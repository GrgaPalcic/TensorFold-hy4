//! CUDA GPU verification for Hy4-flash ViT (`Tower.encode`) and `tail_index_keys` / `kv8` (`656B` FP8 MLA cache).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4");
const check = @import("check.zig");

const st = core.safetensors;
const idx = hy4.cuda_index;
const mla = hy4.cuda_mla;
const cache_mod = hy4.cuda_cache;

fn bf16ToF64(bytes: []const u8, i: usize) f64 {
    const bits = std.mem.readInt(u16, bytes[2 * i ..][0..2], .little);
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}

pub fn run(gpu: check.Gpu, cmd: []const u8, rest: []const [:0]const u8) !void {
    if (std.mem.eql(u8, cmd, "hy4-vit")) return runVit(gpu, rest);
    if (std.mem.eql(u8, cmd, "hy4-tail-kv8")) return runTailKv8(gpu, rest);
    return error.UnknownCommand;
}

/// `tf-cuda-test hy4-vit MODEL IMAGE_PATH OUT_BIN`: runs `Vision.prepare` + `Tower.encode`, writes header + bf16 rows.
fn runVit(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 3) return error.MissingArgument;
    const model_dir = rest[0];
    const img_path = rest[1];
    const out_path = rest[2];
    const a = gpu.gpa;

    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var rows_op = try hy4.cuda_mlp.cuda_rows.Rows.load(gpu.d);
    defer rows_op.unload();
    var norms = try cuda.norms.Norms.load(gpu.d);
    defer norms.unload();

    const vis = try hy4.cuda_vision.Vision.open(a, gpu.io, gpu.d, model_dir, 0, .{});
    defer vis.deinit();

    const img_bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, img_path, a, .limited(32 << 20));
    defer a.free(img_bytes);
    const prep = try vis.prepare(a, img_bytes, vis.limits.image_tokens);
    defer a.free(prep.pixels);

    const out_hidden = vis.c.out_hidden;
    const out_bytes = @as(usize, prep.tokens) * out_hidden * 2;
    var out_dev = try cuda.DeviceBuffer.alloc(gpu.d, out_bytes);
    defer out_dev.free();
    try vis.tower.?.encode(s, &rows_op, &norms, prep.pixels, &.{.{ .h = prep.gh, .w = prep.gw }}, out_dev.ptr);

    const host_out = try a.alloc(u8, 16 + out_bytes);
    defer a.free(host_out);
    std.mem.writeInt(u32, host_out[0..4], prep.gh, .little);
    std.mem.writeInt(u32, host_out[4..8], prep.gw, .little);
    std.mem.writeInt(u32, host_out[8..12], prep.tokens, .little);
    std.mem.writeInt(u32, host_out[12..16], @intCast(out_hidden), .little);
    try out_dev.download(0, host_out[16..]);
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = out_path, .data = host_out });
    std.debug.print("PASS hy4-vit encoded {s}: grid={d}x{d}, tokens={d}, out_hidden={d} -> {s}\n", .{ img_path, prep.gh, prep.gw, prep.tokens, out_hidden, out_path });
}

/// `tf-cuda-test hy4-tail-kv8 MODEL FIX_DIR`: checks 0-bit `runTail == run` across cuts + BF16 vs FP8 KV.
fn runTailKv8(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 2) return error.MissingArgument;
    const model_dir = rest[0];
    const fix_dir = rest[1];
    const a = gpu.gpa;

    const c = try hy4.Config.read(a, gpu.io, model_dir);
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var all_k = try hy4.cuda_model.AllKernels.load(gpu.ctx);
    defer all_k.deinit();

    var ck = try core.checkpoint.Checkpoint.openModel(a, gpu.io, model_dir);
    defer ck.close();
    const l0_names = try hy4.names.layer(&ck, c, 0);
    var layer0 = try hy4.cuda_layer.Layer.load(a, gpu.d, l0_names, c, 0, .{});
    defer layer0.deinit();
    var idx_w = try idx.weights.Weights.load(gpu.d, a, l0_names.attn.indexer.?, c);
    defer idx_w.deinit();
    const l2_names = try hy4.names.layer(&ck, c, 2);
    var layer2 = try hy4.cuda_layer.Layer.load(a, gpu.d, l2_names, c, 2, .{});
    defer layer2.deinit();

    const f0_path = try std.fs.path.join(a, &.{ fix_dir, "layer0_t2560.safetensors" });
    defer a.free(f0_path);
    var f0 = try st.File.open(a, gpu.io, f0_path);
    defer f0.close(gpu.io);
    const x_ten = f0.get("attn_in16") orelse f0.get("in") orelse return error.MissingTap;
    const t: usize = 2560;
    const hd = c.hidden;

    var x_dev = try cuda.DeviceBuffer.fromHost(gpu.d, x_ten.bytes);
    defer x_dev.free();
    const pos_h = try a.alloc(i32, t);
    defer a.free(pos_h);
    for (pos_h, 0..) |*p, i| p.* = @intCast(i);
    var pos_dev = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(pos_h));
    defer pos_dev.free();

    var mla_sc = try mla.Scratch.init(gpu.d, &all_k.layer.mlp.lane, &layer0.mla, t);
    defer mla_sc.deinit();
    var idx_sc = try idx.Scratch.init(gpu.d, &all_k.layer.mlp.lane, &idx_w, t, t);
    defer idx_sc.deinit();

    // 1. Reference full `run` vs chunked `runTail` (`[2051, 506, 1, 1, 1]`)
    var full_cache = try idx.Cache.init(gpu.d, t);
    defer full_cache.deinit();
    const n_sel = t - 2048;
    var ref_idx = try cuda.DeviceBuffer.alloc(gpu.d, n_sel * 2048 * 4);
    defer ref_idx.free();
    try mla.qResid(all_k.layer.mlaOps(), s, &layer0.mla, &mla_sc, x_dev.ptr, t);
    try idx.run(all_k.idxOps(), s, &idx_w, &full_cache, &idx_sc, .{
        .x = x_dev.ptr,
        .q_resid = mla_sc.qres(),
        .rows = t,
        .first = 0,
        .pos = pos_dev.ptr,
        .idx_out = ref_idx.ptr,
    });

    var tail_buf = try cuda.DeviceBuffer.alloc(gpu.d, cache_mod.tail_index_rows * idx.dim * 2);
    defer tail_buf.free();
    var tail_gk = try cuda.DeviceBuffer.alloc(gpu.d, (t / idx.ratio) * idx.dim * 2);
    defer tail_gk.free();
    var tail_idx = try cuda.DeviceBuffer.alloc(gpu.d, n_sel * 2048 * 4);
    defer tail_idx.free();
    const cuts = [_]usize{ 2051, 506, 1, 1, 1 };
    var at: usize = 0;
    for (cuts) |len| {
        try mla.qResid(all_k.layer.mlaOps(), s, &layer0.mla, &mla_sc, x_dev.ptr + at * hd * 2, len);
        var io_ptr: u64 = 0;
        if (at + len > 2048) io_ptr = tail_idx.ptr + (@max(at, 2048) - 2048) * 2048 * 4;
        try idx.runTail(all_k.idxOps(), gpu.d, s, &idx_w, tail_buf.ptr, tail_gk.ptr, t, full_cache.rope_cs.ptr, full_cache.rope_cs.ptr, 0, 0, 0, &idx_sc, .{
            .x = x_dev.ptr + at * hd * 2,
            .q_resid = mla_sc.qres(),
            .rows = len,
            .first = at,
            .pos = pos_dev.ptr + at * 4,
            .idx_out = io_ptr,
        });
        at += len;
    }
    try s.synchronize();
    const gk_a = try a.alloc(u8, full_cache.group_keys.len);
    defer a.free(gk_a);
    const gk_b = try a.alloc(u8, tail_gk.len);
    defer a.free(gk_b);
    try full_cache.group_keys.download(0, gk_a);
    try tail_gk.download(0, gk_b);
    try check.sameBytes("hy4-tail-kv8 group_keys 0-bit", gk_b, gk_a);

    const id_a = try a.alloc(u8, ref_idx.len);
    defer a.free(id_a);
    const id_b = try a.alloc(u8, tail_idx.len);
    defer a.free(id_b);
    try ref_idx.download(0, id_a);
    try tail_idx.download(0, id_b);
    try check.sameBytes("hy4-tail-kv8 recalled idx 0-bit", id_b, id_a);
    std.debug.print("PASS hy4-tail-kv8: runTail[2051,506,1,1,1] == run[2560] 0 bit diffs on group_keys & recalled idx\n", .{});

    // 2. Compare MLA output on Layer 0 (indexed, 2560 tok) and Layer 2 (sliding, 2560 tok): BF16 vs FP8 (656B) cache
    const mla_ops16 = all_k.layer.mlaOpsProf(null);
    var mla_ops8 = mla_ops16;
    mla_ops8.kv8 = true;
    inline for (.{ .{ 0, false }, .{ 2, true } }) |spec| {
        const l_id = spec[0];
        const is_sld = spec[1];
        const ly = if (is_sld) &layer2 else &layer0;
        var l_sc = try mla.Scratch.init(gpu.d, &all_k.layer.mlp.lane, &ly.mla, t);
        defer l_sc.deinit();
        const mode: cache_mod.SlidingMode = if (is_sld) .{ .ring = 4608 } else .{ .ring = 512 };
        const b16 = try cache_mod.layerBytesEx(c, l_id, t, mode, .{ .kv8 = false, .tail_index_keys = true });
        const b8 = try cache_mod.layerBytesEx(c, l_id, t, mode, .{ .kv8 = true, .tail_index_keys = true });
        var c16 = try cuda.DeviceBuffer.alloc(gpu.d, b16);
        defer c16.free();
        var c8 = try cuda.DeviceBuffer.alloc(gpu.d, b8);
        defer c8.free();
        var out16 = try cuda.DeviceBuffer.alloc(gpu.d, t * hd * 4);
        defer out16.free();
        var out8 = try cuda.DeviceBuffer.alloc(gpu.d, t * hd * 4);
        defer out8.free();
        var y16 = try cuda.DeviceBuffer.alloc(gpu.d, t * hd * 2);
        defer y16.free();
        var y8 = try cuda.DeviceBuffer.alloc(gpu.d, t * hd * 2);
        defer y8.free();

        const seg16 = [1]mla.Seg{.{ .row0 = 0, .rows = t, .first = 0, .cache = c16.ptr, .ring = if (is_sld) 4608 else 0, .list_from = if (is_sld) null else 2048, .idx = if (is_sld) 0 else ref_idx.ptr, .nidx = if (is_sld) 0 else 2048 }};
        const seg8 = [1]mla.Seg{.{ .row0 = 0, .rows = t, .first = 0, .cache = c8.ptr, .ring = if (is_sld) 4608 else 0, .list_from = if (is_sld) null else 2048, .idx = if (is_sld) 0 else ref_idx.ptr, .nidx = if (is_sld) 0 else 2048 }};
        try mla.attendSegs(mla_ops16, s, &ly.mla, &l_sc, x_dev.ptr, t, pos_dev.ptr, out16.ptr, &seg16, 0, false);
        try mla.finish(mla_ops16, s, &ly.mla, out16.ptr, 1, t, y16.ptr);
        try mla.attendSegs(mla_ops8, s, &ly.mla, &l_sc, x_dev.ptr, t, pos_dev.ptr, out8.ptr, &seg8, 0, false);
        try mla.finish(mla_ops8, s, &ly.mla, out8.ptr, 1, t, y8.ptr);
        try s.synchronize();

        const h16 = try a.alloc(u8, y16.len);
        defer a.free(h16);
        const h8 = try a.alloc(u8, y8.len);
        defer a.free(h8);
        try y16.download(0, h16);
        try y8.download(0, h8);
        var dot: f64 = 0;
        var na: f64 = 0;
        var nb: f64 = 0;
        var sq_err: f64 = 0;
        var max_abs: f64 = 0;
        const n = t * hd;
        for (0..n) |i| {
            const va = bf16ToF64(h16, i);
            const vb = bf16ToF64(h8, i);
            dot += va * vb;
            na += va * va;
            nb += vb * vb;
            const d = @abs(va - vb);
            sq_err += d * d;
            max_abs = @max(max_abs, d);
        }
        const cos = dot / (@sqrt(na) * @sqrt(nb));
        const rmse = @sqrt(sq_err / @as(f64, @floatFromInt(n)));
        std.debug.print("PASS hy4-tail-kv8 layer{d} ({s}, T=2560): cos={d:.7}, rmse={d:.6}, max_abs={d:.5}\n", .{ l_id, if (is_sld) "sliding" else "indexed", cos, rmse, max_abs });
        try check.expect(cos > 0.9995, "layer{d} kv8 cosine {d:.6} too low", .{ l_id, cos });
    }

    // 3. Scrambled 256-token `page_table` (`[9,2,7,0,5,8,1,6,3,4]`) == contiguous 0-bit on `runTail` + MLA!
    const perm = [10]u32{ 9, 2, 7, 0, 5, 8, 1, 6, 3, 4 };
    var pt_dev = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&perm));
    defer pt_dev.free();
    var pg_tail_buf = try cuda.DeviceBuffer.alloc(gpu.d, cache_mod.tail_index_rows * idx.dim * 2);
    defer pg_tail_buf.free();
    var pg_gk = try cuda.DeviceBuffer.alloc(gpu.d, 12 * 64 * idx.dim * 2);
    defer pg_gk.free();
    try pg_gk.fill8(0xFF, s.handle);
    var pg_idx = try cuda.DeviceBuffer.alloc(gpu.d, n_sel * 2048 * 4);
    defer pg_idx.free();
    at = 0;
    for (cuts) |len| {
        try mla.qResid(all_k.layer.mlaOps(), s, &layer0.mla, &mla_sc, x_dev.ptr + at * hd * 2, len);
        var io_ptr: u64 = 0;
        if (at + len > 2048) io_ptr = pg_idx.ptr + (@max(at, 2048) - 2048) * 2048 * 4;
        try idx.runTail(all_k.idxOps(), gpu.d, s, &idx_w, pg_tail_buf.ptr, pg_gk.ptr, 12 * 256, full_cache.rope_cs.ptr, full_cache.rope_cs.ptr, 0, 0, 0, &idx_sc, .{
            .x = x_dev.ptr + at * hd * 2,
            .q_resid = mla_sc.qres(),
            .rows = len,
            .first = at,
            .pos = pos_dev.ptr + at * 4,
            .idx_out = io_ptr,
            .page_table = pt_dev.ptr,
            .pt_stride = 10,
        });
        at += len;
    }
    try s.synchronize();
    const pg_id_h = try a.alloc(u8, pg_idx.len);
    defer a.free(pg_id_h);
    try pg_idx.download(0, pg_id_h);
    try check.sameBytes("hy4-paged-kv runTail recalled idx 0-bit", pg_id_h, id_a);
    const gk_chunk = 64 * idx.dim * 2;
    const pg_gk_page = try a.alloc(u8, gk_chunk);
    defer a.free(pg_gk_page);
    for (perm, 0..) |pid, lp| {
        try pg_gk.download(@as(usize, pid) * gk_chunk, pg_gk_page);
        try check.sameBytes("hy4-paged-kv group_keys page 0-bit", pg_gk_page, gk_a[lp * gk_chunk ..][0..gk_chunk]);
    }

    inline for (.{ false, true }) |kv8| {
        const ops = if (kv8) mla_ops8 else mla_ops16;
        const rb = cache_mod.mlaRowBytes(kv8);
        var l_sc = try mla.Scratch.init(gpu.d, &all_k.layer.mlp.lane, &layer0.mla, t);
        defer l_sc.deinit();
        var ref_c = try cuda.DeviceBuffer.alloc(gpu.d, t * rb);
        defer ref_c.free();
        var pg_c = try cuda.DeviceBuffer.alloc(gpu.d, 12 * 256 * rb);
        defer pg_c.free();
        try pg_c.fill8(0xFF, s.handle);
        var ref_y = try cuda.DeviceBuffer.alloc(gpu.d, t * hd * 2);
        defer ref_y.free();
        var pg_y = try cuda.DeviceBuffer.alloc(gpu.d, t * hd * 2);
        defer pg_y.free();
        var out_tmp = try cuda.DeviceBuffer.alloc(gpu.d, t * hd * 4);
        defer out_tmp.free();
        var cur: usize = 0;
        for (cuts) |len| {
            const lf: ?usize = if (cur + len > 2048) @max(cur, 2048) - cur else null;
            const ip: u64 = if (lf != null) ref_idx.ptr + (@max(cur, 2048) - 2048) * 2048 * 4 else 0;
            const s_ref = [1]mla.Seg{.{ .row0 = 0, .rows = len, .first = cur, .cache = ref_c.ptr, .list_from = lf, .idx = ip, .nidx = if (lf != null) 2048 else 0 }};
            try mla.attendSegs(ops, s, &layer0.mla, &l_sc, x_dev.ptr + cur * hd * 2, len, pos_dev.ptr + cur * 4, out_tmp.ptr, &s_ref, 0, false);
            try mla.finish(ops, s, &layer0.mla, out_tmp.ptr, 1, len, ref_y.ptr + cur * hd * 2);
            const s_pg = [1]mla.Seg{.{ .row0 = 0, .rows = len, .first = cur, .cache = pg_c.ptr, .page_table = pt_dev.ptr, .pt_stride = 10, .list_from = lf, .idx = ip, .nidx = if (lf != null) 2048 else 0 }};
            try mla.attendSegs(ops, s, &layer0.mla, &l_sc, x_dev.ptr + cur * hd * 2, len, pos_dev.ptr + cur * 4, out_tmp.ptr, &s_pg, 0, false);
            try mla.finish(ops, s, &layer0.mla, out_tmp.ptr, 1, len, pg_y.ptr + cur * hd * 2);
            cur += len;
        }
        try s.synchronize();
        const rh = try a.alloc(u8, ref_y.len);
        defer a.free(rh);
        const ph = try a.alloc(u8, pg_y.len);
        defer a.free(ph);
        try ref_y.download(0, rh);
        try pg_y.download(0, ph);
        try check.sameBytes(if (kv8) "hy4-paged-kv FP8 MLA 0-bit" else "hy4-paged-kv BF16 MLA 0-bit", ph, rh);
    }

    // 4. Speculative draft-rejection rewind across 4-tok groups & 32-slot ring wraps + x16_qreg(kv8) == x16(kv8) 0-bit!
    var bad_x = try cuda.DeviceBuffer.alloc(gpu.d, 8 * hd * 2);
    defer bad_x.free();
    try bad_x.fill8(0x3C, s.handle);
    var rw_tail = try cuda.DeviceBuffer.alloc(gpu.d, cache_mod.tail_index_rows * idx.dim * 2);
    defer rw_tail.free();
    var rw_gk = try cuda.DeviceBuffer.alloc(gpu.d, 12 * 64 * idx.dim * 2);
    defer rw_gk.free();
    try rw_gk.fill8(0xFF, s.handle);
    var rw_idx = try cuda.DeviceBuffer.alloc(gpu.d, n_sel * 2048 * 4);
    defer rw_idx.free();
    var rw_tmp_idx = try cuda.DeviceBuffer.alloc(gpu.d, 8 * 2048 * 4);
    defer rw_tmp_idx.free();
    try mla.qResid(all_k.layer.mlaOps(), s, &layer0.mla, &mla_sc, x_dev.ptr, 2045);
    try idx.runTail(all_k.idxOps(), gpu.d, s, &idx_w, rw_tail.ptr, rw_gk.ptr, 12 * 256, full_cache.rope_cs.ptr, full_cache.rope_cs.ptr, 0, 0, 0, &idx_sc, .{
        .x = x_dev.ptr,
        .q_resid = mla_sc.qres(),
        .rows = 2045,
        .first = 0,
        .pos = pos_dev.ptr,
        .page_table = pt_dev.ptr,
        .pt_stride = 10,
    });
    at = 2045;
    while (at < t) {
        const kept = @min(1 + (at % 3), t - at);
        const win = @min(kept + 3, t - at);
        try gpu.d.check(gpu.d.api.cuMemcpyDtoDAsync_v2(bad_x.ptr, x_dev.ptr + at * hd * 2, kept * hd * 2, s.handle), "cuMemcpyDtoDAsync");
        if (win > kept) try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = bad_x.ptr + kept * hd * 2, .len = (win - kept) * hd * 2, .borrowed = true }).fill8(0x3E, s.handle);
        try mla.qResid(all_k.layer.mlaOps(), s, &layer0.mla, &mla_sc, bad_x.ptr, win);
        try idx.runTail(all_k.idxOps(), gpu.d, s, &idx_w, rw_tail.ptr, rw_gk.ptr, 12 * 256, full_cache.rope_cs.ptr, full_cache.rope_cs.ptr, 0, 0, 0, &idx_sc, .{
            .x = bad_x.ptr,
            .q_resid = mla_sc.qres(),
            .rows = win,
            .first = at,
            .pos = pos_dev.ptr + at * 4,
            .idx_out = if (at + win > 2048) rw_tmp_idx.ptr else 0,
            .page_table = pt_dev.ptr,
            .pt_stride = 10,
        });
        if (at + kept > 2048) {
            const sel0 = if (at >= 2048) 0 else 2048 - at;
            const sel_n = (at + kept) - @max(at, 2048);
            const dst_off = (@max(at, 2048) - 2048) * 2048 * 4;
            try gpu.d.check(gpu.d.api.cuMemcpyDtoDAsync_v2(rw_idx.ptr + dst_off, rw_tmp_idx.ptr + sel0 * 2048 * 4, sel_n * 2048 * 4, s.handle), "cuMemcpyDtoDAsync");
        }
        at += kept;
    }
    try s.synchronize();
    try rw_idx.download(0, pg_id_h);
    try check.sameBytes("hy4-rewind runTail recalled idx 0-bit", pg_id_h, id_a);
    for (perm, 0..) |pid, lp| {
        try rw_gk.download(@as(usize, pid) * gk_chunk, pg_gk_page);
        try check.sameBytes("hy4-rewind group_keys page 0-bit", pg_gk_page, gk_a[lp * gk_chunk ..][0..gk_chunk]);
    }
    // Check R=4 (x16_qreg_paged) == 4x R=1 (x16_paged) under kv8=true
    {
        const rb8 = cache_mod.mlaRowBytes(true);
        var l_sc = try mla.Scratch.init(gpu.d, &all_k.layer.mlp.lane, &layer0.mla, t);
        defer l_sc.deinit();
        var pg_c = try cuda.DeviceBuffer.alloc(gpu.d, 12 * 256 * rb8);
        defer pg_c.free();
        var y_r4 = try cuda.DeviceBuffer.alloc(gpu.d, 4 * hd * 2);
        defer y_r4.free();
        var y_r1 = try cuda.DeviceBuffer.alloc(gpu.d, 4 * hd * 2);
        defer y_r1.free();
        var out_tmp = try cuda.DeviceBuffer.alloc(gpu.d, t * hd * 4);
        defer out_tmp.free();
        const s_pre = [1]mla.Seg{.{ .row0 = 0, .rows = 2556, .first = 0, .cache = pg_c.ptr, .page_table = pt_dev.ptr, .pt_stride = 10, .list_from = 2048, .idx = ref_idx.ptr, .nidx = 2048 }};
        try mla.attendSegs(mla_ops8, s, &layer0.mla, &l_sc, x_dev.ptr, 2556, pos_dev.ptr, out_tmp.ptr, &s_pre, 0, false);
        const s_4 = [1]mla.Seg{.{ .row0 = 0, .rows = 4, .first = 2556, .cache = pg_c.ptr, .page_table = pt_dev.ptr, .pt_stride = 10, .list_from = 0, .idx = ref_idx.ptr + (2556 - 2048) * 2048 * 4, .nidx = 2048 }};
        try mla.attendSegs(mla_ops8, s, &layer0.mla, &l_sc, x_dev.ptr + 2556 * hd * 2, 4, pos_dev.ptr + 2556 * 4, out_tmp.ptr, &s_4, 0, false);
        try mla.finish(mla_ops8, s, &layer0.mla, out_tmp.ptr, 1, 4, y_r4.ptr);
        for (0..4) |j| {
            const cur = 2556 + j;
            const s_1 = [1]mla.Seg{.{ .row0 = 0, .rows = 1, .first = cur, .cache = pg_c.ptr, .page_table = pt_dev.ptr, .pt_stride = 10, .list_from = 0, .idx = ref_idx.ptr + (cur - 2048) * 2048 * 4, .nidx = 2048 }};
            try mla.attendSegs(mla_ops8, s, &layer0.mla, &l_sc, x_dev.ptr + cur * hd * 2, 1, pos_dev.ptr + cur * 4, out_tmp.ptr, &s_1, 0, false);
            try mla.finish(mla_ops8, s, &layer0.mla, out_tmp.ptr, 1, 1, y_r1.ptr + j * hd * 2);
        }
        try s.synchronize();
        var h4: [4 * 3072 * 2]u8 = undefined;
        var h1: [4 * 3072 * 2]u8 = undefined;
        try y_r4.download(0, &h4);
        try y_r1.download(0, &h1);
        try check.sameBytes("hy4 x16_qreg_paged(kv8) == x16_paged(kv8) 0-bit", &h4, &h1);
    }
    std.debug.print("PASS hy4-tail-kv8: scrambled 256-tok page_table == contiguous 0 bit diffs across cuts on runTail + BF16/FP8 MLA!\n", .{});
}
