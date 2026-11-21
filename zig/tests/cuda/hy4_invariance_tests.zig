//! Hy4-flash row-width, prompt-chunk, sparse-seam and multi-stream batch invariance checks at world 1, 2 and 4.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4");
const check = @import("check.zig");

const Gpu = check.Gpu;
const st = core.safetensors;
const Config = hy4.config.Config;
const ch = hy4.cuda_head;
const cc = hy4.cuda_cache;
const cl = hy4.cuda_layer;
const idx = hy4.cuda_index;
const Stream = cuda.Stream;
const DeviceBuffer = cuda.DeviceBuffer;

fn tap(f: *const st.File, name: []const u8) !st.Tensor {
    return f.get(name) orelse error.MissingTap;
}

fn openIn(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) !st.File {
    const p = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(p);
    return st.File.open(gpa, io, p);
}

fn makePositions(gpu: Gpu, n: usize) !DeviceBuffer {
    const host = try gpu.gpa.alloc(i32, n);
    defer gpu.gpa.free(host);
    for (host, 0..) |*v, i| v.* = @intCast(i);
    return DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
}

fn syncDownload(s: Stream, dev: DeviceBuffer, dst: []u8) !void {
    try s.synchronize();
    try dev.download(0, dst);
}

fn wordsDifferU16(a: []const u8, b: []const u8) usize {
    var n: usize = 0;
    for (0..a.len / 2) |i| n += @intFromBool(!std.mem.eql(u8, a[2 * i ..][0..2], b[2 * i ..][0..2]));
    return n;
}

fn wordsDifferU32(a: []const u8, b: []const u8) usize {
    var n: usize = 0;
    for (0..a.len / 4) |i| n += @intFromBool(!std.mem.eql(u8, a[4 * i ..][0..4], b[4 * i ..][0..4]));
    return n;
}

fn verdict(failures: *usize, ok: bool, comptime fmt: []const u8, args: anytype) void {
    if (!ok) failures.* += 1;
    std.debug.print("{s} " ++ fmt ++ "\n", .{if (ok) "PASS" else "FAIL"} ++ args);
}

/// Live GPU compressed-indexer hook supporting multi-segment batched steps (`call.seg` selects its own `idx.Cache`).
const LiveIndexer = struct {
    ops: idx.Ops,
    w: *const idx.weights.Weights,
    sc: *const idx.Scratch,
    caches: []idx.Cache,

    fn hook(self: *LiveIndexer) cl.IndexerHook {
        return .{ .ctx = self, .run = runHook, .prep = prepHook };
    }

    fn prepHook(ctx: *anyopaque, _: usize, s: Stream, attn_in: u64, q_resid: u64, rows: usize) anyerror!void {
        const self: *LiveIndexer = @ptrCast(@alignCast(ctx));
        try idx.preproject(self.ops, s, self.w, self.sc, attn_in, q_resid, rows);
    }

    fn runHook(ctx: *anyopaque, call: cl.IndexerCall) anyerror!?cl.Recall {
        const self: *LiveIndexer = @ptrCast(@alignCast(ctx));
        const cache = &self.caches[call.seg];
        const sel = idx.selectRange(call.first, call.rows, null, call.recall_limit);
        try idx.run(self.ops, call.s, self.w, cache, self.sc, .{
            .x = call.attn_in,
            .q_resid = call.q_resid,
            .rows = call.rows,
            .first = call.first,
            .pos = call.pos,
            .recall_from = call.recall_limit,
            .idx_out = call.idx_out,
            .preprojected = call.preprojected,
            .row_off = call.row0,
        });
        // Mirror token_keys into LayerCache's index_keys view so LayerView stays populated identically.
        if (call.index_keys != 0 and call.index_keys != cache.token_keys.ptr) {
            const bytes = call.rows * idx.dim * 2;
            const dst = call.index_keys + call.first * idx.dim * 2;
            const src = cache.token_keys.ptr + call.first * idx.dim * 2;
            try cache.token_keys.d.check(cache.token_keys.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, call.s.handle), "cuMemcpyDtoDAsync_v2");
        }
        if (sel.count == 0) return null;
        return .{ .list_from = sel.from, .idx = call.idx_out, .nidx = call.recall_limit };
    }
};

fn headWidths(gpu: Gpu, k: *const cl.Kernels, s: Stream, ck: *core.Checkpoint, c: Config, dir: []const u8, failures: *usize) !void {
    const gpa = gpu.gpa;
    var f = try openIn(gpa, gpu.io, dir, "head_t256.safetensors");
    defer f.close(gpu.io);
    const in_tap = try tap(&f, "in");
    const rows: usize = 64;
    const row_bytes = 4 * c.hidden * 2;
    const logit_bytes = rows * c.vocab * 4;

    var dev_in = try DeviceBuffer.fromHost(gpu.d, in_tap.bytes[0 .. rows * row_bytes]);
    defer dev_in.free();
    var dev_logits = try DeviceBuffer.alloc(gpu.d, logit_bytes);
    defer dev_logits.free();
    var sc = try ch.Scratch.init(gpu.d, c, rows);
    defer sc.deinit();

    const top_t = try hy4.names.top(ck, c);
    const w1_ref = try gpa.alloc(u8, logit_bytes);
    defer gpa.free(w1_ref);
    const full_buf = try gpa.alloc(u8, logit_bytes);
    defer gpa.free(full_buf);
    const cut_buf = try gpa.alloc(u8, logit_bytes);
    defer gpa.free(cut_buf);

    // Spans exercise dense_rows_1 (1, 4) and hy4_rows (5, 6, 16, 17, 15) against one 64-row pass.
    const cuts = [_][2]usize{ .{ 0, 1 }, .{ 1, 4 }, .{ 5, 5 }, .{ 10, 6 }, .{ 16, 16 }, .{ 32, 17 }, .{ 49, 15 } };
    for ([_]usize{ 1, 2, 4 }) |world| {
        var ranks: [4]ch.Top = undefined;
        var loaded: usize = 0;
        defer for (ranks[0..loaded]) |*r| r.deinit();
        for (0..world) |r| {
            ranks[r] = try ch.Top.load(gpu.d, top_t, c, .{ .rank = r, .world = world });
            loaded += 1;
        }
        try dev_logits.fill8(0xff, s.handle);
        for (ranks[0..world]) |*rk| {
            try rk.forward(k.headOps(), s, &sc, dev_in.ptr, null, dev_logits.ptr + rk.share.lo * 4, c.vocab, rows);
        }
        try syncDownload(s, dev_logits, full_buf);
        if (world == 1) @memcpy(w1_ref, full_buf);

        try dev_logits.fill8(0xff, s.handle);
        for (cuts) |sp| for (ranks[0..world]) |*rk| {
            const in_p = dev_in.ptr + sp[0] * row_bytes;
            const out_p = dev_logits.ptr + (sp[0] * c.vocab + rk.share.lo) * 4;
            try rk.forward(k.headOps(), s, &sc, in_p, null, out_p, c.vocab, sp[1]);
        };
        try syncDownload(s, dev_logits, cut_buf);

        const diff_cuts = wordsDifferU32(full_buf, cut_buf);
        const diff_w1 = wordsDifferU32(full_buf, w1_ref);
        verdict(failures, diff_cuts == 0 and diff_w1 == 0, "hy4-invariance head w{d}: 64 vs [1,4,5,6,16,17,15] logit diffs {d}, vs w1 {d}", .{ world, diff_cuts, diff_w1 });
    }
}

fn runLayerSpans(wl: *const cl.WorldLayer, k: *const cl.Kernels, s: Stream, c: Config, sc: *const cl.Scratch, cache: *const cc.LayerCache, li: ?*LiveIndexer, dev_in: u64, dev_pos: u64, dev_out: DeviceBuffer, spans: []const [2]usize, prompt_pass: bool, dst: []u8) !void {
    const row_bytes = 4 * c.hidden * 2;
    try cache.fill(s, 0xff);
    try dev_out.fill8(0xff, s.handle);
    if (li) |x| {
        try x.caches[0].token_keys.fill8(0xff, s.handle);
        try x.caches[0].group_keys.fill8(0xff, s.handle);
        try x.caches[0].rope_cs.fill8(0xff, s.handle);
        x.caches[0].reset();
    }
    const ih: ?cl.IndexerHook = if (li) |x| x.hook() else null;
    for (spans) |sp| {
        try wl.forward(k, s, c, sc, .{
            .streams_in = dev_in + sp[0] * row_bytes,
            .streams_out = dev_out.ptr + sp[0] * row_bytes,
            .rows = sp[1],
            .first = sp[0],
            .pos = dev_pos + sp[0] * 4,
            .cache = cache.view,
            .indexer = ih,
            .single_split_attn = prompt_pass,
        });
    }
    try syncDownload(s, dev_out, dst);
}

fn layerWidthsAndBatch(gpu: Gpu, k: *const cl.Kernels, idx_k: *const idx.Kernels, s: Stream, ck: *core.Checkpoint, c: Config, dir: []const u8, layer_idx: usize, fname: []const u8, failures: *usize) !void {
    const gpa = gpu.gpa;
    var f = try openIn(gpa, gpu.io, dir, fname);
    defer f.close(gpu.io);
    const lt = try hy4.names.layer(ck, c, layer_idx);
    const in_tap = try tap(&f, "in");
    const rows = in_tap.dim(1);
    const row_bytes = 4 * c.hidden * 2;
    const out_bytes = rows * row_bytes;

    var dev_in = try DeviceBuffer.fromHost(gpu.d, in_tap.bytes);
    defer dev_in.free();
    var dev_out = try DeviceBuffer.alloc(gpu.d, out_bytes);
    defer dev_out.free();
    var dev_pos = try makePositions(gpu, rows);
    defer dev_pos.free();

    var idx_w: ?idx.weights.Weights = if (lt.attn.indexer) |ix| try idx.weights.Weights.load(gpu.d, gpa, ix, c) else null;
    defer if (idx_w) |*w| w.deinit();
    var idx_sc: ?idx.Scratch = if (idx_w) |*w| try idx.Scratch.init(gpu.d, &k.mlp.lane, w, rows, rows) else null;
    defer if (idx_sc) |*sc_| sc_.deinit();
    const idx_ops: idx.Ops = .{ .k = idx_k, .lane = &k.mlp.lane, .norms = &k.mlp.norms, .dense = &k.mlp.dense };

    const ref_out = try gpa.alloc(u8, out_bytes);
    defer gpa.free(ref_out);
    const got_out = try gpa.alloc(u8, out_bytes);
    defer gpa.free(got_out);

    const subchunks = [_][2]usize{ .{ 0, 512 }, .{ 512, 512 }, .{ 1024, 1024 }, .{ 2048, 512 } };
    const widths = [_][2]usize{ .{ 0, 2036 }, .{ 2036, 6 }, .{ 2042, 16 }, .{ 2058, 17 }, .{ 2075, 24 }, .{ 2099, 1 }, .{ 2100, 1 }, .{ 2101, 459 } };
    const b_starts = [4]usize{ 37, 510, 2045, 2300 };
    const b_win: usize = 6;
    const b_total = 4 * b_win; // 24 rows > 16 in one batched forward

    for ([_]usize{ 1, 2, 4 }) |world| {
        var wl = try cl.WorldLayer.load(gpa, gpu.d, lt, c, layer_idx, world);
        defer wl.deinit();
        var sc = try cl.Scratch.init(gpa, gpu.d, &k.mlp.lane, c, world, rows);
        defer sc.deinit();

        // 1. Width & chunk-cut invariance across 0..2560 (linear + ring for sliding, live indexer for indexed).
        const mode: cc.SlidingMode = if (c.kind[layer_idx] == .sliding) .{ .ring = cc.minRingSlots(c.window, rows) } else .linear;
        var cache = try cc.LayerCache.init(gpu.d, c, layer_idx, cc.minRingSlots(c.window, rows), mode);
        defer cache.deinit();
        var ic1: [1]idx.Cache = undefined;
        if (idx_w != null) ic1[0] = try idx.Cache.init(gpu.d, rows);
        defer if (idx_w != null) ic1[0].deinit();
        var li1: ?LiveIndexer = if (idx_w) |*w| .{ .ops = idx_ops, .w = w, .sc = if (idx_sc) |*p| p else unreachable, .caches = &ic1 } else null;
        const p_li1: ?*LiveIndexer = if (li1) |*p| p else null;

        try runLayerSpans(&wl, k, s, c, &sc, &cache, p_li1, dev_in.ptr, dev_pos.ptr, dev_out, &.{.{ 0, rows }}, true, ref_out);
        try runLayerSpans(&wl, k, s, c, &sc, &cache, p_li1, dev_in.ptr, dev_pos.ptr, dev_out, &subchunks, true, got_out);
        const diff_chunks = wordsDifferU16(ref_out, got_out);
        try runLayerSpans(&wl, k, s, c, &sc, &cache, p_li1, dev_in.ptr, dev_pos.ptr, dev_out, &widths, true, got_out);
        const diff_widths = wordsDifferU16(ref_out, got_out);

        // 2. 4-stream batched `segs` (24 rows > 16) vs solo 6-row window vs solo 6x1-row plain serial decode.
        const b_mode: cc.SlidingMode = if (c.kind[layer_idx] == .sliding) .{ .ring = c.window + 64 } else .linear;
        var s_caches: [4]cc.LayerCache = undefined;
        var b_caches: [4]cc.LayerCache = undefined;
        var s_ic: [4]idx.Cache = undefined;
        var b_ic: [4]idx.Cache = undefined;
        for (0..4) |i| {
            s_caches[i] = try cc.LayerCache.init(gpu.d, c, layer_idx, rows, b_mode);
            b_caches[i] = try cc.LayerCache.init(gpu.d, c, layer_idx, rows, b_mode);
            if (idx_w != null) {
                s_ic[i] = try idx.Cache.init(gpu.d, rows);
                b_ic[i] = try idx.Cache.init(gpu.d, rows);
            }
        }
        defer for (0..4) |i| {
            s_caches[i].deinit();
            b_caches[i].deinit();
            if (idx_w != null) {
                s_ic[i].deinit();
                b_ic[i].deinit();
            }
        };

        const solo_6 = try gpa.alloc(u8, b_total * row_bytes);
        defer gpa.free(solo_6);
        const solo_1 = try gpa.alloc(u8, b_total * row_bytes);
        defer gpa.free(solo_1);
        const batch_24 = try gpa.alloc(u8, b_total * row_bytes);
        defer gpa.free(batch_24);

        var b_in = try DeviceBuffer.alloc(gpu.d, b_total * row_bytes);
        defer b_in.free();
        var b_out = try DeviceBuffer.alloc(gpu.d, b_total * row_bytes);
        defer b_out.free();
        var pos_host: [b_total]i32 = undefined;
        var caches_host: [b_total]u64 = undefined;
        var segs: [4]cl.Seg = undefined;

        for (b_starts, 0..) |st0, i| {
            var s_li: ?LiveIndexer = if (idx_w) |*w| .{ .ops = idx_ops, .w = w, .sc = if (idx_sc) |*p| p else unreachable, .caches = s_ic[i .. i + 1] } else null;
            var b_li: ?LiveIndexer = if (idx_w) |*w| .{ .ops = idx_ops, .w = w, .sc = if (idx_sc) |*p| p else unreachable, .caches = b_ic[i .. i + 1] } else null;

            // Build 64-row chunked prefill up to st0 so a 576-slot sliding ring wraps repeatedly without collision.
            var p_spans: [64][2]usize = undefined;
            var np: usize = 0;
            var cur: usize = 0;
            while (cur < st0) {
                const step = @min(@as(usize, 64), st0 - cur);
                p_spans[np] = .{ cur, step };
                np += 1;
                cur += step;
            }

            // Solo 6x1 plain decode after chunked [0, st0) prefill.
            try runLayerSpans(&wl, k, s, c, &sc, &s_caches[i], if (s_li) |*p| p else null, dev_in.ptr, dev_pos.ptr, dev_out, p_spans[0..np], true, got_out);
            for (0..b_win) |j| {
                try wl.forward(k, s, c, &sc, .{
                    .streams_in = dev_in.ptr + (st0 + j) * row_bytes,
                    .streams_out = dev_out.ptr + (st0 + j) * row_bytes,
                    .rows = 1,
                    .first = st0 + j,
                    .pos = dev_pos.ptr + (st0 + j) * 4,
                    .cache = s_caches[i].view,
                    .indexer = if (s_li != null) s_li.?.hook() else null,
                    .single_split_attn = false,
                });
            }
            try syncDownload(s, dev_out, got_out);
            @memcpy(solo_1[i * b_win * row_bytes ..][0 .. b_win * row_bytes], got_out[st0 * row_bytes ..][0 .. b_win * row_bytes]);

            // Solo 6-row window after chunked [0, st0) prefill.
            try runLayerSpans(&wl, k, s, c, &sc, &s_caches[i], if (s_li) |*p| p else null, dev_in.ptr, dev_pos.ptr, dev_out, p_spans[0..np], true, got_out);
            try wl.forward(k, s, c, &sc, .{
                .streams_in = dev_in.ptr + st0 * row_bytes,
                .streams_out = dev_out.ptr + st0 * row_bytes,
                .rows = b_win,
                .first = st0,
                .pos = dev_pos.ptr + st0 * 4,
                .cache = s_caches[i].view,
                .indexer = if (s_li != null) s_li.?.hook() else null,
                .single_split_attn = false,
            });
            try syncDownload(s, dev_out, got_out);
            @memcpy(solo_6[i * b_win * row_bytes ..][0 .. b_win * row_bytes], got_out[st0 * row_bytes ..][0 .. b_win * row_bytes]);

            // Chunked [0, st0) prefill into b_caches[i] / b_ic[i] for the shared 24-row batched step.
            try runLayerSpans(&wl, k, s, c, &sc, &b_caches[i], if (b_li) |*p| p else null, dev_in.ptr, dev_pos.ptr, dev_out, p_spans[0..np], true, got_out);
            try gpu.d.check(gpu.d.api.cuMemcpyDtoDAsync_v2(b_in.ptr + i * b_win * row_bytes, dev_in.ptr + st0 * row_bytes, b_win * row_bytes, s.handle), "cuMemcpyDtoDAsync_v2");
            for (0..b_win) |j| {
                pos_host[i * b_win + j] = @intCast(st0 + j);
                caches_host[i * b_win + j] = b_caches[i].view.mla;
            }
            segs[i] = .{ .row0 = i * b_win, .rows = b_win, .first = st0, .cache = b_caches[i].view };
        }

        var b_pos = try DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&pos_host));
        defer b_pos.free();
        var b_row_caches = try DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&caches_host));
        defer b_row_caches.free();
        try b_out.fill8(0xff, s.handle);
        var b_li_all: ?LiveIndexer = if (idx_w) |*w| .{ .ops = idx_ops, .w = w, .sc = if (idx_sc) |*p| p else unreachable, .caches = &b_ic } else null;
        try wl.forward(k, s, c, &sc, .{
            .streams_in = b_in.ptr,
            .streams_out = b_out.ptr,
            .rows = b_total,
            .first = 0,
            .pos = b_pos.ptr,
            .cache = b_caches[0].view,
            .indexer = if (b_li_all != null) b_li_all.?.hook() else null,
            .segs = &segs,
            .caches = b_row_caches.ptr,
        });
        try syncDownload(s, b_out, batch_24);

        const diff_draft_plain = wordsDifferU16(solo_6, solo_1);
        const diff_batch_solo = wordsDifferU16(batch_24, solo_6);
        const ok = diff_chunks == 0 and diff_widths == 0 and diff_draft_plain == 0 and diff_batch_solo == 0;
        verdict(failures, ok, "hy4-invariance layer{d} w{d}: subchunks {d}, widths[1,6,16,17,24] {d}, 6==6x1 {d}, batch24==solo {d}", .{ layer_idx, world, diff_chunks, diff_widths, diff_draft_plain, diff_batch_solo });
    }
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 2) return error.MissingArgument;
    const model = rest[0];
    const dir = rest[1];
    const gpa = gpu.gpa;
    const c = try Config.read(gpa, gpu.io, model);
    var ck = try core.Checkpoint.openModel(gpa, gpu.io, model);
    defer ck.close();
    var k = try cl.Kernels.load(gpu.ctx);
    defer k.deinit();
    var idx_k = try idx.Kernels.load(gpu.d);
    defer idx_k.unload();
    var s = try Stream.init(gpu.d, false);
    defer s.deinit();

    var failures: usize = 0;
    try headWidths(gpu, &k, s, &ck, c, dir, &failures);
    try layerWidthsAndBatch(gpu, &k, &idx_k, s, &ck, c, dir, 2, "layer2_t2560.safetensors", &failures);
    try layerWidthsAndBatch(gpu, &k, &idx_k, s, &ck, c, dir, 5, "layer5_t2560.safetensors", &failures);
    if (failures > 0) return error.Mismatch;
}
