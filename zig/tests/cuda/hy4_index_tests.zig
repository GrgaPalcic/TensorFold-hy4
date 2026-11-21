//! tf-cuda-test hy4-index MODEL FIXTURE... (live indexer + MLA, worlds 1/2/4, cut=2051) and hy4-index-contract.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4");
const check = @import("check.zig");

const mla = hy4.cuda_mla;
const idx = hy4.cuda_index;
const DeviceBuffer = cuda.DeviceBuffer;
const recall_from = idx.tokens_per_row;
const groups = idx.groups;
const ratio = idx.ratio;

pub fn run(gpu: check.Gpu, cmd: []const u8, rest: []const [:0]const u8) !void {
    if (std.mem.eql(u8, cmd, "hy4-index-contract")) return @import("contract_hy4_index.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "hy4-index")) {
        if (rest.len < 2) return error.MissingArgument;
        return fixtures(gpu, rest[0], rest[1..]);
    }
    return error.UnknownCommand;
}

const Ops = struct {
    k_idx: idx.Kernels,
    k_mla: mla.Kernels,
    lane: cuda.qmmf.Lane,
    norms: cuda.norms.Norms,
    dense: cuda.dense_rows.DenseRows,
    rope: cuda.mla.Mla,

    fn idxOps(o: *const Ops) idx.Ops {
        return .{ .k = &o.k_idx, .lane = &o.lane, .norms = &o.norms, .dense = &o.dense };
    }

    fn mlaOps(o: *const Ops) mla.Ops {
        return .{ .k = &o.k_mla, .lane = &o.lane, .norms = &o.norms, .mla = &o.rope };
    }
};

const Err = struct { max: f64 = 0, mean: f64 = 0, col: f64 = 0, at: usize = 0 };

fn errors(a: std.mem.Allocator, got: []const u16, want: []const u8, t: usize, hd: usize) !Err {
    const cols = try a.alloc(f64, hd);
    defer a.free(cols);
    @memset(cols, 0);
    var e: Err = .{};
    var sum: f64 = 0;
    for (0..t * hd) |i| {
        const w: f64 = @bitCast(std.mem.readInt(u64, want[8 * i ..][0..8], .little));
        const d = @abs(@as(f64, core.contract.spec.fromBf16(got[i])) - w);
        if (std.math.isNan(d) or d > e.max) e.max = d;
        sum += d;
        cols[i % hd] += d;
    }
    e.mean = sum / @as(f64, @floatFromInt(t * hd));
    for (cols, 0..) |v, i| if (v / @as(f64, @floatFromInt(t)) > e.col) {
        e.col = v / @as(f64, @floatFromInt(t));
        e.at = i;
    };
    return e;
}

fn recallSetDiffers(got_bytes: []const u8, ref_bytes: []const u8, t: usize) usize {
    var n: usize = 0;
    const n_sel = t - recall_from;
    for (0..n_sel) |i| {
        const r = recall_from + i;
        var a: [groups]i32 = undefined;
        var b: [groups]i32 = undefined;
        for (&a, &b, 0..) |*x, *y, j| {
            x.* = std.mem.readInt(i32, got_bytes[(i * groups + j) * 4 ..][0..4], .little);
            y.* = std.mem.readInt(i32, ref_bytes[(r * groups + j) * 4 ..][0..4], .little);
        }
        std.mem.sort(i32, &a, {}, std.sort.asc(i32));
        std.mem.sort(i32, &b, {}, std.sort.asc(i32));
        n += @intFromBool(!std.mem.eql(i32, &a, &b));
    }
    return n;
}

fn verifyGroupsAndTokens(grp_bytes: []const u8, tok_bytes: []const u8, t: usize) !void {
    const n_sel = t - recall_from;
    for (0..n_sel) |i| {
        const r = recall_from + i;
        const own: i32 = @intCast(r / ratio);
        const last = std.mem.readInt(i32, grp_bytes[(i * groups + groups - 1) * 4 ..][0..4], .little);
        try check.expect(last == own, "row {d}: LIST[511] = {d}, want {d}", .{ r, last, own });
        var prev: i32 = -1;
        for (0..groups - 1) |j| {
            const g = std.mem.readInt(i32, grp_bytes[(i * groups + j) * 4 ..][0..4], .little);
            try check.expect(g > prev and g < own, "row {d} slot {d}: {d} not strictly ascending < {d} (prev {d})", .{ r, j, g, own, prev });
            prev = g;
        }
        for (0..groups) |j| {
            const g = std.mem.readInt(i32, grp_bytes[(i * groups + j) * 4 ..][0..4], .little);
            for (0..ratio) |o| {
                const tok = std.mem.readInt(i32, tok_bytes[(i * recall_from + j * ratio + o) * 4 ..][0..4], .little);
                try check.expect(tok == 4 * g + @as(i32, @intCast(o)), "row {d} tok {d}: {d} != {d}", .{ r, j * ratio + o, tok, 4 * g + @as(i32, @intCast(o)) });
            }
        }
    }
}

const IndexRun = struct {
    t: usize,
    hd: usize,
    x: DeviceBuffer,
    pos: DeviceBuffer,
    groups_dev: DeviceBuffer,
    idx_dev: DeviceBuffer,
    groups_seg: DeviceBuffer,
    idx_seg: DeviceBuffer,
    tk_ref: []u8,
    gk_ref: []u8,
    grp_ref: []u8,
    idx_ref: []u8,

    fn deinit(self: *IndexRun, a: std.mem.Allocator) void {
        for ([_]*DeviceBuffer{ &self.x, &self.pos, &self.groups_dev, &self.idx_dev, &self.groups_seg, &self.idx_seg }) |b| b.free();
        a.free(self.tk_ref);
        a.free(self.gk_ref);
        a.free(self.grp_ref);
        a.free(self.idx_ref);
    }
};

fn runIndexPass(s: cuda.Stream, ops: *const Ops, idx_w: *const idx.weights.Weights, idx_cache: *idx.Cache, idx_sc: *const idx.Scratch, mla_w0: *const mla.weights.Weights, mla_sc0: *const mla.Scratch, ir: *const IndexRun, lo: usize, len: usize, grp_dst: u64, idx_dst: u64) !void {
    try mla.qResid(ops.mlaOps(), s, mla_w0, mla_sc0, ir.x.ptr + lo * ir.hd * 2, len);
    var g_out: u64 = 0;
    var i_out: u64 = 0;
    if (lo + len > recall_from) {
        const sel_start = @max(lo, recall_from) - recall_from;
        g_out = grp_dst + sel_start * groups * 4;
        i_out = idx_dst + sel_start * recall_from * 4;
    }
    try idx.run(ops.idxOps(), s, idx_w, idx_cache, idx_sc, .{
        .x = ir.x.ptr + lo * ir.hd * 2,
        .q_resid = mla_sc0.qres(),
        .rows = len,
        .first = lo,
        .pos = ir.pos.ptr + lo * 4,
        .groups_out = g_out,
        .idx_out = i_out,
    });
}

fn checkIndexerParity(gpu: check.Gpu, s: cuda.Stream, ops: *const Ops, idx_w: *const idx.weights.Weights, idx_cache: *idx.Cache, idx_sc: *const idx.Scratch, mla_w0: *const mla.weights.Weights, mla_sc0: *const mla.Scratch, ir: *IndexRun) !void {
    const a = gpu.gpa;
    const n_sel = ir.t - recall_from;
    try idx_cache.token_keys.fill8(0xff, s.handle);
    try idx_cache.group_keys.fill8(0xff, s.handle);
    try idx_cache.rope_cs.fill8(0xff, s.handle);
    idx_cache.reset();
    try runIndexPass(s, ops, idx_w, idx_cache, idx_sc, mla_w0, mla_sc0, ir, 0, ir.t, ir.groups_dev.ptr, ir.idx_dev.ptr);
    try s.synchronize();
    try idx_cache.token_keys.download(0, ir.tk_ref);
    try idx_cache.group_keys.download(0, ir.gk_ref);
    try ir.groups_dev.download(0, ir.grp_ref);
    try ir.idx_dev.download(0, ir.idx_ref);
    try verifyGroupsAndTokens(ir.grp_ref, ir.idx_ref, ir.t);

    const tmp_tk = try a.alloc(u8, ir.tk_ref.len);
    defer a.free(tmp_tk);
    const tmp_gk = try a.alloc(u8, ir.gk_ref.len);
    defer a.free(tmp_gk);
    const tmp_grp = try a.alloc(u8, ir.grp_ref.len);
    defer a.free(tmp_grp);
    const tmp_idx = try a.alloc(u8, ir.idx_ref.len);
    defer a.free(tmp_idx);
    for (0..4) |_| {
        try idx_cache.token_keys.fill8(0xff, s.handle);
        try idx_cache.group_keys.fill8(0xff, s.handle);
        try idx_cache.rope_cs.fill8(0xff, s.handle);
        idx_cache.reset();
        try runIndexPass(s, ops, idx_w, idx_cache, idx_sc, mla_w0, mla_sc0, ir, 0, ir.t, ir.groups_dev.ptr, ir.idx_dev.ptr);
        try s.synchronize();
        try idx_cache.token_keys.download(0, tmp_tk);
        try idx_cache.group_keys.download(0, tmp_gk);
        try ir.groups_dev.download(0, tmp_grp);
        try ir.idx_dev.download(0, tmp_idx);
        try check.sameBytes("hy4-index token_keys repeat", tmp_tk, ir.tk_ref);
        try check.sameBytes("hy4-index group_keys repeat", tmp_gk, ir.gk_ref);
        try check.sameBytes("hy4-index groups repeat", tmp_grp, ir.grp_ref);
        try check.sameBytes("hy4-index idx repeat", tmp_idx, ir.idx_ref);
    }

    const cut: usize = 2051;
    const cuts = [_][2]usize{ .{ 0, cut }, .{ cut, ir.t - 3 - cut }, .{ ir.t - 3, 1 }, .{ ir.t - 2, 1 }, .{ ir.t - 1, 1 } };
    try idx_cache.token_keys.fill8(0xff, s.handle);
    try idx_cache.group_keys.fill8(0xff, s.handle);
    try idx_cache.rope_cs.fill8(0xff, s.handle);
    idx_cache.reset();
    for (cuts) |seg| try runIndexPass(s, ops, idx_w, idx_cache, idx_sc, mla_w0, mla_sc0, ir, seg[0], seg[1], ir.groups_seg.ptr, ir.idx_seg.ptr);
    try s.synchronize();
    try idx_cache.token_keys.download(0, tmp_tk);
    try idx_cache.group_keys.download(0, tmp_gk);
    try ir.groups_seg.download(0, tmp_grp);
    try ir.idx_seg.download(0, tmp_idx);
    try check.sameBytes("hy4-index token_keys cut=2051+decode", tmp_tk, ir.tk_ref);
    try check.sameBytes("hy4-index group_keys cut=2051+decode", tmp_gk, ir.gk_ref);
    try check.sameBytes("hy4-index groups cut=2051+decode", tmp_grp, ir.grp_ref);
    try check.sameBytes("hy4-index idx cut=2051+decode", tmp_idx, ir.idx_ref);
    _ = n_sel;
}

fn mlaCall(ir: *const IndexRun, cache: u64, parts: u64, lo: usize, n: usize, rank: usize, idx_ptr: u64) mla.Call {
    var c: mla.Call = .{ .x = ir.x.ptr + lo * ir.hd * 2, .rows = n, .first = lo, .pos = ir.pos.ptr + lo * 4, .cache = cache, .part = parts + (rank * ir.t + lo) * ir.hd * 4 };
    if (lo + n > recall_from) {
        const from = @max(lo, recall_from);
        c.list_from = from - lo;
        c.idx = idx_ptr + (from - recall_from) * recall_from * 4;
        c.nidx = recall_from;
    }
    return c;
}

fn forwardWorld(gpu: check.Gpu, ops: *const Ops, s: cuda.Stream, ck: *core.Checkpoint, c: hy4.Config, layer: usize, idx_w: *const idx.weights.Weights, idx_cache: *idx.Cache, idx_sc: *const idx.Scratch, ir: *const IndexRun, cache_buf: DeviceBuffer, parts_buf: DeviceBuffer, out_buf: DeviceBuffer, world: usize, got: []u16) !f64 {
    const a = gpu.gpa;
    const t0 = check.now(gpu.io);
    const lt = try hy4.names.layer(ck, c, layer);
    var ws: [4]mla.weights.Weights = undefined;
    var scs: [4]mla.Scratch = undefined;
    var loaded: usize = 0;
    defer for (0..loaded) |i| {
        scs[i].deinit();
        ws[i].deinit();
    };
    for (0..world) |rank| {
        ws[rank] = try mla.weights.Weights.load(gpu.d, a, lt.attn, c, layer, rank, world);
        errdefer ws[rank].deinit();
        scs[rank] = try mla.Scratch.init(gpu.d, &ops.lane, &ws[rank], ir.t);
        loaded += 1;
    }
    for (0..world) |rank| {
        try cache_buf.fill8(0xff, s.handle);
        try mla.attend(ops.mlaOps(), s, &ws[rank], &scs[rank], mlaCall(ir, cache_buf.ptr, parts_buf.ptr, 0, ir.t, rank, ir.idx_dev.ptr));
    }
    try mla.finish(ops.mlaOps(), s, &ws[0], parts_buf.ptr, world, ir.t, out_buf.ptr);
    try s.synchronize();
    try out_buf.download(0, std.mem.sliceAsBytes(got));
    const ms = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / std.time.ns_per_ms;

    const rep = try a.alloc(u16, got.len);
    defer a.free(rep);
    for (0..4) |_| {
        try idx_cache.token_keys.fill8(0xff, s.handle);
        try idx_cache.group_keys.fill8(0xff, s.handle);
        try idx_cache.rope_cs.fill8(0xff, s.handle);
        idx_cache.reset();
        try runIndexPass(s, ops, idx_w, idx_cache, idx_sc, &ws[0], &scs[0], ir, 0, ir.t, ir.groups_dev.ptr, ir.idx_dev.ptr);
        for (0..world) |rank| {
            try cache_buf.fill8(0xff, s.handle);
            try mla.attend(ops.mlaOps(), s, &ws[rank], &scs[rank], mlaCall(ir, cache_buf.ptr, parts_buf.ptr, 0, ir.t, rank, ir.idx_dev.ptr));
        }
        try mla.finish(ops.mlaOps(), s, &ws[0], parts_buf.ptr, world, ir.t, out_buf.ptr);
        try s.synchronize();
        try out_buf.download(0, std.mem.sliceAsBytes(rep));
        try check.sameBytes("hy4-index + mla repeat", std.mem.sliceAsBytes(rep), std.mem.sliceAsBytes(got));
    }

    const cut: usize = 2051;
    const cuts = [_][2]usize{ .{ 0, cut }, .{ cut, ir.t - 3 - cut }, .{ ir.t - 3, 1 }, .{ ir.t - 2, 1 }, .{ ir.t - 1, 1 } };
    try idx_cache.token_keys.fill8(0xff, s.handle);
    try idx_cache.group_keys.fill8(0xff, s.handle);
    try idx_cache.rope_cs.fill8(0xff, s.handle);
    idx_cache.reset();
    for (cuts) |seg| try runIndexPass(s, ops, idx_w, idx_cache, idx_sc, &ws[0], &scs[0], ir, seg[0], seg[1], ir.groups_seg.ptr, ir.idx_seg.ptr);
    for (0..world) |rank| {
        try cache_buf.fill8(0xff, s.handle);
        for (cuts) |seg| try mla.attend(ops.mlaOps(), s, &ws[rank], &scs[rank], mlaCall(ir, cache_buf.ptr, parts_buf.ptr, seg[0], seg[1], rank, ir.idx_seg.ptr));
    }
    try mla.finish(ops.mlaOps(), s, &ws[0], parts_buf.ptr, world, ir.t, out_buf.ptr);
    try s.synchronize();
    try out_buf.download(0, std.mem.sliceAsBytes(rep));
    try check.sameBytes("hy4-index + mla cut=2051+decode", std.mem.sliceAsBytes(rep), std.mem.sliceAsBytes(got));
    return ms;
}

fn fixtures(gpu: check.Gpu, model: []const u8, paths: []const [:0]const u8) !void {
    const a = gpu.gpa;
    const c = try hy4.Config.read(a, gpu.io, model);
    var ck = try core.Checkpoint.openModel(a, gpu.io, model);
    defer ck.close();
    var ops: Ops = undefined;
    ops.k_idx = try idx.Kernels.load(gpu.d);
    defer ops.k_idx.unload();
    ops.k_mla = try mla.Kernels.load(gpu.d);
    defer ops.k_mla.unload();
    ops.lane = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer ops.lane.unload();
    ops.norms = try cuda.norms.Norms.load(gpu.d);
    defer ops.norms.unload();
    ops.dense = try cuda.dense_rows.DenseRows.load(gpu.d);
    defer ops.dense.unload();
    ops.rope = try cuda.mla.Mla.load(gpu.d);
    defer ops.rope.unload();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var failed: usize = 0;
    for (paths) |path| failed += @intFromBool(!try fixture(gpu, &ops, s, &ck, c, path));
    try check.expect(failed == 0, "hy4-index: {d} of {d} fixtures failed", .{ failed, paths.len });
    check.pass("hy4-index: {d} indexed fixtures at world 1, 2 and 4 no worse than bf16 ref; cut=2051 + 5x repeats exact", .{paths.len});
}

fn fixture(gpu: check.Gpu, ops: *const Ops, s: cuda.Stream, ck: *core.Checkpoint, c: hy4.Config, path: []const u8) !bool {
    const a = gpu.gpa;
    const base = std.fs.path.basename(path);
    if (!std.mem.startsWith(u8, base, "layer")) return error.FixtureName;
    const layer = try std.fmt.parseInt(usize, base[5..std.mem.indexOfScalar(u8, base, '_').?], 10);
    if (c.kind[layer] != .indexed) return error.NotAnIndexedLayer;
    var f = try core.safetensors.File.open(a, gpu.io, path);
    defer f.close(gpu.io);
    const in16 = f.get("attn_in16") orelse return error.MissingTap;
    const out16 = f.get("attn_out16") orelse return error.MissingTap;
    const out64 = f.get("attn_out64") orelse return error.MissingTap;
    const r16 = f.get("recall16") orelse return error.MissingTap;
    const r64 = f.get("recall64") orelse return error.MissingTap;
    const hd = c.hidden;
    const t = in16.dim(1);
    if (in16.dim(2) != hd or out64.numel() != t * hd or t <= recall_from) return error.FixtureShape;

    const lt = try hy4.names.layer(ck, c, layer);
    var idx_w = try idx.weights.Weights.load(gpu.d, a, lt.attn.indexer.?, c);
    defer idx_w.deinit();
    var idx_cache = try idx.Cache.init(gpu.d, t);
    defer idx_cache.deinit();
    var idx_sc = try idx.Scratch.init(gpu.d, &ops.lane, &idx_w, t, t);
    defer idx_sc.deinit();
    var mla_w0 = try mla.weights.Weights.load(gpu.d, a, lt.attn, c, layer, 0, 1);
    defer mla_w0.deinit();
    var mla_sc0 = try mla.Scratch.init(gpu.d, &ops.lane, &mla_w0, t);
    defer mla_sc0.deinit();

    const n_sel = t - recall_from;
    const pos = try a.alloc(i32, t);
    defer a.free(pos);
    for (pos, 0..) |*p, i| p.* = @intCast(i);
    var ir: IndexRun = .{
        .t = t,
        .hd = hd,
        .x = try DeviceBuffer.fromHost(gpu.d, in16.bytes),
        .pos = try DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(pos)),
        .groups_dev = try DeviceBuffer.alloc(gpu.d, n_sel * groups * 4),
        .idx_dev = try DeviceBuffer.alloc(gpu.d, n_sel * recall_from * 4),
        .groups_seg = try DeviceBuffer.alloc(gpu.d, n_sel * groups * 4),
        .idx_seg = try DeviceBuffer.alloc(gpu.d, n_sel * recall_from * 4),
        .tk_ref = try a.alloc(u8, t * idx.dim * 2),
        .gk_ref = try a.alloc(u8, (t / ratio) * idx.dim * 2),
        .grp_ref = try a.alloc(u8, n_sel * groups * 4),
        .idx_ref = try a.alloc(u8, n_sel * recall_from * 4),
    };
    defer ir.deinit(a);

    try checkIndexerParity(gpu, s, ops, &idx_w, &idx_cache, &idx_sc, &mla_w0, &mla_sc0, &ir);
    const diff16 = recallSetDiffers(ir.grp_ref, r16.bytes, t);
    const diff64 = recallSetDiffers(ir.grp_ref, r64.bytes, t);
    const ref_diff = blk: {
        var n: usize = 0;
        for (recall_from..t) |r| {
            var x: [groups]i32 = undefined;
            var y: [groups]i32 = undefined;
            for (&x, &y, 0..) |*xv, *yv, j| {
                xv.* = std.mem.readInt(i32, r16.bytes[(r * groups + j) * 4 ..][0..4], .little);
                yv.* = std.mem.readInt(i32, r64.bytes[(r * groups + j) * 4 ..][0..4], .little);
            }
            std.mem.sort(i32, &x, {}, std.sort.asc(i32));
            std.mem.sort(i32, &y, {}, std.sort.asc(i32));
            n += @intFromBool(!std.mem.eql(i32, &x, &y));
        }
        break :blk n;
    };

    var cache_buf = try DeviceBuffer.alloc(gpu.d, t * mla.row * 2);
    defer cache_buf.free();
    var parts_buf = try DeviceBuffer.alloc(gpu.d, 4 * t * hd * 4);
    defer parts_buf.free();
    var out_buf = try DeviceBuffer.alloc(gpu.d, t * hd * 2);
    defer out_buf.free();
    const torch16 = try a.alloc(u16, out16.bytes.len / 2);
    defer a.free(torch16);
    @memcpy(std.mem.sliceAsBytes(torch16), out16.bytes);
    const ref = try errors(a, torch16, out64.bytes, t, hd);
    const got = try a.alloc(u16, t * hd);
    defer a.free(got);

    std.debug.print("INFO {s}: layer {d}, {d} rows; live recall diffs: {d}/{d} vs recall16, {d}/{d} vs recall64 (torch recall16 vs recall64: {d}); torch bf16 vs fp64 max {e:.3} mean {e:.3}, worst col {d} mean {e:.3}\n", .{ base, layer, t, diff16, n_sel, diff64, n_sel, ref_diff, ref.max, ref.mean, ref.at, ref.col });

    var most: f64 = 0;
    for (0..t * hd) |i| most = @max(most, @abs(@as(f64, @bitCast(std.mem.readInt(u64, out64.bytes[8 * i ..][0..8], .little)))));
    const slack = std.math.ldexp(most, -9);
    var ok = true;
    const heads_total = c.heads(layer).count;
    for ([_]usize{ 1, 2, 4 }) |world| {
        const ms = try forwardWorld(gpu, ops, s, ck, c, layer, &idx_w, &idx_cache, &idx_sc, &ir, cache_buf, parts_buf, out_buf, world, got);
        const e = try errors(a, got, out64.bytes, t, hd);
        const pass = e.mean <= ref.mean and e.max <= ref.max + slack and e.col <= ref.col;
        ok = ok and pass;
        std.debug.print("{s} {s} world {d} ({d} heads/rank): native(live idx)+mla vs fp64 max {e:.3} mean {e:.3}, worst col {d} mean {e:.3} ({d:.0} ms)\n", .{ if (pass) "PASS" else "FAIL", base, world, heads_total / world, e.max, e.mean, e.at, e.col, ms });
    }
    return ok;
}
