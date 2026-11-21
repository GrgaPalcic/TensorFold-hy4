//! tf-cuda-test hy4-mla MODEL FIXTURE... (worlds 1, 2, 4 vs fp64/bf16 reference) and hy4-mla-contract (gates 1-5).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4");
const check = @import("check.zig");

const mla = hy4.cuda_mla;
const Weights = mla.weights.Weights;
const DeviceBuffer = cuda.DeviceBuffer;
const recall_from = 2048; // compressed_indexer_candidate_topk: an indexed row at or past it reads the indexer's list
const groups = 512; // the groups a row recalls, its own last
const ratio = 4; // tokens a group

pub fn run(gpu: check.Gpu, cmd: []const u8, rest: []const [:0]const u8) !void {
    if (std.mem.eql(u8, cmd, "hy4-mla-contract")) return @import("contract_hy4_mla.zig").run(gpu);
    if (std.mem.eql(u8, cmd, "hy4-mla")) {
        if (rest.len < 2) return error.MissingArgument;
        return fixtures(gpu, rest[0], rest[1..]);
    }
    return error.UnknownCommand;
}

const Ops = struct {
    k: mla.Kernels,
    lane: cuda.qmmf.Lane,
    norms: cuda.norms.Norms,
    rope: cuda.mla.Mla,

    fn view(o: *const Ops) mla.Ops {
        return .{ .k = &o.k, .lane = &o.lane, .norms = &o.norms, .mla = &o.rope };
    }
};

/// One fixture on the device: attn_in16 rows, positions, the indexer's lists, a cache, the partials and the output.
const Rows = struct {
    t: usize,
    x: DeviceBuffer,
    pos: DeviceBuffer,
    idx: ?DeviceBuffer,
    cache: DeviceBuffer,
    parts: DeviceBuffer,
    out: DeviceBuffer,

    fn deinit(r: *Rows) void {
        for ([_]*DeviceBuffer{ &r.x, &r.pos, &r.cache, &r.parts, &r.out }) |b| b.free();
        if (r.idx) |*b| b.free();
    }
};

/// Errors against fp64: the largest, the mean, and the worst column's mean.
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

fn bf16s(a: std.mem.Allocator, bytes: []const u8) ![]u16 {
    const out = try a.alloc(u16, bytes.len / 2);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

/// The lists rows recall_from..t-1 read: each recalled group's four tokens (recall16, the bf16 run's picks).
fn lists(a: std.mem.Allocator, recall: []const u8, t: usize) ![]i32 {
    const n = t - recall_from;
    const out = try a.alloc(i32, n * groups * ratio);
    for (0..n) |r| for (0..groups) |j| {
        const gi = std.mem.readInt(i32, recall[((recall_from + r) * groups + j) * 4 ..][0..4], .little);
        for (0..ratio) |o| out[(r * groups + j) * ratio + o] = gi * ratio + @as(i32, @intCast(o));
    };
    return out;
}

/// Rows at or past recall_from whose recalled groups differ between the bf16 and fp64 runs (as sets).
fn recallDiffers(recall16: []const u8, recall64: []const u8, t: usize) usize {
    var n: usize = 0;
    for (recall_from..t) |r| {
        var a: [groups]i32 = undefined;
        var b: [groups]i32 = undefined;
        for (&a, &b, 0..) |*x, *y, j| {
            x.* = std.mem.readInt(i32, recall16[(r * groups + j) * 4 ..][0..4], .little);
            y.* = std.mem.readInt(i32, recall64[(r * groups + j) * 4 ..][0..4], .little);
        }
        std.mem.sort(i32, &a, {}, std.sort.asc(i32));
        std.mem.sort(i32, &b, {}, std.sort.asc(i32));
        n += @intFromBool(!std.mem.eql(i32, &a, &b));
    }
    return n;
}

/// Rows [lo, lo + n) of the fixture as one call of rank `rank`'s partial plane.
fn call(r: *const Rows, indexed: bool, hd: usize, lo: usize, n: usize, rank: usize) mla.Call {
    var c: mla.Call = .{ .x = r.x.ptr + lo * hd * 2, .rows = n, .first = lo, .pos = r.pos.ptr + lo * 4, .cache = r.cache.ptr, .part = r.parts.ptr + (rank * r.t + lo) * hd * 4, .single_split = r.t > 64 };
    if (indexed and r.idx != null and lo + n > recall_from) {
        const from = @max(lo, recall_from);
        c.list_from = from - lo;
        c.idx = r.idx.?.ptr + (from - recall_from) * groups * ratio * 4;
        c.nidx = groups * ratio;
    }
    return c;
}

/// Every rank of `world` through attend, then post: 5 byte-identical runs, plus prompt chunks and 3 decode rows.
fn forward(gpu: check.Gpu, ops: *const Ops, s: cuda.Stream, ck: *core.Checkpoint, c: hy4.Config, layer: usize, r: *Rows, world: usize, got: []u16) !f64 {
    const a = gpu.gpa;
    const t0 = check.now(gpu.io);
    const lt = try hy4.names.layer(ck, c, layer);
    var ws: [4]Weights = undefined;
    var scs: [4]mla.Scratch = undefined;
    var loaded: usize = 0;
    defer for (0..loaded) |i| {
        scs[i].deinit();
        ws[i].deinit();
    };
    for (0..world) |rank| {
        ws[rank] = try Weights.load(gpu.d, a, lt.attn, c, layer, rank, world);
        errdefer ws[rank].deinit();
        scs[rank] = try mla.Scratch.init(gpu.d, &ops.lane, &ws[rank], r.t);
        loaded += 1;
    }
    const indexed = c.kind[layer] == .indexed;
    for (0..world) |rank| {
        try r.cache.fill8(0xff, s.handle);
        try mla.attend(ops.view(), s, &ws[rank], &scs[rank], call(r, indexed, c.hidden, 0, r.t, rank));
    }
    try mla.finish(ops.view(), s, &ws[0], r.parts.ptr, world, r.t, r.out.ptr);
    try s.synchronize();
    try r.out.download(0, std.mem.sliceAsBytes(got));
    const ms = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / std.time.ns_per_ms;
    const rep = try a.alloc(u16, got.len);
    defer a.free(rep);
    for (0..4) |_| {
        for (0..world) |rank| {
            try r.cache.fill8(0xff, s.handle);
            try mla.attend(ops.view(), s, &ws[rank], &scs[rank], call(r, indexed, c.hidden, 0, r.t, rank));
        }
        try mla.finish(ops.view(), s, &ws[0], r.parts.ptr, world, r.t, r.out.ptr);
        try s.synchronize();
        try r.out.download(0, std.mem.sliceAsBytes(rep));
        try check.sameBytes("hy4-mla repeat", std.mem.sliceAsBytes(rep), std.mem.sliceAsBytes(got));
    }
    const cut = r.t / 2 + 7;
    const cuts = [_][2]usize{ .{ 0, cut }, .{ cut, r.t - 3 - cut }, .{ r.t - 3, 1 }, .{ r.t - 2, 1 }, .{ r.t - 1, 1 } };
    for (0..world) |rank| {
        try r.cache.fill8(0xff, s.handle);
        for (cuts) |p| try mla.attend(ops.view(), s, &ws[rank], &scs[rank], call(r, indexed, c.hidden, p[0], p[1], rank));
    }
    try mla.finish(ops.view(), s, &ws[0], r.parts.ptr, world, r.t, r.out.ptr);
    try s.synchronize();
    try r.out.download(0, std.mem.sliceAsBytes(rep));
    try check.sameBytes("hy4-mla prompt chunks and decode rows", std.mem.sliceAsBytes(rep), std.mem.sliceAsBytes(got));
    return ms;
}

fn fixtures(gpu: check.Gpu, model: []const u8, paths: []const [:0]const u8) !void {
    const a = gpu.gpa;
    const c = try hy4.Config.read(a, gpu.io, model);
    var ck = try core.Checkpoint.openModel(a, gpu.io, model);
    defer ck.close();
    var ops: Ops = undefined;
    ops.k = try mla.Kernels.load(gpu.d);
    defer ops.k.unload();
    ops.lane = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer ops.lane.unload();
    ops.norms = try cuda.norms.Norms.load(gpu.d);
    defer ops.norms.unload();
    ops.rope = try cuda.mla.Mla.load(gpu.d);
    defer ops.rope.unload();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var failed: usize = 0;
    for (paths) |path| failed += @intFromBool(!try fixture(gpu, &ops, s, &ck, c, path));
    try check.expect(failed == 0, "hy4-mla: {d} of {d} fixtures worse than the bf16 reference or not repeatable", .{ failed, paths.len });
    check.pass("hy4-mla: {d} fixtures at world 1, 2 and 4 no worse than the bf16 reference; repeats and row cuts byte-equal", .{paths.len});
}

fn fixture(gpu: check.Gpu, ops: *const Ops, s: cuda.Stream, ck: *core.Checkpoint, c: hy4.Config, path: []const u8) !bool {
    const a = gpu.gpa;
    const base = std.fs.path.basename(path);
    if (!std.mem.startsWith(u8, base, "layer")) return error.FixtureName;
    const layer = try std.fmt.parseInt(usize, base[5..std.mem.indexOfScalar(u8, base, '_').?], 10);
    var f = try core.safetensors.File.open(a, gpu.io, path);
    defer f.close(gpu.io);
    const in16 = f.get("attn_in16") orelse return error.MissingTap;
    const out16 = f.get("attn_out16") orelse return error.MissingTap;
    const out64 = f.get("attn_out64") orelse return error.MissingTap;
    const hd = c.hidden;
    const t = in16.dim(1);
    if (in16.dim(2) != hd or out64.numel() != t * hd) return error.FixtureShape;
    const indexed = c.kind[layer] == .indexed;
    var rows: Rows = undefined;
    rows.t = t;
    rows.x = try DeviceBuffer.fromHost(gpu.d, in16.bytes);
    const pos = try a.alloc(i32, t);
    defer a.free(pos);
    for (pos, 0..) |*p, i| p.* = @intCast(i);
    rows.pos = try DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(pos));
    rows.idx = null;
    var differ: usize = 0;
    if (indexed and t > recall_from) {
        const r16 = f.get("recall16") orelse return error.MissingTap;
        const r64 = f.get("recall64") orelse return error.MissingTap;
        const l = try lists(a, r16.bytes, t);
        defer a.free(l);
        rows.idx = try DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(l));
        differ = recallDiffers(r16.bytes, r64.bytes, t);
    }
    rows.cache = try DeviceBuffer.alloc(gpu.d, t * mla.row * 2);
    rows.parts = try DeviceBuffer.alloc(gpu.d, 4 * t * hd * 4);
    rows.out = try DeviceBuffer.alloc(gpu.d, t * hd * 2);
    defer rows.deinit();
    const torch16 = try bf16s(a, out16.bytes);
    defer a.free(torch16);
    const ref = try errors(a, torch16, out64.bytes, t, hd);
    const got = try a.alloc(u16, t * hd);
    defer a.free(got);
    const kind = if (indexed) "indexed" else "sliding";
    const heads = c.heads(layer).count;
    std.debug.print("INFO {s}: layer {d} ({s}, {d} heads), {d} rows; torch bf16 vs fp64 max {e:.3} mean {e:.3}, worst column {d} mean {e:.3}{s}", .{ base, layer, kind, heads, t, ref.max, ref.mean, ref.at, ref.col, if (rows.idx != null) "" else "\n" });
    if (rows.idx != null) std.debug.print("; rows {d} on read recall16, {d} of {d} recall other groups than recall64\n", .{ recall_from, differ, t - recall_from });
    var ok = true;
    var most: f64 = 0;
    for (0..t * hd) |i| most = @max(most, @abs(@as(f64, @bitCast(std.mem.readInt(u64, out64.bytes[8 * i ..][0..8], .little)))));
    const slack = std.math.ldexp(most, -9); // half a bf16 ulp of the largest output: the rounding every bf16 row carries
    for ([_]usize{ 1, 2, 4 }) |world| {
        const ms = forward(gpu, ops, s, ck, c, layer, &rows, world, got) catch |e| switch (e) {
            error.TestFailed => return false,
            else => return e,
        };
        const e = try errors(a, got, out64.bytes, t, hd);
        const pass = e.mean <= ref.mean and e.max <= ref.max + slack and e.col <= ref.col;
        ok = ok and pass;
        std.debug.print("{s} {s} world {d} ({d} heads a rank): native vs fp64 max {e:.3} mean {e:.3}, worst column {d} mean {e:.3} ({d:.0} ms with loads)\n", .{ if (pass) "PASS" else "FAIL", base, world, heads / world, e.max, e.mean, e.at, e.col, ms });
    }
    check.pass("{s}: five runs and prompt/decode cuts byte-identical at world 1, 2 and 4", .{base});
    return ok;
}
