//! Hy4-flash's Engram sublayer on one rank: row lookup (NVMe or resident), per-head norm, 5 projections, 4-stream gate.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const ep = @import("engram_pack.zig");
const er = @import("engram_reader.zig");
const partition = @import("partition.zig");
const weights = @import("cuda_weights.zig");

const Config = config.Config;
const qmmf = cuda.qmmf;
const launch = cuda.launch;
const Stream = cuda.Stream;
const DeviceBuffer = cuda.DeviceBuffer;

pub const streams = 4;
pub const projections = 1 + streams;
pub const threads = 256;
pub const warps = threads / 32;
pub const norm_eps: f32 = std.math.floatEps(f32); // torch.nn.RMSNorm(eps=None) uses finfo(float32).eps

pub const Control = enum { production, nan, repeat, tail, sums };

pub const Ops = struct {
    mod: cuda.Module,
    embed_fns: [5]cuda.Function,
    bias_fns: [4]cuda.Function,
    gate_fns: [5]cuda.Function,

    pub fn load(d: *const cuda.Driver) !Ops {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.hy4_engram);
        errdefer mod.unload();
        var o: Ops = .{ .mod = mod, .embed_fns = undefined, .bias_fns = undefined, .gate_fns = undefined };
        for ([_][:0]const u8{ "hy4_engram_embed", "hy4_engram_embed_bad_nan", "hy4_engram_embed_bad_repeat", "hy4_engram_embed_bad_tail", "hy4_engram_embed_bad_sums" }, 0..) |n, i| o.embed_fns[i] = try mod.function(n);
        for ([_][:0]const u8{ "hy4_engram_bias", "hy4_engram_bias_bad_nan", "hy4_engram_bias_bad_repeat", "hy4_engram_bias_bad_tail" }, 0..) |n, i| o.bias_fns[i] = try mod.function(n);
        for ([_][:0]const u8{ "hy4_engram_gate", "hy4_engram_gate_bad_nan", "hy4_engram_gate_bad_repeat", "hy4_engram_gate_bad_tail", "hy4_engram_gate_bad_sums" }, 0..) |n, i| o.gate_fns[i] = try mod.function(n);
        return o;
    }

    pub fn unload(o: *Ops) void {
        o.mod.unload();
    }

    pub fn embed(o: *const Ops, s: Stream, raw: u64, head_norms: u64, eps: f32, out: u64, rows: usize, heads: usize, dim: usize, ctl: Control, repeat: i32) !void {
        const pairs = rows * heads;
        if (pairs == 0 or dim == 0) return;
        var a: launch.Args = .{};
        a.add(raw);
        a.add(head_norms);
        a.add(eps);
        a.add(out);
        a.add(@as(i32, @intCast(pairs)));
        a.add(@as(i32, @intCast(heads)));
        a.add(@as(i32, @intCast(dim)));
        a.add(repeat);
        return launch.launch(o.embed_fns[@backingInt(ctl)], .{ .grid = .{ .x = @intCast((pairs + warps - 1) / warps), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }

    pub fn bias(o: *const Ops, s: Stream, x: u64, b: u64, out: u64, rows: usize, cols: usize, ctl: Control, repeat: i32) !void {
        const count = rows * cols;
        if (count == 0) return;
        var a: launch.Args = .{};
        a.add(x);
        a.add(b);
        a.add(out);
        a.add(@as(i32, @intCast(rows)));
        a.add(@as(i32, @intCast(cols)));
        a.add(repeat);
        return launch.launch(o.bias_fns[@backingInt(ctl)], .{ .grid = .{ .x = @intCast((count + threads - 1) / threads), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }

    pub const GateWeights = struct { norm_keys: u64, norm_queries: u64, pre_gamma: u64, stream_scales: u64, eps: f32 = norm_eps };

    pub fn gate(o: *const Ops, s: Stream, gw: GateWeights, streams_in: u64, proj: u64, mask: u64, engram_out: u64, streams_out: u64, rows: usize, hidden: usize, ctl: Control, repeat: i32) !void {
        if (rows == 0 or hidden == 0) return;
        var a: launch.Args = .{};
        a.add(streams_in);
        a.add(proj);
        a.add(gw.norm_keys);
        a.add(gw.norm_queries);
        a.add(gw.pre_gamma);
        a.add(gw.stream_scales);
        a.add(mask);
        a.add(gw.eps);
        a.add(engram_out);
        a.add(streams_out);
        a.add(@as(i32, @intCast(hidden)));
        a.add(repeat);
        return launch.launch(o.gate_fns[@backingInt(ctl)], .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }
};

pub const Kernels = struct {
    lane: qmmf.Lane,
    ops: Ops,
    sum: cuda.rank_scatter.RankScatter,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        const d = ctx.d;
        var lane = try qmmf.Lane.load(d, (try ctx.capability()) / 10);
        errdefer lane.unload();
        try lane.loadPrompt(d);
        var ops = try Ops.load(d);
        errdefer ops.unload();
        const sum = try cuda.rank_scatter.RankScatter.load(d);
        return .{ .lane = lane, .ops = ops, .sum = sum };
    }

    pub fn deinit(k: *Kernels) void {
        k.sum.unload();
        k.ops.unload();
        k.lane.unload();
        k.* = undefined;
    }
};

/// Host + device scratch for calls of up to `rows` tokens at any world 1, 2 or 4 (sized by world 1's 16 heads).
pub const Scratch = struct {
    gpa: std.mem.Allocator,
    rows: usize,
    idx: []i64,
    mask: []u8,
    host_rows: []u8,
    dev_raw: DeviceBuffer,
    dev_mask: DeviceBuffer,
    dev_emb: DeviceBuffer,
    dev_part: DeviceBuffer,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lane: *const qmmf.Lane, c: Config, rows: usize) !Scratch {
        const ext_rows = rows + config.engram_orders;
        const max_pairs = ext_rows * config.engram_heads;
        const idx = try gpa.alloc(i64, @max(max_pairs, 1));
        errdefer gpa.free(idx);
        const mask = try gpa.alloc(u8, @max(ext_rows, 1));
        errdefer gpa.free(mask);
        const host_rows = try gpa.alloc(u8, @max(rows * config.engram_heads * ep.row_bytes, 16));
        errdefer gpa.free(host_rows);
        var dev_raw = try DeviceBuffer.alloc(d, @max(rows * config.engram_heads * ep.row_bytes, 16));
        errdefer dev_raw.free();
        var dev_mask = try DeviceBuffer.alloc(d, @max(rows, 16));
        errdefer dev_mask.free();
        var dev_emb = try DeviceBuffer.alloc(d, @max(rows * config.engram_heads * c.engram_dim * 2, 16));
        errdefer dev_emb.free();
        const dev_part = try DeviceBuffer.alloc(d, @max(try partBytes(lane, c, rows), 16));
        return .{ .gpa = gpa, .rows = rows, .idx = idx, .mask = mask, .host_rows = host_rows, .dev_raw = dev_raw, .dev_mask = dev_mask, .dev_emb = dev_emb, .dev_part = dev_part };
    }

    pub fn deinit(sc: *Scratch) void {
        sc.dev_part.free();
        sc.dev_emb.free();
        sc.dev_mask.free();
        sc.dev_raw.free();
        sc.gpa.free(sc.host_rows);
        sc.gpa.free(sc.mask);
        sc.gpa.free(sc.idx);
        sc.* = undefined;
    }
};

fn partBytes(lane: *const qmmf.Lane, c: Config, rows: usize) !usize {
    var most: usize = 0;
    const n = projections * c.hidden;
    for ([_]usize{ 1, 2, 4 }) |world| {
        const k = (try partition.Engram.of(c, 0, world)).cols.len;
        const w = cuda.modelopt.bf16Weight(0, n, k);
        var m: usize = 1;
        while (m <= rows) : (m = if (m < 64) m + 1 else m * 2) most = @max(most, lane.partBytes(m, w));
        most = @max(most, lane.partBytes(rows, w));
    }
    return most;
}

/// Starts async NVMe prefetch of `[skip .. tokens.len)` using `0..skip` as n-gram history prefix.
pub fn prefetchWindow(c: Config, w: weights.Engram, r: *er.Reader, sc: *Scratch, tokens: []const u32, positions: []const u32, skip: usize) !void {
    if (tokens.len <= skip) return;
    const rows = tokens.len - skip;
    if (rows > sc.rows or skip > config.engram_orders) return error.TooManyRows;
    const slot_idx = r.indexOfSlot(w.slot) orelse return error.MissingEngramLayer;
    const heads = w.part.heads.len;
    const all_pairs = tokens.len * heads;
    try er.hashRank(c, w.slot, w.part, w.map, tokens, positions, sc.idx[0..all_pairs], sc.mask[0..tokens.len]);
    try r.issue(slot_idx, sc.idx[skip * heads ..][0 .. rows * heads]);
}

/// Starts one merged async NVMe prefetch across all `wins` for `w.slot` (deduplicating ids & merging 4 KiB extents).
pub fn prefetchWindows(c: Config, w: weights.Engram, r: *er.Reader, sc: *Scratch, wins: []const Window) !void {
    if (wins.len == 1) return prefetchWindow(c, w, r, sc, wins[0].tokens, wins[0].positions, wins[0].skip);
    const slot_idx = r.indexOfSlot(w.slot) orelse return error.MissingEngramLayer;
    const heads = w.part.heads.len;
    var rows: usize = 0;
    for (wins) |win| {
        if (win.tokens.len <= win.skip) continue;
        const n = win.tokens.len - win.skip;
        if (rows + n > sc.rows or win.skip > config.engram_orders) return error.TooManyRows;
        try er.hashRank(c, w.slot, w.part, w.map, win.tokens, win.positions, sc.idx[rows * heads ..][0 .. win.tokens.len * heads], sc.mask[rows..][0..win.tokens.len]);
        if (win.skip > 0) std.mem.copyForwards(i64, sc.idx[rows * heads ..][0 .. n * heads], sc.idx[(rows + win.skip) * heads ..][0 .. n * heads]);
        rows += n;
    }
    if (rows == 0) return;
    try r.issue(slot_idx, sc.idx[0 .. rows * heads]);
}

/// Starts async NVMe prefetch of this rank's Engram rows for `tokens` at `positions`.
pub fn prefetch(c: Config, w: weights.Engram, r: *er.Reader, sc: *Scratch, tokens: []const u32, positions: []const u32) !void {
    return prefetchWindow(c, w, r, sc, tokens, positions, 0);
}

/// Rank partial sums `out` (`fp32 [rows, 5 * hidden]`) for `tokens[skip..]` using `0..skip` as n-gram history.
pub fn partialWindow(k: *const Kernels, s: Stream, c: Config, w: weights.Engram, r: *er.Reader, sc: *Scratch, tokens: []const u32, positions: []const u32, skip: usize, out: u64) !void {
    if (tokens.len <= skip) return;
    const rows = tokens.len - skip;
    if (rows > sc.rows or skip > config.engram_orders) return error.TooManyRows;
    const slot_idx = r.indexOfSlot(w.slot) orelse return error.MissingEngramLayer;
    const heads = w.part.heads.len;
    const all_pairs = tokens.len * heads;
    const pairs = rows * heads;
    const raw_bytes = pairs * ep.row_bytes;
    try er.hashRank(c, w.slot, w.part, w.map, tokens, positions, sc.idx[0..all_pairs], sc.mask[0..tokens.len]);
    if (skip > 0) std.mem.copyForwards(u8, sc.mask[0..rows], sc.mask[skip..][0..rows]);
    try r.fetch(slot_idx, sc.idx[skip * heads ..][0..pairs], sc.host_rows[0..raw_bytes]);
    try sc.dev_raw.uploadAsync(0, sc.host_rows[0..raw_bytes], s.handle);
    try k.ops.embed(s, sc.dev_raw.ptr, w.head_norms, norm_eps, sc.dev_emb.ptr, rows, heads, c.engram_dim, .production, 0);
    try k.lane.matmulF32(s, sc.dev_emb.ptr, w.part.cols.len, rows, w.proj, out, sc.dev_part.ptr);
    if (w.bias) |b| try k.ops.bias(s, out, b, out, rows, projections * c.hidden, .production, 0);
}

/// One sequence's n-gram window: `tokens[skip..]` are its rows, `tokens[0..skip]` their history.
pub const Window = struct { tokens: []const u32, positions: []const u32, skip: usize };

/// `partialWindow` over several sequences' windows: rows in window order, one fetch, upload and matmul for all.
pub fn partialWindows(k: *const Kernels, s: Stream, c: Config, w: weights.Engram, r: *er.Reader, sc: *Scratch, wins: []const Window, out: u64) !void {
    if (wins.len == 1) return partialWindow(k, s, c, w, r, sc, wins[0].tokens, wins[0].positions, wins[0].skip, out);
    const slot_idx = r.indexOfSlot(w.slot) orelse return error.MissingEngramLayer;
    const heads = w.part.heads.len;
    var rows: usize = 0;
    for (wins) |win| {
        if (win.tokens.len <= win.skip) continue;
        const n = win.tokens.len - win.skip;
        if (rows + n > sc.rows or win.skip > config.engram_orders) return error.TooManyRows;
        try er.hashRank(c, w.slot, w.part, w.map, win.tokens, win.positions, sc.idx[rows * heads ..][0 .. win.tokens.len * heads], sc.mask[rows..][0..win.tokens.len]);
        if (win.skip > 0) {
            std.mem.copyForwards(i64, sc.idx[rows * heads ..][0 .. n * heads], sc.idx[(rows + win.skip) * heads ..][0 .. n * heads]);
            std.mem.copyForwards(u8, sc.mask[rows..][0..n], sc.mask[rows + win.skip ..][0..n]);
        }
        rows += n;
    }
    if (rows == 0) return;
    const pairs = rows * heads;
    const raw_bytes = pairs * ep.row_bytes;
    try r.fetch(slot_idx, sc.idx[0..pairs], sc.host_rows[0..raw_bytes]);
    try sc.dev_raw.uploadAsync(0, sc.host_rows[0..raw_bytes], s.handle);
    try k.ops.embed(s, sc.dev_raw.ptr, w.head_norms, norm_eps, sc.dev_emb.ptr, rows, heads, c.engram_dim, .production, 0);
    try k.lane.matmulF32(s, sc.dev_emb.ptr, w.part.cols.len, rows, w.proj, out, sc.dev_part.ptr);
    if (w.bias) |b| try k.ops.bias(s, out, b, out, rows, projections * c.hidden, .production, 0);
}

/// Rank's partial sums `out` (`fp32 [tokens.len, 5 * hidden]`): fetch 100 B rows, per-head norm, matmul, rank-0 bias.
pub fn partial(k: *const Kernels, s: Stream, c: Config, w: weights.Engram, r: *er.Reader, sc: *Scratch, tokens: []const u32, positions: []const u32, out: u64) !void {
    return partialWindow(k, s, c, w, r, sc, tokens, positions, 0, out);
}

/// Sums rank partials `[world, rows, 5 * hidden]` fp32 in rank order into `out` (`[rows, 5 * hidden]`).
pub fn rankSum(k: *const Kernels, s: Stream, partials: u64, world: usize, rows: usize, hidden: usize, out: u64) !void {
    try k.sum.rankSum(s, partials, partials, world, 0, rows * projections * hidden, out);
}

/// Applies unified RMSNorm, 4-stream signed-sqrt gates and mask (`sc.mask` from `partial` or `mask_override`).
pub fn finish(k: *const Kernels, s: Stream, c: Config, w: weights.Engram, sc: *Scratch, rows: usize, mask_override: ?[]const u8, streams_in: u64, sum: u64, engram_out: u64, streams_out: u64) !void {
    if (rows == 0) return;
    if (rows > sc.rows) return error.TooManyRows;
    const m = mask_override orelse sc.mask[0..rows];
    if (m.len != rows) return error.BadEngramArgs;
    try sc.dev_mask.uploadAsync(0, m, s.handle);
    const gw: Ops.GateWeights = .{ .norm_keys = w.norm_keys, .norm_queries = w.norm_queries, .pre_gamma = w.pre_gamma, .stream_scales = w.stream_scales };
    try k.ops.gate(s, gw, streams_in, sum, sc.dev_mask.ptr, engram_out, streams_out, rows, c.hidden, .production, 0);
}

test "cuda_engram launchers compile on host without embedded fatbins" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Ops);
    std.testing.refAllDecls(Kernels);
    std.testing.refAllDecls(Scratch);
}
