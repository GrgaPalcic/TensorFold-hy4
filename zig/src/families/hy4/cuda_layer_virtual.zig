//! Single-GPU virtual-rank (`world` 1, 2, 4) DecoderLayer runner for Hy4-flash verification suites.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const names = @import("names.zig");
const partition = @import("partition.zig");
const cuda_mla = @import("cuda_mla.zig");
const cuda_mlp = @import("cuda_mlp.zig");
const cl = @import("cuda_layer.zig");

const Config = config.Config;
const Stream = cuda.Stream;

/// Runs `ranks` (`world` 1, 2 or 4 `Layer` slices on one GPU), summing partial `o_proj` and `mlp` planes at seams.
pub fn forwardVirtual(k: *const cl.Kernels, s: Stream, c: Config, ranks: []const cl.Layer, sc: *const cl.Scratch, call: cl.Call) !void {
    const world = ranks.len;
    if ((world != 1 and world != 2 and world != 4) or world != sc.world) return error.UnsupportedWorld;
    if (c.hidden != sc.hidden or call.rows > sc.max_rows) return error.ScratchShape;
    const head = &ranks[0];
    for (ranks, 0..) |*rk, i| {
        if (rk.layer_idx != head.layer_idx or rk.at.rank != i or rk.at.world != world) return error.BadRankSet;
    }
    var one: [1]cl.Seg = undefined;
    const segs = try cl.checkSegs(c, head.layer_idx, call, &one);
    if (call.rows == 0) return;

    const d = c.hidden;
    const rows = call.rows;
    const plane = rows * d * 4;
    const attn_streams = try cl.runEngram(head, sc, s, call);
    try k.hc.pre(s, head.hc_attn, attn_streams, sc.mixed.ptr, sc.post.ptr, rows, d, .production, 0);
    try k.mlp.norms.addRmsH(s, sc.mixed.ptr, null, head.input_norm, .bf16, 0, sc.normed.ptr, null, rows, d, c.eps);

    const mla_sc = sc.mlaFor(head.is_sliding);
    try cuda_mla.qkvPrep(k.mlaOps(), s, &head.mla, mla_sc, sc.normed.ptr, rows, null, &k.mlp.rows);
    var msegs: [cl.max_segs]cuda_mla.Seg = undefined;
    try cl.runIndexer(head, k, s, c, sc, mla_sc, call, segs, msegs[0..segs.len]);
    for (ranks, 0..) |*rk, i| {
        try cuda_mla.attendCore(k.mlaOps(), s, &rk.mla, mla_sc, sc.normed.ptr, rows, call.pos, msegs[0..segs.len], call.caches, true, call.single_split_attn);
        try cuda_mla.attendOut(k.mlaOps(), s, &rk.mla, mla_sc, sc.normed.ptr, rows, sc.partial.ptr + i * plane);
    }
    try cuda_mla.finish(k.mlaOps(), s, &head.mla, sc.partial.ptr, world, rows, sc.branch.ptr);
    try k.hc.merge(s, attn_streams, sc.branch.ptr, sc.post.ptr, sc.mid.ptr, rows, d, .production, 0);

    try k.hc.pre(s, head.hc_mlp, sc.mid.ptr, sc.mixed.ptr, sc.post.ptr, rows, d, .production, 0);
    try cuda_mlp.norm(&k.mlp, s, c, head.mlp, sc.mixed.ptr, sc.normed.ptr, rows);
    const is_prompt = call.prompt orelse (rows > 128);
    for (ranks, 0..) |*rk, i| {
        try cuda_mlp.partial(&k.mlp, s, c, rk.mlp, &sc.mlp, sc.normed.ptr, sc.partial.ptr + i * plane, rows, is_prompt);
    }
    const mlp_sum = if (world == 1) sc.partial.ptr else blk: {
        try cuda_mlp.rankSum(&k.mlp, s, sc.partial.ptr, world, rows, d, sc.reduced.ptr);
        break :blk sc.reduced.ptr;
    };
    try cuda_mlp.finish(&k.mlp, s, c, head.mlp, mlp_sum, sc.branch.ptr, rows);
    try k.hc.merge(s, sc.mid.ptr, sc.branch.ptr, sc.post.ptr, call.streams_out, rows, d, .production, 0);
}

/// Convenience bundle holding all `world` virtual rank `Layer` slices for single-GPU TP verification.
pub const WorldLayer = struct {
    world: usize,
    ranks: [partition.max_world]cl.Layer = undefined,

    pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, l: names.Layer, c: Config, layer_idx: usize, world: usize) !WorldLayer {
        if (world != 1 and world != 2 and world != 4) return error.UnsupportedWorld;
        var wl: WorldLayer = .{ .world = world };
        var n: usize = 0;
        errdefer for (wl.ranks[0..n]) |*r| r.deinit();
        for (0..world) |r| {
            wl.ranks[r] = try cl.Layer.load(gpa, d, l, c, layer_idx, .{ .rank = r, .world = world });
            n += 1;
        }
        return wl;
    }

    pub fn slice(wl: *const WorldLayer) []const cl.Layer {
        return wl.ranks[0..wl.world];
    }

    pub fn forward(wl: *const WorldLayer, k: *const cl.Kernels, s: Stream, c: Config, sc: *const cl.Scratch, call: cl.Call) !void {
        return forwardVirtual(k, s, c, wl.slice(), sc, call);
    }

    pub fn deinit(wl: *WorldLayer) void {
        for (wl.ranks[0..wl.world]) |*r| r.deinit();
        wl.* = undefined;
    }
};
