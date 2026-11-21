//! Hy4-flash's MLP sublayer on one rank: post_attention_layernorm, the dense FFN or the LatentMoE, the all-reduce seam.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const partition = @import("partition.zig");
const weights = @import("cuda_weights.zig");
pub const cuda_rows = @import("cuda_rows.zig");
const cuda_route = @import("cuda_route.zig");
const cuda_prof = @import("cuda_prof.zig");

const Config = config.Config;
const grouped = cuda.grouped;
const experts = cuda.experts;
const qmmf = cuda.qmmf;
const Stream = cuda.Stream;
const DeviceBuffer = cuda.DeviceBuffer;

/// The shared operators the sublayer launches; nothing of the family's own.
pub const Kernels = struct {
    lane: qmmf.Lane,
    fp8x: experts.Fp8Experts,
    plan_mod: cuda.Module, // experts.cu: the plan kernels
    router: grouped.Router,
    norms: cuda.norms.Norms,
    moe: cuda.moe_glue.MoeGlue,
    route: cuda_route.Route,
    dense: cuda.dense_rows.DenseRows,
    rows: cuda_rows.Rows, // the router (and lm_head) at 5 or more rows, dense's bits
    act: cuda.act_rows.ActRows,
    sum: cuda.rank_scatter.RankScatter,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        const d = ctx.d;
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.lane = try qmmf.Lane.load(d, (try ctx.capability()) / 10);
        errdefer k.lane.unload();
        try k.lane.loadPrompt(d);
        k.fp8x = try experts.Fp8Experts.load(d, @intCast(try ctx.attribute(.multiprocessor_count)));
        errdefer k.fp8x.unload();
        k.plan_mod = try cuda.Module.load(d, cuda.kernels.experts);
        errdefer k.plan_mod.unload();
        k.router = try grouped.Router.resolve(k.plan_mod);
        k.norms = try cuda.norms.Norms.load(d);
        errdefer k.norms.unload();
        k.moe = try cuda.moe_glue.MoeGlue.load(d);
        errdefer k.moe.unload();
        k.route = try cuda_route.Route.load(d);
        errdefer k.route.unload();
        k.dense = try cuda.dense_rows.DenseRows.load(d);
        errdefer k.dense.unload();
        k.rows = try cuda_rows.Rows.load(d);
        errdefer k.rows.unload();
        k.act = try cuda.act_rows.ActRows.load(d);
        errdefer k.act.unload();
        k.sum = try cuda.rank_scatter.RankScatter.load(d);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.sum.unload();
        k.act.unload();
        k.rows.unload();
        k.dense.unload();
        k.route.unload();
        k.moe.unload();
        k.norms.unload();
        k.plan_mod.unload();
        k.fp8x.unload();
        k.lane.unload();
        k.* = undefined;
    }
};

/// Device scratch for calls of up to `rows` rows at any world 1, 2 or 4 (sized by world 1's widths).
pub const Scratch = struct {
    gpa: std.mem.Allocator,
    rows: usize,
    bufs: std.ArrayList(DeviceBuffer) = .empty,
    gu: u64 = 0, // bf16 [rows, 2 x FFN width]: gate then up
    act: u64 = 0, // bf16 [rows, FFN width] or [pairs, expert width]
    logits: u64 = 0, // fp32 [rows, experts]
    picks: u64 = 0, // int32 [rows, top_k]
    wts: u64 = 0, // fp32 [rows, top_k]: normalized, scaled
    plan: grouped.Plan = undefined,
    latent: u64 = 0, // bf16 [rows, latent]: routed_expert_down_proj's rows
    y: u64 = 0, // fp32 [pairs, latent]: each pick's expert output
    comb: u64 = 0, // bf16 [rows, latent]: the picks' weighted sum
    two: u64 = 0, // fp32 [2, rows, hidden]: the routed and shared partials
    part: u64 = 0, // the lane's slice partials

    fn alloc(x: *Scratch, d: *const cuda.Driver, len: usize) !u64 {
        var b = try DeviceBuffer.alloc(d, @max(len, 16));
        errdefer b.free();
        try x.bufs.append(x.gpa, b);
        return b.ptr;
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lane: *const qmmf.Lane, c: Config, rows: usize) !Scratch {
        var x: Scratch = .{ .gpa = gpa, .rows = rows };
        errdefer x.deinit();
        const pairs = rows * c.top_k;
        const widest = @max(c.dense_width, c.shared_width);
        const wide = pairs > grouped.small;
        const items = @max(grouped.maxItems(pairs, c.experts, 16), grouped.maxItems(pairs, c.experts, experts.promptTile(.fp8g)));
        x.gu = try x.alloc(d, rows * 2 * widest * 2);
        x.act = try x.alloc(d, @max(rows * widest, pairs * c.expert_width) * 2);
        x.logits = try x.alloc(d, rows * c.experts * 4);
        x.picks = try x.alloc(d, pairs * 4);
        x.wts = try x.alloc(d, pairs * 4);
        x.plan = .{
            .members = try x.alloc(d, pairs * 4),
            .items = try x.alloc(d, items * 12),
            .counts = try x.alloc(d, 8),
            .rank = try x.alloc(d, (if (wide) pairs else 1) * 4),
            .hist = try x.alloc(d, (if (wide) (pairs + grouped.small - 1) / grouped.small * c.experts else 1) * 4),
        };
        x.latent = try x.alloc(d, rows * c.latent * 2);
        x.y = try x.alloc(d, pairs * c.latent * 4);
        x.comb = try x.alloc(d, rows * c.latent * 2);
        x.two = try x.alloc(d, 2 * rows * c.hidden * 4);
        x.part = try x.alloc(d, try partBytes(lane, c, rows));
        return x;
    }

    pub fn deinit(x: *Scratch) void {
        for (x.bufs.items) |*b| b.free();
        x.bufs.deinit(x.gpa);
        x.* = undefined;
    }
};

/// The most slice partials any of the sublayer's lane calls needs, at every world and every row count up to `rows`.
fn partBytes(lane: *const qmmf.Lane, c: Config, rows: usize) !usize {
    var most: usize = 0;
    for ([_]usize{ 1, 2, 4 }) |world| {
        const dw = (try partition.ffn(c.dense_width, 0, world)).len;
        const sw = (try partition.ffn(c.shared_width, 0, world)).len;
        const shapes = [_][2]usize{ .{ 2 * dw, c.hidden }, .{ c.hidden, dw }, .{ 2 * sw, c.hidden }, .{ c.hidden, sw }, .{ c.latent, c.hidden }, .{ c.latent + 2 * sw, c.hidden }, .{ c.latent / 2 + 2 * sw, c.hidden }, .{ c.hidden, c.latent }, .{ c.experts, c.hidden } };
        for (shapes) |nk| {
            const w = cuda.modelopt.bf16Weight(0, nk[0], nk[1]);
            var m: usize = 1;
            while (m <= rows) : (m = if (m < 64) m + 1 else m * 2) most = @max(most, lane.partBytes(m, w));
            most = @max(most, lane.partBytes(rows, w));
        }
    }
    return most;
}

/// post_attention_layernorm on `x` bf16 [rows, hidden] as HYV4FlashRMSNorm rounds: bf16(w bf16(x inv)); replicated.
pub fn norm(k: *const Kernels, s: Stream, c: Config, m: weights.Mlp, x: u64, out: u64, rows: usize) !void {
    try k.norms.addRmsTail(s, x, null, m.norm, 0, out, null, rows, c.hidden, c.eps, .{ .round = true });
}

pub const ShardedComm = struct {
    ctx: *anyopaque,
    rank: u32 = 0,
    gather_bf16: ?*const fn (ctx: *anyopaque, s: Stream, send: u64, recv: u64, count_bf16: usize) anyerror!void = null,
    exchange_f32: ?*const fn (ctx: *anyopaque, s: Stream, send: u64, recv: u64, count_f32: usize) anyerror!void = null,
};

/// A rank's share of the MLP on normed rows `x` (fp32 [rows, hidden] partial sums); `prompt` selects 64-pair items.
pub fn partial(k: *const Kernels, s: Stream, c: Config, m: weights.Mlp, x_: *const Scratch, x: u64, out: u64, rows: usize, prompt: bool) !void {
    return partialProf(k, s, c, m, x_, x, out, rows, prompt, null, null);
}

pub fn partialProf(k: *const Kernels, s: Stream, c: Config, m: weights.Mlp, x_: *const Scratch, x: u64, out: u64, rows: usize, prompt: bool, prof: ?*cuda_prof.Prof, comm: ?ShardedComm) !void {
    if (rows > x_.rows) return error.TooManyRows;
    switch (m.body) {
        .dense => |f| try ffn(k, s, c, f, x_, x, out, rows, prof),
        .moe => |e| try moe(k, s, c, e, x_, x, out, rows, prompt, prof, comm),
    }
}

/// gate and up in one call, bf16(silu(gate) up), then down's fp32 sums over the rank's inputs.
fn ffn(k: *const Kernels, s: Stream, c: Config, f: weights.Ffn, x_: *const Scratch, x: u64, out: u64, rows: usize, prof: ?*cuda_prof.Prof) !void {
    try k.lane.matmul(s, x, c.hidden, rows, f.gate_up, x_.gu, x_.part);
    try k.act.swiglu(s, x_.gu, 2 * f.width, x_.gu + 2 * f.width, 2 * f.width, x_.act, null, rows, f.width);
    try k.lane.matmulF32(s, x_.act, f.width, rows, f.down, out, x_.part);
    if (prof) |p| p.mark(s, .l0_ffn);
}

/// Runs fp32 router, fused `[latent_down, shared.gate, shared.up]`, routed experts, `latent_up`, and shared down.
fn moe(k: *const Kernels, s: Stream, c: Config, e: weights.Moe, x_: *const Scratch, x: u64, out: u64, rows: usize, prompt: bool, prof: ?*cuda_prof.Prof, comm: ?ShardedComm) !void {
    const pairs = rows * c.top_k;
    // Tensor-core router GEMM across all widths (SK = 4 invariant for R = 1..4096: zero R=64 resumed==fresh seam).
    try k.lane.matmulF32(s, x, c.hidden, rows, e.router_q, x_.logits, x_.part);
    const tile: usize = if (prompt) experts.promptTile(.fp8g) else 16;
    if (rows == 1 and tile == 16 and cuda_route.serves(c) and k.route.m4_plan1 != null) {
        try k.route.topkPlan1(s, c, x_.logits, e.bias, x_.picks, x_.wts, x_.plan);
    } else {
        const route: cuda.moe_glue.Route = .{ .experts = c.experts, .topk = c.top_k, .shared = false, .normalize = c.norm_topk, .scale = c.routed_scale, .score = true };
        if (cuda_route.serves(c)) try k.route.topk(s, c, x_.logits, e.bias, x_.picks, x_.wts, rows) else try k.moe.topk(s, x_.logits, e.bias, route, x_.picks, x_.wts, rows);
        try k.router.route(s, x_.picks, pairs, c.experts, tile, x_.plan);
    }
    if (prof) |p| {
        p.mark(s, .moe_route);
        p.recordMoeItems(s, x_.plan.counts);
    }
    const sw = e.shared_width;
    const use_half_ds = rows <= 64 and e.down_shared_half != null and comm != null and comm.?.gather_bf16 != null;
    const ld_local = if (use_half_ds) c.latent / 2 else c.latent;
    const ds_stride = ld_local + 2 * sw;
    var up_x: u64 = x_.gu;
    var up_stride: usize = ds_stride;
    if (use_half_ds) {
        const cm = comm.?;
        try k.lane.matmul(s, x, c.hidden, rows, e.down_shared_half.?, x_.gu, x_.part);
        if (rows == 1) {
            try cm.gather_bf16.?(cm.ctx, s, x_.gu, x_.latent, ld_local);
        } else {
            try k.rows.sliceBf16(s, x_.gu, ds_stride, x_.comb, ld_local, rows);
            try cm.gather_bf16.?(cm.ctx, s, x_.comb, x_.act, rows * ld_local);
            try k.rows.interleave2Bf16(s, x_.act, x_.latent, ld_local, rows);
        }
        up_x = x_.latent;
        up_stride = c.latent;
    } else {
        try k.lane.matmul(s, x, c.hidden, rows, e.down_shared, x_.gu, x_.part);
    }
    if (prof) |p| p.mark(s, .moe_ds);
    const items = grouped.maxItems(pairs, c.experts, tile);
    const gx: experts.Grouped = .{ .fp8g = &k.fp8x };
    const up_in: experts.Rows = .{ .x = up_x, .stride = up_stride, .slots = c.top_k, .contig_act = prompt };
    const down_in: experts.Rows = .{ .x = x_.act, .stride = e.experts.width, .contig_act = prompt };
    if (prompt) {
        try gx.prompt(s, .up, up_in, e.experts, x_.plan, x_.act, items, pairs, tile, -1);
        if (prof) |p| p.mark(s, .moe_exp_up);
        try gx.prompt(s, .down_f32, down_in, e.experts, x_.plan, x_.y, items, pairs, tile, -1);
        if (prof) |p| p.mark(s, .moe_exp_dn);
    } else {
        try gx.decode(s, .up, up_in, e.experts, x_.plan, x_.act, items, pairs, tile, -1);
        if (prof) |p| p.mark(s, .moe_exp_up);
        try gx.decode(s, .down_f32, down_in, e.experts, x_.plan, x_.y, items, pairs, tile, -1);
        if (prof) |p| p.mark(s, .moe_exp_dn);
    }
    try k.moe.combine(s, x_.y, .f32, x_.wts, x_.comb, .bf16, rows, c.top_k, c.latent);
    if (prof) |p| p.mark(s, .moe_comb);
    // the routed rows back to hidden and the shared expert's share, each fp32; routed + shared is the partial
    const n = rows * c.hidden;
    if (k.lane.canAdd(e.shared_down)) {
        try k.lane.matmulF32(s, x_.comb, c.latent, rows, e.latent_up, out, x_.part);
        try k.act.swiglu(s, x_.gu + ld_local * 2, ds_stride, x_.gu + (ld_local + sw) * 2, ds_stride, x_.act, null, rows, sw);
        try k.lane.matmulF32Add(s, x_.act, sw, rows, e.shared_down, out, out);
    } else {
        try k.lane.matmulF32(s, x_.comb, c.latent, rows, e.latent_up, x_.two, x_.part);
        try k.act.swiglu(s, x_.gu + ld_local * 2, ds_stride, x_.gu + (ld_local + sw) * 2, ds_stride, x_.act, null, rows, sw);
        try k.lane.matmulF32(s, x_.act, sw, rows, e.shared_down, x_.two + n * 4, x_.part);
        try k.sum.rankSum(s, x_.two, x_.two, 2, 0, n, out);
    }
    if (prof) |p| p.mark(s, .moe_lup);
}

/// After the all-reduce: the fp32 sums [rows, hidden] through post_fc2_norm or combine_output_norm into bf16.
pub fn finish(k: *const Kernels, s: Stream, c: Config, m: weights.Mlp, sum: u64, out: u64, rows: usize) !void {
    try k.norms.rms(s, sum, .f32, m.post_norm, out, .bf16, rows, c.hidden, c.eps);
}

/// The ranks' partials [world, rows, hidden] fp32 added in rank order (rank 0 first), as the all-reduce sums them.
pub fn rankSum(k: *const Kernels, s: Stream, partials: u64, world: usize, rows: usize, hidden: usize, out: u64) !void {
    try k.sum.rankSum(s, partials, partials, world, 0, rows * hidden, out);
}

test "every launcher compiles on a host without kernels" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Kernels);
    std.testing.refAllDecls(Scratch);
    _ = cuda_rows;
}
