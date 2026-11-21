//! Hy4-flash's LatentMoE routing (hy4_route.cu): bit-equal to moe_glue's tf_moe_topk_sigmoid_score_warp, faster.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");

const Config = config.Config;
const Stream = cuda.Stream;

/// The one form tf_hy4_route_m4 serves: 512 experts, top 16, normalized, no shared slot; others take moe_glue's.
pub fn serves(c: Config) bool {
    return c.experts == 512 and c.top_k == 16 and c.norm_topk;
}

pub const Route = struct {
    mod: cuda.Module,
    m4: cuda.Function,
    m4_plan1: ?cuda.Function = null,

    pub fn load(d: *const cuda.Driver) !Route {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.hy4_route);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .m4 = try mod.function("tf_hy4_route_m4"),
            .m4_plan1 = mod.function("tf_hy4_route_m4_plan1") catch null,
        };
    }

    pub fn unload(r: *Route) void {
        r.mod.unload();
    }

    /// fp32 logits [rows, 512] + fp32 bias [512] into int32 picks and fp32 weights [rows, 16], in pick order.
    pub fn topk(r: *const Route, s: Stream, c: Config, logits: u64, bias: u64, pick: u64, wts: u64, rows: usize) !void {
        if (!serves(c)) return error.RouteOutOfRange;
        try launch(r.m4, s, logits, bias, rows, c.routed_scale, pick, wts);
    }

    /// Fused single-row top-16 + grouped MoE plan (`rows == 1`, `tile == 16`), bit-equal to `topk` + `plan_small`.
    pub fn topkPlan1(r: *const Route, s: Stream, c: Config, logits: u64, bias: u64, pick: u64, wts: u64, plan: cuda.grouped.Plan) !void {
        if (!serves(c)) return error.RouteOutOfRange;
        const f = r.m4_plan1 orelse return error.MissingKernel;
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(bias);
        a.add(c.routed_scale);
        a.add(pick);
        a.add(wts);
        a.add(plan.members);
        a.add(plan.items);
        a.add(plan.counts);
        try cuda.launch.launch(f, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 128 } }, s, &a);
    }
};

/// One block of 128 threads a row; `f` is tf_hy4_route_m4 or one of its planted controls.
pub fn launch(f: cuda.Function, s: Stream, logits: u64, bias: u64, rows: usize, scale: f32, pick: u64, wts: u64) !void {
    if (rows == 0) return;
    if (rows > std.math.maxInt(i32)) return error.RouteOutOfRange;
    var a: cuda.Args = .{};
    a.add(logits);
    a.add(bias);
    a.add(@as(i32, @intCast(rows)));
    a.add(scale);
    a.add(pick);
    a.add(wts);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 128 } }, s, &a);
}

test "hy4 route: only 512 experts, top 16, normalized; any other form keeps moe_glue's kernel" {
    var c: Config = undefined;
    c.experts = 512;
    c.top_k = 16;
    c.norm_topk = true;
    try std.testing.expect(serves(c));
    for ([_][3]usize{ .{ 496, 16, 1 }, .{ 512, 8, 1 }, .{ 512, 16, 0 } }) |x| {
        c.experts = x[0];
        c.top_k = x[1];
        c.norm_topk = x[2] == 1;
        try std.testing.expect(!serves(c));
    }
}
