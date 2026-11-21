//! Combined CUDA contract set for PR #22's Hy4-flash wave-1 kernels (`hy4_rows_argmax`, `hy4_route`, `hy4_ihc`).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4");
const options = @import("contract_options");
const check = @import("check.zig");
const argmax_k = @import("contract_hy4_argmax.zig");
const sample_rows_k = @import("contract_hy4_sample_rows.zig");
const fp8x_k64_k = @import("contract_hy4_fp8x_k64.zig");
const route_k = @import("contract_hy4_route.zig");
const ihc_k = @import("contract_hy4_ihc.zig");

const g = core.contract.gates;

const ctl_img = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

pub const Set = struct {
    gpa: std.mem.Allocator,
    rows: hy4.cuda_mlp.cuda_rows.Rows,
    route: hy4.cuda_route.Route,
    moe: cuda.moe_glue.MoeGlue,
    fp8x: cuda.experts.Fp8Experts,
    ihc: hy4.cuda_ihc.Ihc,
    hc: hy4.cuda_hc.Hc,
    norms: cuda.norms.Norms,
    mla_mod: cuda.Module,
    ctl_mod: ?cuda.Module = null,
    argmax_cases: [3]argmax_k.Case = undefined,
    argmax_n: usize = 0,
    sample_cases: [4]sample_rows_k.Case = undefined,
    sample_n: usize = 0,
    fp8x_cases: [3]fp8x_k64_k.Case = undefined,
    fp8x_n: usize = 0,
    route_cases: [4]route_k.Case = undefined,
    route_n: usize = 0,
    ihc_cases: [7]ihc_k.Case = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var rows = try hy4.cuda_mlp.cuda_rows.Rows.load(gpu.d);
        errdefer rows.unload();
        var route = try hy4.cuda_route.Route.load(gpu.d);
        errdefer route.unload();
        var moe = try cuda.moe_glue.MoeGlue.load(gpu.d);
        errdefer moe.unload();
        var fp8x = try cuda.experts.Fp8Experts.load(gpu.d, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
        errdefer fp8x.unload();
        var ihc = try hy4.cuda_ihc.Ihc.load(gpu.d);
        errdefer ihc.unload();
        var hc = try hy4.cuda_hc.Hc.load(gpu.d);
        errdefer hc.unload();
        var norms = try cuda.norms.Norms.load(gpu.d);
        errdefer norms.unload();
        var mla_mod = try cuda.Module.load(gpu.d, cuda.kernels.hy4_mla);
        errdefer mla_mod.unload();
        const post_fn = try mla_mod.function("tf_hy4_mla_post");
        const ctl_mod = if (options.controls) try cuda.Module.load(gpu.d, &ctl_img.bytes) else null;

        set.* = .{
            .gpa = gpu.gpa,
            .rows = rows,
            .route = route,
            .moe = moe,
            .fp8x = fp8x,
            .ihc = ihc,
            .hc = hc,
            .norms = norms,
            .mla_mod = mla_mod,
            .ctl_mod = ctl_mod,
        };

        set.argmax_cases[0] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = set.rows.argmax_fn, .fault = .none, .name = "cuda hy4 hy4_rows_argmax" };
        set.argmax_n = 1;
        if (set.ctl_mod) |m| {
            set.argmax_cases[1] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try m.function("tf_ctl_hy4_argmax_bad_nan"), .fault = .bad_nan, .name = "control: hy4 hy4_rows_argmax drops NaN" };
            set.argmax_cases[2] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try m.function("tf_ctl_hy4_argmax_bad_tie"), .fault = .bad_tie, .name = "control: hy4 hy4_rows_argmax breaks ties high" };
            set.argmax_n = 3;
        }
        for (set.argmax_cases[0..set.argmax_n]) |*x| try list.append(a, x.kernel());

        set.sample_cases[0] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = set.rows.f32_to_bf16_fn, .mode = .f32_to_bf16, .fault = .none, .name = "cuda hy4 hy4_rows_f32_to_bf16" };
        set.sample_cases[1] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = set.rows.interleave_bf16_fn, .mode = .interleave_bf16, .fault = .none, .name = "cuda hy4 hy4_rows_interleave_bf16" };
        set.sample_n = 2;
        if (set.ctl_mod) |m| {
            set.sample_cases[2] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try m.function("tf_ctl_hy4_f32_to_bf16_bad_nan"), .mode = .f32_to_bf16, .fault = .bad_nan, .name = "control: hy4 hy4_rows_f32_to_bf16 drops NaN" };
            set.sample_cases[3] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try m.function("tf_ctl_hy4_interleave_bad_nan"), .mode = .interleave_bf16, .fault = .bad_nan, .name = "control: hy4 hy4_rows_interleave_bf16 drops NaN" };
            set.sample_n = 4;
        }
        for (set.sample_cases[0..set.sample_n]) |*x| try list.append(a, x.kernel());

        set.fp8x_cases[0] = .{ .rig = .{ .gpu = gpu, .s = s }, .fp8x = &set.fp8x, .mode = .down_f32, .fault = .none, .name = "cuda hy4 fp8_expert_prompt_k64_f32" };
        set.fp8x_cases[1] = .{ .rig = .{ .gpu = gpu, .s = s }, .fp8x = &set.fp8x, .mode = .down_bf16, .fault = .none, .name = "cuda hy4 fp8_expert_prompt_k64_bf16" };
        set.fp8x_n = 2;
        if (set.ctl_mod) |m| {
            set.fp8x_cases[2] = .{ .rig = .{ .gpu = gpu, .s = s }, .fp8x = &set.fp8x, .ctl_dn = try m.function("tf_ctl_hy4_fp8x_k64_bad_scale"), .mode = .down_f32, .fault = .bad_scale, .name = "control: hy4 fp8_expert_prompt_k64 drops scale" };
            set.fp8x_n = 3;
        }
        for (set.fp8x_cases[0..set.fp8x_n]) |*x| try list.append(a, x.kernel());

        set.route_cases[0] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = set.route.m4, .moe = &set.moe, .fault = .none, .name = "cuda hy4 tf_hy4_route_m4" };
        set.route_n = 1;
        if (set.route.m4_plan1) |p1| {
            set.route_cases[set.route_n] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = p1, .moe = &set.moe, .fault = .none, .plan1 = true, .name = "cuda hy4 tf_hy4_route_m4_plan1" };
            set.route_n += 1;
        }
        if (set.ctl_mod) |m| {
            set.route_cases[set.route_n] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try m.function("tf_ctl_hy4_route_high_ties"), .moe = &set.moe, .fault = .high_ties, .name = "control: hy4 tf_hy4_route_m4 high ties" };
            set.route_n += 1;
            set.route_cases[set.route_n] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try m.function("tf_ctl_hy4_route_distinct"), .moe = &set.moe, .fault = .distinct, .name = "control: hy4 tf_hy4_route_m4 distinct past -inf" };
            set.route_n += 1;
        }
        for (set.route_cases[0..set.route_n]) |*x| try list.append(a, x.kernel());

        const modes = [5]struct { mode: hy4.cuda_ihc.Mode, name: []const u8 }{
            .{ .mode = .boundary, .name = "cuda hy4 hy4_ihc boundary" },
            .{ .mode = .entry, .name = "cuda hy4 hy4_ihc entry" },
            .{ .mode = .head, .name = "cuda hy4 hy4_ihc epilogue_head" },
            .{ .mode = .post_only, .name = "cuda hy4 hy4_ihc post_only" },
            .{ .mode = .head_only, .name = "cuda hy4 hy4_ihc head_only" },
        };
        for (modes, 0..) |m, i| {
            set.ihc_cases[i] = .{ .rig = .{ .gpu = gpu, .s = s }, .ihc = &set.ihc, .hc = &set.hc, .post_fn = post_fn, .norms = &set.norms, .mode = m.mode, .name = m.name, .arena = .init(gpu.gpa) };
        }
        set.ihc_cases[5] = .{ .rig = .{ .gpu = gpu, .s = s }, .ihc = &set.ihc, .hc = &set.hc, .post_fn = post_fn, .norms = &set.norms, .mode = .entry, .fault = .bad_nan, .name = "control: hy4 hy4_ihc entry drops NaN", .arena = .init(gpu.gpa) };
        set.ihc_cases[6] = .{ .rig = .{ .gpu = gpu, .s = s }, .ihc = &set.ihc, .hc = &set.hc, .post_fn = post_fn, .norms = &set.norms, .mode = .entry, .fault = .bad_sums, .name = "control: hy4 hy4_ihc entry sums in fp16", .arena = .init(gpu.gpa) };
        for (&set.ihc_cases) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.ihc_cases) |*x| x.deinit();
        for (set.route_cases[0..set.route_n]) |*x| x.deinit();
        for (set.fp8x_cases[0..set.fp8x_n]) |*x| x.deinit();
        for (set.sample_cases[0..set.sample_n]) |*x| x.deinit();
        for (set.argmax_cases[0..set.argmax_n]) |*x| x.deinit();
        if (set.ctl_mod) |*m| m.unload();
        set.mla_mod.unload();
        set.norms.unload();
        set.hc.unload();
        set.ihc.unload();
        set.fp8x.unload();
        set.moe.unload();
        set.route.unload();
        set.rows.unload();
        set.gpa.destroy(set);
    }
};

pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x68793477, &t);
    std.debug.print("SUMMARY hy4-wave1: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "hy4-wave1-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
    const lim = try @import("contract_limits.zig").live(gpu, &.{ "hy4_rows", "hy4_route", "hy4_ihc", "hy4_dfly2_glue", "hy4_mla_attn", "hy4_mla_c2", "hy4_index", "fp8_experts", "qmmf" });
    try check.expect(lim == 0, "hy4-wave1-contract: {d} live Gate 5 limit mismatches", .{lim});
}
