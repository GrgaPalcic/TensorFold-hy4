//! Hy4-flash's MLA kernels (hy4_mla.cu) under the contract with their planted controls: tf-cuda-test hy4-mla-contract.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4").cuda_mla;
const options = @import("contract_options");
const check = @import("check.zig");
const attn_k = @import("contract_hy4_mla_attn.zig");
const proj_k = @import("contract_hy4_mla_proj.zig");

const g = core.contract.gates;

pub const eps: f32 = 1e-5; // Hy4-flash's rms_norm_eps

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

/// A planted fault (hy4_mla.cuh's tf_hy4::Fault) and the gate it must turn red.
pub const Fault = enum {
    none,
    bad_nan,
    bad_repeat,
    bad_tail,
    bad_sums,

    pub fn gate(f: Fault) ?g.Gate {
        return switch (f) {
            .none => null,
            .bad_nan => .nonfinite,
            .bad_repeat => .repeat,
            .bad_tail => .shapes,
            .bad_sums => .sums,
        };
    }
};

const faults = [_]Fault{ .bad_nan, .bad_repeat, .bad_tail, .bad_sums };
const kernels = [_][]const u8{ "absorb", "attn", "attn_rows", "value", "post" };

fn symbolOf(comptime k: []const u8, comptime f: Fault) [:0]const u8 {
    return "tf_ctl_hy4_mla_" ++ k ++ "_" ++ @tagName(f);
}

fn labelOf(comptime k: []const u8, comptime f: Fault) []const u8 {
    const what = switch (f) {
        .none => unreachable,
        .bad_nan => "reads a NaN input as 0",
        .bad_repeat => "bumps its first input every run",
        .bad_tail => "leaves its reduction's last term out",
        .bad_sums => "keeps its running sums in fp16",
    };
    return "control: hy4 tf_hy4_mla_" ++ k ++ " " ++ what;
}

const labels = blk: {
    var out: [kernels.len * faults.len][]const u8 = undefined;
    for (kernels, 0..) |k, i| for (faults, 0..) |f, j| {
        out[i * faults.len + j] = labelOf(k, f);
    };
    break :blk out;
};

/// The control's name for kernel `k` (absorb, attn, value, post) planted with `f`.
pub fn controlName(k: []const u8, f: Fault) []const u8 {
    for (kernels, 0..) |name, i| if (std.mem.eql(u8, name, k)) for (faults, 0..) |fault, j| {
        if (fault == f) return labels[i * faults.len + j];
    };
    unreachable;
}

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: hy4.Kernels,
    ctl: cuda.Module,
    attns: [5]attn_k.Attn = undefined,
    projs: [5]proj_k.Proj = undefined,
    attn_ctl: [2 * faults.len]attn_k.Attn = undefined,
    proj_ctl: [3 * faults.len]proj_k.Proj = undefined,

    /// The set, its kernels appended to `list`: the served forms, then each kernel's four controls.
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        if (!options.controls) return error.BuiltWithoutControls;
        const self = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(self);
        var k = try hy4.Kernels.load(gpu.d);
        errdefer k.unload();
        var ctl = try cuda.Module.load(gpu.d, &controls.bytes);
        errdefer ctl.unload();
        self.* = .{ .gpa = gpu.gpa, .k = k, .ctl = ctl };
        const rig: @import("contract_rig.zig").Rig = .{ .gpu = gpu, .s = s };
        for (&self.attns, std.enums.values(attn_k.Form)) |*x, form| x.* = .{ .rig = rig, .form = form, .f = if (form == .rows) self.k.attn_rows else self.k.attn };
        const kinds = [_]proj_k.Kind{ .absorb, .value, .post, .post, .post };
        const worlds = [_]usize{ 1, 1, 1, 2, 4 };
        for (&self.projs, kinds, worlds) |*x, kind, world| x.* = .{ .rig = rig, .kind = kind, .f = self.fn_(kind), .world = world };
        inline for (faults, 0..) |fault, j| {
            const attn_fn = try self.ctl.function(comptime symbolOf("attn", fault));
            try attn_fn.allowDynamicShared(hy4.attn_smem);
            self.attn_ctl[j] = .{ .rig = rig, .form = .window, .f = attn_fn, .fault = fault };
            const rows_fn = try self.ctl.function(comptime symbolOf("attn_rows", fault));
            try rows_fn.allowDynamicShared(hy4.attn_smem);
            self.attn_ctl[faults.len + j] = .{ .rig = rig, .form = .rows, .f = rows_fn, .fault = fault };
            inline for ([_]proj_k.Kind{ .absorb, .value, .post }, 0..) |kind, i| {
                const f = try self.ctl.function(comptime symbolOf(@tagName(kind), fault));
                self.proj_ctl[i * faults.len + j] = .{ .rig = rig, .kind = kind, .f = f, .fault = fault, .world = if (kind == .post) 2 else 1 };
            }
        }
        for (&self.attns) |*x| try list.append(a, x.kernel());
        for (&self.projs) |*x| try list.append(a, x.kernel());
        for (&self.attn_ctl) |*x| try list.append(a, x.kernel());
        for (&self.proj_ctl) |*x| try list.append(a, x.kernel());
        return self;
    }

    fn fn_(self: *const Set, kind: proj_k.Kind) cuda.Function {
        return switch (kind) {
            .absorb => self.k.absorb,
            .value => self.k.value,
            .post => self.k.post,
        };
    }

    pub fn deinit(self: *Set) void {
        for (&self.attns) |*x| x.deinit();
        for (&self.projs) |*x| x.deinit();
        for (&self.attn_ctl) |*x| x.deinit();
        for (&self.proj_ctl) |*x| x.deinit();
        self.ctl.unload();
        self.k.unload();
        self.gpa.destroy(self);
    }
};

/// Hy4-flash's MLA kernels' gates 1-4 and their controls alone, then their launches against this GPU (gate 5).
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const self = try Set.init(gpu, s, a, &list);
    defer self.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x6879344d, &t);
    const lim = try @import("contract_limits.zig").live(gpu, &.{"hy4"});
    std.debug.print("SUMMARY hy4 mla: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed; {d} limit problems\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed, lim });
    try check.expect(t.problems() + lim == 0, "hy4-mla-contract: {d} gate failures, {d} controls missed, {d} limit problems", .{ t.failed, t.missed, lim });
}
