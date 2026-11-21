//! Hy4-flash family construction, `tf-hy4-engram-pack` CLI, and host checks without a GPU.
const std = @import("std");
const tp_build = @import("tp.zig");

pub fn moduleShared(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, tp_mod: *std.Build.Module, api: *std.Build.Module, tp_image: ?std.Build.LazyPath) *std.Build.Module {
    const engram_rows = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/engram_rows.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const tp_kern = tp_build.kernelModule(b, tp_image);
    return b.createModule(.{
        .root_source_file = b.path("zig/src/families/hy4/cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "cuda", .module = cuda },
            .{ .name = "core", .module = core },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "engram_rows", .module = engram_rows },
            .{ .name = "tp", .module = tp_mod },
            .{ .name = "tp_kernel", .module = tp_kern },
            .{ .name = "engine_api", .module = api },
        },
    });
}

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    const verbs = tp_build.verbsModule(b, target, optimize);
    const tp_mod = tp_build.moduleWith(b, target, optimize, cuda, verbs);
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    return moduleShared(b, target, optimize, cuda, core, lanes, tp_mod, api, null);
}

/// Builds and installs `tf-hy4-engram-pack` (`zig build hy4-engram-pack` or the default Linux install step).
pub fn installTools(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: ?bool, hy4: *std.Build.Module) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/tools/hy4_engram_pack.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{.{ .name = "hy4", .module = hy4 }} });
    const exe = b.addExecutable(.{ .name = "tf-hy4-engram-pack", .root_module = root });
    const inst = b.addInstallArtifact(exe, .{});
    b.getInstallStep().dependOn(&inst.step);
    b.step("hy4-engram-pack", "Build and install tf-hy4-engram-pack (offline fp64 Engram row packer)").dependOn(&inst.step);
}

/// `test-hy4-host`; TF_HY4_DIR (a checkpoint, whole or a per-layer part) adds the release's own tensors.
pub fn host(b: *std.Build, m: *std.Build.Module, server_tests: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    run.has_side_effects = true; // TF_HY4_DIR is read at run time
    const hy4_step = b.step("test-hy4-host", "Hy4-flash config, tensor map, Engram hash, FP8 packer and HYTK markup without a GPU");
    hy4_step.dependOn(&run.step);
    const srv_run = b.addRunArtifact(b.addTest(.{ .root_module = server_tests, .filters = &.{"hy4 markup:"} }));
    srv_run.has_side_effects = true;
    hy4_step.dependOn(&srv_run.step);
    all.dependOn(&run.step);
}
