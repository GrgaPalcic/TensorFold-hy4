//! Hy4-flash's kernels on the release's tensors against tools/zig/hy4_reference.py's fixtures, one or more files a run.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4");
const check = @import("check.zig");
const Gpu = check.Gpu;

const st = core.safetensors;
const Hc = hy4.cuda_hc.Hc;

fn tap(f: *const st.File, name: []const u8) !st.Tensor {
    return f.get(name) orelse {
        std.debug.print("the fixture has no {s}\n", .{name});
        return error.MissingTap;
    };
}

fn bf16At(bytes: []const u8, i: usize) f64 {
    return @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)) << 16));
}

fn f32At(bytes: []const u8, i: usize) f64 {
    return @as(f32, @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little)));
}

/// A bf16 ulp at |v|: 2^(e - 8) for |v| in [2^(e - 1), 2^e), the subnormal step below the normals.
fn ulp(v: f64) f64 {
    const a = @abs(v);
    if (a < 0x1p-126) return 0x1p-133;
    return std.math.ldexp(@as(f64, 1), std.math.frexp(a).exponent - 8);
}

/// Distance of bf16 outputs from float64: worst fraction of the Wilkinson bound, and outputs past 0.5 bf16 ulp.
const OutGap = struct {
    bound_frac: f64 = 0,
    past_half: usize = 0,

    fn of(got: []const u8, want: []const f64, allow: []const f64) OutGap {
        var g: OutGap = .{};
        for (want, allow, 0..) |w, a, i| {
            const d = @abs(bf16At(got, i) - w);
            g.bound_frac = @max(g.bound_frac, d / @max(a, 0x1p-133));
            if (d > 0.5 * ulp(w)) g.past_half += 1;
        }
        return g;
    }
};

fn bitsDiffer(a: []const u8, b: []const u8) usize {
    var n: usize = 0;
    for (0..a.len / 2) |i| n += @intFromBool(!std.mem.eql(u8, a[2 * i ..][0..2], b[2 * i ..][0..2]));
    return n;
}

fn postGap(got: []const u8, want: []const f64, allow: []const f64) struct { rel: f64, bound_frac: f64 } {
    var rel: f64 = 0;
    var bound_frac: f64 = 0;
    for (want, allow, 0..) |r, a, i| {
        const d = @abs(f32At(got, i) - r);
        rel = @max(rel, d / @max(@abs(r), 0x1p-30));
        bound_frac = @max(bound_frac, d / @max(a, 0x1p-120));
    }
    return .{ .rel = rel, .bound_frac = bound_frac };
}

/// Float64 reference for `hy4_hc_pre` on `x` bf16 [rows, 4, d], with Wilkinson bounds `out_allow` and `post_allow`.
fn hcRef64(x: []const u8, fn_b: []const u8, base_b: []const u8, scale_b: []const u8, eps: f32, hc_eps: f32, mag_c: f32, rows: usize, d: usize, out64: []f64, out_allow: []f64, post64: []f64, post_allow: []f64) void {
    const width = 4 * d;
    const gamma61 = 61.0 * 0x1p-24;
    const pre_s = f32At(scale_b, 0);
    const post_s = f32At(scale_b, 1);
    for (0..rows) |row| {
        var sq: f64 = 0;
        var dot: [8]f64 = @splat(0);
        var mdot: [8]f64 = @splat(0);
        for (0..width) |k| {
            const v = bf16At(x, row * width + k);
            sq += v * v;
            for (0..8) |j| {
                const p = v * f32At(fn_b, j * width + k);
                dot[j] += p;
                mdot[j] += @abs(p);
            }
        }
        const inv = 1.0 / @sqrt(sq / @as(f64, @floatFromInt(width)) + @as(f64, eps));
        var pre: [4]f64 = undefined;
        var pre_err: [4]f64 = undefined;
        for (0..4) |i| {
            const z0 = dot[i] * inv * pre_s + f32At(base_b, i);
            pre[i] = 1.0 / (1.0 + @exp(-z0)) + @as(f64, hc_eps);
            pre_err[i] = 0.25 * (gamma61 * mdot[i] * inv * @abs(pre_s) + 8.0 * 0x1p-24 * @abs(z0)) + 4.0 * 0x1p-24 * @abs(pre[i]);
            const z = dot[4 + i] * inv * post_s + f32At(base_b, 4 + i);
            const p = @as(f64, mag_c) / (1.0 + @exp(-z)) + @as(f64, hc_eps);
            post64[row * 4 + i] = p;
            post_allow[row * 4 + i] = 0.25 * @abs(@as(f64, mag_c)) * (gamma61 * mdot[4 + i] * inv * @abs(post_s) + 8.0 * 0x1p-24 * @abs(z)) + 8.0 * 0x1p-24 * @abs(p);
        }
        for (0..d) |h| {
            var s: f64 = 0;
            var work: f64 = 0;
            for (0..4) |i| {
                const xv = bf16At(x, row * width + i * d + h);
                s += pre[i] * xv;
                work += pre_err[i] * @abs(xv) + 4.0 * 0x1p-24 * @abs(pre[i] * xv);
            }
            out64[row * d + h] = s;
            out_allow[row * d + h] = work + 0.5 * ulp(s + work) + 0x1p-133;
        }
    }
}

/// The layer number of a fixture named layer<i>_t<T>.safetensors.
fn layerOf(path: []const u8) !usize {
    const base = std.fs.path.basename(path);
    if (!std.mem.startsWith(u8, base, "layer")) return error.NotALayerFixture;
    const end = std.mem.indexOfScalar(u8, base, '_') orelse return error.NotALayerFixture;
    return std.fmt.parseInt(usize, base[5..end], 10);
}

const Rig = struct {
    gpu: Gpu,
    s: cuda.Stream,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn put(r: *Rig, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, bytes);
        try r.bufs.append(r.gpu.gpa, b);
        return b.ptr;
    }

    fn zeros(r: *Rig, len: usize) !cuda.DeviceBuffer {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, len);
        try r.bufs.append(r.gpu.gpa, b);
        return b;
    }

    fn get(r: *Rig, b: cuda.DeviceBuffer) ![]u8 {
        try r.s.synchronize();
        const out = try r.gpu.gpa.alloc(u8, b.len);
        try b.download(0, out);
        return out;
    }

    fn deinit(r: *Rig) void {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(r.gpu.gpa);
        r.s.deinit();
    }
};

var failures: usize = 0;

fn verdict(ok: bool, comptime fmt: []const u8, args: anytype) void {
    if (!ok) failures += 1;
    std.debug.print("{s} " ++ fmt ++ "\n", .{if (ok) "PASS" else "FAIL"} ++ args);
}

pub fn run(gpu: Gpu, cmd: []const u8, rest: []const [:0]const u8) !void {
    if (std.mem.eql(u8, cmd, "hy4-contract")) return runContract(gpu);
    if (std.mem.eql(u8, cmd, "hy4-wave1-contract")) return @import("contract_hy4_wave1.zig").run(gpu);
    if (std.mem.startsWith(u8, cmd, "hy4-mla")) return @import("hy4_mla_tests.zig").run(gpu, cmd, rest);
    if (std.mem.eql(u8, cmd, "hy4-engram-contract")) return runEngramContract(gpu);
    if (std.mem.startsWith(u8, cmd, "hy4-index")) return @import("hy4_index_tests.zig").run(gpu, cmd, rest);
    if (std.mem.eql(u8, cmd, "hy4-head") or std.mem.eql(u8, cmd, "hy4-layers")) return @import("hy4_layer_tests.zig").run(gpu, cmd, rest);
    if (std.mem.eql(u8, cmd, "hy4-invariance")) return @import("hy4_invariance_tests.zig").run(gpu, rest);
    if (std.mem.eql(u8, cmd, "hy4-vit") or std.mem.eql(u8, cmd, "hy4-tail-kv8")) return @import("hy4_vit_kv8_tests.zig").run(gpu, cmd, rest);
    if (rest.len < 2) return error.MissingArgument;
    if (std.mem.eql(u8, cmd, "hy4-hc")) return hc(gpu, rest[0], rest[1..]);
    if (std.mem.eql(u8, cmd, "hy4-mlp")) return @import("hy4_mlp_tests.zig").mlp(gpu, rest[0], rest[1..]);
    if (std.mem.eql(u8, cmd, "hy4-engram")) return @import("hy4_engram_tests.zig").engram(gpu, rest[0], rest[1..]);
    return error.UnknownCommand;
}

fn runEngramContract(gpu: Gpu) !void {
    const g = core.contract.gates;
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try @import("contract_hy4_engram.zig").Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x68793465, &t);
    std.debug.print("SUMMARY hy4-engram: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "hy4-engram-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}

fn runContract(gpu: Gpu) !void {
    const g = core.contract.gates;
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try @import("contract_hy4.zig").Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x687934, &t);
    std.debug.print("SUMMARY hy4: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "hy4-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}

/// `hy4-hc MODEL FIXTURE...`: each layer's two iHC mixes (vs fp64) and two merges (bit-equal to torch's bf16).
pub fn hc(gpu: Gpu, model: []const u8, fixtures: []const [:0]const u8) !void {
    failures = 0;
    const gpa = gpu.gpa;
    const c = try hy4.config.Config.read(gpa, gpu.io, model);
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, gpu.io, model);
    defer ck.close();
    var k = try Hc.load(gpu.d);
    defer k.unload();
    for (fixtures) |fixture| try hcOne(gpu, c, &ck, &k, fixture);
    if (failures > 0) return error.Mismatch;
}

fn hcOne(gpu: Gpu, c: hy4.config.Config, ck: *core.checkpoint.Checkpoint, k: *const Hc, fixture: []const u8) !void {
    const gpa = gpu.gpa;
    const layer = try layerOf(fixture);
    var f = try st.File.open(gpa, gpu.io, fixture);
    defer f.close(gpu.io);
    const l = try hy4.names.layer(ck, c, layer);
    var r: Rig = .{ .gpu = gpu, .s = try cuda.Stream.init(gpu.d, false) };
    defer r.deinit();

    const rows = (try tap(&f, "in")).dim(1);
    const d = c.hidden;
    const n = rows * d;
    var mixes: [2]Hc.Mix = undefined;
    for ([_]hy4.names.Hc{ l.hc_attn, l.hc_mlp }, &mixes) |w, *m|
        m.* = .{ .fn_ = try r.put(w.fn_.bytes), .base = try r.put(w.base.bytes), .scale = try r.put(w.scale.bytes), .eps = c.eps, .hc_eps = c.hc_eps, .magnitude = c.hc_magnitude };
    const out = try r.zeros(n * 2);
    const post = try r.zeros(rows * 4 * 4);
    const streams = try r.zeros(n * 4 * 2);
    const out64 = try gpa.alloc(f64, n);
    defer gpa.free(out64);
    const out_allow = try gpa.alloc(f64, n);
    defer gpa.free(out_allow);
    const post64 = try gpa.alloc(f64, rows * 4);
    defer gpa.free(post64);
    const post_allow = try gpa.alloc(f64, rows * 4);
    defer gpa.free(post_allow);
    std.debug.print("hy4-hc layer {d}, {d} rows\n", .{ layer, rows });

    for ([_][]const u8{ "attn_hc", "ffn_hc" }, [_][]const u8{ "attn_hc_in16", "mid16" }, [_]hy4.names.Hc{ l.hc_attn, l.hc_mlp }, mixes) |name, input, w, m| {
        var key: [32]u8 = undefined;
        const in_bytes = (try tap(&f, input)).bytes;
        hcRef64(in_bytes, w.fn_.bytes, w.base.bytes, w.scale.bytes, c.eps, c.hc_eps, c.hc_magnitude, rows, d, out64, out_allow, post64, post_allow);
        try k.pre(r.s, m, try r.put(in_bytes), out.ptr, post.ptr, rows, d, .production, 0);
        const got = try r.get(out);
        defer gpa.free(got);
        const got_post = try r.get(post);
        defer gpa.free(got_post);
        const torch = (try tap(&f, try std.fmt.bufPrint(&key, "{s}_out16", .{name}))).bytes;
        const torch_post = (try tap(&f, try std.fmt.bufPrint(&key, "{s}_post16", .{name}))).bytes;
        const mine = OutGap.of(got, out64, out_allow);
        const theirs = OutGap.of(torch, out64, out_allow);
        const mp = postGap(got_post, post64, post_allow);
        const tp = postGap(torch_post, post64, post_allow);
        const differ = bitsDiffer(got, torch);
        verdict(mine.bound_frac <= 1 and mp.bound_frac <= 1, "{s} mix vs fp64: out {d:.3}x bound ({d} >0.5 ulp; torch {d:.3}x, {d}), {d}/{d} bits != torch; post rel {e:.3} ({d:.3}x bound; torch {e:.3})", .{ name, mine.bound_frac, mine.past_half, theirs.bound_frac, theirs.past_half, differ, n, mp.rel, mp.bound_frac, tp.rel });
        var again: [4][]u8 = undefined;
        for (&again) |*a| {
            try k.pre(r.s, m, try r.put(in_bytes), out.ptr, post.ptr, rows, d, .production, 0);
            a.* = try r.get(out);
        }
        defer for (again) |a| gpa.free(a);
        var same = true;
        for (again) |a| same = same and std.mem.eql(u8, a, got);
        verdict(same, "{s} mix: five runs byte-identical", .{name});
    }

    const merges = [_][4][]const u8{ .{ "attn_hc_in16", "attn_out16", "attn_hc_post16", "mid16" }, .{ "mid16", "mlp_out16", "ffn_hc_post16", "out16" } };
    for (merges) |mg| {
        const p = try r.put((try tap(&f, mg[2])).bytes);
        try k.merge(r.s, try r.put((try tap(&f, mg[0])).bytes), try r.put((try tap(&f, mg[1])).bytes), p, streams.ptr, rows, d, .production, 0);
        const got = try r.get(streams);
        defer gpa.free(got);
        const differ = bitsDiffer(got, (try tap(&f, mg[3])).bytes);
        verdict(differ == 0, "merge into {s}: {d} of {d} outputs differ from torch's bits", .{ mg[3], differ, n * 4 });
    }
}
