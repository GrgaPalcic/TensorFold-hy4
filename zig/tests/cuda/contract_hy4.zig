//! Hy4-flash CUDA contract set: `hy4_hc_pre` (both outputs), `hy4_hc_head`, `hy4_hc_merge`, and planted controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows_ = core.contract.rows;
const Hc = hy4.cuda_hc.Hc;
const Control = hy4.cuda_hc.Control;

const mix_served = [_]c.Shape{ .{ .rows = 8, .n = 1, .k = 4 * 3072 }, .{ .rows = 1, .n = 1, .k = 4 * 3072 }, .{ .rows = 3, .n = 1, .k = 4 * 256 } };
const merge_served = [_]c.Shape{ .{ .rows = 8, .n = 3072, .k = 1 }, .{ .rows = 1, .n = 3072, .k = 1 }, .{ .rows = 3, .n = 257, .k = 1 } };

fn mixDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 5 + 8;
}

fn zeroDepth(_: c.Shape) usize {
    return 0;
}

pub const MixKind = enum { pre_out, pre_post, head_out };

fn controlGate(ctl: Control) ?g.Gate {
    return switch (ctl) {
        .production => null,
        .nan => .nonfinite,
        .repeat => .repeat,
        .tail => .shapes,
        .sums => .sums,
    };
}

pub const MixCase = struct {
    rig: rig_.Rig,
    hc: *const Hc,
    kind: MixKind,
    ctl: Control,
    name: []const u8,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    fn_w: []f32 = &.{},
    base: [8]f32 = @splat(0),
    scale: [2]f32 = .{ 1, 1 },
    eps: f32 = 1e-6,
    hc_eps: f32 = 1e-4,
    magnitude: f32 = 2.0,
    dev_m: Hc.Mix = undefined,
    dev_x: u64 = 0,
    dev_out: u64 = 0,
    dev_post: u64 = 0,
    runs: i32 = 0,

    pub fn kernel(self: *MixCase) g.Kernel {
        const is_post = self.kind == .pre_post;
        return .{
            .spec = .{
                .name = self.name,
                .accum = .f32,
                .depth = mixDepth,
                .ops = 16,
                .out = if (is_post) .f32 else .bf16,
                .tiles = .{ .{}, .{}, .{ .size = 256, .step = 4, .min = 4 } },
                .served = &mix_served,
            },
            .ctx = self,
            .vt = &vt,
            .control = controlGate(self.ctl),
        };
    }
    const vt = g.hooks(MixCase);

    pub fn deinit(self: *MixCase) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    fn mixesCount(self: *const MixCase) usize {
        return if (self.kind == .head_out) 4 else 8;
    }

    pub fn count(self: *const MixCase) usize {
        return self.shape.rows * (if (self.kind == .pre_post) 4 else self.shape.k / 4);
    }

    pub fn prepare(self: *MixCase, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k < 4 or shape.k % 4 != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.runs = 0;
        const a = self.arena.allocator();
        const width = shape.k;
        const hidden = width / 4;
        const mcount = self.mixesCount();
        self.x = try a.alloc(u16, shape.rows * width);
        self.fn_w = try a.alloc(f32, mcount * width);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            self.scale = .{ 100.0, 100.0 };
            self.base = @splat(0);
            self.eps = 1e-6;
            self.hc_eps = 0;
            self.magnitude = 2.0;
            for (0..shape.rows) |r| for (0..hidden) |h| {
                self.x[r * width + h] = c.toBf16(8.0);
                self.x[r * width + hidden + h] = c.toBf16(-8.0);
                self.x[r * width + 2 * hidden + h] = 0;
                self.x[r * width + 3 * hidden + h] = 0;
            };
            for (0..mcount) |j| {
                const sgn: f32 = if (j % 4 == 0) 1.0 else -1.0;
                for (0..width) |k| self.fn_w[j * width + k] = if (k < 2 * hidden) sgn * 8.0 else 0.0;
                self.fn_w[j * width + hidden] = sgn * 7.875;
            }
        } else {
            self.scale = .{ 0.8, 1.1 };
            for (&self.base, 0..) |*b, i| b.* = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 3)) * 0.15;
            self.eps = 1e-6;
            self.hc_eps = 1e-4;
            self.magnitude = 2.0;
            for (self.x) |*v| v.* = c.toBf16(rng.float(f32) * 4.0 - 2.0);
            for (self.fn_w) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * 0.25;
            if (fill == .nonfinite) {
                if (shape.rows > 0) self.x[0] = c.toBf16(std.math.nan(f32));
                if (shape.rows > 1) self.x[width] = c.toBf16(std.math.inf(f32));
                if (shape.rows > 2) self.x[2 * width] = c.toBf16(-std.math.inf(f32));
            }
        }
        self.dev_x = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.dev_m = .{
            .fn_ = try self.rig.input(std.mem.sliceAsBytes(self.fn_w)),
            .base = try self.rig.input(std.mem.sliceAsBytes(self.base[0..mcount])),
            .scale = try self.rig.input(std.mem.sliceAsBytes(self.scale[0..if (self.kind == .head_out) 1 else 2])),
            .eps = self.eps,
            .hc_eps = self.hc_eps,
            .magnitude = self.magnitude,
        };
        if (self.kind == .pre_post) {
            self.dev_out = try self.rig.scratch(shape.rows * hidden * 2);
            self.dev_post = try self.rig.output(shape.rows * 4 * 4);
        } else {
            self.dev_out = try self.rig.output(shape.rows * hidden * 2);
            self.dev_post = if (self.kind == .pre_out) try self.rig.scratch(shape.rows * 4 * 4) else 0;
        }
        return true;
    }

    pub fn run(self: *MixCase) !void {
        const hidden = self.shape.k / 4;
        if (self.kind == .head_out) {
            try self.hc.head(self.rig.s, self.dev_m, self.dev_x, self.dev_out, self.shape.rows, hidden, self.ctl, self.runs);
        } else {
            try self.hc.pre(self.rig.s, self.dev_m, self.dev_x, self.dev_out, self.dev_post, self.shape.rows, hidden, self.ctl, self.runs);
        }
        self.runs += 1;
        try self.rig.s.synchronize();
    }

    fn evalRow(self: *const MixCase, r: usize, sq: f64, msq: f64, dot: []const f64, mdot: []const f64, want: []f64, mag: []f64) void {
        const width = self.shape.k;
        const hidden = width / 4;
        const inv = 1.0 / @sqrt(sq / @as(f64, @floatFromInt(width)) + @as(f64, self.eps));
        const sq_rel = if (sq > 0) 0.5 * msq / sq else 0;
        if (self.kind == .pre_post) {
            for (0..4) |i| {
                const z = (dot[4 + i] * inv) * @as(f64, self.scale[1]) + @as(f64, self.base[4 + i]);
                const e = @exp(-@abs(z));
                const sig = if (z < 0) e / (1.0 + e) else 1.0 / (1.0 + e);
                want[r * 4 + i] = @as(f64, self.magnitude) * sig + @as(f64, self.hc_eps);
                mag[r * 4 + i] = 0.25 * @abs(@as(f64, self.magnitude)) * (mdot[4 + i] * inv + @abs(dot[4 + i] * inv) * sq_rel) * @abs(@as(f64, self.scale[1]));
            }
            return;
        }
        var pre: [4]f64 = undefined;
        var wmag: f64 = 0;
        for (0..4) |i| {
            const z = (dot[i] * inv) * @as(f64, self.scale[0]) + @as(f64, self.base[i]);
            const e = @exp(-@abs(z));
            const sig = if (z < 0) e / (1.0 + e) else 1.0 / (1.0 + e);
            pre[i] = sig + @as(f64, self.hc_eps);
            wmag += 0.25 * (mdot[i] * inv + @abs(dot[i] * inv) * sq_rel) * @abs(@as(f64, self.scale[0]));
        }
        for (0..hidden) |h| {
            var s: f64 = 0;
            var xmax: f64 = 0;
            var smag: f64 = 0;
            for (0..4) |i| {
                const xv: f64 = c.fromBf16(self.x[r * width + i * hidden + h]);
                s += pre[i] * xv;
                xmax = @max(xmax, @abs(xv));
                smag += @abs(pre[i] * xv);
            }
            want[r * hidden + h] = s;
            mag[r * hidden + h] = wmag * xmax + smag;
        }
    }

    pub fn reference(self: *MixCase, want: []f64, mag: []f64) !void {
        const width = self.shape.k;
        const mcount = self.mixesCount();
        for (0..self.shape.rows) |r| {
            var sq: f64 = 0;
            var dot: [8]f64 = @splat(0);
            var mdot: [8]f64 = @splat(0);
            for (0..width) |k| {
                const v: f64 = c.fromBf16(self.x[r * width + k]);
                sq += v * v;
                for (0..mcount) |j| {
                    const p = v * @as(f64, self.fn_w[j * width + k]);
                    dot[j] += p;
                    mdot[j] += @abs(p);
                }
            }
            self.evalRow(r, sq, sq, dot[0..mcount], mdot[0..mcount], want, mag);
        }
    }

    pub fn simulate(self: *MixCase, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const width = self.shape.k;
        const mcount = self.mixesCount();
        const terms = try a.alloc(f64, width);
        defer a.free(terms);
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        for (0..self.shape.rows) |r| {
            for (terms, 0..) |*t, k| {
                const v: f64 = c.fromBf16(self.x[r * width + k]);
                t.* = v * v;
            }
            const sq = rows_.sumIn(acc, order, terms);
            var dot: [8]f64 = @splat(0);
            for (0..mcount) |j| {
                for (terms, 0..) |*t, k| t.* = @as(f64, c.fromBf16(self.x[r * width + k])) * @as(f64, self.fn_w[j * width + k]);
                dot[j] = rows_.sumIn(acc, order, terms);
            }
            self.evalRow(r, sq, sq, dot[0..mcount], dot[0..mcount], out, mag);
        }
    }
};

pub const MergeCase = struct {
    rig: rig_.Rig,
    hc: *const Hc,
    ctl: Control,
    name: []const u8,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    res: []u16 = &.{},
    f: []u16 = &.{},
    post: []f32 = &.{},
    dev_res: u64 = 0,
    dev_f: u64 = 0,
    dev_post: u64 = 0,
    dev_out: u64 = 0,
    runs: i32 = 0,

    pub fn kernel(self: *MergeCase) g.Kernel {
        return .{
            .spec = .{
                .name = self.name,
                .accum = .none,
                .depth = zeroDepth,
                .ops = 2,
                .out = .bf16,
                .tiles = .{ .{}, .{ .size = 256 }, .{} },
                .served = &merge_served,
            },
            .ctx = self,
            .vt = &vt,
            .control = controlGate(self.ctl),
        };
    }
    const vt = g.hooks(MergeCase);

    pub fn deinit(self: *MergeCase) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn count(self: *const MergeCase) usize {
        return self.shape.rows * 4 * self.shape.n;
    }

    pub fn prepare(self: *MergeCase, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.runs = 0;
        const a = self.arena.allocator();
        const total = shape.rows * 4 * shape.n;
        self.res = try a.alloc(u16, total);
        self.f = try a.alloc(u16, shape.rows * shape.n);
        self.post = try a.alloc(f32, shape.rows * 4);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.res) |*v| v.* = c.toBf16(rng.float(f32) * 4.0 - 2.0);
        for (self.f) |*v| v.* = c.toBf16(rng.float(f32) * 4.0 - 2.0);
        for (self.post) |*v| v.* = rng.float(f32) * 2.0;
        if (fill == .nonfinite) {
            self.res[0] = c.toBf16(std.math.nan(f32));
            if (self.f.len > 1) self.f[1] = c.toBf16(std.math.inf(f32));
            if (self.post.len > 2) self.post[2] = -std.math.inf(f32);
        }
        self.dev_res = try self.rig.input(std.mem.sliceAsBytes(self.res));
        self.dev_f = try self.rig.input(std.mem.sliceAsBytes(self.f));
        self.dev_post = try self.rig.input(std.mem.sliceAsBytes(self.post));
        self.dev_out = try self.rig.output(total * 2);
        return true;
    }

    pub fn run(self: *MergeCase) !void {
        try self.hc.merge(self.rig.s, self.dev_res, self.dev_f, self.dev_post, self.dev_out, self.shape.rows, self.shape.n, self.ctl, self.runs);
        self.runs += 1;
        try self.rig.s.synchronize();
    }

    pub fn reference(self: *MergeCase, want: []f64, mag: []f64) !void {
        const hidden = self.shape.n;
        for (0..self.count()) |i| {
            const row = i / (4 * hidden);
            const s = (i / hidden) % 4;
            const h = i % hidden;
            const p: f64 = self.post[row * 4 + s];
            const fv: f64 = c.fromBf16(self.f[row * hidden + h]);
            const rv: f64 = c.fromBf16(self.res[i]);
            want[i] = p * fv + rv;
            mag[i] = @abs(p * fv) + @abs(rv);
        }
    }
};

pub const Set = struct {
    gpa: std.mem.Allocator,
    hc: Hc,
    mixes: [11]MixCase = undefined,
    merges: [4]MergeCase = undefined,
    engram: *@import("contract_hy4_engram.zig").Set = undefined,
    rows: *@import("contract_hy4_rows.zig").Set = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        set.* = .{ .gpa = gpu.gpa, .hc = try Hc.load(gpu.d) };
        errdefer set.hc.unload();
        const pre_names = [5][]const u8{
            "cuda hy4 hy4_hc_pre out",
            "control: hy4 hy4_hc_pre drops NaN",
            "control: hy4 hy4_hc_pre drifts on repeat",
            "control: hy4 hy4_hc_pre skips tail",
            "control: hy4 hy4_hc_pre sums in bf16",
        };
        const head_names = [5][]const u8{
            "cuda hy4 hy4_hc_head",
            "control: hy4 hy4_hc_head drops NaN",
            "control: hy4 hy4_hc_head drifts on repeat",
            "control: hy4 hy4_hc_head skips tail",
            "control: hy4 hy4_hc_head sums in bf16",
        };
        for (std.enums.values(Control), pre_names, 0..) |ctl, n, i| {
            set.mixes[i] = .{ .rig = .{ .gpu = gpu, .s = s }, .hc = &set.hc, .kind = .pre_out, .ctl = ctl, .name = n, .arena = .init(gpu.gpa) };
        }
        set.mixes[5] = .{ .rig = .{ .gpu = gpu, .s = s }, .hc = &set.hc, .kind = .pre_post, .ctl = .production, .name = "cuda hy4 hy4_hc_pre post", .arena = .init(gpu.gpa) };
        for (std.enums.values(Control), head_names, 0..) |ctl, n, i| {
            set.mixes[6 + i] = .{ .rig = .{ .gpu = gpu, .s = s }, .hc = &set.hc, .kind = .head_out, .ctl = ctl, .name = n, .arena = .init(gpu.gpa) };
        }
        const merge_names = [4][]const u8{
            "cuda hy4 hy4_hc_merge",
            "control: hy4 hy4_hc_merge drops NaN",
            "control: hy4 hy4_hc_merge drifts on repeat",
            "control: hy4 hy4_hc_merge skips tail",
        };
        for ([_]Control{ .production, .nan, .repeat, .tail }, merge_names, 0..) |ctl, n, i| {
            set.merges[i] = .{ .rig = .{ .gpu = gpu, .s = s }, .hc = &set.hc, .ctl = ctl, .name = n, .arena = .init(gpu.gpa) };
        }
        for (&set.mixes) |*x| try list.append(a, x.kernel());
        for (&set.merges) |*x| try list.append(a, x.kernel());
        set.engram = try @import("contract_hy4_engram.zig").Set.init(gpu, s, a, list);
        errdefer set.engram.deinit();
        set.rows = try @import("contract_hy4_rows.zig").Set.init(gpu, s, a, list);
        return set;
    }

    pub fn deinit(set: *Set) void {
        set.rows.deinit();
        set.engram.deinit();
        for (&set.mixes) |*x| x.deinit();
        for (&set.merges) |*x| x.deinit();
        set.hc.unload();
        set.gpa.destroy(set);
    }
};
