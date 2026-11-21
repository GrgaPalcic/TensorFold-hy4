//! hy4_rows.cu under the contract: Hy4-flash's lm_head and router at 5+ rows, bit-equal to tf_dense_rows_1 a row.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;
const hr = hy4.cuda_mlp.cuda_rows;
const Control = hr.Control;
const Tile = hr.Tile;

/// A lane's every 256th octet in order (k / 32 fma), then the warp's 5-step tree.
fn chain(s: c.Shape) usize {
    return s.k / 32 + 5;
}

/// An odd 4,103-column cut of lm_head's vocab slice and the router (512 x 3,072), at rows the wrapper sends each tile.
const served = [2][2]c.Shape{
    .{ .{ .rows = 16, .n = 4103, .k = 3072 }, .{ .rows = 200, .n = 512, .k = 3072 } },
    .{ .{ .rows = 16, .n = 512, .k = 3072 }, .{ .rows = 5, .n = 512, .k = 3072 } },
};

fn controlGate(ctl: Control) ?g.Gate {
    return switch (ctl) {
        .production => null,
        .nan => .nonfinite,
        .repeat => .repeat,
        .tail => .shapes,
        .sums => .sums,
    };
}

pub const Case = struct {
    rig: rig_.Rig,
    rows: *const hr.Rows,
    dense: *const cuda.dense_rows.DenseRows,
    tile: Tile,
    ctl: Control,
    name: []const u8,
    d: ?Dense = null,
    dev: [3]u64 = @splat(0),
    runs: i32 = 0,
    base: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *Case) void {
        if (self.d) |d| d.deinit();
        self.base.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    pub fn kernel(self: *Case) g.Kernel {
        const i = @backingInt(self.tile);
        return .{
            .spec = .{
                .name = self.name,
                .accum = .f32,
                .depth = chain,
                .out = .f32,
                .replaces = "tf_dense_rows_1 launched a row at a time",
                .tiles = .{ .{ .size = hr.tile_rows[i] }, .{ .size = hr.warps * hr.tile_cols[i] }, .{ .size = 256, .step = 8, .min = 8 } },
                .served = &served[i],
            },
            .ctx = self,
            .vt = &vt,
            .control = controlGate(self.ctl),
        };
    }

    const vt = g.hooks(Case);

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k % 8 != 0 or shape.k == 0 or shape.n == 0 or shape.rows == 0) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 1);
        const bits = try a.alloc(u16, d.w.len);
        defer a.free(bits);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x77);
        const rng = prng.random();
        const k = shape.k;
        for (bits) |*b| b.* = c.toBf16(if (fill == .stress) 0 else rng.float(f32) * 2 - 1);
        if (fill == .stress) for (0..shape.n) |j| for (0..k / 2) |i| {
            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
            bits[j * k + i] = c.toBf16(v);
            bits[j * k + k / 2 + i] = c.toBf16(-v);
        };
        for (d.w, bits) |*w, b| w.* = c.fromBf16(b);
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(bits));
        self.dev[2] = try self.rig.output(d.outputs() * 4);
        return true;
    }

    /// One launch at this entry's tile whatever the row count (the repeat control drifts with the run count).
    pub fn run(self: *Case) !void {
        const sh = self.d.?.shape;
        self.runs += 1;
        try self.rows.tiled(self.tile, self.ctl, self.runs, self.rig.s, self.dev[0], sh.k, self.dev[1], self.dev[2], sh.n, sh.rows, sh.n, sh.k);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Case) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        return self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.d.?.simulate(acc, order, out);
    }

    /// tf_dense_rows_1 on the same inputs, one launch a row: the bits the wide and narrow tiles must give.
    pub fn replaced(self: *Case) ![]const u8 {
        const sh = self.d.?.shape;
        const n = self.count();
        var tmp = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, n * 4);
        defer tmp.free();
        for (0..sh.rows) |r| try self.dense.rows(self.rig.s, self.dev[0] + r * sh.k * 2, sh.k, self.dev[1], tmp.ptr + r * sh.n * 4, sh.n, 1, sh.n, sh.k);
        try self.rig.s.synchronize();
        try self.base.resize(self.rig.gpu.gpa, n * 4);
        try tmp.download(0, self.base.items);
        return self.base.items;
    }
};

pub const Set = struct {
    gpa: std.mem.Allocator,
    rows: hr.Rows,
    dense: cuda.dense_rows.DenseRows,
    cases: [10]Case = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var rows = try hr.Rows.load(gpu.d);
        errdefer rows.unload();
        set.* = .{ .gpa = gpu.gpa, .rows = rows, .dense = try cuda.dense_rows.DenseRows.load(gpu.d) };
        const names = [2][5][]const u8{ .{
            "cuda hy4 hy4_rows_wide",
            "control: hy4 hy4_rows_wide drops NaN",
            "control: hy4 hy4_rows_wide drifts on repeat",
            "control: hy4 hy4_rows_wide skips tail",
            "control: hy4 hy4_rows_wide sums in fp16",
        }, .{
            "cuda hy4 hy4_rows_narrow",
            "control: hy4 hy4_rows_narrow drops NaN",
            "control: hy4 hy4_rows_narrow drifts on repeat",
            "control: hy4 hy4_rows_narrow skips tail",
            "control: hy4 hy4_rows_narrow sums in fp16",
        } };
        for ([_]Tile{ .wide, .narrow }, names, 0..) |t, row, i| for (std.enums.values(Control), row, 0..) |ctl, n, j| {
            set.cases[i * 5 + j] = .{ .rig = .{ .gpu = gpu, .s = s }, .rows = &set.rows, .dense = &set.dense, .tile = t, .ctl = ctl, .name = n };
        };
        for (&set.cases) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.cases) |*x| x.deinit();
        set.dense.unload();
        set.rows.unload();
        set.gpa.destroy(set);
    }
};
