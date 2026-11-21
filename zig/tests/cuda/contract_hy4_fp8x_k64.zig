//! Contract suite for `fp8_expert_prompt_k64_kernel`: bit-equal to `fp8_expert_prompt_kernel` (f32 & bf16 down).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const fx8 = cuda.experts.Fp8Experts;

fn zeroDepth(_: c.Shape) usize {
    return 0;
}

pub const Mode = enum { down_f32, down_bf16 };
pub const Fault = enum { none, bad_scale };

const rule = "bit-identical to fp8_expert_prompt_kernel across finite and non-finite inputs (NaN/Inf propagates cleanly)";
const served = [_]c.Shape{
    .{ .rows = 1, .n = 256, .k = 256 },
    .{ .rows = 17, .n = 256, .k = 256 },
    .{ .rows = 65, .n = 256, .k = 256 },
    .{ .rows = 130, .n = 256, .k = 256 },
};

const n_experts: usize = 4;
const top_k: usize = 2;

pub const Case = struct {
    rig: rig_.Rig,
    fp8x: *const fx8,
    ctl_dn: ?cuda.Function = null,
    mode: Mode,
    fault: Fault,
    name: []const u8,
    shape: c.Shape = undefined,
    n_items: usize = 0,
    layer: cuda.experts.Layer = undefined,
    plan: cuda.grouped.Plan = undefined,
    dev_x: u64 = 0,
    dev_act: u64 = 0,
    dev_out: u64 = 0,
    dev_ref_act: u64 = 0,
    dev_ref_out: u64 = 0,
    base: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *Case) void {
        self.base.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    pub fn kernel(self: *Case) g.Kernel {
        const ctl: ?g.Gate = if (self.fault == .none) null else .shapes;
        return .{
            .spec = .{
                .name = self.name,
                .accum = .none,
                .depth = zeroDepth,
                .out = .raw,
                .replaces = "fp8_expert_prompt_kernel",
                .nan = .{ .stated = rule },
                .tiles = .{ .{}, .{ .size = 128, .step = 128, .min = 128 }, .{ .size = 128, .step = 128, .min = 128 } },
                .served = &served,
            },
            .ctx = self,
            .vt = &vt,
            .control = ctl,
        };
    }

    const vt = g.hooks(Case);

    fn f32ToBf16(v: f32) u16 {
        const bits: u32 = @bitCast(v);
        if ((bits & 0x7fffffff) > 0x7f800000) return 0x7fc0;
        const lsb = (bits >> 16) & 1;
        return @truncate((bits +% 0x7fff +% lsb) >> 16);
    }

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.n % 128 != 0 or shape.k % 128 != 0) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        const pairs = shape.rows * top_k;
        const d = shape.k;
        const ni = shape.n;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6b3634);
        const rng = prng.random();

        const x_h = try a.alloc(u16, shape.rows * d);
        defer a.free(x_h);
        for (x_h) |*v| v.* = f32ToBf16(rng.float(f32) * 1.5 - 0.75);
        if (fill == .nonfinite) {
            x_h[0] = 0x7fc0; // qNaN
            if (x_h.len > d) x_h[d + 7] = 0x7f80; // +Inf
        }

        // Pack synthetic e4m3 gate/up & down weights for `n_experts`
        const up_raw = try a.alloc(u32, n_experts * ni * d / 2);
        defer a.free(up_raw);
        const dn_raw = try a.alloc(u32, n_experts * ni * d / 4);
        defer a.free(dn_raw);
        const g_tmp = try a.alloc(u8, ni * d);
        defer a.free(g_tmp);
        const u_tmp = try a.alloc(u8, ni * d);
        defer a.free(u_tmp);
        const d_tmp = try a.alloc(u8, d * ni);
        defer a.free(d_tmp);
        for (0..n_experts) |e| {
            for (g_tmp) |*b| b.* = @intCast(0x28 + (rng.int(u8) & 0x1f)); // finite e4m3
            for (u_tmp) |*b| b.* = @intCast(0x28 + (rng.int(u8) & 0x1f) | ((rng.int(u8) & 1) << 7));
            for (d_tmp) |*b| b.* = @intCast(0x28 + (rng.int(u8) & 0x1f) | ((rng.int(u8) & 1) << 7));
            try fx8.packGateUp(a, up_raw[e * ni * d / 2 ..][0 .. ni * d / 2], g_tmp, u_tmp, ni, d);
            fx8.packOne(dn_raw[e * ni * d / 4 ..][0 .. ni * d / 4], d_tmp, d, ni);
        }

        const n_su = n_experts * 2 * (ni / 128) * (d / 128);
        const n_sd = n_experts * (d / 128) * (ni / 128);
        const su_h = try a.alloc(f32, n_su);
        defer a.free(su_h);
        const sd_h = try a.alloc(f32, n_sd);
        defer a.free(sd_h);
        for (su_h) |*v| v.* = 0.0625 + rng.float(f32) * 0.25;
        for (sd_h) |*v| v.* = 0.0625 + rng.float(f32) * 0.25;

        // Build tile=64 grouped plan on host (expert 0 splits into multiple items at rows=65, 130)
        var buckets: [n_experts]std.ArrayList(i32) = .{ .empty, .empty, .empty, .empty };
        defer for (&buckets) |*b| b.deinit(a);
        for (0..pairs) |pr| {
            const e: usize = if (pr % 3 == 0) 0 else (pr % n_experts);
            try buckets[e].append(a, @intCast(pr));
        }
        var items_h: std.ArrayList(i32) = .empty;
        defer items_h.deinit(a);
        var mem_h: std.ArrayList(i32) = .empty;
        defer mem_h.deinit(a);
        for (0..n_experts) |e| {
            const list = buckets[e].items;
            var off: usize = 0;
            while (off < list.len) : (off += 64) {
                const cnt = @min(@as(usize, 64), list.len - off);
                try items_h.appendSlice(a, &.{ @intCast(e), @intCast(mem_h.items.len), @intCast(cnt) });
                try mem_h.appendSlice(a, list[off..][0..cnt]);
            }
        }
        self.n_items = items_h.items.len / 3;
        const cnt_h = [1]i32{@intCast(self.n_items)};

        self.dev_x = try self.rig.input(std.mem.sliceAsBytes(x_h));
        self.layer = .{
            .format = .fp8g,
            .up = try self.rig.input(std.mem.sliceAsBytes(up_raw)),
            .down = try self.rig.input(std.mem.sliceAsBytes(dn_raw)),
            .up_scale = try self.rig.input(std.mem.sliceAsBytes(su_h)),
            .down_scale = try self.rig.input(std.mem.sliceAsBytes(sd_h)),
            .width = ni,
            .dims = d,
            .experts = n_experts,
            .limit = 10.0,
        };
        self.plan = .{
            .items = try self.rig.input(std.mem.sliceAsBytes(items_h.items)),
            .counts = try self.rig.input(std.mem.sliceAsBytes(&cnt_h)),
            .members = try self.rig.input(std.mem.sliceAsBytes(mem_h.items)),
            .rank = 0,
            .hist = 0,
        };
        const out_elem: usize = if (self.mode == .down_f32) 4 else 2;
        self.dev_act = try self.rig.scratch(pairs * ni * 2);
        self.dev_ref_act = try self.rig.scratch(pairs * ni * 2);
        self.dev_ref_out = try self.rig.scratch(pairs * d * out_elem);
        self.dev_out = try self.rig.output(pairs * d * out_elem);
        return true;
    }

    pub fn run(self: *Case) !void {
        const s = self.rig.s;
        const d = self.shape.k;
        const ni = self.shape.n;
        const dn_epi: cuda.experts.Epi = if (self.mode == .down_f32) .down_f32 else .down_bf16;
        try self.fp8x.run(s, .up, .{ .x = self.dev_x, .stride = d, .slots = top_k, .contig_act = true }, self.layer, self.plan, true, self.dev_act, self.n_items, -1);
        if (self.fault == .bad_scale) {
            var a: cuda.Args = .{};
            a.add(self.dev_act);
            a.add(@as(i32, @intCast(ni)));
            a.add(@as(i32, 0));
            a.add(self.layer.down);
            a.add(self.layer.down_scale);
            a.add(@as(i32, @intCast(ni / 32)));
            a.add(@as(i32, @intCast(d / 32)));
            inline for (.{ self.plan.items, self.plan.counts, self.plan.members, self.dev_out }) |v| a.add(v);
            a.add(@as(i32, @intCast(d)));
            a.add(@as(f32, 0.0));
            a.add(@as(i32, -1));
            try cuda.launch.launch(self.ctl_dn.?, .{ .grid = .{ .x = @intCast(self.n_items) }, .block = .{ .x = 128 } }, s, &a);
        } else {
            try self.fp8x.run(s, dn_epi, .{ .x = self.dev_act, .stride = ni, .contig_act = true }, self.layer, self.plan, true, self.dev_out, self.n_items, -1);
        }
        try s.synchronize();
    }

    pub fn count(self: *const Case) usize {
        const out_elem: usize = if (self.mode == .down_f32) 4 else 2;
        return self.shape.rows * top_k * self.shape.k * out_elem;
    }

    pub fn reference(_: *Case, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn replaced(self: *Case) ![]const u8 {
        const s = self.rig.s;
        const d = self.shape.k;
        const ni = self.shape.n;
        const dn_epi: cuda.experts.Epi = if (self.mode == .down_f32) .down_f32 else .down_bf16;
        try self.fp8x.run(s, .up, .{ .x = self.dev_x, .stride = d, .slots = top_k, .contig_act = false }, self.layer, self.plan, true, self.dev_ref_act, self.n_items, -1);
        try self.fp8x.run(s, dn_epi, .{ .x = self.dev_ref_act, .stride = ni, .contig_act = false }, self.layer, self.plan, true, self.dev_ref_out, self.n_items, -1);
        try s.synchronize();
        try self.base.resize(self.rig.gpu.gpa, self.count());
        try self.rig.gpu.d.check(self.rig.gpu.d.api.cuMemcpyDtoH_v2(self.base.items.ptr, self.dev_ref_out, self.base.items.len), "cuMemcpyDtoH");
        return self.base.items;
    }

    pub fn check(self: *Case, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (!std.mem.eql(u8, got, want)) {
            return "fp8_expert_prompt_k64_kernel output differs from fp8_expert_prompt_kernel";
        }
        if (!try self.rig.intact()) return "fp8_expert_prompt_k64_kernel overwrote a guard region";
        return null;
    }
};
