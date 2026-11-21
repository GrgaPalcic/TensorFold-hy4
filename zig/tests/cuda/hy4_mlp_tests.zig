//! Hy4-flash's dense MLP and LatentMoE at world 1, 2 and 4 (decode and prompt) against hy4_mlp_ref's float64.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4");
const check = @import("check.zig");
const ref = @import("hy4_mlp_ref.zig");

const Gpu = check.Gpu;
const st = core.safetensors;
const Config = hy4.config.Config;
const cw = hy4.cuda_weights;
const cm = hy4.cuda_mlp;
const DeviceBuffer = cuda.DeviceBuffer;

fn tap(f: *const st.File, name: []const u8) !st.Tensor {
    return f.get(name) orelse {
        std.debug.print("the fixture has no {s}\n", .{name});
        return error.MissingTap;
    };
}

fn bf16At(bytes: []const u8, i: usize) f64 {
    return @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)) << 16));
}

fn layerOf(path: []const u8) !usize {
    const base = std.fs.path.basename(path);
    if (!std.mem.startsWith(u8, base, "layer")) return error.NotALayerFixture;
    const end = std.mem.indexOfScalar(u8, base, '_') orelse return error.NotALayerFixture;
    return std.fmt.parseInt(usize, base[5..end], 10);
}

fn verdict(failures: *usize, ok: bool, comptime fmt: []const u8, args: anytype) void {
    if (!ok) failures.* += 1;
    std.debug.print("{s} " ++ fmt ++ "\n", .{if (ok) "PASS" else "FAIL"} ++ args);
}

/// Elementwise and relative L2 error of bf16 rows against float64, optionally skipping near-tie rows.
const Gap = struct {
    max_abs: f64 = 0,
    mean_abs: f64 = 0,
    rel_l2: f64 = 0,
    bad_vals: usize = 0,
    rows_used: usize = 0,

    fn of(got: []const u8, want: []const f64, rows: usize, d: usize, skip: ?[]const bool) Gap {
        var g: Gap = .{};
        var sum_abs: f64 = 0;
        var err_sq: f64 = 0;
        var ref_sq: f64 = 0;
        var count: usize = 0;
        for (0..rows) |r| {
            if (skip) |sk| if (sk[r]) continue;
            g.rows_used += 1;
            for (0..d) |i| {
                const idx = r * d + i;
                const v = bf16At(got, idx);
                if (!std.math.isFinite(v)) {
                    g.bad_vals += 1;
                    continue;
                }
                const e = @abs(v - want[idx]);
                g.max_abs = @max(g.max_abs, e);
                sum_abs += e;
                err_sq += e * e;
                ref_sq += want[idx] * want[idx];
                count += 1;
            }
        }
        if (count > 0) g.mean_abs = sum_abs / @as(f64, @floatFromInt(count));
        if (ref_sq > 0) g.rel_l2 = @sqrt(err_sq / ref_sq);
        return g;
    }

    fn within(mine: Gap, theirs: Gap) bool {
        if (mine.bad_vals != 0 or mine.rows_used == 0) return false;
        return mine.rel_l2 <= theirs.rel_l2 and mine.mean_abs <= theirs.mean_abs and mine.max_abs <= theirs.max_abs;
    }

    fn withinSlice(mine: Gap, slice_theirs: Gap, layer_theirs: Gap) bool {
        if (mine.bad_vals != 0 or mine.rows_used == 0) return false;
        return mine.rel_l2 <= @max(1.1 * slice_theirs.rel_l2, 2.5e-3) and
            mine.mean_abs <= @max(1.1 * slice_theirs.mean_abs, 1e-3) and
            mine.max_abs <= @max(layer_theirs.max_abs, 1e-2);
    }
};

fn distinctValid(picks: []const i32, experts: usize) bool {
    for (picks, 0..) |p, i| {
        if (p < 0 or @as(usize, @intCast(p)) >= experts) return false;
        for (picks[0..i]) |q| if (p == q) return false;
    }
    return true;
}

fn sameSetU32(a: []const i32, b: []const u32) usize {
    var miss: usize = 0;
    for (a) |x| {
        if (x < 0 or std.mem.indexOfScalar(u32, b, @intCast(x)) == null) miss += 1;
    }
    return miss;
}

fn sameSetI32(a: []const i32, b: []const i32) usize {
    var miss: usize = 0;
    for (a) |x| {
        if (std.mem.indexOfScalar(i32, b, x) == null) miss += 1;
    }
    return miss;
}

const WorldMlp = struct {
    w: cw.Weights,
    ranks: [4]cw.Mlp = undefined,
    world: usize,

    fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, l: hy4.names.Layer, c: Config, world: usize) !WorldMlp {
        var wm: WorldMlp = .{ .w = cw.Weights.init(gpa, d), .world = world };
        errdefer wm.w.deinit();
        for (0..world) |r| wm.ranks[r] = try wm.w.mlp(l, c, .{ .rank = r, .world = world });
        return wm;
    }

    fn deinit(wm: *WorldMlp) void {
        wm.w.deinit();
    }
};

const Work = struct {
    gpu: Gpu,
    s: cuda.Stream,
    x_in: DeviceBuffer,
    x_norm: DeviceBuffer,
    partials: DeviceBuffer,
    sum: DeviceBuffer,
    out: DeviceBuffer,

    fn init(gpu: Gpu, in_bytes: []const u8, n: usize) !Work {
        var s = try cuda.Stream.init(gpu.d, false);
        errdefer s.deinit();
        var x_in = try DeviceBuffer.fromHost(gpu.d, in_bytes);
        errdefer x_in.free();
        var x_norm = try DeviceBuffer.alloc(gpu.d, n * 2);
        errdefer x_norm.free();
        var partials = try DeviceBuffer.alloc(gpu.d, 4 * n * 4);
        errdefer partials.free();
        var sum = try DeviceBuffer.alloc(gpu.d, n * 4);
        errdefer sum.free();
        const out = try DeviceBuffer.alloc(gpu.d, n * 2);
        return .{ .gpu = gpu, .s = s, .x_in = x_in, .x_norm = x_norm, .partials = partials, .sum = sum, .out = out };
    }

    fn deinit(w: *Work) void {
        w.out.free();
        w.sum.free();
        w.partials.free();
        w.x_norm.free();
        w.x_in.free();
        w.s.deinit();
    }

    /// Runs `wm` over the first `m` rows (summing rank slices when world > 1) and downloads the bf16 output.
    fn forward(w: *Work, k: *const cm.Kernels, c: Config, wm: *const WorldMlp, sc: *const cm.Scratch, m: usize, prompt: bool) ![]u8 {
        try w.out.fill8(0xff, w.s.handle);
        try w.sum.fill8(0xff, w.s.handle);
        if (wm.world == 1) {
            try cm.norm(k, w.s, c, wm.ranks[0], w.x_in.ptr, w.x_norm.ptr, m);
            try cm.partial(k, w.s, c, wm.ranks[0], sc, w.x_norm.ptr, w.sum.ptr, m, prompt);
        } else {
            try w.partials.fill8(0xff, w.s.handle);
            for (0..wm.world) |r| {
                try cm.norm(k, w.s, c, wm.ranks[r], w.x_in.ptr, w.x_norm.ptr, m);
                try cm.partial(k, w.s, c, wm.ranks[r], sc, w.x_norm.ptr, w.partials.ptr + r * m * c.hidden * 4, m, prompt);
            }
            try cm.rankSum(k, w.s, w.partials.ptr, wm.world, m, c.hidden, w.sum.ptr);
        }
        try cm.finish(k, w.s, c, wm.ranks[0], w.sum.ptr, w.out.ptr, m);
        try w.s.synchronize();
        const got = try w.gpu.gpa.alloc(u8, m * c.hidden * 2);
        errdefer w.gpu.gpa.free(got);
        try w.out.download(0, got);
        return got;
    }
};

/// Downloads `len` bytes from a device address `ptr`.
fn readDev(gpu: Gpu, ptr: u64, out: []u8) !void {
    const b: DeviceBuffer = .{ .d = gpu.d, .ptr = ptr, .len = out.len };
    try b.download(0, out);
}

const RouteCheck = struct {
    picks: []i32,
    wts: []f32,
    tie_skip: []bool,
    both_skip: []bool,
    tie_rows: usize = 0,
    torch_tie_rows: usize = 0,
    forced_cuda: ?ref.Result = null,
    forced_torch: ?ref.Result = null,

    fn deinit(rc: *RouteCheck, gpa: std.mem.Allocator) void {
        if (rc.forced_cuda) |*r| r.deinit(gpa);
        if (rc.forced_torch) |*r| r.deinit(gpa);
        gpa.free(rc.both_skip);
        gpa.free(rc.tie_skip);
        gpa.free(rc.wts);
        gpa.free(rc.picks);
    }
};

fn checkRouter(gpu: Gpu, c: Config, l: hy4.names.Layer, f: *const st.File, sc: *const cm.Scratch, x64: []const f64, ref64: *const ref.Result, layer: usize, rows: usize, failures: *usize) !RouteCheck {
    const gpa = gpu.gpa;
    const kk = c.top_k;
    const pairs = rows * kk;
    var rc: RouteCheck = .{
        .picks = try gpa.alloc(i32, pairs),
        .wts = try gpa.alloc(f32, pairs),
        .tie_skip = try gpa.alloc(bool, rows),
        .both_skip = try gpa.alloc(bool, rows),
    };
    errdefer rc.deinit(gpa);
    try readDev(gpu, sc.picks, std.mem.sliceAsBytes(rc.picks));
    try readDev(gpu, sc.wts, std.mem.sliceAsBytes(rc.wts));

    const torch_picks = try gpa.alloc(i32, pairs);
    defer gpa.free(torch_picks);
    const torch_wts = try gpa.alloc(f32, pairs);
    defer gpa.free(torch_wts);
    @memcpy(std.mem.sliceAsBytes(torch_picks), (try tap(f, "router_ids16")).bytes);
    @memcpy(std.mem.sliceAsBytes(torch_wts), (try tap(f, "router_w16")).bytes);

    var valid = true;
    var diff_pairs: usize = 0;
    var vs_torch_rows: usize = 0;
    var max_margin: f64 = 0;
    var max_wt_err: f64 = 0;
    for (0..rows) |r| {
        const gp = rc.picks[r * kk ..][0..kk];
        const gw = rc.wts[r * kk ..][0..kk];
        const rp = ref64.picks[r * kk ..][0..kk];
        const rw = ref64.wts[r * kk ..][0..kk];
        const tp = torch_picks[r * kk ..][0..kk];
        valid = valid and distinctValid(gp, c.experts);
        var wsum: f64 = 0;
        var exact_order = true;
        for (gw, gp, rp, 0..) |w, p, want_p, k| {
            if (!std.math.isFinite(w) or w <= 0) valid = false;
            wsum += w;
            if (p < 0 or @as(u32, @intCast(p)) != want_p) exact_order = false;
            _ = k;
        }
        if (@abs(wsum - @as(f64, c.routed_scale)) > 1e-4) valid = false;
        if (exact_order) for (gw, rw) |w, want_w| {
            max_wt_err = @max(max_wt_err, @abs(@as(f64, w) - want_w));
        };
        const miss = sameSetU32(gp, rp);
        const tmiss = sameSetU32(tp, rp);
        rc.tie_skip[r] = miss != 0;
        rc.both_skip[r] = miss != 0 or tmiss != 0;
        if (miss != 0) {
            rc.tie_rows += 1;
            diff_pairs += miss;
            max_margin = @max(max_margin, ref64.margin[r]);
        }
        if (tmiss != 0) rc.torch_tie_rows += 1;
        if (sameSetI32(gp, tp) != 0) vs_torch_rows += 1;
    }
    const ok = valid and max_margin <= 1e-2 and max_wt_err <= 2e-2;
    verdict(failures, ok, "layer {d} t{d} router: {d}/{d} rows == fp64 ({d}/{d} pairs differ, max tie margin {e:.3}; != torch16 on {d} rows), max |wt - fp64| {e:.3}", .{ layer, rows, rows - rc.tie_rows, rows, diff_pairs, pairs, max_margin, vs_torch_rows, max_wt_err });
    if (rc.tie_rows > 0) rc.forced_cuda = try ref.run(gpa, c, l, x64, rows, .{ .picks = rc.picks, .wts = rc.wts });
    if (rc.torch_tie_rows > 0) rc.forced_torch = try ref.run(gpa, c, l, x64, rows, .{ .picks = torch_picks, .wts = torch_wts });
    return rc;
}

/// `hy4-mlp MODEL FIXTURE...`: dense MLP and LatentMoE at world 1, 2 and 4 (decode and prompt) against float64.
pub fn mlp(gpu: Gpu, model: []const u8, fixtures: []const [:0]const u8) !void {
    if (fixtures.len == 0) return error.MissingArgument;
    const gpa = gpu.gpa;
    const c = try Config.read(gpa, gpu.io, model);
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, gpu.io, model);
    defer ck.close();
    var k = try cm.Kernels.load(gpu.ctx);
    defer k.deinit();
    var failures: usize = 0;
    for (fixtures) |fixture| try mlpOne(gpu, c, &ck, &k, fixture, &failures);
    if (failures > 0) return error.Mismatch;
}

fn mlpOne(gpu: Gpu, c: Config, ck: *core.checkpoint.Checkpoint, k: *const cm.Kernels, fixture: []const u8, failures: *usize) !void {
    const gpa = gpu.gpa;
    const layer = try layerOf(fixture);
    var f = try st.File.open(gpa, gpu.io, fixture);
    defer f.close(gpu.io);
    const l = try hy4.names.layer(ck, c, layer);
    const in_tap = try tap(&f, "ffn_hc_out16");
    const torch_bytes = (try tap(&f, "mlp_out16")).bytes;
    const rows = in_tap.dim(1);
    const d = c.hidden;
    const n = rows * d;
    std.debug.print("hy4-mlp layer {d} ({s}), {d} rows\n", .{ layer, if (c.dense[layer]) "dense" else "moe", rows });

    const x64 = try gpa.alloc(f64, n);
    defer gpa.free(x64);
    for (0..n) |i| x64[i] = bf16At(in_tap.bytes, i);
    var ref64 = try ref.run(gpa, c, l, x64, rows, null);
    defer ref64.deinit(gpa);

    var work = try Work.init(gpu, in_tap.bytes, n);
    defer work.deinit();
    var sc = try cm.Scratch.init(gpa, gpu.d, &k.lane, c, rows);
    defer sc.deinit();
    var small_sc = try cm.Scratch.init(gpa, gpu.d, &k.lane, c, 16);
    defer small_sc.deinit();

    var rc: ?RouteCheck = null;
    defer if (rc) |*r| r.deinit(gpa);
    var w1_ref: ?[]u8 = null;
    defer if (w1_ref) |b| gpa.free(b);

    for ([_]usize{ 1, 2, 4 }) |world| {
        var wm = try WorldMlp.load(gpa, gpu.d, l, c, world);
        defer wm.deinit();

        for ([_]bool{ false, true }) |prompt| {
            const mode = if (prompt) "prompt" else "decode";
            const got = try work.forward(k, c, &wm, &sc, rows, prompt);
            defer gpa.free(got);

            if (world == 1 and !prompt) {
                w1_ref = try gpa.dupe(u8, got);
                if (!c.dense[layer]) rc = try checkRouter(gpu, c, l, &f, &sc, x64, &ref64, layer, rows, failures);
            }

            const want_cuda = if (rc) |r| (if (r.forced_cuda) |fc| fc.out else ref64.out) else ref64.out;
            const want_torch = if (rc) |r| (if (r.forced_torch) |ft| ft.out else ref64.out) else ref64.out;
            const mine = Gap.of(got, want_cuda, rows, d, null);
            const theirs = Gap.of(torch_bytes, want_torch, rows, d, null);
            var unforced_ok = true;
            if (rc) |r| if (r.tie_rows > 0 or r.torch_tie_rows > 0) {
                const mu = Gap.of(got, ref64.out, rows, d, r.both_skip);
                const tu = Gap.of(torch_bytes, ref64.out, rows, d, r.both_skip);
                unforced_ok = Gap.within(mu, tu);
            };
            var w1_max: f64 = 0;
            for (0..n) |i| w1_max = @max(w1_max, @abs(bf16At(got, i) - bf16At(w1_ref.?, i)));
            verdict(failures, Gap.within(mine, theirs) and unforced_ok, "layer {d} t{d} w{d} {s} vs fp64: rel_l2 {e:.3} (torch {e:.3}), max {e:.3} (torch {e:.3}), mean {e:.3} (torch {e:.3}), max |.-w1| {e:.3}", .{ layer, rows, world, mode, mine.rel_l2, theirs.rel_l2, mine.max_abs, theirs.max_abs, mine.mean_abs, theirs.mean_abs, w1_max });

            var same = true;
            for (0..4) |_| {
                const again = try work.forward(k, c, &wm, &sc, rows, prompt);
                defer gpa.free(again);
                same = same and std.mem.eql(u8, again, got);
            }
            verdict(failures, same, "layer {d} t{d} w{d} {s}: five runs byte-identical", .{ layer, rows, world, mode });
        }

        if (world == 1) {
            const want_cuda = if (rc) |r| (if (r.forced_cuda) |fc| fc.out else ref64.out) else ref64.out;
            const want_torch = if (rc) |r| (if (r.forced_torch) |ft| ft.out else ref64.out) else ref64.out;
            const layer_theirs = Gap.of(torch_bytes, want_torch, rows, d, null);
            var small_ok = true;
            var r1: f64 = 0;
            var r16: f64 = 0;
            for ([_]usize{ 1, 16 }) |m| {
                const got_m = try work.forward(k, c, &wm, &small_sc, m, false);
                defer gpa.free(got_m);
                const gm = Gap.of(got_m, want_cuda[0 .. m * d], m, d, null);
                const tm = Gap.of(torch_bytes[0 .. m * d * 2], want_torch[0 .. m * d], m, d, null);
                if (m == 1) r1 = gm.rel_l2 else r16 = gm.rel_l2;
                small_ok = small_ok and Gap.withinSlice(gm, tm, layer_theirs);
            }
            verdict(failures, small_ok, "layer {d} t{d} w1 small-batch decode vs fp64: m=1 rel_l2 {e:.3}, m=16 rel_l2 {e:.3}", .{ layer, rows, r1, r16 });
        } else if (rc) |r| {
            var same_routes = true;
            const pbytes = std.mem.sliceAsBytes(r.picks);
            const wbytes = std.mem.sliceAsBytes(r.wts);
            const pbuf = try gpa.alloc(u8, pbytes.len);
            defer gpa.free(pbuf);
            const wbuf = try gpa.alloc(u8, wbytes.len);
            defer gpa.free(wbuf);
            for (0..world) |rk| {
                try cm.norm(k, work.s, c, wm.ranks[rk], work.x_in.ptr, work.x_norm.ptr, rows);
                try cm.partial(k, work.s, c, wm.ranks[rk], &sc, work.x_norm.ptr, work.partials.ptr + rk * n * 4, rows, false);
                try work.s.synchronize();
                try readDev(gpu, sc.picks, pbuf);
                try readDev(gpu, sc.wts, wbuf);
                same_routes = same_routes and std.mem.eql(u8, pbuf, pbytes) and std.mem.eql(u8, wbuf, wbytes);
            }
            verdict(failures, same_routes, "layer {d} t{d} w{d}: replicated router picks and weights match w1 on all {d} ranks", .{ layer, rows, world, world });
        }
    }
}
