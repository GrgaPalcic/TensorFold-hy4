//! Hy4-flash Engram fixture gates on GPU: NVMe O_DIRECT & resident modes at world 1, 2, 4 vs fp64 & bf16 references.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4");
const check = @import("check.zig");

const Gpu = check.Gpu;
const st = core.safetensors;
const Config = hy4.config.Config;
const cw = hy4.cuda_weights;
const ce = hy4.cuda_engram;
const ep = hy4.engram_pack;
const er = hy4.engram_reader;
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

fn f64At(bytes: []const u8, i: usize) f64 {
    return @bitCast(std.mem.readInt(u64, bytes[8 * i ..][0..8], .little));
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

const Gap = struct {
    max_abs: f64 = 0,
    mean_abs: f64 = 0,
    rel_l2: f64 = 0,
    bad_vals: usize = 0,

    fn of(got: []const u8, want: []const f64) Gap {
        var g: Gap = .{};
        var sum_abs: f64 = 0;
        var err_sq: f64 = 0;
        var ref_sq: f64 = 0;
        for (want, 0..) |w, i| {
            const v = bf16At(got, i);
            if (!std.math.isFinite(v)) {
                g.bad_vals += 1;
                continue;
            }
            const e = @abs(v - w);
            g.max_abs = @max(g.max_abs, e);
            sum_abs += e;
            err_sq += e * e;
            ref_sq += w * w;
        }
        if (want.len > 0) g.mean_abs = sum_abs / @as(f64, @floatFromInt(want.len));
        if (ref_sq > 0) g.rel_l2 = @sqrt(err_sq / ref_sq);
        return g;
    }

    fn within(mine: Gap, theirs: Gap) bool {
        if (mine.bad_vals != 0) return false;
        return mine.rel_l2 <= theirs.rel_l2 and mine.mean_abs <= theirs.mean_abs and mine.max_abs <= theirs.max_abs;
    }

    fn withinSlice(mine: Gap, slice_theirs: Gap, full_theirs: Gap) bool {
        if (mine.bad_vals != 0) return false;
        return mine.rel_l2 <= @max(1.1 * slice_theirs.rel_l2, 4e-3) and
            mine.mean_abs <= @max(1.1 * slice_theirs.mean_abs, 1e-4) and
            mine.max_abs <= @max(full_theirs.max_abs, 5e-3);
    }
};

const PromptIds = struct {
    ids: []u32,
    positions: []u32,

    fn load(gpa: std.mem.Allocator, io: std.Io, fixture: []const u8, rows: usize) !PromptIds {
        const dir = std.fs.path.dirname(fixture) orelse ".";
        var nbuf: [64]u8 = undefined;
        const path = try std.fs.path.join(gpa, &.{ dir, try std.fmt.bufPrint(&nbuf, "ids_t{d}.json", .{rows}) });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
        defer gpa.free(text);
        var parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.BadPromptJson;
        const iv = parsed.value.object.get("ids") orelse return error.BadPromptJson;
        if (iv != .array or iv.array.items.len != rows) return error.BadPromptJson;
        const ids = try gpa.alloc(u32, rows);
        errdefer gpa.free(ids);
        for (iv.array.items, ids) |x, *o| {
            if (x != .integer or x.integer < 0 or x.integer > std.math.maxInt(u32)) return error.BadPromptJson;
            o.* = @intCast(x.integer);
        }
        const positions = try gpa.alloc(u32, rows);
        errdefer gpa.free(positions);
        if (parsed.value.object.get("positions")) |pv| {
            if (pv != .array or pv.array.items.len != rows) return error.BadPromptJson;
            for (pv.array.items, positions) |x, *o| {
                if (x != .integer or x.integer < 0 or x.integer > std.math.maxInt(u32)) return error.BadPromptJson;
                o.* = @intCast(x.integer);
            }
        } else for (positions, 0..) |*p, i| {
            p.* = @intCast(i);
        }
        return .{ .ids = ids, .positions = positions };
    }

    fn deinit(p: *PromptIds, gpa: std.mem.Allocator) void {
        gpa.free(p.positions);
        gpa.free(p.ids);
        p.* = undefined;
    }
};

const WorldEngram = struct {
    gpa: std.mem.Allocator,
    w: cw.Weights,
    ranks: [4]cw.Engram = undefined,
    readers: [4]er.Reader = undefined,
    world: usize,

    fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, engram_dir: []const u8, l: hy4.names.Layer, c: Config, layer: usize, world: usize, mode: er.Mode) !WorldEngram {
        var we: WorldEngram = .{ .gpa = gpa, .w = cw.Weights.init(gpa, d), .world = world };
        errdefer we.w.deinit();
        var made: usize = 0;
        errdefer for (we.readers[0..made]) |*r| r.close();
        for (0..world) |rk| {
            const at: cw.Rank = .{ .rank = rk, .world = world };
            we.ranks[rk] = try we.w.engram(l, c, layer, at);
            we.readers[rk] = try er.Reader.open(gpa, io, engram_dir, c, .{ .rank = rk, .world = world }, &.{we.ranks[rk].slot}, mode, er.default_cache_rows);
            made += 1;
        }
        return we;
    }

    fn deinit(we: *WorldEngram) void {
        for (we.readers[0..we.world]) |*r| r.close();
        we.w.deinit();
    }
};

const RunOut = struct {
    engram: []u8,
    streams: []u8,
    raw: []u8,

    fn deinit(o: *RunOut, gpa: std.mem.Allocator) void {
        gpa.free(o.raw);
        gpa.free(o.streams);
        gpa.free(o.engram);
        o.* = undefined;
    }
};

const Work = struct {
    gpu: Gpu,
    s: cuda.Stream,
    x_in: DeviceBuffer,
    partials: DeviceBuffer,
    sum: DeviceBuffer,
    engram_out: DeviceBuffer,
    streams_out: DeviceBuffer,

    fn init(gpu: Gpu, in_bytes: []const u8, rows: usize, d: usize) !Work {
        var s = try cuda.Stream.init(gpu.d, false);
        errdefer s.deinit();
        var x_in = try DeviceBuffer.fromHost(gpu.d, in_bytes);
        errdefer x_in.free();
        var partials = try DeviceBuffer.alloc(gpu.d, 4 * rows * ce.projections * d * 4);
        errdefer partials.free();
        var sum = try DeviceBuffer.alloc(gpu.d, rows * ce.projections * d * 4);
        errdefer sum.free();
        var engram_out = try DeviceBuffer.alloc(gpu.d, rows * ce.streams * d * 2);
        errdefer engram_out.free();
        const streams_out = try DeviceBuffer.alloc(gpu.d, rows * ce.streams * d * 2);
        return .{ .gpu = gpu, .s = s, .x_in = x_in, .partials = partials, .sum = sum, .engram_out = engram_out, .streams_out = streams_out };
    }

    fn deinit(w: *Work) void {
        w.streams_out.free();
        w.engram_out.free();
        w.sum.free();
        w.partials.free();
        w.x_in.free();
        w.s.deinit();
    }

    fn forward(w: *Work, k: *const ce.Kernels, c: Config, we: *WorldEngram, sc: *ce.Scratch, tokens: []const u32, positions: []const u32, do_prefetch: bool, want_raw: bool) !RunOut {
        const m = tokens.len;
        const d = c.hidden;
        const elem_bytes = m * ce.streams * d * 2;
        try w.engram_out.fill8(0xff, w.s.handle);
        try w.streams_out.fill8(0xff, w.s.handle);
        try w.sum.fill8(0xff, w.s.handle);
        if (do_prefetch) {
            for (0..we.world) |rk| try ce.prefetch(c, we.ranks[rk], &we.readers[rk], sc, tokens, positions);
        }
        const raw_total = if (want_raw) m * hy4.config.engram_heads * ep.row_bytes else 0;
        const raw = try w.gpu.gpa.alloc(u8, raw_total);
        errdefer w.gpu.gpa.free(raw);
        if (we.world == 1) {
            try ce.partial(k, w.s, c, we.ranks[0], &we.readers[0], sc, tokens, positions, w.sum.ptr);
            if (want_raw) {
                try w.s.synchronize();
                try sc.dev_raw.download(0, raw);
            }
        } else {
            try w.partials.fill8(0xff, w.s.handle);
            var raw_at: usize = 0;
            for (0..we.world) |rk| {
                const off = rk * m * ce.projections * d * 4;
                try ce.partial(k, w.s, c, we.ranks[rk], &we.readers[rk], sc, tokens, positions, w.partials.ptr + off);
                if (want_raw) {
                    const rk_bytes = m * we.ranks[rk].part.heads.len * ep.row_bytes;
                    try w.s.synchronize();
                    try sc.dev_raw.download(0, raw[raw_at..][0..rk_bytes]);
                    raw_at += rk_bytes;
                }
            }
            try ce.rankSum(k, w.s, w.partials.ptr, we.world, m, d, w.sum.ptr);
        }
        try ce.finish(k, w.s, c, we.ranks[0], sc, m, null, w.x_in.ptr, w.sum.ptr, w.engram_out.ptr, w.streams_out.ptr);
        try w.s.synchronize();
        const eo_buf = try w.gpu.gpa.alloc(u8, elem_bytes);
        errdefer w.gpu.gpa.free(eo_buf);
        const st_buf = try w.gpu.gpa.alloc(u8, elem_bytes);
        errdefer w.gpu.gpa.free(st_buf);
        try w.engram_out.download(0, eo_buf);
        try w.streams_out.download(0, st_buf);
        return .{ .engram = eo_buf, .streams = st_buf, .raw = raw };
    }
};

/// `hy4-engram MODEL ENGRAM_DIR FIXTURE...` (or `hy4-engram MODEL FIXTURE...` when `.hy4e` files live inside `MODEL`).
pub fn engram(gpu: Gpu, model: []const u8, rest: []const [:0]const u8) !void {
    if (rest.len == 0) return error.MissingArgument;
    const has_dir = !std.mem.endsWith(u8, rest[0], ".safetensors");
    const engram_dir = if (has_dir) rest[0] else model;
    const fixtures = if (has_dir) rest[1..] else rest;
    if (fixtures.len == 0) return error.MissingArgument;

    const gpa = gpu.gpa;
    const c = try Config.read(gpa, gpu.io, model);
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, gpu.io, model);
    defer ck.close();
    var k = try ce.Kernels.load(gpu.ctx);
    defer k.deinit();
    var failures: usize = 0;
    for (fixtures) |fx| try engramOne(gpu, c, &ck, engram_dir, &k, fx, &failures);
    if (failures > 0) return error.Mismatch;
}

fn engramOne(gpu: Gpu, c: Config, ck: *core.checkpoint.Checkpoint, engram_dir: []const u8, k: *const ce.Kernels, fixture: []const u8, failures: *usize) !void {
    const gpa = gpu.gpa;
    const layer = try layerOf(fixture);
    var f = try st.File.open(gpa, gpu.io, fixture);
    defer f.close(gpu.io);
    const l = try hy4.names.layer(ck, c, layer);
    const in_tap = try tap(&f, "in");
    const eo64_tap = try tap(&f, "engram_out64");
    const eo16_bytes = (try tap(&f, "engram_out16")).bytes;
    const rows = in_tap.dim(1);
    const d = c.hidden;
    const total = rows * ce.streams * d;
    std.debug.print("hy4-engram layer {d}, {d} rows\n", .{ layer, rows });

    var pids = try PromptIds.load(gpa, gpu.io, fixture, rows);
    defer pids.deinit(gpa);

    const eo64 = try gpa.alloc(f64, total);
    defer gpa.free(eo64);
    for (0..total) |i| eo64[i] = f64At(eo64_tap.bytes, i);
    const eo_theirs = Gap.of(eo16_bytes, eo64);

    const ah64_opt = f.get("attn_hc_in64");
    const ah16_opt = f.get("attn_hc_in16");
    var ah64: ?[]f64 = null;
    defer if (ah64) |b| gpa.free(b);
    if (ah64_opt) |t| {
        const buf = try gpa.alloc(f64, total);
        for (0..total) |i| buf[i] = f64At(t.bytes, i);
        ah64 = buf;
    }

    var work = try Work.init(gpu, in_tap.bytes, rows, d);
    defer work.deinit();
    var sc = try ce.Scratch.init(gpa, gpu.d, &k.lane, c, rows);
    defer sc.deinit();
    var small_sc = try ce.Scratch.init(gpa, gpu.d, &k.lane, c, 16);
    defer small_sc.deinit();

    var w1_eo: ?[]u8 = null;
    defer if (w1_eo) |b| gpa.free(b);

    for ([_]usize{ 1, 2, 4 }) |world| {
        var nvme_we = try WorldEngram.load(gpa, gpu.io, gpu.d, engram_dir, l, c, layer, world, .nvme);
        var nvme_out = try work.forward(k, c, &nvme_we, &sc, pids.ids, pids.positions, false, true);
        defer nvme_out.deinit(gpa);

        // Verify async prefetch + repeat determinism on NVMe before closing NVMe handles.
        var pref_out = try work.forward(k, c, &nvme_we, &sc, pids.ids, pids.positions, true, false);
        defer pref_out.deinit(gpa);
        var nvme_same = std.mem.eql(u8, pref_out.engram, nvme_out.engram) and std.mem.eql(u8, pref_out.streams, nvme_out.streams);
        for (0..3) |_| {
            var rep = try work.forward(k, c, &nvme_we, &sc, pids.ids, pids.positions, false, false);
            defer rep.deinit(gpa);
            nvme_same = nvme_same and std.mem.eql(u8, rep.engram, nvme_out.engram) and std.mem.eql(u8, rep.streams, nvme_out.streams);
        }

        if (world == 1) {
            w1_eo = try gpa.dupe(u8, nvme_out.engram);
            try checkSmallAndMask(gpa, c, k, &work, &nvme_we, &small_sc, pids, in_tap.bytes, eo64, eo16_bytes, eo_theirs, layer, rows, failures);
        }
        nvme_we.deinit();

        // Resident mode: loads [share.lo .. share.lo + share.len] into RAM (`15.37 GiB`) with zero SSD I/O.
        var res_we = try WorldEngram.load(gpa, gpu.io, gpu.d, engram_dir, l, c, layer, world, .resident);
        defer res_we.deinit();
        var res_out = try work.forward(k, c, &res_we, &sc, pids.ids, pids.positions, false, true);
        defer res_out.deinit(gpa);

        var res_same = true;
        for (0..4) |_| {
            var rep = try work.forward(k, c, &res_we, &sc, pids.ids, pids.positions, false, false);
            defer rep.deinit(gpa);
            res_same = res_same and std.mem.eql(u8, rep.engram, res_out.engram) and std.mem.eql(u8, rep.streams, res_out.streams);
        }

        const raw_eq = std.mem.eql(u8, nvme_out.raw, res_out.raw);
        const out_eq = std.mem.eql(u8, nvme_out.engram, res_out.engram) and std.mem.eql(u8, nvme_out.streams, res_out.streams);
        verdict(failures, raw_eq and out_eq, "layer {d} t{d} w{d}: NVMe == resident byte-for-byte (raw {d} B, out {d} B)", .{ layer, rows, world, nvme_out.raw.len, nvme_out.engram.len });

        const mine = Gap.of(nvme_out.engram, eo64);
        var w1_max: f64 = 0;
        for (0..total) |i| w1_max = @max(w1_max, @abs(bf16At(nvme_out.engram, i) - bf16At(w1_eo.?, i)));
        verdict(failures, Gap.within(mine, eo_theirs), "layer {d} t{d} w{d} engram vs fp64: rel_l2 {e:.3} (torch16 {e:.3}), max {e:.3} (torch16 {e:.3}), mean {e:.3} (torch16 {e:.3}), max |.-w1| {e:.3}", .{ layer, rows, world, mine.rel_l2, eo_theirs.rel_l2, mine.max_abs, eo_theirs.max_abs, mine.mean_abs, eo_theirs.mean_abs, w1_max });

        if (ah64 != null and ah16_opt != null) {
            const st_mine = Gap.of(nvme_out.streams, ah64.?);
            const st_theirs = Gap.of(ah16_opt.?.bytes, ah64.?);
            var diff16: usize = 0;
            for (0..total) |i| diff16 += @intFromBool(!std.mem.eql(u8, nvme_out.streams[2 * i ..][0..2], ah16_opt.?.bytes[2 * i ..][0..2]));
            verdict(failures, Gap.within(st_mine, st_theirs), "layer {d} t{d} w{d} in+engram vs attn_hc_in64: rel_l2 {e:.3} (torch16 {e:.3}), max {e:.3} (torch16 {e:.3}), {d}/{d} bf16 != attn_hc_in16", .{ layer, rows, world, st_mine.rel_l2, st_theirs.rel_l2, st_mine.max_abs, st_theirs.max_abs, diff16, total });
        }

        verdict(failures, nvme_same and res_same, "layer {d} t{d} w{d}: five NVMe runs (incl. prefetch) and five resident runs byte-identical", .{ layer, rows, world });
    }
}

fn checkSmallAndMask(gpa: std.mem.Allocator, c: Config, k: *const ce.Kernels, work: *Work, we: *WorldEngram, small_sc: *ce.Scratch, pids: PromptIds, in_bytes: []const u8, eo64: []const f64, eo16_bytes: []const u8, full_theirs: Gap, layer: usize, rows: usize, failures: *usize) !void {
    const d = c.hidden;
    const stride = ce.streams * d;
    var small_ok = true;
    var r1: f64 = 0;
    var r16: f64 = 0;
    for ([_]usize{ 1, 16 }) |m| {
        var sub = try work.forward(k, c, we, small_sc, pids.ids[0..m], pids.positions[0..m], false, false);
        defer sub.deinit(gpa);
        const gm = Gap.of(sub.engram, eo64[0 .. m * stride]);
        const tm = Gap.of(eo16_bytes[0 .. m * stride * 2], eo64[0 .. m * stride]);
        if (m == 1) r1 = gm.rel_l2 else r16 = gm.rel_l2;
        small_ok = small_ok and Gap.withinSlice(gm, tm, full_theirs);
    }
    verdict(failures, small_ok, "layer {d} t{d} w1 small-batch decode vs fp64: m=1 rel_l2 {e:.3}, m=16 rel_l2 {e:.3}", .{ layer, rows, r1, r16 });

    // Mask check: pad (120002), image (120120), video (120124) zero engram_out and preserve streams_in bit-for-bit.
    const mtoks = [4]u32{ c.engram_pad, c.image_token, c.video_token, pids.ids[0] };
    const mpos = [4]u32{ 0, 1, 2, 0 };
    var mout = try work.forward(k, c, we, small_sc, &mtoks, &mpos, false, false);
    defer mout.deinit(gpa);
    var mask_ok = true;
    for (0..3 * stride) |i| {
        if (std.mem.readInt(u16, mout.engram[2 * i ..][0..2], .little) != 0) mask_ok = false;
    }
    if (!std.mem.eql(u8, mout.streams[0 .. 3 * stride * 2], in_bytes[0 .. 3 * stride * 2])) mask_ok = false;
    // Row 3 has position 0 and pids.ids[0] -> its looked-up hashes equal row 0 of the fixture!
    verdict(failures, mask_ok, "layer {d} t{d} w1 mask: pad/image/video zero engram_out and preserve streams bit-for-bit", .{ layer, rows });
}
