//! End-to-end GB10 verification of `cuda_head`, `cuda_cache`, and `cuda_layer` at world 1, 2 and 4 against fixtures.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4");
const check = @import("check.zig");

const Gpu = check.Gpu;
const st = core.safetensors;
const Config = hy4.Config;
const DeviceBuffer = cuda.DeviceBuffer;
const Stream = cuda.Stream;
const ch = hy4.cuda_head;
const cc = hy4.cuda_cache;
const cl = hy4.cuda_layer;

const recall_from = 2048;
const groups = 512;
const ratio = 4;
const chain_ids = [4]usize{ 2, 3, 4, 5 };

pub fn run(gpu: Gpu, cmd: []const u8, rest: []const [:0]const u8) !void {
    if (rest.len < 2) return error.MissingArgument;
    if (std.mem.eql(u8, cmd, "hy4-head")) return runHead(gpu, rest[0], rest[1..]);
    if (std.mem.eql(u8, cmd, "hy4-layers")) return runLayers(gpu, rest[0], rest[1..]);
    return error.UnknownCommand;
}

fn tap(f: *const st.File, name: []const u8) !st.Tensor {
    return f.get(name) orelse error.MissingTap;
}

fn bf16At(bytes: []const u8, i: usize) f64 {
    return @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)) << 16));
}

fn f32At(bytes: []const u8, i: usize) f64 {
    return @as(f32, @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little)));
}

fn f64At(bytes: []const u8, i: usize) f64 {
    return @bitCast(std.mem.readInt(u64, bytes[8 * i ..][0..8], .little));
}

fn wordDiffs(a: []const u8, b: []const u8, elem: usize) usize {
    var n: usize = 0;
    for (0..a.len / elem) |i| n += @intFromBool(!std.mem.eql(u8, a[elem * i ..][0..elem], b[elem * i ..][0..elem]));
    return n;
}

fn verdict(failures: *usize, ok: bool, comptime fmt: []const u8, args: anytype) void {
    if (!ok) failures.* += 1;
    std.debug.print("{s} " ++ fmt ++ "\n", .{if (ok) "PASS" else "FAIL"} ++ args);
}

fn syncDownload(s: Stream, buf: DeviceBuffer, dst: []u8) !void {
    try s.synchronize();
    try buf.download(0, dst);
}

const Stats = struct { max_abs: f64 = 0, mean_abs: f64 = 0, rel_l2: f64 = 0, cos: f64 = 1, bad: usize = 0 };

fn bf16StatsVs64(got: []const u8, want64: []const u8) Stats {
    const n = want64.len / 8;
    var s: Stats = .{};
    var sums = [4]f64{ 0, 0, 0, 0 }; // abs, err2, ref2, got2
    var dot: f64 = 0;
    for (0..n) |i| {
        const g = bf16At(got, i);
        const w = f64At(want64, i);
        if (!std.math.isFinite(g)) {
            s.bad += 1;
            continue;
        }
        const e = @abs(g - w);
        s.max_abs = @max(s.max_abs, e);
        sums[0] += e;
        sums[1] += e * e;
        sums[2] += w * w;
        sums[3] += g * g;
        dot += g * w;
    }
    if (n > 0) s.mean_abs = sums[0] / @as(f64, @floatFromInt(n));
    if (sums[2] > 0) s.rel_l2 = @sqrt(sums[1] / sums[2]);
    if (sums[2] > 0 and sums[3] > 0) s.cos = dot / (@sqrt(sums[2]) * @sqrt(sums[3]));
    return s;
}

const LogitCompare = struct { mine: Stats = .{}, theirs: Stats = .{}, mine_top1: usize = 0, theirs_top1: usize = 0, mine_gap: f64 = 0, theirs_gap: f64 = 0 };

fn compareLogits(got32: []const u8, torch32: []const u8, want64: []const u8, rows: usize, vocab: usize) LogitCompare {
    var out: LogitCompare = .{};
    var sums = [5]f64{ 0, 0, 0, 0, 0 }; // m_abs, t_abs, m_err2, t_err2, ref2
    for (0..rows) |r| {
        var best = [3]f64{ -std.math.inf(f64), -std.math.inf(f64), -std.math.inf(f64) };
        var args = [3]usize{ 0, 0, 0 }; // mine, torch, fp64
        for (0..vocab) |v| {
            const idx = r * vocab + v;
            const gm = f32At(got32, idx);
            const gt = f32At(torch32, idx);
            const gw = f64At(want64, idx);
            if (!std.math.isFinite(gm)) {
                out.mine.bad += 1;
                continue;
            }
            const em = @abs(gm - gw);
            const et = @abs(gt - gw);
            out.mine.max_abs = @max(out.mine.max_abs, em);
            out.theirs.max_abs = @max(out.theirs.max_abs, et);
            sums[0] += em;
            sums[1] += et;
            sums[2] += em * em;
            sums[3] += et * et;
            sums[4] += gw * gw;
            for ([3]f64{ gm, gt, gw }, 0..) |val, k| if (val > best[k]) {
                best[k] = val;
                args[k] = v;
            };
        }
        out.mine_top1 += @intFromBool(args[0] == args[2]);
        out.theirs_top1 += @intFromBool(args[1] == args[2]);
        out.mine_gap = @max(out.mine_gap, best[2] - f64At(want64, r * vocab + args[0]));
        out.theirs_gap = @max(out.theirs_gap, best[2] - f64At(want64, r * vocab + args[1]));
    }
    const denom = @as(f64, @floatFromInt(rows * vocab));
    out.mine.mean_abs = sums[0] / denom;
    out.theirs.mean_abs = sums[1] / denom;
    if (sums[4] > 0) {
        out.mine.rel_l2 = @sqrt(sums[2] / sums[4]);
        out.theirs.rel_l2 = @sqrt(sums[3] / sums[4]);
    }
    return out;
}

fn parseIdsJson(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]i32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
    defer gpa.free(text);
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const arr = parsed.value.object.get("ids").?.array.items;
    const out = try gpa.alloc(i32, arr.len);
    for (arr, out) |v, *o| o.* = @intCast(v.integer);
    return out;
}

fn openSibling(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, comptime fmt: []const u8, args: anytype) !st.File {
    var buf: [96]u8 = undefined;
    const p = try std.fs.path.join(gpa, &.{ dir, try std.fmt.bufPrint(&buf, fmt, args) });
    defer gpa.free(p);
    return st.File.open(gpa, io, p);
}

fn checkEmbed(gpu: Gpu, ops: ch.Ops, s: Stream, embed: *const ch.Embed, c: Config, head_fixture: []const u8, failures: *usize) !void {
    const gpa = gpu.gpa;
    const base = std.fs.path.basename(head_fixture);
    if (!std.mem.startsWith(u8, base, "head_") or !std.mem.endsWith(u8, base, ".safetensors")) return;
    const tag = base[5 .. base.len - ".safetensors".len];
    const dir = std.fs.path.dirname(head_fixture) orelse ".";
    var leaf_buf: [64]u8 = undefined;
    const ids_path = try std.fs.path.join(gpa, &.{ dir, try std.fmt.bufPrint(&leaf_buf, "ids_{s}.json", .{tag}) });
    defer gpa.free(ids_path);
    const ids = parseIdsJson(gpa, gpu.io, ids_path) catch |e| return if (std.mem.eql(u8, tag, "t256")) e else {};
    defer gpa.free(ids);
    var f0 = openSibling(gpa, gpu.io, dir, "layer0_{s}.safetensors", .{tag}) catch |e| return if (std.mem.eql(u8, tag, "t256")) e else {};
    defer f0.close(gpu.io);

    const rows = ids.len;
    const want = (try tap(&f0, "in")).bytes;
    var dev_ids = try DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ids));
    defer dev_ids.free();
    var dev_streams = try DeviceBuffer.alloc(gpu.d, want.len);
    defer dev_streams.free();
    var dev_one = try DeviceBuffer.alloc(gpu.d, rows * c.hidden * 2);
    defer dev_one.free();

    try dev_streams.fill8(0xff, s.handle);
    try embed.expand(ops, s, dev_ids.ptr, dev_streams.ptr, rows);
    try embed.lookup(ops, s, dev_ids.ptr, dev_one.ptr, rows);
    const host_mem = try gpa.alloc(u8, 2 * want.len + rows * c.hidden * 2);
    defer gpa.free(host_mem);
    const got_streams = host_mem[0..want.len];
    const again = host_mem[want.len .. 2 * want.len];
    const got_one = host_mem[2 * want.len ..];
    try syncDownload(s, dev_streams, got_streams);
    try dev_one.download(0, got_one);

    const diff = wordDiffs(got_streams, want, 2);
    var lookup_ok = true;
    for (0..rows) |r| for (0..4) |st_i| {
        lookup_ok = lookup_ok and std.mem.eql(u8, got_streams[(r * 4 + st_i) * c.hidden * 2 ..][0 .. c.hidden * 2], got_one[r * c.hidden * 2 ..][0 .. c.hidden * 2]);
    };

    var rep_ok = true;
    for (0..4) |_| {
        try dev_streams.fill8(0xff, s.handle);
        try embed.expand(ops, s, dev_ids.ptr, dev_streams.ptr, rows);
        try syncDownload(s, dev_streams, again);
        rep_ok = rep_ok and std.mem.eql(u8, again, got_streams);
    }
    const cut = rows / 2 + 5;
    try dev_streams.fill8(0xff, s.handle);
    try embed.expand(ops, s, dev_ids.ptr, dev_streams.ptr, cut);
    try embed.expand(ops, s, dev_ids.ptr + cut * 4, dev_streams.ptr + cut * 4 * c.hidden * 2, rows - cut);
    try syncDownload(s, dev_streams, again);
    verdict(failures, diff == 0 and lookup_ok and rep_ok and std.mem.eql(u8, again, got_streams), "hy4-head embed {s} ({d} tokens): {d} bits differ vs layer0.in, lookup==streams {}, 5x+cuts {}", .{ tag, rows, diff, lookup_ok, rep_ok and std.mem.eql(u8, again, got_streams) });
}

fn runHead(gpu: Gpu, model: []const u8, fixtures: []const [:0]const u8) !void {
    const gpa = gpu.gpa;
    const c = try Config.read(gpa, gpu.io, model);
    var ck = try core.Checkpoint.openModel(gpa, gpu.io, model);
    defer ck.close();
    const top_names = try hy4.names.top(&ck, c);
    var k = try ch.Kernels.load(gpu.d);
    defer k.deinit();
    var embed = try ch.Embed.load(gpu.d, top_names, c);
    defer embed.deinit();
    var s = try Stream.init(gpu.d, false);
    defer s.deinit();

    var failures: usize = 0;
    for (fixtures) |fixture| {
        try checkEmbed(gpu, k.ops(), s, &embed, c, fixture, &failures);
        try headOne(gpu, k.ops(), s, top_names, c, fixture, &failures);
    }
    if (failures > 0) return error.Mismatch;
}

fn headOne(gpu: Gpu, ops: ch.Ops, s: Stream, top_names: hy4.names.Top, c: Config, path: []const u8, failures: *usize) !void {
    const gpa = gpu.gpa;
    const base = std.fs.path.basename(path);
    var f = try st.File.open(gpa, gpu.io, path);
    defer f.close(gpu.io);
    const in_t = try tap(&f, "in");
    const rows = in_t.dim(1);

    var dev_in = try DeviceBuffer.fromHost(gpu.d, in_t.bytes);
    defer dev_in.free();
    var sc = try ch.Scratch.init(gpu.d, c, rows);
    defer sc.deinit();
    var dev_h = try DeviceBuffer.alloc(gpu.d, rows * c.hidden * 2);
    defer dev_h.free();
    var dev_l = try DeviceBuffer.alloc(gpu.d, rows * c.vocab * 4);
    defer dev_l.free();

    const h_bytes = rows * c.hidden * 2;
    const l_bytes = rows * c.vocab * 4;
    const host_mem = try gpa.alloc(u8, 2 * h_bytes + 3 * l_bytes);
    defer gpa.free(host_mem);
    const w1_h = host_mem[0..h_bytes];
    const host_h = host_mem[h_bytes .. 2 * h_bytes];
    const w1_l = host_mem[2 * h_bytes ..][0..l_bytes];
    const rep_l = host_mem[2 * h_bytes + l_bytes ..][0..l_bytes];
    const host_shard_all = host_mem[2 * h_bytes + 2 * l_bytes ..][0..l_bytes];

    {
        var top1 = try ch.Top.load(gpu.d, top_names, c, .{ .rank = 0, .world = 1 });
        defer top1.deinit();
        try dev_h.fill8(0xff, s.handle);
        try dev_l.fill8(0xff, s.handle);
        try top1.forward(ops, s, &sc, dev_in.ptr, dev_h.ptr, dev_l.ptr, c.vocab, rows);
        try syncDownload(s, dev_h, w1_h);
        try dev_l.download(0, w1_l);

        const hm = bf16StatsVs64(w1_h, (try tap(&f, "hidden64")).bytes);
        const ht = bf16StatsVs64((try tap(&f, "hidden16")).bytes, (try tap(&f, "hidden64")).bytes);
        const lc = compareLogits(w1_l, (try tap(&f, "logits16")).bytes, (try tap(&f, "logits64")).bytes, rows, c.vocab);
        const h_ok = hm.bad == 0 and hm.mean_abs <= ht.mean_abs and hm.rel_l2 <= ht.rel_l2 and hm.max_abs <= ht.max_abs;
        const l_ok = lc.mine.bad == 0 and lc.mine.mean_abs < lc.theirs.mean_abs and lc.mine.rel_l2 < lc.theirs.rel_l2 and lc.mine.max_abs < lc.theirs.max_abs and lc.mine_gap <= lc.mine.max_abs and lc.mine_top1 * 100 >= rows * 98;
        verdict(failures, h_ok, "{s} w1 hidden vs fp64: mean {e:.3} (torch {e:.3}), rel_l2 {e:.3} (torch {e:.3}), max {e:.3} (torch {e:.3})", .{ base, hm.mean_abs, ht.mean_abs, hm.rel_l2, ht.rel_l2, hm.max_abs, ht.max_abs });
        verdict(failures, l_ok, "{s} w1 logits vs fp64: mean {e:.3} < torch {e:.3}, rel_l2 {e:.3} < torch {e:.3}, max {e:.3} < torch {e:.3}, top1 {d}/{d} (torch {d}/{d}, max tie gap {e:.3} vs torch {e:.3})", .{ base, lc.mine.mean_abs, lc.theirs.mean_abs, lc.mine.rel_l2, lc.theirs.rel_l2, lc.mine.max_abs, lc.theirs.max_abs, lc.mine_top1, rows, lc.theirs_top1, rows, lc.mine_gap, lc.theirs_gap });

        var same = true;
        for (0..4) |_| {
            try dev_l.fill8(0xff, s.handle);
            try top1.forward(ops, s, &sc, dev_in.ptr, dev_h.ptr, dev_l.ptr, c.vocab, rows);
            try syncDownload(s, dev_l, rep_l);
            same = same and std.mem.eql(u8, rep_l, w1_l);
        }
        const cut = rows / 2 + 3;
        try dev_l.fill8(0xff, s.handle);
        for ([_][2]usize{ .{ 0, cut }, .{ cut, rows - 1 - cut }, .{ rows - 1, 1 } }) |p| {
            try top1.forward(ops, s, &sc, dev_in.ptr + p[0] * 4 * c.hidden * 2, dev_h.ptr + p[0] * c.hidden * 2, dev_l.ptr + p[0] * c.vocab * 4, c.vocab, p[1]);
        }
        try syncDownload(s, dev_l, rep_l);
        verdict(failures, same and std.mem.eql(u8, rep_l, w1_l), "{s} w1: five runs and chunk/decode cuts byte-identical", .{base});
    }

    for ([_]usize{ 2, 4 }) |world| {
        var tops: [4]ch.Top = undefined;
        var loaded: usize = 0;
        defer for (tops[0..loaded]) |*t| t.deinit();
        for (0..world) |rk| {
            tops[rk] = try ch.Top.load(gpu.d, top_names, c, .{ .rank = rk, .world = world });
            loaded += 1;
        }
        const v_loc = tops[0].share.len;
        var dev_shard = try DeviceBuffer.alloc(gpu.d, rows * v_loc * 4);
        defer dev_shard.free();
        const host_shard = host_shard_all[0 .. rows * v_loc * 4];

        var h_same = true;
        var repeats_ok = true;
        for (0..5) |_| {
            @memset(rep_l, 0xff);
            for (0..world) |rk| {
                const sh = tops[rk].share;
                try dev_h.fill8(0xff, s.handle);
                try dev_shard.fill8(0xff, s.handle);
                try tops[rk].forward(ops, s, &sc, dev_in.ptr, dev_h.ptr, dev_shard.ptr, sh.len, rows);
                try syncDownload(s, dev_h, host_h);
                try dev_shard.download(0, host_shard);
                h_same = h_same and std.mem.eql(u8, host_h, w1_h);
                for (0..rows) |r| @memcpy(rep_l[(r * c.vocab + sh.lo) * 4 ..][0 .. sh.len * 4], host_shard[r * sh.len * 4 ..][0 .. sh.len * 4]);
            }
            repeats_ok = repeats_ok and std.mem.eql(u8, rep_l, w1_l);
        }
        const diff = wordDiffs(rep_l, w1_l, 4);
        verdict(failures, diff == 0 and h_same and repeats_ok, "{s} w{d} ({d} vocab/rank): {d} logit words differ vs w1, hidden==w1 {}, 5 runs identical {}", .{ base, world, v_loc, diff, h_same, repeats_ok });
    }
}

const FixtureRecall = struct {
    dev_idx: DeviceBuffer,

    fn init(gpu: Gpu, recall16: []const u8, total_rows: usize) !FixtureRecall {
        const n = total_rows - recall_from;
        const expanded = try gpu.gpa.alloc(i32, n * groups * ratio);
        defer gpu.gpa.free(expanded);
        for (0..n) |r| for (0..groups) |g| {
            const gi = std.mem.readInt(i32, recall16[((recall_from + r) * groups + g) * 4 ..][0..4], .little);
            for (0..ratio) |o| expanded[(r * groups + g) * ratio + o] = gi * ratio + @as(i32, @intCast(o));
        };
        return .{ .dev_idx = try DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(expanded)) };
    }

    fn hook(fr: *FixtureRecall) cl.IndexerHook {
        return .{ .ctx = fr, .run = onRun };
    }

    fn onRun(ctx: *anyopaque, call: cl.IndexerCall) anyerror!?cl.Recall {
        const fr: *FixtureRecall = @ptrCast(@alignCast(ctx));
        if (call.first + call.rows <= call.recall_limit) return null;
        if (call.q_resid == 0) return error.MissingQResid;
        const from = @max(call.first, call.recall_limit);
        return .{ .list_from = from - call.first, .idx = fr.dev_idx.ptr + (from - recall_from) * groups * ratio * 4, .nidx = groups * ratio };
    }

    fn deinit(fr: *FixtureRecall) void {
        fr.dev_idx.free();
    }
};

const FixtureEngram = struct {
    gpu: Gpu,
    in_bytes: []const u8,
    engram_bytes: []const u8,
    row_bytes: usize,
    called: usize = 0,

    fn hook(fe: *FixtureEngram) cl.EngramHook {
        return .{ .ctx = fe, .run = onRun };
    }

    fn onRun(ctx: *anyopaque, call: cl.EngramCall) anyerror!void {
        const fe: *FixtureEngram = @ptrCast(@alignCast(ctx));
        fe.called += 1;
        const total = call.rows * fe.row_bytes;
        const off = call.first * fe.row_bytes;
        const sum = try fe.gpu.gpa.alloc(u16, total / 2);
        defer fe.gpu.gpa.free(sum);
        for (sum, 0..) |*o, i| o.* = core.bf16_round.bits(@as(f32, @floatCast(bf16At(fe.in_bytes[off..], i))) + @as(f32, @floatCast(bf16At(fe.engram_bytes[off..], i))));
        const dst: DeviceBuffer = .{ .d = fe.gpu.d, .ptr = call.streams_out, .len = total };
        try dst.upload(0, std.mem.sliceAsBytes(sum));
    }
};

fn layerOf(path: []const u8) !usize {
    const base = std.fs.path.basename(path);
    if (!std.mem.startsWith(u8, base, "layer")) return error.NotALayerFixture;
    const end = std.mem.indexOfScalar(u8, base, '_') orelse return error.NotALayerFixture;
    return std.fmt.parseInt(usize, base[5..end], 10);
}

fn makePositions(gpu: Gpu, n: usize) !DeviceBuffer {
    const pos = try gpu.gpa.alloc(i32, n);
    defer gpu.gpa.free(pos);
    for (pos, 0..) |*p, i| p.* = @intCast(i);
    return DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(pos));
}

fn runLayers(gpu: Gpu, model: []const u8, fixtures: []const [:0]const u8) !void {
    const gpa = gpu.gpa;
    const c = try Config.read(gpa, gpu.io, model);
    var ck = try core.Checkpoint.openModel(gpa, gpu.io, model);
    defer ck.close();
    var k = try cl.Kernels.load(gpu.ctx);
    defer k.deinit();
    var s = try Stream.init(gpu.d, false);
    defer s.deinit();

    var failures: usize = 0;
    for (fixtures) |fixture| try layerOne(gpu, &k, s, &ck, c, fixture, &failures);
    if (fixtures.len > 0) try chainLayers(gpu, &k, s, &ck, c, std.fs.path.dirname(fixtures[0]) orelse ".", &failures);
    if (failures > 0) return error.Mismatch;
}

fn execRankOrWorld(any_layer: anytype, k: *const cl.Kernels, s: Stream, c: Config, sc: *const cl.Scratch, cache: *const cc.LayerCache, dev_out: DeviceBuffer, call: cl.Call, dst: []u8) !void {
    try cache.fill(s, 0xff);
    try dev_out.fill8(0xff, s.handle);
    try any_layer.forward(k, s, c, sc, call);
    try syncDownload(s, dev_out, dst);
}

fn layerOne(gpu: Gpu, k: *const cl.Kernels, s: Stream, ck: *core.Checkpoint, c: Config, path: []const u8, failures: *usize) !void {
    const gpa = gpu.gpa;
    const base = std.fs.path.basename(path);
    const layer_idx = try layerOf(path);
    var f = try st.File.open(gpa, gpu.io, path);
    defer f.close(gpu.io);
    const lt = try hy4.names.layer(ck, c, layer_idx);

    const has_engram = c.engramSlot(layer_idx) != null;
    const in_tap = try tap(&f, if (has_engram) "attn_hc_in16" else "in");
    const out16 = try tap(&f, "out16");
    const out64 = try tap(&f, "out64");
    const rows = in_tap.dim(1);
    const out_bytes = rows * 4 * c.hidden * 2;

    var dev_in = try DeviceBuffer.fromHost(gpu.d, in_tap.bytes);
    defer dev_in.free();
    var dev_out = try DeviceBuffer.alloc(gpu.d, out_bytes);
    defer dev_out.free();
    var dev_pos = try makePositions(gpu, rows);
    defer dev_pos.free();

    var fr: ?FixtureRecall = if (c.kind[layer_idx] == .indexed and rows > recall_from) try FixtureRecall.init(gpu, (try tap(&f, "recall16")).bytes, rows) else null;
    defer if (fr) |*r| r.deinit();
    const idx_hook: ?cl.IndexerHook = if (fr) |*r| r.hook() else null;

    const theirs = bf16StatsVs64(out16.bytes, out64.bytes);
    const host_mem = try gpa.alloc(u8, 2 * out_bytes);
    defer gpa.free(host_mem);
    const got = host_mem[0..out_bytes];
    const rep = host_mem[out_bytes..];
    std.debug.print("INFO {s}: layer {d} ({s}, {s}, engram={}), {d} rows; torch vs fp64 rel_l2 {e:.3} mean {e:.3} max {e:.3}\n", .{ base, layer_idx, if (c.kind[layer_idx] == .indexed) "indexed" else "sliding", if (c.dense[layer_idx]) "dense" else "moe", has_engram, rows, theirs.rel_l2, theirs.mean_abs, theirs.max_abs });

    for ([_]usize{ 1, 2, 4 }) |world| {
        var wl = try cl.WorldLayer.load(gpa, gpu.d, lt, c, layer_idx, world);
        defer wl.deinit();
        var sc = try cl.Scratch.init(gpa, gpu.d, &k.mlp.lane, c, world, rows);
        defer sc.deinit();
        var cache = try cc.LayerCache.init(gpu.d, c, layer_idx, rows, .linear);
        defer cache.deinit();

        const call: cl.Call = .{ .streams_in = dev_in.ptr, .streams_out = dev_out.ptr, .rows = rows, .first = 0, .pos = dev_pos.ptr, .cache = cache.view, .indexer = idx_hook };
        try execRankOrWorld(&wl, k, s, c, &sc, &cache, dev_out, call, got);

        const mine = bf16StatsVs64(got, out64.bytes);
        const acc_ok = mine.bad == 0 and mine.rel_l2 <= @max(theirs.rel_l2 * 1.15, theirs.rel_l2 + 1e-3) and
            mine.mean_abs <= @max(theirs.mean_abs * 1.15, theirs.mean_abs + 5e-4) and
            mine.max_abs <= @max(theirs.max_abs * 1.35, theirs.max_abs + 2e-2);

        var same = true;
        for (0..4) |_| {
            try execRankOrWorld(&wl, k, s, c, &sc, &cache, dev_out, call, rep);
            same = same and std.mem.eql(u8, rep, got);
        }

        var single_ok = true;
        if (world == 1) {
            try execRankOrWorld(&wl.ranks[0], k, s, c, &sc, &cache, dev_out, call, rep);
            single_ok = std.mem.eql(u8, rep, got);
            if (has_engram) {
                const raw_in = (try tap(&f, "in")).bytes;
                var dev_raw = try DeviceBuffer.fromHost(gpu.d, raw_in);
                defer dev_raw.free();
                var fe: FixtureEngram = .{ .gpu = gpu, .in_bytes = raw_in, .engram_bytes = (try tap(&f, "engram_out16")).bytes, .row_bytes = 4 * c.hidden * 2 };
                var eng_call = call;
                eng_call.streams_in = dev_raw.ptr;
                eng_call.engram = fe.hook();
                try execRankOrWorld(&wl.ranks[0], k, s, c, &sc, &cache, dev_out, eng_call, rep);
                single_ok = single_ok and fe.called == 1 and std.mem.eql(u8, rep, got);
            }
        } else single_ok = if (wl.ranks[0].forward(k, s, c, &sc, call)) false else |e| e == error.MissingAllReduce;
        verdict(failures, acc_ok and same and single_ok, "{s} w{d} vs fp64: rel_l2 {e:.3} (torch {e:.3}), mean {e:.3} (torch {e:.3}), max {e:.3} (torch {e:.3}), cos {d:.6}, 5x+seams {}", .{ base, world, mine.rel_l2, theirs.rel_l2, mine.mean_abs, theirs.mean_abs, mine.max_abs, theirs.max_abs, mine.cos, same and single_ok });
    }
}

fn runChainSchedule(k: *const cl.Kernels, s: Stream, c: Config, wls: *const [4]cl.WorldLayer, sc: *const cl.Scratch, cache: *const cc.SequenceCache, dev_in: u64, dev_pos: u64, dev_work: u64, dev_out: u64, spans: []const [2]usize, dst_host: []u8) !void {
    const row_bytes = 4 * c.hidden * 2;
    try cache.fill(s, 0xff);
    for (spans) |sp| for (chain_ids, 0..) |lid, ci| {
        const src = if (ci == 0) dev_in + sp[0] * row_bytes else dev_work;
        const dst = if (ci == 3) dev_out + sp[0] * row_bytes else dev_work;
        try wls[ci].forward(k, s, c, sc, .{ .streams_in = src, .streams_out = dst, .rows = sp[1], .first = sp[0], .pos = dev_pos + sp[0] * 4, .cache = try cache.view(lid) });
    };
    const out_view: DeviceBuffer = .{ .d = cache.buf.d, .ptr = dev_out, .len = dst_host.len };
    try syncDownload(s, out_view, dst_host);
}

fn chainLayers(gpu: Gpu, k: *const cl.Kernels, s: Stream, ck: *core.Checkpoint, c: Config, dir: []const u8, failures: *usize) !void {
    const gpa = gpu.gpa;
    var f2 = openSibling(gpa, gpu.io, dir, "layer2_t256.safetensors", .{}) catch return;
    defer f2.close(gpu.io);
    var f5 = try openSibling(gpa, gpu.io, dir, "layer5_t256.safetensors", .{});
    defer f5.close(gpu.io);
    var f2_long: ?st.File = openSibling(gpa, gpu.io, dir, "layer2_t2560.safetensors", .{}) catch null;
    defer if (f2_long) |*fl| fl.close(gpu.io);

    var lts: [4]hy4.names.Layer = undefined;
    for (chain_ids, 0..) |lid, i| lts[i] = try hy4.names.layer(ck, c, lid);

    const rows256: usize = 256;
    const rows_wrap: usize = 640;
    const max_rows: usize = if (f2_long != null) rows_wrap else rows256;
    const row_bytes = 4 * c.hidden * 2;

    var dev_in256 = try DeviceBuffer.fromHost(gpu.d, (try tap(&f2, "in")).bytes);
    defer dev_in256.free();
    var dev_pos = try makePositions(gpu, max_rows);
    defer dev_pos.free();
    var dev_work = try DeviceBuffer.alloc(gpu.d, max_rows * row_bytes);
    defer dev_work.free();
    var dev_out = try DeviceBuffer.alloc(gpu.d, max_rows * row_bytes);
    defer dev_out.free();
    const host_buf = try gpa.alloc(u8, 2 * max_rows * row_bytes);
    defer gpa.free(host_buf);
    const full_out = host_buf[0 .. rows256 * row_bytes];
    const alt_out = host_buf[max_rows * row_bytes ..][0 .. rows256 * row_bytes];

    for ([_]usize{ 1, 2, 4 }) |world| {
        var wls: [4]cl.WorldLayer = undefined;
        var loaded: usize = 0;
        defer for (wls[0..loaded]) |*wl| wl.deinit();
        for (chain_ids, 0..) |lid, i| {
            wls[i] = try cl.WorldLayer.load(gpa, gpu.d, lts[i], c, lid, world);
            loaded += 1;
        }
        var sc = try cl.Scratch.init(gpa, gpu.d, &k.mlp.lane, c, world, max_rows);
        defer sc.deinit();
        var lin_cache = try cc.SequenceCache.initLayers(gpu.d, c, &chain_ids, rows256, .linear);
        defer lin_cache.deinit();
        var ring_cache = try cc.SequenceCache.initLayers(gpu.d, c, &chain_ids, rows256, .{ .ring = c.window });
        defer ring_cache.deinit();

        try runChainSchedule(k, s, c, &wls, &sc, &lin_cache, dev_in256.ptr, dev_pos.ptr, dev_work.ptr, dev_out.ptr, &.{.{ 0, 256 }}, full_out);
        const st256 = bf16StatsVs64(full_out, (try tap(&f5, "out64")).bytes);

        var rep_ok = true;
        for (0..4) |_| {
            try runChainSchedule(k, s, c, &wls, &sc, &lin_cache, dev_in256.ptr, dev_pos.ptr, dev_work.ptr, dev_out.ptr, &.{.{ 0, 256 }}, alt_out);
            rep_ok = rep_ok and std.mem.eql(u8, alt_out, full_out);
        }

        const chunk_spans = [_][2]usize{ .{ 0, 192 }, .{ 192, 64 } };
        const dec_spans = [_][2]usize{ .{ 0, 192 }, .{ 192, 60 }, .{ 252, 1 }, .{ 253, 1 }, .{ 254, 1 }, .{ 255, 1 } };
        var sched_ok = true;
        for ([_]*const cc.SequenceCache{ &lin_cache, &ring_cache }) |cache_ptr| {
            for ([_][]const [2]usize{ &chunk_spans, &dec_spans }) |spans| {
                try runChainSchedule(k, s, c, &wls, &sc, cache_ptr, dev_in256.ptr, dev_pos.ptr, dev_work.ptr, dev_out.ptr, spans, alt_out);
                sched_ok = sched_ok and std.mem.eql(u8, alt_out, full_out);
            }
        }

        var wrap_ok = true;
        if (f2_long) |*fl| {
            var dev_in_long = try DeviceBuffer.fromHost(gpu.d, (try tap(fl, "in")).bytes[0 .. rows_wrap * row_bytes]);
            defer dev_in_long.free();
            var lin640 = try cc.SequenceCache.initLayers(gpu.d, c, &chain_ids, rows_wrap, .linear);
            defer lin640.deinit();
            var ring512 = try cc.SequenceCache.initLayers(gpu.d, c, &chain_ids, 520, .{ .ring = c.window });
            defer ring512.deinit();
            var ring576 = try cc.SequenceCache.initLayers(gpu.d, c, &chain_ids, rows_wrap, .{ .ring = cc.minRingSlots(c.window, 64) + 1 });
            defer ring576.deinit();
            const ref640 = host_buf[0 .. rows_wrap * row_bytes];
            const got640 = host_buf[max_rows * row_bytes ..][0 .. rows_wrap * row_bytes];
            try runChainSchedule(k, s, c, &wls, &sc, &lin640, dev_in_long.ptr, dev_pos.ptr, dev_work.ptr, dev_out.ptr, &.{.{ 0, rows_wrap }}, ref640);
            var chunks64: [10][2]usize = undefined;
            for (&chunks64, 0..) |*sp, i| sp.* = .{ i * 64, 64 };
            try runChainSchedule(k, s, c, &wls, &sc, &ring576, dev_in_long.ptr, dev_pos.ptr, dev_work.ptr, dev_out.ptr, &chunks64, got640);
            wrap_ok = wrap_ok and std.mem.eql(u8, got640, ref640);
            var dec520: [13][2]usize = undefined;
            dec520[0] = .{ 0, 508 };
            for (0..12) |i| dec520[1 + i] = .{ 508 + i, 1 };
            try runChainSchedule(k, s, c, &wls, &sc, &ring512, dev_in_long.ptr, dev_pos.ptr, dev_work.ptr, dev_out.ptr, &dec520, got640[0 .. 520 * row_bytes]);
            wrap_ok = wrap_ok and std.mem.eql(u8, got640[0 .. 520 * row_bytes], ref640[0 .. 520 * row_bytes]);
        }

        verdict(failures, st256.bad == 0 and st256.cos > 0.999 and rep_ok and sched_ok and wrap_ok, "chain layers 2->3->4->5 w{d}: vs layer5.out64 rel_l2 {e:.3} cos {d:.6}; 5x {}, 192+64 & decode & ring==linear {}, wrap {}", .{ world, st256.rel_l2, st256.cos, rep_ok, sched_ok, wrap_ok });
    }
}
