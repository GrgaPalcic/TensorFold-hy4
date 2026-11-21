//! Unit tests for Hy4 Engram FP8 RNE conversion, `.hy4e` packer, and NVMe vs resident parity at world 1/2/4.
const std = @import("std");
const testing = std.testing;
const er = @import("engram_rows");
const fp8 = @import("core").quant.fp8;
const config = @import("config.zig");
const engram_hash = @import("engram_hash.zig");
const partition = @import("partition.zig");
const ep = @import("engram_pack.zig");
const reader = @import("engram_reader.zig");

const release_cfg = @embedFile("fixtures/config.json");

fn refE4m3Rne(x: f32) u8 {
    if (std.math.isNan(x)) return if (@as(u32, @bitCast(x)) >> 31 != 0) 0xff else 0x7f;
    const sign: u8 = if (@as(u32, @bitCast(x)) >> 31 != 0) 0x80 else 0x00;
    const a: f64 = @abs(@as(f64, x));
    if (a > 464.0) return sign | 0x7f;
    if (a == 0.0) return sign;
    var best_code: u8 = 0;
    var best_dist: f64 = a;
    for (1..0x7f) |code| {
        const c: u8 = @intCast(code);
        const v = @abs(fp8.e4m3(c));
        const d = @abs(a - v);
        if (d < best_dist or (d == best_dist and (c & 1) == 0)) {
            best_dist = d;
            best_code = c;
        }
    }
    return sign | best_code;
}

test "e4m3Rne matches exact nearest-even e4m3fn on all 65536 bf16 values, subnormals, ties, +-0 and +-448" {
    try testing.expectEqual(@as(u8, 0x00), ep.e4m3Rne(0.0));
    try testing.expectEqual(@as(u8, 0x80), ep.e4m3Rne(-0.0));
    try testing.expectEqual(@as(u8, 0x7e), ep.e4m3Rne(448.0));
    try testing.expectEqual(@as(u8, 0xfe), ep.e4m3Rne(-448.0));
    try testing.expectEqual(@as(u8, 0x7e), ep.e4m3Rne(464.0));
    try testing.expectEqual(@as(u8, 0x7f), ep.e4m3Rne(464.0001));
    try testing.expectEqual(@as(u8, 0x7f), ep.e4m3Rne(std.math.inf(f32)));
    try testing.expectEqual(@as(u8, 0xff), ep.e4m3Rne(-std.math.inf(f32)));
    try testing.expectEqual(@as(u8, 0x7f), ep.e4m3Rne(std.math.nan(f32)));
    // Subnormal midpoints: 0.5*2^-9 -> 0, 1.5*2^-9 -> 2, 2.5*2^-9 -> 2, 7.5*2^-9 -> 8 (min normal 2^-6).
    try testing.expectEqual(@as(u8, 0x00), ep.e4m3Rne(0.5 * 0x1.0p-9));
    try testing.expectEqual(@as(u8, 0x01), ep.e4m3Rne(0.5 * 0x1.0p-9 + 0x1.0p-20));
    try testing.expectEqual(@as(u8, 0x02), ep.e4m3Rne(1.5 * 0x1.0p-9));
    try testing.expectEqual(@as(u8, 0x02), ep.e4m3Rne(2.5 * 0x1.0p-9));
    try testing.expectEqual(@as(u8, 0x08), ep.e4m3Rne(7.5 * 0x1.0p-9));
    // Normal midpoints: 1.0625 -> 1.0 (0x38, even), 1.1875 -> 1.25 (0x3a, even).
    try testing.expectEqual(@as(u8, 0x38), ep.e4m3Rne(1.0625));
    try testing.expectEqual(@as(u8, 0x3a), ep.e4m3Rne(1.1875));

    for (0..65536) |u| {
        const x: f32 = @bitCast(@as(u32, @intCast(u)) << 16);
        try testing.expectEqual(refE4m3Rne(x), ep.e4m3Rne(x));
    }
    for ([_]u32{ 0, 0x8000_0000 }) |sgn| {
        for (115..137) |exp| for (0..16) |m8| {
            const base: u32 = sgn | (@as(u32, @intCast(exp)) << 23) | (@as(u32, @intCast(m8)) << 19);
            for ([_]i32{ -2, -1, 0, 1, 2 }) |delta| {
                const bits: u32 = @bitCast(@as(i32, @bitCast(base)) +% delta);
                const x: f32 = @bitCast(bits);
                try testing.expectEqual(refE4m3Rne(x), ep.e4m3Rne(x));
            }
        };
    }
}

test "packRow fp64 arithmetic picks exact near-tie code and round-trips within fp32 scale rounding" {
    var raw: [ep.dim * 2]u8 = @splat(0);
    var rec: [ep.row_bytes]u8 = undefined;
    // All-zero row clamps amax to 1e-12 and encodes +0 codes.
    ep.packRow(&raw, &rec);
    for (rec[0..ep.dim]) |b| try testing.expectEqual(@as(u8, 0), b);
    const zscale: f32 = @bitCast(std.mem.readInt(u32, rec[ep.dim..][0..4], .little));
    try testing.expectEqual(@as(f32, @floatCast(1e-12 / 448.0)), zscale);

    var prng = std.Random.DefaultPrng.init(2026);
    const rng = prng.random();
    for (0..256) |_| {
        var amax: f64 = 1e-12;
        var x64: [ep.dim]f64 = undefined;
        for (0..ep.dim) |j| {
            const u: u16 = @intCast(rng.uintLessThan(u32, 0x7f00) | (rng.uintLessThan(u32, 2) << 15));
            std.mem.writeInt(u16, raw[2 * j ..][0..2], u, .little);
            const v: f64 = @as(f32, @bitCast(@as(u32, u) << 16));
            x64[j] = v;
            amax = @max(amax, @abs(v));
        }
        ep.packRow(&raw, &rec);
        var vals: [ep.dim]f32 = undefined;
        ep.unpackRowF32(&rec, &vals);
        const s64 = amax / 448.0;
        for (0..ep.dim) |j| {
            const q64 = fp8.e4m3(rec[j]) * s64;
            try testing.expect(@abs(@as(f64, vals[j]) - q64) / amax <= 1e-6);
        }
    }
}

test "multithreaded packLayer + Reader: NVMe and resident modes match byte-for-byte across world 1, 2 and 4" {
    const a = testing.allocator;
    const io = testing.io;
    var why: config.Why = .{};
    var c = try config.parse(a, release_cfg, &why);
    // Test hashRank on the release config, including document restart padding and pad/image/video masks.
    var map = try a.alloc(u8, engram_hash.map_size * 4);
    defer a.free(map);
    for (0..engram_hash.map_size) |i| {
        const v: i32 = @intCast(if (i == engram_hash.map_size - 1) engram_hash.compressed - 1 else i % (engram_hash.compressed - 1));
        std.mem.writeInt(i32, map[4 * i ..][0..4], v, .little);
    }
    try engram_hash.checkMap(map, &why);
    const toks = [_]u32{ 120000, 42, 120002, 120120, 120124, 99, 100 };
    const pos = [_]u32{ 0, 1, 2, 3, 4, 0, 1 };
    var w1_idx: [toks.len * 16]i64 = undefined;
    var mask: [toks.len]u8 = undefined;
    try reader.hashRank(c, 0, try partition.Engram.of(c, 0, 1), map, &toks, &pos, &w1_idx, &mask);
    try testing.expectEqualSlices(u8, &.{ 1, 1, 0, 0, 0, 1, 1 }, &mask);
    // Document restart at index 5 (pos[5] == 0) pads lookback, never crossing into index 4.
    const solo = engram_hash.rows(&engram_hash.layers[0], map, toks[5..], pos[5..], 0, c.engram_pad);
    for (0..16) |h| try testing.expectEqual(@as(i64, solo[h]), w1_idx[5 * 16 + h]);

    // Verify TP=2 and TP=4 hashRank slices partition w1_idx identically.
    for ([_]usize{ 2, 4 }) |world| {
        for (0..world) |rk| {
            const p = try partition.Engram.of(c, rk, world);
            const sub = try a.alloc(i64, toks.len * p.heads.len);
            defer a.free(sub);
            var m2: [toks.len]u8 = undefined;
            try reader.hashRank(c, 0, p, map, &toks, &pos, sub, &m2);
            try testing.expectEqualSlices(u8, &mask, &m2);
            for (0..toks.len) |i| {
                try testing.expectEqualSlices(i64, w1_idx[i * 16 + p.heads.lo ..][0..p.heads.len], sub[i * p.heads.len ..][0..p.heads.len]);
            }
        }
    }

    // Synthetic shard -> Header + packRow -> NVMe vs resident byte-for-byte at world 1, 2 and 4.
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const small_sizes = [16]u64{ 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607 };
    c.engram_sizes[0] = small_sizes;
    const h = ep.Header.of(c, 0);
    const total: usize = @intCast(h.rows);
    const bf_in = try a.alloc(u8, total * ep.dim * 2);
    defer a.free(bf_in);
    var prng = std.Random.DefaultPrng.init(77);
    for (0..total * ep.dim) |i| {
        const u: u16 = @intCast(prng.random().uintLessThan(u32, 0x4200));
        std.mem.writeInt(u16, bf_in[2 * i ..][0..2], u, .little);
    }
    const file_bytes = try a.alloc(u8, @intCast(ep.header_bytes + h.rows * ep.row_bytes));
    defer a.free(file_bytes);
    const hbytes = h.encode();
    @memcpy(file_bytes[0..ep.header_bytes], &hbytes);
    for (0..total) |r| ep.packRow(bf_in[r * ep.dim * 2 ..][0 .. ep.dim * 2], file_bytes[ep.header_bytes + r * ep.row_bytes ..][0..ep.row_bytes]);
    try tmp.dir.writeFile(io, .{ .sub_path = "engram_layer1.hy4e", .data = file_bytes });
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    const ppath = try std.fs.path.join(a, &.{ root, "engram_layer1.hy4e" });
    defer a.free(ppath);
    const parsed_h = try ep.Header.parse(file_bytes[0..ep.header_bytes], file_bytes.len);
    try parsed_h.checkConfig(c, 0, &why);

    for ([_]usize{ 1, 2, 4 }) |world| {
        for (0..world) |rk| {
            const p = try partition.Engram.of(c, rk, world);
            const sh = p.rows(c, 0);
            const spec: er.OpenSpec = .{
                .path = ppath,
                .h = .{ .layer = 1, .lo = sh.lo, .hi = sh.lo + sh.len, .total = h.rows, .row_bytes = ep.row_bytes },
                .file_lo = 0,
                .file_hi = h.rows,
                .align_bytes = er.page,
            };
            const nvme = try er.Rows.openSpecs(a, &.{spec}, 4096);
            defer nvme.close();
            var res = reader.Reader{
                .gpa = a,
                .c = c,
                .at = .{ .rank = rk, .world = world },
                .part = p,
                .mode = .resident,
                .slots = try a.dupe(usize, &.{0}),
                .shares = try a.dupe(partition.Share, &.{sh}),
                .resident = blk: {
                    const arr = try a.alloc([]u8, 1);
                    arr[0] = try a.dupe(u8, file_bytes[ep.header_bytes + sh.lo * ep.row_bytes ..][0 .. sh.len * ep.row_bytes]);
                    break :blk arr;
                },
            };
            defer res.close();

            var q: [64]i64 = undefined;
            q[0] = @intCast(sh.lo);
            q[1] = @intCast(sh.lo + sh.len - 1);
            for (q[2..]) |*id| id.* = @intCast(sh.lo + prng.random().uintLessThan(usize, sh.len));
            var got_nvme: [q.len * ep.row_bytes]u8 = undefined;
            var got_res: [q.len * ep.row_bytes]u8 = undefined;
            try nvme.rawRows(0, &q, &got_nvme);
            try res.fetch(0, &q, &got_res);
            try testing.expectEqualSlices(u8, &got_nvme, &got_res);
        }
    }
}
