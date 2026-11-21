//! The rank loader's packing: shares of bf16 and block-FP8 projections, stacked parts, and the sliced fused experts.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const names = @import("names.zig");
const partition = @import("partition.zig");
const cw = @import("cuda_weights.zig");

const testing = std.testing;
const Tensor = core.checkpoint.Tensor;
const fp8 = cuda.fp8;
const fx8 = cuda.experts.Fp8Experts;

fn tensor(dtype: core.safetensors.DType, shape: []const usize, bytes: []const u8) Tensor {
    var t: Tensor = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = undefined, .bytes = bytes };
    @memcpy(t.shape[0..shape.len], shape);
    return t;
}

fn fill(bytes: []u8, seed: u8) void {
    for (bytes, 0..) |*b, i| b.* = @truncate(i *% 131 +% seed);
}

test "bf16 shares: rows stack and inputs slice before the lane's packing" {
    const a = testing.allocator;
    const n = 256;
    const k = 192;
    var g: [n * k]u16 = undefined;
    var u: [n * k]u16 = undefined;
    for (&g, &u, 0..) |*x, *y, i| {
        x.* = @truncate(i * 3 + 1);
        y.* = @truncate(i * 5 + 2);
    }
    const gt = tensor(.bf16, &.{ n, k }, std.mem.sliceAsBytes(&g));
    const ut = tensor(.bf16, &.{ n, k }, std.mem.sliceAsBytes(&u));
    const rows: partition.Share = .{ .lo = 128, .len = 128 };
    const cols: partition.Share = .{ .lo = 64, .len = 128 };
    var p = try cw.pack(a, &.{ .{ .p = .{ .bf16 = gt }, .rows = rows }, .{ .p = .{ .bf16 = ut }, .rows = rows } }, cols);
    defer p.deinit(a);
    try testing.expect(p.mode == .bf16 and p.n == 256 and p.k == 128 and p.scales.len == 0);
    var want_rows: [256 * 128]u16 = undefined;
    for (0..256) |r| for (0..128) |c| {
        const src = if (r < 128) &g else &u;
        want_rows[r * 128 + c] = src[(rows.lo + r % 128) * k + cols.lo + c];
    };
    const want = try a.alloc(u16, 256 * 128);
    defer a.free(want);
    cuda.qmmf.packBf16(want, &want_rows, 256, 128);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), p.codes);
}

test "block-FP8 shares: codes repacked, bf16 scale_inv widened into the share's tiles, stacked parts too" {
    const a = testing.allocator;
    const n = 384;
    const k = 256;
    var codes: [2][n * k]u8 = undefined;
    fill(&codes[0], 3);
    fill(&codes[1], 77);
    var inv: [2][3 * 2]u16 = undefined;
    for (&inv, 0..) |*s, p| for (s, 0..) |*v, i| {
        v.* = core.bf16_round.bits(@as(f32, @floatFromInt(i + 1 + 10 * p)) * 0.25);
    };
    var parts: [2]names.Proj = undefined;
    for (&parts, 0..) |*q, p| q.* = .{ .fp8 = .{ .w = tensor(.f8_e4m3, &.{ n, k }, &codes[p]), .s = tensor(.bf16, &.{ 3, 2 }, std.mem.sliceAsBytes(&inv[p])) } };
    const rows: partition.Share = .{ .lo = 128, .len = 128 };
    const cols: partition.Share = .{ .lo = 128, .len = 128 };
    // one part: packCodes of the share, packScalesAt of the whole's blocks
    var one = try cw.pack(a, &.{.{ .p = parts[0], .rows = rows }}, cols);
    defer one.deinit(a);
    var sub: [128 * 128]u8 = undefined;
    for (0..128) |r| @memcpy(sub[r * 128 ..][0..128], codes[0][(rows.lo + r) * k + cols.lo ..][0..128]);
    var want: [128 * 128]u8 = undefined;
    fp8.packCodes(&want, &sub, 128, 128);
    try testing.expectEqualSlices(u8, &want, one.codes);
    var whole: [6]f32 = undefined;
    for (&whole, inv[0]) |*w, b| w.* = core.bf16_round.value(b);
    var tiles: [128 * 2]f32 = undefined;
    fp8.packScalesAt(&tiles, &whole, n, k, rows.lo, cols.lo, 128, 128);
    try testing.expectEqualSlices(f32, &tiles, one.scales);
    try testing.expectEqual(@as(f32, 4 * 0.25), one.scales[0]); // block (1, 1) of [3, 2]: the fourth value
    // two parts: each row's group scale is its own part's block, in stacked order
    var two = try cw.pack(a, &.{ .{ .p = parts[0], .rows = rows }, .{ .p = parts[1], .rows = .{ .lo = 256, .len = 128 } } }, cols);
    defer two.deinit(a);
    try testing.expectEqual(@as(usize, 256), two.n);
    for (0..256) |r| for (0..2) |gi| {
        const p = r / 128;
        const src_row = if (p == 0) rows.lo + r else 256 + r - 128;
        const want_s = core.bf16_round.value(inv[p][(src_row / 128) * 2 + (cols.lo + gi * 64) / 128]);
        try testing.expectEqual(want_s, two.scales[(r / 64 * 2 + gi) * 64 + r % 64]);
    };
}

test "shares off the 64-value groups, past the rows, or mixed formats are refused" {
    const a = testing.allocator;
    var w: [256 * 128]u16 = @splat(0);
    var c: [256 * 128]u8 = @splat(0);
    var s: [2]u16 = @splat(0x3f80);
    const b: names.Proj = .{ .bf16 = tensor(.bf16, &.{ 256, 128 }, std.mem.sliceAsBytes(&w)) };
    const f: names.Proj = .{ .fp8 = .{ .w = tensor(.f8_e4m3, &.{ 256, 128 }, &c), .s = tensor(.bf16, &.{ 2, 1 }, std.mem.sliceAsBytes(&s)) } };
    const all: partition.Share = .{ .lo = 0, .len = 128 };
    try testing.expectError(error.SplitNotOnBlocks, cw.pack(a, &.{.{ .p = b, .rows = all }}, .{ .lo = 32, .len = 64 }));
    try testing.expectError(error.SplitOutOfRange, cw.pack(a, &.{.{ .p = b, .rows = .{ .lo = 200, .len = 128 } }}, all));
    try testing.expectError(error.UnexpectedTensor, cw.pack(a, &.{ .{ .p = b, .rows = all }, .{ .p = f, .rows = all } }, all));
    try testing.expectError(error.SplitNotOnBlocks, cw.pack(a, &.{ .{ .p = f, .rows = .{ .lo = 0, .len = 96 } }, .{ .p = f, .rows = all } }, all));
}

test "fused experts: gate rows then up rows of gate_up_proj, down's inputs, each rank's blocks of both scales" {
    const a = testing.allocator;
    const e = 2;
    const full = 256; // an expert's intermediate width: two 128-row blocks
    const l = 128; // the latent width
    var gu: [e * 2 * full * l]u8 = undefined;
    var dn: [e * l * full]u8 = undefined;
    fill(&gu, 5);
    fill(&dn, 91);
    var gus: [e * 4 * 1]u16 = undefined;
    var dns: [e * 1 * 2]u16 = undefined;
    for (&gus, 0..) |*v, i| v.* = core.bf16_round.bits(@floatFromInt(i + 1));
    for (&dns, 0..) |*v, i| v.* = core.bf16_round.bits(@floatFromInt(100 + i));
    var m: names.Moe = undefined;
    m.gate_up = tensor(.f8_e4m3, &.{ e, 2 * full, l }, &gu);
    m.gate_up_s = tensor(.bf16, &.{ e, 4, 1 }, std.mem.sliceAsBytes(&gus));
    m.down = tensor(.f8_e4m3, &.{ e, l, full }, &dn);
    m.down_s = tensor(.bf16, &.{ e, 1, 2 }, std.mem.sliceAsBytes(&dns));
    for (0..2) |rank| {
        const share = try partition.even(full, 128, rank, 2);
        var h = try cw.hostExperts(a, m, share);
        defer h.deinit(a);
        const w = share.len;
        for (0..e) |x| {
            const base = x * 2 * full * l;
            const want = try a.alloc(u32, w * l / 2);
            defer a.free(want);
            try fx8.packGateUp(a, want, gu[base + share.lo * l ..][0 .. w * l], gu[base + (full + share.lo) * l ..][0 .. w * l], w, l);
            try testing.expectEqualSlices(u32, want, h.up[x * w * l / 2 ..][0 .. w * l / 2]);
            var cols: [l * 128]u8 = undefined;
            for (0..l) |r| @memcpy(cols[r * w ..][0..w], dn[x * l * full + r * full + share.lo ..][0..w]);
            var wd: [l * 128 / 4]u32 = undefined;
            fx8.packOne(&wd, &cols, l, w);
            try testing.expectEqualSlices(u32, &wd, h.down[x * w * l / 4 ..][0 .. w * l / 4]);
            // gate block (x, rank), up block (x, 2 + rank); down block (x, rank)
            try testing.expectEqual(@as(f32, @floatFromInt(x * 4 + rank + 1)), h.up_scale[x * 2]);
            try testing.expectEqual(@as(f32, @floatFromInt(x * 4 + 2 + rank + 1)), h.up_scale[x * 2 + 1]);
            try testing.expectEqual(@as(f32, @floatFromInt(100 + x * 2 + rank)), h.down_scale[x]);
        }
    }
    try testing.expectError(error.SplitNotOnBlocks, cw.hostExperts(a, m, .{ .lo = 64, .len = 128 }));
}
