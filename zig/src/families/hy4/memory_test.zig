//! The TP split and memory floor against the release: byte totals equal its safetensors headers, shares tile the whole.
const std = @import("std");
const config = @import("config.zig");
const part = @import("partition.zig");
const memory = @import("memory.zig");

const testing = std.testing;
const GiB = 1 << 30;

fn release() !config.Config {
    var why: config.Why = .{};
    return config.parse(testing.allocator, @embedFile("fixtures/config.json"), &why);
}

const experts_bytes = 3_624_321_024; // a layer's routed expert stacks: e4m3 codes and bf16 scales
const engram_bytes = 47_782_856; // an Engram layer's tensors besides its table

test "one rank holds the release's text weights as stored, byte for byte (headers, DFly2 and vision skipped)" {
    const c = try release();
    try testing.expectEqual(@as(usize, 402_782_160), try memory.layerBytes(c, 0, 0, 1, .stored)); // dense, bf16 attention
    try testing.expectEqual(@as(usize, 139_598_800 + experts_bytes + engram_bytes), try memory.layerBytes(c, 1, 0, 1, .stored));
    try testing.expectEqual(@as(usize, 207_793_488 + experts_bytes), try memory.layerBytes(c, 2, 0, 1, .stored)); // sliding
    try testing.expectEqual(@as(usize, 139_598_800 + experts_bytes), try memory.layerBytes(c, 5, 0, 1, .stored)); // indexed
    try testing.expectEqual(@as(usize, 158_445_453_396), try memory.weightBytes(c, 0, 1, .stored));
    try testing.expectEqual(@as(usize, 126_721_231_872), try memory.engramTableBytes(c, 0, 1, .stored));
}

test "two ranks: equal halves of every split, on whole heads and 128-row blocks" {
    const c = try release();
    var why: config.Why = .{};
    try part.check(c, 2, &why);
    for (0..c.layers) |l| {
        const a = [2]part.Attn{ try part.Attn.of(c, l, 0, 2), try part.Attn.of(c, l, 1, 2) };
        const h = c.heads(l);
        try testing.expectEqual(h.count / 2, a[0].heads.len);
        try testing.expectEqual(a[0].q_rows.len, a[1].q_rows.lo);
        try testing.expectEqual(h.count * (h.nope + h.rope), a[1].q_rows.lo + a[1].q_rows.len);
        try testing.expectEqual(h.count * (h.nope + h.v), a[1].kv_rows.lo + a[1].kv_rows.len);
        try testing.expectEqual(@as(usize, 0), a[1].o_cols.lo % part.block);
    }
    try testing.expectEqual(part.Share{ .lo = 768, .len = 768 }, try part.expert(c, 1, 2));
    try testing.expectEqual(part.Share{ .lo = 60_416, .len = 60_416 }, try part.vocab(c, 1, 2));
    const e = [2]part.Engram{ try part.Engram.of(c, 0, 2), try part.Engram.of(c, 1, 2) };
    try testing.expect(e[0].bias and !e[1].bias);
    try testing.expectEqual(part.Share{ .lo = 8, .len = 8 }, e[1].heads); // the 3-gram heads
    for (0..config.engram_layers) |s| {
        const r = [2]part.Share{ e[0].rows(c, s), e[1].rows(c, s) };
        try testing.expectEqual(r[0].len, r[1].lo);
        var all: usize = 0;
        for (c.engram_sizes[s]) |p| all += p;
        try testing.expectEqual(all, r[1].lo + r[1].len);
    }
}

test "two ranks: about 75 GiB of weights each, 30 GiB of Engram rows packed, 16.5 KiB of KV a token" {
    const c = try release();
    const w = [2]usize{ try memory.weightBytes(c, 0, 2, .packed_), try memory.weightBytes(c, 1, 2, .packed_) };
    try testing.expectEqual(w[0], w[1]);
    try testing.expect(w[0] > 74 * GiB and w[0] < 76 * GiB);
    try testing.expect(w[0] + w[1] > try memory.weightBytes(c, 0, 1, .packed_)); // replicated parts counted twice
    const rows = [2]usize{ try memory.engramTableBytes(c, 0, 2, .packed_), try memory.engramTableBytes(c, 1, 2, .packed_) };
    try testing.expectEqual(@as(usize, 660_006_416 * 100), rows[0] + rows[1]);
    try testing.expect(rows[0] > 30 * GiB and rows[0] < 31 * GiB);
    try testing.expectEqual(@as(usize, 12 * (1_152 + 256)), memory.tokenBytes(c));
    try testing.expectEqual(@as(usize, 30 * 512 * 1_152), memory.ringBytes(c));
}

test "four ranks: equal quarters on whole heads and 128-row blocks, ~38.4 GiB weights and ~15.4 GiB Engram each" {
    const c = try release();
    var why: config.Why = .{};
    try part.check(c, 4, &why);
    try testing.expectEqual(part.Share{ .lo = 24, .len = 8 }, (try part.Attn.of(c, 0, 3, 4)).heads); // indexed: 32 -> 4 x 8
    try testing.expectEqual(part.Share{ .lo = 48, .len = 16 }, (try part.Attn.of(c, 2, 3, 4)).heads); // sliding: 64 -> 4 x 16
    try testing.expectEqual(part.Share{ .lo = 1_152, .len = 384 }, try part.expert(c, 3, 4)); // 12 blocks -> 4 x 3 blocks
    try testing.expectEqual(part.Share{ .lo = 90_624, .len = 30_208 }, try part.vocab(c, 3, 4));
    var w_sum: usize = 0;
    var row_sum: usize = 0;
    for (0..4) |r| {
        const w = try memory.weightBytes(c, r, 4, .packed_);
        try testing.expectEqual(try memory.weightBytes(c, 0, 4, .packed_), w);
        try testing.expect(w > 38 * GiB and w < 39 * GiB);
        w_sum += w;
        const e = try part.Engram.of(c, r, 4);
        try testing.expectEqual(@as(usize, 4), e.heads.len);
        try testing.expectEqual(r == 0, e.bias);
        const rows = try memory.engramTableBytes(c, r, 4, .packed_);
        try testing.expect(rows > 15 * GiB and rows < 16 * GiB);
        row_sum += rows;
    }
    try testing.expect(w_sum > try memory.weightBytes(c, 0, 1, .packed_));
    try testing.expectEqual(@as(usize, 660_006_416 * 100), row_sum);
}

test "worlds the split refuses: three ranks, zero, five, or a width off the blocks" {
    var c = try release();
    var why: config.Why = .{};
    try testing.expectError(error.UnsupportedWorld, part.check(c, 0, &why));
    try testing.expectError(error.UnsupportedWorld, part.check(c, 3, &why));
    try testing.expectError(error.UnsupportedWorld, part.check(c, 5, &why));
    try testing.expectError(error.UnsupportedWorld, part.Attn.of(c, 0, 2, 2));
    c.expert_width = 1_408;
    try testing.expectError(error.SplitNotOnBlocks, part.check(c, 2, &why));
    try testing.expect(std.mem.indexOf(u8, why.text(), "routed expert width 1408") != null);
    try part.check(c, 1, &why);
}
