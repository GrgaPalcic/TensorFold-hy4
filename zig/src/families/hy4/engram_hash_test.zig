//! The Engram hash against NumPy's run of transformers' formulas: 2 documents, pad and out-of-map ids, 4 layers.
const std = @import("std");
const config = @import("config.zig");
const hash = @import("engram_hash.zig");

const testing = std.testing;

const Golden = struct {
    compressed_vocab: u64,
    ids: []const u32,
    positions: []const u32,
    windows: []const [hash.span]u32,
    folded: []const [hash.span]u64,
    layers: std.json.ArrayHashMap(struct { multipliers: [hash.span]u64, head_sizes: [hash.heads]u32, hash_ids: []const [hash.heads]u32 }),
};

fn golden() !std.json.Parsed(Golden) {
    return std.json.parseFromSlice(Golden, testing.allocator, @embedFile("fixtures/engram_hash_golden.json"), .{});
}

test "windows stop at document starts; hash ids match NumPy's in every layer and head" {
    const g = try golden();
    defer g.deinit();
    const v = g.value;
    try testing.expectEqual(@as(u64, hash.compressed), v.compressed_vocab);
    for (v.windows, 0..) |want, i| try testing.expectEqual(want, hash.window(v.ids, v.positions, i, 120002));
    for (hash.layers) |l| {
        var key: [8]u8 = undefined;
        const want = v.layers.map.get(try std.fmt.bufPrint(&key, "{d}", .{l.id})) orelse return error.TestUnexpectedResult;
        try testing.expectEqual(want.multipliers, l.mult);
        try testing.expectEqual(want.head_sizes, l.sizes);
        for (v.folded, 0..) |f, i| try testing.expectEqual(want.hash_ids[i], hash.ids(&l, f));
    }
}

test "each layer's rows are its primes' sum, the heads back to back, as the release's tables hold them" {
    const rows = [_]u32{ 164994380, 164999008, 165004086, 165008942 };
    for (hash.layers, rows) |l, n| {
        try testing.expectEqual(n, l.rows);
        try testing.expectEqual(@as(u32, 0), l.offsets[0]);
        try testing.expectEqual(l.rows - l.sizes[15], l.offsets[15]);
    }
}

test "the release config passes check; another seed, map or prime is refused by name" {
    var why: config.Why = .{};
    const c = try config.parse(testing.allocator, @embedFile("fixtures/config.json"), &why);
    try hash.check(c, &why);
    var seeded = c;
    seeded.engram_seed = 1;
    try testing.expectError(error.UnsupportedEngram, hash.check(seeded, &why));
    try testing.expect(std.mem.indexOf(u8, why.text(), "seed 1") != null);
    var prime = c;
    prime.engram_sizes[2][7] += 2;
    try testing.expectError(error.UnsupportedEngram, hash.check(prime, &why));
    try testing.expect(std.mem.indexOf(u8, why.text(), "head 7") != null);
}

test "a lookup_table must fold into 0..92,619 and reach its top" {
    var why: config.Why = .{};
    const bytes = try testing.allocator.alloc(u8, hash.map_size * 4);
    defer testing.allocator.free(bytes);
    for (0..hash.map_size) |i| std.mem.writeInt(i32, bytes[i * 4 ..][0..4], @intCast(i % hash.compressed), .little);
    try hash.checkMap(bytes, &why);
    std.mem.writeInt(i32, bytes[0..4], hash.compressed, .little);
    try testing.expectError(error.UnsupportedEngram, hash.checkMap(bytes, &why));
    for (0..hash.map_size) |i| std.mem.writeInt(i32, bytes[i * 4 ..][0..4], @intCast(i % 1000), .little);
    try testing.expectError(error.UnsupportedEngram, hash.checkMap(bytes, &why));
    try testing.expectError(error.UnsupportedEngram, hash.checkMap(bytes[4..], &why));
}

test "the release's own lookup_table (TF_HY4_DIR): folds as NumPy did, rows include the head offsets" {
    const dir = @import("sys").getenv("TF_HY4_DIR") orelse return error.SkipZigTest;
    const core = @import("core");
    var ck = try core.checkpoint.Checkpoint.openModel(testing.allocator, testing.io, dir);
    defer ck.close();
    const map = try ck.expect("model.layers.1.engram_layer.lookup_table", .i32, &.{hash.map_size});
    var why: config.Why = .{};
    try hash.checkMap(map.bytes, &why);
    const g = try golden();
    defer g.deinit();
    const v = g.value;
    for (v.windows, v.folded) |w, f| for (w, f) |t, want| try testing.expectEqual(want, hash.fold(map.bytes, t, 120002));
    const l = &hash.layers[0];
    const want = v.layers.map.get("2").?;
    for (0..v.ids.len) |i| {
        const got = hash.rows(l, map.bytes, v.ids, v.positions, i, 120002);
        for (got, want.hash_ids[i], l.offsets) |r, h, o| try testing.expectEqual(h + o, r);
    }
}
