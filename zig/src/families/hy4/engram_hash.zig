//! Hy4-flash's Engram hash as transformers' hash_ngrams computes it, the release's multipliers written out.
const std = @import("std");
const config = @import("config.zig");

pub const orders = config.engram_orders; // 2-grams and 3-grams
pub const span = orders + 1; // tokens an n-gram window holds: the current one and the two before it
pub const heads = config.engram_heads;
pub const map_size = 120818;
pub const compressed = 92620; // max(lookup_table) + 1: the bound the multipliers were drawn under
pub const seed = 0;

pub const Layer = struct {
    id: u32, // one-based, as config.json names it
    mult: [span]u64,
    sizes: [heads]u32, // order-major: the 2-gram heads, then the 3-gram heads
    offsets: [heads]u32, // each head's first row in the layer's table
    rows: u32,
};

fn layer(id: u32, mult: [span]u64, sizes: [heads]u32) Layer {
    var l: Layer = .{ .id = id, .mult = mult, .sizes = sizes, .offsets = undefined, .rows = 0 };
    for (sizes, 0..) |s, h| {
        l.offsets[h] = l.rows;
        l.rows += s;
    }
    return l;
}

pub const layers = [config.engram_layers]Layer{
    layer(2, .{ 81853575799317, 39690085502883, 24229915018165 }, .{
        10312007, 10312021, 10312039, 10312069, 10312091, 10312121, 10312123, 10312147,
        10312177, 10312187, 10312207, 10312213, 10312229, 10312241, 10312249, 10312259,
    }),
    layer(12, .{ 19457948045719, 44729556510419, 84779352922315 }, .{
        10312277, 10312283, 10312301, 10312343, 10312373, 10312397, 10312409, 10312453,
        10312457, 10312483, 10312507, 10312513, 10312531, 10312543, 10312567, 10312571,
    }),
    layer(22, .{ 72974230057307, 21530602121221, 55998057243097 }, .{
        10312613, 10312619, 10312633, 10312657, 10312681, 10312699, 10312723, 10312747,
        10312759, 10312789, 10312811, 10312823, 10312867, 10312873, 10312889, 10312903,
    }),
    layer(32, .{ 29128509562787, 16745155128523, 22474588302733 }, .{
        10312909, 10312919, 10312927, 10312957, 10312979, 10313011, 10313021, 10313057,
        10313063, 10313117, 10313123, 10313129, 10313161, 10313179, 10313189, 10313201,
    }),
};

/// Refuses a config whose Engram block isn't the release's: layer ids, every head's prime, the seed and map size.
pub fn check(c: config.Config, why: *config.Why) !void {
    if (c.engram_seed != seed or c.engram_map != map_size) {
        why.set("Engram seed {d} over a {d}-entry map: this family hashes the release's (seed 0, 120,818 entries)", .{ c.engram_seed, c.engram_map });
        return error.UnsupportedEngram;
    }
    for (layers, 0..) |l, s| {
        if (c.engram_ids[s] != l.id) {
            why.set("Engram layer {d} where the release has {d}", .{ c.engram_ids[s], l.id });
            return error.UnsupportedEngram;
        }
        for (l.sizes, 0..) |p, h| if (c.engram_sizes[s][h] != p) {
            why.set("Engram layer {d} head {d}: {d} rows where the release has {d}", .{ l.id, h, c.engram_sizes[s][h], p });
            return error.UnsupportedEngram;
        };
    }
}

/// The lookup_table (int32 [120,818]) the multipliers were drawn for: every id below 92,620, the largest at it.
pub fn checkMap(bytes: []const u8, why: *config.Why) !void {
    if (bytes.len != map_size * 4) {
        why.set("Engram lookup_table holds {d} bytes, not {d}", .{ bytes.len, map_size * 4 });
        return error.UnsupportedEngram;
    }
    var most: i32 = -1;
    for (0..map_size) |i| {
        const v = std.mem.readInt(i32, bytes[i * 4 ..][0..4], .little);
        if (v < 0 or v >= compressed) {
            why.set("Engram lookup_table maps token {d} to {d}, outside 0..{d}", .{ i, v, compressed - 1 });
            return error.UnsupportedEngram;
        }
        most = @max(most, v);
    }
    if (most + 1 != compressed) {
        why.set("Engram lookup_table's largest id is {d}: its multipliers would differ from the release's", .{most});
        return error.UnsupportedEngram;
    }
}

/// A token's normalized id: through the map, ids past it read as the pad token's.
pub fn fold(map: []const u8, token: u32, pad: u32) u64 {
    const t = if (token < map_size) token else pad;
    return @intCast(std.mem.readInt(i32, map[t * 4 ..][0..4], .little));
}

/// Row `i`'s n-gram window, current token first; a slot before its document's start (positions restart) is pad.
pub fn window(tokens: []const u32, positions: []const u32, i: usize, pad: u32) [span]u32 {
    var w: [span]u32 = undefined;
    for (0..span) |o| w[o] = if (positions[i] >= o and i >= o) tokens[i - o] else pad;
    return w;
}

/// The 16 heads' hash ids (each below its head's prime) of one window's normalized ids, current first.
pub fn ids(l: *const Layer, folded: [span]u64) [heads]u32 {
    var mixed: [span]u64 = undefined;
    for (0..span) |o| mixed[o] = folded[o] *% l.mult[o];
    var out: [heads]u32 = undefined;
    for (0..orders) |r| {
        var m = mixed[0];
        for (1..r + 2) |o| m ^= mixed[o];
        for (0..config.engram_heads_per_order) |h| {
            const at = r * config.engram_heads_per_order + h;
            out[at] = @intCast(m % l.sizes[at]);
        }
    }
    return out;
}

/// Row `i`'s 16 rows in layer `l`'s table (head offsets added).
pub fn rows(l: *const Layer, map: []const u8, tokens: []const u32, positions: []const u32, i: usize, pad: u32) [heads]u32 {
    const w = window(tokens, positions, i, pad);
    var folded: [span]u64 = undefined;
    for (w, 0..) |t, o| folded[o] = fold(map, t, pad);
    var out = ids(l, folded);
    for (&out, 0..) |*r, h| r.* += l.offsets[h];
    return out;
}

test {
    _ = @import("engram_hash_test.zig");
}
