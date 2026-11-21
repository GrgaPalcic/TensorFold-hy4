//! Hy4-flash's config: the release's own config.json (its module list cut to two), and each refusal by name.
const std = @import("std");
const config = @import("config.zig");
const Config = config.Config;
const Why = config.Why;

const testing = std.testing;
const release = @embedFile("fixtures/config.json");

fn refused(text: []const u8, err: anyerror, word: []const u8) !void {
    var why: Why = .{};
    try testing.expectError(err, config.parse(testing.allocator, text, &why));
    if (std.mem.indexOf(u8, why.text(), word) == null) {
        std.debug.print("refusal \"{s}\" does not name \"{s}\"\n", .{ why.text(), word });
        return error.TestUnexpectedResult;
    }
}

fn replaced(from: []const u8, to: []const u8) ![]u8 {
    if (std.mem.indexOf(u8, release, from) == null) {
        std.debug.print("fixture has no \"{s}\"\n", .{from});
        return error.TestUnexpectedResult;
    }
    return std.mem.replaceOwned(u8, testing.allocator, release, from, to);
}

test "the release config: layers, heads, experts, indexer, Engram" {
    var why: Why = .{};
    const c = try config.parse(testing.allocator, release, &why);
    try config.checkShapes(c, &why);
    try testing.expectEqual(@as(usize, 42), c.layers);
    try testing.expectEqual(@as(usize, 3072), c.hidden);
    try testing.expectEqual(@as(usize, 120832), c.vocab);
    try testing.expectEqual(@as(usize, 32), c.indexed.count);
    try testing.expectEqual(@as(usize, 64), c.sliding.count);
    try testing.expectEqual(@as(f64, 1e7), c.indexed.theta);
    try testing.expectEqual(@as(f64, 1e4), c.sliding.theta);
    try testing.expectEqual([3]usize{ 12, 10, 10 }, c.mrope);
    // swa_pattern 0,0,1,1,1,0,…: indexed at 0, 1, 5, 9, …, 41
    var indexed: usize = 0;
    for (0..c.layers) |i| indexed += @intFromBool(c.kind[i] == .indexed);
    try testing.expectEqual(@as(usize, 12), indexed);
    try testing.expect(c.kind[0] == .indexed and c.kind[1] == .indexed and c.kind[2] == .sliding and c.kind[5] == .indexed and c.kind[41] == .indexed);
    try testing.expect(c.dense[0] and !c.dense[1] and !c.dense[41]);
    try testing.expectEqual(@as(usize, 512), c.experts);
    try testing.expectEqual(@as(usize, 16), c.top_k);
    try testing.expectEqual(@as(usize, 1536), c.latent);
    try testing.expectEqual(@as(usize, 1536), c.expert_width);
    try testing.expectEqual(@as(usize, 1536), c.shared_width);
    try testing.expectEqual(@as(usize, 13824), c.dense_width);
    try testing.expectEqual(@as(f32, 3.998), c.routed_scale);
    try testing.expectEqual(@as(f32, 10), c.swiglu_limit);
    try testing.expect(c.norm_topk);
    try testing.expectEqual(@as(f32, 2), c.hc_magnitude);
    try testing.expectEqual(@as(usize, 2048), c.recall());
    try testing.expectEqual(@as(usize, 512), c.window);
    try testing.expectEqual([4]u32{ 2, 12, 22, 32 }, c.engram_ids);
    try testing.expectEqual(@as(?usize, 0), c.engramSlot(1));
    try testing.expectEqual(@as(?usize, 3), c.engramSlot(31));
    try testing.expectEqual(@as(?usize, null), c.engramSlot(2));
    try testing.expectEqual(@as(u64, 10312007), c.engram_sizes[0][0]);
    try testing.expectEqual(@as(u64, 10313201), c.engram_sizes[3][15]);
    try testing.expectEqual(@as(usize, 96), c.engram_dim);
    try testing.expectEqual(@as(u32, 120002), c.engram_pad);
    try testing.expectEqual(@as(usize, 120818), c.engram_map);
    try testing.expect(c.vision);
    try testing.expectEqual(@as(usize, 5), c.dfly_layers);
    try testing.expect(c.isEos(120025) and !c.isEos(120000));
}

test "generation_config.json's end tokens join config.json's" {
    var why: Why = .{};
    var c = try config.parse(testing.allocator, release, &why);
    try c.addEos(testing.allocator, "{\"bos_token_id\": 120000, \"eos_token_id\": 120025, \"pad_token_id\": 120002}");
    try testing.expectEqualSlices(u32, &.{120025}, c.eos[0..c.eos_count]);
    try c.addEos(testing.allocator, "{\"eos_token_id\": [120025, 120020]}");
    try testing.expectEqualSlices(u32, &.{ 120025, 120020 }, c.eos[0..c.eos_count]);
}

test "the indexed kind under transformers' newer spelling reads the same" {
    const renamed = try std.mem.replaceOwned(u8, testing.allocator, release, "deepseek_sparse_attention", "indexed_attention");
    defer testing.allocator.free(renamed);
    var why: Why = .{};
    const c = try config.parse(testing.allocator, renamed, &why);
    try testing.expectEqual(@as(f64, 1e7), c.indexed.theta);
    try testing.expect(c.kind[0] == .indexed and c.kind[2] == .sliding);
}

test "refusals name what they refuse" {
    const cases = .{
        .{ "\"model_type\": \"hy_v4_flash\",\n \"processor_class\"", "\"model_type\": \"hy_v3\",\n \"processor_class\"", error.NotHy4Flash, "hy_v4_flash" },
        .{ "\"quant_method\": \"fp8\"", "\"quant_method\": \"awq\"", error.UnsupportedQuantization, "block-FP8" },
        .{ "\"weight_block_size\": [\n    128,\n    128\n   ]", "\"weight_block_size\": [\n    64,\n    64\n   ]", error.UnsupportedQuantization, "128 x 128" },
        .{ "\"n_group\": 1", "\"n_group\": 8", error.UnsupportedRouting, "groups" },
        .{ "\"routed_expert_hidden_size\": 1536", "\"routed_expert_hidden_size\": null", error.UnsupportedLayout, "latent experts" },
        .{ "\"engram_fp8_embedding\": true", "\"engram_fp8_embedding\": false", error.UnsupportedEngram, "FP8 round trip" },
        .{ "\"engram_max_ngram_size\": 3", "\"engram_max_ngram_size\": 4", error.UnsupportedEngram, "2- and 3-grams" },
        .{ "\"rope_type\": \"default\"", "\"rope_type\": \"yarn\"", error.UnsupportedRope, "yarn" },
        .{ "\"sliding_attention\",\n   \"sliding_attention\",\n   \"deepseek_sparse_attention\"", "\"sliding_attention\",\n   \"linear_attention\",\n   \"deepseek_sparse_attention\"", error.UnsupportedLayer, "linear_attention" },
        .{ "\"num_hidden_layers\": 42", "\"num_hidden_layers\": 41", error.BadConfig, "41 layers" },
    };
    inline for (cases) |case| {
        const text = try replaced(case[0], case[1]);
        defer testing.allocator.free(text);
        try refused(text, case[2], case[3]);
    }
}

test "shapes outside the family's kernels are refused before any weight is read" {
    var why: Why = .{};
    const c = try config.parse(testing.allocator, release, &why);
    var narrow = c;
    narrow.sliding.nope = 128;
    try testing.expectError(error.UnsupportedShapes, config.checkShapes(narrow, &why));
    var latent = c;
    latent.kv_lora = 576;
    try testing.expectError(error.UnsupportedShapes, config.checkShapes(latent, &why));
    var many = c;
    many.top_k = 65;
    try testing.expectError(error.UnsupportedShapes, config.checkShapes(many, &why));
    var rows = c;
    rows.engram_dim = 128;
    try testing.expectError(error.UnsupportedShapes, config.checkShapes(rows, &why));
}
