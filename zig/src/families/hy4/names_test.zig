//! The tensor map on a small synthetic checkpoint, and on the release itself (TF_HY4_DIR, whole or a part).
const std = @import("std");
const core = @import("core");
const config = @import("config.zig");
const names = @import("names.zig");

const testing = std.testing;
const io = testing.io;
const Entry = core.checkpoint_fixture.Entry;

/// The release config shrunk: 4 layers (indexed, sliding, indexed, sliding), layer 0 dense, Engram on all four.
fn tiny() !config.Config {
    var why: config.Why = .{};
    var c = try config.parse(testing.allocator, @embedFile("fixtures/config.json"), &why);
    c.layers = 4;
    c.hidden = 256;
    c.vocab = 8;
    c.q_lora = 128;
    c.indexed.count = 1;
    c.sliding.count = 2;
    c.kind = @splat(.sliding);
    c.kind[0] = .indexed;
    c.kind[2] = .indexed;
    c.dense = @splat(false);
    c.dense[0] = true;
    c.dense_width = 128;
    c.experts = 2;
    c.expert_width = 128;
    c.latent = 128;
    c.shared_width = 128;
    c.engram_ids = .{ 1, 2, 3, 4 };
    for (&c.engram_sizes) |*s| s.* = @splat(2);
    return c;
}

const Fixture = struct {
    a: std.mem.Allocator,
    list: std.ArrayList(Entry) = .empty,

    fn add(f: *Fixture, dtype: []const u8, shape: []const usize, comptime fmt: []const u8, args: anytype) !void {
        var n: usize = if (std.mem.eql(u8, dtype, "BF16")) 2 else if (std.mem.eql(u8, dtype, "F8_E4M3")) 1 else 4;
        for (shape) |x| n *= x;
        const bytes = try f.a.alloc(u8, n);
        @memset(bytes, 0);
        try f.list.append(f.a, .{ .name = try std.fmt.allocPrint(f.a, fmt, args), .dtype = dtype, .shape = try f.a.dupe(usize, shape), .bytes = bytes });
    }

    /// A projection in bf16, or as e4m3 + bf16 scale_inv.
    fn proj(f: *Fixture, q: bool, n: usize, k: usize, comptime fmt: []const u8, args: anytype) !void {
        if (!q) return f.add("BF16", &.{ n, k }, fmt ++ ".weight", args);
        try f.add("F8_E4M3", &.{ n, k }, fmt ++ ".weight", args);
        try f.add("BF16", &.{ core.quant.fp8.blocks(n), core.quant.fp8.blocks(k) }, fmt ++ ".weight_scale_inv", args);
    }
};

fn write(a: std.mem.Allocator, tmp: testing.TmpDir, c: config.Config, extra: bool) !void {
    var f: Fixture = .{ .a = a };
    const d = c.hidden;
    const m = c.hc_mult;
    try f.add("BF16", &.{ c.vocab, d }, "model.embed_tokens.weight", .{});
    try f.add("BF16", &.{d}, "model.norm.weight", .{});
    try f.add("BF16", &.{ c.vocab, d }, "lm_head.weight", .{});
    try f.add("F32", &.{ m, m * d }, "model.hc_head.hc_head_fn", .{});
    try f.add("F32", &.{m}, "model.hc_head.hc_head_base", .{});
    try f.add("F32", &.{1}, "model.hc_head.hc_head_scale", .{});
    try f.add("BF16", &.{ 2, 2 }, "model.dfly_layers.0.mlp.up_proj.weight", .{}); // skipped on purpose
    try f.add("BF16", &.{ 2, 2 }, "vit.embeddings.patch_embedding.weight", .{});
    if (extra) try f.add("BF16", &.{2}, "model.layers.0.self_attn.q_norm.weight", .{});
    for (0..c.layers) |i| {
        const h = c.heads(i);
        const q = i > 0; // layer 0 stays bf16, as in the release
        for ([_][]const u8{ "input_layernorm", "post_attention_layernorm" }) |n| try f.add("BF16", &.{d}, "model.layers.{d}.{s}.weight", .{ i, n });
        for ([_][]const u8{ "hc_attn_layer", "hc_mlp_layer" }) |s| {
            try f.add("F32", &.{ 2 * m, m * d }, "model.layers.{d}.{s}.hc_pre.hc_fn", .{ i, s });
            try f.add("F32", &.{2 * m}, "model.layers.{d}.{s}.hc_pre.hc_base", .{ i, s });
            try f.add("F32", &.{2}, "model.layers.{d}.{s}.hc_pre.hc_scale", .{ i, s });
        }
        const a_ = "model.layers.{d}.self_attn.";
        try f.add("BF16", &.{ c.q_lora, d }, a_ ++ "q_a_proj.weight", .{i});
        try f.add("BF16", &.{c.q_lora}, a_ ++ "q_a_layernorm.weight", .{i});
        try f.proj(q, h.count * (h.nope + h.rope), c.q_lora, a_ ++ "q_b_proj", .{i});
        try f.add("BF16", &.{ c.kv_lora + h.rope, d }, a_ ++ "kv_a_proj_with_mqa.weight", .{i});
        try f.add("BF16", &.{c.kv_lora}, a_ ++ "kv_a_layernorm.weight", .{i});
        try f.add("BF16", &.{ h.count * (h.nope + h.v), c.kv_lora }, a_ ++ "kv_b_proj.weight", .{i});
        try f.proj(q, h.count * h.v, d, a_ ++ "linear_gate", .{i});
        try f.proj(q, d, h.count * h.v, a_ ++ "o_proj", .{i});
        try f.add("BF16", &.{d}, a_ ++ "post_proj_norm.weight", .{i});
        try f.add("F32", &.{h.count}, a_ ++ "learnable_sink_param", .{i});
        if (c.kind[i] == .indexed) {
            const x = a_ ++ "compressed_indexer.";
            const n = c.index_dim;
            try f.add("BF16", &.{ c.index_heads * n, c.q_lora }, x ++ "query_projection.weight", .{i});
            try f.add("BF16", &.{ c.index_heads, d }, x ++ "head_weight_projection.weight", .{i});
            try f.add("BF16", &.{ n, d }, x ++ "token_key_projection.weight", .{i});
            try f.add("BF16", &.{n}, x ++ "token_key_norm.weight", .{i});
            try f.add("F32", &.{ n, n }, x ++ "key_value_projection.weight", .{i});
            try f.add("F32", &.{ n, n }, x ++ "key_gate_projection.weight", .{i});
            try f.add("F32", &.{n}, x ++ "key_norm.weight", .{i});
            try f.add("F32", &.{ c.index_ratio, n }, x ++ "intra_group_ape", .{i});
        }
        const p = "model.layers.{d}.mlp.";
        if (c.dense[i]) {
            try f.add("BF16", &.{ c.dense_width, d }, p ++ "gate_proj.weight", .{i});
            try f.add("BF16", &.{ c.dense_width, d }, p ++ "up_proj.weight", .{i});
            try f.add("BF16", &.{ d, c.dense_width }, p ++ "down_proj.weight", .{i});
            try f.add("BF16", &.{d}, p ++ "post_fc2_norm.weight", .{i});
        } else {
            const e = c.experts;
            const w = c.expert_width;
            const l = c.latent;
            try f.add("BF16", &.{ e, d }, p ++ "gate.weight", .{i});
            try f.add("F32", &.{e}, p ++ "gate.e_score_correction_bias", .{i});
            try f.add("BF16", &.{ l, d }, p ++ "routed_expert_down_proj.weight", .{i});
            try f.add("BF16", &.{ d, l }, p ++ "routed_expert_up_proj.weight", .{i});
            try f.add("F8_E4M3", &.{ e, 2 * w, l }, p ++ "experts.gate_up_proj", .{i});
            try f.add("BF16", &.{ e, 2 * w / 128, l / 128 }, p ++ "experts.gate_up_proj_scale_inv", .{i});
            try f.add("F8_E4M3", &.{ e, l, w }, p ++ "experts.down_proj", .{i});
            try f.add("BF16", &.{ e, l / 128, w / 128 }, p ++ "experts.down_proj_scale_inv", .{i});
            try f.add("BF16", &.{ c.shared_width, d }, p ++ "shared_experts.gate_proj.weight", .{i});
            try f.add("BF16", &.{ c.shared_width, d }, p ++ "shared_experts.up_proj.weight", .{i});
            try f.add("BF16", &.{ d, c.shared_width }, p ++ "shared_experts.down_proj.weight", .{i});
            try f.add("BF16", &.{d}, p ++ "combine_output_norm.weight", .{i});
        }
        if (c.engramSlot(i)) |s| {
            const g = "model.layers.{d}.engram_layer.";
            var rows: usize = 0;
            for (c.engram_sizes[s]) |r| rows += r;
            const k = config.engram_heads * c.engram_dim;
            try f.add("BF16", &.{ rows, c.engram_dim }, g ++ "multi_head_embedding.embedding.weight", .{i});
            try f.add("I32", &.{c.engram_map}, g ++ "lookup_table", .{i});
            for (0..config.engram_heads) |head| try f.add("BF16", &.{c.engram_dim}, g ++ "lookup_per_head_norms.{d}.weight", .{ i, head });
            try f.add("BF16", &.{ d, k }, g ++ "value_proj.weight", .{i});
            try f.add("BF16", &.{d}, g ++ "value_proj.bias", .{i});
            for (0..4) |j| {
                try f.add("BF16", &.{ d, k }, g ++ "key_projs.{d}.weight", .{ i, j });
                try f.add("BF16", &.{d}, g ++ "key_projs.{d}.bias", .{ i, j });
                try f.add("BF16", &.{d}, g ++ "norm_key.{d}.weight", .{ i, j });
                try f.add("BF16", &.{d}, g ++ "norm_query.{d}.weight", .{ i, j });
                try f.add("BF16", &.{d}, g ++ "unified_per_hc_scales.{d}", .{ i, j });
            }
            try f.add("BF16", &.{d}, g ++ "unified_pre_gamma", .{i});
        }
    }
    try core.checkpoint_fixture.write(a, io, tmp.dir, "model.safetensors", f.list.items);
}

fn tmpPath(a: std.mem.Allocator, tmp: testing.TmpDir) ![]u8 {
    const cwd = try std.process.currentPathAlloc(io, a);
    return std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
}

test "every tensor maps: bf16 and e4m3 projections by their own tensors, the drafter and tower skipped" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const c = try tiny();
    try write(a, tmp, c, false);
    var ck = try core.checkpoint.Checkpoint.openModel(testing.allocator, io, try tmpPath(a, tmp));
    defer ck.close();
    var m = try names.map(testing.allocator, &ck, c);
    defer m.deinit();
    try testing.expect(m.layers[0].attn.q_b == .bf16 and m.layers[1].attn.q_b == .fp8 and m.layers[3].attn.o == .fp8);
    try testing.expectEqual(core.checkpoint.DType.bf16, m.layers[1].attn.o.fp8.s.dtype);
    try testing.expect(m.layers[0].attn.indexer != null and m.layers[1].attn.indexer == null);
    try testing.expect(m.layers[0].mlp == .dense and m.layers[1].mlp == .moe);
    try testing.expectEqual(@as(usize, 32), m.layers[2].engram.?.table.dim(0));
    try testing.expectEqual(@as(usize, 2 * 256), m.layers[3].attn.q_b.rows()); // a sliding layer's 2 heads of 192 + 64
}

test "a tensor the family doesn't read is the one left over (map refuses on it, logging its name)" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const c = try tiny();
    try write(a, tmp, c, true);
    var ck = try core.checkpoint.Checkpoint.openModel(testing.allocator, io, try tmpPath(a, tmp));
    defer ck.close();
    var m = try names.mapped(testing.allocator, &ck, c);
    defer m.deinit();
    var left: usize = 0;
    for (ck.files.items) |*f| for (f.names.keys()) |k| {
        if (ck.used.contains(k)) continue;
        try testing.expectEqualStrings("model.layers.0.self_attn.q_norm.weight", k);
        left += 1;
    };
    try testing.expectEqual(@as(usize, 1), left);
}

test "the release (TF_HY4_DIR): every layer present maps, with the release's formats" {
    const dir = @import("sys").getenv("TF_HY4_DIR") orelse return error.SkipZigTest;
    const c = try config.Config.read(testing.allocator, io, dir);
    var ck = try core.checkpoint.Checkpoint.openModel(testing.allocator, io, dir);
    defer ck.close();
    const t = try names.top(&ck, c);
    try testing.expectEqual(@as(usize, 120832), t.head.dim(0));
    var seen: usize = 0;
    for (0..c.layers) |i| {
        var buf: [128]u8 = undefined;
        if (!ck.has(try std.fmt.bufPrint(&buf, "model.layers.{d}.input_layernorm.weight", .{i}))) continue;
        // a per-layer part may hold a layer without its Engram table
        if (c.engramSlot(i) != null and !ck.has(try std.fmt.bufPrint(&buf, "model.layers.{d}.engram_layer.multi_head_embedding.embedding.weight", .{i}))) continue;
        const l = try names.layer(&ck, c, i);
        seen += 1;
        try testing.expect((l.attn.q_b == .bf16) == (i == 0));
        try testing.expect((l.attn.indexer != null) == (c.kind[i] == .indexed));
        try testing.expect((l.engram != null) == (c.engramSlot(i) != null));
    }
    try testing.expect(seen > 0);
}
