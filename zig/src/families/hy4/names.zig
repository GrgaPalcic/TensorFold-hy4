//! Hy4-flash's checkpoint tensors by name, dtype and shape; a projection's format comes from its own tensors.
const std = @import("std");
const core = @import("core");
const config = @import("config.zig");
const Config = config.Config;

const Tensor = core.checkpoint.Tensor;
const Checkpoint = core.checkpoint.Checkpoint;
const fp8 = core.quant.fp8;

/// A projection [n, k] as stored: bf16, or e4m3 codes with bf16 scale_inv [ceil(n/128), ceil(k/128)].
pub const Proj = union(enum) {
    bf16: Tensor,
    fp8: struct { w: Tensor, s: Tensor },

    pub fn rows(p: Proj) usize {
        return switch (p) {
            .bf16 => |t| t.dim(0),
            .fp8 => |f| f.w.dim(0),
        };
    }
};

/// One iHC site: fn [2 streams, streams x hidden] fp32, base [2 streams], scale [2] (pre, post).
pub const Hc = struct { fn_: Tensor, base: Tensor, scale: Tensor };

pub const Indexer = struct {
    query: Tensor, // bf16 [heads x dim, q_lora]
    head_weights: Tensor, // bf16 [heads, hidden]
    token_key: Tensor, // bf16 [dim, hidden]
    token_key_norm: Tensor, // bf16 [dim]
    values: Tensor, // fp32 [dim, dim] (key_value_projection)
    gates: Tensor, // fp32 [dim, dim] (key_gate_projection)
    key_norm: Tensor, // fp32 [dim]
    ape: Tensor, // fp32 [ratio, dim] (intra_group_ape)
};

pub const Attn = struct {
    q_a: Tensor, // bf16 [q_lora, hidden]
    q_a_norm: Tensor,
    q_b: Proj, // [heads x (nope + rope), q_lora]
    kv_a: Tensor, // bf16 [kv_lora + rope, hidden]
    kv_a_norm: Tensor,
    kv_b: Tensor, // bf16 [heads x (nope + v), kv_lora]
    gate: Proj, // linear_gate [heads x v, hidden]
    o: Proj, // [hidden, heads x v]
    post_norm: Tensor, // post_proj_norm
    sinks: Tensor, // fp32 [heads] (learnable_sink_param)
    indexer: ?Indexer,
};

pub const Dense = struct { gate: Tensor, up: Tensor, down: Tensor, post_norm: Tensor };

pub const Moe = struct {
    router: Tensor, // bf16 [experts, hidden]
    bias: Tensor, // fp32 [experts] (e_score_correction_bias)
    latent_down: Tensor, // bf16 [latent, hidden] (routed_expert_down_proj)
    latent_up: Tensor, // bf16 [hidden, latent] (routed_expert_up_proj)
    gate_up: Tensor, // e4m3 [experts, 2 x width, latent]
    gate_up_s: Tensor, // bf16 [experts, 2 x width / 128, latent / 128]
    down: Tensor, // e4m3 [experts, latent, width]
    down_s: Tensor, // bf16 [experts, latent / 128, width / 128]
    shared: Dense, // bf16; its post_norm is the block's combine_output_norm
};

pub const Engram = struct {
    table: Tensor, // bf16 [rows, 96]: read by the packer, never uploaded
    map: Tensor, // int32 [120,818] lookup_table
    head_norms: [config.engram_heads]Tensor, // bf16 [96]
    value: Tensor, // bf16 [hidden, 16 x 96]
    value_bias: Tensor,
    keys: [4]Tensor, // bf16 [hidden, 16 x 96], one a stream
    key_biases: [4]Tensor,
    norm_keys: [4]Tensor,
    norm_queries: [4]Tensor,
    pre_gamma: Tensor,
    stream_scales: [4]Tensor,
};

pub const Layer = struct {
    input_norm: Tensor,
    post_attn_norm: Tensor, // post_attention_layernorm: the norm before the MLP
    hc_attn: Hc,
    hc_mlp: Hc,
    attn: Attn,
    mlp: union(enum) { dense: Dense, moe: Moe },
    engram: ?Engram,
};

/// The tensors outside the layers.
pub const Top = struct { embed: Tensor, norm: Tensor, head: Tensor, hc_head: Hc };

pub const Map = struct {
    gpa: std.mem.Allocator,
    top: Top,
    layers: []Layer,

    pub fn deinit(m: *Map) void {
        m.gpa.free(m.layers);
        m.* = undefined;
    }
};

const Names = struct {
    ck: *Checkpoint,
    i: usize,
    buf: [192]u8 = undefined,

    fn name(self: *Names, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.bufPrint(&self.buf, "model.layers.{d}." ++ fmt, .{self.i} ++ args);
    }

    fn t(self: *Names, dtype: core.checkpoint.DType, shape: []const usize, comptime fmt: []const u8, args: anytype) !Tensor {
        return self.ck.expect(try self.name(fmt, args), dtype, shape);
    }

    fn bf(self: *Names, shape: []const usize, comptime fmt: []const u8, args: anytype) !Tensor {
        return self.t(.bf16, shape, fmt, args);
    }

    fn f32s(self: *Names, shape: []const usize, comptime fmt: []const u8, args: anytype) !Tensor {
        return self.t(.f32, shape, fmt, args);
    }

    /// `module` [n, k] in whichever format its tensors are stored: bf16 alone, or e4m3 with bf16 scale_inv.
    fn proj(self: *Names, n: usize, k: usize, comptime module: []const u8) !Proj {
        const w = try self.ck.get(try self.name(module ++ ".weight", .{}));
        switch (w.dtype) {
            .bf16 => {
                if (self.ck.has(try self.name(module ++ ".weight_scale_inv", .{}))) return error.UnexpectedTensor;
                return .{ .bf16 = try self.bf(&.{ n, k }, module ++ ".weight", .{}) };
            },
            .f8_e4m3 => {
                const prefix = try self.name(module, .{});
                var own: [192]u8 = undefined;
                @memcpy(own[0..prefix.len], prefix);
                const h = try fp8.read(self.ck, own[0..prefix.len], n, k, .bf16, null);
                return .{ .fp8 = .{ .w = h.weight, .s = h.scale_inv } };
            },
            else => {
                std.log.err("{s}: {t}; this family reads bf16 or e4m3 projections", .{ try self.name(module ++ ".weight", .{}), w.dtype });
                return error.UnsupportedQuantization;
            },
        }
    }

    fn hc(self: *Names, c: Config, comptime site: []const u8) !Hc {
        const m = c.hc_mult;
        return .{
            .fn_ = try self.f32s(&.{ 2 * m, m * c.hidden }, site ++ ".hc_pre.hc_fn", .{}),
            .base = try self.f32s(&.{2 * m}, site ++ ".hc_pre.hc_base", .{}),
            .scale = try self.f32s(&.{2}, site ++ ".hc_pre.hc_scale", .{}),
        };
    }
};

fn attn(nm: *Names, c: Config) !Attn {
    const h = c.heads(nm.i);
    const d = c.hidden;
    const p = "self_attn.";
    return .{
        .q_a = try nm.bf(&.{ c.q_lora, d }, p ++ "q_a_proj.weight", .{}),
        .q_a_norm = try nm.bf(&.{c.q_lora}, p ++ "q_a_layernorm.weight", .{}),
        .q_b = try nm.proj(h.count * (h.nope + h.rope), c.q_lora, p ++ "q_b_proj"),
        .kv_a = try nm.bf(&.{ c.kv_lora + h.rope, d }, p ++ "kv_a_proj_with_mqa.weight", .{}),
        .kv_a_norm = try nm.bf(&.{c.kv_lora}, p ++ "kv_a_layernorm.weight", .{}),
        .kv_b = try nm.bf(&.{ h.count * (h.nope + h.v), c.kv_lora }, p ++ "kv_b_proj.weight", .{}),
        .gate = try nm.proj(h.count * h.v, d, p ++ "linear_gate"),
        .o = try nm.proj(d, h.count * h.v, p ++ "o_proj"),
        .post_norm = try nm.bf(&.{d}, p ++ "post_proj_norm.weight", .{}),
        .sinks = try nm.f32s(&.{h.count}, p ++ "learnable_sink_param", .{}),
        .indexer = if (c.kind[nm.i] == .indexed) try indexer(nm, c) else null,
    };
}

fn indexer(nm: *Names, c: Config) !Indexer {
    const p = "self_attn.compressed_indexer.";
    const n = c.index_dim;
    return .{
        .query = try nm.bf(&.{ c.index_heads * n, c.q_lora }, p ++ "query_projection.weight", .{}),
        .head_weights = try nm.bf(&.{ c.index_heads, c.hidden }, p ++ "head_weight_projection.weight", .{}),
        .token_key = try nm.bf(&.{ n, c.hidden }, p ++ "token_key_projection.weight", .{}),
        .token_key_norm = try nm.bf(&.{n}, p ++ "token_key_norm.weight", .{}),
        .values = try nm.f32s(&.{ n, n }, p ++ "key_value_projection.weight", .{}),
        .gates = try nm.f32s(&.{ n, n }, p ++ "key_gate_projection.weight", .{}),
        .key_norm = try nm.f32s(&.{n}, p ++ "key_norm.weight", .{}),
        .ape = try nm.f32s(&.{ c.index_ratio, n }, p ++ "intra_group_ape", .{}),
    };
}

fn moe(nm: *Names, c: Config) !Moe {
    const e = c.experts;
    const w = c.expert_width;
    const l = c.latent;
    const b = fp8.blocks;
    return .{
        .router = try nm.bf(&.{ e, c.hidden }, "mlp.gate.weight", .{}),
        .bias = try nm.f32s(&.{e}, "mlp.gate.e_score_correction_bias", .{}),
        .latent_down = try nm.bf(&.{ l, c.hidden }, "mlp.routed_expert_down_proj.weight", .{}),
        .latent_up = try nm.bf(&.{ c.hidden, l }, "mlp.routed_expert_up_proj.weight", .{}),
        .gate_up = try nm.t(.f8_e4m3, &.{ e, 2 * w, l }, "mlp.experts.gate_up_proj", .{}),
        .gate_up_s = try nm.bf(&.{ e, b(2 * w), b(l) }, "mlp.experts.gate_up_proj_scale_inv", .{}),
        .down = try nm.t(.f8_e4m3, &.{ e, l, w }, "mlp.experts.down_proj", .{}),
        .down_s = try nm.bf(&.{ e, b(l), b(w) }, "mlp.experts.down_proj_scale_inv", .{}),
        .shared = .{
            .gate = try nm.bf(&.{ c.shared_width, c.hidden }, "mlp.shared_experts.gate_proj.weight", .{}),
            .up = try nm.bf(&.{ c.shared_width, c.hidden }, "mlp.shared_experts.up_proj.weight", .{}),
            .down = try nm.bf(&.{ c.hidden, c.shared_width }, "mlp.shared_experts.down_proj.weight", .{}),
            .post_norm = try nm.bf(&.{c.hidden}, "mlp.combine_output_norm.weight", .{}),
        },
    };
}

fn engram(nm: *Names, c: Config, slot: usize) !Engram {
    const p = "engram_layer.";
    const d = c.hidden;
    const k = config.engram_heads * c.engram_dim;
    var rows: usize = 0;
    for (c.engram_sizes[slot]) |s| rows += s;
    var g: Engram = undefined;
    g.table = try nm.bf(&.{ rows, c.engram_dim }, p ++ "multi_head_embedding.embedding.weight", .{});
    g.map = try nm.t(.i32, &.{c.engram_map}, p ++ "lookup_table", .{});
    for (&g.head_norms, 0..) |*n, h| n.* = try nm.bf(&.{c.engram_dim}, p ++ "lookup_per_head_norms.{d}.weight", .{h});
    g.value = try nm.bf(&.{ d, k }, p ++ "value_proj.weight", .{});
    g.value_bias = try nm.bf(&.{d}, p ++ "value_proj.bias", .{});
    for (0..4) |s| {
        g.keys[s] = try nm.bf(&.{ d, k }, p ++ "key_projs.{d}.weight", .{s});
        g.key_biases[s] = try nm.bf(&.{d}, p ++ "key_projs.{d}.bias", .{s});
        g.norm_keys[s] = try nm.bf(&.{d}, p ++ "norm_key.{d}.weight", .{s});
        g.norm_queries[s] = try nm.bf(&.{d}, p ++ "norm_query.{d}.weight", .{s});
        g.stream_scales[s] = try nm.bf(&.{d}, p ++ "unified_per_hc_scales.{d}", .{s});
    }
    g.pre_gamma = try nm.bf(&.{d}, p ++ "unified_pre_gamma", .{});
    return g;
}

/// Layer `i`'s tensors (0-based), for a whole map or a per-layer tool.
pub fn layer(ck: *Checkpoint, c: Config, i: usize) !Layer {
    var nm: Names = .{ .ck = ck, .i = i };
    const d = c.hidden;
    return .{
        .input_norm = try nm.bf(&.{d}, "input_layernorm.weight", .{}),
        .post_attn_norm = try nm.bf(&.{d}, "post_attention_layernorm.weight", .{}),
        .hc_attn = try nm.hc(c, "hc_attn_layer"),
        .hc_mlp = try nm.hc(c, "hc_mlp_layer"),
        .attn = try attn(&nm, c),
        .mlp = if (c.dense[i]) .{ .dense = .{
            .gate = try nm.bf(&.{ c.dense_width, d }, "mlp.gate_proj.weight", .{}),
            .up = try nm.bf(&.{ c.dense_width, d }, "mlp.up_proj.weight", .{}),
            .down = try nm.bf(&.{ d, c.dense_width }, "mlp.down_proj.weight", .{}),
            .post_norm = try nm.bf(&.{d}, "mlp.post_fc2_norm.weight", .{}),
        } } else .{ .moe = try moe(&nm, c) },
        .engram = if (c.engramSlot(i)) |s| try engram(&nm, c, s) else null,
    };
}

pub fn top(ck: *Checkpoint, c: Config) !Top {
    const m = c.hc_mult;
    return .{
        .embed = try ck.expect("model.embed_tokens.weight", .bf16, &.{ c.vocab, c.hidden }),
        .norm = try ck.expect("model.norm.weight", .bf16, &.{c.hidden}),
        .head = try ck.expect("lm_head.weight", .bf16, &.{ c.vocab, c.hidden }),
        .hc_head = .{
            .fn_ = try ck.expect("model.hc_head.hc_head_fn", .f32, &.{ m, m * c.hidden }),
            .base = try ck.expect("model.hc_head.hc_head_base", .f32, &.{m}),
            .scale = try ck.expect("model.hc_head.hc_head_scale", .f32, &.{1}),
        },
    };
}

/// Every tensor the engine reads, the draft layers and image tower marked as left out on purpose.
pub fn mapped(gpa: std.mem.Allocator, ck: *Checkpoint, c: Config) !Map {
    const m: Map = .{ .gpa = gpa, .top = try top(ck, c), .layers = try gpa.alloc(Layer, c.layers) };
    errdefer gpa.free(m.layers);
    for (m.layers, 0..) |*l, i| l.* = try layer(ck, c, i);
    try ck.skip("model.dfly_layers.");
    try ck.skip("vit.");
    return m;
}

/// `mapped`, and a refusal if any tensor is left over (they are logged by name).
pub fn map(gpa: std.mem.Allocator, ck: *Checkpoint, c: Config) !Map {
    var m = try mapped(gpa, ck, c);
    errdefer m.deinit();
    if (ck.unused() != 0) {
        std.log.err("{d} checkpoint tensors are not Hy4-flash's", .{ck.unused()});
        return error.UnusedTensors;
    }
    return m;
}

test {
    _ = @import("names_test.zig");
}
