//! Hy4-flash's dimensions from config.json (`text_config`), and the layouts this family refuses by name.
const std = @import("std");
const eos_ids = @import("core").tokenizer.eos;

pub const max_layers = 64;
pub const max_eos = 4;
/// Engram memories: layers carrying one, n-gram orders (2..3) and hash heads an order.
pub const engram_layers = 4;
pub const engram_orders = 2;
pub const engram_heads_per_order = 8;
pub const engram_heads = engram_orders * engram_heads_per_order;

/// Why a check refused the checkpoint, kept for the caller's log line (tests read it instead).
pub const Why = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    pub fn set(self: *Why, comptime fmt: []const u8, args: anytype) void {
        var w: std.Io.Writer = .fixed(&self.buf);
        w.print(fmt, args) catch {};
        self.len = w.end;
    }

    pub fn text(self: *const Why) []const u8 {
        return self.buf[0..self.len];
    }
};

/// The attention a layer runs: keys the compressed indexer recalls, or a sliding window.
pub const Attention = enum { indexed, sliding };

/// One attention kind's head geometry (gated MLA: query/key nope + rope columns, value width).
pub const Heads = struct { count: usize, nope: usize, rope: usize, v: usize, theta: f64 };

pub const Config = struct {
    layers: usize,
    hidden: usize,
    vocab: usize,
    max_positions: usize = 1_048_576,
    q_lora: usize,
    kv_lora: usize,
    indexed: Heads,
    sliding: Heads,
    window: usize, // keys a sliding row sees, its own included
    kind: [max_layers]Attention = @splat(.sliding),
    dense: [max_layers]bool = @splat(false), // a dense FFN instead of the MoE
    dense_width: usize,
    experts: usize,
    top_k: usize,
    expert_width: usize, // a routed expert's intermediate width
    latent: usize, // the routed experts' input and output width (LatentMoE)
    shared_width: usize, // the shared expert's intermediate width
    routed_scale: f32,
    swiglu_limit: f32,
    norm_topk: bool,
    hc_mult: usize,
    hc_eps: f32,
    hc_magnitude: f32,
    eps: f32,
    mrope: [3]usize,
    index_heads: usize,
    index_dim: usize,
    index_ratio: usize, // tokens a pooled group holds
    index_groups: usize, // groups a query recalls, its own included
    engram_ids: [engram_layers]u32, // one-based layer ids
    engram_sizes: [engram_layers][engram_heads]u64, // each hash head's prime, order-major
    engram_dim: usize, // values a head's row holds
    engram_pad: u32,
    engram_seed: u64,
    engram_map: usize, // lookup_table entries
    image_start_token: u32 = 120118,
    image_token: u32 = 120120,
    image_end_token: u32 = 120119,
    image_newline_token: u32 = 120121,
    video_token: u32 = 120124,
    vision: bool, // the checkpoint carries an image tower (refused when asked for, v1)
    dfly_layers: usize, // DFly2 draft layers in the checkpoint (not loaded, v1)
    eos: [max_eos]u32 = @splat(0),
    eos_count: usize = 0,

    pub fn heads(self: Config, layer: usize) Heads {
        return if (self.kind[layer] == .indexed) self.indexed else self.sliding;
    }

    /// Keys an indexed query attends at most: its recalled groups' tokens.
    pub fn recall(self: Config) usize {
        return self.index_groups * self.index_ratio;
    }

    /// Whether 0-based layer `layer` carries an Engram memory, and which of the four.
    pub fn engramSlot(self: Config, layer: usize) ?usize {
        for (self.engram_ids, 0..) |id, s| if (id == layer + 1) return s;
        return null;
    }

    pub fn isEos(self: Config, token: u32) bool {
        for (self.eos[0..self.eos_count]) |e| if (e == token) return true;
        return false;
    }

    /// Whether a token position contributes zero Engram residual (padding, image or video placeholders).
    pub fn isEngramMasked(self: Config, token: u32) bool {
        return token == self.engram_pad or token == self.image_token or token == self.video_token;
    }

    /// End tokens from generation_config.json's text, after config.json's own.
    pub fn addEos(self: *Config, allocator: std.mem.Allocator, generation: []const u8) !void {
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, generation, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.BadEos;
        const ids = try eos_ids.merge(allocator, self.eos[0..self.eos_count], parsed.value, self.vocab);
        defer allocator.free(ids);
        try setEos(self, ids);
    }

    /// `dir`/config.json, checked for the family's shapes, and generation_config.json's end tokens when present.
    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(text);
        var why: Why = .{};
        var c = parse(gpa, text, &why) catch |e| {
            if (why.len > 0) std.log.err("{s}: {s}", .{ path, why.text() });
            return e;
        };
        checkShapes(c, &why) catch |e| {
            std.log.err("{s}: {s}", .{ path, why.text() });
            return e;
        };
        const gen_path = try std.fs.path.join(gpa, &.{ dir, "generation_config.json" });
        defer gpa.free(gen_path);
        const gen = std.Io.Dir.cwd().readFileAlloc(io, gen_path, gpa, .limited(1 << 20)) catch |e| switch (e) {
            error.FileNotFound => return c,
            else => return e,
        };
        defer gpa.free(gen);
        try c.addEos(gpa, gen);
        return c;
    }
};

const Obj = std.json.ObjectMap;

fn int(obj: Obj, key: []const u8, why: *Why) !usize {
    const v = obj.get(key) orelse {
        why.set("config.json's text_config has no {s}", .{key});
        return error.BadConfig;
    };
    if (v != .integer or v.integer <= 0) {
        why.set("config.json's {s} is not a positive integer", .{key});
        return error.BadConfig;
    }
    return @intCast(v.integer);
}

fn number(obj: Obj, key: []const u8, why: *Why) !f64 {
    const v = obj.get(key) orelse {
        why.set("config.json's text_config has no {s}", .{key});
        return error.BadConfig;
    };
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => {
            why.set("config.json's {s} is not a number", .{key});
            return error.BadConfig;
        },
    };
}

fn flag(obj: Obj, key: []const u8, default: bool) !bool {
    const v = obj.get(key) orelse return default;
    if (v != .bool) return error.BadConfig;
    return v.bool;
}

fn string(v: ?std.json.Value, want: []const u8) bool {
    const s = v orelse return false;
    return s == .string and std.mem.eql(u8, s.string, want);
}

fn optToken(obj: Obj, key: []const u8, default: u32) !u32 {
    const v = obj.get(key) orelse return default;
    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return error.BadConfig;
    return @intCast(v.integer);
}

fn setEos(c: *Config, ids: []const u32) !void {
    if (ids.len > max_eos) return error.TooManyEos;
    @memcpy(c.eos[0..ids.len], ids);
    c.eos_count = ids.len;
}

/// The layer kinds, under either of transformers' spellings of the indexed kind.
fn attention(s: []const u8) ?Attention {
    if (std.mem.eql(u8, s, "deepseek_sparse_attention") or std.mem.eql(u8, s, "indexed_attention")) return .indexed;
    if (std.mem.eql(u8, s, "sliding_attention")) return .sliding;
    return null;
}

/// Plain RoPE of one attention kind (`rope_parameters` keyed by either spelling): its theta and M-RoPE sections.
fn rope(text: Obj, kind: Attention, mrope: *[3]usize, why: *Why) !f64 {
    const params = text.get("rope_parameters") orelse {
        why.set("config.json has no rope_parameters", .{});
        return error.BadConfig;
    };
    if (params != .object) return error.BadConfig;
    const keys: []const []const u8 = if (kind == .indexed) &.{ "indexed_attention", "deepseek_sparse_attention" } else &.{"sliding_attention"};
    const p: std.json.Value = for (keys) |k| {
        if (params.object.get(k)) |v| break v;
    } else {
        why.set("rope_parameters has no entry for the {t} layers", .{kind});
        return error.BadConfig;
    };
    if (p != .object) return error.BadConfig;
    if (p.object.get("rope_type")) |t| if (!string(t, "default")) {
        why.set("rope_parameters asks for {f} RoPE on the {t} layers; Hy4-flash runs the default kind", .{ std.json.fmt(t, .{}), kind });
        return error.UnsupportedRope;
    };
    if (p.object.get("mrope_section")) |m| {
        if (m != .array or m.array.items.len != 3) return error.BadConfig;
        for (m.array.items, 0..) |x, i| {
            if (x != .integer or x.integer < 0) return error.BadConfig;
            mrope[i] = @intCast(x.integer);
        }
    } else mrope.* = .{ 12, 10, 10 };
    return number(p.object, "rope_theta", why);
}

/// Hy4's Engram block: four layers of 2- and 3-gram hash heads, eight an order.
fn engram(c: *Config, text: Obj, why: *Why) !void {
    const ids = text.get("engram_layer_ids") orelse {
        why.set("config.json has no engram_layer_ids; this family reads Hy4-flash's Engram memories", .{});
        return error.BadConfig;
    };
    if (ids != .array or ids.array.items.len != engram_layers) {
        why.set("engram_layer_ids names {d} layers; Hy4-flash has {d}", .{ if (ids == .array) ids.array.items.len else 0, engram_layers });
        return error.UnsupportedEngram;
    }
    if (try int(text, "engram_max_ngram_size", why) != engram_orders + 1 or try int(text, "engram_n_head_per_ngram", why) != engram_heads_per_order) {
        why.set("Engram n-grams up to {d}, {d} heads an order: this family hashes 2- and 3-grams, 8 heads each", .{ try int(text, "engram_max_ngram_size", why), try int(text, "engram_n_head_per_ngram", why) });
        return error.UnsupportedEngram;
    }
    c.engram_dim = try int(text, "engram_n_embed_per_ngram", why) / engram_heads_per_order;
    if (!try flag(text, "engram_fp8_embedding", true)) {
        why.set("engram_fp8_embedding is false; this family reads rows through Hy4-flash's FP8 round trip", .{});
        return error.UnsupportedEngram;
    }
    c.engram_pad = @intCast(try int(text, "engram_pad_id", why));
    c.engram_seed = if (text.get("engram_seed")) |s| (if (s == .integer and s.integer >= 0) @intCast(s.integer) else return error.BadConfig) else 0;
    c.engram_map = if (text.get("engram_lookup_table_size")) |_| try int(text, "engram_lookup_table_size", why) else 120818;
    const sizes = text.get("engram_layer_head_vocab_sizes") orelse return error.BadConfig;
    if (sizes != .object) return error.BadConfig;
    for (ids.array.items, 0..) |v, s| {
        if (v != .integer or v.integer < 1 or v.integer > c.layers) return error.BadConfig;
        c.engram_ids[s] = @intCast(v.integer);
        var key: [16]u8 = undefined;
        const per = sizes.object.get(try std.fmt.bufPrint(&key, "{d}", .{v.integer})) orelse {
            why.set("engram_layer_head_vocab_sizes has no entry for layer {d}", .{v.integer});
            return error.BadConfig;
        };
        if (per != .array or per.array.items.len != engram_orders) return error.BadConfig;
        for (per.array.items, 0..) |order, o| {
            if (order != .array or order.array.items.len != engram_heads_per_order) return error.BadConfig;
            for (order.array.items, 0..) |x, h| {
                if (x != .integer or x.integer <= 0) return error.BadConfig;
                c.engram_sizes[s][o * engram_heads_per_order + h] = @intCast(x.integer);
            }
        }
    }
}

/// config.json's text; a refusal's reason goes to `why`.
pub fn parse(allocator: std.mem.Allocator, json_text: []const u8, why: *Why) !Config {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadConfig;
    const top = parsed.value.object;
    if (!string(top.get("model_type"), "hy_v4_flash")) {
        why.set("model_type {f} is not hy_v4_flash", .{std.json.fmt(top.get("model_type") orelse .null, .{})});
        return error.NotHy4Flash;
    }
    const tv = top.get("text_config") orelse {
        why.set("config.json has no text_config", .{});
        return error.BadConfig;
    };
    if (tv != .object) return error.BadConfig;
    const text = tv.object;
    const qv = text.get("quantization_config") orelse top.get("quantization_config");
    const q = if (qv) |v| (if (v == .object) v.object else null) else null;
    const fp8 = if (q) |m| string(m.get("quant_method"), "fp8") and (m.get("fmt") == null or string(m.get("fmt"), "e4m3")) else false;
    const blocks = if (q) |m| (if (m.get("weight_block_size")) |x| x == .array and x.array.items.len == 2 and
        x.array.items[0] == .integer and x.array.items[0].integer == 128 and
        x.array.items[1] == .integer and x.array.items[1].integer == 128 else false) else false;
    if (!fp8 or !blocks) {
        why.set("Hy4-flash on CUDA reads its block-FP8 release (e4m3, 128 x 128 blocks)", .{});
        return error.UnsupportedQuantization;
    }
    if (!string(text.get("hidden_act") orelse std.json.Value{ .string = "silu" }, "silu") or try flag(text, "attention_bias", false)) {
        why.set("hidden_act other than silu, or attention biases: not Hy4-flash's layout", .{});
        return error.UnsupportedLayout;
    }
    var c = Config{
        .layers = try int(text, "num_hidden_layers", why),
        .hidden = try int(text, "hidden_size", why),
        .vocab = try int(text, "vocab_size", why),
        .max_positions = if (text.get("max_position_embeddings")) |_| try int(text, "max_position_embeddings", why) else 1_048_576,
        .q_lora = try int(text, "q_lora_rank", why),
        .kv_lora = try int(text, "kv_lora_rank", why),
        .indexed = .{ .count = try int(text, "num_attention_heads", why), .nope = try int(text, "qk_nope_head_dim", why), .rope = try int(text, "qk_rope_head_dim", why), .v = try int(text, "v_head_dim", why), .theta = 0 },
        .sliding = .{ .count = try int(text, "swa_num_attention_heads", why), .nope = try int(text, "swa_qk_nope_head_dim", why), .rope = try int(text, "swa_qk_rope_head_dim", why), .v = try int(text, "swa_v_head_dim", why), .theta = 0 },
        .window = try int(text, "sliding_window", why),
        .dense_width = try int(text, "intermediate_size", why),
        .experts = try int(text, "n_routed_experts", why),
        .top_k = try int(text, "num_experts_per_tok", why),
        .expert_width = try int(text, "moe_intermediate_size", why),
        .latent = 0,
        .shared_width = try int(text, "moe_intermediate_size", why) * try int(text, "n_shared_experts", why),
        .routed_scale = @floatCast(try number(text, "routed_scaling_factor", why)),
        .swiglu_limit = @floatCast(try number(text, "swiglu_limit", why)),
        .norm_topk = try flag(text, "norm_topk_prob", true),
        .hc_mult = try int(text, "hc_mult", why),
        .hc_eps = @floatCast(try number(text, "hc_eps", why)),
        .hc_magnitude = @floatCast(try number(text, "hc_magnitude", why)),
        .eps = @floatCast(try number(text, "rms_norm_eps", why)),
        .mrope = undefined,
        .index_heads = try int(text, "compressed_indexer_num_attention_heads", why),
        .index_dim = try int(text, "compressed_indexer_head_dim", why),
        .index_ratio = try int(text, "compressed_indexer_compress_ratio", why),
        .index_groups = try int(text, "compressed_indexer_group_topk", why),
        .engram_ids = undefined,
        .engram_sizes = undefined,
        .engram_dim = 0,
        .engram_pad = 0,
        .engram_seed = 0,
        .engram_map = 0,
        .image_start_token = try optToken(top, "image_start_token_id", 120118),
        .image_token = try optToken(top, "image_token_id", 120120),
        .image_end_token = try optToken(top, "image_end_token_id", 120119),
        .image_newline_token = try optToken(top, "image_newline_token_id", 120121),
        .video_token = try optToken(top, "video_token_id", 120124),
        .vision = top.get("vision_config") != null,
        .dfly_layers = if (top.get("dfly_num_hidden_layers")) |v| (if (v == .integer and v.integer >= 0) @intCast(v.integer) else 0) else 0,
    };
    if (c.layers > max_layers) {
        why.set("{d} layers; this build holds at most {d}", .{ c.layers, max_layers });
        return error.BadConfig;
    }
    const latent = text.get("routed_expert_hidden_size") orelse std.json.Value{ .null = {} };
    if (latent != .integer or latent.integer <= 0) {
        why.set("routed_expert_hidden_size is not set: this family runs Hy4-flash's latent experts only", .{});
        return error.UnsupportedLayout;
    }
    c.latent = @intCast(latent.integer);
    const groups = if (text.get("n_group")) |_| try int(text, "n_group", why) else 1;
    const keep = if (text.get("topk_group")) |_| try int(text, "topk_group", why) else 1;
    if (groups != 1 or keep != 1) {
        why.set("expert groups ({d}, keeping {d}): this family routes over one group", .{ groups, keep });
        return error.UnsupportedRouting;
    }
    var mrope_sliding: [3]usize = undefined;
    c.indexed.theta = try rope(text, .indexed, &c.mrope, why);
    c.sliding.theta = try rope(text, .sliding, &mrope_sliding, why);
    if (!std.mem.eql(usize, &c.mrope, &mrope_sliding)) {
        why.set("the two attention kinds' mrope_section differ", .{});
        return error.UnsupportedRope;
    }
    inline for (.{ .{ "layer_types", "attention" }, .{ "mlp_layer_types", "mlp" } }) |pair| {
        const v = text.get(pair[0]) orelse {
            why.set("config.json has no {s}", .{pair[0]});
            return error.BadConfig;
        };
        if (v != .array or v.array.items.len != c.layers) {
            why.set("{s} does not name each of the {d} layers", .{ pair[0], c.layers });
            return error.BadConfig;
        }
        for (v.array.items, 0..) |t, i| {
            if (t != .string) return error.BadConfig;
            if (comptime std.mem.eql(u8, pair[1], "attention")) {
                c.kind[i] = attention(t.string) orelse {
                    why.set("layer {d} is {s}; Hy4-flash has indexed and sliding attention layers", .{ i, t.string });
                    return error.UnsupportedLayer;
                };
            } else if (std.mem.eql(u8, t.string, "dense")) {
                c.dense[i] = true;
            } else if (!std.mem.eql(u8, t.string, "sparse")) {
                why.set("layer {d}'s MLP is {s}; Hy4-flash has dense and sparse ones", .{ i, t.string });
                return error.UnsupportedLayer;
            }
        }
    }
    try engram(&c, text, why);
    const ids = try eos_ids.resolve(allocator, parsed.value, .null, c.vocab);
    defer allocator.free(ids);
    try setEos(&c, ids);
    return c;
}

/// What this family's kernels are built for; anything else is refused before a weight is read.
pub fn checkShapes(c: Config, why: *Why) !void {
    for ([_]Heads{ c.indexed, c.sliding }) |h| if (h.nope != 192 or h.rope != 64 or h.v != 256 or h.count % 16 != 0 or h.count > 64) {
        why.set("heads {d} of query/key {d}+{d}, value {d}: this family's MLA takes 192+64, value 256, 16 to 64 heads in sixteens", .{ h.count, h.nope, h.rope, h.v });
        return error.UnsupportedShapes;
    };
    if (c.kv_lora != 512 or c.q_lora % 128 != 0 or c.hidden % 128 != 0) {
        why.set("kv_lora {d}, q_lora {d}, hidden {d}: this family takes a 512-wide latent and whole 128 blocks", .{ c.kv_lora, c.q_lora, c.hidden });
        return error.UnsupportedShapes;
    }
    if (c.latent % 128 != 0 or c.expert_width % 128 != 0 or c.dense_width % 128 != 0 or c.experts > 4096 or c.top_k == 0 or c.top_k > 64 or c.top_k > c.experts) {
        why.set("{d} experts top {d}, widths {d} / {d} / {d}: outside what the expert and routing kernels take", .{ c.experts, c.top_k, c.latent, c.expert_width, c.dense_width });
        return error.UnsupportedShapes;
    }
    if (c.hc_mult != 4 or c.index_dim != 128 or c.index_ratio != 4 or c.index_heads != 16 or c.window < 2 or c.engram_dim != 96) {
        why.set("streams {d}, indexer {d} heads of {d} over groups of {d}, window {d}, Engram rows of {d}: not Hy4-flash's", .{ c.hc_mult, c.index_heads, c.index_dim, c.index_ratio, c.window, c.engram_dim });
        return error.UnsupportedShapes;
    }
    if (c.mrope[0] + c.mrope[1] + c.mrope[2] != c.indexed.rope / 2) {
        why.set("mrope_section sums to {d}, not half the 64 roped columns", .{c.mrope[0] + c.mrope[1] + c.mrope[2]});
        return error.UnsupportedRope;
    }
}

test {
    _ = @import("config_test.zig");
}
