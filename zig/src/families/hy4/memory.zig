//! What Hy4-flash holds on a rank: weight bytes by the TP split, Engram rows, KV a token and the sliding rings.
const config = @import("config.zig");
const part = @import("partition.zig");

const Config = config.Config;

/// A 128 x 128 block scale's bytes: bf16 as the checkpoint stores it, fp32 as the GPU packs hold it.
pub const Scale = enum(usize) { stored = 2, packed_ = 4 };

pub const Format = enum { bf16, fp8 };

/// An [n, k] projection's bytes: bf16, or e4m3 codes plus a scale a block.
pub fn projBytes(f: Format, n: usize, k: usize, scale: Scale) usize {
    return switch (f) {
        .bf16 => n * k * 2,
        .fp8 => n * k + (n + part.block - 1) / part.block * ((k + part.block - 1) / part.block) * @backingInt(scale),
    };
}

/// The release's formats: layer 0's attention is all bf16; later layers keep q_b, linear_gate and o_proj in e4m3.
fn attnFormat(layer: usize) Format {
    return if (layer == 0) .bf16 else .fp8;
}

fn attnBytes(c: Config, layer: usize, rank: usize, world: usize, scale: Scale) !usize {
    const d = c.hidden;
    const a = try part.Attn.of(c, layer, rank, world);
    const f = attnFormat(layer);
    var n: usize = 2 * (c.q_lora * d + c.q_lora + (c.kv_lora + c.heads(layer).rope) * d + c.kv_lora + d); // q_a, kv_a, norms
    n += 2 * a.kv_rows.len * c.kv_lora + 4 * a.heads.len; // kv_b, sinks
    n += projBytes(f, a.q_rows.len, c.q_lora, scale) + projBytes(f, a.gate_rows.len, d, scale) + projBytes(f, d, a.o_cols.len, scale);
    if (c.kind[layer] == .indexed) { // the indexer, replicated
        const t = c.index_dim;
        n += 2 * (c.index_heads * d + c.index_heads * t * c.q_lora + t + t * d) + 4 * (c.index_ratio * t + 2 * t * t + t);
    }
    return n;
}

fn ffnBytes(c: Config, layer: usize, rank: usize, world: usize, scale: Scale) !usize {
    const d = c.hidden;
    if (c.dense[layer]) return 2 * 3 * (try part.ffn(c.dense_width, rank, world)).len * d + 2 * d; // and post_fc2_norm
    const w = (try part.expert(c, rank, world)).len;
    const l = c.latent;
    var n: usize = 2 * c.experts * d + 4 * c.experts + 2 * 2 * l * d; // router, its bias, latent down and up
    n += 2 * 3 * (try part.ffn(c.shared_width, rank, world)).len * d + 2 * d; // shared expert, combine_output_norm
    return n + c.experts * (projBytes(.fp8, 2 * w, l, scale) + projBytes(.fp8, l, w, scale));
}

/// Engram's tensors besides its table: the token map, per-head norms, the value and key projections, the gate's norms.
fn engramBytes(c: Config, rank: usize, world: usize) !usize {
    const d = c.hidden;
    const e = try part.Engram.of(c, rank, world);
    const projections = 1 + c.hc_mult; // value_proj and a key_proj a stream
    return 4 * c.engram_map + 2 * e.heads.len * c.engram_dim + projections * 2 * (d * e.cols.len + d) + 2 * d * (3 * c.hc_mult + 1);
}

/// One decoder layer on a rank, Engram's table excluded.
pub fn layerBytes(c: Config, layer: usize, rank: usize, world: usize, scale: Scale) !usize {
    const d = c.hidden;
    const hc = c.hc_mult;
    var n = 2 * 2 * d + 2 * (4 * 2 * hc * hc * d + 4 * 2 * hc + 4 * 2); // the two sandwich norms, iHC pre and post twice
    n += try attnBytes(c, layer, rank, world, scale) + try ffnBytes(c, layer, rank, world, scale);
    if (c.engramSlot(layer) != null) n += try engramBytes(c, rank, world);
    return n;
}

/// Embeddings (replicated), lm_head (split by vocabulary), the final norm and hc_head.
pub fn topBytes(c: Config, rank: usize, world: usize) !usize {
    const d = c.hidden;
    return 2 * c.vocab * d + 2 * (try part.vocab(c, rank, world)).len * d + 2 * d + 4 * (c.hc_mult * c.hc_mult * d + c.hc_mult + 1);
}

/// Every weight a rank holds in GPU memory: everything but the Engram tables.
pub fn weightBytes(c: Config, rank: usize, world: usize, scale: Scale) !usize {
    var n = try topBytes(c, rank, world);
    for (0..c.layers) |l| n += try layerBytes(c, l, rank, world, scale);
    return n;
}

/// A rank's Engram rows across the four layers: bf16 as stored, or packed (e4m3 codes and an fp32 scale a row).
pub fn engramTableBytes(c: Config, rank: usize, world: usize, row: enum { stored, packed_ }) !usize {
    const e = try part.Engram.of(c, rank, world);
    const per_row = if (row == .stored) 2 * c.engram_dim else c.engram_dim + 4;
    var n: usize = 0;
    for (0..config.engram_layers) |s| n += e.rows(c, s).len * per_row;
    return n;
}

/// KV a token on each rank: the indexed layers' latent and roped columns (bf16, whole on both ranks) and token key.
pub fn tokenBytes(c: Config) usize {
    var n: usize = 0;
    for (c.kind[0..c.layers]) |k| if (k == .indexed) {
        n += 2 * (c.kv_lora + c.indexed.rope + c.index_dim);
    };
    return n;
}

/// A stream's sliding rings: `window` latent rows in each sliding layer, whatever the context.
pub fn ringBytes(c: Config) usize {
    var n: usize = 0;
    for (c.kind[0..c.layers]) |k| if (k == .sliding) {
        n += 2 * (c.kv_lora + c.sliding.rope) * c.window;
    };
    return n;
}

test {
    _ = @import("memory_test.zig");
}
