//! Replicated device weights for Hy4-flash's compressed indexer (docs/recipes/hy4-flash.md).
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const names = @import("names.zig");
const partition = @import("partition.zig");
const mla_w = @import("cuda_mla_weights.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const Share = partition.Share;

pub const heads = 16;
pub const dim = 128;
pub const ratio = 4;
pub const groups = 512;
pub const rope = 64;
pub const rope_pairs = rope / 2;

/// Exact op-for-op match to `(16^-0.5 * 128^-0.5)` in HYV4FlashCompressedIndexer.
pub const head_weight_scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(heads * dim))));

/// Refuses any indexer geometry outside 16 heads of 128, 4:1 pooling, 512 groups, 64 roped dims, and 64-aligned inputs.
pub fn geometry(c: config.Config) !void {
    if (c.index_heads != heads or c.index_dim != dim or c.index_ratio != ratio or c.index_groups != groups)
        return error.IndexGeometry;
    if (c.indexed.rope != rope or c.q_lora % cuda.qmmf.group != 0 or c.hidden % cuda.qmmf.group != 0)
        return error.IndexGeometry;
}

/// One indexed layer's compressed-indexer weights on device (replicated on every rank).
pub const Weights = struct {
    q_lora: usize,
    hidden: usize,
    eps: f32,
    query: mla_w.Packed, // [2048, q_lora] bf16 packed for lane matmul
    token_key: mla_w.Packed, // [128, hidden] bf16 packed for lane matmul
    head_weights: DeviceBuffer, // [16, hidden] bf16 row-major for dense_rows.rowsScaled
    head_scale: DeviceBuffer, // [16] fp32 filled with head_weight_scale
    token_key_norm: DeviceBuffer, // [128] bf16
    values: DeviceBuffer, // [128, 128] fp32
    gates: DeviceBuffer, // [128, 128] fp32
    key_norm: DeviceBuffer, // [128] fp32
    ape: DeviceBuffer, // [4, 128] fp32
    freqs: DeviceBuffer, // [32] fp32

    pub fn load(d: *const cuda.Driver, gpa: std.mem.Allocator, t: names.Indexer, c: config.Config) !Weights {
        try geometry(c);
        const whole = struct {
            fn of(n: usize) Share {
                return .{ .lo = 0, .len = n };
            }
        }.of;
        var w: Weights = undefined;
        w.q_lora = c.q_lora;
        w.hidden = c.hidden;
        w.eps = c.eps;
        w.query = try mla_w.pack(d, gpa, .{ .bf16 = t.query }, heads * dim, c.q_lora, whole(heads * dim), whole(c.q_lora));
        errdefer w.query.deinit();
        w.query.w = try cuda.qmmf.pinned(w.query.w, 1);
        w.token_key = try mla_w.pack(d, gpa, .{ .bf16 = t.token_key }, dim, c.hidden, whole(dim), whole(c.hidden));
        errdefer w.token_key.deinit();
        w.head_weights = try DeviceBuffer.fromHost(d, t.head_weights.bytes);
        errdefer w.head_weights.free();
        const hs: [heads]f32 = @splat(head_weight_scale);
        w.head_scale = try DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(&hs));
        errdefer w.head_scale.free();
        w.token_key_norm = try DeviceBuffer.fromHost(d, t.token_key_norm.bytes);
        errdefer w.token_key_norm.free();
        w.values = try DeviceBuffer.fromHost(d, t.values.bytes);
        errdefer w.values.free();
        w.gates = try DeviceBuffer.fromHost(d, t.gates.bytes);
        errdefer w.gates.free();
        w.key_norm = try DeviceBuffer.fromHost(d, t.key_norm.bytes);
        errdefer w.key_norm.free();
        w.ape = try DeviceBuffer.fromHost(d, t.ape.bytes);
        errdefer w.ape.free();
        const f = cuda.mla.ropeFreqs(c.indexed.theta);
        w.freqs = try DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(&f));
        return w;
    }

    pub fn deinit(w: *Weights) void {
        w.query.deinit();
        w.token_key.deinit();
        for ([_]*DeviceBuffer{
            &w.head_weights,
            &w.head_scale,
            &w.token_key_norm,
            &w.values,
            &w.gates,
            &w.key_norm,
            &w.ape,
            &w.freqs,
        }) |b| b.free();
    }
};

test "head_weight_scale matches (16 * 128)^-0.5 to 0 bits" {
    try std.testing.expectEqual(@as(f32, 0.02209708691207961), head_weight_scale);
}

test "indexer geometry admits Hy4-flash and refuses mismatched head or ratio sizes" {
    var c: config.Config = undefined;
    c.index_heads = heads;
    c.index_dim = dim;
    c.index_ratio = ratio;
    c.index_groups = groups;
    c.indexed = .{ .count = 32, .nope = 192, .rope = rope, .v = 256, .theta = 1e7 };
    c.q_lora = 1024;
    c.hidden = 3072;
    try geometry(c);
    c.index_heads = 32;
    try std.testing.expectError(error.IndexGeometry, geometry(c));
    c.index_heads = heads;
    c.index_ratio = 8;
    try std.testing.expectError(error.IndexGeometry, geometry(c));
    c.index_ratio = ratio;
    c.q_lora = 1000;
    try std.testing.expectError(error.IndexGeometry, geometry(c));
}
