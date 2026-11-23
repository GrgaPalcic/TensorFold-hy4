//! A rank's gated-MLA weights for one Hy4-flash layer on the device: whole heads by partition.zig, formats as stored.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const names = @import("names.zig");
const partition = @import("partition.zig");

const qmmf = cuda.qmmf;
const fp8 = cuda.fp8;
const DeviceBuffer = cuda.DeviceBuffer;
const Share = partition.Share;

pub const latent = 512; // kv_lora: the cached latent
pub const rope = 64; // roped columns a q head and the shared key carry
pub const nope = 192;
pub const v_width = 256;
pub const q_head = nope + rope; // a q_b head's rows
pub const kv_head = nope + v_width; // a kv_b head's rows: key, then value
pub const heads_tile = 8; // tf_hy4_mla_attn's heads a block: a rank's heads come in eights

/// Refuses any layer geometry at `world` outside latent 512, nope 192, rope 64, v 256, and whole 8-head rank tiles.
pub fn geometry(c: config.Config, layer: usize, world: usize) !void {
    const h = c.heads(layer);
    if (c.kv_lora != latent or h.rope != rope or h.nope != nope or h.v != v_width) return error.MlaGeometry;
    if (h.count % (world * heads_tile) != 0) return error.MlaHeadsSplit;
}

/// A projection packed for the lane matmul: bf16 words, or e4m3 codes and their fp32 scale tiles.
pub const Packed = struct {
    w: qmmf.Weight,
    codes: DeviceBuffer,
    scales: ?DeviceBuffer = null,

    pub fn deinit(p: *Packed) void {
        p.codes.free();
        if (p.scales) |*s| s.free();
    }
};

/// Rows `rows` and columns `cols` of a row-major [n_all, k_all] matrix of `elem`-byte values, copied out.
pub fn block(gpa: std.mem.Allocator, bytes: []const u8, k_all: usize, elem: usize, rows: Share, cols: Share) ![]u8 {
    const out = try gpa.alloc(u8, rows.len * cols.len * elem);
    for (0..rows.len) |r| {
        const src = ((rows.lo + r) * k_all + cols.lo) * elem;
        @memcpy(out[r * cols.len * elem ..][0 .. cols.len * elem], bytes[src..][0 .. cols.len * elem]);
    }
    return out;
}

/// `p`'s rows and columns of [n_all, k_all], packed as stored: bf16 as words, e4m3 with its blocks' scales widened.
pub fn pack(d: *const cuda.Driver, gpa: std.mem.Allocator, p: names.Proj, n_all: usize, k_all: usize, rows: Share, cols: Share) !Packed {
    const n = rows.len;
    const k = cols.len;
    if (k % qmmf.group != 0) return error.ProjectionShape;
    switch (p) {
        .bf16 => |t| {
            const src = try block(gpa, t.bytes, k_all, 2, rows, cols);
            defer gpa.free(src);
            const words = try gpa.alloc(u16, n * k);
            defer gpa.free(words);
            @memcpy(std.mem.sliceAsBytes(words), src);
            const npad = (n + 127) / 128 * 128;
            const out = try gpa.alloc(u16, npad * k);
            defer gpa.free(out);
            qmmf.packBf16(out, words, n, k);
            const codes = try DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(out));
            return .{ .w = .{ .mode = .bf16, .codes = codes.ptr, .scales = 0, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(npad) }, .codes = codes };
        },
        .fp8 => |f| {
            const src = try block(gpa, f.w.bytes, k_all, 1, rows, cols);
            defer gpa.free(src);
            const out = try gpa.alloc(u8, fp8.padded(n) * k);
            defer gpa.free(out);
            fp8.packCodes(out, src, n, k);
            const inv = try gpa.alloc(f32, f.s.numel());
            defer gpa.free(inv);
            try core.quant.fp8.widen(f.s, inv, true);
            const tiles = try gpa.alloc(f32, fp8.padded(n) * (k / fp8.group));
            defer gpa.free(tiles);
            fp8.packScalesAt(tiles, inv, n_all, k_all, rows.lo, cols.lo, n, k);
            var codes = try DeviceBuffer.fromHost(d, out);
            errdefer codes.free();
            const scales = try DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(tiles));
            return .{ .w = fp8.weight(codes.ptr, scales.ptr, n, k), .codes = codes, .scales = scales };
        },
    }
}

/// Packs slice `[row_lo .. row_lo + row_len)` of `[a (n0, k), b (n1, k)]` bf16 into one `qmmf` BF16 weight.
fn packBf16PairSlice(d: *const cuda.Driver, gpa: std.mem.Allocator, a: []const u8, n0: usize, b: []const u8, n1: usize, k: usize, row_lo: usize, row_len: usize) !Packed {
    if (k % qmmf.group != 0 or row_lo + row_len > n0 + n1) return error.ProjectionShape;
    const words = try gpa.alloc(u16, row_len * k);
    defer gpa.free(words);
    var dst_r: usize = 0;
    for (row_lo..row_lo + row_len) |r| {
        const src = if (r < n0) a[r * k * 2 ..][0 .. k * 2] else b[(r - n0) * k * 2 ..][0 .. k * 2];
        @memcpy(std.mem.sliceAsBytes(words[dst_r * k ..][0..k]), src);
        dst_r += 1;
    }
    const npad = (row_len + 127) / 128 * 128;
    const out = try gpa.alloc(u16, npad * k);
    defer gpa.free(out);
    qmmf.packBf16(out, words, row_len, k);
    const codes = try DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(out));
    return .{ .w = .{ .mode = .bf16, .codes = codes.ptr, .scales = 0, .n = @intCast(row_len), .k = @intCast(k), .npad = @intCast(npad) }, .codes = codes };
}

fn upload(d: *const cuda.Driver, bytes: []const u8) !DeviceBuffer {
    return DeviceBuffer.fromHost(d, bytes);
}

/// One layer's attention on one rank of `world`.
pub const Weights = struct {
    heads: usize, // this rank's
    window: usize, // keys a sliding row sees; 0 on an indexed layer (causal, or the indexer's list)
    eps: f32,
    hidden: usize,
    q_lora: usize,
    qkv_shards: usize = 1, // ranks `qkv_a_part` is row-split over for decode (R <= 64); 1 when every rank reads it whole
    qkv_a: Packed, // stacked [q_a (q_lora), kv_a (576)] x hidden bf16, full (prefill R > 64, or unsharded)
    qkv_a_part: ?Packed = null, // this rank's rows [(q_lora + 576) / world, hidden] bf16 for decode (R <= 64)
    q_b: Packed, // this rank's heads' rows
    gate: Packed, // linear_gate: this rank's heads' rows
    o: Packed, // o_proj: this rank's heads' input columns
    kv_b: DeviceBuffer, // unused placeholder (16 B): c2 reads uk / uv instead of kv_b as stored
    uk: DeviceBuffer, // bf16 W_UK_T [heads][512][192], c2 TILED unit order
    uv: DeviceBuffer, // bf16 W_UV [heads][256][512], c2 TILED unit order
    q_a_norm: DeviceBuffer, // bf16
    kv_a_norm: DeviceBuffer, // bf16
    post_norm: DeviceBuffer, // bf16
    sinks: DeviceBuffer, // fp32 [heads]
    freqs: DeviceBuffer, // fp32 [rope / 2]: the layer kind's theta^(-2i / 64)

    pub fn load(d: *const cuda.Driver, gpa: std.mem.Allocator, t: names.Attn, c: config.Config, layer: usize, rank: usize, world: usize) !Weights {
        return loadEx(d, gpa, t, c, layer, rank, world, false);
    }

    pub fn loadEx(d: *const cuda.Driver, gpa: std.mem.Allocator, t: names.Attn, c: config.Config, layer: usize, rank: usize, world: usize, shard_replicated: bool) !Weights {
        try geometry(c, layer, world);
        const share = try partition.Attn.of(c, layer, rank, world);
        const h = c.heads(layer);
        const hd = c.hidden;
        const all = h.count;
        const whole = struct {
            fn of(n: usize) Share {
                return .{ .lo = 0, .len = n };
            }
        }.of;
        var w: Weights = undefined;
        w.heads = share.heads.len;
        w.window = if (c.kind[layer] == .sliding) c.window else 0;
        w.eps = c.eps;
        w.hidden = hd;
        w.q_lora = c.q_lora;
        const qkv_total = c.q_lora + latent + rope;
        const do_shard = shard_replicated and world >= 2 and (qkv_total % (16 * world) == 0);
        w.qkv_shards = if (do_shard) world else 1;
        w.qkv_a = try packBf16PairSlice(d, gpa, t.q_a.bytes, c.q_lora, t.kv_a.bytes, latent + rope, hd, 0, qkv_total);
        errdefer w.qkv_a.deinit();
        w.qkv_a.w = try qmmf.pinned(w.qkv_a.w, 1);
        w.qkv_a_part = if (do_shard) try packBf16PairSlice(d, gpa, t.q_a.bytes, c.q_lora, t.kv_a.bytes, latent + rope, hd, rank * (qkv_total / world), qkv_total / world) else null;
        errdefer if (w.qkv_a_part) |*qh| qh.deinit();
        if (w.qkv_a_part) |*qh| qh.w = try qmmf.pinned(qh.w, 1);
        w.q_b = try pack(d, gpa, t.q_b, all * q_head, c.q_lora, share.q_rows, whole(c.q_lora));
        errdefer w.q_b.deinit();
        w.q_b.w = try qmmf.pinned(w.q_b.w, 1);
        w.gate = try pack(d, gpa, t.gate, all * v_width, hd, share.gate_rows, whole(hd));
        errdefer w.gate.deinit();
        w.gate.w = try qmmf.pinned(w.gate.w, 1);
        w.o = try pack(d, gpa, t.o, hd, all * v_width, whole(hd), share.o_cols);
        errdefer w.o.deinit();
        w.o.w = try qmmf.pinned(w.o.w, 1);
        {
            const tiled = try @import("cuda_mla_c2.zig").repack(gpa, t.kv_b.bytes[share.kv_rows.lo * latent * 2 ..][0 .. share.kv_rows.len * latent * 2], share.heads.len);
            defer gpa.free(tiled.uk);
            defer gpa.free(tiled.uv);
            w.uk = try upload(d, std.mem.sliceAsBytes(tiled.uk));
            errdefer w.uk.free();
            w.uv = try upload(d, std.mem.sliceAsBytes(tiled.uv));
        }
        errdefer w.uk.free();
        errdefer w.uv.free();
        w.kv_b = try DeviceBuffer.alloc(d, 16);
        errdefer w.kv_b.free();
        w.q_a_norm = try upload(d, t.q_a_norm.bytes);
        errdefer w.q_a_norm.free();
        w.kv_a_norm = try upload(d, t.kv_a_norm.bytes);
        errdefer w.kv_a_norm.free();
        w.post_norm = try upload(d, t.post_norm.bytes);
        errdefer w.post_norm.free();
        w.sinks = try upload(d, t.sinks.bytes[share.heads.lo * 4 ..][0 .. share.heads.len * 4]);
        errdefer w.sinks.free();
        const f = cuda.mla.ropeFreqs(h.theta);
        w.freqs = try upload(d, std.mem.sliceAsBytes(&f));
        return w;
    }

    pub fn deinit(w: *Weights) void {
        if (w.qkv_a_part) |*qh| qh.deinit();
        for ([_]*Packed{ &w.qkv_a, &w.q_b, &w.gate, &w.o }) |p| p.deinit();
        for ([_]*DeviceBuffer{ &w.kv_b, &w.uk, &w.uv, &w.q_a_norm, &w.kv_a_norm, &w.post_norm, &w.sinks, &w.freqs }) |b| b.free();
    }
};

test "block: a rank's rows and input columns of a row-major matrix, in order" {
    const gpa = std.testing.allocator;
    var m: [6 * 8]u16 = undefined;
    for (&m, 0..) |*v, i| v.* = @intCast(i);
    const got = try block(gpa, std.mem.sliceAsBytes(&m), 8, 2, .{ .lo = 2, .len = 3 }, .{ .lo = 4, .len = 4 });
    defer gpa.free(got);
    for (0..3) |r| for (0..4) |c| try std.testing.expectEqual(@as(u16, @intCast((2 + r) * 8 + 4 + c)), std.mem.readInt(u16, got[(r * 4 + c) * 2 ..][0..2], .little));
}

test "geometry: Hy4-flash's heads split in whole tiles at worlds 1, 2 and 4, and nothing else" {
    var c: config.Config = undefined;
    c.kv_lora = latent;
    c.indexed = .{ .count = 32, .nope = nope, .rope = rope, .v = v_width, .theta = 1e7 };
    c.sliding = .{ .count = 64, .nope = nope, .rope = rope, .v = v_width, .theta = 1e4 };
    c.kind = @splat(.sliding);
    c.kind[0] = .indexed;
    for ([_]usize{ 1, 2, 4 }) |world| {
        try geometry(c, 0, world);
        try geometry(c, 1, world);
    }
    try std.testing.expectError(error.MlaHeadsSplit, geometry(c, 0, 8));
    c.indexed.nope = 128;
    try std.testing.expectError(error.MlaGeometry, geometry(c, 0, 1));
}
