//! Hy4-flash's per-sequence and per-layer KV caches: indexed MLA+key rows (1,408 B/token/layer) and sliding rings.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const memory = @import("memory.zig");
const cuda_mla = @import("cuda_mla.zig");

const Config = config.Config;
const Attention = config.Attention;
const Stream = cuda.Stream;
const DeviceBuffer = cuda.DeviceBuffer;

pub const latent_cols = cuda_mla.latent; // 512 normed latent columns
pub const rope_cols = cuda_mla.rope; // 64 roped key columns
pub const mla_cols = cuda_mla.row; // 576 bf16 columns per MLA cache row
pub const mla_row_bytes = mla_cols * 2; // 1,152 B/token/layer (bf16)
pub const mla_row_fp8_bytes: usize = 656; // 512B e4m3 + 128B bf16 k_rope + 4B fp32 scale + 12B pad
pub const index_cols = 128; // compressed indexer token-key columns
pub const index_row_bytes = index_cols * 2; // 256 B/token/indexed-layer
pub const tail_index_rows: usize = 32; // rewind-safe ring of unpooled token keys across verify windows
pub const indexed_layer_token_bytes = mla_row_bytes + index_row_bytes; // 1,408 B/token/indexed-layer

pub const Format = struct { kv8: bool = false, tail_index_keys: bool = false, paged: bool = false };

pub fn mlaRowBytes(kv8: bool) usize {
    return if (kv8) mla_row_fp8_bytes else mla_row_bytes;
}

/// Storage layout for the 30 sliding-window layers (`window = 512`).
pub const SlidingMode = union(enum) {
    linear: void, // [capacity, 576] bf16
    ring: usize, // [ring_slots, 576] bf16 (`512` for 1-token decode, `>= 511 + max_chunk` for chunked steps)
};

pub fn indexedLayerCount(c: Config) usize {
    var n: usize = 0;
    for (c.kind[0..c.layers]) |k| if (k == .indexed) {
        n += 1;
    };
    return n;
}

pub fn slidingLayerCount(c: Config) usize {
    var n: usize = 0;
    for (c.kind[0..c.layers]) |k| if (k == .sliding) {
        n += 1;
    };
    return n;
}

/// Bytes added per token across all indexed layers (12 x 1,408 = 16,896 B/token on Hy4-flash).
pub fn indexedTokenBytes(c: Config) usize {
    return indexedLayerCount(c) * 2 * (c.kv_lora + c.indexed.rope + c.index_dim);
}

/// Total bytes of all sliding layers' rings at `ring_slots` rows each (17,694,720 B at `ring_slots = 512`).
pub fn slidingRingBytes(c: Config, ring_slots: usize) usize {
    return slidingLayerCount(c) * 2 * (c.kv_lora + c.sliding.rope) * ring_slots;
}

/// Smallest collision-free ring slot count for `window` when `attend` processes up to `max_step_rows` rows at once.
pub fn minRingSlots(window: usize, max_step_rows: usize) usize {
    if (max_step_rows <= 1) return window;
    return window - 1 + max_step_rows;
}

pub fn slotOf(pos: usize, ring: usize) usize {
    return if (ring == 0) pos else pos % ring;
}

pub fn runs(first: usize, n: usize, ring: usize) [2][2]usize {
    return cuda_mla.runs(first, n, ring);
}

/// Validates that writing `[first, first + rows)` and attending from `first` fits without ring slot collisions.
pub fn checkStep(kind: Attention, capacity: usize, window: usize, ring: usize, first: usize, rows: usize) !void {
    if (rows == 0) return;
    if (kind == .indexed or ring == 0) {
        if (ring != 0) return error.RingShape;
        if (first + rows > capacity) return error.CacheOverflow;
        return;
    }
    if (window < 2 or ring < window or ring > capacity or rows > ring) return error.RingShape;
    const lo = if (first + 1 > window) first + 1 - window else 0;
    if ((first + rows - 1) - lo >= ring) return error.RingStepTooWide;
}

/// Device bytes one layer's cache occupies for `capacity` tokens under `mode` and `fmt`.
pub fn layerBytesEx(c: Config, layer: usize, capacity: usize, mode: SlidingMode, fmt: Format) !usize {
    if (layer >= c.layers or capacity == 0) return error.CacheShape;
    if (c.kv_lora != latent_cols or c.heads(layer).rope != rope_cols or c.index_dim != index_cols) return error.CacheShape;
    const rb = mlaRowBytes(fmt.kv8);
    if (c.kind[layer] == .indexed) {
        if (fmt.paged) return tail_index_rows * index_row_bytes;
        const ik_rows = if (fmt.tail_index_keys) tail_index_rows else capacity;
        return capacity * rb + ik_rows * index_row_bytes;
    }
    return switch (mode) {
        .linear => capacity * rb,
        .ring => |slots| if (slots < c.window) error.RingShape else slots * rb,
    };
}

pub fn layerBytes(c: Config, layer: usize, capacity: usize, mode: SlidingMode) !usize {
    return layerBytesEx(c, layer, capacity, mode, .{});
}

pub fn sequenceBytesEx(c: Config, layer_ids: []const usize, capacity: usize, mode: SlidingMode, fmt: Format) !usize {
    var total: usize = 0;
    for (layer_ids) |l| total += try layerBytesEx(c, l, capacity, mode, fmt);
    return total;
}

/// Device bytes across `layer_ids` (`0..c.layers` or a subset) for `capacity` tokens under `mode`.
pub fn sequenceBytes(c: Config, layer_ids: []const usize, capacity: usize, mode: SlidingMode) !usize {
    return sequenceBytesEx(c, layer_ids, capacity, mode, .{});
}

/// Non-owning device pointers for one layer's cache, passed to `cuda_layer` and `cuda_mla`.
pub const LayerView = struct {
    kind: Attention,
    mla: u64, // bf16 [slots, 576]
    index_keys: u64 = 0, // bf16 [capacity, 128] on indexed layers, 0 on sliding layers
    capacity: usize, // linear capacity on indexed/linear layers, ring_slots on ring layers
    window: usize = 0, // c.window on sliding layers, 0 on indexed layers
    ring: usize = 0, // 0 when linear, ring_slots when ring
    kv8: bool = false,
    tail_index_keys: bool = false,
    page_table: u64 = 0, // non-zero when indexed MLA + group_keys live in 256-token pool pages
    pt_stride: usize = 0,
    pub const Made = struct { view: LayerView, bytes: usize };

    pub fn ofPtrEx(c: Config, layer: usize, base: u64, capacity: usize, mode: SlidingMode, fmt: Format) !Made {
        const bytes = try layerBytesEx(c, layer, capacity, mode, fmt);
        const rb = mlaRowBytes(fmt.kv8);
        if (c.kind[layer] == .indexed) {
            if (fmt.paged) {
                return .{
                    .view = .{ .kind = .indexed, .mla = 0, .index_keys = base, .capacity = capacity, .kv8 = fmt.kv8, .tail_index_keys = true },
                    .bytes = bytes,
                };
            }
            const mla_bytes = capacity * rb;
            return .{
                .view = .{ .kind = .indexed, .mla = base, .index_keys = base + mla_bytes, .capacity = capacity, .kv8 = fmt.kv8, .tail_index_keys = fmt.tail_index_keys },
                .bytes = bytes,
            };
        }
        return switch (mode) {
            .linear => .{
                .view = .{ .kind = .sliding, .mla = base, .capacity = capacity, .window = c.window, .ring = 0, .kv8 = fmt.kv8 },
                .bytes = bytes,
            },
            .ring => |slots| .{
                .view = .{ .kind = .sliding, .mla = base, .capacity = slots, .window = c.window, .ring = slots, .kv8 = fmt.kv8 },
                .bytes = bytes,
            },
        };
    }

    pub fn ofPtr(c: Config, layer: usize, base: u64, capacity: usize, mode: SlidingMode) !Made {
        return ofPtrEx(c, layer, base, capacity, mode, .{});
    }

    pub fn check(v: LayerView, first: usize, rows: usize) !void {
        return checkStep(v.kind, v.capacity, v.window, v.ring, first, rows);
    }

    pub fn mlaRow(v: LayerView, pos: usize) u64 {
        return v.mla + slotOf(pos, v.ring) * mlaRowBytes(v.kv8);
    }
    pub fn indexKeyRow(v: LayerView, pos: usize) u64 {
        return v.index_keys + pos * index_row_bytes;
    }

    pub fn mlaRuns(v: LayerView, first: usize, rows: usize) [2][2]usize {
        return runs(first, rows, v.ring);
    }
};

/// Single-layer KV cache allocation on the device.
pub const LayerCache = struct {
    buf: DeviceBuffer,
    view: LayerView,

    pub fn init(d: *const cuda.Driver, c: Config, layer: usize, capacity: usize, mode: SlidingMode) !LayerCache {
        const bytes = try layerBytes(c, layer, capacity, mode);
        var buf = try DeviceBuffer.alloc(d, bytes);
        errdefer buf.free();
        const made = try LayerView.ofPtr(c, layer, buf.ptr, capacity, mode);
        return .{ .buf = buf, .view = made.view };
    }

    pub fn fill(lc: *const LayerCache, s: Stream, byte: u8) !void {
        return lc.buf.fill8(byte, s.handle);
    }

    pub fn deinit(lc: *LayerCache) void {
        lc.buf.free();
        lc.* = undefined;
    }
};

/// Contiguous device KV cache across all layers (or a chosen subset) for one sequence.
pub const SequenceCache = struct {
    buf: DeviceBuffer,
    views: [config.max_layers]?LayerView = @splat(null),
    capacity: usize,
    mode: SlidingMode,
    fmt: Format = .{},

    pub fn init(d: *const cuda.Driver, c: Config, capacity: usize, mode: SlidingMode) !SequenceCache {
        var all: [config.max_layers]usize = undefined;
        for (0..c.layers) |i| all[i] = i;
        return initLayers(d, c, all[0..c.layers], capacity, mode);
    }

    pub fn initLayers(d: *const cuda.Driver, c: Config, layer_ids: []const usize, capacity: usize, mode: SlidingMode) !SequenceCache {
        return initLayersEx(d, c, layer_ids, capacity, mode, .{});
    }

    pub fn initLayersEx(d: *const cuda.Driver, c: Config, layer_ids: []const usize, capacity: usize, mode: SlidingMode, fmt: Format) !SequenceCache {
        if (layer_ids.len == 0) return error.CacheShape;
        const total = try sequenceBytesEx(c, layer_ids, capacity, mode, fmt);
        var buf = try DeviceBuffer.alloc(d, total);
        errdefer buf.free();
        var sc: SequenceCache = .{ .buf = buf, .capacity = capacity, .mode = mode, .fmt = fmt };
        var off: usize = 0;
        for (layer_ids) |l| {
            if (sc.views[l] != null) return error.DuplicateLayer;
            const made = try LayerView.ofPtrEx(c, l, buf.ptr + off, capacity, mode, fmt);
            sc.views[l] = made.view;
            off += made.bytes;
        }
        std.debug.assert(off == total);
        return sc;
    }

    pub fn view(sc: *const SequenceCache, layer: usize) !LayerView {
        if (layer >= config.max_layers) return error.CacheShape;
        return sc.views[layer] orelse error.UnallocatedLayer;
    }

    pub fn fill(sc: *const SequenceCache, s: Stream, byte: u8) !void {
        return sc.buf.fill8(byte, s.handle);
    }

    pub fn deinit(sc: *SequenceCache) void {
        sc.buf.free();
        sc.* = undefined;
    }
};

test "cache byte geometry matches memory.zig (16,896 B/token indexed, 17,694,720 B sliding ring)" {
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, @embedFile("fixtures/config.json"), &why);
    try std.testing.expectEqual(@as(usize, 12), indexedLayerCount(c));
    try std.testing.expectEqual(@as(usize, 30), slidingLayerCount(c));
    try std.testing.expectEqual(@as(usize, 1_152), mla_row_bytes);
    try std.testing.expectEqual(@as(usize, 256), index_row_bytes);
    try std.testing.expectEqual(@as(usize, 1_408), indexed_layer_token_bytes);
    try std.testing.expectEqual(@as(usize, 16_896), indexedTokenBytes(c));
    try std.testing.expectEqual(memory.tokenBytes(c), indexedTokenBytes(c));
    try std.testing.expectEqual(@as(usize, 30 * 512 * 1_152), slidingRingBytes(c, c.window));
    try std.testing.expectEqual(memory.ringBytes(c), slidingRingBytes(c, c.window));

    var all: [config.max_layers]usize = undefined;
    for (0..c.layers) |i| all[i] = i;
    for ([_]usize{ 256, 2_560, 1_048_576 }) |cap| {
        const ring_total = try sequenceBytes(c, all[0..c.layers], cap, .{ .ring = c.window });
        try std.testing.expectEqual(cap * memory.tokenBytes(c) + memory.ringBytes(c), ring_total);
        const lin_total = try sequenceBytes(c, all[0..c.layers], cap, .linear);
        try std.testing.expectEqual(cap * (12 * 1_408 + 30 * 1_152), lin_total);
    }
}

test "ring slot indexing, wrap runs, and step collision guards" {
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, @embedFile("fixtures/config.json"), &why);
    const base: u64 = 0x10000;
    const idx_v = (try LayerView.ofPtr(c, 0, base, 2560, .{ .ring = 512 })).view;
    try std.testing.expectEqual(Attention.indexed, idx_v.kind);
    try std.testing.expectEqual(@as(usize, 0), idx_v.ring);
    try std.testing.expectEqual(base + 2560 * 1_152, idx_v.index_keys);
    try std.testing.expectEqual(base + 7 * 1_152, idx_v.mlaRow(7));
    try std.testing.expectEqual(base + 2560 * 1_152 + 7 * 256, idx_v.indexKeyRow(7));
    try idx_v.check(2500, 60);
    try std.testing.expectError(error.CacheOverflow, idx_v.check(2500, 61));

    const ring_v = (try LayerView.ofPtr(c, 2, base, 2560, .{ .ring = 512 })).view;
    try std.testing.expectEqual(Attention.sliding, ring_v.kind);
    try std.testing.expectEqual(@as(usize, 512), ring_v.ring);
    try std.testing.expectEqual(@as(usize, 0), slotOf(512, ring_v.ring));
    try std.testing.expectEqual(@as(usize, 511), slotOf(1023, ring_v.ring));
    try std.testing.expectEqual(base + 3 * 1_152, ring_v.mlaRow(515));
    try std.testing.expectEqual([2][2]usize{ .{ 508, 4 }, .{ 0, 5 } }, ring_v.mlaRuns(1020, 9));
    try ring_v.check(0, 256);
    try ring_v.check(511, 1);
    try ring_v.check(512, 1);
    try ring_v.check(1_000_000, 1);
    try std.testing.expectError(error.RingStepTooWide, ring_v.check(500, 64));

    const wide_slots = minRingSlots(c.window, 192);
    try std.testing.expectEqual(@as(usize, 703), wide_slots);
    const wide_v = (try LayerView.ofPtr(c, 2, base, 2560, .{ .ring = 1024 })).view;
    try wide_v.check(500, 192);
    try wide_v.check(2368, 192);

    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(LayerView);
    std.testing.refAllDecls(LayerCache);
    std.testing.refAllDecls(SequenceCache);
}
