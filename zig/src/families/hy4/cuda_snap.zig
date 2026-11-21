//! Device prompt-cache snapshots for Hy4-flash: whole copies (`buf`) or `TF_KV_SHARE` ring/tail blobs (`blob`).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const cuda_cache = @import("cuda_cache.zig");
const cuda_index = @import("cuda_index.zig");
const cuda_pages = @import("cuda_pages.zig");
const fwd = @import("cuda_model.zig");

const Blob = core.kv_prompts.Blob;
const DeviceBuffer = cuda.DeviceBuffer;
const Stream = cuda.Stream;
const Model = fwd.Model;

pub const Saved = struct {
    id: u64,
    at: u32,
    from: u32 = 0, // [0, from) held in shared pool pages under TF_KV_SHARE; 0 for whole copies
    active_layers: usize,
    synced_peers: bool = false,
    kv8: bool = false,
    mrope_delta: i32 = 0,
    buf: DeviceBuffer,
    blob: []u32 = &.{}, // pool pages holding [from, at) + sliding rings under TF_KV_SHARE
    pages: []u32 = &.{}, // retained shared prefix pages when recalled from or queued to disk
    history: []u32,
};

/// Freed device snapshot buffers (up to `cap = 2`), reused in place before allocating/freeing on GB10 (#14.1).
pub const Spares = struct {
    bufs: [cap]DeviceBuffer = undefined,
    n: usize = 0,
    pub const cap = 2;

    fn find(s: *const Spares, len: usize) ?usize {
        var i = s.n;
        while (i > 0) {
            i -= 1;
            if (s.bufs[i].len == len) return i;
        }
        return null;
    }

    fn remove(s: *Spares, i: usize) DeviceBuffer {
        const b = s.bufs[i];
        for (i..s.n - 1) |j| s.bufs[j] = s.bufs[j + 1];
        s.n -= 1;
        return b;
    }

    pub fn fits(s: *const Spares, len: usize) bool {
        return s.find(len) != null;
    }

    pub fn take(s: *Spares, d: *const cuda.Driver, len: usize) !DeviceBuffer {
        const i = s.find(len) orelse return DeviceBuffer.alloc(d, len);
        return s.remove(i);
    }

    pub fn give(s: *Spares, b: DeviceBuffer) void {
        if (b.len == 0 or b.borrowed) return;
        if (s.n == cap) {
            var old = s.remove(0);
            old.free();
        }
        s.bufs[s.n] = b;
        s.n += 1;
    }

    pub fn bytes(s: *const Spares) u64 {
        var n: u64 = 0;
        for (s.bufs[0..s.n]) |b| n += b.len;
        return n;
    }

    pub fn trim(s: *Spares, room: u64) void {
        while (s.bytes() > room) {
            var big: usize = 0;
            for (s.bufs[1..s.n], 1..) |b, i| if (b.len > s.bufs[big].len) {
                big = i;
            };
            var old = s.remove(big);
            old.free();
        }
    }

    pub fn deinit(s: *Spares) void {
        s.trim(0);
    }
};

fn takeBuf(spares: ?*Spares, d: *const cuda.Driver, len: usize) !DeviceBuffer {
    return if (spares) |s| s.take(d, len) else DeviceBuffer.alloc(d, len);
}

fn giveBuf(spares: ?*Spares, b: DeviceBuffer) void {
    if (b.len == 0 or b.borrowed) return;
    if (spares) |s| return s.give(b);
    var tmp = b;
    tmp.free();
}

/// Device bytes copied for `[from, at)` across indexed tail rows and sliding rings (excludes `history`).
pub fn copiedDeviceBytes(m: *const Model, at: u32, from: u32) u64 {
    if (at == 0) return 0;
    const pos: u64 = at;
    const f0: u64 = from;
    const rb: u64 = cuda_cache.mlaRowBytes(m.kv8);
    const groups: u64 = (pos / cuda_index.ratio) -| (f0 / cuda_index.ratio);
    const tail_ik: u64 = (pos % cuda_index.ratio) * cuda_cache.index_row_bytes;
    const idx_bytes: u64 = @as(u64, m.n_indexed) * ((pos -| f0) * rb + tail_ik + groups * cuda_index.dim * 2);
    var n_sld: u64 = 0;
    for (m.layer_ids) |l| if (m.c.kind[l] == .sliding) {
        n_sld += 1;
    };
    const win: u64 = @min(pos, m.c.window);
    return idx_bytes + n_sld * win * rb;
}

pub fn bytes(m: *const Model, at: u32) u64 {
    if (at == 0) return 16;
    const hist_bytes: u64 = @as(u64, at) * 4;
    return @max(hist_bytes + copiedDeviceBytes(m, at, 0), 16);
}

pub fn blobPages(m: *const Model, at: u32, from: u32) usize {
    const pg = m.pages orelse return 0;
    return pg.blobPagesFor(copiedDeviceBytes(m, at, from));
}

/// Iterates every device slice of `slot`'s state `[from, at)` in canonical order.
pub fn each(
    m: *const Model,
    slot: usize,
    at: usize,
    from: usize,
    ctx: anytype,
    comptime f: fn (@TypeOf(ctx), dev: u64, off: usize, len: usize) anyerror!void,
) !void {
    if (at == 0) return;
    const sq = if (m.seqs[slot]) |*p| p else return error.UnknownSlot;
    const rb = cuda_cache.mlaRowBytes(m.kv8);
    const ik_len = (at % cuda_index.ratio) * cuda_cache.index_row_bytes;
    const g_from = from / cuda_index.ratio;
    const g_end = at / cuda_index.ratio;
    var off: usize = 0;

    for (m.layer_ids) |l| {
        const v = try sq.cache.view(l);
        if (v.kind == .indexed) {
            const ord = m.indexed_ord[l].?;
            if (m.pages) |pg| {
                const ids = pg.leases[slot].ids.items;
                const pt: usize = cuda_pages.page_tokens;
                var q = from;
                while (q < at) {
                    const pid = ids[q / pt];
                    const rem_in_page = pt - (q % pt);
                    const run = @min(at - q, rem_in_page);
                    try f(ctx, pg.mlaRowPtr(ord, pid, q % pt), off, run * rb);
                    off += run * rb;
                    q += run;
                }
            } else if (at > from) {
                const len_b = (at - from) * rb;
                try f(ctx, v.mla + from * rb, off, len_b);
                off += len_b;
            }
            if (ik_len > 0) {
                const slot0 = if (v.tail_index_keys) (at - (at % cuda_index.ratio)) % cuda_cache.tail_index_rows else g_end * cuda_index.ratio;
                const ik_ptr = v.index_keys + slot0 * cuda_cache.index_row_bytes;
                try f(ctx, ik_ptr, off, ik_len);
                off += ik_len;
            }
            if (g_end > g_from) {
                if (m.pages) |pg| {
                    const ids = pg.leases[slot].ids.items;
                    const pg_g = cuda_pages.page_groups;
                    var g = g_from;
                    while (g < g_end) {
                        const pid = ids[g / pg_g];
                        const rem_g = pg_g - (g % pg_g);
                        const run_g = @min(g_end - g, rem_g);
                        const len_b = run_g * cuda_index.dim * 2;
                        try f(ctx, pg.gkRowPtr(ord, pid, g % pg_g), off, len_b);
                        off += len_b;
                        g += run_g;
                    }
                } else {
                    const len_b = (g_end - g_from) * cuda_index.dim * 2;
                    try f(ctx, sq.groupKeysAt(ord) + g_from * cuda_index.dim * 2, off, len_b);
                    off += len_b;
                }
            }
        } else {
            const win = @min(at, m.c.window);
            const start = at - win;
            for (cuda_cache.runs(start, win, v.ring)) |run| if (run[1] > 0) {
                const len_b = run[1] * rb;
                try f(ctx, v.mla + run[0] * rb, off, len_b);
                off += len_b;
            };
        }
    }
}

const Copy = struct {
    m: *const Model,
    s: Stream,
    buf_base: u64 = 0,
    blob: ?Blob = null,
    out: bool,
};

const Piece = struct {
    c: Copy,
    dev: u64,
    fn copy(p: Piece, region: usize, id: u32, within: u64, from_off: u64, n: u64) anyerror!void {
        const pg = p.c.m.pages.?;
        const kept = pg.blobPieceAt(region, id) + within;
        const src = if (p.c.out) p.dev + from_off else kept;
        const dst = if (p.c.out) kept else p.dev + from_off;
        try p.c.m.d.check(p.c.m.d.api.cuMemcpyDtoDAsync_v2(dst, src, @intCast(n), p.c.s.handle), "cuMemcpyDtoDAsync");
    }
};

fn copySlice(c: Copy, dev: u64, off: usize, len: usize) anyerror!void {
    if (len == 0) return;
    if (c.blob) |b| return b.each(off, len, Piece{ .c = c, .dev = dev }, Piece.copy);
    const kept = c.buf_base + off;
    const src = if (c.out) dev else kept;
    const dst = if (c.out) kept else dev;
    try c.m.d.check(c.m.d.api.cuMemcpyDtoDAsync_v2(dst, src, len, c.s.handle), "cuMemcpyDtoDAsync");
}

pub fn blank(gpa: std.mem.Allocator, spares: ?*Spares, m: *const Model, id: u64, at: u32) !*Saved {
    const total: usize = @intCast(bytes(m, at));
    const buf = try takeBuf(spares, m.d, total);
    errdefer giveBuf(spares, buf);
    const hist = try gpa.alloc(u32, at);
    errdefer gpa.free(hist);
    @memset(hist, 0);
    const out = try gpa.create(Saved);
    out.* = .{ .id = id, .at = at, .active_layers = m.active_layers, .kv8 = m.kv8, .buf = buf, .history = hist };
    return out;
}

pub fn syncHistory(k: *Saved) !void {
    if (k.at == 0 or k.buf.len == 0) return;
    try (DeviceBuffer{ .d = k.buf.d, .ptr = k.buf.ptr, .len = @as(usize, k.at) * 4 }).download(0, std.mem.sliceAsBytes(k.history));
}

pub fn save(gpa: std.mem.Allocator, spares: ?*Spares, m: *const Model, s: Stream, slot: usize, id: u64, at: u32) !*Saved {
    const sq = if (m.seqs[slot]) |*p| p else return error.UnknownSlot;
    if (at > sq.len) return error.BadSnapshotPos;
    const total: usize = @intCast(bytes(m, at));
    const buf = try takeBuf(spares, m.d, total);
    errdefer {
        cuda.completion.drain(m.d, s, "hy4: failed snapshot save did not drain");
        giveBuf(spares, buf);
    }
    const hist = try gpa.alloc(u32, at);
    errdefer gpa.free(hist);
    const pos: usize = at;
    if (pos > 0) {
        @memcpy(hist, sq.history[0..pos]);
        try buf.uploadAsync(0, std.mem.sliceAsBytes(hist), s.handle);
    }
    try each(m, slot, pos, 0, Copy{ .m = m, .s = s, .buf_base = buf.ptr + pos * 4, .out = true }, copySlice);
    const out = try gpa.create(Saved);
    out.* = .{ .id = id, .at = at, .active_layers = m.active_layers, .kv8 = m.kv8, .mrope_delta = sq.mrope_delta, .buf = buf, .history = hist };
    return out;
}

/// `TF_KV_SHARE`: saves `[from, at)` + sliding rings into `blob` pool pages (`[0, from)` stays in shared pool pages).
pub fn saveShare(gpa: std.mem.Allocator, m: *const Model, s: Stream, slot: usize, id: u64, at: u32, from: u32, blob: []u32) !*Saved {
    const sq = if (m.seqs[slot]) |*p| p else return error.UnknownSlot;
    const pg = m.pages orelse return error.NotPaged;
    if (at > sq.len or from > at) return error.BadSnapshotPos;
    const hist = try gpa.alloc(u32, at);
    errdefer gpa.free(hist);
    if (at > 0) @memcpy(hist, sq.history[0..at]);
    if (blob.len > 0) {
        const b_view: Blob = .{ .ids = blob, .chunk = pg.blobChunk(), .regions = pg.blobRegions() };
        try each(m, slot, at, from, Copy{ .m = m, .s = s, .blob = b_view, .out = true }, copySlice);
    }
    const out = try gpa.create(Saved);
    out.* = .{
        .id = id,
        .at = at,
        .from = from,
        .active_layers = m.active_layers,
        .kv8 = m.kv8,
        .mrope_delta = sq.mrope_delta,
        .buf = .{ .d = m.d, .ptr = 0, .len = 0, .borrowed = true },
        .blob = blob,
        .history = hist,
    };
    return out;
}

pub fn restore(m: *Model, s: Stream, slot: usize, k: *const Saved) !void {
    if (k.active_layers != m.active_layers or k.kv8 != m.kv8) return error.SnapshotMismatch;
    const sq = if (m.seqs[slot]) |*p| p else return error.UnknownSlot;
    const pos: usize = k.at;
    if (pos > sq.cap) return error.CacheOverflow;
    if (pos > 0) @memcpy(sq.history[0..pos], k.history);
    if (sq.media) |*old| {
        old.deinit();
        sq.media = null;
    }
    sq.mrope_delta = k.mrope_delta;
    if (k.blob.len > 0 or k.from > 0) {
        if (k.blob.len > 0) {
            const pg = m.pages orelse return error.NotPaged;
            const b_view: Blob = .{ .ids = k.blob, .chunk = pg.blobChunk(), .regions = pg.blobRegions() };
            try each(m, slot, pos, k.from, Copy{ .m = m, .s = s, .blob = b_view, .out = false }, copySlice);
        }
    } else if (pos > 0) {
        try each(m, slot, pos, 0, Copy{ .m = m, .s = s, .buf_base = k.buf.ptr + pos * 4, .out = false }, copySlice);
    }
    sq.len = pos;
}

pub fn drop(gpa: std.mem.Allocator, spares: ?*Spares, s: Stream, k: *Saved) void {
    s.synchronize() catch {};
    giveBuf(spares, k.buf);
    if (k.pages.len > 0) gpa.free(k.pages);
    gpa.free(k.history);
    gpa.destroy(k);
}

test "Spares retains up to two buffers, reuses exact sizes, and trims largest first" {
    const Fake = struct {
        var next: u64 = 0x1000;
        var live: usize = 0;
        fn alloc(out: *cuda.abi.DevicePtr, n: usize) callconv(.c) cuda.abi.Result {
            out.* = next;
            next += @intCast(n + 256);
            live += 1;
            return 0;
        }
        fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
            live -= 1;
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Fake.alloc;
    d.api.cuMemFree_v2 = Fake.free;
    var sp: Spares = .{};
    defer sp.deinit();
    const b1 = try sp.take(&d, 1024);
    const b2 = try sp.take(&d, 4096);
    const b3 = try sp.take(&d, 2048);
    try std.testing.expectEqual(@as(usize, 3), Fake.live);
    sp.give(b1);
    sp.give(b2);
    try std.testing.expect(sp.fits(1024) and sp.fits(4096));
    try std.testing.expectEqual(@as(u64, 5120), sp.bytes());
    sp.give(b3);
    try std.testing.expect(!sp.fits(1024) and sp.fits(4096) and sp.fits(2048));
    try std.testing.expectEqual(@as(usize, 2), Fake.live);
    const reused = try sp.take(&d, 4096);
    try std.testing.expectEqual(b2.ptr, reused.ptr);
    sp.give(reused);
    sp.trim(2048);
    try std.testing.expect(!sp.fits(4096) and sp.fits(2048));
    sp.deinit();
    try std.testing.expectEqual(@as(usize, 0), Fake.live);
}
