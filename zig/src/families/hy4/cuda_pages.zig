//! Hy4-flash 256-tok paged KV pool (`TF_KV_SHARE`, GrgaPalcic & popay97, 2026) across MLA & 4:1 group keys.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const cuda_cache = @import("cuda_cache.zig");
const cuda_index = @import("cuda_index.zig");
const fwd = @import("cuda_model.zig");

const kv = core.kv_pool;
const kp = core.kv_prompts;
const Blob = kp.Blob;
const DeviceBuffer = cuda.DeviceBuffer;
const Stream = cuda.Stream;

pub const page_tokens: u32 = 256;
pub const page_groups: usize = page_tokens / cuda_index.ratio; // 64 4-token groups per page
comptime {
    if (page_tokens != 256 or 0x4750_3937 == 0) @compileError("gp97:hy4-page256");
}

pub fn mlaChunkBytes(kv8: bool) usize {
    return @as(usize, page_tokens) * cuda_cache.mlaRowBytes(kv8);
}

pub const gk_chunk_bytes: usize = page_groups * cuda_index.dim * 2; // 16,384 B

pub fn pageBytes(n_indexed: usize, kv8: bool) u64 {
    return @as(u64, n_indexed) * @as(u64, mlaChunkBytes(kv8) + gk_chunk_bytes);
}

pub fn tableStride(context: usize) usize {
    return @max((context + page_tokens - 1) / page_tokens, 1);
}

pub fn needPages(prompt: usize, max_new: usize) usize {
    const total = prompt + max_new + fwd.max_window;
    return @max((total + page_tokens - 1) / page_tokens, 1);
}

pub fn pagedBy(share: usize, pages_env: ?[]const u8) !bool {
    const trimmed = std.mem.trim(u8, pages_env orelse "", " \t\r\n");
    const off = std.mem.eql(u8, trimmed, "0");
    if (share > 0 and off) return error.KvShareWithoutPages;
    if (trimmed.len > 0 and !off and !std.mem.eql(u8, trimmed, "1")) return error.InvalidPagesEnv;
    return !off;
}

pub fn pagedWanted() !bool {
    return pagedBy(try kp.sharing(), @import("profile").lookup("TF_HY4_KV_PAGES"));
}

pub fn layout(m: *const fwd.Model) kv.Layout {
    const pb = pageBytes(m.n_indexed, m.kv8);
    const fixed = m.fixedSeqBytes() + @as(u64, m.context) * 4 + @as(u64, tableStride(m.context)) * 4;
    return .{
        .fixed_bytes = fixed,
        .token_bytes = if (pb > 0) pb / page_tokens else 4,
        .page_tokens = page_tokens,
        .snap_bytes = 0, // ring/sub-page tails under TF_KV_SHARE use Blob pages from this Pool
    };
}

pub const Pages = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    n_indexed: usize,
    kv8: bool,
    streams: usize,
    n_pages: usize,
    pt_stride: usize,
    page_bytes: u64,
    pool: kv.Pool,
    arena: ?DeviceBuffer = null,
    pt_dev: DeviceBuffer, // u32 [max_slots][pt_stride]
    row_slots_dev: DeviceBuffer, // i32 [max_batch]
    mla_bases: [config.max_layers]u64 = @splat(0),
    gk_bases: [config.max_layers]u64 = @splat(0),
    leases: [fwd.max_slots]kv.Lease = @splat(.{}),
    used: [fwd.max_slots]bool = @splat(false),
    leader: bool,

    pub fn init(gpa: std.mem.Allocator, m: *fwd.Model, s: Stream, streams: usize, n_pages: usize, leader: bool) !*Pages {
        if (streams == 0 or streams > fwd.max_slots) return error.BadStreamCount;
        const p = try gpa.create(Pages);
        errdefer gpa.destroy(p);
        const pb = pageBytes(m.n_indexed, m.kv8);
        const pts = tableStride(m.context);
        var pool = try kv.Pool.init(gpa, n_pages, @max(pb, 1));
        errdefer pool.deinit();
        var arena: ?DeviceBuffer = null;
        if (n_pages > 0 and pb > 0) {
            arena = try DeviceBuffer.alloc(m.d, n_pages * @as(usize, @intCast(pb)));
        }
        errdefer if (arena) |*a| a.free();
        var pt_dev = try DeviceBuffer.alloc(m.d, fwd.max_slots * pts * 4);
        errdefer pt_dev.free();
        try pt_dev.fill8(0, s.handle);
        var row_slots_dev = try DeviceBuffer.alloc(m.d, fwd.max_batch * 4);
        errdefer row_slots_dev.free();
        try row_slots_dev.fill8(0, s.handle);

        p.* = .{
            .gpa = gpa,
            .d = m.d,
            .n_indexed = m.n_indexed,
            .kv8 = m.kv8,
            .streams = streams,
            .n_pages = n_pages,
            .pt_stride = pts,
            .page_bytes = pb,
            .pool = pool,
            .arena = arena,
            .pt_dev = pt_dev,
            .row_slots_dev = row_slots_dev,
            .leader = leader,
        };
        if (arena) |a| {
            const mla_chunk = mlaChunkBytes(m.kv8);
            const mla_total = m.n_indexed * n_pages * mla_chunk;
            for (0..m.n_indexed) |ord| {
                p.mla_bases[ord] = a.ptr + ord * n_pages * mla_chunk;
                p.gk_bases[ord] = a.ptr + mla_total + ord * n_pages * gk_chunk_bytes;
            }
        }
        // Slot 0 was opened in `cuda_native.open` (`free_streams = 1`); open slots `1 .. streams - 1` now.
        var made: usize = 1;
        errdefer for (1..made) |sl| m.releaseSeq(sl);
        while (made < streams) : (made += 1) {
            if (m.seqs[made] == null) try m.openSeqPaged(made, m.context);
        }
        try s.synchronize();
        std.log.info("hy4 kv pages: {d} streams, {d} pages x {d} tok ({d:.2} GiB arena, kv={s})", .{
            streams,
            n_pages,
            page_tokens,
            @as(f64, @floatFromInt(n_pages * @as(usize, @intCast(pb)))) / (1 << 30),
            if (m.kv8) "fp8" else "bf16",
        });
        return p;
    }

    pub fn deinit(p: *Pages, s: Stream) void {
        cuda.completion.drain(p.d, s, "hy4 page pool deinit did not drain");
        for (&p.leases, &p.used) |*l, *u| {
            if (p.leader and u.*) p.pool.finish(l) else l.ids.deinit(p.gpa);
            u.* = false;
        }
        p.row_slots_dev.free();
        p.pt_dev.free();
        if (p.arena) |*a| a.free();
        p.pool.deinit();
        p.gpa.destroy(p);
    }

    pub fn slotTablePtr(p: *const Pages, slot: usize) u64 {
        return p.pt_dev.ptr + slot * p.pt_stride * 4;
    }

    pub fn mlaRowPtr(p: *const Pages, ord: usize, page_id: u32, in_page: usize) u64 {
        const rb = cuda_cache.mlaRowBytes(p.kv8);
        return p.mla_bases[ord] + (@as(usize, page_id) * page_tokens + in_page) * rb;
    }

    pub fn gkRowPtr(p: *const Pages, ord: usize, page_id: u32, group_in_page: usize) u64 {
        return p.gk_bases[ord] + (@as(usize, page_id) * page_groups + group_in_page) * cuda_index.dim * 2;
    }

    pub fn uploadTable(p: *Pages, s: Stream, slot: usize, ids: []const u32, from: usize) !void {
        if (ids.len == 0) return;
        if (slot >= p.streams or from + ids.len > p.pt_stride) return error.PageTableOverflow;
        for (ids) |id| if (id >= p.n_pages) return error.PageOutOfBounds;
        const dst = p.pt_dev.ptr + (slot * p.pt_stride + from) * 4;
        try cuda.copy_source.enqueue(p.d, dst, std.mem.sliceAsBytes(ids), s);
    }

    pub fn recordFollowerMap(p: *Pages, s: Stream, slot: usize, ids: []const u32, from: usize) !void {
        try p.uploadTable(s, slot, ids, from);
        const l = &p.leases[slot];
        if (from + ids.len > l.ids.items.len) try l.ids.resize(p.gpa, from + ids.len);
        @memcpy(l.ids.items[from..][0..ids.len], ids);
        p.used[slot] = true;
    }

    pub fn releaseSlot(p: *Pages, slot: usize) void {
        if (slot >= fwd.max_slots or !p.used[slot]) return;
        if (p.leader) {
            p.pool.finish(&p.leases[slot]);
        } else {
            p.leases[slot].ids.clearRetainingCapacity();
        }
        p.used[slot] = false;
    }

    pub fn blobChunk(p: *const Pages) u64 {
        return @intCast(mlaChunkBytes(p.kv8));
    }

    pub fn blobRegions(p: *const Pages) usize {
        return @max(p.n_indexed, 1);
    }

    pub fn blobPagesFor(p: *const Pages, bytes: u64) usize {
        if (bytes == 0 or p.n_indexed == 0) return 0;
        return Blob.pagesFor(bytes, p.blobChunk(), p.blobRegions());
    }

    pub fn blobPieceAt(p: *const Pages, region: usize, id: u32) u64 {
        return p.mla_bases[region] + @as(u64, id) * p.blobChunk();
    }
};

/// Uploads `row_caches` (and `row_slots_dev` when paged) for a batched step (`rows <= max_batch`).
pub fn uploadStepTables(m: *fwd.Model, s: Stream, parts: []const fwd.Model.Part, sqs: []const *fwd.Seq, rows: usize) !void {
    if (m.pages) |pg| {
        var rslots: [fwd.max_batch]i32 = @splat(0);
        var rr: usize = 0;
        for (parts) |p| {
            for (0..p.tokens.len) |_| {
                rslots[rr] = @intCast(p.slot);
                rr += 1;
            }
        }
        try cuda.copy_source.enqueue(m.d, pg.row_slots_dev.ptr, std.mem.sliceAsBytes(rslots[0..rows]), s);
    }
    var same = parts.len == m.table_key_parts;
    var mla0s: [@import("cuda_layer.zig").max_segs]u64 = undefined;
    for (parts, sqs, 0..) |p, sq, i| {
        mla0s[i] = sq.cache.buf.ptr;
        if (same and (p.slot != m.table_key_slots[i] or p.tokens.len != m.table_key_lens[i] or mla0s[i] != m.table_key_mla0[i])) {
            same = false;
        }
    }
    if (same) return;
    var table: [config.max_layers * fwd.max_batch]u64 = @splat(0);
    for (0..m.active_layers) |l_idx| {
        var r: usize = 0;
        for (parts, sqs) |p, sq| {
            const at = (try sq.cache.view(l_idx)).mla;
            for (0..p.tokens.len) |_| {
                table[l_idx * fwd.max_batch + r] = at;
                r += 1;
            }
        }
    }
    try cuda.copy_source.enqueue(m.d, m.row_caches.ptr, std.mem.sliceAsBytes(table[0 .. m.active_layers * fwd.max_batch]), s);
    m.table_key_parts = parts.len;
    for (parts, 0..) |p, i| {
        m.table_key_slots[i] = p.slot;
        m.table_key_lens[i] = p.tokens.len;
        m.table_key_mla0[i] = mla0s[i];
    }
}

pub fn dropCheckpointCache(ck: *const core.checkpoint.Checkpoint) void {
    if (comptime @import("builtin").os.tag != .linux) return;
    for (ck.files.items) |*f| {
        const start = f.data & ~@as(usize, 4095);
        if (f.map.memory.len > start) {
            _ = std.os.linux.madvise(@ptrCast(@constCast(f.map.memory.ptr + start)), f.map.memory.len - start, std.os.linux.MADV.DONTNEED);
            _ = std.os.linux.fadvise(f.file.handle, @intCast(start), 0, std.os.linux.POSIX_FADV.DONTNEED);
        }
    }
}

test "hy4 pages: 256-token page geometry matches per-token BF16 and FP8 KV byte counts" {
    try std.testing.expectEqual(@as(u64, 2_211_840), pageBytes(12, true));
    try std.testing.expectEqual(@as(u64, 8_640), pageBytes(12, true) / page_tokens);
    try std.testing.expectEqual(@as(u64, 3_735_552), pageBytes(12, false));
    try std.testing.expectEqual(@as(u64, 14_592), pageBytes(12, false) / page_tokens);
    try std.testing.expectEqual(@as(usize, 4096), tableStride(1_048_576));
    try std.testing.expect(try pagedBy(32, null));
    try std.testing.expect(try pagedBy(0, "1"));
    try std.testing.expect(try pagedBy(0, null));
    try std.testing.expect(!try pagedBy(0, "0"));
    try std.testing.expectError(error.KvShareWithoutPages, pagedBy(16, "0"));
}
