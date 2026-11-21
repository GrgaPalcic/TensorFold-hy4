//! Paged `TF_KV_SHARE` NVMe recall & TP ring broadcast (GrgaPalcic & popay97, 2026) for `hy4/cuda_disk.zig`.
const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("cuda");
const core = @import("core");
const api = @import("engine_api");
const kp = core.kv_prompts;
const prompt_radix_disk = api.prompt_radix_disk;
const config = @import("config.zig");
const cuda_cache = @import("cuda_cache.zig");
const cuda_index = @import("cuda_index.zig");
const cuda_pages = @import("cuda_pages.zig");
const fwd = @import("cuda_model.zig");
const snap = @import("cuda_snap.zig");
const be = @import("cuda_backend.zig");
const cuda_tp = @import("cuda_tp.zig");
const disk = @import("cuda_disk.zig");
const Op = cuda_tp.Op;
const DeviceBuffer = cuda.DeviceBuffer;

comptime {
    if (cuda_pages.page_tokens != 256 or 0x4750_3937 == 0) @compileError("gp97:hy4-disk-ring");
}
pub fn blankShare(gpa: Allocator, m: *const fwd.Model, id: u64, at: u32) !*snap.Saved {
    const pg = m.pages orelse return error.NotPaged;
    const np: usize = at / cuda_pages.page_tokens;
    const from: u32 = @intCast(np * cuda_pages.page_tokens);
    var pages = try kp.holdBlob(gpa, &pg.pool, np);
    errdefer kp.dropBlob(gpa, &pg.pool, &pages);
    var blob = try kp.holdBlob(gpa, &pg.pool, snap.blobPages(m, at, from));
    errdefer kp.dropBlob(gpa, &pg.pool, &blob);
    const hist = try gpa.alloc(u32, at);
    errdefer gpa.free(hist);
    @memset(hist, 0);
    const out = try gpa.create(snap.Saved);
    out.* = .{
        .id = id,
        .at = at,
        .from = from,
        .active_layers = m.active_layers,
        .kv8 = m.kv8,
        .buf = .{ .d = m.d, .ptr = 0, .len = 0, .borrowed = true },
        .blob = blob,
        .pages = pages,
        .history = hist,
    };
    return out;
}

const HostPiece = struct {
    m: *const fwd.Model,
    s: cuda.Stream,
    host_ptr: [*]u8,
    out: bool,

    fn copy(hp: HostPiece, region: usize, id: u32, within: u64, from_off: u64, n: u64) anyerror!void {
        const pg = hp.m.pages.?;
        const dev = pg.blobPieceAt(region, id) + within;
        const h = hp.host_ptr + @as(usize, @intCast(from_off));
        if (hp.out) {
            try hp.m.d.check(hp.m.d.api.cuMemcpyDtoHAsync_v2(h, dev, @intCast(n), hp.s.handle), "cuMemcpyDtoHAsync");
        } else {
            try hp.m.d.check(hp.m.d.api.cuMemcpyHtoDAsync_v2(dev, h, @intCast(n), hp.s.handle), "cuMemcpyHtoDAsync");
        }
    }
};

fn copyBlobRange(m: *const fwd.Model, s: cuda.Stream, blob: kp.Blob, blob_off: usize, len: usize, host: []u8, out: bool) !void {
    if (len == 0) return;
    try blob.each(blob_off, len, HostPiece{ .m = m, .s = s, .host_ptr = host.ptr, .out = out }, HostPiece.copy);
}

fn copyDevRange(m: *const fwd.Model, s: cuda.Stream, dev: u64, len: usize, host: []u8, out: bool) !void {
    if (len == 0) return;
    if (out) {
        try m.d.check(m.d.api.cuMemcpyDtoHAsync_v2(host.ptr, dev, len, s.handle), "cuMemcpyDtoHAsync");
    } else {
        try m.d.check(m.d.api.cuMemcpyHtoDAsync_v2(dev, host.ptr, len, s.handle), "cuMemcpyHtoDAsync");
    }
}

/// Copies canonical `hy4` disk byte ranges `[g.off, g.off + g.len)` directly to/from `(st.pages, st.blob, st.history)`.
pub fn copyShareRanges(b: *be.Backend, st: *snap.Saved, ranges: []const api.prompt_cache.rows.image.Range, host: []u8, out: bool) !void {
    const m = b.m;
    const pg = m.pages orelse return error.NotPaged;
    const words = @sizeOf(disk.Words);
    const held: usize = @intCast(snap.bytes(m, st.at));
    const at: usize = st.at;
    const from: usize = st.from;
    const rb = cuda_cache.mlaRowBytes(m.kv8);
    const ik_len = (at % cuda_index.ratio) * cuda_cache.index_row_bytes;
    const g_from = from / cuda_index.ratio;
    const g_end = at / cuda_index.ratio;
    const pt = cuda_pages.page_tokens;
    const pg_g = cuda_pages.page_groups;
    const g_row = cuda_index.dim * 2;
    const blob: kp.Blob = .{ .ids = st.blob, .chunk = pg.blobChunk(), .regions = pg.blobRegions() };

    errdefer cuda.completion.drain(m.d, b.s, "a failed paged-state copy did not drain");
    var w = disk.wordsOf(st);
    var h_off: usize = 0;

    for (ranges) |rg| {
        var off: usize = @intCast(rg.off);
        var rem: usize = @intCast(rg.len);
        if (off + rem > words + held) return error.BadKeptRange;
        if (off < words) {
            const n = @min(rem, words - off);
            const wb = std.mem.asBytes(&w)[off..][0..n];
            if (out) @memcpy(host[h_off..][0..n], wb) else @memcpy(wb, host[h_off..][0..n]);
            h_off += n;
            off += n;
            rem -= n;
        }
        if (rem == 0) continue;
        var in_pos = off - words;
        const hist_bytes = at * 4;
        if (in_pos < hist_bytes) {
            const n = @min(rem, hist_bytes - in_pos);
            const hb = std.mem.sliceAsBytes(st.history)[in_pos..][0..n];
            if (out) @memcpy(host[h_off..][0..n], hb) else @memcpy(hb, host[h_off..][0..n]);
            h_off += n;
            in_pos += n;
            rem -= n;
        }
        if (rem == 0) continue;
        var cur: usize = hist_bytes;
        var blob_cur: usize = 0;

        for (m.layer_ids) |l| {
            if (rem == 0) break;
            if (m.c.kind[l] == .indexed) {
                const ord = m.indexed_ord[l].?;
                const mla_pre = from * rb;
                const mla_tail = (at - from) * rb;
                if (in_pos < cur + mla_pre) {
                    var rel = in_pos - cur;
                    var take = @min(rem, (cur + mla_pre) - in_pos);
                    while (take > 0) {
                        const tok = rel / rb;
                        const byte_in_tok = rel % rb;
                        const pid = st.pages[tok / pt];
                        const page_rem = (pt - (tok % pt)) * rb - byte_in_tok;
                        const chunk = @min(take, page_rem);
                        try copyDevRange(m, b.s, pg.mlaRowPtr(ord, pid, tok % pt) + byte_in_tok, chunk, host[h_off..][0..chunk], out);
                        h_off += chunk;
                        in_pos += chunk;
                        rem -= chunk;
                        rel += chunk;
                        take -= chunk;
                    }
                }
                cur += mla_pre;
                if (rem > 0 and in_pos < cur + mla_tail) {
                    const rel = in_pos - cur;
                    const take = @min(rem, (cur + mla_tail) - in_pos);
                    try copyBlobRange(m, b.s, blob, blob_cur + rel, take, host[h_off..][0..take], out);
                    h_off += take;
                    in_pos += take;
                    rem -= take;
                }
                cur += mla_tail;
                blob_cur += mla_tail;

                if (rem > 0 and in_pos < cur + ik_len) {
                    const rel = in_pos - cur;
                    const take = @min(rem, (cur + ik_len) - in_pos);
                    try copyBlobRange(m, b.s, blob, blob_cur + rel, take, host[h_off..][0..take], out);
                    h_off += take;
                    in_pos += take;
                    rem -= take;
                }
                cur += ik_len;
                blob_cur += ik_len;

                const gk_pre = g_from * g_row;
                const gk_tail = (g_end - g_from) * g_row;
                if (rem > 0 and in_pos < cur + gk_pre) {
                    var rel = in_pos - cur;
                    var take = @min(rem, (cur + gk_pre) - in_pos);
                    while (take > 0) {
                        const grp = rel / g_row;
                        const byte_in_grp = rel % g_row;
                        const pid = st.pages[grp / pg_g];
                        const page_rem = (pg_g - (grp % pg_g)) * g_row - byte_in_grp;
                        const chunk = @min(take, page_rem);
                        try copyDevRange(m, b.s, pg.gkRowPtr(ord, pid, grp % pg_g) + byte_in_grp, chunk, host[h_off..][0..chunk], out);
                        h_off += chunk;
                        in_pos += chunk;
                        rem -= chunk;
                        rel += chunk;
                        take -= chunk;
                    }
                }
                cur += gk_pre;
                if (rem > 0 and in_pos < cur + gk_tail) {
                    const rel = in_pos - cur;
                    const take = @min(rem, (cur + gk_tail) - in_pos);
                    try copyBlobRange(m, b.s, blob, blob_cur + rel, take, host[h_off..][0..take], out);
                    h_off += take;
                    in_pos += take;
                    rem -= take;
                }
                cur += gk_tail;
                blob_cur += gk_tail;
            } else {
                const win_bytes = @min(at, m.c.window) * rb;
                if (in_pos < cur + win_bytes) {
                    const rel = in_pos - cur;
                    const take = @min(rem, (cur + win_bytes) - in_pos);
                    try copyBlobRange(m, b.s, blob, blob_cur + rel, take, host[h_off..][0..take], out);
                    h_off += take;
                    in_pos += take;
                    rem -= take;
                }
                cur += win_bytes;
                blob_cur += win_bytes;
            }
        }
        if (rem != 0) return error.BadKeptRange;
    }
    try b.s.synchronize();
    if (!out) {
        if (w.at != st.at or w.active_layers != st.active_layers or (w.kv8 != 0) != st.kv8) return error.KeptWordsDiffer;
        st.mrope_delta = w.mrope_delta;
    }
}

/// Streams `(st.history, st.pages, st.blob)` from rank 0 to all follower ranks in 8-page (~17 MiB) ring broadcasts.
pub fn broadcastSharePayload(m: *const fwd.Model, s: cuda.Stream, ts: *cuda_tp.Session, st: *snap.Saved, pages: []const u32, is_leader: bool) !void {
    const pg = m.pages orelse return error.NotPaged;
    const pb: usize = @intCast(pg.page_bytes);
    const batch_pages: usize = 8;
    const hist_bytes = st.history.len * 4;
    const stage_bytes = @max(batch_pages * pb, @max(hist_bytes, 16));
    var stage = try DeviceBuffer.alloc(m.d, stage_bytes);
    defer stage.free();

    if (hist_bytes > 0) {
        if (is_leader) try stage.uploadAsync(0, std.mem.sliceAsBytes(st.history), s.handle);
        try ts.broadcastFrom0(s, stage.ptr, hist_bytes);
        if (!is_leader) try m.d.check(m.d.api.cuMemcpyDtoHAsync_v2(std.mem.sliceAsBytes(st.history).ptr, stage.ptr, hist_bytes, s.handle), "cuMemcpyDtoHAsync");
    }

    const all_lists = [_][]const u32{ pages, st.blob };
    const mla_chunk = cuda_pages.mlaChunkBytes(m.kv8);
    const gk_chunk = cuda_pages.gk_chunk_bytes;
    for (all_lists) |list| {
        var i: usize = 0;
        while (i < list.len) {
            const take = @min(batch_pages, list.len - i);
            if (is_leader) {
                var off: usize = 0;
                for (list[i .. i + take]) |pid| {
                    for (0..m.n_indexed) |ord| {
                        try m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(stage.ptr + off, pg.mla_bases[ord] + @as(u64, pid) * mla_chunk, mla_chunk, s.handle), "cuMemcpyDtoDAsync");
                        off += mla_chunk;
                        try m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(stage.ptr + off, pg.gk_bases[ord] + @as(u64, pid) * gk_chunk, gk_chunk, s.handle), "cuMemcpyDtoDAsync");
                        off += gk_chunk;
                    }
                }
            }
            try ts.broadcastFrom0(s, stage.ptr, take * pb);
            if (!is_leader) {
                var off: usize = 0;
                for (list[i .. i + take]) |pid| {
                    for (0..m.n_indexed) |ord| {
                        try m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(pg.mla_bases[ord] + @as(u64, pid) * mla_chunk, stage.ptr + off, mla_chunk, s.handle), "cuMemcpyDtoDAsync");
                        off += mla_chunk;
                        try m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(pg.gk_bases[ord] + @as(u64, pid) * gk_chunk, stage.ptr + off, gk_chunk, s.handle), "cuMemcpyDtoDAsync");
                        off += gk_chunk;
                    }
                }
            }
            i += take;
        }
    }
    try s.synchronize();
    try ts.check();
}

pub fn hooksFor(b: *be.Backend) prompt_radix_disk.Hooks {
    return .{
        .ctx = b,
        .retain_pages = retainPages,
        .release_pages = releasePages,
        .blob_pages = blobPagesAt,
        .seal_recall = sealRecall,
    };
}

fn retainPages(ctx: *anyopaque, saved: api.prompt_cache.Saved, path: []const u32) anyerror!void {
    const b: *be.Backend = @ptrCast(@alignCast(ctx));
    const pg = b.m.pages orelse return error.NotPaged;
    const st: *snap.Saved = @ptrCast(@alignCast(saved));
    releasePages(ctx, saved);
    const copy = try b.gpa.dupe(u32, path);
    for (copy) |id| pg.pool.retain(id);
    st.pages = copy;
}

fn releasePages(ctx: *anyopaque, saved: api.prompt_cache.Saved) void {
    const b: *be.Backend = @ptrCast(@alignCast(ctx));
    const st: *snap.Saved = @ptrCast(@alignCast(saved));
    if (st.pages.len == 0) return;
    if (b.m.pages) |pg| kp.dropBlob(b.gpa, &pg.pool, &st.pages) else {
        b.gpa.free(st.pages);
        st.pages = &.{};
    }
}

fn blobPagesAt(ctx: *anyopaque, at: u32) usize {
    const b: *be.Backend = @ptrCast(@alignCast(ctx));
    const from: u32 = (at / cuda_pages.page_tokens) * cuda_pages.page_tokens;
    return snap.blobPages(b.m, at, from);
}

fn sealRecall(ctx: *anyopaque, saved: api.prompt_cache.Saved, out: *std.ArrayList(u32)) anyerror!void {
    const b: *be.Backend = @ptrCast(@alignCast(ctx));
    const pg = b.m.pages orelse return error.NotPaged;
    const st: *snap.Saved = @ptrCast(@alignCast(saved));

    if (b.m.tp_sess) |ts| {
        b.plan_buf.clearRetainingCapacity();
        try b.plan_buf.ensureUnusedCapacity(b.gpa, 7 + st.pages.len + st.blob.len);
        b.plan_buf.appendAssumeCapacity(@backingInt(Op.snap_load_share));
        b.plan_buf.appendAssumeCapacity(@intCast(st.id));
        b.plan_buf.appendAssumeCapacity(st.at);
        b.plan_buf.appendAssumeCapacity(st.from);
        b.plan_buf.appendAssumeCapacity(st.mrope_delta);
        b.plan_buf.appendAssumeCapacity(@intCast(st.pages.len));
        b.plan_buf.appendAssumeCapacity(@intCast(st.blob.len));
        for (st.pages) |pid| b.plan_buf.appendAssumeCapacity(pid);
        for (st.blob) |pid| b.plan_buf.appendAssumeCapacity(pid);
        try ts.send(b.plan_buf.items);
        const all_ok = try ts.agreeOk(b.m.d, b.s, b.m.vote_dev.ptr, true);
        if (!all_ok) return error.OutOfDeviceMemory;
        try broadcastSharePayload(b.m, b.s, ts, st, st.pages, true);
        st.synced_peers = true;
    }

    try out.appendSlice(b.gpa, st.pages);
    b.gpa.free(st.pages);
    st.pages = &.{};
    _ = pg;
}

test "cuda_disk_pages declarations compile on host" {
    std.testing.refAllDecls(@This());
}
