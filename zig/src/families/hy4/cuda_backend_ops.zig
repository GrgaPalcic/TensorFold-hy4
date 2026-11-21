//! Speculative round `keep`/`draft` and prompt-cache snapshot operations for `hy4.cuda_backend.Backend`.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const core = @import("core");
const cuda_pages = @import("cuda_pages.zig");
const fwd = @import("cuda_model.zig");
const cuda_dfly = @import("cuda_dfly.zig");
const cuda_prof = @import("cuda_prof.zig");
const snap = @import("cuda_snap.zig");
const cuda_tp = @import("cuda_tp.zig");
const cb = @import("cuda_backend.zig");
const vcfg = @import("vision_config.zig");
const cmedia = @import("cuda_media.zig");

const be = lanes.backend;
const Stream = lanes.Stream;
const Op = cuda_tp.Op;
const Backend = cb.Backend;
const Lane = cb.Lane;
const kp = core.kv_prompts;
const kv = core.kv_pool;

pub fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
    const b: *Backend = @ptrCast(@alignCast(ptr));
    const kt0 = if (b.m.prof != null) cuda_prof.nowNs() else 0;
    b.plan_buf.clearRetainingCapacity();
    if (b.m.tp_sess != null) {
        try b.plan_buf.append(b.gpa, @backingInt(Op.keep_round));
        try b.plan_buf.append(b.gpa, @intCast(windows.len));
    }
    var tot_rows: usize = 0;
    var tot_held: usize = 0;
    var tot_copy: usize = 0;
    var tot_kept: usize = 0;
    for (windows, paths) |w, path| {
        const l = b.lanes_map.getPtr(w.stream) orelse return error.UnknownStream;
        if (path.len == 0 or path.len > l.rows.items.len) return error.EmptyPath;
        for (path, 0..) |r, i| if (r != i) return error.ChainsOnly;
        tot_rows += w.rows();
        tot_held += w.held;
        tot_copy += w.tokens.len;
        tot_kept += path.len;
        l.len = l.base + path.len;
        if (b.m.seqs[l.slot]) |*sq| sq.len = @intCast(l.len);
        if (b.m.tp_sess != null) {
            try b.plan_buf.append(b.gpa, @intCast(l.slot));
            try b.plan_buf.append(b.gpa, @intCast(l.len));
        }
    }
    if (b.m.tp_sess) |ts| try ts.send(b.plan_buf.items);
    if (b.m.prof != null) {
        b.prof_round_n += 1;
        const k_ms = @as(f64, @floatFromInt(cuda_prof.nowNs() - kt0)) / 1e6;
        const v_ms = @as(f64, @floatFromInt(b.prof_verify_ns)) / 1e6;
        const d_ms = @as(f64, @floatFromInt(b.prof_draft_ns)) / 1e6;
        if (b.prof_round_n <= 4 or b.prof_round_n % 16 == 0) {
            std.log.info(
                "[HY4P:ROUND] #{d} streams={d} rows={d} held={d} copy={d} kept={d} next_draft_n={d} max_d={d} verify={d:.2} draft={d:.2} keep={d:.2}",
                .{ b.prof_round_n, windows.len, tot_rows, tot_held, tot_copy, tot_kept, b.prof_draft_asked, b.prof_draft_max_d, v_ms, d_ms, k_ms },
            );
        }
    }
}

pub fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
    const b: *Backend = @ptrCast(@alignCast(ptr));
    const dt0 = if (b.m.prof != null) cuda_prof.nowNs() else 0;
    if (!b.m.hasDfly()) {
        for (requests) |r| if (r.depth != 0 or r.lanes != null) return error.NoDraftHead;
        return;
    }
    var items: [fwd.max_slots]fwd.Model.DraftItem = undefined;
    var lanes_ptrs: [fwd.max_slots]*Lane = undefined;
    var depths: [fwd.max_slots]usize = undefined;
    var count: usize = 0;
    for (requests) |r| {
        if (r.lanes != null) return error.TreesNotBuilt;
        const l = b.lanes_map.getPtr(r.stream) orelse return error.UnknownStream;
        l.held_n = 0;
        if (r.depth == 0) continue;
        // shared_round.zig calls draft() before keep(), so accepted length is l.base + rows.len.
        const kept_len: u64 = if (r.rows) |rows| blk: {
            if (r.start != l.base) return error.PositionMismatch;
            break :blk l.base + rows.len;
        } else l.len;
        if (r.position != kept_len + 1) return error.PositionMismatch;
        const d: usize = @min(r.depth, cuda_dfly.max_drafts);
        var anchor_tok: u32 = 0;
        var src_row: usize = 0;
        if (r.rows) |rows| {
            if (rows.len == 0 or r.follow.len < rows.len) return error.BadDraftRequest;
            for (rows, 0..) |row, j| if (row != j) return error.TreesNotBuilt;
            anchor_tok = r.follow[rows.len - 1];
            src_row = rows.len - 1;
        } else {
            anchor_tok = try b.feedVal(r.first orelse return error.NoFirstToken);
            src_row = 0;
        }
        if (count >= fwd.max_slots) return error.TooManyStreams;
        items[count] = .{ .slot = l.slot, .anchor_tok = anchor_tok, .anchor_pos = @intCast(kept_len), .src_row = src_row };
        lanes_ptrs[count] = l;
        depths[count] = d;
        count += 1;
    }
    if (count == 0) return;
    var host_out: [cuda_dfly.max_reqs * cuda_dfly.max_drafts]u32 = undefined;
    var off: usize = 0;
    while (off < count) {
        const batch_n = @min(count - off, cuda_dfly.max_reqs);
        var max_d: usize = 1;
        for (depths[off..][0..batch_n]) |d| max_d = @max(max_d, d);
        if (b.m.tp_sess) |ts| {
            b.plan_buf.clearRetainingCapacity();
            try b.plan_buf.append(b.gpa, @backingInt(Op.draft_round));
            try b.plan_buf.append(b.gpa, @intCast(batch_n));
            try b.plan_buf.append(b.gpa, @intCast(max_d));
            for (items[off..][0..batch_n]) |it| {
                try b.plan_buf.append(b.gpa, @intCast(it.slot));
                try b.plan_buf.append(b.gpa, @intCast(it.anchor_tok));
                try b.plan_buf.append(b.gpa, @intCast(it.anchor_pos));
                try b.plan_buf.append(b.gpa, @intCast(it.src_row));
            }
            try ts.send(b.plan_buf.items);
        }
        try b.m.draftDfly(b.k, b.s, items[off..][0..batch_n], max_d);
        try b.s.synchronize();
        if (b.m.prof) |*p| p.finish(b.s);
        if (b.m.tp_sess) |ts| try ts.check();
        const dsc = if (b.m.dfly_sc) |*p| p else return error.NoDraftHead;
        try (cuda.DeviceBuffer{ .d = b.m.d, .ptr = dsc.draft_out, .len = batch_n * max_d * 4 }).download(0, std.mem.sliceAsBytes(host_out[0 .. batch_n * max_d]));
        for (0..batch_n) |bi| {
            const l = lanes_ptrs[off + bi];
            const d = depths[off + bi];
            var valid: usize = 0;
            for (0..d) |j| {
                const tok = host_out[bi * max_d + j];
                if (tok >= b.m.c.vocab) break;
                l.held[j] = tok;
                valid += 1;
            }
            l.held_n = valid;
        }
        if (b.m.prof != null) {
            b.prof_draft_asked += batch_n;
            b.prof_draft_max_d = @max(b.prof_draft_max_d, max_d);
        }
        off += batch_n;
    }
    if (b.m.prof != null) b.prof_draft_ns = cuda_prof.nowNs() - dt0;
}

pub fn held(ptr: *anyopaque, s: *Stream, out: []u32) anyerror!void {
    const b: *Backend = @ptrCast(@alignCast(ptr));
    const l = b.lanes_map.getPtr(s) orelse return error.UnknownStream;
    if (out.len > l.held_n) return error.NoHeldDrafts;
    @memcpy(out, l.held[0..out.len]);
}

/// Runs GPU `sample.cu` draws for a verified batch and unpacks sampled tokens + draft echoes into `out`.
pub fn finishSampledBatch(b: *Backend, windows: []const be.Window, out: []be.Verified, parts: []const fwd.Model.Part, rows: usize) !void {
    var rules_buf: [fwd.max_batch]cuda.sample.Rule = undefined;
    var metas_buf: [fwd.max_batch]i32 = undefined;
    var picked_buf: [fwd.max_batch]u32 = undefined;
    var r0: usize = 0;
    for (windows, parts) |w, p| {
        const rule = cuda.sample.rule(w.stream.sampling.?);
        for (0..p.tokens.len) |r| {
            rules_buf[r0 + r] = rule;
            metas_buf[r0 + r] = @intCast(w.positions[r] - 1);
        }
        r0 += p.tokens.len;
    }
    try b.m.sampleGpu(b.k, b.s, rules_buf[0..rows], metas_buf[0..rows], picked_buf[0..rows]);
    r0 = 0;
    for (windows, out, parts) |w, *o, p| {
        const n = p.tokens.len;
        @memcpy(o.sampled[0..n], picked_buf[r0..][0..n]);
        @memcpy(o.drafts[0 .. n - 1], p.tokens[1..]);
        b.lanes_map.getPtr(w.stream).?.len = p.first + n;
        r0 += n;
    }
}

pub fn blankState(b: *Backend, at: u32) !*snap.Saved {
    const id = b.next_snap;
    b.next_snap += 1;
    if (b.prompts != null) return @import("cuda_disk_pages.zig").blankShare(b.gpa, b.m, id, at);
    return snap.blank(b.gpa, &b.spares, b.m, id, at);
}

pub fn trimSpares(b: *Backend, room: u64) void {
    if (b.spares.bytes() <= room) return;
    if (b.m.tp_sess) |ts| ts.send(&.{ @backingInt(Op.snap_trim), @bitCast(room) }) catch {};
    b.spares.trim(room);
}

pub fn saveState(b: *Backend, owner: ?*anyopaque, at: u32) !*snap.Saved {
    const s: *const Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
    const l = b.lanes_map.getPtr(s) orelse return error.UnknownStream;
    const id = b.next_snap;
    b.next_snap += 1;
    if (b.prompts != null) return saveShareState(b, l.slot, id, at);
    if (b.m.tp_sess) |ts| {
        try ts.send(&.{ @backingInt(Op.snap_save), @intCast(id), @intCast(l.slot), at });
        const saved_opt = snap.save(b.gpa, &b.spares, b.m, b.s, l.slot, id, at) catch null;
        const all_ok = try ts.agreeOk(b.m.d, b.s, b.m.vote_dev.ptr, saved_opt != null);
        if (!all_ok) {
            if (saved_opt) |sv| snap.drop(b.gpa, &b.spares, b.s, sv);
            return error.OutOfDeviceMemory;
        }
        try b.s.synchronize();
        saved_opt.?.synced_peers = true;
        return saved_opt.?;
    }
    const sv = try snap.save(b.gpa, &b.spares, b.m, b.s, l.slot, id, at);
    try b.s.synchronize();
    return sv;
}

fn saveShareState(b: *Backend, slot: usize, id: u64, at: u32) !*snap.Saved {
    const pg = b.m.pages orelse return error.NotPaged;
    const from: u32 = (at / cuda_pages.page_tokens) * cuda_pages.page_tokens;
    var blob = try kp.holdBlob(b.gpa, &pg.pool, snap.blobPages(b.m, at, from));
    errdefer kp.dropBlob(b.gpa, &pg.pool, &blob);
    if (b.m.tp_sess) |ts| {
        b.plan_buf.clearRetainingCapacity();
        try b.plan_buf.ensureUnusedCapacity(b.gpa, 6 + blob.len);
        b.plan_buf.appendAssumeCapacity(@backingInt(Op.snap_save_share));
        b.plan_buf.appendAssumeCapacity(@intCast(id));
        b.plan_buf.appendAssumeCapacity(@intCast(slot));
        b.plan_buf.appendAssumeCapacity(at);
        b.plan_buf.appendAssumeCapacity(from);
        b.plan_buf.appendAssumeCapacity(@intCast(blob.len));
        for (blob) |pid| b.plan_buf.appendAssumeCapacity(pid);
        try ts.send(b.plan_buf.items);
        const saved_opt = snap.saveShare(b.gpa, b.m, b.s, slot, id, at, from, blob) catch null;
        const all_ok = try ts.agreeOk(b.m.d, b.s, b.m.vote_dev.ptr, saved_opt != null);
        if (!all_ok) {
            if (saved_opt) |sv| {
                b.s.synchronize() catch {};
                sv.blob = &.{};
                snap.drop(b.gpa, &b.spares, b.s, sv);
            }
            return error.OutOfDeviceMemory;
        }
        try b.s.synchronize();
        saved_opt.?.synced_peers = true;
        return saved_opt.?;
    }
    const sv = try snap.saveShare(b.gpa, b.m, b.s, slot, id, at, from, blob);
    try b.s.synchronize();
    return sv;
}

fn syncRecalledState(b: *Backend, k: *snap.Saved) !void {
    const ts = b.m.tp_sess orelse return;
    if (k.synced_peers) return;
    try ts.send(&.{ @backingInt(Op.snap_load), @intCast(k.id), @intCast(k.at) });
    const all_ok = try ts.agreeOk(b.m.d, b.s, b.m.vote_dev.ptr, true);
    if (!all_ok) return error.OutOfDeviceMemory;
    try ts.broadcastFrom0(b.s, k.buf.ptr, k.buf.len);
    try b.s.synchronize();
    try ts.check();
    k.synced_peers = true;
}

pub fn restoreSlot(b: *Backend, slot: usize, k: *const snap.Saved) !void {
    try syncRecalledState(b, @constCast(k));
    if (b.m.tp_sess) |ts| try ts.send(&.{ @backingInt(Op.snap_restore), @intCast(k.id), @intCast(slot) });
    try snap.restore(b.m, b.s, slot, k);
    try b.s.synchronize();
}

pub fn restoreState(b: *Backend, owner: ?*anyopaque, k: *const snap.Saved) !void {
    if (b.prompts != null) {
        b.pending_restore = k;
        return;
    }
    const s: *const Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
    const l = b.lanes_map.getPtr(s) orelse return error.UnknownStream;
    try restoreSlot(b, l.slot, k);
    l.len = k.at;
}

pub fn dropState(b: *Backend, k: *snap.Saved) void {
    if (k.synced_peers) if (b.m.tp_sess) |ts| ts.send(&.{ @backingInt(Op.snap_drop), @intCast(k.id) }) catch {};
    if (k.pages.len > 0 or k.blob.len > 0) b.s.synchronize() catch {};
    if (k.pages.len > 0) {
        if (b.m.pages) |pg| kp.dropBlob(b.gpa, &pg.pool, &k.pages) else {
            b.gpa.free(k.pages);
            k.pages = &.{};
        }
    }
    if (k.blob.len > 0) {
        if (b.m.pages) |pg| kp.dropBlob(b.gpa, &pg.pool, &k.blob) else {
            b.gpa.free(k.blob);
            k.blob = &.{};
        }
    }
    snap.drop(b.gpa, &b.spares, b.s, k);
}

const TableMap = struct {
    b: *Backend,
    pg: *cuda_pages.Pages,
    slot: usize,

    fn write(ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void {
        const tm: *const TableMap = @ptrCast(@alignCast(ctx));
        const tail = ids[from..];
        if (tail.len == 0) return;
        if (tm.b.m.tp_sess) |ts| {
            tm.b.plan_buf.clearRetainingCapacity();
            try tm.b.plan_buf.ensureUnusedCapacity(tm.b.gpa, 4 + tail.len);
            tm.b.plan_buf.appendAssumeCapacity(@backingInt(Op.map_pages));
            tm.b.plan_buf.appendAssumeCapacity(@intCast(tm.slot));
            tm.b.plan_buf.appendAssumeCapacity(@intCast(from));
            tm.b.plan_buf.appendAssumeCapacity(@intCast(tail.len));
            for (tail) |id| tm.b.plan_buf.appendAssumeCapacity(id);
            try ts.send(tm.b.plan_buf.items);
        }
        try tm.pg.uploadTable(tm.b.s, tm.slot, tail, from);
    }
};

fn leaseSlotAllRanks(b: *Backend, pg: *cuda_pages.Pages, slot: usize, prompt_len: usize, max_new: usize, adopt: []const u32) !void {
    const n = cuda_pages.needPages(prompt_len, max_new);
    const lease = &pg.leases[slot];
    pg.pool.promise(lease, n -| adopt.len) catch |e| {
        for (adopt) |id| pg.pool.release(id);
        return e;
    };
    pg.used[slot] = true;
    errdefer pg.releaseSlot(slot);
    try pg.pool.adopt(lease, adopt);
    var tm: TableMap = .{ .b = b, .pg = pg, .slot = slot };
    const table: kv.Table = .{ .ctx = &tm, .write = TableMap.write };
    if (adopt.len > 0) try table.write(table.ctx, lease.ids.items, 0);
    try pg.pool.cover(lease, n, table);
}

fn buildImages(gpa: std.mem.Allocator, imgs: []const lanes.stream.Image) ![]kp.Image {
    if (imgs.len == 0) return &.{};
    const out = try gpa.alloc(kp.Image, imgs.len);
    for (imgs, out) |im, *o| o.* = .{ .at = im.at, .tokens = im.tokens, .gh = im.gh, .gw = im.gw, .hash = im.hash };
    return out;
}

pub fn opening(b: *Backend, s: *Stream) !*Lane {
    const p = s.prompt();
    if (p.len == 0) return error.EmptyPrompt;
    if (!cb.fits(b.m.context, p.len, s.max_new)) return error.PromptTooLong;
    const got = try b.lanes_map.getOrPut(b.gpa, s);
    if (got.found_existing and b.prompts != null) return error.StreamAlreadyOpen;
    if (!got.found_existing) {
        const slot_lim = if (b.m.pages) |pg| pg.streams else fwd.max_slots;
        const slot = std.mem.indexOfScalar(bool, b.free[0..slot_lim], true) orelse return error.NoFreeSlot;
        if (b.m.pages == null) {
            const cap = p.len + @as(usize, s.max_new) + fwd.max_window;
            b.openSeqAllRanks(slot, cap) catch |e| {
                b.lanes_map.removeByPtr(got.key_ptr);
                return e;
            };
        } else if (b.m.seqs[slot]) |*sq| {
            sq.len = 0;
            sq.mrope_delta = 0;
            if (sq.media) |*sm| {
                sm.deinit();
                sq.media = null;
            }
        }
        const ll = b.gpa.alloc(f32, b.m.c.vocab) catch |e| {
            if (b.m.pages == null) b.releaseSeqAllRanks(slot);
            b.lanes_map.removeByPtr(got.key_ptr);
            return e;
        };
        if (b.m.pages) |pg| {
            var adopt: std.ArrayList(u32) = .empty;
            defer adopt.deinit(b.gpa);
            var plan: kp.Plan = .{};
            b.pending_restore = null;
            if (b.prompts) |pr| {
                const imgs = buildImages(b.gpa, s.images) catch |e| {
                    b.gpa.free(ll);
                    b.lanes_map.removeByPtr(got.key_ptr);
                    return e;
                };
                defer if (imgs.len > 0) b.gpa.free(imgs);
                plan = pr.plan(b.gpa, &pg.pool, .{ .key = p, .prompt_len = @intCast(p.len), .images = imgs }, s.history_len, s.shared_prefixes, @ptrCast(s), &adopt) catch |e| {
                    b.gpa.free(ll);
                    b.lanes_map.removeByPtr(got.key_ptr);
                    return e;
                };
            }
            leaseSlotAllRanks(b, pg, slot, p.len, s.max_new, adopt.items) catch |e| {
                if (plan.marks.len > 0) b.gpa.free(plan.marks);
                b.gpa.free(ll);
                b.lanes_map.removeByPtr(got.key_ptr);
                return e;
            };
            if (b.pending_restore) |k| {
                b.pending_restore = null;
                restoreSlot(b, slot, k) catch |e| {
                    pg.releaseSlot(slot);
                    if (b.m.tp_sess) |ts| ts.send(&.{ @backingInt(Op.release_pages), @intCast(slot) }) catch {};
                    if (plan.marks.len > 0) b.gpa.free(plan.marks);
                    b.gpa.free(ll);
                    b.lanes_map.removeByPtr(got.key_ptr);
                    return e;
                };
            }
            b.free[slot] = false;
            got.value_ptr.* = .{ .slot = slot, .last_logits = ll, .marks = plan.marks, .share_from = plan.from };
        } else {
            b.free[slot] = false;
            got.value_ptr.* = .{ .slot = slot, .last_logits = ll };
        }
    }
    const l = got.value_ptr;
    s.cached = 0;
    l.first_tok = null;
    if (b.prompts != null) {
        l.len = l.share_from;
        l.fill = l.share_from;
        s.cached = l.share_from;
    } else {
        l.fill = resumeLegacy(b, s, l, p.len);
    }
    s.fill_at.store(@intCast(l.fill.?), .monotonic);
    try setupMedia(b, s, l.slot, l.fill.?);
    return l;
}

fn resumeLegacy(b: *Backend, s: *Stream, l: *Lane, len: usize) usize {
    const saved = s.reuse.saved orelse return 0;
    const k: *const snap.Saved = @ptrCast(@alignCast(saved));
    if (k.at != s.reuse.at or k.at == 0 or k.at >= len) {
        s.reuse_failed = true;
        return 0;
    }
    restoreSlot(b, l.slot, k) catch {
        s.reuse_failed = true;
        return 0;
    };
    l.len = k.at;
    s.cached = k.at;
    return k.at;
}

pub fn afterChunk(b: *Backend, s: *Stream, l: *Lane, end: usize, done: bool) !void {
    const p = s.prompt();
    if (b.prompts) |pr| {
        if (std.mem.indexOfScalar(u32, l.marks, @intCast(end)) != null) {
            b.s.synchronize() catch {};
            const pg = b.m.pages.?;
            const imgs = try buildImages(b.gpa, s.images);
            defer if (imgs.len > 0) b.gpa.free(imgs);
            var tm: TableMap = .{ .b = b, .pg = pg, .slot = l.slot };
            if (try pr.keepPages(b.gpa, &pg.pool, &pg.leases[l.slot], .{ .key = p, .prompt_len = @intCast(p.len), .images = imgs }, end, @ptrCast(s), cuda_pages.page_tokens, .{ .ctx = &tm, .write = TableMap.write })) {
                l.kept += 1;
            }
        }
        if (done) pr.report(p.len, l.share_from, l.kept);
        return;
    }
    if (s.reuse.hook) |hk| if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(end)) != null) {
        b.s.synchronize() catch {};
        hk.at(hk.ptr, s, @intCast(end));
    };
}

pub fn releaseLane(b: *Backend, l: *Lane) void {
    if (b.m.pages) |pg| {
        if (b.m.seqs[l.slot]) |*sq| cmedia.clearSeq(&sq.media, &sq.mrope_delta);
        if (b.m.tp_sess) |ts| ts.send(&.{ @backingInt(Op.release_pages), @intCast(l.slot) }) catch {};
        pg.releaseSlot(l.slot);
    } else {
        b.releaseSeqAllRanks(l.slot);
    }
    b.free[l.slot] = true;
    if (l.marks.len > 0) b.gpa.free(l.marks);
    l.rows.deinit(b.gpa);
    b.gpa.free(l.last_logits);
}

/// Prepares per-sequence multimodal features & 3D M-RoPE (`s.images`) at `opening` (skips ViT when cached past images).
pub fn setupMedia(b: *Backend, s: *Stream, slot: usize, resumed_at: usize) !void {
    const sq = if (b.m.seqs[slot]) |*p| p else return error.UnknownSlot;
    if (sq.media) |*old| {
        old.deinit();
        sq.media = null;
    }
    const imgs = s.images;
    if (imgs.len == 0) {
        if (resumed_at == 0 or b.prompts != null) sq.mrope_delta = 0;
        if (b.m.tp_sess) |ts| try ts.send(&.{ @backingInt(Op.clear_media), @intCast(slot), sq.mrope_delta });
        return;
    }
    const vis = b.vis orelse return error.NoHy4Vision;
    const p = s.prompt();
    var need_encode = false;
    var runs: [64]vcfg.Run = undefined;
    if (imgs.len > vis.limits.max_images or imgs.len > runs.len) return error.BadImageCount;
    for (imgs, 0..) |im, i| {
        runs[i] = .{ .at = im.at, .gh = im.gh, .gw = im.gw };
        if (im.at + im.tokens > resumed_at) need_encode = true;
    }
    if (!need_encode) {
        const src_tmp = try b.gpa.alloc(i32, p.len);
        defer b.gpa.free(src_tmp);
        const pos3_tmp = try b.gpa.alloc(i32, 3 * p.len);
        defer b.gpa.free(pos3_tmp);
        const delta = vcfg.layout(p.len, runs[0..imgs.len], src_tmp, pos3_tmp);
        sq.mrope_delta = delta;
        if (b.m.tp_sess) |ts| try ts.send(&.{ @backingInt(Op.clear_media), @intCast(slot), delta });
        return;
    }
    const idx_freqs = if (b.m.first_idx_layer) |fi| b.m.index_weights[fi].?.freqs.ptr else 0;
    const sm = try cmedia.prepareRank0(
        b.gpa,
        b.m.d,
        b.s,
        vis,
        &b.k.layer.mlp.rows,
        &b.k.layer.mlp.norms,
        b.k.idx.rope_table,
        idx_freqs,
        b.m.tp_sess,
        slot,
        p,
        imgs,
        resumed_at,
    );
    sq.mrope_delta = sm.delta;
    sq.media = sm;
}
