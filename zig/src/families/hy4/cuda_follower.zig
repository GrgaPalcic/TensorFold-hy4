//! Follower-rank (`rank > 0`) `PlanLink` replay loop for Hy4-flash TP serving.
const std = @import("std");
const be = @import("cuda_backend.zig");
const fwd = @import("cuda_model.zig");
const cuda_dfly = @import("cuda_dfly.zig");
const batched = @import("cuda_batch.zig");
const snap = @import("cuda_snap.zig");
const cuda_tp = @import("cuda_tp.zig");
const cmedia = @import("cuda_media.zig");

const Op = cuda_tp.Op;

/// Zero-host-read follower loop (`rank > 0`): reads `PlanLink` frames and queues all 42 layers onto `b.s`.
pub fn loop(b: *be.Backend) !void {
    const ts = b.m.tp_sess orelse return error.NotFollower;
    var tokens_tmp: std.ArrayList(u32) = .empty;
    defer tokens_tmp.deinit(b.gpa);
    while (true) {
        const msg = ts.recvGrow(&b.plan_buf) catch |err| switch (err) {
            error.PeerClosed => return,
            else => return err,
        };
        if (msg.len == 0) continue;
        const op: Op = @fromBackingInt(@intCast(msg[0]));
        switch (op) {
            .stop => return,
            .open_seq => {
                const slot: usize = @intCast(msg[1]);
                const cap: usize = @intCast(msg[2]);
                const local_ok = if (b.m.openSeq(slot, cap)) |_| true else |_| false;
                const all_ok = try ts.agreeOk(b.m.d, b.s, b.m.vote_dev.ptr, local_ok);
                if (!all_ok and local_ok) b.m.releaseSeq(slot);
            },
            .release_seq => if (b.m.pages) |pg| {
                const slot: usize = @intCast(msg[1]);
                if (b.m.seqs[slot]) |*sq| cmedia.clearSeq(&sq.media, &sq.mrope_delta);
                pg.releaseSlot(slot);
            } else b.m.releaseSeq(@intCast(msg[1])),
            .release_pages => {
                const slot: usize = @intCast(msg[1]);
                if (b.m.seqs[slot]) |*sq| cmedia.clearSeq(&sq.media, &sq.mrope_delta);
                if (b.m.pages) |pg| pg.releaseSlot(slot);
            },
            .map_pages => {
                const pg = b.m.pages orelse return error.NotPaged;
                const slot: usize = @intCast(msg[1]);
                const from: usize = @intCast(msg[2]);
                const n: usize = @intCast(msg[3]);
                if (msg.len < 4 + n) return error.BadPlan;
                try tokens_tmp.resize(b.gpa, n);
                for (0..n) |i| tokens_tmp.items[i] = @intCast(msg[4 + i]);
                try pg.recordFollowerMap(b.s, slot, tokens_tmp.items, from);
            },
            .forward, .forward_greedy, .forward_prompt, .forward_sample => {
                const slot: usize = @intCast(msg[1]);
                const p0: usize = @intCast(msg[2]);
                const lrows: usize = @intCast(msg[3]);
                const n: usize = @intCast(msg[4]);
                try tokens_tmp.resize(b.gpa, n);
                for (0..n) |i| tokens_tmp.items[i] = @intCast(msg[5 + i]);
                if (op == .forward_greedy) {
                    try b.m.forwardGreedy(b.k, b.s, slot, p0, tokens_tmp.items, lrows);
                } else if (op == .forward_sample) {
                    try b.m.forwardSample(b.k, b.s, slot, p0, tokens_tmp.items, lrows);
                } else if (op == .forward_prompt) {
                    try b.m.forwardPrompt(b.k, b.s, slot, p0, tokens_tmp.items, lrows);
                } else {
                    try b.m.forward(b.k, b.s, slot, p0, tokens_tmp.items, lrows);
                }
            },
            .verify_round => try batched.replayVerify(b.m, b.k, b.s, b.gpa, &tokens_tmp, msg, .full_f32, b.m.hasDfly()),
            .verify_round_greedy => try batched.replayVerify(b.m, b.k, b.s, b.gpa, &tokens_tmp, msg, .greedy, true),
            .verify_round_greedy_plain => try batched.replayVerify(b.m, b.k, b.s, b.gpa, &tokens_tmp, msg, .greedy, false),
            .verify_round_sample => try batched.replayVerify(b.m, b.k, b.s, b.gpa, &tokens_tmp, msg, .sample_bf16, true),
            .verify_round_sample_plain => try batched.replayVerify(b.m, b.k, b.s, b.gpa, &tokens_tmp, msg, .sample_bf16, false),
            .draft_round => {
                const batch_n: usize = @intCast(msg[1]);
                const max_d: usize = @intCast(msg[2]);
                if (batch_n == 0 or batch_n > cuda_dfly.max_reqs or msg.len < 3 + batch_n * 4) return error.BadPlan;
                var items: [cuda_dfly.max_reqs]fwd.Model.DraftItem = undefined;
                for (0..batch_n) |bi| {
                    const base = 3 + bi * 4;
                    items[bi] = .{
                        .slot = @intCast(msg[base]),
                        .anchor_tok = @intCast(msg[base + 1]),
                        .anchor_pos = @intCast(msg[base + 2]),
                        .src_row = @intCast(msg[base + 3]),
                    };
                }
                try b.m.draftDfly(b.k, b.s, items[0..batch_n], max_d);
            },
            .keep_round => {
                const nw: usize = @intCast(msg[1]);
                for (0..nw) |i| {
                    const slot: usize = @intCast(msg[2 + 2 * i]);
                    const new_len: usize = @intCast(msg[3 + 2 * i]);
                    if (b.m.seqs[slot]) |*sq| sq.len = new_len;
                }
            },
            .snap_save => {
                const id: u64 = @intCast(msg[1]);
                const slot: usize = @intCast(msg[2]);
                const at: u32 = @intCast(msg[3]);
                const saved_opt = snap.save(b.gpa, &b.spares, b.m, b.s, slot, id, at) catch null;
                const all_ok = try ts.agreeOk(b.m.d, b.s, b.m.vote_dev.ptr, saved_opt != null);
                if (!all_ok) {
                    if (saved_opt) |sv| snap.drop(b.gpa, &b.spares, b.s, sv);
                } else {
                    try b.s.synchronize();
                    saved_opt.?.synced_peers = true;
                    try b.follower_snaps.put(b.gpa, id, saved_opt.?);
                }
            },
            .snap_restore => {
                const id: u64 = @intCast(msg[1]);
                const slot: usize = @intCast(msg[2]);
                if (b.follower_snaps.get(id)) |sv| {
                    try snap.restore(b.m, b.s, slot, sv);
                    try b.s.synchronize();
                }
            },
            .snap_drop => {
                const id: u64 = @intCast(msg[1]);
                if (b.follower_snaps.fetchRemove(id)) |kv| {
                    if (kv.value.blob.len > 0) b.gpa.free(kv.value.blob);
                    snap.drop(b.gpa, &b.spares, b.s, kv.value);
                }
            },
            .snap_load => {
                const id: u64 = @intCast(msg[1]);
                const at: u32 = @intCast(msg[2]);
                const blank_opt = snap.blank(b.gpa, &b.spares, b.m, id, at) catch null;
                const all_ok = try ts.agreeOk(b.m.d, b.s, b.m.vote_dev.ptr, blank_opt != null);
                if (!all_ok) {
                    if (blank_opt) |sv| snap.drop(b.gpa, &b.spares, b.s, sv);
                } else {
                    const sv = blank_opt.?;
                    errdefer snap.drop(b.gpa, &b.spares, b.s, sv);
                    try ts.broadcastFrom0(b.s, sv.buf.ptr, sv.buf.len);
                    try b.s.synchronize();
                    try ts.check();
                    try snap.syncHistory(sv);
                    sv.synced_peers = true;
                    try b.follower_snaps.put(b.gpa, id, sv);
                }
            },
            .snap_trim => {
                const room: u64 = @bitCast(msg[1]);
                b.spares.trim(room);
            },
            .set_media => {
                const slot: usize = @intCast(msg[1]);
                const prompt_len: usize = @intCast(msg[2]);
                const feat_rows: usize = @intCast(msg[3]);
                const out_hidden: usize = @intCast(msg[4]);
                const delta: i32 = @intCast(msg[5]);
                const sq = if (b.m.seqs[slot]) |*p| p else return error.UnknownSlot;
                if (sq.media) |*old| {
                    old.deinit();
                    sq.media = null;
                }
                const idx_freqs = if (b.m.first_idx_layer) |fi| b.m.index_weights[fi].?.freqs.ptr else 0;
                sq.media = try cmedia.receiveFollower(b.gpa, b.m.d, b.s, b.k.idx.rope_table, idx_freqs, ts, prompt_len, feat_rows, out_hidden, delta);
                sq.mrope_delta = delta;
            },
            .clear_media => {
                const slot: usize = @intCast(msg[1]);
                const delta: i32 = @intCast(msg[2]);
                if (b.m.seqs[slot]) |*sq| {
                    if (sq.media) |*old| {
                        old.deinit();
                        sq.media = null;
                    }
                    sq.mrope_delta = delta;
                }
            },
            .snap_save_share => {
                const id: u64 = @intCast(msg[1]);
                const slot: usize = @intCast(msg[2]);
                const at: u32 = @intCast(msg[3]);
                const from: u32 = @intCast(msg[4]);
                const nb: usize = @intCast(msg[5]);
                if (msg.len < 6 + nb) return error.BadPlan;
                const blob = b.gpa.alloc(u32, nb) catch null;
                const saved_opt = if (blob) |bl| blk: {
                    for (0..nb) |i| bl[i] = @intCast(msg[6 + i]);
                    break :blk snap.saveShare(b.gpa, b.m, b.s, slot, id, at, from, bl) catch {
                        b.gpa.free(bl);
                        break :blk null;
                    };
                } else null;
                const all_ok = try ts.agreeOk(b.m.d, b.s, b.m.vote_dev.ptr, saved_opt != null);
                if (!all_ok) {
                    if (saved_opt) |sv| {
                        if (sv.blob.len > 0) b.gpa.free(sv.blob);
                        snap.drop(b.gpa, &b.spares, b.s, sv);
                    }
                } else {
                    try b.s.synchronize();
                    saved_opt.?.synced_peers = true;
                    try b.follower_snaps.put(b.gpa, id, saved_opt.?);
                }
            },
            .snap_load_share => {
                const id: u64 = @intCast(msg[1]);
                const at: u32 = @intCast(msg[2]);
                const from: u32 = @intCast(msg[3]);
                const mrope_delta: i32 = @intCast(msg[4]);
                const np: usize = @intCast(msg[5]);
                const nb: usize = @intCast(msg[6]);
                if (msg.len < 7 + np + nb) return error.BadPlan;
                const pg = b.m.pages orelse return error.NotPaged;
                var valid = true;
                for (0..np + nb) |i| {
                    const pid: usize = @intCast(msg[7 + i]);
                    if (pid >= pg.n_pages) valid = false;
                }
                const pages = if (valid) b.gpa.alloc(u32, np) catch null else null;
                const blob = if (pages != null) b.gpa.alloc(u32, nb) catch null else null;
                const hist = if (blob != null) b.gpa.alloc(u32, at) catch null else null;
                const sv_opt = if (hist != null) b.gpa.create(snap.Saved) catch null else null;
                const local_ok = sv_opt != null;
                const all_ok = try ts.agreeOk(b.m.d, b.s, b.m.vote_dev.ptr, local_ok);
                if (!all_ok) {
                    if (sv_opt) |sv| b.gpa.destroy(sv);
                    if (hist) |h| b.gpa.free(h);
                    if (blob) |bl| b.gpa.free(bl);
                    if (pages) |ps| b.gpa.free(ps);
                } else {
                    const ps = pages.?;
                    defer b.gpa.free(ps);
                    const bl = blob.?;
                    const h = hist.?;
                    @memset(h, 0);
                    for (0..np) |i| ps[i] = @intCast(msg[7 + i]);
                    for (0..nb) |i| bl[i] = @intCast(msg[7 + np + i]);
                    const sv = sv_opt.?;
                    sv.* = .{
                        .id = id,
                        .at = at,
                        .from = from,
                        .active_layers = b.m.active_layers,
                        .synced_peers = true,
                        .kv8 = b.m.kv8,
                        .mrope_delta = mrope_delta,
                        .buf = .{ .d = b.m.d, .ptr = 0, .len = 0, .borrowed = true },
                        .blob = bl,
                        .history = h,
                    };
                    errdefer {
                        if (sv.blob.len > 0) b.gpa.free(sv.blob);
                        snap.drop(b.gpa, &b.spares, b.s, sv);
                    }
                    try @import("cuda_disk_pages.zig").broadcastSharePayload(b.m, b.s, ts, sv, ps, false);
                    try b.follower_snaps.put(b.gpa, id, sv);
                }
            },
        }
    }
}

test "cuda_follower declarations compile on host" {
    std.testing.refAllDecls(@This());
}
