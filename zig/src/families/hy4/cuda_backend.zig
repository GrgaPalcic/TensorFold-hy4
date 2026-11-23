//! Hy4-flash behind `lanes.Backend`: chunked prefill, multi-rank `PlanLink` round queueing, sampling, and `logprobs`.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const fwd = @import("cuda_model.zig");
const cuda_dfly = @import("cuda_dfly.zig");
const cuda_prof = @import("cuda_prof.zig");
const batched = @import("cuda_batch.zig");
const cuda_sample = @import("cuda_sample.zig");
const snap = @import("cuda_snap.zig");
const cuda_tp = @import("cuda_tp.zig");
const cvis = @import("cuda_vision.zig");

const be = lanes.backend;
const Stream = lanes.Stream;
const Row = lanes.logprob.Row;
const Op = cuda_tp.Op;

pub const prompt_rows: u32 = @intCast(fwd.prompt_rows);
pub const max_window: u32 = @intCast(fwd.max_window);
pub const max_batch: u32 = @intCast(fwd.max_batch);
pub const round_rows: u32 = @intCast(fwd.round_rows);
const handle_ring: usize = 1024;

// Measured TP=2 verify(R) curve across short and long context (captures MoE expert-union growth with rows).
const hy4_window_costs = [fwd.max_window]lanes.config.Cost{
    .{ .width = 1, .ms = 38.0 },   .{ .width = 2, .ms = 50.0 },   .{ .width = 3, .ms = 57.5 },   .{ .width = 4, .ms = 64.5 },
    .{ .width = 5, .ms = 73.5 },   .{ .width = 6, .ms = 79.0 },   .{ .width = 7, .ms = 86.5 },   .{ .width = 8, .ms = 92.5 },
    .{ .width = 9, .ms = 97.5 },   .{ .width = 10, .ms = 102.5 }, .{ .width = 11, .ms = 107.0 }, .{ .width = 12, .ms = 111.5 },
    .{ .width = 13, .ms = 116.0 }, .{ .width = 14, .ms = 120.0 }, .{ .width = 15, .ms = 124.0 }, .{ .width = 16, .ms = 128.0 },
};

// Measured TP=4 verify(R) on the four-rank ring (2026-11-23): a row costs about half of what it does at TP=2.
const hy4_window_costs_w4 = [fwd.max_window]lanes.config.Cost{
    .{ .width = 1, .ms = 25.9 },  .{ .width = 2, .ms = 29.9 },  .{ .width = 3, .ms = 33.4 },  .{ .width = 4, .ms = 37.8 },
    .{ .width = 5, .ms = 40.4 },  .{ .width = 6, .ms = 45.1 },  .{ .width = 7, .ms = 48.5 },  .{ .width = 8, .ms = 51.8 },
    .{ .width = 9, .ms = 53.5 },  .{ .width = 10, .ms = 56.0 }, .{ .width = 11, .ms = 58.5 }, .{ .width = 12, .ms = 61.0 },
    .{ .width = 13, .ms = 63.3 }, .{ .width = 14, .ms = 65.6 }, .{ .width = 15, .ms = 67.9 }, .{ .width = 16, .ms = 70.1 },
};

/// A shared round's cost by rows: `base` ms at one row, then `slopes` ms a row up to 8, 16, 32 rows and past 32.
fn sharedCostTable(comptime base: f64, comptime slopes: [4]f64) [fwd.max_batch]lanes.config.Cost {
    var c: [fwd.max_batch]lanes.config.Cost = undefined;
    const at8 = base + slopes[0] * 7;
    const at16 = at8 + slopes[1] * 8;
    const at32 = at16 + slopes[2] * 16;
    for (&c, 0..) |*x, i| {
        const w = i + 1;
        const ms: f64 = if (w <= 8)
            base + slopes[0] * @as(f64, @floatFromInt(w - 1))
        else if (w <= 16)
            at8 + slopes[1] * @as(f64, @floatFromInt(w - 8))
        else if (w <= 32)
            at16 + slopes[2] * @as(f64, @floatFromInt(w - 16))
        else
            at32 + slopes[3] * @as(f64, @floatFromInt(w - 32));
        x.* = .{ .width = @intCast(w), .ms = ms };
    }
    return c;
}

const hy4_shared_costs = sharedCostTable(38.0, .{ 7.8, 4.4, 3.7, 2.7 });
const hy4_shared_costs_w4 = sharedCostTable(26.0, .{ 4.0, 2.6, 2.0, 1.6 });

pub const Lane = struct {
    slot: usize,
    len: u64 = 0,
    base: u64 = 0,
    fill: ?usize = null,
    first_tok: ?u32 = null,
    rows: std.ArrayList(u32) = .empty,
    last_logits: []f32, // [vocab] fp32 from prompt's final row
    held: [cuda_dfly.max_drafts]u32 = @splat(0),
    held_n: usize = 0,
    marks: []const u32 = &.{},
    kept: u32 = 0,
    share_from: u32 = 0,
};

pub fn fits(physical: usize, prompt: usize, reply: usize) bool {
    return prompt <= physical and reply <= physical - prompt and fwd.max_window <= physical - prompt - reply;
}

/// Stable fp64 log-sum-exp over all untempered fp32 head columns (`#13` exact TP logprobs).
pub const scoreLogprobs = @import("cuda_logprobs.zig").scoreLogprobs;

pub const Backend = struct {
    gpa: std.mem.Allocator,
    m: *fwd.Model,
    k: *const fwd.AllKernels,
    s: cuda.Stream,
    priority_copies: bool = false,
    lanes_map: std.AutoHashMapUnmanaged(*const Stream, Lane) = .empty,
    free: [fwd.max_slots]bool = @splat(true),
    handles: [handle_ring]u32 = @splat(0),
    next_handle: u64 = 0,
    next_snap: u64 = 1,
    logits_host: []f32, // [max_batch * vocab]
    argmax_host: [cuda_tp.max_world * fwd.max_batch * 2]u32 = @splat(0),
    bf16_row: []u16, // [vocab] for lanes.sampling_row.draw
    work_row: []f32, // [vocab] penalized scratch
    counts: []u32, // [vocab] token counts when penalties are active
    plan_buf: std.ArrayList(i64) = .empty,
    follower_snaps: std.AutoHashMapUnmanaged(u64, *snap.Saved) = .empty,
    prof_round_n: usize = 0,
    prof_verify_ns: i64 = 0,
    prof_draft_ns: i64 = 0,
    prof_draft_asked: usize = 0,
    prof_draft_max_d: usize = 0,

    spares: snap.Spares = .{},
    ctx: ?*const cuda.Context = null,
    vis: ?*cvis.Vision = null,
    prompts: ?@import("core").kv_prompts.Prompts = null,
    pending_restore: ?*const snap.Saved = null,
    pub fn init(gpa: std.mem.Allocator, m: *fwd.Model, k: *const fwd.AllKernels, s: cuda.Stream, priority_copies: bool) !Backend {
        const v = m.c.vocab;
        const lh = try gpa.alloc(f32, fwd.max_batch * v);
        errdefer gpa.free(lh);
        const br = try gpa.alloc(u16, v);
        errdefer gpa.free(br);
        const wr = try gpa.alloc(f32, v);
        errdefer gpa.free(wr);
        const cnt = try gpa.alloc(u32, v);
        @memset(cnt, 0);
        return .{
            .gpa = gpa,
            .m = m,
            .k = k,
            .s = s,
            .priority_copies = priority_copies,
            .logits_host = lh,
            .bf16_row = br,
            .work_row = wr,
            .counts = cnt,
        };
    }

    pub fn deinit(b: *Backend) void {
        var sit = b.follower_snaps.valueIterator();
        while (sit.next()) |sv| snap.drop(b.gpa, null, b.s, sv.*);
        b.follower_snaps.deinit(b.gpa);
        b.spares.deinit();
        var it = b.lanes_map.valueIterator();
        while (it.next()) |l| {
            l.rows.deinit(b.gpa);
            b.gpa.free(l.last_logits);
        }
        b.lanes_map.deinit(b.gpa);
        b.plan_buf.deinit(b.gpa);
        b.gpa.free(b.counts);
        b.gpa.free(b.work_row);
        b.gpa.free(b.bf16_row);
        b.gpa.free(b.logits_host);
        b.* = undefined;
    }

    pub fn backend(b: *Backend) be.Backend {
        return .{ .ptr = b, .vtable = &vtable, .penalties = true, .chunk_invariant = true };
    }

    pub fn facts(b: *const Backend) lanes.Model {
        const dfly = b.m.hasDfly();
        const w4 = if (b.m.tp_sess) |ts| ts.world >= 4 else false;
        return .{
            .fixed_draft_block = dfly,
            .root_copies = b.priority_copies,
            .context_copies = b.priority_copies,
            .copy_match = if (b.priority_copies) 4 else 8,
            .exact_width = max_window,
            .gpu_tokens = false,
            .mtp = dfly,
            .speculate = dfly,
            .speculate_early = false,
            .drafts = if (dfly) cuda_dfly.max_drafts else 0,
            .draft_streams = dfly,
            .mtp_step_ms = if (dfly) 1.1 else 0,
            .draft_step_ms = if (dfly) 1.1 else 0,
            .window_costs = if (!dfly) &.{} else if (w4) &hy4_window_costs_w4 else &hy4_window_costs,
            .shared_costs = if (!dfly) &.{} else if (w4) &hy4_shared_costs_w4 else &hy4_shared_costs,
            .allocate_shared_drafts = false,
            .shared_drafts = 0,
            .hidden_rows = true,
            .batch_rows = max_batch, // a shared round is one forward: up to 64 batched decode/verify rows
            .max_streams = fwd.max_slots,
        };
    }

    const vtable: be.Backend.VTable = .{
        .prefill = prefill,
        .prefill_step = prefillStep,
        .first = first,
        .first_masked = firstMasked,
        .first_row = firstRow,
        .queue = queue,
        .read = read,
        .verify = verify,
        .keep = ops.keep,
        .draft = ops.draft,
        .held = ops.held,
        .release = release,
        .reports_fill = true,
    };

    fn self(ptr: *anyopaque) *Backend {
        return @ptrCast(@alignCast(ptr));
    }

    fn pushHandle(b: *Backend, tok: u32) u64 {
        const h = b.next_handle;
        b.handles[@intCast(h % handle_ring)] = tok;
        b.next_handle = h + 1;
        return h;
    }

    /// Draws one token from `raw` (`fp32 [vocab]`), applying penalties, grammar `mask`, and `sampling_row.draw`.
    fn drawOne(b: *Backend, s: *const Stream, raw: []const f32, extra_emitted: []const u32, position: u64, mask: []const u32) !u32 {
        var row = raw;
        if (s.penalties.active()) {
            @memcpy(b.work_row, raw);
            const em = s.emitted();
            for (em) |t| if (t < b.counts.len) {
                b.counts[t] += 1;
            };
            for (extra_emitted) |t| if (t < b.counts.len) {
                b.counts[t] += 1;
            };
            for (em) |t| if (t < b.counts.len and b.counts[t] > 0) {
                b.work_row[t] -= s.penalties.of(b.counts[t]);
                b.counts[t] = 0;
            };
            for (extra_emitted) |t| if (t < b.counts.len and b.counts[t] > 0) {
                b.work_row[t] -= s.penalties.of(b.counts[t]);
                b.counts[t] = 0;
            };
            row = b.work_row;
        }
        const greedy = if (s.sampling) |sp| sp.temperature <= 0 else true;
        if (greedy) {
            var best: ?usize = null;
            for (row, 0..) |x, id| {
                if (!std.math.isFinite(x)) return error.NonfiniteLogits;
                if (mask.len > 0 and !lanes.grammar.allows(mask, id)) continue;
                if (best == null or x > row[best.?]) best = id;
            }
            return @intCast(best orelse return error.NoAllowedToken);
        }
        for (row, b.bf16_row) |x, *w| {
            if (!std.math.isFinite(x)) return error.NonfiniteLogits;
            w.* = core.bf16_round.bits(x);
        }
        return lanes.sampling_row.draw(b.gpa, b.bf16_row, s.sampling, position, mask);
    }

    fn isFastGreedy(s: *const Stream, mask: []const u32) bool {
        if (s.logprobs != null or s.penalties.active() or mask.len != 0) return false;
        return if (s.sampling) |sp| sp.temperature <= 0 else true;
    }

    fn downloadArgmax(b: *Backend, rows: usize, dst: []u32) !void {
        try b.s.synchronize();
        if (b.m.prof) |*p| p.finish(b.s);
        if (b.m.tp_sess) |ts| try ts.check();
        const world = b.m.at.world;
        const words = world * rows * 2;
        try (cuda.DeviceBuffer{ .d = b.m.d, .ptr = b.m.argmaxPtr(), .len = words * 4 }).download(0, std.mem.sliceAsBytes(b.argmax_host[0..words]));
        for (0..rows) |r| {
            var best_v: f32 = -std.math.inf(f32);
            var best_id: u32 = std.math.maxInt(u32);
            for (0..world) |rk| {
                const idx = (rk * rows + r) * 2;
                const v: f32 = @bitCast(b.argmax_host[idx]);
                const id = b.argmax_host[idx + 1];
                if (!std.math.isFinite(v)) return error.NonfiniteLogits;
                if (v > best_v or (v == best_v and id < best_id)) {
                    best_v = v;
                    best_id = id;
                }
            }
            dst[r] = best_id;
        }
    }

    fn sendForwardOp(b: *Backend, op: Op, slot: usize, p0: usize, tokens: []const u32, logits_rows: usize) !void {
        const ts = b.m.tp_sess orelse return;
        b.plan_buf.clearRetainingCapacity();
        try b.plan_buf.ensureUnusedCapacity(b.gpa, 5 + tokens.len);
        b.plan_buf.appendAssumeCapacity(@backingInt(op));
        b.plan_buf.appendAssumeCapacity(@intCast(slot));
        b.plan_buf.appendAssumeCapacity(@intCast(p0));
        b.plan_buf.appendAssumeCapacity(@intCast(logits_rows));
        b.plan_buf.appendAssumeCapacity(@intCast(tokens.len));
        for (tokens) |t| b.plan_buf.appendAssumeCapacity(t);
        try ts.send(b.plan_buf.items);
    }

    fn sendForward(b: *Backend, slot: usize, p0: usize, tokens: []const u32, logits_rows: usize) !void {
        return b.sendForwardOp(.forward, slot, p0, tokens, logits_rows);
    }

    fn downloadLogits(b: *Backend, rows: usize, dst: []f32) !void {
        try b.s.synchronize();
        if (b.m.prof) |*p| p.finish(b.s);
        if (b.m.tp_sess) |ts| try ts.check();
        const bytes = rows * b.m.c.vocab * 4;
        try (cuda.DeviceBuffer{ .d = b.m.d, .ptr = b.m.logitsPtr(), .len = bytes }).download(0, std.mem.sliceAsBytes(dst[0 .. rows * b.m.c.vocab]));
    }

    pub fn openSeqAllRanks(b: *Backend, slot: usize, cap: usize) !void {
        if (b.m.tp_sess) |ts| {
            try ts.send(&.{ @backingInt(Op.open_seq), @intCast(slot), @intCast(cap) });
            const local_ok = if (b.m.openSeq(slot, cap)) |_| true else |_| false;
            const all_ok = try ts.agreeOk(b.m.d, b.s, b.m.vote_dev.ptr, local_ok);
            if (!all_ok) {
                if (local_ok) b.m.releaseSeq(slot);
                return error.OutOfDeviceMemory;
            }
            return;
        }
        try b.m.openSeq(slot, cap);
    }

    pub fn releaseSeqAllRanks(b: *Backend, slot: usize) void {
        if (b.m.tp_sess) |ts| ts.send(&.{ @backingInt(Op.release_seq), @intCast(slot) }) catch {};
        b.m.releaseSeq(slot);
    }

    fn opening(b: *Backend, s: *Stream) !*Lane {
        return ops.opening(b, s);
    }

    fn chunk(b: *Backend, s: *Stream, l: *Lane, max_rows: u32) !bool {
        const p = s.prompt();
        const at = l.fill.?;
        if (s.isCancelled()) return error.Cancelled;
        const step_cap: usize = if (max_rows > 0 and max_rows < fwd.prompt_rows) max_rows else fwd.prompt_rows;
        var end = @min(p.len, (at / step_cap + 1) * step_cap);
        const marks = if (b.prompts != null) l.marks else s.reuse.marks;
        for (marks) |mk| if (mk > at and mk < end) {
            end = mk;
        };
        const done = end == p.len;
        const lrows: usize = @intFromBool(done);
        try b.sendForwardOp(.forward_prompt, l.slot, at, p[at..end], lrows);
        try b.m.forwardPrompt(b.k, b.s, l.slot, at, p[at..end], lrows);
        if (done) {
            try b.downloadLogits(1, l.last_logits);
        }
        s.fill_at.store(@intCast(end), .monotonic);
        l.fill = if (done) null else end;
        if (done) l.len = p.len else l.len = end;
        try ops.afterChunk(b, s, l, end, done);
        return done;
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const b = self(ptr);
        const l = try b.opening(s);
        errdefer b.dropLane(s);
        while (!try b.chunk(s, l, 0)) {}
    }

    fn prefillStep(ptr: *anyopaque, s: *Stream, max_rows: u32) anyerror!bool {
        const b = self(ptr);
        const going = if (b.lanes_map.getPtr(s)) |l| l.fill != null else false;
        const l = if (going) b.lanes_map.getPtr(s).? else try b.opening(s);
        errdefer b.dropLane(s);
        return b.chunk(s, l, max_rows);
    }

    fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        return firstMasked(ptr, s, position, &.{});
    }

    fn firstMasked(ptr: *anyopaque, s: *Stream, position: u64, mask: []const u32) anyerror!u64 {
        const b = self(ptr);
        const l = b.lanes_map.getPtr(s) orelse return error.UnknownStream;
        if (position != l.len or l.fill != null) return error.PositionMismatch;
        const tok = try b.drawOne(s, l.last_logits, &.{}, position, mask);
        l.first_tok = tok;
        return b.pushHandle(tok);
    }

    fn firstRow(ptr: *anyopaque, s: *Stream) anyerror!Row {
        const b = self(ptr);
        const l = b.lanes_map.getPtr(s) orelse return error.UnknownStream;
        const k = s.logprobs orelse return error.LogprobsUnsupported;
        const tok = l.first_tok orelse return error.LogprobRow;
        return scoreLogprobs(l.last_logits, tok, k);
    }

    pub fn feedVal(b: *Backend, feed: be.Feed) !u32 {
        return switch (feed) {
            .value => |v| v,
            .handle => |h| read(b, h),
        };
    }

    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const b = self(ptr);
        const l = b.lanes_map.getPtr(s) orelse return error.UnknownStream;
        const tok_in = try b.feedVal(feed);
        if (position != l.len + 1) return error.PositionMismatch;
        if (isFastGreedy(s, &.{})) {
            try b.sendForwardOp(.forward_greedy, l.slot, @intCast(l.len), &.{tok_in}, 1);
            try b.m.forwardGreedy(b.k, b.s, l.slot, @intCast(l.len), &.{tok_in}, 1);
            l.len += 1;
            var picked: [1]u32 = undefined;
            try b.downloadArgmax(1, &picked);
            return b.pushHandle(picked[0]);
        }
        if (cuda_sample.isFastSampled(s, &.{})) {
            try b.sendForwardOp(.forward_sample, l.slot, @intCast(l.len), &.{tok_in}, 1);
            try b.m.forwardSample(b.k, b.s, l.slot, @intCast(l.len), &.{tok_in}, 1);
            l.len += 1;
            var picked: [1]u32 = undefined;
            try b.m.sampleGpu(b.k, b.s, &.{cuda.sample.rule(s.sampling.?)}, &.{@intCast(position - 1)}, &picked);
            return b.pushHandle(picked[0]);
        }
        try b.sendForward(l.slot, @intCast(l.len), &.{tok_in}, 1);
        try b.m.forward(b.k, b.s, l.slot, @intCast(l.len), &.{tok_in}, 1);
        l.len += 1;
        try b.downloadLogits(1, l.last_logits);
        const tok_out = try b.drawOne(s, l.last_logits, &.{}, position, &.{});
        return b.pushHandle(tok_out);
    }

    fn read(ptr: *anyopaque, h: u64) anyerror!u32 {
        const b = self(ptr);
        if (h >= b.next_handle or b.next_handle - h > handle_ring) return error.NoSuchToken;
        return b.handles[@intCast(h % handle_ring)];
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = self(ptr);
        const vt0 = if (b.m.prof != null) cuda_prof.nowNs() else 0;
        const v = b.m.c.vocab;
        const mask_words = (v + 31) / 32;

        var all_greedy = true;
        var all_sampled_fast = true;
        var any_drafts = false;
        for (windows) |w| {
            if (!isFastGreedy(w.stream, w.masks)) all_greedy = false;
            if (!cuda_sample.isFastSampled(w.stream, w.masks)) all_sampled_fast = false;
            if (w.stream.drafts) any_drafts = true;
        }
        const logit_mode: fwd.Model.LogitMode = if (all_greedy) .greedy else if (all_sampled_fast) .sample_bf16 else .full_f32;
        const save_dfly = b.m.hasDfly() and any_drafts;
        b.plan_buf.clearRetainingCapacity();
        if (b.m.tp_sess != null) {
            const op: Op = switch (logit_mode) {
                .greedy => if (save_dfly) .verify_round_greedy else .verify_round_greedy_plain,
                .sample_bf16 => if (save_dfly) .verify_round_sample else .verify_round_sample_plain,
                .full_f32 => .verify_round,
            };
            try b.plan_buf.append(b.gpa, @backingInt(op));
            try b.plan_buf.append(b.gpa, @intCast(windows.len));
        }

        for (windows) |w| {
            if (w.parents != null) return error.ChainsOnly;
            const l = b.lanes_map.getPtr(w.stream) orelse return error.UnknownStream;
            if (w.held > l.held_n) return error.NoHeldDrafts;
            const n = w.rows();
            if (n > fwd.max_window) return error.WindowTooWide;
            if (w.masks.len != 0 and w.masks.len != n * mask_words) return error.BadMaskShape;
            l.base = l.len;
            l.rows.clearRetainingCapacity();
            try l.rows.append(b.gpa, w.pending);
            if (w.held > 0) try l.rows.appendSlice(b.gpa, l.held[0..w.held]);
            if (w.tokens.len > 0) try l.rows.appendSlice(b.gpa, w.tokens);
            if (l.rows.items.len != n) return error.BadStepShape;
            for (w.positions, 0..) |pos, r| if (pos != l.base + r + 1) return error.PositionMismatch;
            if (b.m.tp_sess != null) {
                try b.plan_buf.append(b.gpa, @intCast(l.slot));
                try b.plan_buf.append(b.gpa, @intCast(l.base));
                try b.plan_buf.append(b.gpa, @intCast(n));
                for (l.rows.items) |t| try b.plan_buf.append(b.gpa, t);
            }
        }
        if (b.m.tp_sess) |ts| try ts.send(b.plan_buf.items);

        if (windows.len > fwd.max_slots) return error.WindowTooWide;
        var parts: [fwd.max_slots]fwd.Model.Part = undefined;
        for (windows, parts[0..windows.len]) |w, *p| {
            const l = b.lanes_map.getPtr(w.stream).?;
            p.* = .{ .slot = l.slot, .first = l.base, .tokens = l.rows.items };
        }
        var done: usize = 0;
        var picked_buf: [fwd.max_batch]u32 = undefined;
        while (done < windows.len) {
            const bt = batched.batch(parts[done..windows.len]);
            try b.m.forwardPartsLogits(b.k, b.s, parts[done..][0..bt.n], bt.rows, logit_mode, save_dfly);
            switch (logit_mode) {
                .greedy => {
                    try b.downloadArgmax(bt.rows, picked_buf[0..bt.rows]);
                    var r0: usize = 0;
                    for (windows[done..][0..bt.n], out[done..][0..bt.n], parts[done..][0..bt.n]) |w, *o, p| {
                        const n = p.tokens.len;
                        @memcpy(o.sampled[0..n], picked_buf[r0..][0..n]);
                        @memcpy(o.drafts[0 .. n - 1], p.tokens[1..]);
                        b.lanes_map.getPtr(w.stream).?.len = p.first + n;
                        r0 += n;
                    }
                },
                .sample_bf16 => try ops.finishSampledBatch(b, windows[done..][0..bt.n], out[done..][0..bt.n], parts[done..][0..bt.n], bt.rows),
                .full_f32 => {
                    try b.downloadLogits(bt.rows, b.logits_host[0 .. bt.rows * v]);
                    var r0: usize = 0;
                    for (windows[done..][0..bt.n], out[done..][0..bt.n], parts[done..][0..bt.n]) |w, *o, p| {
                        const n = p.tokens.len;
                        for (0..n) |r| {
                            const row_logits = b.logits_host[(r0 + r) * v ..][0..v];
                            const rmask = if (w.masks.len > 0) w.masks[r * mask_words ..][0..mask_words] else &.{};
                            o.sampled[r] = try b.drawOne(w.stream, row_logits, p.tokens[0 .. r + 1], w.positions[r], rmask);
                            if (w.stream.logprobs) |k_top| if (r < o.rows.len) {
                                o.rows[r] = try scoreLogprobs(row_logits, o.sampled[r], k_top);
                            };
                        }
                        @memcpy(o.drafts[0 .. n - 1], p.tokens[1..]);
                        b.lanes_map.getPtr(w.stream).?.len = p.first + n;
                        r0 += n;
                    }
                },
            }
            done += bt.n;
        }
        if (b.m.prof != null) {
            b.prof_verify_ns = cuda_prof.nowNs() - vt0;
            b.prof_draft_ns = 0;
            b.prof_draft_asked = 0;
            b.prof_draft_max_d = 0;
        }
    }

    const ops = @import("cuda_backend_ops.zig");

    fn dropLane(b: *Backend, s: *const Stream) void {
        if (b.lanes_map.fetchRemove(s)) |entry| {
            var l = entry.value;
            ops.releaseLane(b, &l);
        }
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        self(ptr).dropLane(s);
    }

    pub const blankState = ops.blankState;
    pub const trimSpares = ops.trimSpares;
    pub const saveState = ops.saveState;
    pub const restoreState = ops.restoreState;
    pub const dropState = ops.dropState;

    pub const followLoop = @import("cuda_follower.zig").loop;
};

test "cuda_backend declarations compile on host" {
    std.testing.refAllDecls(@This());
}
