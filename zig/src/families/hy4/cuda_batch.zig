//! Hy4's batched decode rounds: which windows share one forward, and a follower rank's replay of a verify plan.
const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("cuda_model.zig");

const Part = fwd.Model.Part;

/// The leading parts one batched step carries (at least one): rows within `max_batch`, the logits buffer's room.
pub fn batch(parts: []const Part) struct { n: usize, rows: usize } {
    var n: usize = 1;
    var rows = parts[0].tokens.len;
    while (n < parts.len and n < fwd.max_slots and rows + parts[n].tokens.len <= fwd.max_batch) : (n += 1) {
        rows += parts[n].tokens.len;
    }
    return .{ .n = n, .rows = rows };
}

/// A follower's verify round from plan words `[op, windows, (slot, base, n, tokens[n])...]`, batched as rank 0 does.
pub fn replayVerify(m: *fwd.Model, k: *const fwd.AllKernels, s: cuda.Stream, gpa: std.mem.Allocator, tmp: *std.ArrayList(u32), msg: []const i64, logit_mode: fwd.Model.LogitMode, save_dfly: bool) !void {
    const nw: usize = @intCast(msg[1]);
    if (nw == 0 or nw > fwd.max_slots) return error.WindowTooWide;
    var parts: [fwd.max_slots]Part = undefined;
    try tmp.resize(gpa, msg.len);
    var cur: usize = 2;
    for (parts[0..nw]) |*p| {
        if (cur + 3 > msg.len) return error.BadPlan;
        const n: usize = @intCast(msg[cur + 2]);
        if (cur + 3 + n > msg.len) return error.BadPlan;
        for (0..n) |i| tmp.items[cur + i] = @intCast(msg[cur + 3 + i]);
        p.* = .{ .slot = @intCast(msg[cur]), .first = @intCast(msg[cur + 1]), .tokens = tmp.items[cur..][0..n] };
        cur += 3 + n;
    }
    var done: usize = 0;
    while (done < nw) {
        const bt = batch(parts[done..nw]);
        try m.forwardPartsLogits(k, s, parts[done..][0..bt.n], bt.rows, logit_mode, save_dfly);
        done += bt.n;
    }
}

test "batch: decode rows share a step up to max_batch, a full window goes alone" {
    const one = [_]u32{7};
    const wide: [fwd.max_batch]u32 = @splat(7);
    var ps: [fwd.max_slots]Part = undefined;
    for (&ps, 0..) |*p, i| p.* = .{ .slot = i, .first = 0, .tokens = &one };
    try std.testing.expectEqual(fwd.max_slots, batch(&ps).n);
    ps[1].tokens = &wide;
    try std.testing.expectEqual(@as(usize, 1), batch(&ps).n);
    try std.testing.expectEqual(@as(usize, 1), batch(ps[1..]).n);
    try std.testing.expectEqual(fwd.max_batch, batch(ps[1..]).rows);
}
