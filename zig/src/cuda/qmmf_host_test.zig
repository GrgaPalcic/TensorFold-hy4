//! Host unit tests for `qmmf.zig` split-K pinning and slice planning.
const std = @import("std");
const qmmf = @import("qmmf.zig");

test "a cut weight pinned to the whole weight's K slices plans as the whole one at every row count" {
    const whole: qmmf.Weight = .{ .mode = .bf16, .codes = 0, .scales = 0, .n = 16480, .k = 2560, .npad = 16512 };
    var cut = whole;
    cut.n = 8240;
    cut.npad = 8320;
    try std.testing.expectEqual(@as(u32, 1), qmmf.splitK(whole.n, whole.k));
    try std.testing.expectEqual(@as(u32, 2), qmmf.splitK(cut.n, cut.k)); // unpinned, the half would sum two slices
    const kept = try qmmf.pinned(cut, qmmf.splitK(whole.n, whole.k));
    for ([_]usize{ 1, 16, 64, 1536 }) |m| for ([_]bool{ false, true }) |cl| {
        try std.testing.expectEqual(qmmf.plan(m, whole.n, whole.k, cl), qmmf.planFor(m, kept, cl));
        try std.testing.expectEqual(qmmf.plan(m, cut.n, cut.k, cl), qmmf.planFor(m, cut, cl));
    };
    try std.testing.expect(qmmf.planFor(1536, cut, false).sk != qmmf.planFor(1536, kept, false).sk);
    try std.testing.expectEqual(qmmf.splitK(whole.n, whole.k), qmmf.slices(kept));
    try std.testing.expectEqual(qmmf.splitK(cut.n, cut.k), qmmf.slices(cut));
    try std.testing.expectEqual(@as(u32, 4), (try qmmf.pinned(cut, 4)).sk);
    for ([_]u32{ 0, 3, 8, 16 }) |bad| try std.testing.expectError(error.BadSplit, qmmf.pinned(cut, bad));
}
