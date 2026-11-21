//! A row's logprobs for the hy4 backend: fp64 log-softmax of the fp32 logits, the token's rank and the top k.
const std = @import("std");
const lanes = @import("lanes");

const Row = lanes.logprob.Row;

pub fn scoreLogprobs(values: []const f32, token: u32, k: u8) !Row {
    if (values.len == 0 or token >= values.len or k > lanes.logprob.max_top) return error.LogprobRow;
    var maximum: f64 = -std.math.inf(f64);
    for (values) |v| {
        if (std.math.isNan(v) or v == std.math.inf(f32)) return error.NonFiniteLogit;
        maximum = @max(maximum, v);
    }
    if (!std.math.isFinite(maximum)) return error.NoFiniteLogit;
    var sum: f64 = 0;
    for (values) |v| sum += @exp(@as(f64, v) - maximum);
    const mass = @log(sum);
    var out: Row = .{ .token = token, .logprob = @floatCast((@as(f64, values[token]) - maximum) - mass) };
    const want = @min(@as(usize, k), values.len);
    for (values, 0..) |v, id| {
        out.rank += @intFromBool(v >= values[token]);
        var at: usize = 0;
        while (at < out.count and values[out.ids[at]] >= v) : (at += 1) {}
        if (at >= want) continue;
        const end = @min(@as(usize, out.count), want - 1);
        var j = end;
        while (j > at) : (j -= 1) out.ids[j] = out.ids[j - 1];
        out.ids[at] = @intCast(id);
        out.count = @intCast(@min(@as(usize, out.count) + 1, want));
    }
    for (out.ids[0..out.count], out.logprobs[0..out.count]) |id, *lp| {
        lp.* = @floatCast((@as(f64, values[id]) - maximum) - mass);
    }
    return out;
}

test "scoreLogprobs normalizes in fp64 and ranks top tokens" {
    const vals = [_]f32{ 1.0, 3.0, 2.0, -1.0 };
    const r = try scoreLogprobs(&vals, 2, 3);
    try std.testing.expectEqual(@as(u32, 2), r.token);
    try std.testing.expectEqual(@as(u32, 2), r.rank);
    try std.testing.expectEqual(@as(u8, 3), r.count);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 0 }, r.ids[0..3]);
}
