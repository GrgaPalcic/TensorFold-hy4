//! Hy4-flash's iHC launches (zig/kernels/cuda/hy4_hc.cu): a layer's stream mixes, the branch merge, and hc_head.
const cuda = @import("cuda");

const launch = cuda.launch;

pub const streams = 4;
const threads = 256;

/// Which build of each kernel runs: the production one, or a planted contract control.
pub const Control = enum { production, nan, repeat, tail, sums };

pub const Hc = struct {
    mod: cuda.Module,
    pre_fns: [5]cuda.Function,
    head_fns: [5]cuda.Function,
    merge_fns: [4]cuda.Function,

    pub fn load(d: *const cuda.Driver) !Hc {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.hy4_hc);
        errdefer mod.unload();
        var h: Hc = .{ .mod = mod, .pre_fns = undefined, .head_fns = undefined, .merge_fns = undefined };
        for ([_][:0]const u8{ "hy4_hc_pre", "hy4_hc_pre_bad_nan", "hy4_hc_pre_bad_repeat", "hy4_hc_pre_bad_tail", "hy4_hc_pre_bad_sums" }, 0..) |n, i| h.pre_fns[i] = try mod.function(n);
        for ([_][:0]const u8{ "hy4_hc_head", "hy4_hc_head_bad_nan", "hy4_hc_head_bad_repeat", "hy4_hc_head_bad_tail", "hy4_hc_head_bad_sums" }, 0..) |n, i| h.head_fns[i] = try mod.function(n);
        for ([_][:0]const u8{ "hy4_hc_merge", "hy4_hc_merge_bad_nan", "hy4_hc_merge_bad_repeat", "hy4_hc_merge_bad_tail" }, 0..) |n, i| h.merge_fns[i] = try mod.function(n);
        return h;
    }

    pub fn unload(h: *Hc) void {
        h.mod.unload();
    }

    /// The fp32 sums behind one mix: an iHC's (pre and post) or hc_head's, as `pre` and `head` take them.
    pub const Mix = struct { fn_: u64, base: u64, scale: u64, eps: f32, hc_eps: f32, magnitude: f32 = 0 };

    /// `x` bf16 [rows, 4, hidden] to `out` bf16 [rows, hidden], with each row's post weights in `post` f32 [rows, 4].
    pub fn pre(h: *const Hc, s: cuda.Stream, m: Mix, x: u64, out: u64, post: u64, rows: usize, hidden: usize, control: Control, repeat: i32) !void {
        return h.mix(h.pre_fns[@backingInt(control)], s, m, x, out, post, rows, hidden, repeat);
    }

    /// hc_head: the last streams collapsed to one row (bf16), before the final norm.
    pub fn head(h: *const Hc, s: cuda.Stream, m: Mix, x: u64, out: u64, rows: usize, hidden: usize, control: Control, repeat: i32) !void {
        return h.mix(h.head_fns[@backingInt(control)], s, m, x, out, 0, rows, hidden, repeat);
    }

    fn mix(_: *const Hc, f: cuda.Function, s: cuda.Stream, m: Mix, x: u64, out: u64, post: u64, rows: usize, hidden: usize, repeat: i32) !void {
        if (rows == 0) return;
        var a: launch.Args = .{};
        a.add(x);
        a.add(m.fn_);
        a.add(m.base);
        a.add(m.scale);
        a.add(m.eps);
        a.add(m.hc_eps);
        a.add(m.magnitude);
        a.add(out);
        a.add(post);
        a.add(@as(i32, @intCast(hidden)));
        a.add(repeat);
        return launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }

    /// The branch `f` bf16 [rows, hidden] back into the streams: out[r, i] = bf16(post[r, i] f[r] + residual[r, i]).
    pub fn merge(h: *const Hc, s: cuda.Stream, residual: u64, f: u64, post: u64, out: u64, rows: usize, hidden: usize, control: Control, repeat: i32) !void {
        const count = rows * streams * hidden;
        if (count == 0) return;
        var a: launch.Args = .{};
        a.add(residual);
        a.add(f);
        a.add(post);
        a.add(out);
        a.add(@as(i32, @intCast(rows)));
        a.add(@as(i32, @intCast(hidden)));
        a.add(repeat);
        return launch.launch(h.merge_fns[@backingInt(control)], .{ .grid = .{ .x = @intCast((count + threads - 1) / threads), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }
};
