//! Float64 HYV4FlashMLP and HYV4FlashMoE on the release's tensors: weights exact (e4m3 x scale, bf16), every sum fp64.
const std = @import("std");
const core = @import("core");
const hy4 = @import("hy4");

const Config = hy4.config.Config;
const names = hy4.names;
const Tensor = core.checkpoint.Tensor;
const page = std.heap.page_allocator; // the workers' own buffers: thread-safe

/// A route given rather than chosen: the fixture's own picks and weights (fp32 in the reference's fp64 run too).
pub const Forced = struct { picks: []const i32, wts: []const f32 };

pub const Result = struct {
    out: []f64, // [rows, hidden]
    picks: []u32 = &.{}, // [rows, top_k], by choice score
    wts: []f64 = &.{}, // [rows, top_k]
    margin: []f64 = &.{}, // [rows]: the last pick's choice score minus the next one's

    pub fn deinit(r: *Result, gpa: std.mem.Allocator) void {
        gpa.free(r.out);
        gpa.free(r.picks);
        gpa.free(r.wts);
        gpa.free(r.margin);
        r.* = undefined;
    }
};

fn bf16At(bytes: []const u8, i: usize) f32 {
    return core.bf16_round.value(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little));
}

/// A bf16 tensor's values as fp32: exact.
pub fn widen(gpa: std.mem.Allocator, t: Tensor) ![]f32 {
    if (t.dtype != .bf16) return error.UnexpectedTensor;
    const out = try gpa.alloc(f32, t.numel());
    for (out, 0..) |*o, i| o.* = bf16At(t.bytes, i);
    return out;
}

fn dot(w: []const f32, x: []const f64) f64 {
    const V = 8;
    var acc: @Vector(V, f64) = @splat(0);
    var i: usize = 0;
    while (i + V <= w.len) : (i += V) {
        const wv: @Vector(V, f32) = w[i..][0..V].*;
        const xv: @Vector(V, f64) = x[i..][0..V].*;
        const wd: @Vector(V, f64) = @floatCast(wv);
        acc += wd * xv;
    }
    var s = @reduce(.Add, acc);
    while (i < w.len) : (i += 1) s += @as(f64, w[i]) * x[i];
    return s;
}

/// `f(ctx, lo, hi)` over [0, n) in one contiguous range a thread.
fn parallel(comptime Ctx: type, ctx: *const Ctx, n: usize, comptime f: fn (*const Ctx, usize, usize) void) void {
    const most = 64;
    const threads = @min(most, @max(1, std.Thread.getCpuCount() catch 1));
    var hs: [most]?std.Thread = @splat(null);
    const per = (n + threads - 1) / threads;
    for (0..threads) |t| {
        const lo = @min(n, t * per);
        const hi = @min(n, lo + per);
        if (lo >= hi) continue;
        hs[t] = std.Thread.spawn(.{}, f, .{ ctx, lo, hi }) catch blk: {
            f(ctx, lo, hi);
            break :blk null;
        };
    }
    for (hs) |h| if (h) |th| th.join();
}

/// out [rows, n] (ldo apart) = x [rows, k] (ldx apart) times w [n, k] transposed, rows in blocks of 16.
const Gemm = struct {
    w: []const f32,
    n: usize,
    k: usize,
    x: []const f64,
    ldx: usize,
    out: []f64,
    ldo: usize,

    fn run(g: *const Gemm, lo: usize, hi: usize) void {
        var r0 = lo;
        while (r0 < hi) : (r0 += 16) {
            const r1 = @min(hi, r0 + 16);
            for (0..g.n) |j| {
                const wr = g.w[j * g.k ..][0..g.k];
                for (r0..r1) |r| g.out[r * g.ldo + j] = dot(wr, g.x[r * g.ldx ..][0..g.k]);
            }
        }
    }
};

fn gemm(w: []const f32, n: usize, k: usize, x: []const f64, ldx: usize, out: []f64, ldo: usize, rows: usize) void {
    const g: Gemm = .{ .w = w, .n = n, .k = k, .x = x, .ldx = ldx, .out = out, .ldo = ldo };
    parallel(Gemm, &g, rows, Gemm.run);
}

/// x * rsqrt(mean(x^2) + eps) * w, every row in fp64.
fn norm(x: []const f64, w: Tensor, eps: f32, out: []f64, rows: usize, d: usize) void {
    for (0..rows) |r| {
        var sq: f64 = 0;
        for (x[r * d ..][0..d]) |v| sq += v * v;
        const inv = 1.0 / @sqrt(sq / @as(f64, @floatFromInt(d)) + @as(f64, eps));
        for (0..d) |i| out[r * d + i] = x[r * d + i] * inv * @as(f64, bf16At(w.bytes, i));
    }
}

fn silu(x: f64) f64 {
    return x / (1.0 + @exp(-x));
}

/// HYV4FlashMLP without its norm: down(silu(gate(y)) up(y)), [rows, hidden].
fn ffn(gpa: std.mem.Allocator, gate: Tensor, up: Tensor, down: Tensor, y: []const f64, rows: usize, d: usize, out: []f64) !void {
    const width = gate.dim(0);
    const wg = try widen(gpa, gate);
    defer gpa.free(wg);
    const wu = try widen(gpa, up);
    defer gpa.free(wu);
    const g = try gpa.alloc(f64, rows * width);
    defer gpa.free(g);
    const u = try gpa.alloc(f64, rows * width);
    defer gpa.free(u);
    gemm(wg, width, d, y, d, g, width, rows);
    gemm(wu, width, d, y, d, u, width, rows);
    for (g, u) |*a, b| a.* = silu(a.*) * b;
    const wd = try widen(gpa, down);
    defer gpa.free(wd);
    gemm(wd, d, width, g, width, out, d, rows);
}

/// One layer's MLP on `x` [rows, hidden] (the norm's input): post_attention_layernorm, the body, its closing norm.
pub fn run(gpa: std.mem.Allocator, c: Config, l: names.Layer, x: []const f64, rows: usize, forced: ?Forced) !Result {
    const d = c.hidden;
    const y = try gpa.alloc(f64, rows * d);
    defer gpa.free(y);
    norm(x, l.post_attn_norm, c.eps, y, rows, d);
    const h = try gpa.alloc(f64, rows * d);
    defer gpa.free(h);
    var res: Result = .{ .out = try gpa.alloc(f64, rows * d) };
    errdefer res.deinit(gpa);
    switch (l.mlp) {
        .dense => |m| {
            try ffn(gpa, m.gate, m.up, m.down, y, rows, d, h);
            norm(h, m.post_norm, c.eps, res.out, rows, d);
        },
        .moe => |m| {
            try moe(gpa, c, m, y, rows, forced, &res, h);
            norm(h, m.shared.post_norm, c.eps, res.out, rows, d);
        },
    }
    return res;
}

/// HYV4FlashTopkRouter on one row: sigmoid scores, + bias to choose (ties: the lower expert), normalized, scaled.
fn route(c: Config, logits: []const f64, bias: Tensor, picks: []u32, wts: []f64) f64 {
    var choice: [1024]f64 = undefined;
    var score: [1024]f64 = undefined;
    for (logits, 0..) |v, e| {
        score[e] = 1.0 / (1.0 + @exp(-v));
        choice[e] = score[e] + @as(f64, @as(f32, @bitCast(std.mem.readInt(u32, bias.bytes[4 * e ..][0..4], .little))));
    }
    var taken: [1024]bool = @splat(false);
    var next: f64 = 0;
    for (0..c.top_k + 1) |k| {
        var best: usize = logits.len;
        for (0..logits.len) |e| if (!taken[e] and (best == logits.len or choice[e] > choice[best])) {
            best = e;
        };
        if (k == c.top_k) {
            next = choice[best];
            break;
        }
        taken[best] = true;
        picks[k] = @intCast(best);
    }
    var sum: f64 = 0;
    for (picks) |p| sum += score[p];
    for (picks, wts) |p, *w| w.* = score[p] / (sum + 1e-20) * @as(f64, c.routed_scale);
    return choice[picks[c.top_k - 1]] - next;
}

fn moe(gpa: std.mem.Allocator, c: Config, m: names.Moe, y: []const f64, rows: usize, forced: ?Forced, res: *Result, h: []f64) !void {
    const d = c.hidden;
    const kk = c.top_k;
    res.picks = try gpa.alloc(u32, rows * kk);
    res.wts = try gpa.alloc(f64, rows * kk);
    res.margin = try gpa.alloc(f64, rows);
    {
        const wr = try widen(gpa, m.router);
        defer gpa.free(wr);
        const logits = try gpa.alloc(f64, rows * c.experts);
        defer gpa.free(logits);
        gemm(wr, c.experts, d, y, d, logits, c.experts, rows);
        for (0..rows) |r| res.margin[r] = route(c, logits[r * c.experts ..][0..c.experts], m.bias, res.picks[r * kk ..][0..kk], res.wts[r * kk ..][0..kk]);
    }
    if (forced) |f| for (res.picks, res.wts, f.picks, f.wts) |*p, *w, fp, fw| {
        p.* = @intCast(fp);
        w.* = fw;
    };
    const lat = try gpa.alloc(f64, rows * c.latent);
    defer gpa.free(lat);
    {
        const wl = try widen(gpa, m.latent_down);
        defer gpa.free(wl);
        gemm(wl, c.latent, d, y, d, lat, c.latent, rows);
    }
    const pairs = try gpa.alloc(f64, rows * kk * c.latent);
    defer gpa.free(pairs);
    try routed(gpa, c, m, res.picks, lat, rows, pairs);
    const sum = try gpa.alloc(f64, rows * c.latent);
    defer gpa.free(sum);
    @memset(sum, 0);
    for (0..rows) |r| for (0..kk) |k| for (0..c.latent) |i| {
        sum[r * c.latent + i] += res.wts[r * kk + k] * pairs[(r * kk + k) * c.latent + i];
    };
    const wu = try widen(gpa, m.latent_up);
    defer gpa.free(wu);
    gemm(wu, d, c.latent, sum, c.latent, h, d, rows);
    const shared = try gpa.alloc(f64, rows * d);
    defer gpa.free(shared);
    try ffn(gpa, m.shared.gate, m.shared.up, m.shared.down, y, rows, d, shared);
    for (h, shared) |*a, b| a.* += b;
}

/// Each routed pair's expert output in the latent width (unweighted): experts in parallel, their pairs grouped.
const Experts = struct {
    m: names.Moe,
    lut: *const [256]f32,
    latent: []const f64, // [rows, L]
    order: []const u32, // pair ids grouped by expert
    start: []const usize, // [experts + 1]
    slots: usize,
    limit: f64,
    out: []f64, // [pairs, L]

    fn run(x: *const Experts, lo: usize, hi: usize) void {
        x.span(lo, hi) catch @panic("hy4-mlp reference: out of memory");
    }

    fn span(x: *const Experts, lo: usize, hi: usize) !void {
        const two = x.m.gate_up.dim(1);
        const l = x.m.gate_up.dim(2);
        const w = two / 2;
        const b = 128;
        const gu = try page.alloc(f32, two * l);
        defer page.free(gu);
        const dn = try page.alloc(f32, l * w);
        defer page.free(dn);
        for (lo..hi) |e| {
            const ps = x.order[x.start[e]..x.start[e + 1]];
            if (ps.len == 0) continue;
            for (0..two) |j| for (0..l) |i| {
                const s = bf16At(x.m.gate_up_s.bytes, (e * (two / b) + j / b) * (l / b) + i / b);
                gu[j * l + i] = x.lut[x.m.gate_up.bytes[(e * two + j) * l + i]] * s;
            };
            for (0..l) |j| for (0..w) |i| {
                const s = bf16At(x.m.down_s.bytes, (e * (l / b) + j / b) * (w / b) + i / b);
                dn[j * w + i] = x.lut[x.m.down.bytes[(e * l + j) * w + i]] * s;
            };
            const act = try page.alloc(f64, ps.len * two);
            defer page.free(act);
            for (0..two) |j| for (ps, 0..) |p, q| {
                act[q * two + j] = dot(gu[j * l ..][0..l], x.latent[(p / x.slots) * l ..][0..l]);
            };
            for (0..ps.len) |q| for (0..w) |i| {
                const g = @min(act[q * two + i], x.limit);
                const u = std.math.clamp(act[q * two + w + i], -x.limit, x.limit);
                act[q * two + i] = silu(g) * u;
            };
            for (0..l) |j| for (ps, 0..) |p, q| {
                x.out[p * l + j] = dot(dn[j * w ..][0..w], act[q * two ..][0..w]);
            };
        }
    }
};

fn routed(gpa: std.mem.Allocator, c: Config, m: names.Moe, picks: []const u32, lat: []const f64, rows: usize, out: []f64) !void {
    var lut: [256]f32 = undefined;
    for (&lut, 0..) |*v, i| v.* = @floatCast(core.quant.fp8.e4m3(@intCast(i)));
    const start = try gpa.alloc(usize, c.experts + 1);
    defer gpa.free(start);
    @memset(start, 0);
    for (picks) |p| start[p + 1] += 1;
    for (1..start.len) |e| start[e] += start[e - 1];
    const order = try gpa.alloc(u32, picks.len);
    defer gpa.free(order);
    const at = try gpa.dupe(usize, start[0..c.experts]);
    defer gpa.free(at);
    for (picks, 0..) |p, i| {
        order[at[p]] = @intCast(i);
        at[p] += 1;
    }
    _ = rows;
    const x: Experts = .{ .m = m, .lut = &lut, .latent = lat, .order = order, .start = start, .slots = c.top_k, .limit = c.swiglu_limit, .out = out };
    parallel(Experts, &x, c.experts, Experts.run);
}
