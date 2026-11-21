//! Hy4-flash's absorbed gated MLA on CUDA (docs/recipes/hy4-flash.md): launches and rank forward to post_proj_norm.
const std = @import("std");
const cuda = @import("cuda");
pub const weights = @import("cuda_mla_weights.zig");
const c1 = @import("cuda_mla_c1.zig");
const c2 = @import("cuda_mla_c2.zig");
const cuda_prof = @import("cuda_prof.zig");

const Stream = cuda.Stream;
const Function = cuda.Function;
const DeviceBuffer = cuda.DeviceBuffer;
const Args = cuda.Args;
const Weights = weights.Weights;

pub const latent = weights.latent;
pub const rope = weights.rope;
pub const row = latent + rope; // a cache row: the normed latent, then the roped key (bf16)
pub const heads_tile = weights.heads_tile;
pub const scale: f32 = 0.0625; // (nope + rope)^-0.5, exact
/// tf_hy4::kAttnSmem: q, a 32-key chunk on padded rows, its weights and positions, the heads' m, l and rescale.
pub const attn_smem: u32 = heads_tile * row * 4 + 32 * (row + 2) * 2 + heads_tile * 32 * 4 + 32 * 4 + 3 * heads_tile * 4;
/// tf_hy4_mla_post's widest row: 256 threads of 16.
pub const post_most = 4096;

pub const Kernels = struct {
    mod: cuda.Module,
    absorb: Function,
    attn: Function,
    attn_rows: Function, // tf_hy4_mla_attn_rows: row r's cache at a device u64 table's entry r
    value: Function,
    post: Function,
    c1: c1.C1,
    c2: c2.C2,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.hy4_mla);
        errdefer mod.unload();
        var fast = try c1.C1.load(d);
        errdefer fast.unload();
        var tc = try c2.C2.load(d);
        errdefer tc.unload();
        const k: Kernels = .{
            .mod = mod,
            .absorb = try mod.function("tf_hy4_mla_absorb"),
            .attn = try mod.function("tf_hy4_mla_attn"),
            .attn_rows = try mod.function("tf_hy4_mla_attn_rows"),
            .value = try mod.function("tf_hy4_mla_value"),
            .post = try mod.function("tf_hy4_mla_post"),
            .c1 = fast,
            .c2 = tc,
        };
        try k.attn.allowDynamicShared(attn_smem);
        try k.attn_rows.allowDynamicShared(attn_smem);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.c2.unload();
        k.c1.unload();
        k.mod.unload();
        k.* = undefined;
    }
};

fn cdiv(a: usize, b: usize) u32 {
    return @intCast((a + b - 1) / b);
}

fn int(v: usize) c_int {
    return @intCast(v);
}

/// ql [rows][heads][512] fp32 = each head's 192 nope q columns (rows `qs` values apart, heads 256) . kv_b's key rows.
pub fn absorb(f: Function, s: Stream, q: u64, qs: usize, kvb: u64, ql: u64, rows: usize, heads: usize) !void {
    if (rows == 0) return;
    var a: Args = .{};
    inline for (.{ q, @as(i64, @intCast(qs)), kvb, ql, int(rows), int(heads) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(heads), .y = latent / 128, .z = cdiv(rows, 16) }, .block = .{ .x = 128 } }, s, &a);
}

/// Keys a row sees: `window` ending at pos (0: causal), else `nidx` listed positions (< 0 or > pos masked); p % ring.
pub const Keys = struct { pos: u64, window: usize = 0, idx: u64 = 0, nidx: usize = 0, ring: usize = 0 };

/// ol [rows][heads][512] fp32: the sinked softmax over the latent cache, by tf_hy4_mla_attn (or a control in its form).
pub fn attn(f: Function, s: Stream, ql: u64, qr: u64, cache: u64, keys: Keys, sinks: u64, ol: u64, rows: usize, heads: usize) !void {
    if (rows == 0) return;
    if (heads % heads_tile != 0) return error.MlaHeadsSplit;
    var a: Args = .{};
    inline for (.{ ql, qr, cache, keys.pos, keys.idx, sinks, ol }) |v| a.add(v);
    inline for (.{ heads, keys.window, keys.nidx, keys.ring }) |v| a.add(int(v));
    a.add(scale);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(heads / heads_tile) }, .block = .{ .x = 256 }, .shared = attn_smem }, s, &a);
}

/// out [rows][heads x 256] bf16 (rows `os` apart) = (ol . kv_b's value rows) * sigmoid(gate rows `gs` apart).
pub fn value(f: Function, s: Stream, ol: u64, kvb: u64, gate: u64, gs: usize, out: u64, os: usize, rows: usize, heads: usize) !void {
    if (rows == 0) return;
    var a: Args = .{};
    inline for (.{ ol, kvb, gate, @as(i64, @intCast(gs)), out, @as(i64, @intCast(os)), int(rows), int(heads) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(heads), .y = weights.v_width / 32, .z = cdiv(rows, 16) }, .block = .{ .x = 128 } }, s, &a);
}

/// out [rows][width] bf16 = post_proj_norm(bf16(the `world` fp32 partial planes [rows][width] summed in rank order)).
pub fn post(f: Function, s: Stream, parts: u64, world: usize, w: u64, out: u64, rows: usize, width: usize, eps: f32) !void {
    if (rows == 0) return;
    if (width > post_most or world == 0) return error.RowShape;
    var a: Args = .{};
    inline for (.{ parts, @as(i64, @intCast(rows * width)), int(world), w, out, int(width) }) |v| a.add(v);
    a.add(eps);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
}

/// The shared operators the forward also launches.
pub const Ops = struct {
    k: *const Kernels,
    lane: *const cuda.qmmf.Lane,
    norms: *const cuda.norms.Norms,
    mla: *const cuda.mla.Mla, // tf_mla_rope: GPT-J pairs at fp32 angles, as Hy4's interleaved rope
    prof: ?*cuda_prof.Prof = null,
    rows_op: ?*const @import("cuda_rows.zig").Rows = null,
    kv8: bool = false,
    rot_pos: u64 = 0,
    pos3: u64 = 0,
    ld3: usize = 0,
    row_stream: u64 = 0, // batched paged indexed KV: device i32 [rows] slot indices
    pt_base: u64 = 0, // batched paged indexed KV: base of [max_slots, pt_stride] u32
};

/// One rank's working rows for up to `rows` rows of a layer with `heads` heads on it.
pub const Scratch = struct {
    rows: usize,
    heads: usize,
    bufs: [9]DeviceBuffer,
    part: ?DeviceBuffer, // the lane's K slices where they meet in memory (no clusters)

    const Name = enum { qa, qres, kva, q, qr, ql, ol, gate, gated };

    pub fn init(d: *const cuda.Driver, lane: *const cuda.qmmf.Lane, w: *const Weights, rows: usize) !Scratch {
        const h = w.heads;
        const q_lora: usize = w.q_lora;
        const sizes = [9]usize{ rows * (q_lora + row) * 2, rows * q_lora * 2, 16, rows * h * weights.q_head * 2, rows * h * rope * 2, rows * h * row * 4, rows * h * latent * 4, rows * h * weights.v_width * 2, rows * h * weights.v_width * 2 };
        var sc: Scratch = .{ .rows = rows, .heads = h, .bufs = undefined, .part = null };
        var n: usize = 0;
        errdefer for (sc.bufs[0..n]) |*b| b.free();
        for (&sc.bufs, sizes) |*b, len| {
            b.* = try DeviceBuffer.alloc(d, len);
            n += 1;
        }
        var part: usize = 0;
        for ([_]cuda.qmmf.Weight{ w.qkv_a.w, w.q_b.w, w.gate.w, w.o.w }) |pw| part = @max(part, lane.partBytes(rows, pw));
        if (part > 0) sc.part = try DeviceBuffer.alloc(d, part);
        return sc;
    }

    pub fn deinit(sc: *Scratch) void {
        for (&sc.bufs) |*b| b.free();
        if (sc.part) |*p| p.free();
    }

    fn lanePart(sc: *const Scratch) u64 {
        return if (sc.part) |p| p.ptr else 0;
    }

    pub fn at(sc: *const Scratch, comptime name: Name) u64 {
        return sc.bufs[@backingInt(name)].ptr;
    }

    pub fn qres(sc: *const Scratch) u64 {
        return sc.at(.qres);
    }
};

/// A call's rows: attn_in, consecutive positions from `first`, the cache, and optional index lists from `list_from`.
pub const Call = struct {
    x: u64, // bf16 [rows][hidden]
    rows: usize,
    first: usize, // the first row's position
    pos: u64, // i32 [rows]: first, first + 1, ...
    cache: u64, // bf16 [slots][576]
    ring: usize = 0, // slots of a ring; 0: a position's row is its own slot
    list_from: ?usize = null, // rows from here read idx; null: none do
    idx: u64 = 0, // i32 [rows - list_from][nidx]
    nidx: usize = 0,
    part: u64, // fp32 [rows][hidden]: this rank's o_proj partial
    single_split: bool = false,
};

/// Cache rows of `n` consecutive positions from `first`: at most two runs of slots, the second from slot 0 of a ring.
pub fn runs(first: usize, n: usize, ring: usize) [2][2]usize {
    if (ring == 0) return .{ .{ first, n }, .{ 0, 0 } };
    const s0 = first % ring;
    const n0 = @min(n, ring - s0);
    return .{ .{ s0, n0 }, .{ 0, n - n0 } };
}

pub const GatherBf16 = struct {
    ctx: *anyopaque,
    gather: *const fn (ctx: *anyopaque, s: Stream, send: u64, recv: u64, count_bf16: usize) anyerror!void,
};

/// Computes `[q_a, kv_a] = qkv_a(x)` into `.qa` and `q_resid = q_a_layernorm(q_a)` (TP=2 `qkv_a_half` at `rows <= 64`).
pub fn qkvPrep(ops: Ops, s: Stream, w: *const Weights, sc: *const Scratch, x: u64, rows: usize, gather: ?GatherBf16, rows_op: ?*const @import("cuda_rows.zig").Rows) !void {
    if (rows > sc.rows or w.heads != sc.heads) return error.ScratchShape;
    if (rows == 0) return;
    const q_lora: usize = w.q_lora;
    const qkv_stride = q_lora + row;
    if (rows <= 64 and w.qkv_sharded) {
        if (w.qkv_a_half) |qh| {
            const g = gather orelse return error.MissingAllReduce;
            const half = qkv_stride / 2;
            try ops.lane.matmul(s, x, w.hidden, rows, qh.w, sc.at(.q), sc.lanePart());
            if (rows == 1) {
                try g.gather(g.ctx, s, sc.at(.q), sc.at(.qa), half);
            } else {
                try g.gather(g.ctx, s, sc.at(.q), sc.at(.ql), rows * half);
                try rows_op.?.interleave2Bf16(s, sc.at(.ql), sc.at(.qa), half, rows);
            }
            try ops.norms.rmsParts(s, sc.at(.qa), qkv_stride, 0, w.q_a_norm.ptr, 0, sc.at(.qres), q_lora, 0, null, rows, 1, q_lora, w.eps, false);
            return;
        }
    }
    try ops.lane.matmul(s, x, w.hidden, rows, w.qkv_a.w, sc.at(.qa), sc.lanePart());
    try ops.norms.rmsParts(s, sc.at(.qa), qkv_stride, 0, w.q_a_norm.ptr, 0, sc.at(.qres), q_lora, 0, null, rows, 1, q_lora, w.eps, false);
}

/// q_resid [rows, q_lora] bf16 = q_a_layernorm(q_a_proj(x)).
pub fn qResid(ops: Ops, s: Stream, w: *const Weights, sc: *const Scratch, x: u64, rows: usize) !void {
    return qkvPrep(ops, s, w, sc, x, rows, null, null);
}

/// One sequence's rows `[row0, row0 + rows)` of a batched call: its cache, positions from `first`, index lists.
pub const Seg = struct {
    row0: usize,
    rows: usize,
    first: usize,
    cache: u64, // bf16 [slots][576]
    ring: usize = 0,
    list_from: ?usize = null, // segment-relative
    idx: u64 = 0, // i32 [rows - list_from][nidx]
    nidx: usize = 0,
    page_table: u64 = 0,
    pt_stride: usize = 0,
};

/// attn_in to this rank's fp32 o_proj partial: qkv_a, kv_a + k_rope to cache, q_b + q_rope, absorb, attn, gate and o.
pub fn attend(ops: Ops, s: Stream, w: *const Weights, sc: *const Scratch, c: Call) !void {
    const one = [1]Seg{.{ .row0 = 0, .rows = c.rows, .first = c.first, .cache = c.cache, .ring = c.ring, .list_from = c.list_from, .idx = c.idx, .nidx = c.nidx }};
    try attendCore(ops, s, w, sc, c.x, c.rows, c.pos, &one, 0, false, c.single_split);
    return attendOut(ops, s, w, sc, c.x, c.rows, c.part);
}

/// `attend` over segments (`caches`: per-row cache ptrs or 0; `qkv_ready`: `qkvPrep` already ran before indexer).
pub fn attendSegs(ops: Ops, s: Stream, w: *const Weights, sc: *const Scratch, x: u64, rows: usize, pos: u64, part: u64, segs: []const Seg, caches: u64, qkv_ready: bool) !void {
    try attendCore(ops, s, w, sc, x, rows, pos, segs, caches, qkv_ready, false);
    try attendOut(ops, s, w, sc, x, rows, part);
}

/// Position/cache-dependent half of MLA: `kvWrite` -> `q_b` -> `qRope` -> `c2.absorb` -> `c1.attn` into `sc.at(.ol)`.
pub fn attendCore(ops: Ops, s: Stream, w: *const Weights, sc: *const Scratch, x: u64, rows: usize, pos: u64, segs: []const Seg, caches: u64, qkv_ready: bool, single_split: bool) !void {
    if (rows > sc.rows or w.heads != sc.heads) return error.ScratchShape;
    var covered: usize = 0;
    for (segs) |g| {
        if (g.row0 != covered or (g.ring != 0 and g.rows > g.ring)) return error.RingShape;
        covered += g.rows;
    }
    if (covered != rows) return error.ScratchShape;
    const h = w.heads;
    const q_lora: usize = w.q_lora;
    const qkv_stride = q_lora + row;
    const lp = sc.lanePart();
    if (!qkv_ready) try qkvPrep(ops, s, w, sc, x, rows, null, null);
    if (ops.rows_op) |rop| {
        const paged = segs[0].page_table != 0 and segs[0].ring == 0;
        if (paged) {
            if (ops.row_stream != 0 and ops.pt_base != 0 and oneLaunch(segs)) {
                try rop.kvWritePaged(s, sc.at(.qa), qkv_stride, q_lora, w.kv_a_norm.ptr, w.freqs.ptr, pos, ops.rot_pos, ops.pos3, ops.ld3, ops.pt_base, ops.row_stream, segs[0].pt_stride, segs[0].cache, rows, w.eps, ops.kv8);
            } else {
                for (segs) |g| {
                    const r_rot = if (ops.rot_pos != 0) ops.rot_pos + g.row0 * 4 else 0;
                    const r_pos3 = if (ops.pos3 != 0) ops.pos3 + g.row0 * 4 else 0;
                    const ld3 = if (ops.pos3 != 0) ops.ld3 else 0;
                    try rop.kvWritePaged(s, sc.at(.qa) + g.row0 * qkv_stride * 2, qkv_stride, q_lora, w.kv_a_norm.ptr, w.freqs.ptr, pos + g.row0 * 4, r_rot, r_pos3, ld3, g.page_table, 0, g.pt_stride, g.cache, g.rows, w.eps, ops.kv8);
                }
            }
        } else if (caches != 0 and oneLaunch(segs)) {
            try rop.kvWriteEx(s, sc.at(.qa), qkv_stride, q_lora, w.kv_a_norm.ptr, w.freqs.ptr, pos, ops.rot_pos, ops.pos3, ops.ld3, caches, 0, segs[0].ring, rows, w.eps, ops.kv8);
        } else if (segs.len == 1) {
            try rop.kvWriteEx(s, sc.at(.qa), qkv_stride, q_lora, w.kv_a_norm.ptr, w.freqs.ptr, pos, ops.rot_pos, ops.pos3, ops.ld3, 0, segs[0].cache, segs[0].ring, rows, w.eps, ops.kv8);
        } else {
            for (segs) |g| {
                const r_rot = if (ops.rot_pos != 0) ops.rot_pos + g.row0 * 4 else 0;
                const r_pos3 = if (ops.pos3 != 0) ops.pos3 + g.row0 * 4 else 0;
                const ld3 = if (ops.pos3 != 0) ops.ld3 else 0;
                try rop.kvWriteEx(s, sc.at(.qa) + g.row0 * qkv_stride * 2, qkv_stride, q_lora, w.kv_a_norm.ptr, w.freqs.ptr, pos + g.row0 * 4, r_rot, r_pos3, ld3, 0, g.cache, g.ring, g.rows, w.eps, ops.kv8);
            }
        }
    } else {
        for (segs) |g| {
            var r0: usize = g.row0;
            for (runs(g.first, g.rows, g.ring)) |run| {
                if (run[1] == 0) continue;
                const dst = g.cache + run[0] * row * 2;
                const src = sc.at(.qa) + (r0 * qkv_stride + q_lora) * 2;
                try ops.norms.rmsParts(s, src, qkv_stride, 0, w.kv_a_norm.ptr, 0, dst, row, 0, null, run[1], 1, latent, w.eps, false);
                try ops.mla.rope(s, .{ .x = src + latent * 2, .x_row = qkv_stride, .x_head = 0, .out = dst + latent * 2, .o_row = row, .o_head = 0 }, run[1], 1, pos + r0 * 4, w.freqs.ptr, 1.0);
                r0 += run[1];
            }
        }
    }
    if (ops.prof) |p| p.mark(s, .attn_mla);
    const qs = h * weights.q_head;
    try ops.lane.matmul(s, sc.at(.qres), q_lora, rows, w.q_b.w, sc.at(.q), lp);
    if (ops.prof) |p| p.mark(s, .attn_qmmf);
    const qa = sc.at(.ql);
    const qlo: usize = if (ops.k.c1.exact) rows * h * row else 0;
    if (ops.rows_op) |rop| {
        const q_pos = if (ops.rot_pos != 0) ops.rot_pos else pos;
        try rop.qRopeMrope(s, sc.at(.q) + weights.nope * 2, qs, weights.q_head, qa + latent * 2, h * row, row, q_pos, ops.pos3, ops.ld3, w.freqs.ptr, rows, h);
    } else for (segs) |g| { // tf_mla_rope angles row r at pos[0] + r: one launch per segment's consecutive positions
        const q0 = sc.at(.q) + g.row0 * qs * 2 + weights.nope * 2;
        const qr0 = qa + g.row0 * h * row * 2 + latent * 2;
        try ops.mla.rope(s, .{ .x = q0, .x_row = qs, .x_head = weights.q_head, .out = qr0, .o_row = h * row, .o_head = row }, g.rows, h, pos + g.row0 * 4, w.freqs.ptr, 1.0);
    }
    try c2.absorb(&ops.k.c2, s, sc.at(.q), qs, rows, h, w.uk.ptr, qa, qlo);
    const lat = h * latent;
    if (w.window != 0 and w.window != 512) return error.MlaWindow;
    const kind: c_int = if (w.window != 0) 0 else 1;
    if (ops.row_stream != 0 and ops.pt_base != 0 and segs[0].ring == 0 and oneLaunch(segs)) {
        try c1.attn(&ops.k.c1, s, qa, qlo, h, segs[0].cache, .{ .kind = kind, .ring = 0, .single_split = single_split, .kv8 = ops.kv8, .page_table = ops.pt_base, .pt_stride = segs[0].pt_stride, .row_stream = ops.row_stream }, pos, w.sinks.ptr, sc.at(.ol), rows);
    } else if (caches != 0 and oneLaunch(segs)) {
        try c1.attnRows(&ops.k.c1, s, qa, qlo, h, caches, .{ .kind = kind, .ring = segs[0].ring, .single_split = single_split, .kv8 = ops.kv8 }, pos, w.sinks.ptr, sc.at(.ol), rows);
    } else for (segs) |g| {
        const dense = @min(g.list_from orelse g.rows, g.rows);
        const qa_g = qa + g.row0 * h * row * 2;
        const ol = sc.at(.ol) + g.row0 * lat * 4;
        const gp = pos + g.row0 * 4;
        try c1.attn(&ops.k.c1, s, qa_g, qlo, h, g.cache, .{ .kind = kind, .ring = g.ring, .single_split = single_split, .kv8 = ops.kv8, .page_table = g.page_table, .pt_stride = g.pt_stride }, gp, w.sinks.ptr, ol, dense);
        if (dense < g.rows) {
            if (g.nidx == 0 or w.window != 0) return error.IndexList;
            const keys: c1.Keys = .{ .kind = 2, .ring = g.ring, .tok = g.idx, .ntok = g.nidx, .single_split = single_split, .kv8 = ops.kv8, .page_table = g.page_table, .pt_stride = g.pt_stride };
            try c1.attn(&ops.k.c1, s, qa_g + dense * h * row * 2, qlo, h, g.cache, keys, gp + dense * 4, w.sinks.ptr, ol + dense * lat * 4, g.rows - dense);
        }
    }
    if (ops.prof) |p| p.mark(s, .attn_mla);
}

/// Position-independent post-attention half of MLA (`gate` -> `c2.expand` -> `o_proj` from `sc.at(.ol)` into `part`).
pub fn attendOut(ops: Ops, s: Stream, w: *const Weights, sc: *const Scratch, x: u64, rows: usize, part: u64) !void {
    const h = w.heads;
    const hd = w.hidden;
    const lp = sc.lanePart();
    const vs = h * weights.v_width;
    try ops.lane.matmul(s, x, hd, rows, w.gate.w, sc.at(.gate), lp);
    if (ops.prof) |p| p.mark(s, .attn_qmmf);
    try c2.expand(&ops.k.c2, s, sc.at(.ol), rows, h, w.uv.ptr, sc.at(.gate), vs, sc.at(.gated));
    if (ops.prof) |p| p.mark(s, .attn_mla);
    try ops.lane.matmulF32(s, sc.at(.gated), vs, rows, w.o.w, part, lp);
    if (ops.prof) |p| p.mark(s, .attn_qmmf);
}

/// Segments one tf_hy4_mla_attn_rows launch can take: none reads an index list, all on the same ring.
fn oneLaunch(segs: []const Seg) bool {
    for (segs) |g| if (g.list_from != null or g.ring != segs[0].ring) return false;
    return true;
}

/// The layer's output rows from the `world` ranks' partials (planes [rows][hidden], rank order): post_proj_norm.
pub fn finish(ops: Ops, s: Stream, w: *const Weights, parts: u64, world: usize, rows: usize, out: u64) !void {
    try post(ops.k.post, s, parts, world, w.post_norm.ptr, out, rows, w.hidden, w.eps);
}

test "runs: a ring's slots wrap once, a linear cache's never" {
    try std.testing.expectEqual([2][2]usize{ .{ 700, 9 }, .{ 0, 0 } }, runs(700, 9, 0));
    try std.testing.expectEqual([2][2]usize{ .{ 1020, 4 }, .{ 0, 5 } }, runs(2044, 9, 1024));
    try std.testing.expectEqual([2][2]usize{ .{ 3, 9 }, .{ 0, 0 } }, runs(1027, 9, 1024));
}

test "the attention's shared bytes are tf_hy4::kAttnSmem's" {
    try std.testing.expectEqual(@as(u32, 56672), attn_smem);
    _ = weights;
}
