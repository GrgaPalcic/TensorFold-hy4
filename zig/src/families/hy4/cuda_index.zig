//! Hy4-flash's compressed indexer on CUDA (hy4_index.cu): token keys, 4:1 pooling, scores, and top-511 selection.
const std = @import("std");
const cuda = @import("cuda");
pub const weights = @import("cuda_index_weights.zig");

const Stream = cuda.Stream;
const Function = cuda.Function;
const DeviceBuffer = cuda.DeviceBuffer;
const Args = cuda.Args;
const Weights = weights.Weights;

pub const heads = weights.heads;
pub const dim = weights.dim;
pub const ratio = weights.ratio;
pub const groups = weights.groups;
pub const rope = weights.rope;
pub const rope_pairs = weights.rope_pairs;
pub const tokens_per_row = groups * ratio;
pub const small_max_jobs: usize = 48;
pub const select_cap: usize = 4096;
pub const select_scan_span: usize = 2048;
/// Max rows scored/selected per sub-tile (bounds `sc` & `bits` scratch to ~268 MiB even at 512K context).
pub const score_tile: usize = 512;
pub const tail_ring_rows: usize = 32;
pub const pool_1cta_smem: u32 = (16 * 4 * dim + 2 * 64 * 32) * 4;
pub const pool_persist_smem: u32 = (64 * dim + 2 * 128 * 16 + 2 * 32) * 4;

pub const Kernels = struct {
    mod: cuda.Module,
    rope_table: Function,
    pool_1cta: Function,
    pool_small: Function,
    pool_persist: Function,
    q_mrope: Function,
    score_decode: Function,
    score_decode_rope: Function,
    score_prefill: Function,
    select_1cta: Function,
    select_scan: Function,
    select_merge: Function,
    pool_1cta_paged: Function,
    pool_small_paged: Function,
    pool_persist_paged: Function,
    score_decode_paged: Function,
    score_decode_rope_paged: Function,
    score_prefill_paged: Function,
    clusters: bool,
    pdl: bool = false,
    sms: usize,
    occ_pre: usize,
    persist_clusters: usize,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const cap = try d.currentCapability();
        var mod = try cuda.Module.load(d, cuda.kernels.hy4_index);
        errdefer mod.unload();
        var k: Kernels = .{
            .mod = mod,
            .rope_table = try mod.function("tf_hy4_idx_rope_table"),
            .pool_1cta = try mod.function("tf_hy4_idx_pool_1cta"),
            .pool_small = try mod.function("tf_hy4_idx_pool_small"),
            .pool_persist = try mod.function("tf_hy4_idx_pool_persist"),
            .q_mrope = try mod.function("tf_hy4_idx_q_mrope"),
            .score_decode = try mod.function("tf_hy4_idx_score_decode"),
            .score_decode_rope = try mod.function("tf_hy4_idx_score_decode_rope"),
            .score_prefill = try mod.function("tf_hy4_idx_score_prefill"),
            .select_1cta = try mod.function("tf_hy4_idx_select_1cta"),
            .select_scan = try mod.function("tf_hy4_idx_select_scan"),
            .select_merge = try mod.function("tf_hy4_idx_select_merge"),
            .pool_1cta_paged = try mod.function("tf_hy4_idx_pool_1cta_paged"),
            .pool_small_paged = try mod.function("tf_hy4_idx_pool_small_paged"),
            .pool_persist_paged = try mod.function("tf_hy4_idx_pool_persist_paged"),
            .score_decode_paged = try mod.function("tf_hy4_idx_score_decode_paged"),
            .score_decode_rope_paged = try mod.function("tf_hy4_idx_score_decode_rope_paged"),
            .score_prefill_paged = try mod.function("tf_hy4_idx_score_prefill_paged"),
            .clusters = cap >= 120,
            .pdl = cap >= 90,
            .sms = 1,
            .occ_pre = 1,
            .persist_clusters = 1,
        };
        try k.pool_1cta.allowDynamicShared(pool_1cta_smem);
        if (k.clusters) try k.pool_persist.allowDynamicShared(pool_persist_smem);
        try k.pool_1cta_paged.allowDynamicShared(pool_1cta_smem);
        if (k.clusters) try k.pool_persist_paged.allowDynamicShared(pool_persist_smem);
        var dev: cuda.abi.Device = 0;
        try d.check(d.api.cuCtxGetDevice(&dev), "cuCtxGetDevice");
        var sm_count: c_int = 1;
        try d.check(d.api.cuDeviceGetAttribute(&sm_count, .multiprocessor_count, dev), "cuDeviceGetAttribute");
        k.sms = @intCast(@max(sm_count, 1));
        k.occ_pre = @max(1, try k.score_prefill.occupancy(256, 0));
        if (k.clusters) {
            const occ_p = try k.pool_persist.occupancy(256, pool_persist_smem);
            k.persist_clusters = @max(1, (occ_p * k.sms) / 4);
        }
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.* = undefined;
    }
};

pub const Ops = struct {
    k: *const Kernels,
    lane: *const cuda.qmmf.Lane,
    norms: *const cuda.norms.Norms,
    dense: *const cuda.dense_rows.DenseRows,
};

/// Persistent indexer state for one stream up to `cap` tokens (`cap % 4 == 0`).
pub const Cache = struct {
    cap: usize,
    token_keys: DeviceBuffer, // bf16 [cap, 128]
    group_keys: DeviceBuffer, // bf16 [cap / 4, 128]
    rope_cs: DeviceBuffer, // float2 [cap, 32]
    rope_ready: usize = 0,

    pub fn init(d: *const cuda.Driver, cap: usize) !Cache {
        if (cap == 0 or cap % ratio != 0) return error.IndexCacheShape;
        var tk = try DeviceBuffer.alloc(d, cap * dim * 2);
        errdefer tk.free();
        var gk = try DeviceBuffer.alloc(d, (cap / ratio) * dim * 2);
        errdefer gk.free();
        const cs = try DeviceBuffer.alloc(d, cap * rope_pairs * 8);
        return .{ .cap = cap, .token_keys = tk, .group_keys = gk, .rope_cs = cs };
    }

    pub fn deinit(c: *Cache) void {
        c.token_keys.free();
        c.group_keys.free();
        c.rope_cs.free();
        c.* = undefined;
    }

    pub fn reset(c: *Cache) void {
        c.rope_ready = 0;
    }

    /// Populates rope_cs[0 .. end) incrementally for any positions not yet written.
    pub fn ensureRope(c: *Cache, ops: Ops, s: Stream, w: *const Weights, end: usize) !void {
        if (end > c.cap) return error.IndexCacheOverflow;
        if (end <= c.rope_ready) return;
        const first = c.rope_ready;
        try ropeTable(ops.k.rope_table, s, c.rope_cs.ptr, first, end - first, w.freqs.ptr, 0, 0, 10, 10, false);
        c.rope_ready = end;
    }
};

/// Working scratch for up to `rows` rows per call and contexts up to `cap` tokens.
pub const Scratch = struct {
    rows: usize,
    score_rows: usize,
    max_groups: usize,
    sc_stride: usize,
    arena: DeviceBuffer,
    ptrs: [7]u64,
    part: ?DeviceBuffer,

    const Name = enum(usize) { tk_raw = 0, tk_stage = 1, iq = 2, qi = 3, hw = 4, sc = 5, bits = 6 };

    pub fn init(d: *const cuda.Driver, lane: *const cuda.qmmf.Lane, w: *const Weights, rows: usize, cap: usize) !Scratch {
        if (rows == 0 or cap == 0 or cap % ratio != 0) return error.ScratchShape;
        const score_rows: usize = @min(rows, score_tile);
        const max_groups = cap / ratio;
        const sc_stride = std.mem.alignForward(usize, max_groups, 4);
        const words = (max_groups + 31) / 32;
        const sizes = [7]usize{
            rows * dim * 2,
            (rows + ratio) * dim * 2,
            rows * heads * dim * 2,
            score_rows * heads * dim * 2,
            rows * heads * 4,
            score_rows * sc_stride * 4,
            @max(@as(usize, 16), score_rows * 2 * words * 4),
        };
        var offs: [7]usize = undefined;
        var total: usize = 0;
        for (sizes, 0..) |len, i| {
            offs[i] = total;
            total += std.mem.alignForward(usize, @max(len, 16), 256);
        }
        var arena = try DeviceBuffer.alloc(d, total);
        errdefer arena.free();
        var ptrs: [7]u64 = undefined;
        for (&ptrs, offs) |*p, o| p.* = arena.ptr + o;
        var part: usize = 0;
        for ([_]cuda.qmmf.Weight{ w.query.w, w.token_key.w }) |pw| {
            var m: usize = 1;
            while (m <= rows) : (m = if (m < 64) m + 1 else m * 2) part = @max(part, lane.partBytes(m, pw));
            if (rows >= 64) part = @max(part, lane.partBytes(@min(rows, cuda.qmmf.fused_rows - 1), pw));
            part = @max(part, lane.partBytes(rows, pw));
        }
        const part_buf: ?DeviceBuffer = if (part > 0) try DeviceBuffer.alloc(d, part) else null;
        return .{
            .rows = rows,
            .score_rows = score_rows,
            .max_groups = max_groups,
            .sc_stride = sc_stride,
            .arena = arena,
            .ptrs = ptrs,
            .part = part_buf,
        };
    }

    pub fn deinit(sc: *Scratch) void {
        sc.arena.free();
        if (sc.part) |*p| p.free();
    }

    fn at(sc: *const Scratch, comptime name: Name) u64 {
        return sc.ptrs[@backingInt(name)];
    }

    fn lanePart(sc: *const Scratch) u64 {
        return if (sc.part) |p| p.ptr else 0;
    }
};

fn cdiv(a: usize, b: usize) u32 {
    return @intCast((a + b - 1) / b);
}

fn cint(v: usize) c_int {
    return @intCast(v);
}

/// Newly completed 4-token groups for rows at `[first, first + rows)`: exact partition `[first/4 .. (first+rows)/4)`.
pub fn completedGroups(first: usize, rows: usize) struct { lo: usize, len: usize } {
    const lo = first / ratio;
    const hi = (first + rows) / ratio;
    return .{ .lo = lo, .len = hi - lo };
}

/// Selected row slice `[from, from + count)` starting at `list_from` (default `pos >= recall_from`).
pub fn selectRange(first: usize, rows: usize, list_from: ?usize, recall_from: usize) struct { from: usize, count: usize } {
    const start = if (list_from) |lf| @min(lf, rows) else if (first + rows > recall_from)
        (if (first >= recall_from) 0 else recall_from - first)
    else
        rows;
    return .{ .from = start, .count = rows - start };
}

pub fn scoreSplit(nb: usize, span_own: usize, sms: usize, occ_pre: usize) u32 {
    const ns = @max(@as(usize, 1), (span_own + 15) / 16);
    if (nb == 1) return @intCast(@max(@as(usize, 1), @min(ns, @max(@as(usize, 48), @min(@as(usize, 288), ns / 8)))));
    return @intCast(@max(@as(usize, 1), @min((4 * sms * occ_pre + nb - 1) / nb, @max(@as(usize, 1), ns / 4))));
}

pub fn ropeTable(f: Function, s: Stream, cs: u64, first: usize, npos: usize, freqs: u64, pos3: u64, ld3: usize, sec_h: usize, sec_w: usize, round_bf16: bool) !void {
    if (npos == 0) return;
    var a: Args = .{};
    a.add(cs);
    a.add(cint(first));
    a.add(cint(npos));
    a.add(freqs);
    a.add(pos3);
    inline for (.{ ld3, sec_h, sec_w, @intFromBool(round_bf16) }) |v| a.add(cint(v));
    try cuda.launch.launch(f, .{ .grid = .{ .x = cdiv(npos * rope_pairs, 256) }, .block = .{ .x = 256 } }, s, &a);
}

pub fn poolWith(f: Function, s: Stream, cluster: ?u32, smem: u32, grid_x: u32, block_x: u32, js: u64, jg: u64, g0: usize, njobs: usize, tkc: u64, ctx: usize, w: *const Weights, rt: u64, rt_stream_stride: usize, rt_group_stride: usize, gkc: u64, pt: u64, pt_stride: usize, g_base: usize) !void {
    if (njobs == 0) return;
    var a: Args = .{};
    inline for (.{ js, jg, cint(g0), cint(njobs), tkc, cint(ctx) }) |v| a.add(v);
    inline for (.{ w.values.ptr, w.gates.ptr, w.ape.ptr, w.key_norm.ptr, rt }) |v| a.add(v);
    inline for (.{ cint(rt_stream_stride), cint(rt_group_stride), w.eps, gkc }) |v| a.add(v);
    if (pt != 0) inline for (.{ pt, cint(pt_stride), cint(g_base) }) |v| a.add(v);
    try cuda.launch.launch(f, .{
        .grid = .{ .x = grid_x },
        .block = .{ .x = block_x },
        .shared = smem,
        .cluster = if (cluster) |cl| .{ .x = cl, .y = 1, .z = 1 } else null,
    }, s, &a);
}

pub fn poolEx(k: *const Kernels, s: Stream, w: *const Weights, tkc: u64, gkc: u64, rt: u64, ctx: usize, g0: usize, njobs: usize, js: u64, jg: u64, rt_stream_stride: usize, rt_group_stride: usize, pt: u64, pt_stride: usize, g_base: usize) !void {
    if (njobs == 0) return;
    const pg = pt != 0;
    if (!k.clusters) {
        return poolWith(if (pg) k.pool_1cta_paged else k.pool_1cta, s, null, pool_1cta_smem, cdiv(njobs, 16), 128, js, jg, g0, njobs, tkc, ctx, w, rt, rt_stream_stride, rt_group_stride, gkc, pt, pt_stride, g_base);
    }
    if (njobs <= small_max_jobs) {
        return poolWith(if (pg) k.pool_small_paged else k.pool_small, s, 4, 0, @intCast(4 * njobs), 128, js, jg, g0, njobs, tkc, ctx, w, rt, rt_stream_stride, rt_group_stride, gkc, pt, pt_stride, g_base);
    }
    const nblk = (njobs + 31) / 32;
    const ncl = @min(nblk, k.persist_clusters);
    return poolWith(if (pg) k.pool_persist_paged else k.pool_persist, s, 4, pool_persist_smem, @intCast(4 * ncl), 256, js, jg, g0, njobs, tkc, ctx, w, rt, rt_stream_stride, rt_group_stride, gkc, pt, pt_stride, g_base);
}

pub fn pool(k: *const Kernels, s: Stream, w: *const Weights, tkc: u64, gkc: u64, rt: u64, ctx: usize, g0: usize, njobs: usize, js: u64, jg: u64, rt_stream_stride: usize, rt_group_stride: usize) !void {
    return poolEx(k, s, w, tkc, gkc, rt, ctx, g0, njobs, js, jg, rt_stream_stride, rt_group_stride, 0, 0, 0);
}

pub fn qMrope(f: Function, s: Stream, sel: u64, n_sel: usize, iq: u64, row_pos: u64, pos3: u64, r_stride: usize, cs: u64, qi: u64) !void {
    if (n_sel == 0) return;
    if (qi == 0 or iq == 0 or cs == 0) return error.ScratchShape;
    var a: Args = .{};
    a.add(sel);
    a.add(cint(n_sel));
    a.add(iq);
    a.add(row_pos);
    a.add(pos3);
    a.add(cint(r_stride));
    a.add(cs);
    a.add(qi);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(n_sel) }, .block = .{ .x = 256 } }, s, &a);
}

pub const ScoreCall = struct {
    sel: u64 = 0,
    n_sel: usize,
    iq_or_qi: u64,
    hw: u64,
    row_stream: u64 = 0,
    row_pos: u64,
    gkc: u64,
    gpool: usize,
    sc: u64,
    gcap: usize,
    span_own: usize,
    cs: u64,
    pos3: u64 = 0,
    r_stride: usize,
    qi_scratch: u64,
    page_table: u64 = 0,
    pt_stride: usize = 0,
};

pub fn score(k: *const Kernels, s: Stream, c: ScoreCall) !void {
    if (c.n_sel == 0) return;
    const fused = c.n_sel <= 4;
    const rb: usize = if (fused) 4 else 8;
    const nb = cdiv(c.n_sel, rb);
    const split = scoreSplit(nb, c.span_own, k.sms, k.occ_pre);
    if (fused) {
        var a: Args = .{};
        a.add(c.sel);
        a.add(cint(c.n_sel));
        a.add(c.iq_or_qi);
        a.add(c.hw);
        a.add(c.row_stream);
        a.add(c.row_pos);
        a.add(c.gkc);
        a.add(cint(c.gpool));
        a.add(c.sc);
        a.add(cint(c.gcap));
        a.add(c.cs);
        a.add(c.pos3);
        a.add(cint(c.r_stride));
        if (c.page_table != 0) {
            a.add(c.page_table);
            a.add(cint(c.pt_stride));
        }
        try cuda.launch.launch(if (c.page_table != 0) k.score_decode_rope_paged else k.score_decode_rope, .{ .grid = .{ .x = split, .y = nb }, .block = .{ .x = 128 } }, s, &a);
    } else {
        try qMrope(k.q_mrope, s, c.sel, c.n_sel, c.iq_or_qi, c.row_pos, c.pos3, c.r_stride, c.cs, c.qi_scratch);
        var a: Args = .{};
        a.add(c.sel);
        a.add(cint(c.n_sel));
        a.add(c.qi_scratch);
        a.add(c.hw);
        a.add(c.row_stream);
        a.add(c.row_pos);
        a.add(c.gkc);
        a.add(cint(c.gpool));
        a.add(c.sc);
        a.add(cint(c.gcap));
        if (c.page_table != 0) {
            a.add(c.page_table);
            a.add(cint(c.pt_stride));
        }
        try cuda.launch.launch(if (c.page_table != 0) k.score_prefill_paged else k.score_prefill, .{ .grid = .{ .x = split, .y = nb }, .block = .{ .x = 256 }, .pdl = k.pdl }, s, &a);
    }
}

pub fn select(k: *const Kernels, s: Stream, sel: u64, n_sel: usize, row_pos: u64, sc: u64, gcap: usize, bits: u64, groups_out: u64, tokens_out: u64, max_own: usize) !void {
    if (n_sel == 0) return;
    if (max_own <= select_cap) {
        var a: Args = .{};
        inline for (.{ sel, row_pos, sc, cint(gcap), groups_out, tokens_out }) |v| a.add(v);
        try cuda.launch.launch(k.select_1cta, .{ .grid = .{ .x = @intCast(n_sel) }, .block = .{ .x = 1024 }, .pdl = k.pdl }, s, &a);
    } else {
        const words = (gcap + 31) / 32;
        const scan_blocks = cdiv(max_own, select_scan_span);
        if (scan_blocks > 0) {
            var sa: Args = .{};
            inline for (.{ sel, row_pos, sc, cint(gcap), bits, cint(words) }) |v| sa.add(v);
            try cuda.launch.launch(k.select_scan, .{ .grid = .{ .x = scan_blocks, .y = @intCast(n_sel) }, .block = .{ .x = 256 }, .pdl = k.pdl }, s, &sa);
        }
        var ma: Args = .{};
        inline for (.{ sel, row_pos, sc, cint(gcap), bits, cint(words), groups_out, tokens_out, @as(u64, 0) }) |v| ma.add(v);
        try cuda.launch.launch(k.select_merge, .{ .grid = .{ .x = @intCast(n_sel) }, .block = .{ .x = 1024 }, .pdl = k.pdl }, s, &ma);
    }
}

pub const Call = struct {
    x: u64, // bf16 [rows, hidden]
    q_resid: u64, // bf16 [rows, q_lora]
    rows: usize,
    first: usize,
    pos: u64, // i32 [rows]: first, first + 1, ...
    pos3: u64 = 0, // optional i32 [3, pos3_stride] M-RoPE query positions
    pos3_stride: usize = 0,
    list_from: ?usize = null,
    recall_from: usize = tokens_per_row,
    groups_out: u64 = 0, // i32 [n_sel, 512], optional
    idx_out: u64 = 0, // i32 [n_sel, 2048], optional
    page_table: u64 = 0,
    pt_stride: usize = 0,
    preprojected: bool = false, // when true, tk_raw/iq/hw are already projected for the full batch at `row_off`
    row_off: usize = 0,
};

/// Batched projections for multi-segment decode (`segs.len > 1`): one `token_key`, `query`, and `head_weights` launch.
pub fn preproject(ops: Ops, s: Stream, w: *const Weights, sc: *const Scratch, x: u64, q_resid: u64, rows: usize) !void {
    if (rows == 0) return;
    if (rows > sc.rows) return error.ScratchShape;
    const lp = sc.lanePart();
    try ops.lane.matmul(s, x, w.hidden, rows, w.token_key.w, sc.at(.tk_raw), lp);
    if (q_resid != 0) {
        try ops.lane.matmul(s, q_resid, w.q_lora, rows, w.query.w, sc.at(.iq), lp);
        try ops.dense.rowsScaled(s, x, w.hidden, w.head_weights.ptr, w.head_scale.ptr, sc.at(.hw), heads, rows, heads, w.hidden);
    }
}

fn runSelect(ops: Ops, s: Stream, w: *const Weights, group_keys: u64, gpool: usize, rope_cs: u64, sc: *const Scratch, c: Call) !void {
    const sel = selectRange(c.first, c.rows, c.list_from, c.recall_from);
    if (sel.count == 0) return;
    const n_sel = sel.count;
    const pos_sel = c.pos + sel.from * 4;
    const lp = sc.lanePart();
    const iq_base = if (c.preprojected) sc.at(.iq) + (c.row_off + sel.from) * heads * dim * 2 else blk: {
        try ops.lane.matmul(s, c.q_resid + sel.from * w.q_lora * 2, w.q_lora, n_sel, w.query.w, sc.at(.iq), lp);
        break :blk sc.at(.iq);
    };
    const hw_base = if (c.preprojected) sc.at(.hw) + (c.row_off + sel.from) * heads * 4 else blk: {
        try ops.dense.rowsScaled(s, c.x + sel.from * w.hidden * 2, w.hidden, w.head_weights.ptr, w.head_scale.ptr, sc.at(.hw), heads, n_sel, heads, w.hidden);
        break :blk sc.at(.hw);
    };
    var off: usize = 0;
    while (off < n_sel) {
        const b: usize = @min(sc.score_rows, n_sel - off);
        const max_own = (c.first + sel.from + off + b - 1) / ratio;
        const gcap = std.mem.alignForward(usize, @max(max_own, 1), 4);
        if (gcap > sc.sc_stride) return error.ScratchShape;
        const iq_b = iq_base + off * heads * dim * 2;
        const hw_b = hw_base + off * heads * 4;
        const pos_b = pos_sel + off * 4;
        const pos3_b = if (c.pos3 != 0) c.pos3 + (sel.from + off) * 4 else 0;
        const r_stride = if (c.pos3 != 0) c.pos3_stride else b;
        const grp_b = if (c.groups_out != 0) c.groups_out + off * groups * 4 else 0;
        const idx_b = if (c.idx_out != 0) c.idx_out + off * tokens_per_row * 4 else 0;
        try score(ops.k, s, .{
            .n_sel = b,
            .iq_or_qi = iq_b,
            .hw = hw_b,
            .row_pos = pos_b,
            .gkc = group_keys,
            .gpool = gpool,
            .sc = sc.at(.sc),
            .gcap = gcap,
            .span_own = max_own,
            .cs = rope_cs,
            .pos3 = pos3_b,
            .r_stride = r_stride,
            .qi_scratch = sc.at(.qi),
            .page_table = c.page_table,
            .pt_stride = c.pt_stride,
        });
        try select(ops.k, s, 0, b, pos_b, sc.at(.sc), gcap, sc.at(.bits), grp_b, idx_b, max_own);
        off += b;
    }
}

/// Incremental indexer step: token keys -> completed 4:1 group pooling -> score + top-511 + 2,048-token recall.
pub fn run(ops: Ops, s: Stream, w: *const Weights, cache: *Cache, sc: *const Scratch, c: Call) !void {
    if (c.rows == 0) return;
    if (c.row_off + c.rows > sc.rows) return error.ScratchShape;
    if (c.first + c.rows > cache.cap) return error.IndexCacheOverflow;
    const lp = sc.lanePart();
    const tk_src = if (c.preprojected) sc.at(.tk_raw) + c.row_off * dim * 2 else blk: {
        try ops.lane.matmul(s, c.x, w.hidden, c.rows, w.token_key.w, sc.at(.tk_raw), lp);
        break :blk sc.at(.tk_raw);
    };
    const tk_dst = cache.token_keys.ptr + c.first * dim * 2;
    try ops.norms.rmsParts(s, tk_src, dim, 0, w.token_key_norm.ptr, 0, tk_dst, dim, 0, null, c.rows, 1, dim, w.eps, true);
    try cache.ensureRope(ops, s, w, c.first + c.rows);
    const cg = completedGroups(c.first, c.rows);
    if (cg.len > 0) {
        try pool(ops.k, s, w, cache.token_keys.ptr, cache.group_keys.ptr, cache.rope_cs.ptr, cache.cap, cg.lo, cg.len, 0, 0, 0, ratio);
    }
    return runSelect(ops, s, w, cache.group_keys.ptr, cache.cap / ratio, cache.rope_cs.ptr, sc, c);
}

fn copyTailRing(d: *const cuda.Driver, s: Stream, ring_ptr: u64, pos0: usize, lin_ptr: u64, n: usize, to_ring: bool) !void {
    if (n == 0) return;
    const s0 = pos0 % tail_ring_rows;
    const n0 = @min(n, tail_ring_rows - s0);
    const rb = dim * 2;
    try d.check(d.api.cuMemcpyDtoDAsync_v2(if (to_ring) ring_ptr + s0 * rb else lin_ptr, if (to_ring) lin_ptr else ring_ptr + s0 * rb, n0 * rb, s.handle), "cuMemcpyDtoDAsync");
    if (n > n0) try d.check(d.api.cuMemcpyDtoDAsync_v2(if (to_ring) ring_ptr else lin_ptr + n0 * rb, if (to_ring) lin_ptr + n0 * rb else ring_ptr, (n - n0) * rb, s.handle), "cuMemcpyDtoDAsync");
}

/// Ring-backed `tail_keys` (`[32, 128]` bf16 per sequence): rewind-safe and 0-bit identical to `run`.
pub fn runTail(
    ops: Ops,
    d: *const cuda.Driver,
    s: Stream,
    w: *const Weights,
    tail_keys: u64,
    group_keys: u64,
    cap: usize,
    query_rope_cs: u64,
    pool_rope_cs: u64,
    pool_rope_delta: isize,
    sm_mrope_cs: u64,
    sm_prompt_len: usize,
    sc: *const Scratch,
    c: Call,
) !void {
    if (c.rows == 0) return;
    if (c.row_off + c.rows > sc.rows) return error.ScratchShape;
    if (c.first + c.rows > cap) return error.IndexCacheOverflow;
    const rem = c.first % ratio;
    const stage = sc.at(.tk_stage);
    try copyTailRing(d, s, tail_keys, c.first - rem, stage, rem, false);
    const lp = sc.lanePart();
    const tk_src = if (c.preprojected) sc.at(.tk_raw) + c.row_off * dim * 2 else blk: {
        try ops.lane.matmul(s, c.x, w.hidden, c.rows, w.token_key.w, sc.at(.tk_raw), lp);
        break :blk sc.at(.tk_raw);
    };
    try ops.norms.rmsParts(s, tk_src, dim, 0, w.token_key_norm.ptr, 0, stage + rem * dim * 2, dim, 0, null, c.rows, 1, dim, w.eps, true);
    const cg = completedGroups(c.first, c.rows);
    if (cg.len > 0) {
        const prompt_groups = if (sm_mrope_cs != 0) (sm_prompt_len + ratio - 1) / ratio else 0;
        const n0 = if (cg.lo < prompt_groups) @min(cg.len, prompt_groups - cg.lo) else 0;
        if (n0 > 0) {
            const rt0 = sm_mrope_cs + cg.lo * ratio * rope_pairs * 8;
            const gkc0 = if (c.page_table != 0) group_keys else group_keys + cg.lo * dim * 2;
            try poolEx(ops.k, s, w, stage, gkc0, rt0, rem + c.rows, 0, n0, 0, 0, 0, ratio, c.page_table, c.pt_stride, cg.lo);
        }
        if (cg.len > n0) {
            const g1 = cg.lo + n0;
            const base_pos: usize = @intCast(@as(isize, @intCast(g1 * ratio)) + pool_rope_delta);
            const rt1 = pool_rope_cs + base_pos * rope_pairs * 8;
            const gkc1 = if (c.page_table != 0) group_keys else group_keys + g1 * dim * 2;
            try poolEx(ops.k, s, w, stage + n0 * ratio * dim * 2, gkc1, rt1, rem + c.rows - n0 * ratio, 0, cg.len - n0, 0, 0, 0, ratio, c.page_table, c.pt_stride, g1);
        }
    }
    const save_n = @min(c.rows, tail_ring_rows);
    const skip_new = c.rows - save_n;
    try copyTailRing(d, s, tail_keys, c.first + skip_new, stage + (rem + skip_new) * dim * 2, save_n, true);
    return runSelect(ops, s, w, group_keys, cap / ratio, query_rope_cs, sc, c);
}

test "Scratch sub-tile sizes stay in usize (avoid Zig 0.17 @min narrowing)" {
    const rows: usize = 4096;
    const score_rows: usize = @min(rows, score_tile);
    const cap: usize = 524_288;
    const max_groups = cap / ratio;
    const words = (max_groups + 31) / 32;
    try std.testing.expectEqual(@as(usize, 2_097_152), score_rows * heads * dim * 2);
    try std.testing.expectEqual(@as(usize, 16_777_216), @max(@as(usize, 16), score_rows * 2 * words * 4));
}

test "completedGroups partitions any chunk and decode cut across 4-token boundaries" {
    try std.testing.expectEqual(@TypeOf(completedGroups(0, 0)){ .lo = 0, .len = 0 }, completedGroups(0, 3));
    try std.testing.expectEqual(@TypeOf(completedGroups(0, 0)){ .lo = 0, .len = 1 }, completedGroups(0, 4));
    try std.testing.expectEqual(@TypeOf(completedGroups(0, 0)){ .lo = 0, .len = 512 }, completedGroups(0, 2051));
    try std.testing.expectEqual(@TypeOf(completedGroups(0, 0)){ .lo = 512, .len = 127 }, completedGroups(2051, 506));
    try std.testing.expectEqual(@TypeOf(completedGroups(0, 0)){ .lo = 639, .len = 0 }, completedGroups(2557, 1));
    try std.testing.expectEqual(@TypeOf(completedGroups(0, 0)){ .lo = 639, .len = 0 }, completedGroups(2558, 1));
    try std.testing.expectEqual(@TypeOf(completedGroups(0, 0)){ .lo = 639, .len = 1 }, completedGroups(2559, 1));
    try std.testing.expectEqual(@TypeOf(completedGroups(0, 0)){ .lo = 0, .len = 640 }, completedGroups(0, 2560));
}

test "selectRange selects only positions at or past recall_from across cuts" {
    try std.testing.expectEqual(@TypeOf(selectRange(0, 0, null, 2048)){ .from = 2048, .count = 0 }, selectRange(0, 2048, null, 2048));
    try std.testing.expectEqual(@TypeOf(selectRange(0, 0, null, 2048)){ .from = 2048, .count = 3 }, selectRange(0, 2051, null, 2048));
    try std.testing.expectEqual(@TypeOf(selectRange(0, 0, null, 2048)){ .from = 0, .count = 506 }, selectRange(2051, 506, null, 2048));
    try std.testing.expectEqual(@TypeOf(selectRange(0, 0, null, 2048)){ .from = 0, .count = 1 }, selectRange(2559, 1, null, 2048));
    try std.testing.expectEqual(@TypeOf(selectRange(0, 0, null, 2048)){ .from = 2048, .count = 512 }, selectRange(0, 2560, 2048, 2048));
}

test "every index launcher compiles on a host without kernels" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(weights);
    std.testing.refAllDecls(Kernels);
    std.testing.refAllDecls(Cache);
    std.testing.refAllDecls(Scratch);
}
