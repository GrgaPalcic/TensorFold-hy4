//! hy4_rows.cu: Hy4-flash's lm_head and router at 5 or more rows, each row's bits those of `tf_dense_rows_1`.
const std = @import("std");
const cuda = @import("cuda");

const launch = cuda.launch;
const Stream = cuda.Stream;

pub const threads = 256;
pub const warps = threads / 32;
pub const Control = enum { production, nan, repeat, tail, sums };
/// A launch tile: rows each weight load serves and columns each x load serves.
pub const Tile = enum { wide, narrow };
pub const tile_rows = [_]usize{ 16, 4 };
pub const tile_cols = [_]usize{ 4, 2 };

pub const Rows = struct {
    mod: cuda.Module,
    fns: [2][5]cuda.Function,
    argmax_fn: cuda.Function,
    slice_bf16_fn: cuda.Function,
    interleave2_bf16_fn: cuda.Function,
    slice_f32_fn: cuda.Function,
    comb_swiglu_bf16_fn: cuda.Function,
    kv_write_fn: cuda.Function,
    q_rope_fn: cuda.Function,
    f32_to_bf16_fn: cuda.Function = undefined,
    interleave_bf16_fn: cuda.Function = undefined,
    kv_write_ex_fn: cuda.Function = undefined,
    kv_write_paged_fn: cuda.Function = undefined,
    q_rope_mrope_fn: cuda.Function = undefined,
    splice_streams_fn: cuda.Function = undefined,
    vit_qkv_fn: cuda.Function = undefined,
    vit_merge_gather_fn: cuda.Function = undefined,
    vit_append_newline_fn: cuda.Function = undefined,

    pub fn load(d: *const cuda.Driver) !Rows {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.hy4_rows);
        errdefer mod.unload();
        var r: Rows = .{
            .mod = mod,
            .fns = undefined,
            .argmax_fn = undefined,
            .slice_bf16_fn = undefined,
            .interleave2_bf16_fn = undefined,
            .slice_f32_fn = undefined,
            .comb_swiglu_bf16_fn = undefined,
            .kv_write_fn = undefined,
            .q_rope_fn = undefined,
            .f32_to_bf16_fn = undefined,
            .interleave_bf16_fn = undefined,
        };
        const ends = [_][]const u8{ "", "_bad_nan", "_bad_repeat", "_bad_tail", "_bad_sums" };
        inline for ([_][]const u8{ "wide", "narrow" }, 0..) |t, i| inline for (ends, 0..) |e, j| {
            r.fns[i][j] = try mod.function(std.fmt.comptimePrint("hy4_rows_{s}{s}", .{ t, e }));
        };
        r.argmax_fn = try mod.function("hy4_rows_argmax");
        r.slice_bf16_fn = try mod.function("hy4_rows_slice_bf16");
        r.interleave2_bf16_fn = try mod.function("hy4_rows_interleave2_bf16");
        r.slice_f32_fn = try mod.function("hy4_rows_slice_f32");
        r.comb_swiglu_bf16_fn = try mod.function("hy4_rows_comb_swiglu_bf16");
        r.kv_write_fn = try mod.function("hy4_rows_kv_write");
        r.q_rope_fn = try mod.function("hy4_rows_q_rope");
        r.f32_to_bf16_fn = try mod.function("hy4_rows_f32_to_bf16");
        r.interleave_bf16_fn = try mod.function("hy4_rows_interleave_bf16");
        r.kv_write_ex_fn = try mod.function("hy4_rows_kv_write_ex");
        r.kv_write_paged_fn = try mod.function("hy4_rows_kv_write_paged");
        r.q_rope_mrope_fn = try mod.function("hy4_rows_q_rope_mrope");
        r.splice_streams_fn = try mod.function("hy4_rows_splice_streams");
        r.vit_qkv_fn = try mod.function("hy4_rows_vit_qkv");
        r.vit_merge_gather_fn = try mod.function("hy4_rows_vit_merge_gather");
        r.vit_append_newline_fn = try mod.function("hy4_rows_vit_append_newline");
        return r;
    }

    pub fn unload(r: *Rows) void {
        r.mod.unload();
    }

    /// Extracts `[rows, dst_cols]` bf16 from strided `[rows, src_stride]` bf16 (both multiples of 8).
    pub fn sliceBf16(r: *const Rows, s: Stream, src: u64, src_stride: usize, dst: u64, dst_cols: usize, rows_n: usize) !void {
        if (rows_n == 0) return;
        const total = rows_n * (dst_cols / 8);
        var a: launch.Args = .{};
        a.add(src);
        a.add(@as(i32, @intCast(src_stride / 8)));
        a.add(dst);
        a.add(@as(i32, @intCast(dst_cols / 8)));
        a.add(@as(i32, @intCast(rows_n)));
        try launch.launch(r.slice_bf16_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Interleaves `[2, rows, half_cols]` bf16 into `[rows, 2 * half_cols]` bf16 (`half_cols` multiple of 8).
    pub fn interleave2Bf16(r: *const Rows, s: Stream, src: u64, dst: u64, half_cols: usize, rows_n: usize) !void {
        if (rows_n == 0) return;
        const total = rows_n * 2 * (half_cols / 8);
        var a: launch.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(@as(i32, @intCast(half_cols / 8)));
        a.add(@as(i32, @intCast(rows_n)));
        try launch.launch(r.interleave2_bf16_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Converts `[total]` fp32 logits into bf16 (RNE), setting `nonfinite` (if non-zero) on NaN/Inf.
    pub fn f32ToBf16(r: *const Rows, s: Stream, src: u64, dst: u64, total: usize, nonfinite: u64) !void {
        if (total == 0) return;
        const n4 = (total + 3) / 4;
        var a: launch.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(@as(i32, @intCast(total)));
        a.add(nonfinite);
        try launch.launch(r.f32_to_bf16_fn, .{ .grid = .{ .x = @intCast((n4 + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Interleaves `[world, rows, local_cols]` bf16 into `[rows, world * local_cols]` bf16 and flags Inf/NaN.
    pub fn interleaveBf16(r: *const Rows, s: Stream, src: u64, dst: u64, local_cols: usize, rows_n: usize, world: usize, nonfinite: u64) !void {
        if (rows_n == 0 or world == 0) return;
        if (local_cols % 8 != 0) return error.BadStepShape;
        const total = rows_n * world * (local_cols / 8);
        var a: launch.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(@as(i32, @intCast(local_cols / 8)));
        a.add(@as(i32, @intCast(rows_n)));
        a.add(@as(i32, @intCast(world)));
        a.add(nonfinite);
        try launch.launch(r.interleave_bf16_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Extracts `[rows, dst_cols]` fp32 at `col_off` from strided `[rows, src_stride]` fp32 (multiples of 4).
    pub fn sliceF32(r: *const Rows, s: Stream, src: u64, src_stride: usize, col_off: usize, dst: u64, dst_cols: usize, rows_n: usize) !void {
        if (rows_n == 0) return;
        const total = rows_n * (dst_cols / 4);
        var a: launch.Args = .{};
        a.add(src);
        a.add(@as(i32, @intCast(src_stride / 4)));
        a.add(@as(i32, @intCast(col_off / 4)));
        a.add(dst);
        a.add(@as(i32, @intCast(dst_cols / 4)));
        a.add(@as(i32, @intCast(rows_n)));
        try launch.launch(r.slice_f32_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Fuses world=2 `comb` fp32 rank-order sum -> bf16 `[0..half)` and shared SwiGLU `[half..half+sw)` into `out`.
    pub fn combSwigluBf16(
        r: *const Rows,
        s: Stream,
        comb_f32: u64,
        comb_stride: usize,
        my_col: usize,
        peer_f32: u64,
        gu: u64,
        gu_stride: usize,
        gate_off: usize,
        up_off: usize,
        out: u64,
        half: usize,
        sw: usize,
        rows_n: usize,
        rank: u32,
    ) !void {
        if (rows_n == 0) return;
        const total = rows_n * (half + sw);
        var a: launch.Args = .{};
        a.add(comb_f32);
        a.add(@as(i32, @intCast(comb_stride)));
        a.add(@as(i32, @intCast(my_col)));
        a.add(peer_f32);
        a.add(gu);
        a.add(@as(i32, @intCast(gu_stride)));
        a.add(@as(i32, @intCast(gate_off)));
        a.add(@as(i32, @intCast(up_off)));
        a.add(out);
        a.add(@as(i32, @intCast(half)));
        a.add(@as(i32, @intCast(sw)));
        a.add(@as(i32, @intCast(rows_n)));
        a.add(@as(i32, @intCast(rank)));
        try launch.launch(r.comb_swiglu_bf16_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// GPU greedy argmax over `[m, vocab_local]` fp32 logits; writes `[m][2]` u32 `{float_bits(max), vocab_lo + id}`.
    pub fn argmax(r: *const Rows, s: Stream, logits: u64, vocab_local: usize, vocab_lo: usize, m: usize, out: u64) !void {
        if (m == 0) return;
        var a: launch.Args = .{};
        a.add(logits);
        a.add(@as(i32, @intCast(vocab_local)));
        a.add(@as(i32, @intCast(vocab_lo)));
        a.add(@as(i32, @intCast(m)));
        a.add(out);
        try launch.launch(r.argmax_fn, .{ .grid = .{ .x = @intCast(m), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }

    /// Fused batched `kv_a_norm` (512) + `k_rope` (64) into single (`caches == 0`) or per-row (`caches != 0`) KV cache.
    pub fn kvWrite(
        r: *const Rows,
        s: Stream,
        qa: u64,
        qa_stride: usize,
        q_lora: usize,
        kv_norm_w: u64,
        freqs: u64,
        pos: u64,
        caches: u64,
        single_cache: u64,
        ring: usize,
        rows_n: usize,
        eps: f32,
    ) !void {
        if (rows_n == 0) return;
        var a: launch.Args = .{};
        a.add(qa);
        a.add(@as(i32, @intCast(qa_stride)));
        a.add(@as(i32, @intCast(q_lora)));
        a.add(kv_norm_w);
        a.add(freqs);
        a.add(pos);
        a.add(caches);
        a.add(single_cache);
        a.add(@as(i32, @intCast(ring)));
        a.add(eps);
        try launch.launch(r.kv_write_fn, .{ .grid = .{ .x = @intCast(rows_n), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Batched per-row GPT-J RoPE (64 cols) reading `pos[r]` directly for all `rows_n` x `heads` in one launch.
    pub fn qRope(
        r: *const Rows,
        s: Stream,
        x: u64,
        xs: usize,
        xh: usize,
        out: u64,
        os: usize,
        oh: usize,
        pos: u64,
        freqs: u64,
        rows_n: usize,
        heads: usize,
    ) !void {
        if (rows_n == 0 or heads == 0) return;
        var a: launch.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(xs)));
        a.add(@as(i64, @intCast(xh)));
        a.add(out);
        a.add(@as(i64, @intCast(os)));
        a.add(@as(i64, @intCast(oh)));
        a.add(pos);
        a.add(freqs);
        try launch.launch(r.q_rope_fn, .{ .grid = .{ .x = @intCast(rows_n), .y = @intCast(heads), .z = 1 }, .block = .{ .x = 32 } }, s, &a);
    }

    /// Extended `kvWrite` supporting `--kv-cache fp8` (`kv8`) and 3D M-RoPE (`rot_pos` / `pos3`).
    pub fn kvWriteEx(
        r: *const Rows,
        s: Stream,
        qa: u64,
        qa_stride: usize,
        q_lora: usize,
        kv_norm_w: u64,
        freqs: u64,
        slot_pos: u64,
        rot_pos: u64,
        pos3: u64,
        ld3: usize,
        caches: u64,
        single_cache: u64,
        ring: usize,
        rows_n: usize,
        eps: f32,
        kv8: bool,
    ) !void {
        if (rows_n == 0) return;
        if (!kv8 and rot_pos == 0 and pos3 == 0) {
            return r.kvWrite(s, qa, qa_stride, q_lora, kv_norm_w, freqs, slot_pos, caches, single_cache, ring, rows_n, eps);
        }
        var a: launch.Args = .{};
        a.add(qa);
        a.add(@as(i32, @intCast(qa_stride)));
        a.add(@as(i32, @intCast(q_lora)));
        a.add(kv_norm_w);
        a.add(freqs);
        a.add(slot_pos);
        a.add(rot_pos);
        a.add(pos3);
        a.add(@as(i32, @intCast(ld3)));
        a.add(caches);
        a.add(single_cache);
        a.add(@as(i32, @intCast(ring)));
        a.add(eps);
        a.add(@as(i32, @intFromBool(kv8)));
        try launch.launch(r.kv_write_ex_fn, .{ .grid = .{ .x = @intCast(rows_n), .y = 1, .z = 1 }, .block = .{ .x = 512 } }, s, &a);
    }

    /// Paged `kvWrite` writing indexed MLA rows into `256`-token pool pages via `page_table`.
    pub fn kvWritePaged(
        r: *const Rows,
        s: Stream,
        qa: u64,
        qa_stride: usize,
        q_lora: usize,
        kv_norm_w: u64,
        freqs: u64,
        slot_pos: u64,
        rot_pos: u64,
        pos3: u64,
        ld3: usize,
        page_table: u64,
        row_stream: u64,
        pt_stride: usize,
        arena_base: u64,
        rows_n: usize,
        eps: f32,
        kv8: bool,
    ) !void {
        if (rows_n == 0) return;
        var a: launch.Args = .{};
        a.add(qa);
        a.add(@as(i32, @intCast(qa_stride)));
        a.add(@as(i32, @intCast(q_lora)));
        a.add(kv_norm_w);
        a.add(freqs);
        a.add(slot_pos);
        a.add(rot_pos);
        a.add(pos3);
        a.add(@as(i32, @intCast(ld3)));
        a.add(page_table);
        a.add(row_stream);
        a.add(@as(i32, @intCast(pt_stride)));
        a.add(arena_base);
        a.add(eps);
        a.add(@as(i32, @intFromBool(kv8)));
        try launch.launch(r.kv_write_paged_fn, .{ .grid = .{ .x = @intCast(rows_n), .y = 1, .z = 1 }, .block = .{ .x = 512 } }, s, &a);
    }

    /// Batched per-row query RoPE supporting 3D interleaved M-RoPE (`pos3` [3, ld3] when non-zero).
    pub fn qRopeMrope(
        r: *const Rows,
        s: Stream,
        x: u64,
        xs: usize,
        xh: usize,
        out: u64,
        os: usize,
        oh: usize,
        pos: u64,
        pos3: u64,
        ld3: usize,
        freqs: u64,
        rows_n: usize,
        heads: usize,
    ) !void {
        if (rows_n == 0 or heads == 0) return;
        if (pos3 == 0) return r.qRope(s, x, xs, xh, out, os, oh, pos, freqs, rows_n, heads);
        var a: launch.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(xs)));
        a.add(@as(i64, @intCast(xh)));
        a.add(out);
        a.add(@as(i64, @intCast(os)));
        a.add(@as(i64, @intCast(oh)));
        a.add(pos);
        a.add(pos3);
        a.add(@as(i32, @intCast(ld3)));
        a.add(freqs);
        try launch.launch(r.q_rope_mrope_fn, .{ .grid = .{ .x = @intCast(rows_n), .y = @intCast(heads), .z = 1 }, .block = .{ .x = 32 } }, s, &a);
    }

    /// Splices `feat[src_idx[r]]` into all 4 iHC streams `streams[r, 0..3]` (`hidden` multiple of 8).
    pub fn spliceStreams(r: *const Rows, s: Stream, feat: u64, src_idx: u64, streams: u64, hidden: usize, rows_n: usize) !void {
        if (rows_n == 0 or hidden == 0) return;
        const hu4 = hidden / 8;
        var a: launch.Args = .{};
        a.add(feat);
        a.add(src_idx);
        a.add(streams);
        a.add(@as(i32, @intCast(hu4)));
        a.add(@as(i32, @intCast(rows_n)));
        try launch.launch(r.splice_streams_fn, .{ .grid = .{ .x = @intCast(rows_n), .y = @intCast((hu4 + 255) / 256), .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn vitQkv(r: *const Rows, s: Stream, acc: u64, bias: u64, q: u64, k: u64, v: u64, rows_n: usize, heads: usize, hd: usize) !void {
        if (rows_n == 0) return;
        var a: launch.Args = .{};
        inline for (.{ acc, bias, q, k, v }) |p| a.add(p);
        inline for (.{ rows_n, heads, hd }) |x| a.add(@as(i32, @intCast(x)));
        try launch.launch(r.vit_qkv_fn, .{ .grid = .{ .x = @intCast(rows_n), .y = @intCast(heads), .z = 1 }, .block = .{ .x = 128 } }, s, &a);
    }

    pub fn vitMergeGather(r: *const Rows, s: Stream, src: u64, dst: u64, gh: usize, gw: usize, d: usize) !void {
        const total = (gh / 2) * (gw / 2) * 4 * (d / 8);
        if (total == 0) return;
        var a: launch.Args = .{};
        a.add(src);
        a.add(dst);
        inline for (.{ gh, gw, d / 8 }) |x| a.add(@as(i32, @intCast(x)));
        try launch.launch(r.vit_merge_gather_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn vitAppendNewline(r: *const Rows, s: Stream, merged: u64, newline: u64, out: u64, mh: usize, mw: usize, out_d: usize) !void {
        const total = mh * (mw + 1) * (out_d / 8);
        if (total == 0) return;
        var a: launch.Args = .{};
        inline for (.{ merged, newline, out }) |p| a.add(p);
        inline for (.{ mh, mw, out_d / 8 }) |x| a.add(@as(i32, @intCast(x)));
        try launch.launch(r.vit_append_newline_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// The tile `m` rows of `n` outputs take: wide while it fills twice 48 SMs' 2 CTAs, else narrow.
    pub fn pick(m: usize, n: usize) Tile {
        const ctas = gridX(.wide, n) * ((m + tile_rows[0] - 1) / tile_rows[0]);
        return if (ctas >= 192) .wide else .narrow;
    }

    fn gridX(t: Tile, n: usize) usize {
        const per = warps * tile_cols[@backingInt(t)];
        return (n + per - 1) / per;
    }

    /// out [m, n] fp32 = x [m, k] bf16 times w [n, k] bf16 transposed, as `DenseRows.rows`; up to 4 rows go there.
    pub fn rows(r: *const Rows, dense: *const cuda.dense_rows.DenseRows, s: Stream, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, m: usize, n: usize, k: usize) !void {
        if (m <= 4) return dense.rows(s, x, x_stride, w, out, out_stride, m, n, k);
        return r.tiled(pick(m, n), .production, 0, s, x, x_stride, w, out, out_stride, m, n, k);
    }

    /// One launch at tile `t` (contract tests name the tile and control).
    pub fn tiled(r: *const Rows, t: Tile, ctl: Control, repeat: i32, s: Stream, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, m: usize, n: usize, k: usize) !void {
        if (k % 8 != 0 or x % 16 != 0 or w % 16 != 0 or x_stride % 8 != 0) return error.Misaligned;
        if (m == 0 or n == 0) return;
        var a: launch.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(x_stride)));
        a.add(w);
        a.add(out);
        a.add(@as(i64, @intCast(out_stride)));
        a.add(@as(i32, @intCast(m)));
        a.add(@as(i32, @intCast(n)));
        a.add(@as(i32, @intCast(k)));
        a.add(repeat);
        const rt = tile_rows[@backingInt(t)];
        const grid: launch.Dim3 = .{ .x = @intCast(gridX(t, n)), .y = @intCast((m + rt - 1) / rt), .z = 1 };
        try launch.launch(r.fns[@backingInt(t)][@backingInt(ctl)], .{ .grid = grid, .block = .{ .x = threads } }, s, &a);
    }
};

test "pick: lm_head's vocab slice and prompt routers go wide, a decode round's router narrow" {
    try std.testing.expectEqual(Tile.wide, Rows.pick(16, 60416));
    try std.testing.expectEqual(Tile.wide, Rows.pick(5, 30208));
    try std.testing.expectEqual(Tile.narrow, Rows.pick(16, 512));
    try std.testing.expectEqual(Tile.narrow, Rows.pick(32, 512));
    try std.testing.expectEqual(Tile.wide, Rows.pick(2048, 512));
    std.testing.refAllDecls(Rows);
}
