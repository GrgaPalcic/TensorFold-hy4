//! Hy4-flash's token embeddings (4-stream broadcast) and top head (hc_head, model.norm, vocab-sharded fp32 lm_head).
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const names = @import("names.zig");
const partition = @import("partition.zig");
const memory = @import("memory.zig");
const cuda_hc = @import("cuda_hc.zig");
const cuda_weights = @import("cuda_weights.zig");
const cuda_rows = @import("cuda_rows.zig");

const Config = config.Config;
const Stream = cuda.Stream;
const DeviceBuffer = cuda.DeviceBuffer;
const Rank = cuda_weights.Rank;

pub const streams = cuda_hc.streams;

/// Shared CUDA operators launched by `Embed` and `Top`.
pub const Ops = struct {
    hc: *const cuda_hc.Hc,
    norms: *const cuda.norms.Norms,
    dense: *const cuda.dense_rows.DenseRows,
    embed: *const cuda.embed_rows.EmbedRows,
};

/// Standalone owner of `Ops` for head-only callers and tests; full models share `cuda_layer.Kernels.headOps()`.
pub const Kernels = struct {
    hc: cuda_hc.Hc,
    norms: cuda.norms.Norms,
    dense: cuda.dense_rows.DenseRows,
    embed: cuda.embed_rows.EmbedRows,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var hc = try cuda_hc.Hc.load(d);
        errdefer hc.unload();
        var norms = try cuda.norms.Norms.load(d);
        errdefer norms.unload();
        var dense = try cuda.dense_rows.DenseRows.load(d);
        errdefer dense.unload();
        const embed = try cuda.embed_rows.EmbedRows.load(d);
        return .{ .hc = hc, .norms = norms, .dense = dense, .embed = embed };
    }

    pub fn deinit(k: *Kernels) void {
        k.embed.unload();
        k.dense.unload();
        k.norms.unload();
        k.hc.unload();
        k.* = undefined;
    }

    pub fn ops(k: *const Kernels) Ops {
        return .{ .hc = &k.hc, .norms = &k.norms, .dense = &k.dense, .embed = &k.embed };
    }
};

/// Replicated `model.embed_tokens.weight` (`bf16 [vocab, hidden]`) with 1-stream lookup and 4-stream expansion.
pub const Embed = struct {
    vocab: usize,
    hidden: usize,
    w: DeviceBuffer,

    pub fn load(d: *const cuda.Driver, t: names.Top, c: Config) !Embed {
        if (c.hc_mult != streams or c.hidden % 8 != 0) return error.UnsupportedShapes;
        if (t.embed.dtype != .bf16 or t.embed.dim(0) != c.vocab or t.embed.dim(1) != c.hidden) return error.UnexpectedTensor;
        return .{ .vocab = c.vocab, .hidden = c.hidden, .w = try DeviceBuffer.fromHost(d, t.embed.bytes) };
    }

    pub fn deinit(e: *Embed) void {
        e.w.free();
        e.* = undefined;
    }

    /// Looks up `ids` (`i32 [rows]`) into `out` (`bf16 [rows, hidden]`).
    pub fn lookup(e: *const Embed, ops: Ops, s: Stream, ids: u64, out: u64, rows: usize) !void {
        return ops.embed.bf16(s, ids, e.w.ptr, out, rows, e.hidden);
    }

    /// Looks up `ids` (`i32 [rows]`) and broadcasts exact bf16 bits across all 4 streams (`bf16 [rows, 4, hidden]`).
    pub fn expand(e: *const Embed, ops: Ops, s: Stream, ids: u64, out: u64, rows: usize) !void {
        return ops.embed.bf16Copies(s, ids, e.w.ptr, out, rows, e.hidden, streams);
    }
};

/// Working rows for `Top.forward` up to `max_rows` rows.
pub const Scratch = struct {
    max_rows: usize,
    hidden_dim: usize,
    collapsed: DeviceBuffer, // bf16 [max_rows, hidden]: hc_head output
    hidden: DeviceBuffer, // bf16 [max_rows, hidden]: model.norm output when caller passes null

    pub fn init(d: *const cuda.Driver, c: Config, max_rows: usize) !Scratch {
        const bytes = @max(max_rows, 1) * c.hidden * 2;
        var collapsed = try DeviceBuffer.alloc(d, bytes);
        errdefer collapsed.free();
        const hidden = try DeviceBuffer.alloc(d, bytes);
        return .{ .max_rows = max_rows, .hidden_dim = c.hidden, .collapsed = collapsed, .hidden = hidden };
    }

    pub fn deinit(sc: *Scratch) void {
        sc.hidden.free();
        sc.collapsed.free();
        sc.* = undefined;
    }
};

/// Replicated `hc_head` + `model.norm` and the rank's vocab slice of `lm_head.weight` (`bf16 [vocab_local, hidden]`).
pub const Top = struct {
    at: Rank,
    share: partition.Share,
    hidden: usize,
    eps: f32,
    hc_eps: f32,
    hc_fn: DeviceBuffer, // fp32 [4, 4 * hidden]
    hc_base: DeviceBuffer, // fp32 [4]
    hc_scale: DeviceBuffer, // fp32 [1]
    norm: DeviceBuffer, // bf16 [hidden]
    lm_head: DeviceBuffer, // bf16 [share.len, hidden]
    rows: cuda_rows.Rows, // lm_head at 5 or more rows (a batched round), ops.dense's bits

    pub fn load(d: *const cuda.Driver, t: names.Top, c: Config, at: Rank) !Top {
        if (c.hc_mult != streams or c.hidden % 64 != 0) return error.UnsupportedShapes;
        const share = try partition.vocab(c, at.rank, at.world);
        if (t.norm.dtype != .bf16 or t.norm.dim(0) != c.hidden) return error.UnexpectedTensor;
        if (t.head.dtype != .bf16 or t.head.dim(0) != c.vocab or t.head.dim(1) != c.hidden) return error.UnexpectedTensor;
        var hc_fn = try DeviceBuffer.fromHost(d, t.hc_head.fn_.bytes);
        errdefer hc_fn.free();
        var hc_base = try DeviceBuffer.fromHost(d, t.hc_head.base.bytes);
        errdefer hc_base.free();
        var hc_scale = try DeviceBuffer.fromHost(d, t.hc_head.scale.bytes);
        errdefer hc_scale.free();
        var norm = try DeviceBuffer.fromHost(d, t.norm.bytes);
        errdefer norm.free();
        const row_bytes = c.hidden * 2;
        const lm_slice = t.head.bytes[share.lo * row_bytes ..][0 .. share.len * row_bytes];
        var lm_head = try DeviceBuffer.fromHost(d, lm_slice);
        errdefer lm_head.free();
        const rows = try cuda_rows.Rows.load(d);
        return .{
            .at = at,
            .share = share,
            .hidden = c.hidden,
            .eps = c.eps,
            .hc_eps = c.hc_eps,
            .hc_fn = hc_fn,
            .hc_base = hc_base,
            .hc_scale = hc_scale,
            .norm = norm,
            .lm_head = lm_head,
            .rows = rows,
        };
    }

    pub fn deinit(w: *Top) void {
        w.rows.unload();
        w.lm_head.free();
        w.norm.free();
        w.hc_scale.free();
        w.hc_base.free();
        w.hc_fn.free();
        w.* = undefined;
    }

    pub fn mix(w: *const Top) cuda_hc.Hc.Mix {
        return .{ .fn_ = w.hc_fn.ptr, .base = w.hc_base.ptr, .scale = w.hc_scale.ptr, .eps = w.eps, .hc_eps = w.hc_eps };
    }

    /// Collapses `streams_in` (`bf16 [rows, 4, hidden]`) via `hc_head` + `model.norm` into `hidden_out` (`bf16`).
    pub fn collapse(w: *const Top, ops: Ops, s: Stream, streams_in: u64, collapsed_tmp: u64, hidden_out: u64, rows: usize) !void {
        if (rows == 0) return;
        try ops.hc.head(s, w.mix(), streams_in, collapsed_tmp, rows, w.hidden, .production, 0);
        try ops.norms.addRmsH(s, collapsed_tmp, null, w.norm.ptr, .bf16, 0, hidden_out, null, rows, w.hidden, w.eps);
    }

    /// Projects `hidden_in` (`bf16 [rows, hidden]`) via `lm_head` into `fp32 [rows, share.len]` (`out_stride` apart).
    pub fn logitsFromHidden(w: *const Top, ops: Ops, s: Stream, hidden_in: u64, logits_out: u64, out_stride: usize, rows: usize) !void {
        if (rows == 0) return;
        if (out_stride < w.share.len) return error.RowShape;
        try w.rows.rows(ops.dense, s, hidden_in, w.hidden, w.lm_head.ptr, logits_out, out_stride, rows, w.share.len, w.hidden);
    }

    /// Runs `hc_head` + `model.norm` + `fp32` `lm_head` on `streams_in` (`bf16 [rows, 4, hidden]`).
    pub fn forward(w: *const Top, ops: Ops, s: Stream, sc: *const Scratch, streams_in: u64, hidden_out: ?u64, logits_out: u64, out_stride: usize, rows: usize) !void {
        if (rows > sc.max_rows or w.hidden != sc.hidden_dim) return error.ScratchShape;
        if (rows == 0) return;
        const h = hidden_out orelse sc.hidden.ptr;
        try w.collapse(ops, s, streams_in, sc.collapsed.ptr, h, rows);
        try w.logitsFromHidden(ops, s, h, logits_out, out_stride, rows);
    }
};

/// Both ends of the backbone on one rank (`Embed` + `Top`), matching `memory.topBytes` byte for byte.
pub const Head = struct {
    embed: Embed,
    top: Top,

    pub fn load(d: *const cuda.Driver, t: names.Top, c: Config, at: Rank) !Head {
        var embed = try Embed.load(d, t, c);
        errdefer embed.deinit();
        const top = try Top.load(d, t, c, at);
        return .{ .embed = embed, .top = top };
    }

    pub fn deinit(h: *Head) void {
        h.top.deinit();
        h.embed.deinit();
        h.* = undefined;
    }

    pub fn byteCount(c: Config, at: Rank) !usize {
        const share = try partition.vocab(c, at.rank, at.world);
        const embed_bytes = 2 * c.vocab * c.hidden;
        const head_bytes = 2 * share.len * c.hidden;
        const norm_bytes = 2 * c.hidden;
        const hc_bytes = 4 * (c.hc_mult * c.hc_mult * c.hidden + c.hc_mult + 1);
        return embed_bytes + head_bytes + norm_bytes + hc_bytes;
    }
};

test "Head.byteCount matches memory.topBytes and tiles vocabulary across worlds 1, 2 and 4" {
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, @embedFile("fixtures/config.json"), &why);
    for ([_]usize{ 1, 2, 4 }) |world| {
        var covered: usize = 0;
        for (0..world) |rank| {
            const at: Rank = .{ .rank = rank, .world = world };
            try std.testing.expectEqual(try memory.topBytes(c, rank, world), try Head.byteCount(c, at));
            const sh = try partition.vocab(c, rank, world);
            try std.testing.expectEqual(covered, sh.lo);
            covered += sh.len;
        }
        try std.testing.expectEqual(c.vocab, covered);
    }
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Kernels);
    std.testing.refAllDecls(Embed);
    std.testing.refAllDecls(Scratch);
    std.testing.refAllDecls(Top);
    std.testing.refAllDecls(Head);
}
