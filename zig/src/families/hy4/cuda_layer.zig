//! Hy4-flash's DecoderLayer on CUDA (worlds 1, 2, 4): iHC, norms, MLA, MLP/LatentMoE, and Engram/Indexer/TP seams.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const names = @import("names.zig");
const partition = @import("partition.zig");
const cuda_hc = @import("cuda_hc.zig");
const cuda_ihc = @import("cuda_ihc.zig");
const cuda_prof = @import("cuda_prof.zig");
const cuda_weights = @import("cuda_weights.zig");
const cuda_mla = @import("cuda_mla.zig");
const cuda_mlp = @import("cuda_mlp.zig");
const cuda_head = @import("cuda_head.zig");
const cuda_cache = @import("cuda_cache.zig");

const Config = config.Config;
const Stream = cuda.Stream;
const DeviceBuffer = cuda.DeviceBuffer;
const Rank = cuda_weights.Rank;

/// Every CUDA operator launched by `cuda_layer` and `cuda_head`, loaded once per device context.
pub const Kernels = struct {
    hc: cuda_hc.Hc,
    ihc: cuda_ihc.Ihc,
    mla: cuda_mla.Kernels,
    rope: cuda.mla.Mla,
    mlp: cuda_mlp.Kernels,
    embed: cuda.embed_rows.EmbedRows,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        const d = ctx.d;
        var hc = try cuda_hc.Hc.load(d);
        errdefer hc.unload();
        var ihc = try cuda_ihc.Ihc.load(d);
        errdefer ihc.unload();
        var mla = try cuda_mla.Kernels.load(d);
        errdefer mla.unload();
        var rope = try cuda.mla.Mla.load(d);
        errdefer rope.unload();
        var mlp = try cuda_mlp.Kernels.load(ctx);
        errdefer mlp.deinit();
        const embed = try cuda.embed_rows.EmbedRows.load(d);
        return .{ .hc = hc, .ihc = ihc, .mla = mla, .rope = rope, .mlp = mlp, .embed = embed };
    }

    pub fn deinit(k: *Kernels) void {
        k.embed.unload();
        k.mlp.deinit();
        k.rope.unload();
        k.mla.unload();
        k.ihc.unload();
        k.hc.unload();
        k.* = undefined;
    }

    pub fn mlaOps(k: *const Kernels) cuda_mla.Ops {
        return .{ .k = &k.mla, .lane = &k.mlp.lane, .norms = &k.mlp.norms, .mla = &k.rope };
    }

    pub fn mlaOpsProf(k: *const Kernels, prof: ?*cuda_prof.Prof) cuda_mla.Ops {
        return .{ .k = &k.mla, .lane = &k.mlp.lane, .norms = &k.mlp.norms, .mla = &k.rope, .prof = prof, .rows_op = &k.mlp.rows };
    }

    pub fn headOps(k: *const Kernels) cuda_head.Ops {
        return .{ .hc = &k.hc, .norms = &k.mlp.norms, .dense = &k.mlp.dense, .embed = &k.embed };
    }
};

/// All-reduce seam summing `count` fp32 values across `world` ranks from `send` into `recv` in rank order.
pub const AllReduce = struct {
    ctx: *anyopaque,
    rank: u32 = 0,
    sum: *const fn (ctx: *anyopaque, s: Stream, send: u64, recv: u64, count: usize) anyerror!void,
    gather_bf16: ?*const fn (ctx: *anyopaque, s: Stream, send: u64, recv: u64, count_bf16: usize) anyerror!void = null,
    exchange_f32: ?*const fn (ctx: *anyopaque, s: Stream, send: u64, recv: u64, count_f32: usize) anyerror!void = null,
};

/// Arguments passed to the optional Engram hook on layers `1, 11, 21, 31` before `attn_hc`.
pub const EngramCall = struct {
    layer_idx: usize,
    slot: usize, // 0..3
    s: Stream,
    streams_in: u64, // bf16 [rows, 4, hidden]
    streams_out: u64, // bf16 [rows, 4, hidden]: streams_in + engram_residual
    rows: usize,
    first: usize,
    pos: u64, // i32 [rows]
    all_reduce: ?AllReduce = null,
};

pub const EngramHook = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, call: EngramCall) anyerror!void,
};

/// Recalled token indices returned by `IndexerHook` on indexed layers when `first + rows > recall_limit` (`2048`).
pub const Recall = struct {
    list_from: usize, // row offset in [0, rows) where sparse recall begins
    idx: u64, // device i32 [rows - list_from, nidx]
    nidx: usize, // recalled tokens per row (2,048)
};

/// Arguments passed to the optional Compressed Indexer hook on indexed layers (`0, 1, 5, 9, ..., 41`).
pub const IndexerCall = struct {
    layer_idx: usize,
    seg: usize = 0, // the segment of `Call.segs` these rows belong to
    s: Stream,
    attn_in: u64, // bf16 [rows, hidden]
    q_resid: u64, // bf16 [rows, q_lora], computed when first + rows > recall_limit (0 otherwise)
    rows: usize,
    first: usize,
    pos: u64, // device i32 [rows]
    index_keys: u64, // device bf16 [capacity, 128]
    capacity: usize,
    idx_out: u64, // preallocated device scratch i32 [rows, recall_limit]
    recall_limit: usize, // c.recall() == 2,048
    page_table: u64 = 0,
    pt_stride: usize = 0,
    row0: usize = 0, // batch row offset when `preprojected` is true
    preprojected: bool = false,
};

pub const IndexerHook = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, call: IndexerCall) anyerror!?Recall,
    prep: ?*const fn (ctx: *anyopaque, layer_idx: usize, s: Stream, attn_in: u64, q_resid: u64, rows: usize) anyerror!void = null,
};

fn mlaPartBytes(lane: *const cuda.qmmf.Lane, c: Config, heads: usize, max_rows: usize) usize {
    const shapes = [5][2]usize{
        .{ c.q_lora + cuda_mla.row, c.hidden },
        .{ (c.q_lora + cuda_mla.row) / 2, c.hidden },
        .{ heads * cuda_mla.weights.q_head, c.q_lora },
        .{ heads * cuda_mla.weights.v_width, c.hidden },
        .{ c.hidden, heads * cuda_mla.weights.v_width },
    };
    var most: usize = 0;
    for (shapes, 0..) |nk, idx| {
        const bf = cuda.modelopt.bf16Weight(0, nk[0], nk[1]);
        const f8 = cuda.fp8.weight(0, 0, nk[0], nk[1]);
        var m: usize = 1;
        while (m <= max_rows) : (m = if (m < 64) m + 1 else m * 2) {
            most = @max(most, lane.partBytes(m, bf));
            if (idx >= 2) most = @max(most, lane.partBytes(m, f8));
        }
        most = @max(most, lane.partBytes(max_rows, bf));
        if (idx >= 2) most = @max(most, lane.partBytes(max_rows, f8));
    }
    return most;
}

fn initMlaScratch(d: *const cuda.Driver, lane: *const cuda.qmmf.Lane, c: Config, heads: usize, rows: usize) !cuda_mla.Scratch {
    const q_lora = c.q_lora;
    const sizes = [9]usize{
        rows * (q_lora + cuda_mla.row) * 2,
        rows * q_lora * 2,
        16,
        rows * heads * cuda_mla.weights.q_head * 2,
        rows * heads * cuda_mla.rope * 2,
        rows * heads * cuda_mla.row * 4, // ql: c1 QA hi and lo planes (bf16 [rows][heads][576] each)
        rows * heads * cuda_mla.latent * 4,
        rows * heads * cuda_mla.weights.v_width * 2,
        rows * heads * cuda_mla.weights.v_width * 2,
    };
    var sc: cuda_mla.Scratch = .{ .rows = rows, .heads = heads, .bufs = undefined, .part = null };
    var n: usize = 0;
    errdefer for (sc.bufs[0..n]) |*b| b.free();
    for (&sc.bufs, sizes) |*b, len| {
        b.* = try DeviceBuffer.alloc(d, @max(len, 16));
        n += 1;
    }
    const part = mlaPartBytes(lane, c, heads, rows);
    if (part > 0) sc.part = try DeviceBuffer.alloc(d, part);
    return sc;
}

/// Reusable scratch across every DecoderLayer (indexed or sliding, dense or MoE) for `rows <= max_rows` at `world`.
pub const Scratch = struct {
    world: usize,
    max_rows: usize,
    hidden: usize,
    mla_indexed: cuda_mla.Scratch,
    mla_sliding: cuda_mla.Scratch,
    mlp: cuda_mlp.Scratch,
    engram_buf: DeviceBuffer, // bf16 [max_rows, 4, hidden]
    mixed: DeviceBuffer, // bf16 [max_rows, hidden]
    post: DeviceBuffer, // fp32 [max_rows, 4]
    normed: DeviceBuffer, // bf16 [max_rows, hidden]
    branch: DeviceBuffer, // bf16 [max_rows, hidden]
    mid: DeviceBuffer, // bf16 [max_rows, 4, hidden]
    partial: DeviceBuffer, // fp32 [world, max_rows, hidden]
    reduced: DeviceBuffer, // fp32 [max_rows, hidden]
    idx: DeviceBuffer, // i32 [max_rows, recall]

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lane: *const cuda.qmmf.Lane, c: Config, world: usize, max_rows: usize) !Scratch {
        if (world != 1 and world != 2 and world != 4) return error.UnsupportedWorld;
        const rows = @max(max_rows, 1);
        const hd = c.hidden;
        var mla_idx = try initMlaScratch(d, lane, c, c.indexed.count / world, rows);
        errdefer mla_idx.deinit();
        var mla_sld = try initMlaScratch(d, lane, c, c.sliding.count / world, rows);
        errdefer mla_sld.deinit();
        var mlp = try cuda_mlp.Scratch.init(gpa, d, lane, c, rows);
        errdefer mlp.deinit();
        var engram_buf = try DeviceBuffer.alloc(d, rows * cuda_hc.streams * hd * 2);
        errdefer engram_buf.free();
        var mixed = try DeviceBuffer.alloc(d, rows * hd * 2);
        errdefer mixed.free();
        var post = try DeviceBuffer.alloc(d, rows * cuda_hc.streams * 4);
        errdefer post.free();
        var normed = try DeviceBuffer.alloc(d, rows * hd * 2);
        errdefer normed.free();
        var branch = try DeviceBuffer.alloc(d, rows * hd * 2);
        errdefer branch.free();
        var mid = try DeviceBuffer.alloc(d, rows * cuda_hc.streams * hd * 2);
        errdefer mid.free();
        var partial = try DeviceBuffer.alloc(d, world * rows * hd * 4);
        errdefer partial.free();
        var reduced = try DeviceBuffer.alloc(d, rows * hd * 4);
        errdefer reduced.free();
        const idx = try DeviceBuffer.alloc(d, rows * c.recall() * 4);
        return .{
            .world = world,
            .max_rows = max_rows,
            .hidden = hd,
            .mla_indexed = mla_idx,
            .mla_sliding = mla_sld,
            .mlp = mlp,
            .engram_buf = engram_buf,
            .mixed = mixed,
            .post = post,
            .normed = normed,
            .branch = branch,
            .mid = mid,
            .partial = partial,
            .reduced = reduced,
            .idx = idx,
        };
    }

    pub fn deinit(sc: *Scratch) void {
        for ([_]*DeviceBuffer{ &sc.idx, &sc.reduced, &sc.partial, &sc.mid, &sc.branch, &sc.normed, &sc.post, &sc.mixed, &sc.engram_buf }) |b| b.free();
        sc.mlp.deinit();
        sc.mla_sliding.deinit();
        sc.mla_indexed.deinit();
        sc.* = undefined;
    }

    pub fn mlaFor(sc: *const Scratch, is_sliding: bool) *const cuda_mla.Scratch {
        return if (is_sliding) &sc.mla_sliding else &sc.mla_indexed;
    }
};

/// One DecoderLayer's rank-sliced weights on `at` (`world` 1, 2 or 4).
pub const Layer = struct {
    layer_idx: usize,
    at: Rank,
    is_sliding: bool,
    has_engram: bool,
    engram_slot: ?usize,
    store: cuda_weights.Weights,
    hc_attn: cuda_hc.Hc.Mix,
    input_norm: u64, // bf16 [hidden]
    mla: cuda_mla.weights.Weights,
    hc_mlp: cuda_hc.Hc.Mix,
    mlp: cuda_weights.Mlp,

    pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, l: names.Layer, c: Config, layer_idx: usize, at: Rank) !Layer {
        return loadEx(gpa, d, l, c, layer_idx, at, false);
    }

    pub fn loadEx(gpa: std.mem.Allocator, d: *const cuda.Driver, l: names.Layer, c: Config, layer_idx: usize, at: Rank, shard_replicated: bool) !Layer {
        if (layer_idx >= c.layers) return error.BadLayer;
        var store = cuda_weights.Weights.init(gpa, d);
        errdefer store.deinit();
        const hc_attn: cuda_hc.Hc.Mix = .{
            .fn_ = try store.upload(l.hc_attn.fn_.bytes),
            .base = try store.upload(l.hc_attn.base.bytes),
            .scale = try store.upload(l.hc_attn.scale.bytes),
            .eps = c.eps,
            .hc_eps = c.hc_eps,
            .magnitude = c.hc_magnitude,
        };
        const input_norm = try store.upload(l.input_norm.bytes);
        const hc_mlp: cuda_hc.Hc.Mix = .{
            .fn_ = try store.upload(l.hc_mlp.fn_.bytes),
            .base = try store.upload(l.hc_mlp.base.bytes),
            .scale = try store.upload(l.hc_mlp.scale.bytes),
            .eps = c.eps,
            .hc_eps = c.hc_eps,
            .magnitude = c.hc_magnitude,
        };
        const mlp = try store.mlpEx(l, c, at, shard_replicated);
        const mla = try cuda_mla.weights.Weights.loadEx(d, gpa, l.attn, c, layer_idx, at.rank, at.world, shard_replicated);
        const slot = c.engramSlot(layer_idx);
        return .{
            .layer_idx = layer_idx,
            .at = at,
            .is_sliding = c.kind[layer_idx] == .sliding,
            .has_engram = slot != null,
            .engram_slot = slot,
            .store = store,
            .hc_attn = hc_attn,
            .input_norm = input_norm,
            .mla = mla,
            .hc_mlp = hc_mlp,
            .mlp = mlp,
        };
    }

    pub fn deinit(self: *Layer) void {
        self.mla.deinit();
        self.store.deinit();
        self.* = undefined;
    }

    /// Runs one rank's DecoderLayer (`streams_in` may equal `streams_out`); uses `call.all_reduce` when `world > 1`.
    pub fn forward(self: *const Layer, k: *const Kernels, s: Stream, c: Config, sc: *const Scratch, call: Call) !void {
        const attn_streams = try self.forwardPreAttn(k, s, c, sc, call);
        if (call.rows == 0) return;
        try self.forwardPostAttn(k, s, c, sc, attn_streams, call);
    }

    /// Runs the Engram hook (if active on this layer) and returns the stream buffer (`engram_dst` or `streams_in`).
    pub fn runEngramOnly(self: *const Layer, sc: *const Scratch, s: Stream, call: Call) !u64 {
        return runEngram(self, sc, s, call);
    }

    /// Position/cache-dependent half of a layer: Engram + optional entry iHC + `qkvPrep` + indexer + `attendCore`.
    pub fn forwardPreAttn(self: *const Layer, k: *const Kernels, s: Stream, c: Config, sc: *const Scratch, call: Call) !u64 {
        if (self.at.world != sc.world or c.hidden != sc.hidden or call.rows > sc.max_rows) return error.ScratchShape;
        if (self.at.world > 1 and call.all_reduce == null) return error.MissingAllReduce;
        var one: [1]Seg = undefined;
        const segs = try checkSegs(c, self.layer_idx, call, &one);
        if (call.rows == 0) return call.streams_in;

        const d = c.hidden;
        const rows = call.rows;
        const use_ihc = d == cuda_ihc.hidden and (call.streams_in == call.streams_out or call.engram_dst != 0);
        const attn_streams = if (call.skip_entry) call.streams_in else try runEngram(self, sc, s, call);
        if (!call.skip_entry) {
            if (use_ihc) {
                try k.ihc.run(s, .{
                    .mode = .entry,
                    .rows = rows,
                    .x = attn_streams,
                    .mix = self.hc_attn,
                    .next_nw = self.input_norm,
                    .a = sc.normed.ptr,
                    .post_next = sc.post.ptr,
                    .eps = c.eps,
                    .hc_eps = c.hc_eps,
                    .mag = c.hc_magnitude,
                });
            } else {
                try k.hc.pre(s, self.hc_attn, attn_streams, sc.mixed.ptr, sc.post.ptr, rows, d, .production, 0);
                try k.mlp.norms.addRmsH(s, sc.mixed.ptr, null, self.input_norm, .bf16, 0, sc.normed.ptr, null, rows, d, c.eps);
            }
            if (call.prof) |p| p.mark(s, .ihc);
        }

        var mla_ops = k.mlaOpsProf(call.prof);
        mla_ops.kv8 = segs[0].cache.kv8;
        mla_ops.rot_pos = call.rot_pos;
        mla_ops.pos3 = call.pos3;
        mla_ops.ld3 = call.ld3;
        mla_ops.row_stream = call.row_stream;
        mla_ops.pt_base = call.pt_base;
        const mla_sc = sc.mlaFor(self.is_sliding);
        const mla_gather: ?cuda_mla.GatherBf16 = if (call.all_reduce) |ar| if (ar.gather_bf16) |gf| .{ .ctx = ar.ctx, .gather = gf } else null else null;
        try cuda_mla.qkvPrep(mla_ops, s, &self.mla, mla_sc, sc.normed.ptr, rows, mla_gather, &k.mlp.rows);
        if (call.prof) |p| p.mark(s, .qkv);
        var msegs: [max_segs]cuda_mla.Seg = undefined;
        try runIndexer(self, k, s, c, sc, mla_sc, call, segs, msegs[0..segs.len]);
        if (!self.is_sliding) {
            if (call.prof) |p| p.mark(s, .idx);
        }
        try cuda_mla.attendCore(mla_ops, s, &self.mla, mla_sc, sc.normed.ptr, rows, call.pos, msegs[0..segs.len], call.caches, true, call.single_split_attn);
        return attn_streams;
    }

    /// Position-independent second half (`attendOut` + `attn_comm` + iHC + MLP + `mlp_comm` + `next_boundary`).
    pub fn forwardPostAttn(self: *const Layer, k: *const Kernels, s: Stream, c: Config, sc: *const Scratch, attn_streams: u64, call: Call) !void {
        const d = c.hidden;
        const rows = call.rows;
        const use_ihc = d == cuda_ihc.hidden and (attn_streams == call.streams_out or call.engram_dst != 0);
        const mla_ops = k.mlaOpsProf(call.prof);
        const mla_sc = sc.mlaFor(self.is_sliding);
        try cuda_mla.attendOut(mla_ops, s, &self.mla, mla_sc, sc.normed.ptr, rows, sc.partial.ptr);
        const attn_sum = if (call.all_reduce) |ar| blk: {
            try ar.sum(ar.ctx, s, sc.partial.ptr, sc.reduced.ptr, rows * d);
            if (call.prof) |p| p.mark(s, .attn_comm);
            break :blk sc.reduced.ptr;
        } else sc.partial.ptr;
        if (use_ihc) {
            try k.ihc.run(s, .{
                .mode = .boundary,
                .rows = rows,
                .g = attn_sum,
                .x = attn_streams,
                .post_prev = sc.post.ptr,
                .post_nw = self.mla.post_norm.ptr,
                .mix = self.hc_mlp,
                .next_nw = self.mlp.norm,
                .a = sc.normed.ptr,
                .post_next = sc.post.ptr,
                .eps = c.eps,
                .hc_eps = c.hc_eps,
                .mag = c.hc_magnitude,
                .round_in = true,
            });
        } else {
            try cuda_mla.finish(mla_ops, s, &self.mla, attn_sum, 1, rows, sc.branch.ptr);
            try k.hc.merge(s, attn_streams, sc.branch.ptr, sc.post.ptr, sc.mid.ptr, rows, d, .production, 0);
            try k.hc.pre(s, self.hc_mlp, sc.mid.ptr, sc.mixed.ptr, sc.post.ptr, rows, d, .production, 0);
            try cuda_mlp.norm(&k.mlp, s, c, self.mlp, sc.mixed.ptr, sc.normed.ptr, rows);
        }
        if (call.prof) |p| p.mark(s, .ihc);

        const mlp_comm: ?cuda_mlp.ShardedComm = if (call.all_reduce) |ar| .{
            .ctx = ar.ctx,
            .rank = ar.rank,
            .gather_bf16 = ar.gather_bf16,
            .exchange_f32 = ar.exchange_f32,
        } else null;
        try cuda_mlp.partialProf(&k.mlp, s, c, self.mlp, &sc.mlp, sc.normed.ptr, sc.partial.ptr, rows, call.prompt orelse (rows > 128), call.prof, mlp_comm);
        const mlp_sum = if (call.all_reduce) |ar| blk: {
            try ar.sum(ar.ctx, s, sc.partial.ptr, sc.reduced.ptr, rows * d);
            if (call.prof) |p| p.mark(s, .mlp_comm);
            break :blk sc.reduced.ptr;
        } else sc.partial.ptr;
        if (use_ihc) {
            switch (call.next_boundary) {
                .attn => |nb| try k.ihc.run(s, .{
                    .mode = .boundary,
                    .rows = rows,
                    .g = mlp_sum,
                    .x = attn_streams,
                    .post_prev = sc.post.ptr,
                    .post_nw = self.mlp.post_norm_bf16,
                    .mix = nb.mix,
                    .next_nw = nb.norm,
                    .a = sc.normed.ptr,
                    .post_next = sc.post.ptr,
                    .eps = c.eps,
                    .hc_eps = c.hc_eps,
                    .mag = c.hc_magnitude,
                    .round_in = false,
                }),
                .head => |hb| try k.ihc.run(s, .{
                    .mode = .head,
                    .rows = rows,
                    .g = mlp_sum,
                    .x = attn_streams,
                    .post_prev = sc.post.ptr,
                    .post_nw = self.mlp.post_norm_bf16,
                    .mix = hb.mix,
                    .next_nw = hb.norm,
                    .a = hb.out,
                    .eps = c.eps,
                    .hc_eps = c.hc_eps,
                    .mag = c.hc_magnitude,
                    .round_in = false,
                }),
                .post_only => try k.ihc.run(s, .{
                    .mode = .post_only,
                    .rows = rows,
                    .g = mlp_sum,
                    .x = attn_streams,
                    .post_prev = sc.post.ptr,
                    .post_nw = self.mlp.post_norm_bf16,
                    .eps = c.eps,
                    .hc_eps = c.hc_eps,
                    .mag = c.hc_magnitude,
                    .round_in = false,
                }),
            }
        } else {
            try cuda_mlp.finish(&k.mlp, s, c, self.mlp, mlp_sum, sc.branch.ptr, rows);
            try k.hc.merge(s, sc.mid.ptr, sc.branch.ptr, sc.post.ptr, call.streams_out, rows, d, .production, 0);
        }
        if (call.prof) |p| p.mark(s, .ihc);
    }
};

/// Most sequences one batched step carries (the model's slots).
pub const max_segs: usize = 32;

/// One sequence's rows `[row0, row0 + rows)` of a batched step, at positions from `first` in its own cache.
pub const Seg = struct { row0: usize, rows: usize, first: usize, cache: cuda_cache.LayerView };

/// MLP epilogue target: next layer's attention entry, final head, or stream update only (before Engram).
pub const NextBoundary = union(enum) {
    post_only: void,
    attn: struct { mix: cuda_hc.Hc.Mix, norm: u64 },
    head: struct { mix: cuda_hc.Hc.Mix, norm: u64, out: u64 },
};

/// Inputs and seam hooks for one DecoderLayer step of `rows` tokens starting at position `first`.
pub const Call = struct {
    streams_in: u64, // bf16 [rows, 4, hidden]
    streams_out: u64, // bf16 [rows, 4, hidden] (may alias streams_in)
    rows: usize,
    first: usize = 0,
    pos: u64, // device i32 [rows]: first, first + 1, ...
    rot_pos: u64 = 0, // optional device i32 [rows]: rotary angle positions (`pos + mrope_delta`)
    pos3: u64 = 0, // optional device i32 [3, ld3]: 3D M-RoPE positions
    ld3: usize = 0,
    cache: cuda_cache.LayerView,
    engram: ?EngramHook = null,
    indexer: ?IndexerHook = null,
    all_reduce: ?AllReduce = null,
    prompt: ?bool = null,
    segs: []const Seg = &.{}, // several sequences' rows in order; empty: one at `first` in `cache`
    caches: u64 = 0, // with segs: device u64 [rows], each row's MLA cache on this layer (one attention launch)
    row_stream: u64 = 0, // batched paged indexed KV: device i32 [rows] slot indices
    pt_base: u64 = 0, // batched paged indexed KV: base of [max_slots, pt_stride] u32
    skip_entry: bool = false, // previous layer's fused boundary already wrote sc.normed and sc.post
    next_boundary: NextBoundary = .post_only,
    engram_dst: u64 = 0, // non-zero: non-aliasing output buffer for runEngram (ping-pong with streams_in)
    single_split_attn: bool = false, // prompt passes: 1-split c1 at every width for 0-bit resumed==fresh
    prof: ?*cuda_prof.Prof = null,
};

/// The step's segments (one from `first` and `cache` when `call.segs` is empty), checked to tile `call.rows`.
pub fn checkSegs(c: Config, layer_idx: usize, call: Call, one: *[1]Seg) ![]const Seg {
    one[0] = .{ .row0 = 0, .rows = call.rows, .first = call.first, .cache = call.cache };
    const segs: []const Seg = if (call.segs.len > 0) call.segs else one;
    if (segs.len > max_segs) return error.ScratchShape;
    var covered: usize = 0;
    for (segs) |g| {
        if (g.row0 != covered) return error.ScratchShape;
        if (g.cache.kind != c.kind[layer_idx]) return error.CacheShape;
        try g.cache.check(g.first, g.rows);
        covered += g.rows;
    }
    if (covered != call.rows) return error.ScratchShape;
    return segs;
}

pub fn runEngram(layer: *const Layer, sc: *const Scratch, s: Stream, call: Call) !u64 {
    const slot = layer.engram_slot orelse return call.streams_in;
    const eh = call.engram orelse return call.streams_in;
    const dst = if (call.engram_dst != 0) call.engram_dst else sc.engram_buf.ptr;
    try eh.run(eh.ctx, .{ .layer_idx = layer.layer_idx, .slot = slot, .s = s, .streams_in = call.streams_in, .streams_out = dst, .rows = call.rows, .first = call.first, .pos = call.pos, .all_reduce = call.all_reduce });
    return dst;
}

/// Fills each segment's MLA arguments and, on indexed layers, runs the indexer hook per segment for its recall list.
pub fn runIndexer(layer: *const Layer, _: *const Kernels, s: Stream, c: Config, sc: *const Scratch, mla_sc: *const cuda_mla.Scratch, call: Call, segs: []const Seg, out: []cuda_mla.Seg) !void {
    for (segs, out) |g, *o| o.* = .{ .row0 = g.row0, .rows = g.rows, .first = g.first, .cache = g.cache.mla, .ring = g.cache.ring, .page_table = g.cache.page_table, .pt_stride = g.cache.pt_stride };
    if (layer.is_sliding) return;
    const ih = call.indexer orelse return;
    const limit = c.recall();
    const q_lora = c.q_lora;
    var q_resid: u64 = 0;
    for (segs) |g| if (g.first + g.rows > limit) {
        q_resid = mla_sc.qres(); // qkvPrep already computed q_a_layernorm(q_a(normed)) into mla_sc.qres()
        break;
    };
    const preproj = segs.len > 1 and ih.prep != null;
    if (preproj) try ih.prep.?(ih.ctx, layer.layer_idx, s, sc.normed.ptr, q_resid, call.rows);
    for (segs, out, 0..) |g, *o, i| {
        const r = try ih.run(ih.ctx, .{
            .layer_idx = layer.layer_idx,
            .seg = i,
            .s = s,
            .attn_in = sc.normed.ptr + g.row0 * c.hidden * 2,
            .q_resid = if (q_resid != 0) q_resid + g.row0 * q_lora * 2 else 0,
            .rows = g.rows,
            .first = g.first,
            .pos = call.pos + g.row0 * 4,
            .index_keys = g.cache.index_keys,
            .capacity = g.cache.capacity,
            .idx_out = sc.idx.ptr + g.row0 * limit * 4,
            .recall_limit = limit,
            .page_table = g.cache.page_table,
            .pt_stride = g.cache.pt_stride,
            .row0 = g.row0,
            .preprojected = preproj,
        });
        const rec = r orelse continue;
        if (g.first + g.rows <= limit or rec.list_from >= g.rows or rec.nidx == 0) return error.IndexList;
        o.list_from = rec.list_from;
        o.idx = rec.idx;
        o.nidx = rec.nidx;
    }
}

pub const forwardVirtual = @import("cuda_layer_virtual.zig").forwardVirtual;
pub const WorldLayer = @import("cuda_layer_virtual.zig").WorldLayer;

test "cuda_layer declarations compile on a host without kernels" {
    inline for (.{ @This(), Kernels, Scratch, Layer, WorldLayer }) |T| std.testing.refAllDecls(T);
}
