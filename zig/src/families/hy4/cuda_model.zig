//! Whole-model 42-layer Hy4-flash rank engine (`world` 1, 2, 4) with Indexer, Engram, KV slots and TP collectives.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const names = @import("names.zig");
const ep = @import("engram_pack.zig");
const er = @import("engram_reader.zig");
const cuda_weights = @import("cuda_weights.zig");
const cuda_head = @import("cuda_head.zig");
const cuda_cache = @import("cuda_cache.zig");
const cuda_layer = @import("cuda_layer.zig");
const cuda_index = @import("cuda_index.zig");
const cuda_engram = @import("cuda_engram.zig");
const cuda_dfly = @import("cuda_dfly.zig");
const cuda_prof = @import("cuda_prof.zig");
const cuda_sample = @import("cuda_sample.zig");
const cuda_tp = @import("cuda_tp.zig");
const cmedia = @import("cuda_media.zig");
const cuda_pages = @import("cuda_pages.zig");

const Config = config.Config;
const Stream = cuda.Stream;
const DeviceBuffer = cuda.DeviceBuffer;
const Rank = cuda_weights.Rank;

pub const prompt_rows: usize = 4096;
pub const max_window: usize = 16;
pub const max_batch: usize = 64; // max rows in one shared decode/verify forward pass
pub const round_rows: usize = 128;
pub const max_slots: usize = 32;
pub const ring_slots: usize = 4608; // >= 511 + prompt_rows (4,607), 64-aligned
/// `pos_host`: every part's n-gram window positions, then the step's row positions from here.
pub const pos_windows: usize = prompt_rows + max_slots * config.engram_orders;

pub const AllKernels = struct {
    layer: cuda_layer.Kernels,
    idx: cuda_index.Kernels,
    engram: cuda_engram.Kernels,
    dfly: cuda_dfly.Kernels,
    sample_mod: cuda.Module,
    draws: cuda.sample.Draws,

    pub fn load(ctx: *const cuda.Context) !AllKernels {
        var layer = try cuda_layer.Kernels.load(ctx);
        errdefer layer.deinit();
        var idx = try cuda_index.Kernels.load(ctx.d);
        errdefer idx.unload();
        var engram = try cuda_engram.Kernels.load(ctx);
        errdefer engram.deinit();
        var dfly = try cuda_dfly.Kernels.load(ctx.d);
        errdefer dfly.unload();
        var sample_mod = try cuda.Module.load(ctx.d, cuda.kernels.sample);
        errdefer sample_mod.unload();
        return .{ .layer = layer, .idx = idx, .engram = engram, .dfly = dfly, .sample_mod = sample_mod, .draws = try cuda.sample.Draws.resolve(sample_mod) };
    }

    pub fn deinit(k: *AllKernels) void {
        k.sample_mod.unload();
        k.dfly.unload();
        k.engram.deinit();
        k.idx.unload();
        k.layer.deinit();
        k.* = undefined;
    }

    pub fn idxOps(k: *const AllKernels) cuda_index.Ops {
        return .{
            .k = &k.idx,
            .lane = &k.layer.mlp.lane,
            .norms = &k.layer.mlp.norms,
            .dense = &k.layer.mlp.dense,
        };
    }
};

const sys = @import("sys");
const env = sys.environment;

/// Resolves the directory containing packed Engram files (`TF_HY4_ENGRAM_DIR`, `<dir>`, or `<dir>/../hy4-engram`).
pub fn resolveEngramDir(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8) ![]u8 {
    if (try env.path("TF_HY4_ENGRAM_DIR", @import("profile").lookup("TF_HY4_ENGRAM_DIR"))) |s| {
        if (s.len > 0) return gpa.dupe(u8, s);
    }
    var nbuf: [64]u8 = undefined;
    const probe = try ep.fileName(&nbuf, 1);
    const in_model = try std.fs.path.join(gpa, &.{ model_dir, probe });
    defer gpa.free(in_model);
    if (std.Io.Dir.cwd().statFile(io, in_model, .{})) |_| return gpa.dupe(u8, model_dir) else |_| {}
    const sibling = try std.fs.path.join(gpa, &.{ model_dir, "..", "hy4-engram" });
    errdefer gpa.free(sibling);
    const sib_probe = try std.fs.path.join(gpa, &.{ sibling, probe });
    defer gpa.free(sib_probe);
    if (std.Io.Dir.cwd().statFile(io, sib_probe, .{})) |_| return sibling else |_| {}
    gpa.free(sibling);
    return error.MissingEngramPackDir;
}

pub fn parseEngramMode() !er.Mode {
    const idx = try env.read(usize, "TF_HY4_ENGRAM_MODE", 0);
    return if (idx == 1) .resident else .nvme;
}

pub fn parseEngramCache() !u32 {
    return env.read(u32, "TF_HY4_ENGRAM_CACHE", er.default_cache_rows);
}

pub fn parseLayersLimit(total: usize) !usize {
    const n = (try env.read(?usize, "TF_HY4_LAYERS", null)) orelse return total;
    if (n == 0 or n > total) return error.BadLayersLimit;
    return n;
}

const dropCheckpointCache = cuda_pages.dropCheckpointCache;

/// Per-stream KV cache + compressed-indexer pooled group keys + n-gram token history.
pub const Seq = struct {
    gpa: std.mem.Allocator,
    cap: usize,
    len: usize = 0,
    cache: cuda_cache.SequenceCache,
    group_keys: ?DeviceBuffer = null, // bf16 [n_indexed, cap / 4, 128]
    last_hidden: DeviceBuffer, // bf16 [max_window, hidden]: last forward's post-norm hidden rows for DFly2
    dfly_topk: DeviceBuffer, // i32 [5][max_window][2048]: saved target topk for DFly2 past 2K context
    history: []u32,
    mrope_delta: i32 = 0,
    media: ?cmedia.SeqMedia = null,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, c: Config, layer_ids: []const usize, n_indexed: usize, asked_cap: usize, fmt: cuda_cache.Format) !Seq {
        const cap = std.mem.alignForward(usize, @max(asked_cap, 64), 64);
        var cache = try cuda_cache.SequenceCache.initLayersEx(d, c, layer_ids, cap, .{ .ring = ring_slots }, fmt);
        errdefer cache.deinit();
        var gk: ?DeviceBuffer = null;
        if (n_indexed > 0 and !fmt.paged) {
            gk = try DeviceBuffer.alloc(d, n_indexed * (cap / cuda_index.ratio) * cuda_index.dim * 2);
        }
        errdefer if (gk) |*b| b.free();
        var lh = try DeviceBuffer.alloc(d, max_window * c.hidden * 2);
        errdefer lh.free();
        var dtk = try DeviceBuffer.alloc(d, cuda_dfly.num_layers * max_window * cuda_dfly.target_topk * 4);
        errdefer dtk.free();
        const history = try gpa.alloc(u32, cap);
        @memset(history, c.engram_pad);
        return .{ .gpa = gpa, .cap = cap, .cache = cache, .group_keys = gk, .last_hidden = lh, .dfly_topk = dtk, .history = history };
    }

    pub fn deinit(sq: *Seq) void {
        if (sq.media) |*sm| sm.deinit();
        sq.gpa.free(sq.history);
        sq.dfly_topk.free();
        sq.last_hidden.free();
        if (sq.group_keys) |*b| b.free();
        sq.cache.deinit();
        sq.* = undefined;
    }

    pub fn groupKeysAt(sq: *const Seq, ord: usize) u64 {
        const stride = (sq.cap / cuda_index.ratio) * cuda_index.dim * 2;
        return sq.group_keys.?.ptr + ord * stride;
    }

    pub fn dflyTopkAt(sq: *const Seq, dl: usize, row_idx: usize) u64 {
        return sq.dfly_topk.ptr + (dl * max_window + row_idx) * cuda_dfly.target_topk * 4;
    }
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    c: Config,
    at: Rank,
    context: usize,
    kv8: bool = false,
    active_layers: usize,
    layer_ids: []usize,
    n_indexed: usize,
    indexed_ord: [config.max_layers]?usize = @splat(null),
    first_idx_layer: ?usize = null,
    tp_sess: ?*cuda_tp.Session = null,
    head: cuda_head.Head,
    layers: []cuda_layer.Layer,
    index_weights: []?cuda_index.weights.Weights,
    engram_store: cuda_weights.Weights,
    engram_weights: [config.engram_layers]?cuda_weights.Engram = @splat(null),
    engram_reader: ?er.Reader = null,
    dfly_w: ?cuda_dfly.Weights = null,
    dfly_sc: ?cuda_dfly.Scratch = null,
    prof: ?cuda_prof.Prof = null,
    layer_sc: cuda_layer.Scratch,
    head_sc: cuda_head.Scratch,
    idx_sc: ?cuda_index.Scratch = null,
    engram_sc: ?cuda_engram.Scratch = null,
    engram_partial: DeviceBuffer, // fp32 [prompt_rows, 5 * hidden]
    engram_reduced: DeviceBuffer, // fp32 [prompt_rows, 5 * hidden]
    gather_buf: DeviceBuffer, // fp32 [world, prompt_rows, 5 * hidden] disjoint allGather target
    engram_out: DeviceBuffer, // bf16 [prompt_rows, 4, hidden]
    streams_buf: DeviceBuffer, // bf16 [prompt_rows, 4, hidden]
    ids_dev: DeviceBuffer, // u32 [prompt_rows]
    pos_dev: DeviceBuffer, // i32 [prompt_rows]
    rot_pos_dev: DeviceBuffer, // i32 [prompt_rows]: rotary positions (`pos + mrope_delta`)
    pos3_decode_dev: DeviceBuffer, // i32 [3 * prompt_rows]: 3D rotary positions when `mrope_delta != 0`
    logits_local: DeviceBuffer, // fp32 [max_batch, vocab_local]
    logits_full: ?DeviceBuffer = null, // fp32 [max_batch, vocab] when world > 1
    sample_sc: cuda_sample.Scratch,
    argmax_local: DeviceBuffer, // u32 [max_batch, 2]: {max_bits, global_id}
    argmax_full: ?DeviceBuffer = null, // u32 [world, max_batch, 2] when world > 1
    vote_dev: DeviceBuffer, // 64 B scratch for atomic multi-rank allocation votes
    row_caches: DeviceBuffer, // u64 [max_layers][max_batch]: a batched step's MLA cache a row, by layer
    rope_cs: ?DeviceBuffer = null, // float2 [context, 32] shared across all indexed layers
    rope_ready: usize = 0,
    pos_host: []u32,
    rot_pos_host: []i32,
    scratch_bytes: u64 = 0,
    seqs: [max_slots]?Seq = @splat(null),
    pages: ?*cuda_pages.Pages = null,
    swa_spans: [max_swa_spans]SwaSpan = @splat(.{}),
    n_swa_spans: usize = 0,
    swa_span_at: [config.max_layers]?u8 = @splat(null),
    swa_graphs: [max_swa_spans][max_batch + 1]GraphSlot = @splat(@splat(.{})),
    swa_graphs_enabled: bool = true,
    table_key_parts: usize = 0,
    table_key_slots: [cuda_layer.max_segs]usize = @splat(0),
    table_key_lens: [cuda_layer.max_segs]usize = @splat(0),
    table_key_mla0: [cuda_layer.max_segs]u64 = @splat(0),

    pub const max_swa_spans: usize = 16;
    pub const SwaSpan = struct { tail_layer: ?usize = null, swa_lo: usize = 0, swa_hi: usize = 0, fuse_head: bool = false };
    pub const GraphState = enum(u8) { cold, warm, ready, failed };
    pub const GraphSlot = struct { state: GraphState = .cold, exec: cuda.graph.Exec = undefined };

    fn readSwaGraphEnabled() bool {
        const v = @import("profile").lookup("TF_HY4_SWA_GRAPH") orelse return true;
        return v.len == 0 or v[0] != '0';
    }

    pub const InitOptions = struct {
        at: Rank = .{},
        context: usize,
        layers_limit: ?usize = null,
        engram_mode: er.Mode = .nvme,
        engram_cache: u32 = er.default_cache_rows,
        tp_sess: ?*cuda_tp.Session = null,
        kv8: bool = false,
    };

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, k: *const AllKernels, dir: []const u8, opts: InitOptions) !*Model {
        const d = ctx.d;
        const c = try Config.read(gpa, io, dir);
        const active = opts.layers_limit orelse try parseLayersLimit(c.layers);
        if (active == 0 or active > c.layers) return error.BadLayersLimit;
        const cap_aligned = std.mem.alignForward(usize, @max(opts.context + max_window, 64), 64);

        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        const layer_ids = try gpa.alloc(usize, active);
        errdefer gpa.free(layer_ids);
        for (layer_ids, 0..) |*id, i| id.* = i;
        const layers = try gpa.alloc(cuda_layer.Layer, active);
        errdefer gpa.free(layers);
        const idx_w = try gpa.alloc(?cuda_index.weights.Weights, active);
        errdefer gpa.free(idx_w);
        @memset(idx_w, null);
        const pos_host = try gpa.alloc(u32, pos_windows + prompt_rows);
        errdefer gpa.free(pos_host);
        const rot_pos_host = try gpa.alloc(i32, 3 * prompt_rows);
        errdefer gpa.free(rot_pos_host);

        var engram_slots_buf: [config.engram_layers]usize = undefined;
        var n_eng: usize = 0;
        var n_idx: usize = 0;
        var indexed_ord: [config.max_layers]?usize = @splat(null);
        var first_idx: ?usize = null;
        var engram_store = cuda_weights.Weights.init(gpa, d);
        errdefer engram_store.deinit();
        var engram_w: [config.engram_layers]?cuda_weights.Engram = @splat(null);

        var head: cuda_head.Head = undefined;
        var dfly_w: ?cuda_dfly.Weights = null;
        var loaded_layers: usize = 0;
        errdefer {
            if (dfly_w) |*dw| dw.deinit();
            for (idx_w[0..loaded_layers]) |*iw| if (iw.*) |*v| v.deinit();
            for (layers[0..loaded_layers]) |*ly| ly.deinit();
        }
        {
            var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
            defer {
                dropCheckpointCache(&ck);
                ck.close();
            }
            head = try cuda_head.Head.load(d, try names.top(&ck, c), c, opts.at);
            errdefer head.deinit();
            // TF_HY4_SHARD_REP=0 keeps qkv_a and latent_down whole on every rank (on: world 4 gains 0.96 ms a token)
            const shard_rep = opts.tp_sess != null and blk: {
                const v = @import("profile").lookup("TF_HY4_SHARD_REP") orelse break :blk true;
                break :blk v.len == 0 or v[0] != '0';
            };
            for (0..active) |l_idx| {
                const l = try names.layer(&ck, c, l_idx);
                layers[l_idx] = try cuda_layer.Layer.loadEx(gpa, d, l, c, l_idx, opts.at, shard_rep);
                if (l.attn.indexer) |ix| {
                    idx_w[l_idx] = try cuda_index.weights.Weights.load(d, gpa, ix, c);
                    indexed_ord[l_idx] = n_idx;
                    if (first_idx == null) first_idx = l_idx;
                    n_idx += 1;
                }
                if (c.engramSlot(l_idx)) |slot| {
                    engram_w[slot] = try engram_store.engram(l, c, l_idx, opts.at);
                    engram_slots_buf[n_eng] = slot;
                    n_eng += 1;
                }
                loaded_layers = l_idx + 1;
                if (loaded_layers % 4 == 0) dropCheckpointCache(&ck);
            }
            if (cuda_dfly.Weights.enabled() and active == c.layers and c.hidden == cuda_dfly.hidden and cuda_dfly.Weights.present(&ck)) {
                dfly_w = try cuda_dfly.Weights.load(gpa, d, &ck, c, opts.at);
                std.log.info("[tensorfold] hy4: loaded DFly2 5-layer built-in drafter (rank {d}/{d}, fp8_head={}, shard_rep={})", .{ opts.at.rank, opts.at.world, dfly_w.?.lm_head_fp8 != null, shard_rep });
            }
        }
        errdefer head.deinit();

        var dfly_sc: ?cuda_dfly.Scratch = null;
        if (dfly_w) |*dw| {
            dfly_sc = try cuda_dfly.Scratch.init(d, &k.layer.mlp.lane, dw);
        }
        errdefer if (dfly_sc) |*ds| ds.deinit();

        var prof: ?cuda_prof.Prof = null;
        if (opts.at.rank == 0 and cuda_prof.enabled()) {
            prof = try cuda_prof.Prof.init(d);
            std.log.info("[tensorfold] hy4: TF_HY4_PROF=1 CUDA-event profiler enabled on rank 0", .{});
        }
        errdefer if (prof) |*p| p.deinit();

        var reader: ?er.Reader = null;
        if (n_eng > 0) {
            const eng_dir = try resolveEngramDir(gpa, io, dir);
            defer gpa.free(eng_dir);
            reader = try er.Reader.open(gpa, io, eng_dir, c, .{ .rank = opts.at.rank, .world = opts.at.world }, engram_slots_buf[0..n_eng], opts.engram_mode, opts.engram_cache);
        }
        errdefer if (reader) |*r| r.close();

        const dev_before_sc = cuda.usage(false).device;
        var layer_sc = try cuda_layer.Scratch.init(gpa, d, &k.layer.mlp.lane, c, opts.at.world, prompt_rows);
        errdefer layer_sc.deinit();
        var head_sc = try cuda_head.Scratch.init(d, c, max_batch);
        errdefer head_sc.deinit();
        var idx_sc: ?cuda_index.Scratch = null;
        if (first_idx) |fi| {
            idx_sc = try cuda_index.Scratch.init(d, &k.layer.mlp.lane, &idx_w[fi].?, prompt_rows, cap_aligned);
        }
        errdefer if (idx_sc) |*s| s.deinit();
        var engram_sc: ?cuda_engram.Scratch = null;
        if (n_eng > 0) {
            engram_sc = try cuda_engram.Scratch.init(gpa, d, &k.engram.lane, c, prompt_rows);
        }
        errdefer if (engram_sc) |*s| s.deinit();

        const eng_cols = cuda_engram.projections * c.hidden;
        var eng_part = try DeviceBuffer.alloc(d, @max(prompt_rows * eng_cols * 4, 16));
        errdefer eng_part.free();
        var eng_red = try DeviceBuffer.alloc(d, @max(prompt_rows * eng_cols * 4, 16));
        errdefer eng_red.free();
        const g_len = if (opts.tp_sess) |ts| ts.gatherBytes(prompt_rows * eng_cols) else 0;
        var g_buf = try DeviceBuffer.alloc(d, @max(g_len, 16));
        errdefer g_buf.free();
        var eng_out = try DeviceBuffer.alloc(d, prompt_rows * cuda_head.streams * c.hidden * 2);
        errdefer eng_out.free();
        var streams_buf = try DeviceBuffer.alloc(d, prompt_rows * cuda_head.streams * c.hidden * 2);
        errdefer streams_buf.free();
        var ids_dev = try DeviceBuffer.alloc(d, prompt_rows * 4);
        errdefer ids_dev.free();
        var pos_dev = try DeviceBuffer.alloc(d, prompt_rows * 4);
        errdefer pos_dev.free();
        var rot_pos_dev = try DeviceBuffer.alloc(d, prompt_rows * 4);
        errdefer rot_pos_dev.free();
        var pos3_dec_dev = try DeviceBuffer.alloc(d, 3 * prompt_rows * 4);
        errdefer pos3_dec_dev.free();
        const vocab_local = head.top.share.len;
        var logits_local = try DeviceBuffer.alloc(d, max_batch * vocab_local * 4);
        errdefer logits_local.free();
        var logits_full: ?DeviceBuffer = null;
        if (opts.at.world > 1) logits_full = try DeviceBuffer.alloc(d, max_batch * c.vocab * 4);
        errdefer if (logits_full) |*b| b.free();
        var sample_sc = try cuda_sample.Scratch.init(d, max_batch, c.vocab, vocab_local, opts.at.world);
        errdefer sample_sc.deinit();
        var argmax_local = try DeviceBuffer.alloc(d, max_batch * 8);
        errdefer argmax_local.free();
        var argmax_full: ?DeviceBuffer = null;
        if (opts.at.world > 1) argmax_full = try DeviceBuffer.alloc(d, opts.at.world * max_batch * 8);
        errdefer if (argmax_full) |*b| b.free();
        var vote_dev = try DeviceBuffer.alloc(d, 64);
        errdefer vote_dev.free();
        var row_caches = try DeviceBuffer.alloc(d, config.max_layers * max_batch * 8);
        errdefer row_caches.free();
        var rope_cs: ?DeviceBuffer = null;
        if (n_idx > 0) rope_cs = try DeviceBuffer.alloc(d, (cap_aligned + 16_384) * cuda_index.rope_pairs * 8);
        const scratch_bytes = cuda.usage(false).device -| dev_before_sc;

        m.* = .{
            .gpa = gpa,
            .d = d,
            .c = c,
            .at = opts.at,
            .context = cap_aligned,
            .kv8 = opts.kv8,
            .active_layers = active,
            .layer_ids = layer_ids,
            .n_indexed = n_idx,
            .indexed_ord = indexed_ord,
            .first_idx_layer = first_idx,
            .tp_sess = opts.tp_sess,
            .head = head,
            .layers = layers,
            .index_weights = idx_w,
            .engram_store = engram_store,
            .engram_weights = engram_w,
            .engram_reader = reader,
            .dfly_w = dfly_w,
            .dfly_sc = dfly_sc,
            .prof = prof,
            .layer_sc = layer_sc,
            .head_sc = head_sc,
            .idx_sc = idx_sc,
            .engram_sc = engram_sc,
            .engram_partial = eng_part,
            .engram_reduced = eng_red,
            .gather_buf = g_buf,
            .engram_out = eng_out,
            .streams_buf = streams_buf,
            .ids_dev = ids_dev,
            .pos_dev = pos_dev,
            .rot_pos_dev = rot_pos_dev,
            .pos3_decode_dev = pos3_dec_dev,
            .logits_local = logits_local,
            .logits_full = logits_full,
            .sample_sc = sample_sc,
            .argmax_local = argmax_local,
            .argmax_full = argmax_full,
            .vote_dev = vote_dev,
            .row_caches = row_caches,
            .rope_cs = rope_cs,
            .pos_host = pos_host,
            .rot_pos_host = rot_pos_host,
            .scratch_bytes = scratch_bytes,
            .swa_graphs_enabled = readSwaGraphEnabled() and c.hidden == @import("cuda_ihc.zig").hidden,
        };
        fwd_impl.initSwaSpans(m, active);
        return m;
    }

    pub fn deinit(m: *Model) void {
        if (m.pages) |pg| {
            if (cuda.Stream.init(m.d, false) catch null) |ts| {
                pg.deinit(ts);
                var ts_mut = ts;
                ts_mut.deinit();
            }
            m.pages = null;
        }
        for (0..m.n_swa_spans) |si| {
            for (&m.swa_graphs[si]) |*slot| {
                if (slot.state == .ready) slot.exec.deinit();
            }
        }
        for (&m.seqs) |*sq| if (sq.*) |*v| v.deinit();
        if (m.rope_cs) |*b| b.free();
        m.vote_dev.free();
        m.row_caches.free();
        if (m.argmax_full) |*b| b.free();
        m.argmax_local.free();
        m.sample_sc.deinit();
        if (m.logits_full) |*b| b.free();
        m.logits_local.free();
        m.pos3_decode_dev.free();
        m.rot_pos_dev.free();
        m.pos_dev.free();
        m.ids_dev.free();
        m.streams_buf.free();
        m.engram_out.free();
        m.gather_buf.free();
        m.engram_reduced.free();
        m.engram_partial.free();
        if (m.prof) |*p| p.deinit();
        if (m.engram_sc) |*s| s.deinit();
        if (m.dfly_sc) |*s| s.deinit();
        if (m.idx_sc) |*s| s.deinit();
        m.head_sc.deinit();
        m.layer_sc.deinit();
        if (m.engram_reader) |*r| r.close();
        m.engram_store.deinit();
        if (m.dfly_w) |*w| w.deinit();
        for (m.index_weights) |*iw| if (iw.*) |*v| v.deinit();
        for (m.layers) |*ly| ly.deinit();
        m.head.deinit();
        m.gpa.free(m.rot_pos_host);
        m.gpa.free(m.pos_host);
        m.gpa.free(m.index_weights);
        m.gpa.free(m.layers);
        m.gpa.free(m.layer_ids);
        const gpa = m.gpa;
        gpa.destroy(m);
    }

    pub fn fixedSeqBytes(m: *const Model) u64 {
        var n_sld: u64 = 0;
        for (m.layer_ids) |l| if (m.c.kind[l] == .sliding) {
            n_sld += 1;
        };
        const tail_b: u64 = @as(u64, m.n_indexed) * cuda_cache.tail_index_rows * cuda_cache.index_row_bytes;
        return n_sld * ring_slots * cuda_cache.mlaRowBytes(m.kv8) + tail_b;
    }

    pub fn tokenSeqBytes(m: *const Model) u64 {
        const idx_tok: u64 = cuda_cache.mlaRowBytes(m.kv8) + (cuda_index.dim * 2) / cuda_index.ratio;
        return @as(u64, m.n_indexed) * idx_tok + 4;
    }

    pub fn logitsPtr(m: *const Model) u64 {
        return if (m.logits_full) |b| b.ptr else m.logits_local.ptr;
    }

    pub fn hasDfly(m: *const Model) bool {
        return m.dfly_w != null and m.dfly_sc != null;
    }

    pub fn argmaxPtr(m: *const Model) u64 {
        return if (m.argmax_full) |b| b.ptr else m.argmax_local.ptr;
    }

    pub fn openSeq(m: *Model, slot: usize, asked_cap: usize) !void {
        if (slot >= max_slots or m.seqs[slot] != null) return error.BadSlot;
        m.seqs[slot] = try Seq.init(m.gpa, m.d, m.c, m.layer_ids, m.n_indexed, asked_cap, .{ .kv8 = m.kv8, .tail_index_keys = true });
    }
    pub fn openSeqPaged(m: *Model, slot: usize, asked_cap: usize) !void {
        if (slot >= max_slots or m.seqs[slot] != null) return error.BadSlot;
        m.seqs[slot] = try Seq.init(m.gpa, m.d, m.c, m.layer_ids, m.n_indexed, asked_cap, .{ .kv8 = m.kv8, .tail_index_keys = true, .paged = true });
    }

    pub fn releaseSeq(m: *Model, slot: usize) void {
        if (slot < max_slots) if (m.seqs[slot]) |*sq| {
            sq.deinit();
            m.seqs[slot] = null;
            m.table_key_parts = 0;
        };
    }

    pub fn ensureRope(m: *Model, k: *const AllKernels, s: Stream, end: usize) !void {
        const fi = m.first_idx_layer orelse return;
        if (end <= m.rope_ready) return;
        const first = m.rope_ready;
        try cuda_index.ropeTable(k.idx.rope_table, s, m.rope_cs.?.ptr, first, end - first, m.index_weights[fi].?.freqs.ptr, 0, 0, 10, 10, false);
        m.rope_ready = end;
    }

    pub const Part = struct { slot: usize, first: usize, tokens: []const u32 };
    pub const DraftItem = struct { slot: usize, anchor_tok: u32, anchor_pos: usize, src_row: usize };
    pub const LogitMode = cuda_sample.LogitMode;
    const fwd_impl = @import("cuda_forward.zig");

    pub fn sampleGpu(m: *Model, k: *const AllKernels, s: Stream, rules: []const cuda.sample.Rule, metas: []const i32, out: []u32) !void {
        return m.sample_sc.draw(m.d, &k.draws, s, m.tp_sess, if (m.prof) |*p| p else null, rules, metas, out);
    }
    fn fwd1(m: *Model, k: *const AllKernels, s: Stream, slot: usize, first: usize, tokens: []const u32, lr: usize, mode: LogitMode, dfly: bool, pr: bool) !void {
        return fwd_impl.forwardPartsInternal(m, k, s, &.{.{ .slot = slot, .first = first, .tokens = tokens }}, lr, mode, dfly, pr);
    }
    pub fn forward(m: *Model, k: *const AllKernels, s: Stream, slot: usize, first: usize, tokens: []const u32, lr: usize) !void {
        return m.fwd1(k, s, slot, first, tokens, lr, .full_f32, m.dfly_w != null and lr > 0, false);
    }
    pub fn forwardPrompt(m: *Model, k: *const AllKernels, s: Stream, slot: usize, first: usize, tokens: []const u32, lr: usize) !void {
        return m.fwd1(k, s, slot, first, tokens, lr, .full_f32, m.dfly_w != null and lr > 0, true);
    }
    pub fn forwardGreedy(m: *Model, k: *const AllKernels, s: Stream, slot: usize, first: usize, tokens: []const u32, lr: usize) !void {
        return m.fwd1(k, s, slot, first, tokens, lr, .greedy, false, false);
    }
    pub fn forwardSample(m: *Model, k: *const AllKernels, s: Stream, slot: usize, first: usize, tokens: []const u32, lr: usize) !void {
        return m.fwd1(k, s, slot, first, tokens, lr, .sample_bf16, false, false);
    }
    pub fn forwardParts(m: *Model, k: *const AllKernels, s: Stream, parts: []const Part, logits_rows: usize) !void {
        return fwd_impl.forwardPartsInternal(m, k, s, parts, logits_rows, .full_f32, m.dfly_w != null and logits_rows > 0, false);
    }
    pub fn forwardPartsMode(m: *Model, k: *const AllKernels, s: Stream, parts: []const Part, logits_rows: usize, greedy_only: bool) !void {
        return fwd_impl.forwardPartsInternal(m, k, s, parts, logits_rows, if (greedy_only) .greedy else .full_f32, m.dfly_w != null and logits_rows > 0, false);
    }
    pub fn forwardPartsLogits(m: *Model, k: *const AllKernels, s: Stream, parts: []const Part, logits_rows: usize, logit_mode: LogitMode, save_dfly: bool) !void {
        return fwd_impl.forwardPartsInternal(m, k, s, parts, logits_rows, logit_mode, save_dfly, false);
    }
    pub const draftDfly = fwd_impl.draftDfly;
};

test "cuda_model declarations compile on host" {
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest;
    std.testing.refAllDecls(@This());
}
