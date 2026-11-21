//! Forward pass, 42-layer CUDA Graph span capture/replay, and DFly2 proposal dispatch for `hy4.cuda_model.Model`.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const cuda_head = @import("cuda_head.zig");
const cuda_cache = @import("cuda_cache.zig");
const cuda_layer = @import("cuda_layer.zig");
const cuda_index = @import("cuda_index.zig");
const cuda_engram = @import("cuda_engram.zig");
const cuda_dfly = @import("cuda_dfly.zig");
const cuda_ihc = @import("cuda_ihc.zig");
const cuda_prof = @import("cuda_prof.zig");
const cm = @import("cuda_model.zig");

const Stream = cuda.Stream;
const Model = cm.Model;
const Seq = cm.Seq;
const AllKernels = cm.AllKernels;
const Part = Model.Part;
const DraftItem = Model.DraftItem;
const SwaSpan = Model.SwaSpan;

const StepCtx = struct {
    m: *Model,
    k: *const AllKernels,
    sqs: []const *Seq,
    eng: []const cuda_engram.Window,
    save_dfly: bool = false,
    save_last_only: bool = false,
    seg_row0: [cuda_layer.max_segs]usize = @splat(0),
    total_rows: usize = 0,
    any_delta: bool = false,
};

fn reduceHook(ctx: *anyopaque, s: Stream, send: u64, recv: u64, count: usize) anyerror!void {
    const sc: *const StepCtx = @ptrCast(@alignCast(ctx));
    return sc.m.tp_sess.?.sumRankOrder(s, send, sc.m.gather_buf.ptr, recv, count);
}

fn gatherBf16Hook(ctx: *anyopaque, s: Stream, send: u64, recv: u64, count_bf16: usize) anyerror!void {
    const sc: *const StepCtx = @ptrCast(@alignCast(ctx));
    return sc.m.tp_sess.?.gatherBf16(s, send, recv, count_bf16);
}

fn exchangeF32Hook(ctx: *anyopaque, s: Stream, send: u64, recv: u64, count_f32: usize) anyerror!void {
    const sc: *const StepCtx = @ptrCast(@alignCast(ctx));
    return sc.m.tp_sess.?.exchangeF32(s, send, recv, count_f32);
}

fn engramHook(ctx: *anyopaque, call: cuda_layer.EngramCall) anyerror!void {
    const sc: *const StepCtx = @ptrCast(@alignCast(ctx));
    const m = sc.m;
    const w = m.engram_weights[call.slot] orelse return error.MissingEngramLayer;
    const r = if (m.engram_reader) |*p| p else return error.MissingEngramReader;
    const esc = if (m.engram_sc) |*p| p else return error.MissingEngramScratch;
    const t0 = if (m.prof != null) cuda_prof.nowNs() else 0;
    try cuda_engram.partialWindows(&sc.k.engram, call.s, m.c, w, r, esc, sc.eng, m.engram_partial.ptr);
    if (m.prof) |*p| {
        p.addEngCpuNs(cuda_prof.nowNs() - t0);
        p.mark(call.s, .eng_gpu);
    }
    const sum_ptr = if (m.tp_sess) |ts| blk: {
        const count = call.rows * cuda_engram.projections * m.c.hidden;
        try ts.sumRankOrder(call.s, m.engram_partial.ptr, m.gather_buf.ptr, m.engram_reduced.ptr, count);
        if (m.prof) |*p| p.mark(call.s, .eng_comm);
        break :blk m.engram_reduced.ptr;
    } else m.engram_partial.ptr;
    try cuda_engram.finish(&sc.k.engram, call.s, m.c, w, esc, call.rows, null, call.streams_in, sum_ptr, m.engram_out.ptr, call.streams_out);
    if (m.prof) |*p| p.mark(call.s, .eng_gpu);
}

fn indexerPrepHook(ctx: *anyopaque, layer_idx: usize, s: Stream, attn_in: u64, q_resid: u64, rows: usize) anyerror!void {
    const sc: *const StepCtx = @ptrCast(@alignCast(ctx));
    const m = sc.m;
    const iw = &(m.index_weights[layer_idx] orelse return error.MissingIndexWeights);
    const isc = &(m.idx_sc orelse return error.MissingIndexScratch);
    try cuda_index.preproject(sc.k.idxOps(), s, iw, isc, attn_in, q_resid, rows);
}

fn indexerHook(ctx: *anyopaque, call: cuda_layer.IndexerCall) anyerror!?cuda_layer.Recall {
    const sc: *const StepCtx = @ptrCast(@alignCast(ctx));
    const m = sc.m;
    const ord = m.indexed_ord[call.layer_idx] orelse return null;
    const iw = if (m.index_weights[call.layer_idx]) |*p| p else return error.MissingIndexWeights;
    const isc = if (m.idx_sc) |*p| p else return error.MissingIndexScratch;
    if (call.seg >= sc.sqs.len) return error.BadStepShape;
    const sq = sc.sqs[call.seg];
    var q_pos3: u64 = 0;
    var q_pos3_stride: usize = 0;
    var sm_mcs: u64 = 0;
    var sm_plen: usize = 0;
    if (sq.media) |*sm| {
        if (sc.save_last_only) {
            sm_mcs = sm.mrope_cs.ptr;
            sm_plen = sm.prompt_len;
            if (call.first < sm.prompt_len) {
                q_pos3 = sm.pos3.ptr + call.first * 4;
                q_pos3_stride = sm.prompt_len;
            }
        }
    }
    if (q_pos3 == 0 and sc.any_delta) {
        q_pos3 = m.pos3_decode_dev.ptr + sc.seg_row0[call.seg] * 4;
        q_pos3_stride = sc.total_rows;
    }
    const icall: cuda_index.Call = .{
        .x = call.attn_in,
        .q_resid = call.q_resid,
        .rows = call.rows,
        .first = call.first,
        .pos = call.pos,
        .pos3 = q_pos3,
        .pos3_stride = q_pos3_stride,
        .recall_from = call.recall_limit,
        .idx_out = call.idx_out,
        .preprojected = call.preprojected,
        .row_off = call.row0,
        .page_table = call.page_table,
        .pt_stride = call.pt_stride,
    };
    if (sq.cache.fmt.tail_index_keys) {
        const gk_ptr = if (m.pages) |pg| pg.gk_bases[ord] else sq.groupKeysAt(ord);
        try cuda_index.runTail(sc.k.idxOps(), m.d, call.s, iw, call.index_keys, gk_ptr, sq.cap, m.rope_cs.?.ptr, m.rope_cs.?.ptr, sq.mrope_delta, sm_mcs, sm_plen, isc, icall);
    } else {
        var view_cache: cuda_index.Cache = .{
            .cap = sq.cap,
            .token_keys = .{ .d = m.d, .ptr = call.index_keys, .len = sq.cap * cuda_index.dim * 2 },
            .group_keys = .{ .d = m.d, .ptr = sq.groupKeysAt(ord), .len = (sq.cap / cuda_index.ratio) * cuda_index.dim * 2 },
            .rope_cs = m.rope_cs.?,
            .rope_ready = m.rope_ready,
        };
        try cuda_index.run(sc.k.idxOps(), call.s, iw, &view_cache, isc, icall);
    }
    const sel = cuda_index.selectRange(call.first, call.rows, null, call.recall_limit);
    if (sel.count == 0) return null;
    if (sc.save_dfly and m.dfly_w != null) {
        inline for (cuda_dfly.target_kv_layers, 0..) |tl, dl| {
            if (call.layer_idx == tl) {
                const row_b = cuda_dfly.target_topk * 4;
                if (!sc.save_last_only and call.rows <= cm.max_window) {
                    try sq.dfly_topk.copyFrom((dl * cm.max_window + sel.from) * row_b, call.idx_out, sel.count * row_b, call.s.handle);
                } else {
                    try sq.dfly_topk.copyFrom((dl * cm.max_window) * row_b, call.idx_out + (sel.count - 1) * row_b, row_b, call.s.handle);
                }
            }
        }
    }
    return .{ .list_from = sel.from, .idx = call.idx_out, .nidx = call.recall_limit };
}

fn seqLayerView(m: *const Model, slot: usize, sq: *const Seq, l_idx: usize) !cuda_cache.LayerView {
    var v = try sq.cache.view(l_idx);
    if (v.kind == .indexed) if (m.pages) |pg| {
        v.mla = pg.mla_bases[m.indexed_ord[l_idx].?];
        v.page_table = pg.slotTablePtr(slot);
        v.pt_stride = pg.pt_stride;
    };
    return v;
}

fn runSwaSpanBody(m: *Model, k: *const AllKernels, s: Stream, sp: SwaSpan, rows: usize, cur_streams: u64, skip_first: bool, ar: ?cuda_layer.AllReduce) !void {
    const dummy_cache: cuda_cache.LayerView = .{ .kind = .sliding, .mla = 0, .capacity = cm.ring_slots, .window = m.c.window, .ring = cm.ring_slots, .kv8 = m.kv8 };
    const has_engram_reader = m.engram_reader != null;
    var skip_entry = skip_first;
    if (sp.tail_layer) |tl| {
        var next_b: cuda_layer.NextBoundary = .post_only;
        var next_skip = false;
        if (sp.fuse_head) {
            next_b = .{ .head = .{ .mix = m.head.top.mix(), .norm = m.head.top.norm.ptr, .out = m.head_sc.hidden.ptr } };
        } else if (tl + 1 < m.active_layers and !(m.layers[tl + 1].has_engram and has_engram_reader)) {
            next_b = .{ .attn = .{ .mix = m.layers[tl + 1].hc_attn, .norm = m.layers[tl + 1].input_norm } };
            next_skip = true;
        }
        try m.layers[tl].forwardPostAttn(&k.layer, s, m.c, &m.layer_sc, cur_streams, .{
            .streams_in = cur_streams,
            .streams_out = cur_streams,
            .rows = rows,
            .first = 0,
            .pos = m.pos_dev.ptr,
            .cache = dummy_cache,
            .all_reduce = ar,
            .next_boundary = next_b,
        });
        skip_entry = next_skip;
    }
    for (sp.swa_lo..sp.swa_hi) |l_idx| {
        var next_b: cuda_layer.NextBoundary = .post_only;
        var next_skip = false;
        const next_eng = m.layers[l_idx + 1].has_engram and has_engram_reader;
        if (!next_eng) {
            next_b = .{ .attn = .{ .mix = m.layers[l_idx + 1].hc_attn, .norm = m.layers[l_idx + 1].input_norm } };
            next_skip = true;
        }
        try m.layers[l_idx].forward(&k.layer, s, m.c, &m.layer_sc, .{
            .streams_in = cur_streams,
            .streams_out = cur_streams,
            .rows = rows,
            .first = 0,
            .pos = m.pos_dev.ptr,
            .cache = dummy_cache,
            .segs = &.{},
            .caches = m.row_caches.ptr + l_idx * cm.max_batch * 8,
            .engram = null,
            .indexer = null,
            .all_reduce = ar,
            .skip_entry = skip_entry,
            .next_boundary = next_b,
            .engram_dst = 0,
            .prof = null,
        });
        skip_entry = next_skip;
    }
}

fn captureSwaSpan(m: *Model, k: *const AllKernels, s: Stream, sp: SwaSpan, rows: usize, cur_streams: u64, skip_first: bool, ar: ?cuda_layer.AllReduce) !cuda.graph.Exec {
    try cuda.graph.beginCapture(s, .thread_local);
    const body_err: ?anyerror = blk: {
        runSwaSpanBody(m, k, s, sp, rows, cur_streams, skip_first, ar) catch |e| break :blk e;
        break :blk null;
    };
    var g = try cuda.graph.endCapture(s);
    defer g.deinit();
    if (body_err) |e| return e;
    var exec = try g.instantiate();
    errdefer exec.deinit();
    try exec.upload(s);
    return exec;
}

fn runSwaSpan(m: *Model, k: *const AllKernels, s: Stream, si: usize, rows: usize, cur_streams: u64, skip_first: bool, ar: ?cuda_layer.AllReduce) !void {
    const sp = m.swa_spans[si];
    const slot = &m.swa_graphs[si][rows];
    switch (slot.state) {
        .ready => return slot.exec.launchOn(s),
        .warm => {
            if (captureSwaSpan(m, k, s, sp, rows, cur_streams, skip_first, ar)) |exec| {
                slot.exec = exec;
                slot.state = .ready;
                return slot.exec.launchOn(s);
            } else |err| {
                std.log.warn("[tensorfold] hy4 graph span capture (span={d} tail={?} [{d}..{d}), R={d}) failed: {s}; using eager", .{ si, sp.tail_layer, sp.swa_lo, sp.swa_hi, rows, @errorName(err) });
                slot.state = .failed;
            }
        },
        .cold => {
            try runSwaSpanBody(m, k, s, sp, rows, cur_streams, skip_first, ar);
            slot.state = .warm;
            return;
        },
        .failed => {},
    }
    try runSwaSpanBody(m, k, s, sp, rows, cur_streams, skip_first, ar);
}

fn prefetchEngramSlot(m: *Model, eng: []const cuda_engram.Window, slot: usize, prof_ptr: ?*cuda_prof.Prof) !void {
    if (slot >= config.engram_layers) return;
    const r = if (m.engram_reader) |*rd| rd else return;
    const w = m.engram_weights[slot] orelse return;
    const esc = if (m.engram_sc) |*p| p else return;
    const t_pf = if (prof_ptr != null) cuda_prof.nowNs() else 0;
    try cuda_engram.prefetchWindows(m.c, w, r, esc, eng);
    if (prof_ptr) |p| p.addEngCpuNs(cuda_prof.nowNs() - t_pf);
}

fn prefetchEngramAfter(m: *Model, eng: []const cuda_engram.Window, done_slot: usize, prompt_pass: bool, prof_ptr: ?*cuda_prof.Prof) !void {
    if (prompt_pass) return prefetchEngramSlot(m, eng, done_slot + 1, prof_ptr);
    if (done_slot == 0) {
        for (1..config.engram_layers) |sl| try prefetchEngramSlot(m, eng, sl, prof_ptr);
    }
}

pub fn forwardPartsInternal(m: *Model, k: *const AllKernels, s: Stream, parts: []const Part, logits_rows: usize, logit_mode: Model.LogitMode, save_dfly: bool, prompt_pass: bool) !void {
    if (parts.len == 0 or parts.len > cuda_layer.max_segs) return error.BadStepShape;
    var sqs: [cuda_layer.max_segs]*Seq = undefined;
    var eng: [cuda_layer.max_segs]cuda_engram.Window = undefined;
    const row_pos = m.pos_host[cm.pos_windows..];
    var rows: usize = 0;
    var wcur: usize = 0;
    var end: usize = 0;
    for (parts) |p| rows += p.tokens.len;
    const prof_ptr: ?*cuda_prof.Prof = if (m.prof) |*p| p else null;
    if (prof_ptr) |p| p.begin(s, .{ .kind = .target, .rows = rows, .parts = parts.len, .mode = @backingInt(logit_mode) });
    const total_rows = rows;
    rows = 0;
    var any_delta = false;
    var seg_row0: [cuda_layer.max_segs]usize = @splat(0);
    for (parts, 0..) |p, i| {
        const n = p.tokens.len;
        if (n == 0 or rows + n > cm.prompt_rows or p.slot >= cm.max_slots) return error.BadStepShape;
        for (parts[0..i]) |q| if (q.slot == p.slot) return error.BadStepShape;
        const sq: *Seq = if (m.seqs[p.slot]) |*q| q else return error.UnknownSlot;
        if (p.first + n > sq.cap) return error.CacheOverflow;
        seg_row0[i] = rows;
        if (sq.mrope_delta != 0) any_delta = true;
        @memcpy(sq.history[p.first..][0..n], p.tokens);
        const skip = @min(p.first, config.engram_orders);
        const win_start = p.first - skip;
        const wpos = m.pos_host[wcur..][0 .. skip + n];
        for (wpos, 0..) |*x, j| x.* = @intCast(win_start + j);
        for (row_pos[rows..][0..n], 0..) |*x, j| {
            const lp: i32 = @intCast(p.first + j);
            x.* = @intCast(lp);
            const rp = lp + sq.mrope_delta;
            m.rot_pos_host[rows + j] = rp;
            m.rot_pos_host[total_rows + rows + j] = rp;
            m.rot_pos_host[2 * total_rows + rows + j] = rp;
        }
        try cuda.copy_source.enqueue(m.d, m.ids_dev.ptr + rows * 4, std.mem.sliceAsBytes(p.tokens), s);
        sqs[i] = sq;
        eng[i] = .{ .tokens = sq.history[win_start..][0 .. skip + n], .positions = wpos, .skip = skip };
        wcur += skip + n;
        rows += n;
        const rot_end: usize = @intCast(@max(@as(isize, @intCast(p.first + n)) + sq.mrope_delta, @as(isize, @intCast(p.first + n))));
        end = @max(end, rot_end);
    }
    if (logits_rows > rows or logits_rows > cm.max_batch) return error.BadStepShape;
    try cuda.copy_source.enqueue(m.d, m.pos_dev.ptr, std.mem.sliceAsBytes(row_pos[0..rows]), s);
    if (any_delta) {
        try cuda.copy_source.enqueue(m.d, m.rot_pos_dev.ptr, std.mem.sliceAsBytes(m.rot_pos_host[0..rows]), s);
        try cuda.copy_source.enqueue(m.d, m.pos3_decode_dev.ptr, std.mem.sliceAsBytes(m.rot_pos_host[0 .. 3 * rows]), s);
    }
    var prompt_pos3: u64 = 0;
    var prompt_ld3: usize = 0;
    if (sqs[0].media) |*sm| {
        if (prompt_pass and parts.len == 1 and parts[0].first < sm.prompt_len) {
            prompt_pos3 = sm.pos3.ptr + parts[0].first * 4;
            prompt_ld3 = sm.prompt_len;
            var max_p: i32 = 0;
            for (sm.pos3_host) |pv| max_p = @max(max_p, pv);
            end = @max(end, @as(usize, @intCast(max_p + 1)));
        }
    }
    try m.ensureRope(k, s, end);

    try m.head.embed.expand(k.layer.headOps(), s, m.ids_dev.ptr, m.streams_buf.ptr, rows);
    if (sqs[0].media) |*sm| {
        if (prompt_pass and parts.len == 1) {
            try sm.spliceChunk(&k.layer.mlp.rows, s, m.streams_buf.ptr, parts[0].first, rows, m.c.hidden);
        }
    }
    var step_ctx: StepCtx = .{
        .m = m,
        .k = k,
        .sqs = sqs[0..parts.len],
        .eng = eng[0..parts.len],
        .save_dfly = save_dfly,
        .save_last_only = prompt_pass,
        .seg_row0 = seg_row0,
        .total_rows = rows,
        .any_delta = any_delta,
    };
    const ar: ?cuda_layer.AllReduce = if (m.tp_sess) |ts| .{
        .ctx = &step_ctx,
        .rank = @intCast(ts.rank),
        .sum = reduceHook,
        .gather_bf16 = gatherBf16Hook,
        .exchange_f32 = exchangeF32Hook,
    } else null;
    const eh: ?cuda_layer.EngramHook = if (m.engram_reader != null) .{ .ctx = &step_ctx, .run = engramHook } else null;
    const ih: ?cuda_layer.IndexerHook = if (m.n_indexed > 0) .{ .ctx = &step_ctx, .run = indexerHook, .prep = indexerPrepHook } else null;

    var segs: [cuda_layer.max_segs]cuda_layer.Seg = undefined;
    const tabled = rows <= cm.max_batch;
    if (tabled) try @import("cuda_pages.zig").uploadStepTables(m, s, parts, sqs[0..parts.len], rows);
    const rstream_ptr: u64 = if (tabled) (if (m.pages) |pg| pg.row_slots_dev.ptr else 0) else 0;
    const pt_base_ptr: u64 = if (tabled) (if (m.pages) |pg| pg.pt_dev.ptr else 0) else 0;
    if (prof_ptr) |p| p.mark(s, .embed);
    // Issue Engram slot 0 right after embed (non-blocking bulk pool in prefill overlaps full Layer 0 GPU time).
    try prefetchEngramSlot(m, eng[0..parts.len], 0, prof_ptr);
    var cur_streams = m.streams_buf.ptr;
    var alt_streams = m.layer_sc.engram_buf.ptr;
    var skip_entry = false;
    const fuse_head = logits_rows > 0 and logits_rows == rows and m.c.hidden == cuda_ihc.hidden;
    var l_idx: usize = 0;
    while (l_idx < m.active_layers) {
        if (tabled and !prompt_pass and !any_delta and prompt_pos3 == 0 and m.swa_graphs_enabled and prof_ptr == null) {
            if (m.swa_span_at[l_idx]) |si| {
                const sp = m.swa_spans[si];
                if (!sp.fuse_head or fuse_head) {
                    if (sp.tail_layer) |tl| {
                        var row0: usize = 0;
                        for (parts, sqs[0..parts.len], segs[0..parts.len]) |p, sq, *g| {
                            g.* = .{ .row0 = row0, .rows = p.tokens.len, .first = p.first, .cache = try seqLayerView(m, p.slot, sq, tl) };
                            row0 += p.tokens.len;
                        }
                        const has_eng = m.layers[tl].has_engram and eh != null;
                        const out_streams = if (has_eng) alt_streams else cur_streams;
                        _ = try m.layers[tl].forwardPreAttn(&k.layer, s, m.c, &m.layer_sc, .{
                            .streams_in = cur_streams,
                            .streams_out = out_streams,
                            .rows = rows,
                            .first = parts[0].first,
                            .pos = m.pos_dev.ptr,
                            .cache = segs[0].cache,
                            .segs = if (parts.len > 1) segs[0..parts.len] else &.{},
                            .caches = m.row_caches.ptr + tl * cm.max_batch * 8,
                            .row_stream = rstream_ptr,
                            .pt_base = pt_base_ptr,
                            .engram = eh,
                            .indexer = ih,
                            .all_reduce = ar,
                            .skip_entry = skip_entry,
                            .engram_dst = if (has_eng) alt_streams else 0,
                        });
                        if (has_eng) {
                            const tmp = cur_streams;
                            cur_streams = alt_streams;
                            alt_streams = tmp;
                            if (m.layers[tl].engram_slot) |done_slot| {
                                try prefetchEngramAfter(m, eng[0..parts.len], done_slot, prompt_pass, prof_ptr);
                            }
                        }
                        try runSwaSpan(m, k, s, si, rows, cur_streams, true, ar);
                    } else {
                        const has_eng = m.layers[l_idx].has_engram and eh != null;
                        if (has_eng) {
                            _ = try m.layers[l_idx].runEngramOnly(&m.layer_sc, s, .{
                                .streams_in = cur_streams,
                                .streams_out = alt_streams,
                                .rows = rows,
                                .first = parts[0].first,
                                .pos = m.pos_dev.ptr,
                                .cache = try seqLayerView(m, parts[0].slot, sqs[0], l_idx),
                                .engram = eh,
                                .all_reduce = ar,
                                .engram_dst = alt_streams,
                            });
                            const tmp = cur_streams;
                            cur_streams = alt_streams;
                            alt_streams = tmp;
                            if (m.layers[l_idx].engram_slot) |done_slot| {
                                try prefetchEngramAfter(m, eng[0..parts.len], done_slot, prompt_pass, prof_ptr);
                            }
                        }
                        try runSwaSpan(m, k, s, si, rows, cur_streams, skip_entry, ar);
                    }
                    l_idx = sp.swa_hi;
                    skip_entry = if (sp.swa_hi < m.active_layers) !(m.layers[sp.swa_hi].has_engram and eh != null) else false;
                    continue;
                }
            }
        }
        var row0: usize = 0;
        for (parts, sqs[0..parts.len], segs[0..parts.len]) |p, sq, *g| {
            g.* = .{ .row0 = row0, .rows = p.tokens.len, .first = p.first, .cache = try seqLayerView(m, p.slot, sq, l_idx) };
            row0 += p.tokens.len;
        }
        const has_eng = m.layers[l_idx].has_engram and eh != null;
        const out_streams = if (has_eng) alt_streams else cur_streams;
        var next_b: cuda_layer.NextBoundary = .post_only;
        var next_skip = false;
        if (l_idx + 1 < m.active_layers) {
            const next_eng = m.layers[l_idx + 1].has_engram and eh != null;
            if (!next_eng) {
                next_b = .{ .attn = .{ .mix = m.layers[l_idx + 1].hc_attn, .norm = m.layers[l_idx + 1].input_norm } };
                next_skip = true;
            }
        } else if (fuse_head) {
            next_b = .{ .head = .{ .mix = m.head.top.mix(), .norm = m.head.top.norm.ptr, .out = m.head_sc.hidden.ptr } };
        }
        try m.layers[l_idx].forward(&k.layer, s, m.c, &m.layer_sc, .{
            .streams_in = cur_streams,
            .streams_out = out_streams,
            .rows = rows,
            .first = parts[0].first,
            .pos = m.pos_dev.ptr,
            .rot_pos = if (any_delta and prompt_pos3 == 0) m.rot_pos_dev.ptr else 0,
            .pos3 = prompt_pos3,
            .ld3 = prompt_ld3,
            .cache = segs[0].cache,
            .segs = if (parts.len > 1) segs[0..parts.len] else &.{},
            .caches = if (tabled) m.row_caches.ptr + l_idx * cm.max_batch * 8 else 0,
            .row_stream = if (m.layers[l_idx].is_sliding) 0 else rstream_ptr,
            .pt_base = if (m.layers[l_idx].is_sliding) 0 else pt_base_ptr,
            .engram = eh,
            .indexer = ih,
            .all_reduce = ar,
            .skip_entry = skip_entry,
            .next_boundary = next_b,
            .engram_dst = if (has_eng) alt_streams else 0,
            .single_split_attn = prompt_pass,
            .prof = prof_ptr,
        });
        if (has_eng) {
            const tmp = cur_streams;
            cur_streams = alt_streams;
            alt_streams = tmp;
        }
        skip_entry = next_skip;
        if (m.layers[l_idx].engram_slot) |done_slot| {
            try prefetchEngramAfter(m, eng[0..parts.len], done_slot, prompt_pass, prof_ptr);
        }
        l_idx += 1;
    }
    if (logits_rows > 0) {
        const vocab_local = m.head.top.share.len;
        if (fuse_head) {
            try m.head.top.logitsFromHidden(k.layer.headOps(), s, m.head_sc.hidden.ptr, m.logits_local.ptr, vocab_local, logits_rows);
            if (prof_ptr) |p| p.mark(s, .lmhead);
        } else {
            const row_off = rows - logits_rows;
            const st_in = cur_streams + row_off * cuda_head.streams * m.c.hidden * 2;
            if (m.c.hidden == cuda_ihc.hidden) {
                try k.layer.ihc.run(s, .{
                    .mode = .head_only,
                    .rows = logits_rows,
                    .x = st_in,
                    .mix = m.head.top.mix(),
                    .next_nw = m.head.top.norm.ptr,
                    .a = m.head_sc.hidden.ptr,
                    .eps = m.c.eps,
                    .hc_eps = m.c.hc_eps,
                    .mag = m.c.hc_magnitude,
                });
                if (prof_ptr) |p| p.mark(s, .ihc);
                try m.head.top.logitsFromHidden(k.layer.headOps(), s, m.head_sc.hidden.ptr, m.logits_local.ptr, vocab_local, logits_rows);
                if (prof_ptr) |p| p.mark(s, .lmhead);
            } else {
                try m.head.top.forward(k.layer.headOps(), s, &m.head_sc, st_in, null, m.logits_local.ptr, vocab_local, logits_rows);
                if (prof_ptr) |p| p.mark(s, .lmhead);
            }
        }
        if (save_dfly and m.dfly_w != null) {
            const row_b = m.c.hidden * 2;
            if (logits_rows == rows) {
                var r0: usize = 0;
                for (parts, sqs[0..parts.len]) |p, sq| {
                    const n: usize = @min(p.tokens.len, cm.max_window);
                    try sq.last_hidden.copyFrom(0, m.head_sc.hidden.ptr + r0 * row_b, n * row_b, s.handle);
                    r0 += p.tokens.len;
                }
            } else if (logits_rows == 1 and parts.len == 1) {
                try sqs[0].last_hidden.copyFrom(0, m.head_sc.hidden.ptr, row_b, s.handle);
            }
            if (prof_ptr) |p| p.mark(s, .save_hid);
        }
        switch (logit_mode) {
            .greedy => {
                try m.head.top.rows.argmax(s, m.logits_local.ptr, vocab_local, m.head.top.share.lo, logits_rows, m.argmax_local.ptr);
                if (prof_ptr) |p| p.mark(s, .argmax);
                if (m.tp_sess) |ts| {
                    try ts.gatherArgmax(s, m.argmax_local.ptr, m.argmax_full.?.ptr, logits_rows);
                    if (prof_ptr) |p| p.mark(s, .head_comm);
                }
            },
            .sample_bf16 => try m.sample_sc.prepare(m.d, &m.head.top.rows, s, m.logits_local.ptr, vocab_local, logits_rows, m.tp_sess, m.at.rank, prof_ptr),
            .full_f32 => if (m.tp_sess) |ts| {
                try ts.gatherLogits(s, m.logits_local.ptr, vocab_local, m.logits_full.?.ptr, logits_rows);
                if (prof_ptr) |p| p.mark(s, .head_comm);
            },
        }
    }
    if (prof_ptr) |p| {
        p.endSubmit();
        if (logits_rows == 0) p.finish(s);
    }
    for (parts, sqs[0..parts.len]) |p, sq| sq.len = p.first + p.tokens.len;
}

pub const draftDfly = @import("cuda_dfly_run.zig").draftModel;

pub const initSwaSpans = @import("cuda_dfly_run.zig").initSwaSpans;
