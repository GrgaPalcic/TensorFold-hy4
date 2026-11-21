//! Batched DFly2 5-layer speculative proposal execution and per-(B, K) CUDA Graph capture.
const std = @import("std");
const cuda = @import("cuda");
const cuda_head = @import("cuda_head.zig");
const cuda_mla_c1 = @import("cuda_mla_c1.zig");
const cuda_mla_c2 = @import("cuda_mla_c2.zig");
const cuda_prof = @import("cuda_prof.zig");
const cuda_tp = @import("cuda_tp.zig");
const dfly = @import("cuda_dfly.zig");

const Stream = cuda.Stream;
const qmmf = cuda.qmmf;

pub const Req = struct {
    anchor_tok: u32,
    anchor_pos: usize,
    target_hidden_ptr: u64, // bf16 [3072]
    kv_caches: [dfly.num_layers]u64, // per draft layer: target layer [25,29,33,37,41] MLA cache base ptr
    topk_ptrs: [dfly.num_layers]u64, // i32 [2048] saved target topk per draft layer (`anchor_pos >= 2048`)
    mrope_delta: i32 = 0,
    slot: usize = 0,
    page_table: u64 = 0,
    pt_stride: usize = 0,
};

pub const Runtime = struct {
    d: *const cuda.Driver,
    k: *const dfly.Kernels,
    lane: *const qmmf.Lane,
    norms: *const cuda.norms.Norms,
    act: *const cuda.act_rows.ActRows,
    mla_rope: *const cuda.mla.Mla,
    c1: *const cuda_mla_c1.C1,
    c2: *const cuda_mla_c2.C2,
    head: *const cuda_head.Head,
    head_ops: cuda_head.Ops,
    logits_local: u64,
    argmax_local: u64,
    argmax_out: u64,
    tp_sess: ?*cuda_tp.Session,
    gather_buf: u64,
    world: usize,
    vocab: usize,
    eps: f32,
    prof: ?*cuda_prof.Prof = null,
    kv8: bool = false,
    pt_dev: u64 = 0,
    pt_stride: usize = 0,
    mla_bases: [dfly.num_layers]u64 = @splat(0),
};

fn reduce(rt: Runtime, s: Stream, src: u64, dst: u64, count: usize) !u64 {
    if (rt.tp_sess) |ts| {
        try ts.sumRankOrder(s, src, rt.gather_buf, dst, count);
        return dst;
    }
    return src;
}

pub const AttnMode = enum(u8) { dense, seam, sparse };

/// Classifies DFly2 attention at `anchor_pos`: dense (`anchor+qw <= 2048`), seam (`2043..2048`), sparse (`>= 2049`).
pub fn attnMode(anchor_pos: usize) AttnMode {
    if (anchor_pos + dfly.qw <= dfly.target_topk) return .dense;
    if (anchor_pos > dfly.target_topk) return .sparse;
    return .seam;
}

/// Runs one batched DFly2 draft round (`B = 1..max_reqs`, `depth = 1..5`), writing `u32 [B, depth]` to `draft_out`.
pub fn propose(rt: Runtime, s: Stream, w: *const dfly.Weights, sc: *dfly.Scratch, reqs: []const Req, depth: usize) !void {
    const b = reqs.len;
    if (b == 0 or b > dfly.max_reqs or depth == 0 or depth > dfly.max_drafts) return error.BadDraftShape;
    const n = b * dfly.qw;

    var ids_h: [dfly.max_reqs * dfly.qw]i32 = undefined;
    var pos_h: [dfly.max_reqs * dfly.qw]i32 = undefined;
    var rot_h: [dfly.max_reqs * dfly.qw]i32 = undefined;
    var prev_h: [dfly.max_reqs]i32 = undefined;
    var ptrs_h: [dfly.max_reqs]u64 = undefined;
    var ctab_h: [dfly.num_layers * dfly.max_reqs * dfly.qw]u64 = undefined;
    var slots_h: [dfly.max_reqs * dfly.qw]i32 = undefined;
    var all_dense = true;
    var all_sparse = true;
    for (reqs, 0..) |rq, bi| {
        ptrs_h[bi] = rq.target_hidden_ptr;
        prev_h[bi] = @intCast(rq.anchor_tok);
        const am = attnMode(rq.anchor_pos);
        if (am != .dense) all_dense = false;
        if (am != .sparse) all_sparse = false;
        ids_h[bi * dfly.qw] = @intCast(rq.anchor_tok);
        pos_h[bi * dfly.qw] = @intCast(rq.anchor_pos);
        rot_h[bi * dfly.qw] = @as(i32, @intCast(rq.anchor_pos)) + rq.mrope_delta;
        for (1..dfly.qw) |q| {
            ids_h[bi * dfly.qw + q] = @intCast(dfly.mask_token_id);
            pos_h[bi * dfly.qw + q] = @intCast(rq.anchor_pos + q);
            rot_h[bi * dfly.qw + q] = @as(i32, @intCast(rq.anchor_pos + q)) + rq.mrope_delta;
        }
        for (0..dfly.qw) |q| slots_h[bi * dfly.qw + q] = @intCast(rq.slot);
        for (0..dfly.num_layers) |li| {
            for (0..dfly.qw) |q| ctab_h[li * n + bi * dfly.qw + q] = rq.kv_caches[li];
        }
    }
    try cuda.copy_source.enqueue(rt.d, sc.src_ptrs, std.mem.sliceAsBytes(ptrs_h[0..b]), s);
    try cuda.copy_source.enqueue(rt.d, sc.ids_blk, std.mem.sliceAsBytes(ids_h[0..n]), s);
    try cuda.copy_source.enqueue(rt.d, sc.pos_blk, std.mem.sliceAsBytes(pos_h[0..n]), s);
    try cuda.copy_source.enqueue(rt.d, sc.rot_pos_blk, std.mem.sliceAsBytes(rot_h[0..n]), s);
    try cuda.copy_source.enqueue(rt.d, sc.prev_ids, std.mem.sliceAsBytes(prev_h[0..b]), s);
    try cuda.copy_source.enqueue(rt.d, sc.row_caches, std.mem.sliceAsBytes(ctab_h[0 .. dfly.num_layers * n]), s);
    try cuda.copy_source.enqueue(rt.d, sc.row_slots, std.mem.sliceAsBytes(slots_h[0..n]), s);

    const can_dense_graph = all_dense and n <= 64;
    const can_sparse_graph = all_sparse and n <= 64 and b <= dfly.Scratch.max_graph_b;
    if (can_sparse_graph) {
        var meta_h: [3 * dfly.max_reqs]i32 = @splat(0);
        for (reqs, 0..) |rq, bi| {
            meta_h[bi] = 1;
            meta_h[dfly.max_reqs + bi] = @intCast(bi);
            meta_h[2 * dfly.max_reqs + bi] = @intCast(rq.anchor_pos);
            for (0..dfly.num_layers) |li| {
                const dst_off = (bi * dfly.num_layers + li) * dfly.target_topk * 4;
                try rt.d.check(rt.d.api.cuMemcpyDtoDAsync_v2(sc.src_topk.ptr + dst_off, rq.topk_ptrs[li], dfly.target_topk * 4, s.handle), "cuMemcpyDtoDAsync");
            }
        }
        try cuda.copy_source.enqueue(rt.d, sc.ext_meta.ptr, std.mem.sliceAsBytes(&meta_h), s);
    }
    if ((can_dense_graph or can_sparse_graph) and sc.graphs_enabled and b <= dfly.Scratch.max_graph_b and rt.prof == null) {
        const slot = if (can_sparse_graph) &sc.sparse_graphs[b][depth] else &sc.graphs[b][depth];
        switch (slot.state) {
            .ready => return slot.exec.launchOn(s),
            .warm => {
                if (captureGraph(rt, s, w, sc, b, depth, can_sparse_graph)) |exec| {
                    slot.exec = exec;
                    slot.state = .ready;
                    return slot.exec.launchOn(s);
                } else |err| {
                    std.log.warn("[tensorfold] hy4 dfly graph capture (sparse={}, B={d}, K={d}) failed: {s}; using eager", .{ can_sparse_graph, b, depth, @errorName(err) });
                    slot.state = .failed;
                }
            },
            .cold => {
                try proposeBody(rt, s, w, sc, b, depth, null, can_sparse_graph);
                slot.state = .warm;
                return;
            },
            .failed => {},
        }
    }
    try proposeBody(rt, s, w, sc, b, depth, if (can_dense_graph or can_sparse_graph) null else reqs, can_sparse_graph);
}

fn captureGraph(rt: Runtime, s: Stream, w: *const dfly.Weights, sc: *const dfly.Scratch, b: usize, depth: usize, sparse_graphed: bool) !cuda.graph.Exec {
    try cuda.graph.beginCapture(s, .thread_local);
    const body_err: ?anyerror = blk: {
        proposeBody(rt, s, w, sc, b, depth, null, sparse_graphed) catch |e| break :blk e;
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

/// Capturable DFly2 body: dense (`sparse_graphed == false`) or sparse (`sparse_graphed == true`, `anchor >= 2048`).
fn proposeBody(rt: Runtime, s: Stream, w: *const dfly.Weights, sc: *const dfly.Scratch, b: usize, depth: usize, sparse_reqs: ?[]const Req, sparse_graphed: bool) !void {
    const n = b * dfly.qw;
    const hidden = dfly.hidden;
    const q_lora = dfly.q_lora;
    const kv_lora = dfly.kv_lora;
    const nope = dfly.nope;
    const rope = dfly.rope;
    const v_width = dfly.v_width;
    const mla_row = dfly.mla_row;

    try rt.k.runGatherHidden(s, sc.src_ptrs, sc.hidden_in, b);
    try rt.k.runDual(
        s,
        .{ .base = rt.head.embed.w.ptr, .ids = sc.prev_ids, .ld = hidden, .ids_stride = 1, .nrows = @intCast(rt.vocab) },
        w.enorm,
        .{ .base = sc.hidden_in, .ld = hidden },
        w.hnorm,
        rt.eps,
        sc.cat,
        2 * hidden,
        b,
    );
    try rt.lane.matmulF32(s, sc.cat, 2 * hidden, b, w.eh_proj, sc.eh_f32, sc.lane_part);
    try rt.k.runFusion(
        s,
        .{ .base = rt.head.embed.w.ptr, .ids = sc.ids_blk, .ld = hidden, .ids_stride = 1, .nrows = @intCast(rt.vocab) },
        sc.eh_f32,
        w.gate,
        w.layers[0].input_norm,
        rt.eps,
        sc.res,
        sc.normed,
        n,
    );
    if (rt.prof) |p| p.mark(s, .dfly_fuse);

    if (sparse_graphed) {
        try rt.k.runExtendTopk(s, .{
            .mode = 0,
            .src = sc.src_topk.ptr,
            .src_slot_stride = dfly.target_topk,
            .src_row_stride = dfly.num_layers * dfly.target_topk,
            .src_rows_cap = @intCast(b),
            .dense_below = dfly.target_topk,
            .layer_slot = sc.ext_layer_slots.ptr,
            .num_layers = dfly.num_layers,
            .active = sc.ext_meta.ptr,
            .source_rows = sc.ext_meta.ptr + dfly.max_reqs * 4,
            .anchors = sc.ext_meta.ptr + 2 * dfly.max_reqs * 4,
            .b = @intCast(b),
            .qw = dfly.qw,
            .target_topk = dfly.target_topk,
            .local_window = 0,
            .k_pad = dfly.topk_pad,
            .out = sc.ext_topk.ptr,
            .out_layer_stride = @intCast(n * dfly.topk_pad),
        });
    }

    const qkv_stride = q_lora + mla_row;
    const h = w.layers[0].heads;
    const qs = h * (nope + rope);
    const vs = h * v_width;
    const qlo: usize = if (rt.c1.exact) n * h * mla_row else 0;

    for (0..dfly.num_layers) |li| {
        const ly = w.layers[li];
        const layer_caches = sc.row_caches + li * n * 8;
        try rt.lane.matmul(s, sc.normed, hidden, n, ly.qkv_a, sc.qkv, sc.lane_part);
        try rt.norms.rmsParts(s, sc.qkv, qkv_stride, 0, ly.q_a_norm, 0, sc.qres, q_lora, 0, null, n, 1, q_lora, rt.eps, false);
        if (rt.prof) |p| p.mark(s, .dfly_qkv);
        if (rt.pt_dev != 0) {
            try rt.head.top.rows.kvWritePaged(s, sc.qkv, qkv_stride, q_lora, ly.kv_a_norm, w.freqs, sc.pos_blk, sc.rot_pos_blk, 0, 0, rt.pt_dev, sc.row_slots, rt.pt_stride, rt.mla_bases[li], n, rt.eps, rt.kv8);
        } else {
            try rt.head.top.rows.kvWriteEx(s, sc.qkv, qkv_stride, q_lora, ly.kv_a_norm, w.freqs, sc.pos_blk, sc.rot_pos_blk, 0, 0, layer_caches, 0, 0, n, rt.eps, rt.kv8);
        }
        if (rt.prof) |p| p.mark(s, .dfly_mla);
        try rt.lane.matmul(s, sc.qres, q_lora, n, ly.q_b, sc.qb, sc.lane_part);
        if (rt.prof) |p| p.mark(s, .dfly_attn_mm);
        try rt.head.top.rows.qRope(s, sc.qb + nope * 2, qs, nope + rope, sc.qa + kv_lora * 2, h * mla_row, mla_row, sc.rot_pos_blk, w.freqs, n, h);
        try cuda_mla_c2.absorb(rt.c2, s, sc.qb, qs, n, h, ly.uk, sc.qa, qlo);

        if (sparse_reqs) |reqs| {
            for (reqs, 0..) |rq, bi| {
                const r0 = bi * dfly.qw;
                const qa_g = sc.qa + r0 * h * mla_row * 2;
                const ol_g = sc.ol + r0 * h * kv_lora * 4;
                const pos_g = sc.pos_blk + r0 * 4;
                const am = attnMode(rq.anchor_pos);
                if (am == .dense) {
                    try cuda_mla_c1.attn(rt.c1, s, qa_g, qlo, h, rq.kv_caches[li], .{ .kind = 1, .ring = 0, .kv8 = rt.kv8, .page_table = rq.page_table, .pt_stride = rq.pt_stride }, pos_g, ly.sinks, ol_g, dfly.qw);
                } else {
                    const meta_h = [4]i32{ 1, 0, @intCast(rq.anchor_pos), 0 };
                    const meta_ptr = sc.ext_meta.ptr + bi * 16;
                    try cuda.copy_source.enqueue(rt.d, meta_ptr, std.mem.sliceAsBytes(&meta_h), s);
                    const out_tok = sc.ext_topk.ptr + r0 * dfly.topk_pad * 4;
                    try rt.k.runExtendTopk(s, .{
                        .mode = if (am == .sparse) 0 else 1,
                        .src = rq.topk_ptrs[li],
                        .src_slot_stride = dfly.target_topk,
                        .src_row_stride = dfly.target_topk,
                        .src_rows_cap = 1,
                        .dense_below = dfly.target_topk,
                        .layer_slot = meta_ptr + 12,
                        .num_layers = 1,
                        .active = meta_ptr,
                        .source_rows = meta_ptr + 4,
                        .anchors = meta_ptr + 8,
                        .b = 1,
                        .qw = dfly.qw,
                        .target_topk = dfly.target_topk,
                        .local_window = 0,
                        .k_pad = dfly.topk_pad,
                        .out = out_tok,
                        .out_layer_stride = dfly.qw * dfly.topk_pad,
                    });
                    try cuda_mla_c1.attn(rt.c1, s, qa_g, qlo, h, rq.kv_caches[li], .{ .kind = 2, .ring = 0, .tok = out_tok, .ntok = dfly.topk_pad, .kv8 = rt.kv8, .page_table = rq.page_table, .pt_stride = rq.pt_stride }, pos_g, ly.sinks, ol_g, dfly.qw);
                }
            }
        } else if (sparse_graphed) {
            const out_tok = sc.ext_topk.ptr + li * n * dfly.topk_pad * 4;
            if (rt.pt_dev != 0) {
                try cuda_mla_c1.attn(rt.c1, s, sc.qa, qlo, h, rt.mla_bases[li], .{ .kind = 2, .ring = 0, .tok = out_tok, .ntok = dfly.topk_pad, .kv8 = rt.kv8, .page_table = rt.pt_dev, .pt_stride = rt.pt_stride, .row_stream = sc.row_slots }, sc.pos_blk, ly.sinks, sc.ol, n);
            } else {
                try cuda_mla_c1.attnRows(rt.c1, s, sc.qa, qlo, h, layer_caches, .{ .kind = 2, .ring = 0, .tok = out_tok, .ntok = dfly.topk_pad, .kv8 = rt.kv8 }, sc.pos_blk, ly.sinks, sc.ol, n);
            }
        } else {
            if (rt.pt_dev != 0) {
                try cuda_mla_c1.attn(rt.c1, s, sc.qa, qlo, h, rt.mla_bases[li], .{ .kind = 1, .ring = 0, .kv8 = rt.kv8, .page_table = rt.pt_dev, .pt_stride = rt.pt_stride, .row_stream = sc.row_slots }, sc.pos_blk, ly.sinks, sc.ol, n);
            } else {
                try cuda_mla_c1.attnRows(rt.c1, s, sc.qa, qlo, h, layer_caches, .{ .kind = 1, .ring = 0, .kv8 = rt.kv8 }, sc.pos_blk, ly.sinks, sc.ol, n);
            }
        }
        if (rt.prof) |p| p.mark(s, .dfly_mla);

        try rt.lane.matmul(s, sc.normed, hidden, n, ly.gate, sc.gate_buf, sc.lane_part);
        if (rt.prof) |p| p.mark(s, .dfly_attn_mm);
        try cuda_mla_c2.expand(rt.c2, s, sc.ol, n, h, ly.uv, sc.gate_buf, vs, sc.gated);
        if (rt.prof) |p| p.mark(s, .dfly_mla);
        try rt.lane.matmulF32(s, sc.gated, vs, n, ly.o, sc.part_f32, sc.lane_part);
        if (rt.prof) |p| p.mark(s, .dfly_attn_mm);
        const attn_sum = try reduce(rt, s, sc.part_f32, sc.red_f32, n * hidden);
        if (rt.prof) |p| p.mark(s, .dfly_comm);
        try rt.k.runBoundary(s, attn_sum, ly.post_norm, sc.res, hidden, ly.post_attn_norm, sc.res, hidden, sc.normed, rt.eps, n);
        if (rt.prof) |p| p.mark(s, .dfly_bnd);

        try rt.lane.matmul(s, sc.normed, hidden, n, ly.mlp_gate_up, sc.mlp_gu, sc.lane_part);
        try rt.act.swiglu(s, sc.mlp_gu, 2 * ly.ffn_local, sc.mlp_gu + ly.ffn_local * 2, 2 * ly.ffn_local, sc.mlp_act, null, n, ly.ffn_local);
        try rt.lane.matmulF32(s, sc.mlp_act, ly.ffn_local, n, ly.mlp_down, sc.part_f32, sc.lane_part);
        if (rt.prof) |p| p.mark(s, .dfly_mlp_mm);
        const mlp_sum = try reduce(rt, s, sc.part_f32, sc.red_f32, n * hidden);
        if (rt.prof) |p| p.mark(s, .dfly_comm);
        if (li + 1 < dfly.num_layers) {
            try rt.k.runBoundary(s, mlp_sum, 0, sc.res, hidden, w.layers[li + 1].input_norm, sc.res, hidden, sc.normed, rt.eps, n);
        } else {
            try rt.k.runBoundary(s, mlp_sum, 0, sc.res, hidden, w.final_norm, 0, hidden, sc.final_h, rt.eps, n);
        }
        if (rt.prof) |p| p.mark(s, .dfly_bnd);
    }

    const vocab_local = rt.head.top.share.len;
    for (0..depth) |step| {
        const h_col = sc.final_h + (step + 1) * hidden * 2;
        try rt.k.runDual(
            s,
            .{ .base = h_col, .ld = dfly.qw * hidden },
            w.hc_hidden_norm,
            .{ .base = rt.head.embed.w.ptr, .ids = sc.prev_ids, .ld = hidden, .ids_stride = 1, .nrows = @intCast(rt.vocab) },
            w.hc_embed_norm,
            rt.eps,
            sc.cat,
            2 * hidden,
            b,
        );
        try rt.lane.matmul(s, sc.cat, 2 * hidden, b, w.hc_gate_up, sc.mlp_gu, sc.lane_part);
        try rt.act.swiglu(s, sc.mlp_gu, 2 * w.hc_local, sc.mlp_gu + w.hc_local * 2, 2 * w.hc_local, sc.mlp_act, null, b, w.hc_local);
        try rt.lane.matmulF32(s, sc.mlp_act, w.hc_local, b, w.hc_down, sc.part_f32, sc.lane_part);
        if (rt.prof) |p| p.mark(s, .dfly_hc);
        const hc_sum = try reduce(rt, s, sc.part_f32, sc.red_f32, b * hidden);
        if (rt.prof) |p| p.mark(s, .dfly_hc_comm);
        try rt.k.runBoundary(s, hc_sum, 0, h_col, dfly.qw * hidden, 0, sc.normed, hidden, 0, rt.eps, b);
        if (rt.prof) |p| p.mark(s, .dfly_hc);

        if (w.lm_head_fp8) |hw| {
            try rt.lane.matmulF32(s, sc.normed, hidden, b, hw, rt.logits_local, sc.lane_part);
        } else {
            try rt.head.top.logitsFromHidden(rt.head_ops, s, sc.normed, rt.logits_local, vocab_local, b);
        }
        if (rt.prof) |p| p.mark(s, .dfly_lmhead);
        try rt.head.top.rows.argmax(s, rt.logits_local, vocab_local, rt.head.top.share.lo, b, rt.argmax_local);
        if (rt.tp_sess) |ts| {
            try ts.gatherArgmax(s, rt.argmax_local, rt.argmax_out, b);
        }
        try rt.k.runPickArgmax(s, rt.argmax_out, rt.world, b, step, depth, sc.prev_ids, sc.draft_out);
        if (rt.prof) |p| p.mark(s, .dfly_argmax_comm);
    }
}

test "attnMode partitions dense (<2043), seam (2043..2048), and graphable sparse (>=2049)" {
    try std.testing.expectEqual(AttnMode.dense, attnMode(0));
    try std.testing.expectEqual(AttnMode.dense, attnMode(2042));
    try std.testing.expectEqual(AttnMode.seam, attnMode(2043));
    try std.testing.expectEqual(AttnMode.seam, attnMode(2047));
    try std.testing.expectEqual(AttnMode.seam, attnMode(2048));
    try std.testing.expectEqual(AttnMode.sparse, attnMode(2049));
}

/// Runs DFly2 5-layer speculative drafting on `m`, writing `items.len * depth` token IDs into `draft_out`.
pub fn draftModel(m: *@import("cuda_model.zig").Model, k: *const @import("cuda_model.zig").AllKernels, s: Stream, items: []const @import("cuda_model.zig").Model.DraftItem, depth: usize) !void {
    const cm = @import("cuda_model.zig");
    const dw = if (m.dfly_w) |*p| p else return error.NoDraftHead;
    const dsc = if (m.dfly_sc) |*p| p else return error.NoDraftHead;
    if (items.len == 0 or items.len > dfly.max_reqs) return error.BadDraftShape;
    var reqs: [dfly.max_reqs]Req = undefined;
    for (items, 0..) |it, bi| {
        const sq: *const cm.Seq = if (m.seqs[it.slot]) |*q| q else return error.UnknownSlot;
        if (it.anchor_pos + dfly.qw > sq.cap or it.src_row >= cm.max_window) return error.CacheOverflow;
        var kv: [dfly.num_layers]u64 = undefined;
        var tk: [dfly.num_layers]u64 = undefined;
        for (dfly.target_kv_layers, 0..) |tl, dl| {
            kv[dl] = if (m.pages) |pg| pg.mla_bases[m.indexed_ord[tl].?] else (try sq.cache.view(tl)).mla;
            tk[dl] = sq.dflyTopkAt(dl, it.src_row);
        }
        reqs[bi] = .{
            .anchor_tok = it.anchor_tok,
            .anchor_pos = it.anchor_pos,
            .target_hidden_ptr = sq.last_hidden.ptr + it.src_row * m.c.hidden * 2,
            .kv_caches = kv,
            .topk_ptrs = tk,
            .mrope_delta = sq.mrope_delta,
            .slot = it.slot,
            .page_table = if (m.pages) |pg| pg.slotTablePtr(it.slot) else 0,
            .pt_stride = if (m.pages) |pg| pg.pt_stride else 0,
        };
    }
    var mla_bases: [dfly.num_layers]u64 = @splat(0);
    if (m.pages) |pg| {
        for (dfly.target_kv_layers, 0..) |tl, dl| mla_bases[dl] = pg.mla_bases[m.indexed_ord[tl].?];
    }
    const rt: Runtime = .{
        .d = m.d,
        .k = &k.dfly,
        .lane = &k.layer.mlp.lane,
        .norms = &k.layer.mlp.norms,
        .act = &k.layer.mlp.act,
        .mla_rope = &k.layer.rope,
        .c1 = &k.layer.mla.c1,
        .c2 = &k.layer.mla.c2,
        .head = &m.head,
        .head_ops = k.layer.headOps(),
        .logits_local = m.logits_local.ptr,
        .argmax_local = m.argmax_local.ptr,
        .argmax_out = m.argmaxPtr(),
        .tp_sess = m.tp_sess,
        .gather_buf = m.gather_buf.ptr,
        .world = m.at.world,
        .vocab = m.c.vocab,
        .eps = m.c.eps,
        .prof = if (m.prof) |*p| p else null,
        .kv8 = m.kv8,
        .pt_dev = if (m.pages) |pg| pg.pt_dev.ptr else 0,
        .pt_stride = if (m.pages) |pg| pg.pt_stride else 0,
        .mla_bases = mla_bases,
    };
    if (m.prof) |*p| p.begin(s, .{ .kind = .dfly, .rows = items.len * dfly.qw, .parts = items.len, .mode = depth });
    try propose(rt, s, dw, dsc, reqs[0..items.len], depth);
    if (m.prof) |*p| p.endSubmit();
}

/// Partitions `active` layers into merged post-attention and sliding-window CUDA Graph spans.
pub fn initSwaSpans(m: *@import("cuda_model.zig").Model, active: usize) void {
    const cm = @import("cuda_model.zig");
    const has_er = m.engram_reader != null;
    var l_scan: usize = 0;
    while (l_scan < active and m.n_swa_spans < cm.Model.max_swa_spans) {
        const is_idx = !m.layers[l_scan].is_sliding;
        if (!is_idx and l_scan + 1 >= active) break;
        const tl: ?usize = if (is_idx) l_scan else null;
        const swa_lo = l_scan + 1 - @intFromBool(!is_idx);
        var swa_hi = if (is_idx) swa_lo else l_scan + 1;
        while (swa_hi + 1 < active and m.layers[swa_hi].is_sliding and !(m.layers[swa_hi].has_engram and has_er)) : (swa_hi += 1) {}
        const si = m.n_swa_spans;
        m.swa_spans[si] = .{ .tail_layer = tl, .swa_lo = swa_lo, .swa_hi = swa_hi, .fuse_head = is_idx and (l_scan + 1 == active) };
        m.swa_span_at[l_scan] = @intCast(si);
        m.n_swa_spans += 1;
        l_scan = swa_hi;
    }
}
