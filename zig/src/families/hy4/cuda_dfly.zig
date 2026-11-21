//! Built-in 5-layer DFly2 speculative drafter (`hy4_dfly2_glue.cu` + `c1` + `c2` + `qmmf` + shared embed/KV).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const names = @import("names.zig");
const partition = @import("partition.zig");
const cuda_weights = @import("cuda_weights.zig");
const cuda_mla = @import("cuda_mla.zig");
const cuda_mla_c1 = @import("cuda_mla_c1.zig");
const cuda_mla_c2 = @import("cuda_mla_c2.zig");
const cuda_head = @import("cuda_head.zig");
const cuda_tp = @import("cuda_tp.zig");
const cuda_prof = @import("cuda_prof.zig");

const Config = config.Config;
const Stream = cuda.Stream;
const Function = cuda.Function;
const DeviceBuffer = cuda.DeviceBuffer;
const qmmf = cuda.qmmf;
const Rank = cuda_weights.Rank;
const Share = partition.Share;
const fp8 = core.quant.fp8;

pub const num_layers: usize = 5;
pub const max_drafts: usize = 5;
pub const qw: usize = max_drafts + 1; // 6 rows per request: [bonus, MASK x 5]
pub const mask_token_id: u32 = 120009;
pub const target_kv_layers = [num_layers]usize{ 25, 29, 33, 37, 41 };
pub const hidden: usize = 3072;
pub const q_lora: usize = 1024;
pub const kv_lora: usize = 512;
pub const nope: usize = 192;
pub const rope: usize = 64;
pub const v_width: usize = 256;
pub const mla_row: usize = kv_lora + rope; // 576
pub const total_heads: usize = 32;
pub const ffn_width: usize = 9216;
pub const hc_width: usize = 3072;
pub const topk_pad: usize = 2176; // round_up(2048 + 0 + 6, 128)
pub const target_topk: usize = 2048;
pub const max_reqs: usize = 16; // up to 16 concurrent streams drafted in one block pass (96 rows)
pub const rope_theta: f64 = 10_000_000.0;

pub const DflyRows = extern struct {
    base: u64,
    ids: u64 = 0,
    ld: i64,
    ids_stride: i32 = 1,
    nrows: i32 = 0,

    comptime {
        std.debug.assert(@sizeOf(DflyRows) == 32);
    }
};

pub const TopkArgs = extern struct {
    mode: i32 = 0, // HY4D_SRC_TOK = 0
    src: u64 = 0,
    src_slot_stride: i64 = 0,
    src_row_stride: i64 = 0,
    src_rows_cap: i32 = 0,
    dense_below: i32 = 2048,
    layer_slot: u64 = 0,
    num_layers: i32 = 1,
    active: u64 = 0,
    source_rows: u64 = 0,
    anchors: u64 = 0,
    b: i32 = 1,
    qw: i32 = qw,
    target_topk: i32 = target_topk,
    local_window: i32 = 0,
    k_pad: i32 = topk_pad,
    out: u64 = 0,
    out_layer_stride: i64 = 0,
    slot_out: u64 = 0,
    row_pos_out: u64 = 0,
    block_table: u64 = 0,
    bt_stride: i64 = 0,
    blocks_per_req: i32 = 0,
    block_size: i32 = 0,
    max_model_len: i32 = 0,
    req_stream: u64 = 0,
    ctx: i64 = 0,
    pad_slot: i64 = -1,

    comptime {
        std.debug.assert(@sizeOf(TopkArgs) == 192);
    }
};

pub const Kernels = struct {
    mod: cuda.Module,
    dual: Function,
    fusion: Function,
    boundary: Function,
    extend_topk: Function,
    pick_argmax: Function,
    gather_hidden: Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.hy4_dfly2_glue);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .dual = try mod.function("hy4_dfly_dual_rmsnorm_cat_s1"),
            .fusion = try mod.function("hy4_dfly_fusion_gated_add_s1"),
            .boundary = try mod.function("hy4_dfly_sublayer_boundary_s1"),
            .extend_topk = try mod.function("hy4_dfly_extend_topk_k"),
            .pick_argmax = try mod.function("hy4_dfly_pick_argmax"),
            .gather_hidden = try mod.function("hy4_dfly_gather_hidden"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.* = undefined;
    }

    pub fn runDual(k: *const Kernels, s: Stream, a: DflyRows, wa: u64, b: DflyRows, wb: u64, eps: f32, out: u64, ld_out: usize, rows: usize) !void {
        var args: cuda.Args = .{};
        args.add(a);
        args.add(wa);
        args.add(b);
        args.add(wb);
        args.add(eps);
        args.add(out);
        args.add(@as(i64, @intCast(ld_out)));
        try cuda.launch.launch(k.dual, .{ .grid = .{ .x = 1, .y = 2, .z = @intCast(rows) }, .block = .{ .x = 384 } }, s, &args);
    }

    pub fn runFusion(k: *const Kernels, s: Stream, e: DflyRows, eh: u64, gate: u64, w_norm: u64, eps: f32, res_out: u64, norm_out: u64, n: usize) !void {
        var args: cuda.Args = .{};
        args.add(e);
        args.add(eh);
        args.add(@as(i64, hidden));
        args.add(@as(i64, 0));
        args.add(@as(i32, 1));
        args.add(gate);
        args.add(w_norm);
        args.add(eps);
        args.add(@as(i32, qw));
        args.add(res_out);
        args.add(@as(i64, hidden));
        args.add(norm_out);
        args.add(@as(i64, hidden));
        try cuda.launch.launch(k.fusion, .{ .grid = .{ .x = 1, .y = @intCast(n), .z = 1 }, .block = .{ .x = 384 } }, s, &args);
    }

    pub fn runBoundary(k: *const Kernels, s: Stream, part: u64, sandwich_w: u64, res_in: u64, ld_ri: usize, next_w: u64, res_out: u64, ld_ro: usize, next_out: u64, eps: f32, rows: usize) !void {
        var args: cuda.Args = .{};
        args.add(part);
        args.add(@as(i64, hidden));
        args.add(@as(i64, 0));
        args.add(@as(i32, 1));
        args.add(sandwich_w);
        args.add(res_in);
        args.add(@as(i64, @intCast(ld_ri)));
        args.add(next_w);
        args.add(res_out);
        args.add(@as(i64, @intCast(ld_ro)));
        args.add(next_out);
        args.add(@as(i64, hidden));
        args.add(eps);
        try cuda.launch.launch(k.boundary, .{ .grid = .{ .x = 1, .y = @intCast(rows), .z = 1 }, .block = .{ .x = 384 } }, s, &args);
    }

    pub fn runExtendTopk(k: *const Kernels, s: Stream, ta: TopkArgs) !void {
        var args: cuda.Args = .{};
        args.add(ta);
        const gx: u32 = @intCast(ta.b * ta.num_layers);
        const kpad_u: u32 = @intCast(ta.k_pad);
        const gy: u32 = ((kpad_u >> 2) + 127) / 128;
        try cuda.launch.launch(k.extend_topk, .{ .grid = .{ .x = gx, .y = gy, .z = 1 }, .block = .{ .x = 128 } }, s, &args);
    }

    pub fn runPickArgmax(k: *const Kernels, s: Stream, argmax_full: u64, world: usize, b: usize, step: usize, stride: usize, prev_ids: u64, draft_out: u64) !void {
        var args: cuda.Args = .{};
        args.add(argmax_full);
        args.add(@as(i32, @intCast(world)));
        args.add(@as(i32, @intCast(b)));
        args.add(@as(i32, @intCast(step)));
        args.add(@as(i32, @intCast(stride)));
        args.add(prev_ids);
        args.add(draft_out);
        try cuda.launch.launch(k.pick_argmax, .{ .grid = .{ .x = @intCast((b + 63) / 64), .y = 1, .z = 1 }, .block = .{ .x = 64 } }, s, &args);
    }

    pub fn runGatherHidden(k: *const Kernels, s: Stream, src_ptrs: u64, dst: u64, b: usize) !void {
        var args: cuda.Args = .{};
        args.add(src_ptrs);
        args.add(dst);
        args.add(@as(i32, @intCast(b)));
        try cuda.launch.launch(k.gather_hidden, .{ .grid = .{ .x = @intCast(b), .y = 1, .z = 1 }, .block = .{ .x = 384 } }, s, &args);
    }
};

pub const DraftLayer = struct {
    heads: usize,
    input_norm: u64,
    post_attn_norm: u64,
    qkv_a: qmmf.Weight, // bf16 [1024 + 576, 3072]
    q_a_norm: u64,
    kv_a_norm: u64,
    q_b: qmmf.Weight, // fp8g [heads * 256, 1024]
    uk: u64, // bf16 c2 TILED [heads][512][192]
    uv: u64, // bf16 c2 TILED [heads][256][512]
    gate: qmmf.Weight, // fp8g [heads * 256, 3072]
    o: qmmf.Weight, // fp8g [3072, heads * 256]
    post_norm: u64,
    sinks: u64, // fp32 [heads]
    mlp_gate_up: qmmf.Weight, // fp8g [2 * ffn_local, 3072]
    mlp_down: qmmf.Weight, // fp8g [3072, ffn_local]
    ffn_local: usize,
};

pub const Weights = struct {
    store: cuda_weights.Weights,
    enorm: u64,
    hnorm: u64,
    gate: u64,
    eh_proj: qmmf.Weight, // bf16 [3072, 6144], replicated
    final_norm: u64,
    hc_hidden_norm: u64,
    hc_embed_norm: u64,
    hc_gate_up: qmmf.Weight, // bf16 [2 * hc_local, 6144]
    hc_down: qmmf.Weight, // bf16 [3072, hc_local]
    hc_local: usize,
    lm_head_fp8: ?qmmf.Weight = null, // optional 1x64 group FP8 E4M3 copy of rank's lm_head (197 MB vs 371 MB)
    freqs: u64, // fp32 [32]
    layers: [num_layers]DraftLayer,

    pub fn enabled() bool {
        const v = @import("profile").lookup("TF_HY4_DFLY") orelse return true;
        return v.len == 0 or v[0] != '0';
    }

    pub fn fp8HeadEnabled() bool {
        const v = @import("profile").lookup("TF_HY4_DFLY_FP8_HEAD") orelse return true;
        return v.len == 0 or v[0] != '0';
    }

    /// Quantizes BF16 `[n, k]` (`k % 64 == 0`) into 1x64 FP8G for DFly2 drafts; target verify uses BF16 `lm_head`.
    fn packBf16AsFp8g(gpa: std.mem.Allocator, store: *cuda_weights.Weights, bf16_bytes: []const u8, n: usize, k: usize) !qmmf.Weight {
        const npad = cuda.fp8.padded(n);
        const kg = k / cuda.fp8.group;
        const raw = try gpa.alloc(u8, n * k);
        defer gpa.free(raw);
        const scales = try gpa.alloc(f32, npad * kg);
        defer gpa.free(scales);
        @memset(scales, 1.0);
        for (0..n) |r| {
            const t = r / 64;
            const c = r % 64;
            for (0..kg) |g| {
                const base = r * k + g * 64;
                var vals: [64]f32 = undefined;
                var amax: f32 = 0.0;
                for (0..64) |j| {
                    const v = bf16ToF32(bf16_bytes, base + j);
                    vals[j] = v;
                    const av = @abs(v);
                    if (av > amax) amax = av;
                }
                const sc: f32 = if (amax > 0.0) amax / 448.0 else 1.0;
                const inv: f32 = if (amax > 0.0) 448.0 / amax else 0.0;
                scales[(t * kg + g) * 64 + c] = sc;
                for (0..64) |j| {
                    raw[base + j] = core.exlr8.codec.e4m3FromF32(vals[j] * inv);
                }
            }
        }
        const packed_codes = try gpa.alloc(u8, npad * k);
        defer gpa.free(packed_codes);
        cuda.fp8.packCodes(packed_codes, raw, n, k);
        const codes_ptr = try store.upload(packed_codes);
        const scales_ptr = try store.upload(std.mem.sliceAsBytes(scales));
        return cuda.fp8.weight(codes_ptr, scales_ptr, n, k);
    }

    pub fn present(ck: *core.checkpoint.Checkpoint) bool {
        return ck.has("model.dfly_layers.final_norm.weight") and ck.has("model.dfly_layers.0.input_layernorm.weight");
    }

    fn readProj(ck: *core.checkpoint.Checkpoint, prefix: []const u8, n: usize, k: usize) !names.Proj {
        var wbuf: [192]u8 = undefined;
        const wname = try std.fmt.bufPrint(&wbuf, "{s}.weight", .{prefix});
        const w = try ck.get(wname);
        return switch (w.dtype) {
            .bf16 => .{ .bf16 = try ck.expect(wname, .bf16, &.{ n, k }) },
            .f8_e4m3 => blk: {
                const h = try fp8.read(ck, prefix, n, k, .bf16, null);
                break :blk .{ .fp8 = .{ .w = h.weight, .s = h.scale_inv } };
            },
            else => error.UnsupportedQuantization,
        };
    }

    fn bf16ToF32(bytes: []const u8, i: usize) f32 {
        const bits = std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
        return @bitCast(@as(u32, bits) << 16);
    }

    pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, ck: *core.checkpoint.Checkpoint, c: Config, at: Rank) !Weights {
        var store = cuda_weights.Weights.init(gpa, d);
        errdefer store.deinit();
        const wholeH: Share = .{ .lo = 0, .len = hidden };
        const whole2H: Share = .{ .lo = 0, .len = 2 * hidden };
        const hc_sh = try partition.ffn(hc_width, at.rank, at.world);
        const ffn_sh = try partition.ffn(ffn_width, at.rank, at.world);
        const heads_local = total_heads / at.world;
        const head_lo = at.rank * heads_local;
        const q_rows: Share = .{ .lo = head_lo * (nope + rope), .len = heads_local * (nope + rope) };
        const kv_rows: Share = .{ .lo = head_lo * (nope + v_width), .len = heads_local * (nope + v_width) };
        const v_sh: Share = .{ .lo = head_lo * v_width, .len = heads_local * v_width };

        const enorm = try store.upload((try ck.expect("model.dfly_layers.mtp_fusion.enorm.weight", .bf16, &.{hidden})).bytes);
        const hnorm = try store.upload((try ck.expect("model.dfly_layers.mtp_fusion.hnorm.weight", .bf16, &.{hidden})).bytes);
        const gate = try store.upload((try ck.expect("model.dfly_layers.mtp_fusion.gate", .bf16, &.{hidden})).bytes);
        const eh_t = try ck.expect("model.dfly_layers.mtp_fusion.eh_proj.weight", .bf16, &.{ hidden, 2 * hidden });
        const eh_proj = try store.projPin1(&.{.{ .p = .{ .bf16 = eh_t }, .rows = wholeH }}, whole2H);
        const final_norm = try store.upload((try ck.expect("model.dfly_layers.final_norm.weight", .bf16, &.{hidden})).bytes);

        const hc_hn = try store.upload((try ck.expect("model.dfly_layers.hidden_correction.hidden_norm.weight", .bf16, &.{hidden})).bytes);
        const hc_en = try store.upload((try ck.expect("model.dfly_layers.hidden_correction.embed_norm.weight", .bf16, &.{hidden})).bytes);
        const hc_g = try ck.expect("model.dfly_layers.hidden_correction.gate_proj.weight", .bf16, &.{ hc_width, 2 * hidden });
        const hc_u = try ck.expect("model.dfly_layers.hidden_correction.up_proj.weight", .bf16, &.{ hc_width, 2 * hidden });
        const hc_d = try ck.expect("model.dfly_layers.hidden_correction.down_proj.weight", .bf16, &.{ hidden, hc_width });
        const hc_gate_up = try store.projPin1(&.{ .{ .p = .{ .bf16 = hc_g }, .rows = hc_sh }, .{ .p = .{ .bf16 = hc_u }, .rows = hc_sh } }, whole2H);
        const hc_down = try store.projPin1(&.{.{ .p = .{ .bf16 = hc_d }, .rows = wholeH }}, hc_sh);

        const f = cuda.mla.ropeFreqs(rope_theta);
        const freqs = try store.upload(std.mem.sliceAsBytes(&f));

        var layers: [num_layers]DraftLayer = undefined;
        for (0..num_layers) |li| {
            var pbuf: [128]u8 = undefined;
            const in_n = try ck.expect(try std.fmt.bufPrint(&pbuf, "model.dfly_layers.{d}.input_layernorm.weight", .{li}), .bf16, &.{hidden});
            const pa_n = try ck.expect(try std.fmt.bufPrint(&pbuf, "model.dfly_layers.{d}.post_attention_layernorm.weight", .{li}), .bf16, &.{hidden});
            const qa_t = try ck.expect(try std.fmt.bufPrint(&pbuf, "model.dfly_layers.{d}.self_attn.q_a_proj.weight", .{li}), .bf16, &.{ q_lora, hidden });
            const kva_t = try ck.expect(try std.fmt.bufPrint(&pbuf, "model.dfly_layers.{d}.self_attn.kv_a_proj_with_mqa.weight", .{li}), .bf16, &.{ mla_row, hidden });
            const qan = try ck.expect(try std.fmt.bufPrint(&pbuf, "model.dfly_layers.{d}.self_attn.q_a_layernorm.weight", .{li}), .bf16, &.{q_lora});
            const kvan = try ck.expect(try std.fmt.bufPrint(&pbuf, "model.dfly_layers.{d}.self_attn.kv_a_layernorm.weight", .{li}), .bf16, &.{kv_lora});
            const pn = try ck.expect(try std.fmt.bufPrint(&pbuf, "model.dfly_layers.{d}.self_attn.post_proj_norm.weight", .{li}), .bf16, &.{hidden});
            const kvb_t = try ck.expect(try std.fmt.bufPrint(&pbuf, "model.dfly_layers.{d}.self_attn.kv_b_proj.weight", .{li}), .bf16, &.{ total_heads * (nope + v_width), kv_lora });
            const sinks_t = try ck.get(try std.fmt.bufPrint(&pbuf, "model.dfly_layers.{d}.self_attn.learnable_sink_param", .{li}));
            var sinks_f32: [total_heads]f32 = undefined;
            for (0..heads_local) |h_i| {
                sinks_f32[h_i] = switch (sinks_t.dtype) {
                    .bf16 => bf16ToF32(sinks_t.bytes, head_lo + h_i),
                    .f32 => @bitCast(std.mem.readInt(u32, sinks_t.bytes[(head_lo + h_i) * 4 ..][0..4], .little)),
                    else => return error.UnexpectedTensor,
                };
            }
            const tiled = try cuda_mla_c2.repack(gpa, kvb_t.bytes[kv_rows.lo * kv_lora * 2 ..][0 .. kv_rows.len * kv_lora * 2], heads_local);
            defer gpa.free(tiled.uk);
            defer gpa.free(tiled.uv);

            var nb: [128]u8 = undefined;
            const qb_p = try readProj(ck, try std.fmt.bufPrint(&nb, "model.dfly_layers.{d}.self_attn.q_b_proj", .{li}), total_heads * (nope + rope), q_lora);
            const gt_p = try readProj(ck, try std.fmt.bufPrint(&nb, "model.dfly_layers.{d}.self_attn.linear_gate", .{li}), total_heads * v_width, hidden);
            const op_p = try readProj(ck, try std.fmt.bufPrint(&nb, "model.dfly_layers.{d}.self_attn.o_proj", .{li}), hidden, total_heads * v_width);
            const mg_p = try readProj(ck, try std.fmt.bufPrint(&nb, "model.dfly_layers.{d}.mlp.gate_proj", .{li}), ffn_width, hidden);
            const mu_p = try readProj(ck, try std.fmt.bufPrint(&nb, "model.dfly_layers.{d}.mlp.up_proj", .{li}), ffn_width, hidden);
            const md_p = try readProj(ck, try std.fmt.bufPrint(&nb, "model.dfly_layers.{d}.mlp.down_proj", .{li}), hidden, ffn_width);

            layers[li] = .{
                .heads = heads_local,
                .input_norm = try store.upload(in_n.bytes),
                .post_attn_norm = try store.upload(pa_n.bytes),
                .qkv_a = try store.projPin1(&.{
                    .{ .p = .{ .bf16 = qa_t }, .rows = .{ .lo = 0, .len = q_lora } },
                    .{ .p = .{ .bf16 = kva_t }, .rows = .{ .lo = 0, .len = mla_row } },
                }, wholeH),
                .q_a_norm = try store.upload(qan.bytes),
                .kv_a_norm = try store.upload(kvan.bytes),
                .q_b = try store.projPin1(&.{.{ .p = qb_p, .rows = q_rows }}, .{ .lo = 0, .len = q_lora }),
                .uk = try store.upload(std.mem.sliceAsBytes(tiled.uk)),
                .uv = try store.upload(std.mem.sliceAsBytes(tiled.uv)),
                .gate = try store.projPin1(&.{.{ .p = gt_p, .rows = v_sh }}, wholeH),
                .o = try store.projPin1(&.{.{ .p = op_p, .rows = wholeH }}, v_sh),
                .post_norm = try store.upload(pn.bytes),
                .sinks = try store.upload(std.mem.sliceAsBytes(sinks_f32[0..heads_local])),
                .mlp_gate_up = try store.projPin1(&.{ .{ .p = mg_p, .rows = ffn_sh }, .{ .p = mu_p, .rows = ffn_sh } }, wholeH),
                .mlp_down = try store.projPin1(&.{.{ .p = md_p, .rows = wholeH }}, ffn_sh),
                .ffn_local = ffn_sh.len,
            };
        }

        var lm_head_fp8: ?qmmf.Weight = null;
        if (fp8HeadEnabled()) {
            const vsh = try partition.vocab(c, at.rank, at.world);
            const head_t = try ck.expect("lm_head.weight", .bf16, &.{ c.vocab, hidden });
            const slice = head_t.bytes[vsh.lo * hidden * 2 ..][0 .. vsh.len * hidden * 2];
            lm_head_fp8 = try packBf16AsFp8g(gpa, &store, slice, vsh.len, hidden);
        }

        return .{
            .store = store,
            .enorm = enorm,
            .hnorm = hnorm,
            .gate = gate,
            .eh_proj = eh_proj,
            .final_norm = final_norm,
            .hc_hidden_norm = hc_hn,
            .hc_embed_norm = hc_en,
            .hc_gate_up = hc_gate_up,
            .hc_down = hc_down,
            .hc_local = hc_sh.len,
            .lm_head_fp8 = lm_head_fp8,
            .freqs = freqs,
            .layers = layers,
        };
    }

    pub fn deinit(w: *Weights) void {
        w.store.deinit();
        w.* = undefined;
    }
};

pub const Scratch = struct {
    bufs: [26]DeviceBuffer = undefined,
    n_bufs: usize = 0,
    // pointers into bufs
    hidden_in: u64 = 0, // bf16 [max_reqs, 3072]
    src_ptrs: u64 = 0, // u64 [max_reqs]
    cat: u64 = 0, // bf16 [max_reqs, 6144]
    eh_f32: u64 = 0, // fp32 [max_reqs, 3072]
    ids_blk: u64 = 0, // i32 [max_reqs * qw]
    pos_blk: u64 = 0, // i32 [max_reqs * qw]
    rot_pos_blk: u64 = 0, // i32 [max_reqs * qw]: rotary positions (`pos + mrope_delta`)
    res: u64 = 0, // bf16 [max_reqs * qw, 3072]
    normed: u64 = 0, // bf16 [max_reqs * qw, 3072]
    final_h: u64 = 0, // bf16 [max_reqs * qw, 3072]
    qkv: u64 = 0, // bf16 [max_reqs * qw, 1600]
    qres: u64 = 0, // bf16 [max_reqs * qw, 1024]
    qb: u64 = 0, // bf16 [max_reqs * qw, heads * 256]
    qa: u64 = 0, // bf16 [max_reqs * qw, heads * 576] * 2 (hi+lo)
    ol: u64 = 0, // fp32 [max_reqs * qw, heads * 512]
    gate_buf: u64 = 0, // bf16 [max_reqs * qw, heads * 256]
    gated: u64 = 0, // bf16 [max_reqs * qw, heads * 256]
    mlp_gu: u64 = 0, // bf16 [max_reqs * qw, 2 * ffn_local]
    mlp_act: u64 = 0, // bf16 [max_reqs * qw, max(ffn_local, hc_local)]
    part_f32: u64 = 0, // fp32 [max_reqs * qw, 3072]
    red_f32: u64 = 0, // fp32 [max_reqs * qw, 3072]
    prev_ids: u64 = 0, // i32 [max_reqs]
    draft_out: u64 = 0, // u32 [max_reqs * max_drafts]
    row_caches: u64 = 0, // u64 [num_layers * max_reqs * qw]
    row_slots: u64 = 0, // i32 [max_reqs * qw]: slot index per draft row when paged
    lane_part: u64 = 0, // qmmf split-K partials

    // Extended topk scratch (for anchor >= 2048): [num_layers, max_reqs * qw, topk_pad]
    ext_topk: DeviceBuffer,
    ext_meta: DeviceBuffer, // i32 [4 * max_reqs + 8]: active, source_rows, anchors
    src_topk: DeviceBuffer, // i32 [max_graph_b * num_layers * target_topk]: copied before sparse graph launch
    ext_layer_slots: DeviceBuffer, // i32 [num_layers] = {0, 1, 2, 3, 4}

    // Per-(B, depth) CUDA Graph caches for dense (<2048) and sparse (>=2048) DFly2 proposals (n <= 64, B <= 10)
    graphs: [max_graph_b + 1][max_drafts + 1]GraphSlot = @splat(@splat(.{})),
    sparse_graphs: [max_graph_b + 1][max_drafts + 1]GraphSlot = @splat(@splat(.{})),
    graphs_enabled: bool = true,

    pub const max_graph_b: usize = 10;
    pub const GraphState = enum(u8) { cold, warm, ready, failed };
    pub const GraphSlot = struct { state: GraphState = .cold, exec: cuda.graph.Exec = undefined };

    fn readGraphEnabled() bool {
        const v = @import("profile").lookup("TF_HY4_DFLY_GRAPH") orelse return true;
        return v.len == 0 or v[0] != '0';
    }

    fn add(sc: *Scratch, d: *const cuda.Driver, bytes: usize) !u64 {
        const b = try DeviceBuffer.alloc(d, @max(bytes, 16));
        sc.bufs[sc.n_bufs] = b;
        sc.n_bufs += 1;
        return b.ptr;
    }

    pub fn init(d: *const cuda.Driver, lane: *const qmmf.Lane, w: *const Weights) !Scratch {
        const nr = max_reqs * qw; // 96
        const h = w.layers[0].heads;
        const fl = w.layers[0].ffn_local;
        var sc: Scratch = .{ .ext_topk = undefined, .ext_meta = undefined, .src_topk = undefined, .ext_layer_slots = undefined, .graphs_enabled = readGraphEnabled() };
        errdefer for (sc.bufs[0..sc.n_bufs]) |*b| b.free();
        sc.hidden_in = try sc.add(d, max_reqs * hidden * 2);
        sc.src_ptrs = try sc.add(d, max_reqs * 8);
        sc.cat = try sc.add(d, max_reqs * 2 * hidden * 2);
        sc.eh_f32 = try sc.add(d, max_reqs * hidden * 4);
        sc.ids_blk = try sc.add(d, nr * 4);
        sc.pos_blk = try sc.add(d, nr * 4);
        sc.rot_pos_blk = try sc.add(d, nr * 4);
        sc.res = try sc.add(d, nr * hidden * 2);
        sc.normed = try sc.add(d, nr * hidden * 2);
        sc.final_h = try sc.add(d, nr * hidden * 2);
        sc.qkv = try sc.add(d, nr * (q_lora + mla_row) * 2);
        sc.qres = try sc.add(d, nr * q_lora * 2);
        sc.qb = try sc.add(d, nr * h * (nope + rope) * 2);
        sc.qa = try sc.add(d, nr * h * mla_row * 4);
        sc.ol = try sc.add(d, nr * h * kv_lora * 4);
        sc.gate_buf = try sc.add(d, nr * h * v_width * 2);
        sc.gated = try sc.add(d, nr * h * v_width * 2);
        sc.mlp_gu = try sc.add(d, nr * 2 * @max(fl, w.hc_local) * 2);
        sc.mlp_act = try sc.add(d, nr * @max(fl, w.hc_local) * 2);
        sc.part_f32 = try sc.add(d, nr * hidden * 4);
        sc.red_f32 = try sc.add(d, nr * hidden * 4);
        sc.prev_ids = try sc.add(d, max_reqs * 4);
        sc.draft_out = try sc.add(d, max_reqs * max_drafts * 4);
        sc.row_caches = try sc.add(d, num_layers * nr * 8);
        sc.row_slots = try sc.add(d, nr * 4);

        var lp: usize = 0;
        const ws = [_]qmmf.Weight{
            w.eh_proj,         w.hc_gate_up,            w.hc_down,
            w.layers[0].qkv_a, w.layers[0].q_b,         w.layers[0].gate,
            w.layers[0].o,     w.layers[0].mlp_gate_up, w.layers[0].mlp_down,
        };
        for (ws) |pw| {
            var m: usize = 1;
            while (m <= nr) : (m += 1) lp = @max(lp, lane.partBytes(m, pw));
        }
        if (w.lm_head_fp8) |hw| {
            var m: usize = 1;
            while (m <= max_reqs) : (m += 1) lp = @max(lp, lane.partBytes(m, hw));
        }
        sc.lane_part = try sc.add(d, lp);
        sc.ext_topk = try DeviceBuffer.alloc(d, num_layers * nr * topk_pad * 4);
        errdefer sc.ext_topk.free();
        sc.ext_meta = try DeviceBuffer.alloc(d, (4 * max_reqs + 8) * 4);
        errdefer sc.ext_meta.free();
        sc.src_topk = try DeviceBuffer.alloc(d, max_graph_b * num_layers * target_topk * 4);
        errdefer sc.src_topk.free();
        const slots_h = [num_layers]i32{ 0, 1, 2, 3, 4 };
        sc.ext_layer_slots = try DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(&slots_h));
        return sc;
    }

    pub fn deinit(sc: *Scratch) void {
        for (&sc.graphs) |*row| {
            for (row) |*slot| {
                if (slot.state == .ready) slot.exec.deinit();
            }
        }
        for (&sc.sparse_graphs) |*row| {
            for (row) |*slot| {
                if (slot.state == .ready) slot.exec.deinit();
            }
        }
        sc.ext_layer_slots.free();
        sc.src_topk.free();
        sc.ext_meta.free();
        sc.ext_topk.free();
        for (sc.bufs[0..sc.n_bufs]) |*b| b.free();
        sc.* = undefined;
    }
};

pub const Req = @import("cuda_dfly_run.zig").Req;
pub const Runtime = @import("cuda_dfly_run.zig").Runtime;
pub const propose = @import("cuda_dfly_run.zig").propose;

test "cuda_dfly declarations compile on host" {
    std.testing.refAllDecls(@This());
    _ = @import("cuda_dfly_run.zig");
}
