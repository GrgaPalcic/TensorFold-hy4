//! Hy4-flash Wave-1 & Wave-2 CUDA launch records (`hy4_*`, `fp8_experts` TMA, `qmmf` FP8G 64x128; nvcc 13.0).
const std = @import("std");
const registry = @import("registry.zig");
const U = @import("limits.zig").Usage;

fn five(b: u32, r80: u32, s80: u32, r86: u32, s86: u32, r89: u32, s89: u32, r120: u32, s120: u32, r121: u32, s121: u32) [5]U {
    return .{
        .{ .regs = r80, .static = s80, .bound = b },
        .{ .regs = r86, .static = s86, .bound = b },
        .{ .regs = r89, .static = s89, .bound = b },
        .{ .regs = r120, .static = s120, .bound = b },
        .{ .regs = r121, .static = s121, .bound = b },
    };
}

fn sameS(b: u32, s: u32, r80: u32, r86: u32, r89: u32, r120: u32, r121: u32) [5]U {
    return five(b, r80, s, r86, s, r89, s, r120, s, r121, s);
}

pub const records = [_]registry.CudaRecord{
    // hy4_rows: GPU local-vocab greedy argmax, sharding helpers, batched KV write, and per-row Q RoPE
    .{ .launch = .{ .name = "hy4_rows hy4_rows_argmax", .symbol = "hy4_rows_argmax", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 96, 20, 20, 20, 29, 29), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_slice_bf16", .symbol = "hy4_rows_slice_bf16", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 0, 16, 16, 16, 16, 16), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_interleave2_bf16", .symbol = "hy4_rows_interleave2_bf16", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 0, 18, 18, 18, 24, 24), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_slice_f32", .symbol = "hy4_rows_slice_f32", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 0, 16, 16, 16, 16, 16), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_comb_swiglu_bf16", .symbol = "hy4_rows_comb_swiglu_bf16", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 0, 17, 17, 17, 17, 17), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_kv_write", .symbol = "hy4_rows_kv_write", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 32, 25, 27, 27, 25, 25), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_q_rope", .symbol = "hy4_rows_q_rope", .image = "hy4_rows", .threads = 32 }, .usage = sameS(32, 0, 26, 26, 26, 23, 23), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_f32_to_bf16", .symbol = "hy4_rows_f32_to_bf16", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 0, 25, 25, 25, 24, 24), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_interleave_bf16", .symbol = "hy4_rows_interleave_bf16", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 0, 16, 16, 16, 16, 16), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_kv_write_ex", .symbol = "hy4_rows_kv_write_ex", .image = "hy4_rows", .threads = 512 }, .usage = sameS(512, 64, 32, 34, 35, 32, 32), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_kv_write_paged", .symbol = "hy4_rows_kv_write_paged", .image = "hy4_rows", .threads = 512 }, .usage = sameS(512, 64, 32, 34, 36, 32, 32), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_q_rope_mrope", .symbol = "hy4_rows_q_rope_mrope", .image = "hy4_rows", .threads = 32 }, .usage = sameS(32, 0, 27, 27, 28, 22, 22), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_splice_streams", .symbol = "hy4_rows_splice_streams", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 0, 16, 16, 16, 16, 16), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_vit_qkv", .symbol = "hy4_rows_vit_qkv", .image = "hy4_rows", .threads = 128 }, .usage = sameS(128, 0, 30, 29, 29, 30, 30), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_vit_merge_gather", .symbol = "hy4_rows_vit_merge_gather", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 0, 26, 26, 26, 28, 28), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_vit_append_newline", .symbol = "hy4_rows_vit_append_newline", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 0, 18, 18, 18, 22, 22), .later_pending = true },

    // hy4_route: 4-warp bitonic top-16 sigmoid router and R=1 fused plan
    .{ .launch = .{ .name = "hy4_route tf_hy4_route_m4", .symbol = "tf_hy4_route_m4", .image = "hy4_route", .threads = 128 }, .usage = sameS(128, 6144, 56, 56, 56, 56, 56), .later_pending = true },
    .{ .launch = .{ .name = "hy4_route tf_hy4_route_m4_plan1", .symbol = "tf_hy4_route_m4_plan1", .image = "hy4_route", .threads = 128 }, .usage = sameS(128, 6144, 56, 56, 56, 56, 56), .later_pending = true },

    // hy4_ihc: 1-launch ticket (R <= 64), 1-launch coop (R <= 16), 2-launch tiled, and 1-pass fused prefill
    .{ .launch = .{ .name = "hy4_ihc hy4_ihc_fused_boundary", .symbol = "hy4_ihc_fused_boundary", .image = "hy4_ihc", .threads = 256, .dynamic = 46416 }, .usage = sameS(256, 160, 147, 149, 149, 162, 162), .later_pending = true },
    .{ .launch = .{ .name = "hy4_ihc hy4_ihc_entry", .symbol = "hy4_ihc_entry", .image = "hy4_ihc", .threads = 256, .dynamic = 46416 }, .usage = sameS(256, 16, 144, 144, 144, 162, 162), .later_pending = true },
    .{ .launch = .{ .name = "hy4_ihc hy4_ihc_epilogue_head", .symbol = "hy4_ihc_epilogue_head", .image = "hy4_ihc", .threads = 256, .dynamic = 30032 }, .usage = sameS(256, 160, 150, 148, 148, 166, 166), .later_pending = true },
    .{ .launch = .{ .name = "hy4_ihc hy4_ihc_head_only", .symbol = "hy4_ihc_head_only", .image = "hy4_ihc", .threads = 256, .dynamic = 30032 }, .usage = sameS(256, 16, 148, 148, 148, 160, 160), .later_pending = true },
    .{ .launch = .{ .name = "hy4_ihc hy4_ihc_post_only", .symbol = "hy4_ihc_post_only", .image = "hy4_ihc", .threads = 256, .dynamic = 7440 }, .usage = sameS(256, 144, 80, 80, 80, 79, 79), .later_pending = true },
    .{ .launch = .{ .name = "hy4_ihc hy4_ihc_fused_boundary_coop", .symbol = "hy4_ihc_fused_boundary_coop", .image = "hy4_ihc", .threads = 256 }, .usage = sameS(256, 46336, 138, 140, 140, 154, 154), .later_pending = true },
    .{ .launch = .{ .name = "hy4_ihc hy4_ihc_entry_coop", .symbol = "hy4_ihc_entry_coop", .image = "hy4_ihc", .threads = 256 }, .usage = sameS(256, 45952, 139, 141, 141, 153, 153), .later_pending = true },
    .{ .launch = .{ .name = "hy4_ihc hy4_ihc_epilogue_head_coop", .symbol = "hy4_ihc_epilogue_head_coop", .image = "hy4_ihc", .threads = 256 }, .usage = sameS(256, 42240, 102, 104, 104, 117, 117), .later_pending = true },
    .{ .launch = .{ .name = "hy4_ihc hy4_ihc_head_only_coop", .symbol = "hy4_ihc_head_only_coop", .image = "hy4_ihc", .threads = 256 }, .usage = sameS(256, 41856, 107, 109, 109, 116, 116), .later_pending = true },
    .{ .launch = .{ .name = "hy4_ihc hy4_ihc_post_vec", .symbol = "hy4_ihc_post_vec", .image = "hy4_ihc", .threads = 256 }, .usage = sameS(256, 12320, 32, 46, 46, 54, 54), .later_pending = true },
    .{ .launch = .{ .name = "hy4_ihc hy4_ihc_pre_tiled_rb2", .symbol = "hy4_ihc_pre_tiled_rb2", .image = "hy4_ihc", .threads = 256, .dynamic = 7080 }, .usage = sameS(256, 0, 48, 48, 48, 56, 56), .later_pending = true },
    .{ .launch = .{ .name = "hy4_ihc hy4_ihc_pre_tiled_rb4", .symbol = "hy4_ihc_pre_tiled_rb4", .image = "hy4_ihc", .threads = 256, .dynamic = 13904 }, .usage = sameS(256, 0, 48, 48, 48, 63, 63), .later_pending = true },
    .{ .launch = .{ .name = "hy4_ihc hy4_ihc_head_tiled", .symbol = "hy4_ihc_head_tiled", .image = "hy4_ihc", .threads = 256, .dynamic = 27552 }, .usage = sameS(256, 0, 48, 48, 48, 61, 61), .later_pending = true },
    .{ .launch = .{ .name = "hy4_ihc hy4_ihc_fused_prefill_m8", .symbol = "hy4_ihc_fused_prefill_m8", .image = "hy4_ihc", .threads = 256 }, .usage = sameS(256, 18816, 38, 48, 48, 48, 48), .later_pending = true },
    .{ .launch = .{ .name = "hy4_ihc hy4_ihc_fused_prefill_m4", .symbol = "hy4_ihc_fused_prefill_m4", .image = "hy4_ihc", .threads = 256 }, .usage = sameS(256, 18816, 38, 48, 48, 48, 48), .later_pending = true },

    // hy4_dfly2_glue: 5-layer DFly2 drafter glue (S = 1 single-CTA row kernels, top-k extension, argmax, gather)
    .{ .launch = .{ .name = "hy4_dfly2_glue hy4_dfly_dual_rmsnorm_cat_s1", .symbol = "hy4_dfly_dual_rmsnorm_cat_s1", .image = "hy4_dfly2_glue", .threads = 384 }, .usage = sameS(384, 48, 29, 33, 33, 31, 31), .later_pending = true },
    .{ .launch = .{ .name = "hy4_dfly2_glue hy4_dfly_fusion_gated_add_s1", .symbol = "hy4_dfly_fusion_gated_add_s1", .image = "hy4_dfly2_glue", .threads = 384 }, .usage = sameS(384, 48, 38, 38, 38, 40, 40), .later_pending = true },
    .{ .launch = .{ .name = "hy4_dfly2_glue hy4_dfly_sublayer_boundary_s1", .symbol = "hy4_dfly_sublayer_boundary_s1", .image = "hy4_dfly2_glue", .threads = 384 }, .usage = sameS(384, 96, 40, 40, 40, 40, 40), .later_pending = true },
    .{ .launch = .{ .name = "hy4_dfly2_glue hy4_dfly_extend_topk_k", .symbol = "hy4_dfly_extend_topk_k", .image = "hy4_dfly2_glue", .threads = 128 }, .usage = sameS(128, 0, 32, 40, 40, 40, 40), .later_pending = true },
    .{ .launch = .{ .name = "hy4_dfly2_glue hy4_dfly_pick_argmax", .symbol = "hy4_dfly_pick_argmax", .image = "hy4_dfly2_glue", .threads = 64 }, .usage = sameS(64, 0, 12, 12, 12, 14, 14), .later_pending = true },
    .{ .launch = .{ .name = "hy4_dfly2_glue hy4_dfly_gather_hidden", .symbol = "hy4_dfly_gather_hidden", .image = "hy4_dfly2_glue", .threads = 384 }, .usage = sameS(384, 0, 16, 16, 16, 18, 18), .later_pending = true },

    // hy4_mla_attn: c1 tensor-core latent MLA attention (split + fold & exact 2-plane x16 / x16_qreg)
    .{ .launch = .{ .name = "hy4_mla_attn hy4_mla_dec_kt32", .symbol = "hy4_mla_dec_kt32", .image = "hy4_mla_attn", .threads = 256, .dynamic = 97424, .opt_in = true }, .usage = sameS(256, 0, 192, 192, 114, 167, 167), .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla_attn hy4_mla_dec_kt32_rows", .symbol = "hy4_mla_dec_kt32_rows", .image = "hy4_mla_attn", .threads = 256, .dynamic = 97424, .opt_in = true }, .usage = sameS(256, 0, 194, 194, 116, 167, 167), .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla_attn hy4_mla_dec_qreg_kt32", .symbol = "hy4_mla_dec_qreg_kt32", .image = "hy4_mla_attn", .threads = 256, .dynamic = 78992, .opt_in = true }, .usage = sameS(256, 0, 242, 242, 186, 224, 224), .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla_attn hy4_mla_dec_h32_kt16", .symbol = "hy4_mla_dec_h32_kt16", .image = "hy4_mla_attn", .threads = 256, .dynamic = 81936, .opt_in = true }, .usage = sameS(256, 0, 156, 156, 157, 185, 185), .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla_attn hy4_mla_dec_h32_qreg_kt16", .symbol = "hy4_mla_dec_h32_qreg_kt16", .image = "hy4_mla_attn", .threads = 256, .dynamic = 45072 }, .usage = sameS(256, 0, 205, 205, 207, 244, 244), .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla_attn hy4_mla_dec_x16", .symbol = "hy4_mla_dec_x16", .image = "hy4_mla_attn", .threads = 256, .dynamic = 81552, .opt_in = true }, .usage = sameS(256, 0, 111, 111, 110, 140, 140), .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla_attn hy4_mla_dec_x16_rows", .symbol = "hy4_mla_dec_x16_rows", .image = "hy4_mla_attn", .threads = 256, .dynamic = 81552, .opt_in = true }, .usage = sameS(256, 0, 113, 113, 112, 140, 140), .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla_attn hy4_mla_dec_x16_qreg", .symbol = "hy4_mla_dec_x16_qreg", .image = "hy4_mla_attn", .threads = 256, .dynamic = 48896 }, .usage = sameS(256, 0, 128, 128, 128, 128, 128), .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla_attn hy4_mla_dec_x16_qreg_rows", .symbol = "hy4_mla_dec_x16_qreg_rows", .image = "hy4_mla_attn", .threads = 256, .dynamic = 48896 }, .usage = sameS(256, 0, 128, 128, 128, 128, 128), .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla_attn hy4_mla_fold", .symbol = "hy4_mla_fold", .image = "hy4_mla_attn", .threads = 256 }, .usage = sameS(256, 16640, 64, 64, 64, 62, 62), .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla_attn hy4_mla_dec_kt32_paged", .symbol = "hy4_mla_dec_kt32_paged", .image = "hy4_mla_attn", .threads = 256, .dynamic = 97424, .opt_in = true }, .usage = sameS(256, 0, 194, 194, 114, 167, 167), .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla_attn hy4_mla_dec_qreg_kt32_paged", .symbol = "hy4_mla_dec_qreg_kt32_paged", .image = "hy4_mla_attn", .threads = 256, .dynamic = 78992, .opt_in = true }, .usage = sameS(256, 0, 244, 244, 188, 224, 224), .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla_attn hy4_mla_dec_h32_kt16_paged", .symbol = "hy4_mla_dec_h32_kt16_paged", .image = "hy4_mla_attn", .threads = 256, .dynamic = 81936, .opt_in = true }, .usage = sameS(256, 0, 156, 156, 157, 185, 185), .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla_attn hy4_mla_dec_h32_qreg_kt16_paged", .symbol = "hy4_mla_dec_h32_qreg_kt16_paged", .image = "hy4_mla_attn", .threads = 256, .dynamic = 45072 }, .usage = sameS(256, 0, 205, 205, 207, 244, 244), .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla_attn hy4_mla_dec_x16_paged", .symbol = "hy4_mla_dec_x16_paged", .image = "hy4_mla_attn", .threads = 256, .dynamic = 81552, .opt_in = true }, .usage = sameS(256, 0, 111, 111, 110, 139, 139), .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla_attn hy4_mla_dec_x16_qreg_paged", .symbol = "hy4_mla_dec_x16_qreg_paged", .image = "hy4_mla_attn", .threads = 256, .dynamic = 48896 }, .usage = sameS(256, 0, 128, 128, 128, 128, 128), .later_pending = true },

    // hy4_mla_c2: tensor-core absorb (bf16 + exact lo plane) and value expand + sigmoid gate
    .{ .launch = .{ .name = "hy4_mla_c2 hy4c2_absorb_d1_x", .symbol = "hy4c2_absorb_d1_x", .image = "hy4_mla_c2", .threads = 128, .dynamic = 8704 }, .usage = sameS(128, 0, 80, 80, 80, 78, 78), .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla_c2 hy4c2_absorb_d2_x", .symbol = "hy4c2_absorb_d2_x", .image = "hy4_mla_c2", .threads = 128, .dynamic = 14848 }, .usage = sameS(128, 0, 90, 90, 90, 88, 88), .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla_c2 hy4c2_absorb_p_x", .symbol = "hy4c2_absorb_p_x", .image = "hy4_mla_c2", .threads = 256, .dynamic = 33792 }, .usage = sameS(256, 0, 124, 124, 124, 126, 126), .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla_c2 hy4c2_expand_d1_f32lat", .symbol = "hy4c2_expand_d1_f32lat", .image = "hy4_mla_c2", .threads = 128, .dynamic = 27136 }, .usage = sameS(128, 0, 109, 109, 109, 118, 118), .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla_c2 hy4c2_expand_d2_f32lat", .symbol = "hy4c2_expand_d2_f32lat", .image = "hy4_mla_c2", .threads = 128, .dynamic = 51712, .opt_in = true }, .usage = sameS(128, 0, 113, 113, 113, 125, 125), .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla_c2 hy4c2_expand_d4k64s2_f32lat", .symbol = "hy4c2_expand_d4k64s2_f32lat", .image = "hy4_mla_c2", .threads = 128, .dynamic = 33280 }, .usage = sameS(128, 0, 94, 94, 94, 86, 86), .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla_c2 hy4c2_expand_p_f32lat", .symbol = "hy4c2_expand_p_f32lat", .image = "hy4_mla_c2", .threads = 256, .dynamic = 46080 }, .usage = sameS(256, 0, 128, 128, 128, 124, 124), .later_pending = true },

    // fp8_experts: Blackwell TMA 1D decode kernels (sm_120+)
    .{ .launch = .{ .name = "fp8_experts decode tma gate-up", .symbol = "_ZN14tf_fp8_experts21fp8_expert_tma_kernelILi2ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi", .image = "fp8_experts", .threads = 32, .min_sm = 120 }, .usage = five(32, 151, 0, 153, 0, 153, 0, 144, 16400, 144, 16400), .later_pending = true },
    .{ .launch = .{ .name = "fp8_experts decode tma down f32 s3", .symbol = "_ZN14tf_fp8_experts21fp8_expert_tma_kernelILi1ELi0ELi3ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi", .image = "fp8_experts", .threads = 32, .min_sm = 120 }, .usage = five(32, 111, 0, 111, 0, 96, 0, 106, 12320, 106, 12320), .later_pending = true },
    .{ .launch = .{ .name = "fp8_experts decode tma down f32 s2", .symbol = "_ZN14tf_fp8_experts21fp8_expert_tma_kernelILi1ELi0ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi", .image = "fp8_experts", .threads = 32, .min_sm = 120 }, .usage = five(32, 111, 0, 111, 0, 96, 0, 100, 8208, 100, 8208), .later_pending = true },
    .{ .launch = .{ .name = "fp8_experts decode tma down bf16 s2", .symbol = "_ZN14tf_fp8_experts21fp8_expert_tma_kernelILi1ELi3ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi", .image = "fp8_experts", .threads = 32, .min_sm = 120 }, .usage = five(32, 112, 0, 111, 0, 96, 0, 100, 8208, 100, 8208), .later_pending = true },
    .{ .launch = .{ .name = "fp8_experts prompt k64 gate-up", .symbol = "_ZN14tf_fp8_experts28fp8_expert_prompt_k64_kernelILi2ELi2ELi2ELi2ELb0ELb1EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi", .image = "fp8_experts", .threads = 128, .dynamic = 49176, .opt_in = true }, .usage = sameS(128, 0, 215, 215, 218, 220, 220), .later_pending = true },
    .{ .launch = .{ .name = "fp8_experts prompt k64 down f32", .symbol = "_ZN14tf_fp8_experts28fp8_expert_prompt_k64_kernelILi1ELi0ELi4ELi4ELb1ELb0EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi", .image = "fp8_experts", .threads = 128, .dynamic = 49176, .opt_in = true }, .usage = sameS(128, 0, 206, 206, 210, 218, 218), .later_pending = true },
    .{ .launch = .{ .name = "fp8_experts prompt k64 down bf16", .symbol = "_ZN14tf_fp8_experts28fp8_expert_prompt_k64_kernelILi1ELi3ELi4ELi4ELb1ELb0EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi", .image = "fp8_experts", .threads = 128, .dynamic = 49176, .opt_in = true }, .usage = sameS(128, 0, 206, 206, 210, 220, 220), .later_pending = true },

    // qmmf: FP8G 64x128 4-warp 2-stage prompt tile (2 CTAs/SM on sm_12x)
    .{ .launch = .{ .name = "qmmf FP8G prompt 64x128 bf16 unsplit", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi3ELi64ELi128ELi1ELi4ELi2ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 34944, .min_sm = 120 }, .usage = sameS(128, 0, 242, 242, 242, 168, 168), .later_pending = true },
    .{ .launch = .{ .name = "qmmf FP8G prompt 64x128 bf16 fused slices", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi3ELi64ELi128ELi1ELi4ELi2ELb0ELb0ELb1EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 34944, .min_sm = 120 }, .usage = sameS(128, 0, 255, 255, 255, 255, 255), .later_pending = true },
    .{ .launch = .{ .name = "qmmf FP8G prompt 64x128 fp32 unsplit", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi3ELi64ELi128ELi1ELi4ELi2ELb1ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 34944, .min_sm = 120 }, .usage = sameS(128, 0, 242, 242, 242, 170, 170), .later_pending = true },
    .{ .launch = .{ .name = "qmmf FP8G prompt 64x128 fp32 fused slices", .symbol = "_ZN7tf_qmmf11qmmf_kernelILi3ELi64ELi128ELi1ELi4ELi2ELb1ELb0ELb1EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .image = "qmmf", .threads = 128, .dynamic = 34944, .min_sm = 120 }, .usage = sameS(128, 0, 255, 255, 255, 255, 255), .later_pending = true },
};

test "hy4 wave1 records: every launch is registered once and fits every SM" {
    for (records) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(std.mem.eql(u8, active.launch.name, r.launch.name));
        try std.testing.expectEqual(@as(usize, 1), found);
        try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
    }
}
