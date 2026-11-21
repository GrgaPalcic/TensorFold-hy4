//! Hy4-flash CUDA launch records (`hy4_hc`, `hy4_engram` and `hy4_rows`, 256 threads) from `nvcc 13.0` across 5 SMs.
const registry = @import("registry.zig");
const U = @import("limits.zig").Usage;

fn all(regs80: u32, regs86: u32, regs89: u32, regs120: u32, regs121: u32, static: u32) [5]U {
    return .{
        .{ .regs = regs80, .static = static, .bound = 256 },
        .{ .regs = regs86, .static = static, .bound = 256 },
        .{ .regs = regs89, .static = static, .bound = 256 },
        .{ .regs = regs120, .static = static, .bound = 256 },
        .{ .regs = regs121, .static = static, .bound = 256 },
    };
}

pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "hy4_hc hy4_hc_pre", .symbol = "hy4_hc_pre", .image = "hy4_hc", .threads = 256 }, .usage = all(80, 80, 80, 80, 80, 304), .later_pending = true },
    .{ .launch = .{ .name = "hy4_hc hy4_hc_pre_bad_nan", .symbol = "hy4_hc_pre_bad_nan", .image = "hy4_hc", .threads = 256 }, .usage = all(80, 80, 80, 80, 80, 304), .later_pending = true },
    .{ .launch = .{ .name = "hy4_hc hy4_hc_pre_bad_repeat", .symbol = "hy4_hc_pre_bad_repeat", .image = "hy4_hc", .threads = 256 }, .usage = all(80, 80, 80, 80, 80, 304), .later_pending = true },
    .{ .launch = .{ .name = "hy4_hc hy4_hc_pre_bad_tail", .symbol = "hy4_hc_pre_bad_tail", .image = "hy4_hc", .threads = 256 }, .usage = all(80, 80, 80, 80, 80, 304), .later_pending = true },
    .{ .launch = .{ .name = "hy4_hc hy4_hc_pre_bad_sums", .symbol = "hy4_hc_pre_bad_sums", .image = "hy4_hc", .threads = 256 }, .usage = all(48, 44, 44, 46, 46, 304), .later_pending = true },
    .{ .launch = .{ .name = "hy4_hc hy4_hc_head", .symbol = "hy4_hc_head", .image = "hy4_hc", .threads = 256 }, .usage = all(48, 48, 48, 48, 48, 176), .later_pending = true },
    .{ .launch = .{ .name = "hy4_hc hy4_hc_head_bad_nan", .symbol = "hy4_hc_head_bad_nan", .image = "hy4_hc", .threads = 256 }, .usage = all(48, 48, 48, 48, 48, 176), .later_pending = true },
    .{ .launch = .{ .name = "hy4_hc hy4_hc_head_bad_repeat", .symbol = "hy4_hc_head_bad_repeat", .image = "hy4_hc", .threads = 256 }, .usage = all(48, 48, 48, 48, 48, 176), .later_pending = true },
    .{ .launch = .{ .name = "hy4_hc hy4_hc_head_bad_tail", .symbol = "hy4_hc_head_bad_tail", .image = "hy4_hc", .threads = 256 }, .usage = all(48, 48, 48, 48, 48, 176), .later_pending = true },
    .{ .launch = .{ .name = "hy4_hc hy4_hc_head_bad_sums", .symbol = "hy4_hc_head_bad_sums", .image = "hy4_hc", .threads = 256 }, .usage = all(36, 40, 40, 40, 40, 176), .later_pending = true },
    .{ .launch = .{ .name = "hy4_hc hy4_hc_merge", .symbol = "hy4_hc_merge", .image = "hy4_hc", .threads = 256 }, .usage = all(22, 22, 22, 22, 22, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_hc hy4_hc_merge_bad_nan", .symbol = "hy4_hc_merge_bad_nan", .image = "hy4_hc", .threads = 256 }, .usage = all(22, 22, 22, 22, 22, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_hc hy4_hc_merge_bad_repeat", .symbol = "hy4_hc_merge_bad_repeat", .image = "hy4_hc", .threads = 256 }, .usage = all(22, 22, 22, 22, 22, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_hc hy4_hc_merge_bad_tail", .symbol = "hy4_hc_merge_bad_tail", .image = "hy4_hc", .threads = 256 }, .usage = all(22, 22, 22, 22, 22, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_engram hy4_engram_embed", .symbol = "hy4_engram_embed", .image = "hy4_engram", .threads = 256 }, .usage = all(26, 25, 25, 21, 21, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_engram hy4_engram_embed_bad_nan", .symbol = "hy4_engram_embed_bad_nan", .image = "hy4_engram", .threads = 256 }, .usage = all(21, 21, 21, 21, 21, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_engram hy4_engram_embed_bad_repeat", .symbol = "hy4_engram_embed_bad_repeat", .image = "hy4_engram", .threads = 256 }, .usage = all(21, 21, 21, 21, 21, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_engram hy4_engram_embed_bad_tail", .symbol = "hy4_engram_embed_bad_tail", .image = "hy4_engram", .threads = 256 }, .usage = all(20, 21, 21, 22, 22, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_engram hy4_engram_embed_bad_sums", .symbol = "hy4_engram_embed_bad_sums", .image = "hy4_engram", .threads = 256 }, .usage = all(26, 25, 25, 21, 21, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_engram hy4_engram_bias", .symbol = "hy4_engram_bias", .image = "hy4_engram", .threads = 256 }, .usage = all(20, 18, 18, 20, 20, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_engram hy4_engram_bias_bad_nan", .symbol = "hy4_engram_bias_bad_nan", .image = "hy4_engram", .threads = 256 }, .usage = all(20, 18, 18, 20, 20, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_engram hy4_engram_bias_bad_repeat", .symbol = "hy4_engram_bias_bad_repeat", .image = "hy4_engram", .threads = 256 }, .usage = all(20, 18, 18, 20, 20, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_engram hy4_engram_bias_bad_tail", .symbol = "hy4_engram_bias_bad_tail", .image = "hy4_engram", .threads = 256 }, .usage = all(20, 18, 18, 20, 20, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_engram hy4_engram_gate", .symbol = "hy4_engram_gate", .image = "hy4_engram", .threads = 256 }, .usage = all(62, 62, 62, 64, 64, 340), .later_pending = true },
    .{ .launch = .{ .name = "hy4_engram hy4_engram_gate_bad_nan", .symbol = "hy4_engram_gate_bad_nan", .image = "hy4_engram", .threads = 256 }, .usage = all(62, 48, 48, 64, 64, 340), .later_pending = true },
    .{ .launch = .{ .name = "hy4_engram hy4_engram_gate_bad_repeat", .symbol = "hy4_engram_gate_bad_repeat", .image = "hy4_engram", .threads = 256 }, .usage = all(60, 64, 64, 64, 64, 340), .later_pending = true },
    .{ .launch = .{ .name = "hy4_engram hy4_engram_gate_bad_tail", .symbol = "hy4_engram_gate_bad_tail", .image = "hy4_engram", .threads = 256 }, .usage = all(62, 62, 62, 64, 64, 340), .later_pending = true },
    .{ .launch = .{ .name = "hy4_engram hy4_engram_gate_bad_sums", .symbol = "hy4_engram_gate_bad_sums", .image = "hy4_engram", .threads = 256 }, .usage = all(52, 52, 52, 64, 64, 340), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_wide", .symbol = "hy4_rows_wide", .image = "hy4_rows", .threads = 256 }, .usage = all(128, 128, 128, 126, 126, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_wide_bad_nan", .symbol = "hy4_rows_wide_bad_nan", .image = "hy4_rows", .threads = 256 }, .usage = all(128, 128, 128, 126, 126, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_wide_bad_repeat", .symbol = "hy4_rows_wide_bad_repeat", .image = "hy4_rows", .threads = 256 }, .usage = all(128, 128, 128, 126, 126, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_wide_bad_tail", .symbol = "hy4_rows_wide_bad_tail", .image = "hy4_rows", .threads = 256 }, .usage = all(128, 128, 128, 128, 128, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_wide_bad_sums", .symbol = "hy4_rows_wide_bad_sums", .image = "hy4_rows", .threads = 256 }, .usage = all(128, 127, 127, 123, 123, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_narrow", .symbol = "hy4_rows_narrow", .image = "hy4_rows", .threads = 256 }, .usage = all(58, 59, 59, 62, 62, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_narrow_bad_nan", .symbol = "hy4_rows_narrow_bad_nan", .image = "hy4_rows", .threads = 256 }, .usage = all(58, 59, 59, 62, 62, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_narrow_bad_repeat", .symbol = "hy4_rows_narrow_bad_repeat", .image = "hy4_rows", .threads = 256 }, .usage = all(58, 59, 59, 62, 62, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_narrow_bad_tail", .symbol = "hy4_rows_narrow_bad_tail", .image = "hy4_rows", .threads = 256 }, .usage = all(56, 58, 58, 61, 61, 0), .later_pending = true },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_narrow_bad_sums", .symbol = "hy4_rows_narrow_bad_sums", .image = "hy4_rows", .threads = 256 }, .usage = all(57, 58, 58, 62, 62, 0), .later_pending = true },
};
