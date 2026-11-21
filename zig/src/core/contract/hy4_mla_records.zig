//! The hy4_mla image's launches (hy4_mla.cu, families/hy4/cuda_mla.zig): nvcc 13.0's usage, five-SM fatbins.
const std = @import("std");
const registry = @import("registry.zig");

pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "hy4_mla tf_hy4_mla_absorb", .symbol = "tf_hy4_mla_absorb", .image = "hy4_mla", .threads = 128 }, .usage = .{ .{ .regs = 104, .static = 12288, .bound = 128 }, .{ .regs = 106, .static = 12288, .bound = 128 }, .{ .regs = 106, .static = 12288, .bound = 128 }, .{ .regs = 88, .static = 12288, .bound = 128 }, .{ .regs = 94, .static = 12288, .bound = 128 } }, .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla tf_hy4_mla_attn", .symbol = "tf_hy4_mla_attn", .image = "hy4_mla", .threads = 256, .dynamic = 56672, .opt_in = true }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 256 }, .{ .regs = 64, .static = 0, .bound = 256 }, .{ .regs = 64, .static = 0, .bound = 256 }, .{ .regs = 78, .static = 0, .bound = 256 }, .{ .regs = 78, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla tf_hy4_mla_attn_rows", .symbol = "tf_hy4_mla_attn_rows", .image = "hy4_mla", .threads = 256, .dynamic = 56672, .opt_in = true }, .usage = .{ .{ .regs = 74, .static = 0, .bound = 256 }, .{ .regs = 77, .static = 0, .bound = 256 }, .{ .regs = 77, .static = 0, .bound = 256 }, .{ .regs = 80, .static = 0, .bound = 256 }, .{ .regs = 80, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla tf_hy4_mla_value", .symbol = "tf_hy4_mla_value", .image = "hy4_mla", .threads = 128 }, .usage = .{ .{ .regs = 60, .static = 8320, .bound = 128 }, .{ .regs = 56, .static = 8320, .bound = 128 }, .{ .regs = 56, .static = 8320, .bound = 128 }, .{ .regs = 56, .static = 8320, .bound = 128 }, .{ .regs = 56, .static = 8320, .bound = 128 } }, .later_pending = true },
    .{ .launch = .{ .name = "hy4_mla tf_hy4_mla_post", .symbol = "tf_hy4_mla_post", .image = "hy4_mla", .threads = 256 }, .usage = .{ .{ .regs = 40, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 } }, .later_pending = true },
};

test "hy4_mla records: every launch is registered once and fits every SM" {
    for (records) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(std.mem.eql(u8, active.launch.name, r.launch.name));
        try std.testing.expectEqual(@as(usize, 1), found);
        try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
    }
}

test "the attention's dynamic shared bytes are cuda_mla.attn_smem's (tf_hy4::kAttnSmem)" {
    try std.testing.expectEqual(@as(u32, 8 * 576 * 4 + 32 * 578 * 2 + 8 * 32 * 4 + 32 * 4 + 3 * 8 * 4), records[1].launch.dynamic);
    try std.testing.expectEqual(records[1].launch.dynamic, records[2].launch.dynamic); // tf_hy4_mla_attn_rows
}
