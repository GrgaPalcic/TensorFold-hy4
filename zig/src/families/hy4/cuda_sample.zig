//! GPU-side BF16 logit gather and keyed `sample.cu` draw scratch for Hy4-flash (`temperature > 0`).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const cuda_rows = @import("cuda_rows.zig");
const cuda_tp = @import("cuda_tp.zig");
const cuda_prof = @import("cuda_prof.zig");

const Stream = cuda.Stream;
const DeviceBuffer = cuda.DeviceBuffer;

pub const LogitMode = enum(u8) { full_f32, greedy, sample_bf16 };

/// True when `s` can draw directly on GPU via `sample.cu` (positive temperature, no logprobs/penalties/mask).
pub fn isFastSampled(s: *const lanes.Stream, mask: []const u32) bool {
    if (s.logprobs != null or s.penalties.active() or mask.len != 0) return false;
    return if (s.sampling) |sp| sp.temperature > 0 else false;
}

/// Device buffers backing 1-collective BF16 vocab gather and keyed `sample.cu` sampling up to `max_batch` rows.
pub const Scratch = struct {
    max_batch: usize,
    vocab: usize,
    world: usize,
    bf16_local: DeviceBuffer,
    bf16_stage: ?DeviceBuffer = null,
    bf16_full: ?DeviceBuffer = null,
    rules: DeviceBuffer,
    meta: DeviceBuffer,
    tok: DeviceBuffer,
    nonfinite: DeviceBuffer,
    scratch: DeviceBuffer,

    pub fn init(d: *const cuda.Driver, max_batch: usize, vocab: usize, vocab_local: usize, world: usize) !Scratch {
        var bf16_local = try DeviceBuffer.alloc(d, max_batch * vocab_local * 2);
        errdefer bf16_local.free();
        var bf16_stage: ?DeviceBuffer = null;
        var bf16_full: ?DeviceBuffer = null;
        if (world > 1) {
            bf16_stage = try DeviceBuffer.alloc(d, world * max_batch * vocab_local * 2);
            errdefer bf16_stage.?.free();
            bf16_full = try DeviceBuffer.alloc(d, max_batch * vocab * 2);
        }
        errdefer if (bf16_stage) |*b| b.free();
        errdefer if (bf16_full) |*b| b.free();
        var rules = try DeviceBuffer.alloc(d, max_batch * @sizeOf(cuda.sample.Rule));
        errdefer rules.free();
        var meta = try DeviceBuffer.alloc(d, max_batch * 4);
        errdefer meta.free();
        var tok = try DeviceBuffer.alloc(d, max_batch * 4);
        errdefer tok.free();
        var nonfinite = try DeviceBuffer.alloc(d, 4);
        errdefer nonfinite.free();
        const scratch = try DeviceBuffer.alloc(d, max_batch * cuda.sample.Draws.row_scratch);
        return .{
            .max_batch = max_batch,
            .vocab = vocab,
            .world = world,
            .bf16_local = bf16_local,
            .bf16_stage = bf16_stage,
            .bf16_full = bf16_full,
            .rules = rules,
            .meta = meta,
            .tok = tok,
            .nonfinite = nonfinite,
            .scratch = scratch,
        };
    }

    pub fn deinit(sc: *Scratch) void {
        sc.scratch.free();
        sc.nonfinite.free();
        sc.tok.free();
        sc.meta.free();
        sc.rules.free();
        if (sc.bf16_full) |*b| b.free();
        if (sc.bf16_stage) |*b| b.free();
        sc.bf16_local.free();
        sc.* = undefined;
    }

    pub fn logitsPtr(sc: *const Scratch) u64 {
        return if (sc.bf16_full) |b| b.ptr else sc.bf16_local.ptr;
    }

    /// Rounds local fp32 logits to bf16, gathers across ranks in one collective, and interleaves on rank 0.
    pub fn prepare(
        sc: *const Scratch,
        d: *const cuda.Driver,
        rows_ops: *const cuda_rows.Rows,
        s: Stream,
        logits_local: u64,
        vocab_local: usize,
        logits_rows: usize,
        tp_sess: ?*cuda_tp.Session,
        rank: usize,
        prof_ptr: ?*cuda_prof.Prof,
    ) !void {
        if (tp_sess) |ts| {
            try rows_ops.f32ToBf16(s, logits_local, sc.bf16_local.ptr, logits_rows * vocab_local, 0);
            try ts.gatherBf16(s, sc.bf16_local.ptr, sc.bf16_stage.?.ptr, logits_rows * vocab_local);
            if (rank == 0) {
                const zero: u32 = 0;
                try cuda.copy_source.enqueue(d, sc.nonfinite.ptr, std.mem.asBytes(&zero), s);
                try rows_ops.interleaveBf16(s, sc.bf16_stage.?.ptr, sc.bf16_full.?.ptr, vocab_local, logits_rows, sc.world, sc.nonfinite.ptr);
            }
            if (prof_ptr) |p| p.mark(s, .head_comm);
        } else {
            const zero: u32 = 0;
            try cuda.copy_source.enqueue(d, sc.nonfinite.ptr, std.mem.asBytes(&zero), s);
            try rows_ops.f32ToBf16(s, logits_local, sc.bf16_local.ptr, logits_rows * vocab_local, sc.nonfinite.ptr);
            if (prof_ptr) |p| p.mark(s, .argmax);
        }
    }

    /// Draws `rules.len` tokens on GPU from `logitsPtr()` via `sample.cu`, checking nonfinite logits.
    pub fn draw(
        sc: *const Scratch,
        d: *const cuda.Driver,
        draws: *const cuda.sample.Draws,
        s: Stream,
        tp_sess: ?*cuda_tp.Session,
        prof_ptr: ?*cuda_prof.Prof,
        rules: []const cuda.sample.Rule,
        metas: []const i32,
        out: []u32,
    ) !void {
        const rows = rules.len;
        if (rows == 0 or rows > sc.max_batch or metas.len != rows or out.len < rows) return error.BadStepShape;
        try cuda.copy_source.enqueue(d, sc.rules.ptr, std.mem.sliceAsBytes(rules), s);
        try cuda.copy_source.enqueue(d, sc.meta.ptr, std.mem.sliceAsBytes(metas), s);
        try draws.drawRows(s, sc.logitsPtr(), sc.vocab, sc.rules.ptr, sc.meta.ptr, sc.tok.ptr, rows, sc.scratch.ptr);
        try s.synchronize();
        if (prof_ptr) |p| p.finish(s);
        if (tp_sess) |ts| try ts.check();
        var nonfinite: u32 = 0;
        try sc.nonfinite.download(0, std.mem.asBytes(&nonfinite));
        if (nonfinite != 0) return error.NonfiniteLogits;
        try sc.tok.download(0, std.mem.sliceAsBytes(out[0..rows]));
    }
};

test "cuda_sample declarations compile on host" {
    std.testing.refAllDecls(@This());
}
