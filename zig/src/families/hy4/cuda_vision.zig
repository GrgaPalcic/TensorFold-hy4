//! Hy4-flash CUDA Vision Tower (`vit.*`, 27-layer BF16 SigLIP ViT + `HYV4FlashVisionPatchMerger`).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const vcfg = @import("vision_config.zig");
const cuda_rows = @import("cuda_rows.zig");

const lt = cuda.cublaslt;
const DeviceBuffer = cuda.DeviceBuffer;
const Stream = cuda.Stream;

pub const Config = vcfg.Config;
pub const Limits = vcfg.Limits;
pub const Grid = struct { h: u32, w: u32 };

const workspace_bytes: usize = 32 << 20;
const max_depth: usize = 32;

const Block = struct {
    norm1_w: u64,
    norm1_b: u64,
    qkv_w: u64, // bf16 [3 * hidden, hidden] (q_proj, k_proj, v_proj concatenated)
    qkv_b: u64, // bf16 [3 * hidden]
    proj_w: u64,
    proj_b: u64,
    norm2_w: u64,
    norm2_b: u64,
    fc1_w: u64,
    fc1_b: u64,
    fc2_w: u64,
    fc2_b: u64,
};

const QvFns = struct {
    embed: cuda.Function,
    layernorm: cuda.Function,
    attention72: cuda.Function,
    bias_gelu: cuda.Function,
    bias_residual: cuda.Function,
    bias_cast: cuda.Function,
};

fn aligned(n: usize) usize {
    return std.mem.alignForward(usize, n, 256);
}

fn bf16Bits(v: f32) u16 {
    return core.bf16_round.bits(v);
}

pub fn weightArenaBytes(c: Config) usize {
    const h = c.hidden;
    const inter = c.intermediate;
    const m = c.mergedWidth();
    const out_h = c.out_hidden;
    var n: usize = aligned(h * vcfg.patch_width * 2) + aligned(h * 2) + aligned(vcfg.num_positions * h * 2);
    n += c.depth * (4 * aligned(h * 2) + aligned(3 * h * h * 2) + aligned(3 * h * 2) +
        aligned(h * h * 2) + aligned(h * 2) + aligned(inter * h * 2) + aligned(inter * 2) +
        aligned(h * inter * 2) + aligned(h * 2));
    n += aligned(h * 2) + aligned(m * m * 2) + aligned(m * 2) + aligned(out_h * m * 2) + aligned(out_h * 2) + aligned(out_h * 2);
    return n;
}

pub fn scratchArenaBytes(c: Config, max_p: usize) usize {
    const h = c.hidden;
    const inter = c.intermediate;
    const m = c.mergedWidth();
    const m4 = max_p / 4;
    return aligned(max_p * vcfg.patch_width * 2) + // d_px
        aligned(max_p * h * 2) + // x
        aligned(max_p * h * 2) + // h_norm
        aligned(@max(max_p * @max(3 * h, inter), m4 * m) * 4) + // fp32 GEMM acc
        3 * aligned(max_p * h * 2) + // q, k, v
        aligned(@max(max_p * h, m4 * c.out_hidden) * 2) + // att / merged proj2 out
        aligned(@max(max_p * inter, m4 * m) * 2) + // mid / merge_gather
        aligned(max_p * 4 * 4) + // d_idx
        aligned(max_p * 4 * 4) + // d_wts
        workspace_bytes;
}

pub const Tower = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    c: Config,
    qv_mod: cuda.Module,
    f: QvFns,
    lib: lt.Library,
    arena: DeviceBuffer,
    at: usize = 0,
    scratch: u64 = 0,
    patch_w: u64 = 0,
    patch_b: u64 = 0,
    pos: u64 = 0, // [16385, 1152]; row 0 is CLS, patch grid starts at pos + 1152 * 2
    blocks: [max_depth]Block = undefined,
    perceive_norm_w: u64 = 0,
    proj0_w: u64 = 0, // [4608, 2, 2, 1152] bf16 (permuted from [4608, 1152, 2, 2])
    proj0_b: u64 = 0,
    proj2_w: u64 = 0, // [3072, 4608] bf16
    proj2_b: u64 = 0,
    image_newline: u64 = 0, // [3072] bf16

    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, c: Config) !*Tower {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const t = try gpa.create(Tower);
        errdefer gpa.destroy(t);
        t.* = .{ .gpa = gpa, .d = d, .c = c, .qv_mod = undefined, .f = undefined, .lib = undefined, .arena = undefined };
        t.lib = try lt.Library.open();
        errdefer t.lib.close();
        t.qv_mod = try cuda.Module.load(d, cuda.kernels.qwen_vision);
        errdefer t.qv_mod.unload();
        t.f = .{
            .embed = try t.qv_mod.function("tf_qv_embed"),
            .layernorm = try t.qv_mod.function("tf_qv_layernorm"),
            .attention72 = try t.qv_mod.function("tf_qv_attention72"),
            .bias_gelu = try t.qv_mod.function("tf_qv_bias_gelu"),
            .bias_residual = try t.qv_mod.function("tf_qv_bias_residual"),
            .bias_cast = try t.qv_mod.function("tf_qv_bias_cast"),
        };
        const total = weightArenaBytes(c) + scratchArenaBytes(c, vcfg.max_patches);
        t.arena = try DeviceBuffer.alloc(d, total);
        errdefer t.arena.free();
        try t.arena.fill8(0, null);
        try t.loadWeights(gpa, io, dir);
        t.scratch = try t.take(scratchArenaBytes(c, vcfg.max_patches));
        return t;
    }

    pub fn deinit(t: *Tower) void {
        t.arena.free();
        t.qv_mod.unload();
        t.lib.close();
        t.gpa.destroy(t);
    }

    fn take(t: *Tower, len: usize) !u64 {
        const cur = aligned(t.at);
        if (cur + len > t.arena.len) return error.ArenaExhausted;
        t.at = cur + len;
        return t.arena.ptr + cur;
    }

    fn put(t: *Tower, bytes: []const u8) !u64 {
        const ptr = try t.take(bytes.len);
        try cuda.copy_source.synchronousCopy(t.d, ptr, bytes, std.heap.page_allocator);
        return ptr;
    }

    fn loadWeights(t: *Tower, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
        var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, "vit.");
        defer core.checkpoint.cache.close(&ck);
        const h = t.c.hidden;
        const inter = t.c.intermediate;
        const m = t.c.mergedWidth();
        const out_h = t.c.out_hidden;

        t.patch_w = try t.put((try ck.expect("vit.embeddings.patch_embedding.weight", .bf16, &.{ h, 3, 16, 16 })).bytes);
        t.patch_b = try t.put((try ck.expect("vit.embeddings.patch_embedding.bias", .bf16, &.{h})).bytes);
        t.pos = try t.put((try ck.expect("vit.embeddings.position_embedding.weight", .bf16, &.{ vcfg.num_positions, h })).bytes);

        const qkv_w_buf = try gpa.alloc(u8, 3 * h * h * 2);
        defer gpa.free(qkv_w_buf);
        const qkv_b_buf = try gpa.alloc(u8, 3 * h * 2);
        defer gpa.free(qkv_b_buf);
        for (0..t.c.depth) |i| {
            var nb: [128]u8 = undefined;
            const b = &t.blocks[i];
            b.norm1_w = try t.put((try ck.expect(try std.fmt.bufPrint(&nb, "vit.layers.{d}.input_layernorm.weight", .{i}), .bf16, &.{h})).bytes);
            b.norm1_b = try t.put((try ck.expect(try std.fmt.bufPrint(&nb, "vit.layers.{d}.input_layernorm.bias", .{i}), .bf16, &.{h})).bytes);
            inline for (.{ "q_proj", "k_proj", "v_proj" }, 0..) |pn, pi| {
                const pw = try ck.expect(try std.fmt.bufPrint(&nb, "vit.layers.{d}.self_attn.{s}.weight", .{ i, pn }), .bf16, &.{ h, h });
                @memcpy(qkv_w_buf[pi * h * h * 2 ..][0 .. h * h * 2], pw.bytes);
                const pb = try ck.expect(try std.fmt.bufPrint(&nb, "vit.layers.{d}.self_attn.{s}.bias", .{ i, pn }), .bf16, &.{h});
                @memcpy(qkv_b_buf[pi * h * 2 ..][0 .. h * 2], pb.bytes);
            }
            b.qkv_w = try t.put(qkv_w_buf);
            b.qkv_b = try t.put(qkv_b_buf);
            b.proj_w = try t.put((try ck.expect(try std.fmt.bufPrint(&nb, "vit.layers.{d}.self_attn.o_proj.weight", .{i}), .bf16, &.{ h, h })).bytes);
            b.proj_b = try t.put((try ck.expect(try std.fmt.bufPrint(&nb, "vit.layers.{d}.self_attn.o_proj.bias", .{i}), .bf16, &.{h})).bytes);
            b.norm2_w = try t.put((try ck.expect(try std.fmt.bufPrint(&nb, "vit.layers.{d}.post_attention_layernorm.weight", .{i}), .bf16, &.{h})).bytes);
            b.norm2_b = try t.put((try ck.expect(try std.fmt.bufPrint(&nb, "vit.layers.{d}.post_attention_layernorm.bias", .{i}), .bf16, &.{h})).bytes);
            b.fc1_w = try t.put((try ck.expect(try std.fmt.bufPrint(&nb, "vit.layers.{d}.mlp.dense_h_to_4h.weight", .{i}), .bf16, &.{ inter, h })).bytes);
            b.fc1_b = try t.put((try ck.expect(try std.fmt.bufPrint(&nb, "vit.layers.{d}.mlp.dense_h_to_4h.bias", .{i}), .bf16, &.{inter})).bytes);
            b.fc2_w = try t.put((try ck.expect(try std.fmt.bufPrint(&nb, "vit.layers.{d}.mlp.dense_4h_to_h.weight", .{i}), .bf16, &.{ h, inter })).bytes);
            b.fc2_b = try t.put((try ck.expect(try std.fmt.bufPrint(&nb, "vit.layers.{d}.mlp.dense_4h_to_h.bias", .{i}), .bf16, &.{h})).bytes);
        }

        t.perceive_norm_w = try t.put((try ck.expect("vit.perceive.norm.weight", .bf16, &.{h})).bytes);
        const p0 = try ck.expect("vit.perceive.proj.0.weight", .bf16, &.{ m, h, 2, 2 });
        // Permute [out=4608, C=1152, kh=2, kw=2] -> [out=4608, kh=2, kw=2, C=1152] to match `vit_merge_gather`.
        const p0_perm = try gpa.alloc(u16, m * m);
        defer gpa.free(p0_perm);
        const p0_src: []align(1) const u16 = std.mem.bytesAsSlice(u16, p0.bytes);
        for (0..m) |oc| for (0..h) |ic| for (0..4) |sub| {
            p0_perm[oc * m + sub * h + ic] = p0_src[(oc * h + ic) * 4 + sub];
        };
        t.proj0_w = try t.put(std.mem.sliceAsBytes(p0_perm));
        t.proj0_b = try t.put((try ck.expect("vit.perceive.proj.0.bias", .bf16, &.{m})).bytes);
        t.proj2_w = try t.put((try ck.expect("vit.perceive.proj.2.weight", .bf16, &.{ out_h, m, 1, 1 })).bytes);
        t.proj2_b = try t.put((try ck.expect("vit.perceive.proj.2.bias", .bf16, &.{out_h})).bytes);
        t.image_newline = try t.put((try ck.expect("vit.perceive.image_newline", .bf16, &.{out_h})).bytes);
        if (ck.unused() != 0) return error.UnusedVisionTensors;
    }

    /// Encodes `grids` (`pixels` flat across images) into `out` (`bf16 [sum((gh/2)*(gw/2+1)), 3072]`).
    pub fn encode(
        t: *Tower,
        s: Stream,
        rows_op: *const cuda_rows.Rows,
        norms: *const cuda.norms.Norms,
        pixels: []const f32,
        grids: []const Grid,
        out: u64,
    ) !void {
        var n: usize = 0;
        for (grids) |g| {
            if (g.h == 0 or g.w == 0 or g.h % vcfg.merge != 0 or g.w % vcfg.merge != 0) return error.BadImageGrid;
            n += @as(usize, g.h) * g.w;
        }
        if (n == 0 or n > vcfg.max_patches or pixels.len != n * vcfg.patch_width) return error.BadImagePatches;

        const gpa = t.gpa;
        const c = t.c;
        const h = c.hidden;
        const heads = c.heads;
        const hd = c.headDim();
        const inter = c.intermediate;
        const m = c.mergedWidth();
        const out_h = c.out_hidden;

        const px16 = try gpa.alloc(u16, n * vcfg.patch_width);
        defer gpa.free(px16);
        for (px16, pixels) |*dst, v| dst.* = bf16Bits(v);
        const idx = try gpa.alloc(i32, n * 4);
        defer gpa.free(idx);
        const wts = try gpa.alloc(f32, n * 4);
        defer gpa.free(wts);
        var p_off: usize = 0;
        for (grids) |g| {
            const len = @as(usize, g.h) * g.w;
            vcfg.positions(g.h, g.w, idx[p_off * 4 ..][0 .. len * 4], wts[p_off * 4 ..][0 .. len * 4]);
            p_off += len;
        }

        var cur: u64 = t.scratch;
        const end = t.scratch + scratchArenaBytes(c, vcfg.max_patches);
        const sub = struct {
            fn f(p: *u64, len: usize) u64 {
                const r = p.*;
                p.* += aligned(len);
                return r;
            }
        }.f;
        const m4_max = n / 4;
        const d_px = sub(&cur, n * vcfg.patch_width * 2);
        const x = sub(&cur, n * h * 2);
        const h_norm = sub(&cur, n * h * 2);
        const acc = sub(&cur, @max(n * @max(3 * h, inter), m4_max * m) * 4);
        const q = sub(&cur, n * h * 2);
        const k = sub(&cur, n * h * 2);
        const v = sub(&cur, n * h * 2);
        const att = sub(&cur, @max(n * h, m4_max * out_h) * 2);
        const mid = sub(&cur, @max(n * inter, m4_max * m) * 2);
        const d_idx = sub(&cur, n * 4 * 4);
        const d_wts = sub(&cur, n * 4 * 4);
        const ws = cur;
        const ws_len = end - cur;

        var mm: Gemms = .{ .lib = &t.lib };
        defer mm.deinit();
        errdefer cuda.completion.drain(t.d, s, "hy4: failed ViT encode did not drain");

        try cuda.copy_source.enqueue(t.d, d_px, std.mem.sliceAsBytes(px16), s);
        try cuda.copy_source.enqueue(t.d, d_idx, std.mem.sliceAsBytes(idx), s);
        try cuda.copy_source.enqueue(t.d, d_wts, std.mem.sliceAsBytes(wts), s);

        try mm.run(n, h, vcfg.patch_width, d_px, t.patch_w, acc, ws, ws_len, s);
        try go(s, t.f.embed, n, 1, 256, .{ acc, t.patch_b, t.pos + h * 2, d_idx, d_wts, x, cint(h) });

        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        for (t.blocks[0..c.depth]) |b| {
            try go(s, t.f.layernorm, n, 1, 256, .{ x, b.norm1_w, b.norm1_b, h_norm, cint(h), c.eps });
            try mm.run(n, 3 * h, h, h_norm, b.qkv_w, acc, ws, ws_len, s);
            try rows_op.vitQkv(s, acc, b.qkv_b, q, k, v, n, heads, hd);
            var s0: usize = 0;
            for (grids) |g| {
                const len = @as(usize, g.h) * g.w;
                try go(s, t.f.attention72, (len + 63) / 64, heads, 64, .{ q, k, v, att, cint(n), cint(heads), cint(s0), cint(len), scale });
                s0 += len;
            }
            try mm.run(n, h, h, att, b.proj_w, acc, ws, ws_len, s);
            try go(s, t.f.bias_residual, n, 1, 256, .{ acc, b.proj_b, x, cint(h) });
            try go(s, t.f.layernorm, n, 1, 256, .{ x, b.norm2_w, b.norm2_b, h_norm, cint(h), c.eps });
            try mm.run(n, inter, h, h_norm, b.fc1_w, acc, ws, ws_len, s);
            try go(s, t.f.bias_gelu, n, 1, 256, .{ acc, b.fc1_b, mid, cint(inter), @as(c_int, 0) });
            try mm.run(n, h, inter, mid, b.fc2_w, acc, ws, ws_len, s);
            try go(s, t.f.bias_residual, n, 1, 256, .{ acc, b.fc2_b, x, cint(h) });
        }

        // Per-image PatchMerger: RMSNorm -> 2x2 merge conv -> GELU -> 1x1 out conv -> append image_newline
        var patch_at: usize = 0;
        var out_tok_at: usize = 0;
        for (grids) |g| {
            const len = @as(usize, g.h) * g.w;
            const mh: usize = g.h / vcfg.merge;
            const mw: usize = g.w / vcfg.merge;
            const m4 = mh * mw;
            try norms.rmsParts(s, x + patch_at * h * 2, h, 0, t.perceive_norm_w, 0, h_norm, h, 0, null, len, 1, h, c.eps, false);
            try rows_op.vitMergeGather(s, h_norm, mid, g.h, g.w, h);
            try mm.run(m4, m, m, mid, t.proj0_w, acc, ws, ws_len, s);
            try go(s, t.f.bias_gelu, m4, 1, 256, .{ acc, t.proj0_b, mid, cint(m), @as(c_int, 0) });
            try mm.run(m4, out_h, m, mid, t.proj2_w, acc, ws, ws_len, s);
            try go(s, t.f.bias_cast, m4, 1, 256, .{ acc, t.proj2_b, att, cint(out_h) });
            try rows_op.vitAppendNewline(s, att, t.image_newline, out + out_tok_at * out_h * 2, mh, mw, out_h);
            patch_at += len;
            out_tok_at += mh * (mw + 1);
        }
        try s.synchronize();
    }
};

fn cint(x: usize) c_int {
    return @intCast(x);
}

fn go(s: Stream, f: cuda.Function, gx: usize, gy: usize, block: u32, args: anytype) !void {
    var a: cuda.Args = .{};
    inline for (args) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(gx), .y = @intCast(gy) }, .block = .{ .x = block } }, s, &a);
}

const Gemms = struct {
    lib: *const lt.Library,
    plans: [16]?struct { m: usize, n: usize, k: usize, lin: lt.Linear } = @splat(null),

    fn run(g: *Gemms, m: usize, n: usize, k: usize, x: u64, w: u64, out: u64, ws: u64, ws_len: usize, s: Stream) !void {
        for (&g.plans) |*p| {
            if (p.*) |*have| {
                if (have.m == m and have.n == n and have.k == k) return have.lin.run(x, w, out, ws, ws_len, s.handle);
                continue;
            }
            p.* = .{ .m = m, .n = n, .k = k, .lin = try lt.Linear.init(g.lib, m, n, k, .f32, ws_len) };
            return p.*.?.lin.run(x, w, out, ws, ws_len, s.handle);
        }
        return error.TooManyGemmShapes;
    }

    fn deinit(g: *Gemms) void {
        for (&g.plans) |*p| if (p.*) |*have| have.lin.deinit();
    }
};

/// Server-facing Vision handle matching `native/cuda_vision_offer.zig`'s `template_wraps` contract.
pub const Vision = struct {
    pub const template_wraps = true;

    gpa: std.mem.Allocator,
    io: std.Io,
    c: Config,
    tower: ?*Tower = null, // non-null on rank 0; followers receive encoded features via TP broadcast
    marks: struct { open: u32, image: u32, close: u32 },
    mark_ids: [3]u32,
    limits: Limits,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, rank: usize, limits: Limits) !*Vision {
        const c = (try Config.read(gpa, io, dir)) orelse return error.NoHy4Vision;
        if (limits.image_tokens < 2 or limits.max_images == 0 or limits.max_images > 64) return error.BadImageLimits;
        const v = try gpa.create(Vision);
        errdefer gpa.destroy(v);
        const tower = if (rank == 0) try Tower.load(gpa, io, d, dir, c) else null;
        v.* = .{
            .gpa = gpa,
            .io = io,
            .c = c,
            .tower = tower,
            .marks = .{ .open = c.image_start, .image = c.image_token, .close = c.image_end },
            .mark_ids = .{ c.image_start, c.image_token, c.image_end },
            .limits = limits,
        };
        return v;
    }

    pub fn deinit(v: *Vision) void {
        if (v.tower) |t| t.deinit();
        v.gpa.destroy(v);
    }

    pub fn prepare(v: *const Vision, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) !core.picture.patches.Prepared {
        return vcfg.prepare(v.io, a, bytes, v.c, @min(max_tokens, v.limits.image_tokens));
    }

    pub const explain = core.picture.patches.problem;
};

test "cuda_vision declarations and arena geometry compile on host" {
    const c: Config = .{};
    try std.testing.expect(weightArenaBytes(c) > 880 << 20);
    std.testing.refAllDecls(@This());
}
