//! Hy4-flash's weights on one rank of `world` (partition.zig): projection shares packed for the lane, experts sliced.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const config = @import("config.zig");
const engram_hash = @import("engram_hash.zig");
const names = @import("names.zig");
const partition = @import("partition.zig");

const Config = config.Config;
const Share = partition.Share;
const DeviceBuffer = cuda.DeviceBuffer;
const fp8 = cuda.fp8;
const qmmf = cuda.qmmf;
const fx8 = cuda.experts.Fp8Experts;
const bfr = core.bf16_round;

/// One rank of a tensor-parallel group: world 1, 2 or 4.
pub const Rank = struct { rank: usize = 0, world: usize = 1 };

/// A projection's share as the lane reads it: bf16 tiles, or e4m3 tiles with fp32 scale tiles (cuda/fp8.zig).
pub const Packed = struct {
    mode: qmmf.Mode,
    codes: []u8,
    scales: []f32 = &.{},
    n: usize,
    k: usize,

    pub fn deinit(p: *Packed, gpa: std.mem.Allocator) void {
        gpa.free(p.codes);
        gpa.free(p.scales);
        p.* = undefined;
    }
};

/// Rows `rows` of a stored projection; parts stack by rows into one packed projection.
pub const Part = struct { p: names.Proj, rows: Share };

fn inputs(p: names.Proj) usize {
    return switch (p) {
        .bf16 => |t| t.dim(1),
        .fp8 => |f| f.w.dim(1),
    };
}

/// A stored scale_inv (bf16 or fp32) widened to fp32: exact.
fn widened(gpa: std.mem.Allocator, t: core.checkpoint.Tensor) ![]f32 {
    const out = try gpa.alloc(f32, t.numel());
    errdefer gpa.free(out);
    try core.quant.fp8.widen(t, out, true);
    return out;
}

/// `parts` stacked by rows, inputs `cols` (whole 64-value groups); each format packed as stored, nothing requantized.
pub fn pack(gpa: std.mem.Allocator, parts: []const Part, cols: Share) !Packed {
    if (parts.len == 0) return error.UnexpectedTensor;
    const mode: qmmf.Mode = if (parts[0].p == .bf16) .bf16 else .fp8g;
    const width = inputs(parts[0].p);
    var n: usize = 0;
    for (parts) |q| {
        const same = (q.p == .bf16) == (mode == .bf16);
        if (!same or inputs(q.p) != width) return error.UnexpectedTensor;
        if (q.rows.len == 0 or q.rows.lo + q.rows.len > q.p.rows()) return error.SplitOutOfRange;
        if (parts.len > 1 and q.rows.len % 64 != 0) return error.SplitNotOnBlocks; // scale tiles hold 64 rows
        n += q.rows.len;
    }
    if (cols.len == 0 or cols.lo + cols.len > width or cols.lo % qmmf.group != 0 or cols.len % qmmf.group != 0) return error.SplitNotOnBlocks;
    const k = cols.len;
    const npad = fp8.padded(n);
    const elem: usize = if (mode == .bf16) 2 else 1;
    const rows = try gpa.alloc(u8, n * k * elem);
    defer gpa.free(rows);
    var at: usize = 0;
    for (parts) |q| {
        const bytes = switch (q.p) {
            .bf16 => |t| t.bytes,
            .fp8 => |f| f.w.bytes,
        };
        for (q.rows.lo..q.rows.lo + q.rows.len) |r| {
            @memcpy(rows[at * k * elem ..][0 .. k * elem], bytes[(r * width + cols.lo) * elem ..][0 .. k * elem]);
            at += 1;
        }
    }
    const codes = try gpa.alloc(u8, npad * k * elem);
    errdefer gpa.free(codes);
    if (mode == .bf16) {
        qmmf.packBf16(@alignCast(std.mem.bytesAsSlice(u16, codes)), @alignCast(std.mem.bytesAsSlice(u16, rows)), n, k);
        return .{ .mode = .bf16, .codes = codes, .n = n, .k = k };
    }
    fp8.packCodes(codes, rows, n, k);
    const kg = k / fp8.group;
    const scales = try gpa.alloc(f32, npad * kg);
    errdefer gpa.free(scales);
    @memset(scales, 1.0);
    var row: usize = 0;
    for (parts) |q| {
        const f = q.p.fp8;
        const inv = try widened(gpa, f.s);
        defer gpa.free(inv);
        const one = if (parts.len == 1) scales else try gpa.alloc(f32, fp8.padded(q.rows.len) * kg);
        defer if (parts.len > 1) gpa.free(one);
        fp8.packScalesAt(one, inv, f.w.dim(0), f.w.dim(1), q.rows.lo, cols.lo, q.rows.len, k);
        if (parts.len > 1) {
            const tiles = q.rows.len / 64; // a part's own tiles, past its padding
            @memcpy(scales[(row / 64) * kg * 64 ..][0 .. tiles * kg * 64], one[0 .. tiles * kg * 64]);
        }
        row += q.rows.len;
    }
    return .{ .mode = .fp8g, .codes = codes, .scales = scales, .n = n, .k = k };
}

/// A rank's dense FFN or shared expert: gate rows then up rows in one call, down's matching inputs.
pub const Ffn = struct { gate_up: qmmf.Weight, down: qmmf.Weight, width: usize };

pub const Moe = struct {
    router: u64, // bf16 [experts, hidden] as stored
    router_q: qmmf.Weight, // router packed for the bf16 tensor-core lane (prefill widths)
    bias: u64, // fp32 [experts] (e_score_correction_bias)
    down_shared: qmmf.Weight, // stacked [latent_down, shared.gate, shared.up] [latent + 2*sw, hidden] bf16 (R >= 2)
    down_shared_half: ?qmmf.Weight = null, // stacked [latent_down_half (latent/2), shared.gate, shared.up] bf16 (R == 1, world == 2)
    latent_up: qmmf.Weight, // routed_expert_up_proj [hidden, latent], replicated: it takes a rank's partial sums
    experts: cuda.experts.Layer, // fp8g, each expert's share of its intermediate width
    shared_down: qmmf.Weight, // shared.down [hidden, sw] bf16
    shared_width: usize,
};

/// One layer's MLP on one rank: the norm before it, its body, and the norm after the all-reduce.
pub const Mlp = struct {
    norm: u64, // post_attention_layernorm, bf16 [hidden]
    post_norm: u64, // post_fc2_norm (dense) or combine_output_norm (MoE), fp32 [hidden]
    post_norm_bf16: u64, // same post_norm in bf16 [hidden] for fused hy4_ihc boundary
    body: union(enum) { dense: Ffn, moe: Moe },
};

/// One Engram layer's weights on one rank: its head norms, value + 4 key column shares, rank-0 biases, and gate norms.
pub const Engram = struct {
    slot: usize,
    part: partition.Engram,
    map: []const u8, // lookup_table int32 [120,818], kept on host for n-gram hashing
    head_norms: u64, // bf16 [part.heads.len, 96]
    proj: qmmf.Weight, // bf16 [5 * hidden, part.cols.len]: value_proj, then key_projs[0..4]
    bias: ?u64, // bf16 [5 * hidden] on rank 0, null on peers
    norm_keys: u64, // bf16 [4, hidden]
    norm_queries: u64, // bf16 [4, hidden]
    pre_gamma: u64, // bf16 [hidden]
    stream_scales: u64, // bf16 [4, hidden]
};

/// The experts of one layer as cuda/experts.zig's fp8g layer takes them, each sliced to one rank's width.
pub const HostExperts = struct {
    up: []u32, // [E][w/32][L/32][2][32][2][4]: gate and up interleaved
    down: []u32, // [E][L/32][w/32][1][32][2][4]
    up_scale: []f32, // [E][2][w/128][L/128] (gate, then up)
    down_scale: []f32, // [E][1][L/128][w/128]

    pub fn deinit(h: *HostExperts, gpa: std.mem.Allocator) void {
        gpa.free(h.up);
        gpa.free(h.down);
        gpa.free(h.up_scale);
        gpa.free(h.down_scale);
        h.* = undefined;
    }
};

fn bf16At(bytes: []const u8, i: usize) f32 {
    return bfr.value(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little));
}

/// The fused routed experts sliced by rank: gate_up_proj [E, 2w, L] stacks gate rows then up rows (chunk(2, -1)).
pub fn hostExperts(gpa: std.mem.Allocator, m: names.Moe, share: Share) !HostExperts {
    const e = m.gate_up.dim(0);
    const two = m.gate_up.dim(1);
    const l = m.gate_up.dim(2);
    const full = two / 2;
    const w = share.len;
    const b = fp8.block;
    if (w == 0 or share.lo % b != 0 or w % b != 0 or share.lo + w > full or l % b != 0) return error.SplitNotOnBlocks;
    if (m.down.dim(0) != e or m.down.dim(1) != l or m.down.dim(2) != full) return error.UnexpectedTensor;
    var h: HostExperts = .{ .up = &.{}, .down = &.{}, .up_scale = &.{}, .down_scale = &.{} };
    errdefer h.deinit(gpa);
    h.up = try gpa.alloc(u32, e * w * l / 2);
    h.down = try gpa.alloc(u32, e * w * l / 4);
    h.up_scale = try gpa.alloc(f32, e * 2 * (w / b) * (l / b));
    h.down_scale = try gpa.alloc(f32, e * (l / b) * (w / b));
    const cols = try gpa.alloc(u8, l * w);
    defer gpa.free(cols);
    const lb = l / b;
    const fb = full / b;
    for (0..e) |x| {
        const gu = m.gate_up.bytes[x * two * l ..][0 .. two * l];
        try fx8.packGateUp(gpa, h.up[x * w * l / 2 ..][0 .. w * l / 2], gu[share.lo * l ..][0 .. w * l], gu[(full + share.lo) * l ..][0 .. w * l], w, l);
        const dn = m.down.bytes[x * l * full ..][0 .. l * full];
        for (0..l) |r| @memcpy(cols[r * w ..][0..w], dn[r * full + share.lo ..][0..w]);
        fx8.packOne(h.down[x * w * l / 4 ..][0 .. w * l / 4], cols, l, w);
        // scales: gate's row blocks, then up's, each the share's blocks of the expert's [2w/128, L/128]
        const us = h.up_scale[x * 2 * (w / b) * lb ..];
        for (0..2) |half| for (0..w / b) |rb| for (0..lb) |cb| {
            const src = (x * 2 * fb + half * fb + share.lo / b + rb) * lb + cb;
            us[(half * (w / b) + rb) * lb + cb] = bf16At(m.gate_up_s.bytes, src);
        };
        const ds = h.down_scale[x * lb * (w / b) ..];
        for (0..lb) |rb| for (0..w / b) |cb| {
            ds[rb * (w / b) + cb] = bf16At(m.down_s.bytes, (x * lb + rb) * fb + share.lo / b + cb);
        };
    }
    return h;
}

/// Device copies a rank's layers point into; freed together.
pub const Weights = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    host_bufs: std.ArrayList([]u8) = .empty,
    bytes: usize = 0,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver) Weights {
        return .{ .gpa = gpa, .d = d };
    }

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        for (w.host_bufs.items) |hb| w.gpa.free(hb);
        w.host_bufs.deinit(w.gpa);
        w.* = undefined;
    }

    fn stackTensors(w: *Weights, ts: []const core.checkpoint.Tensor) !u64 {
        var total: usize = 0;
        for (ts) |t| total += t.bytes.len;
        const buf = try w.gpa.alloc(u8, total);
        defer w.gpa.free(buf);
        var at: usize = 0;
        for (ts) |t| {
            @memcpy(buf[at..][0..t.bytes.len], t.bytes);
            at += t.bytes.len;
        }
        return w.upload(buf);
    }

    pub fn upload(w: *Weights, bytes: []const u8) !u64 {
        var b = try DeviceBuffer.fromHost(w.d, bytes);
        errdefer b.free();
        try w.buffers.append(w.gpa, b);
        w.bytes += bytes.len;
        return b.ptr;
    }

    /// A packed projection on the device as the lane's weight view.
    pub fn weight(w: *Weights, p: Packed) !qmmf.Weight {
        const codes = try w.upload(p.codes);
        if (p.mode == .bf16) return cuda.modelopt.bf16Weight(codes, p.n, p.k);
        return fp8.weight(codes, try w.upload(std.mem.sliceAsBytes(p.scales)), p.n, p.k);
    }

    /// `parts` stacked by rows and inputs `cols`, packed and uploaded.
    pub fn proj(w: *Weights, parts: []const Part, cols: Share) !qmmf.Weight {
        var p = try pack(w.gpa, parts, cols);
        defer p.deinit(w.gpa);
        return w.weight(p);
    }

    /// Horizontally concatenates two bf16 tensors `a[rows, a_cols]` and `b[rows, b_cols]`, packed for `qmmf` BF16.
    pub fn projHConcat(w: *Weights, a: core.checkpoint.Tensor, a_cols: Share, b: core.checkpoint.Tensor, b_cols: Share, rows: Share) !qmmf.Weight {
        if (a.dtype != .bf16 or b.dtype != .bf16) return error.UnexpectedTensor;
        const n = rows.len;
        const k = a_cols.len + b_cols.len;
        if (n == 0 or k == 0 or k % qmmf.group != 0) return error.SplitNotOnBlocks;
        if (rows.lo + n > a.dim(0) or rows.lo + n > b.dim(0)) return error.SplitOutOfRange;
        if (a_cols.lo + a_cols.len > a.dim(1) or b_cols.lo + b_cols.len > b.dim(1)) return error.SplitOutOfRange;
        const aw = a.dim(1);
        const bw = b.dim(1);
        const npad = fp8.padded(n);
        const raw = try w.gpa.alloc(u16, n * k);
        defer w.gpa.free(raw);
        for (0..n) |i| {
            const r = rows.lo + i;
            @memcpy(std.mem.sliceAsBytes(raw[i * k ..][0..a_cols.len]), a.bytes[(r * aw + a_cols.lo) * 2 ..][0 .. a_cols.len * 2]);
            @memcpy(std.mem.sliceAsBytes(raw[i * k + a_cols.len ..][0..b_cols.len]), b.bytes[(r * bw + b_cols.lo) * 2 ..][0 .. b_cols.len * 2]);
        }
        const packed_buf = try w.gpa.alloc(u16, npad * k);
        defer w.gpa.free(packed_buf);
        qmmf.packBf16(packed_buf, raw, n, k);
        const codes = try w.upload(std.mem.sliceAsBytes(packed_buf));
        return cuda.modelopt.bf16Weight(codes, n, k);
    }

    fn widen(w: *Weights, t: core.checkpoint.Tensor) !u64 {
        if (t.dtype != .bf16) return error.UnexpectedTensor;
        const v = try w.gpa.alloc(f32, t.numel());
        defer w.gpa.free(v);
        for (v, 0..) |*o, i| o.* = bf16At(t.bytes, i);
        return w.upload(std.mem.sliceAsBytes(v));
    }

    /// `parts` stacked by rows and inputs `cols`, packed, uploaded, and pinned to single K-slice (`sk = 1`).
    pub fn projPin1(w: *Weights, parts: []const Part, cols: Share) !qmmf.Weight {
        return qmmf.pinned(try w.proj(parts, cols), 1);
    }

    /// gate and up [width, hidden] by the rank's rows, down [hidden, width] by its inputs.
    fn ffn(w: *Weights, gate: core.checkpoint.Tensor, up: core.checkpoint.Tensor, down: core.checkpoint.Tensor, at: Rank) !Ffn {
        const width = gate.dim(0);
        const s = try partition.ffn(width, at.rank, at.world);
        return .{
            .gate_up = try w.projPin1(&.{ .{ .p = .{ .bf16 = gate }, .rows = s }, .{ .p = .{ .bf16 = up }, .rows = s } }, .{ .lo = 0, .len = gate.dim(1) }),
            .down = try w.projPin1(&.{.{ .p = .{ .bf16 = down }, .rows = .{ .lo = 0, .len = down.dim(0) } }}, s),
            .width = s.len,
        };
    }

    /// Layer `l`'s MLP on rank `at`: replicated norms, router and latent projections; FFN and expert widths split.
    pub fn mlp(w: *Weights, l: names.Layer, c: Config, at: Rank) !Mlp {
        return w.mlpEx(l, c, at, false);
    }

    pub fn mlpEx(w: *Weights, l: names.Layer, c: Config, at: Rank, shard_replicated: bool) !Mlp {
        const all: Share = .{ .lo = 0, .len = c.hidden };
        switch (l.mlp) {
            .dense => |m| return .{
                .norm = try w.upload(l.post_attn_norm.bytes),
                .post_norm = try w.widen(m.post_norm),
                .post_norm_bf16 = try w.upload(m.post_norm.bytes),
                .body = .{ .dense = try w.ffn(m.gate, m.up, m.down, at) },
            },
            .moe => |m| {
                const share = try partition.expert(c, at.rank, at.world);
                const ss = try partition.ffn(m.shared.gate.dim(0), at.rank, at.world);
                const do_shard_ld = shard_replicated and at.world == 2 and (c.latent % 128 == 0);
                const ld_half: Share = .{ .lo = at.rank * (c.latent / 2), .len = c.latent / 2 };
                const ds_half: ?qmmf.Weight = if (do_shard_ld)
                    try w.projPin1(&.{
                        .{ .p = .{ .bf16 = m.latent_down }, .rows = ld_half },
                        .{ .p = .{ .bf16 = m.shared.gate }, .rows = ss },
                        .{ .p = .{ .bf16 = m.shared.up }, .rows = ss },
                    }, all)
                else
                    null;
                var h = try hostExperts(w.gpa, m, share);
                defer h.deinit(w.gpa);
                return .{
                    .norm = try w.upload(l.post_attn_norm.bytes),
                    .post_norm = try w.widen(m.shared.post_norm),
                    .post_norm_bf16 = try w.upload(m.shared.post_norm.bytes),
                    .body = .{ .moe = .{
                        .router = try w.upload(m.router.bytes),
                        .router_q = try w.proj(&.{.{ .p = .{ .bf16 = m.router }, .rows = .{ .lo = 0, .len = c.experts } }}, all),
                        .bias = try w.upload(m.bias.bytes),
                        .down_shared = try w.projPin1(&.{
                            .{ .p = .{ .bf16 = m.latent_down }, .rows = .{ .lo = 0, .len = c.latent } },
                            .{ .p = .{ .bf16 = m.shared.gate }, .rows = ss },
                            .{ .p = .{ .bf16 = m.shared.up }, .rows = ss },
                        }, all),
                        .down_shared_half = ds_half,
                        .latent_up = try w.projPin1(&.{.{ .p = .{ .bf16 = m.latent_up }, .rows = all }}, .{ .lo = 0, .len = c.latent }),
                        .experts = .{
                            .format = .fp8g,
                            .up = try w.upload(std.mem.sliceAsBytes(h.up)),
                            .down = try w.upload(std.mem.sliceAsBytes(h.down)),
                            .up_scale = try w.upload(std.mem.sliceAsBytes(h.up_scale)),
                            .down_scale = try w.upload(std.mem.sliceAsBytes(h.down_scale)),
                            .width = share.len,
                            .dims = c.latent,
                            .experts = c.experts,
                            .limit = c.swiglu_limit,
                        },
                        .shared_down = try w.projPin1(&.{.{ .p = .{ .bf16 = m.shared.down }, .rows = .{ .lo = 0, .len = m.shared.down.dim(0) } }}, ss),
                        .shared_width = ss.len,
                    } },
                };
            },
        }
    }

    /// Layer `l`'s Engram block (at `layer` index) on rank `at`: checks lookup_table, slices heads and input columns.
    pub fn engram(w: *Weights, l: names.Layer, c: Config, layer_idx: usize, at: Rank) !Engram {
        const slot = c.engramSlot(layer_idx) orelse return error.NotEngramLayer;
        const g = l.engram orelse return error.NotEngramLayer;
        var why: config.Why = .{};
        try engram_hash.check(c, &why);
        try engram_hash.checkMap(g.map.bytes, &why);
        const part = try partition.Engram.of(c, at.rank, at.world);
        const map = try w.gpa.dupe(u8, g.map.bytes);
        errdefer w.gpa.free(map);
        try w.host_bufs.append(w.gpa, map);

        const all: Share = .{ .lo = 0, .len = c.hidden };
        const parts = [5]Part{
            .{ .p = .{ .bf16 = g.value }, .rows = all },
            .{ .p = .{ .bf16 = g.keys[0] }, .rows = all },
            .{ .p = .{ .bf16 = g.keys[1] }, .rows = all },
            .{ .p = .{ .bf16 = g.keys[2] }, .rows = all },
            .{ .p = .{ .bf16 = g.keys[3] }, .rows = all },
        };
        const biases = [5]core.checkpoint.Tensor{ g.value_bias, g.key_biases[0], g.key_biases[1], g.key_biases[2], g.key_biases[3] };
        return .{
            .slot = slot,
            .part = part,
            .map = map,
            .head_norms = try w.stackTensors(g.head_norms[part.heads.lo..][0..part.heads.len]),
            .proj = try w.proj(&parts, part.cols),
            .bias = if (part.bias) try w.stackTensors(&biases) else null,
            .norm_keys = try w.stackTensors(&g.norm_keys),
            .norm_queries = try w.stackTensors(&g.norm_queries),
            .pre_gamma = try w.upload(g.pre_gamma.bytes),
            .stream_scales = try w.stackTensors(&g.stream_scales),
        };
    }
};

test {
    _ = @import("cuda_weights_test.zig");
}
