//! Tensor-core absorbed latent MLA attention (`hy4_mla_attn.cu`): split+fold decode/verify and single-split prefill.
const std = @import("std");
const cuda = @import("cuda");

const Stream = cuda.Stream;
const Function = cuda.Function;
const DeviceBuffer = cuda.DeviceBuffer;

pub const row = 576;
// Split+fold (`max_batch = 64`) keeps decode/verify row bits independent of batch width; prompt passes use one split.
const max_split_rows = 64;
const max_batched_rows = 64;
const max_rows = 4096; // ROW_STREAM zeros
const max_hb = 4; // head blocks of 16 at world 1 (64 sliding heads)
const max_ns = 18; // up to 2176 keys (DFly2) / 128 = 17 splits

fn smem(mt: u32, kt: u32, qreg: bool) u32 {
    return (if (qreg) 0 else mt * 16 * 576 * 2) + 2 * kt * 576 * 2 + mt * 16 * kt * 2 + (8 / (mt * kt / 8)) * mt * 16 * (kt + 8) * 4 + 2 * kt * 4 + 6 * mt * 16 * 4 + 128 + 16;
}

fn smemX(mt: u32, kt: u32, qreg: bool) u32 {
    const base = (if (qreg) 0 else mt * 16 * 576 * 2 * 2) + 2 * kt * 576 * 2 + mt * 16 * kt * 2 * 2 + (8 / (mt * kt / 8)) * mt * 16 * (kt + 8) * 4 + 2 * kt * 4 + 6 * mt * 16 * 4 + 128 + 16;
    return if (qreg) ((base + 127) & ~@as(u32, 127)) + 4 * 16 * 32 * 2 else base;
}

/// Reads `TF_HY4_C1_EXACT` (default 1: 2-plane FP32-exact `x16` / `x16_qreg`; 0: 1-plane bf16-P).
fn readExact() bool {
    const v = @import("profile").lookup("TF_HY4_C1_EXACT") orelse return true;
    return v.len == 0 or v[0] != '0';
}

fn dependentLaunches(d: *const cuda.Driver) bool {
    var dev: cuda.abi.Device = 0;
    var major: c_int = 0;
    if (d.api.cuCtxGetDevice(&dev) != cuda.abi.success) return false;
    return d.api.cuDeviceGetAttribute(&major, .compute_capability_major, dev) == cuda.abi.success and major >= 9;
}

pub const C1 = struct {
    mod: cuda.Module,
    dec: Function,
    dec_rows: Function,
    qreg32: Function,
    h32: Function,
    h32q: Function,
    fold: Function,
    x16: Function,
    x16_rows: Function,
    x16_qreg: ?Function = null,
    dec_paged: Function,
    qreg32_paged: Function,
    h32_paged: Function,
    h32q_paged: Function,
    x16_paged: Function,
    x16_qreg_paged: ?Function = null,
    x16_qreg_rows: ?Function = null,
    exact: bool,
    pdl: bool = false,
    pm: DeviceBuffer,
    pl: DeviceBuffer,
    po: DeviceBuffer,
    rs: DeviceBuffer,

    pub fn load(d: *const cuda.Driver) !C1 {
        var mod = try cuda.Module.load(d, cuda.kernels.hy4_mla_attn);
        errdefer mod.unload();
        var c: C1 = undefined;
        c.mod = mod;
        c.exact = readExact();
        c.pdl = dependentLaunches(d);
        c.dec = try mod.function("hy4_mla_dec_kt32");
        try c.dec.allowDynamicShared(smem(1, 32, false));
        c.dec_rows = try mod.function("hy4_mla_dec_kt32_rows");
        try c.dec_rows.allowDynamicShared(smem(1, 32, false));
        c.qreg32 = try mod.function("hy4_mla_dec_qreg_kt32");
        try c.qreg32.allowDynamicShared(smem(1, 32, true));
        c.h32 = try mod.function("hy4_mla_dec_h32_kt16");
        try c.h32.allowDynamicShared(smem(2, 16, false));
        c.h32q = try mod.function("hy4_mla_dec_h32_qreg_kt16");
        try c.h32q.allowDynamicShared(smem(2, 16, true));
        c.fold = try mod.function("hy4_mla_fold");
        c.x16 = try mod.function("hy4_mla_dec_x16");
        try c.x16.allowDynamicShared(smemX(1, 16, false));
        c.x16_rows = try mod.function("hy4_mla_dec_x16_rows");
        try c.x16_rows.allowDynamicShared(smemX(1, 16, false));
        c.x16_qreg = mod.function("hy4_mla_dec_x16_qreg") catch null;
        if (c.x16_qreg) |fq| try fq.allowDynamicShared(smemX(1, 16, true));
        c.dec_paged = try mod.function("hy4_mla_dec_kt32_paged");
        try c.dec_paged.allowDynamicShared(smem(1, 32, false));
        c.qreg32_paged = try mod.function("hy4_mla_dec_qreg_kt32_paged");
        try c.qreg32_paged.allowDynamicShared(smem(1, 32, true));
        c.h32_paged = try mod.function("hy4_mla_dec_h32_kt16_paged");
        try c.h32_paged.allowDynamicShared(smem(2, 16, false));
        c.h32q_paged = try mod.function("hy4_mla_dec_h32_qreg_kt16_paged");
        try c.h32q_paged.allowDynamicShared(smem(2, 16, true));
        c.x16_paged = try mod.function("hy4_mla_dec_x16_paged");
        try c.x16_paged.allowDynamicShared(smemX(1, 16, false));
        c.x16_qreg_paged = mod.function("hy4_mla_dec_x16_qreg_paged") catch null;
        if (c.x16_qreg_paged) |fqp| try fqp.allowDynamicShared(smemX(1, 16, true));
        c.x16_qreg_rows = mod.function("hy4_mla_dec_x16_qreg_rows") catch null;
        if (c.x16_qreg_rows) |fq| try fq.allowDynamicShared(smemX(1, 16, true));
        const nf = max_split_rows * max_hb * max_ns * 16;
        c.pm = try DeviceBuffer.alloc(d, nf * 4);
        errdefer c.pm.free();
        c.pl = try DeviceBuffer.alloc(d, nf * 4);
        errdefer c.pl.free();
        c.po = try DeviceBuffer.alloc(d, nf * 512 * 4);
        errdefer c.po.free();
        const zeros: [max_rows * 4]u8 = @splat(0);
        c.rs = try DeviceBuffer.fromHost(d, &zeros);
        return c;
    }

    pub fn unload(c: *C1) void {
        c.rs.free();
        c.po.free();
        c.pl.free();
        c.pm.free();
        c.mod.unload();
    }
};

/// Keys: kind 0 sliding window 512 (slot pos % ring), 1 causal (p < 2048), 2 token list `tok` [rows][ntok].
pub const Keys = struct {
    kind: c_int,
    ring: usize = 0,
    tok: u64 = 0,
    ntok: usize = 0,
    single_split: bool = false,
    kv8: bool = false,
    page_table: u64 = 0,
    pt_stride: usize = 0,
    row_stream: u64 = 0,
};

fn ci(v: usize) c_int {
    return @intCast(v);
}

pub fn attn(c: *const C1, s: Stream, qa: u64, qlo: usize, heads: usize, cache: u64, k: Keys, pos: u64, sinks: u64, out: u64, rows: usize) !void {
    return attnInner(c, s, qa, qlo, heads, cache, false, k, pos, sinks, out, rows);
}

/// Batched multi-stream decode: `caches` is device u64 [rows], each row's MLA cache base pointer (rows <= 64).
pub fn attnRows(c: *const C1, s: Stream, qa: u64, qlo: usize, heads: usize, caches: u64, k: Keys, pos: u64, sinks: u64, out: u64, rows: usize) !void {
    return attnInner(c, s, qa, qlo, heads, caches, true, k, pos, sinks, out, rows);
}

fn attnInner(c: *const C1, s: Stream, qa: u64, qlo: usize, heads: usize, cache: u64, per_row: bool, k: Keys, pos: u64, sinks: u64, out: u64, rows: usize) !void {
    if (rows == 0) return;
    if (rows > max_rows) return error.RowShape;
    if (per_row and rows > max_batched_rows) return error.RowShape;
    const mk: usize = switch (k.kind) {
        0 => 512,
        1 => 2048,
        else => k.ntok,
    };
    // A linear cache (ring 0): any ring past every position keeps pos % ring == pos; one stream, so ctx is unused.
    const ring: usize = if (k.ring != 0) k.ring else std.math.maxInt(i32);
    const paged = k.page_table != 0 and k.kind != 0;
    var f = if (paged) c.dec_paged else if (per_row) c.dec_rows else c.dec;
    var sm = smem(1, 32, false);
    var nq: usize = 16;
    var ch: usize = if (k.kind == 0) 64 else 128;
    const one_split = k.single_split or rows > max_split_rows;
    if (one_split) {
        ch = @max(32, (mk + 31) / 32 * 32);
        if (!per_row) {
            if (heads % 32 == 0) {
                nq = 32;
                if (rows >= 512) {
                    f = if (paged) c.h32q_paged else c.h32q;
                    sm = smem(2, 16, true);
                } else {
                    f = if (paged) c.h32_paged else c.h32;
                    sm = smem(2, 16, false);
                }
            } else if (rows >= 1024) {
                f = if (paged) c.qreg32_paged else c.qreg32;
                sm = smem(1, 32, true);
            }
        }
    }
    if (c.exact) {
        const want_qreg = one_split or (k.kind != 0 and rows >= 4);
        const qfn = if (paged) c.x16_qreg_paged else if (per_row) c.x16_qreg_rows else c.x16_qreg;
        if (want_qreg and qfn != null) {
            f = qfn.?;
            sm = smemX(1, 16, true);
        } else {
            f = if (paged) c.x16_paged else if (per_row) c.x16_rows else c.x16;
            sm = smemX(1, 16, false);
        }
        nq = 16;
        ch = if (one_split) @max(16, (mk + 15) / 16 * 16) else if (k.kind == 0) 64 else 128;
    }
    const ns = (mk + ch - 1) / ch;
    const hb = (heads + nq - 1) / nq;
    if (ns > 1 and (hb > max_hb or ns > max_ns)) return error.RowShape;
    var a: cuda.Args = .{};
    a.add(qa);
    a.add(ci(heads));
    a.add(cache);
    inline for (.{ k.kind, ci(ring), ci(ring) }) |v| a.add(v);
    inline for (.{ if (k.row_stream != 0) k.row_stream else c.rs.ptr, pos, @as(u64, 0), k.tok }) |v| a.add(v);
    inline for (.{ ci(k.ntok), ci(k.ntok) }) |v| a.add(v);
    a.add(sinks);
    a.add(ci(ch));
    inline for (.{ c.pm.ptr, c.pl.ptr, c.po.ptr, @as(u64, 0), out }) |v| a.add(v);
    a.add(@as(c_int, 1) | (if (k.kv8) @as(c_int, 2) else 0));
    a.add(ci(rows));
    if (c.exact) a.add(@as(i64, @intCast(qlo)));
    if (paged) {
        a.add(k.page_table);
        a.add(ci(k.pt_stride));
    }
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(ns), .y = @intCast(hb), .z = @intCast(rows) }, .block = .{ .x = 256 }, .shared = sm }, s, &a);
    if (ns <= 1) return;
    var b: cuda.Args = .{};
    inline for (.{ ci(heads), ci(nq), k.kind }) |v| b.add(v);
    b.add(pos);
    inline for (.{ ci(k.ntok), ci(ch), ci(ns) }) |v| b.add(v);
    inline for (.{ c.pm.ptr, c.pl.ptr, c.po.ptr, sinks, out }) |v| b.add(v);
    b.add(@as(c_int, 1));
    b.add(ci(rows));
    try cuda.launch.launch(c.fold, .{ .grid = .{ .x = 8, .y = @intCast(hb), .z = @intCast(rows) }, .block = .{ .x = 256 }, .pdl = c.pdl }, s, &b);
}
