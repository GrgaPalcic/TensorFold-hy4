//! Gated CUDA-event + wall-clock profiler for Hy4-flash (`TF_HY4_PROF=1`; zero overhead when unset).
const std = @import("std");
const cuda = @import("cuda");
const sys = @import("sys");

const Stream = cuda.Stream;
const Event = cuda.Event;

pub const max_marks: usize = 1024;
const flush_every: usize = 32;

pub const Bucket = enum(u8) {
    // Target forward buckets
    embed = 0,
    ihc = 1,
    eng_gpu = 2,
    eng_comm = 3,
    qkv = 4,
    idx = 5,
    attn_qmmf = 6,
    attn_mla = 7,
    attn_comm = 8,
    l0_ffn = 9,
    moe_route = 10,
    moe_ds = 11,
    moe_exp_up = 12,
    moe_exp_dn = 13,
    moe_comb = 14,
    moe_lup = 15,
    mlp_comm = 16,
    lmhead = 17,
    save_hid = 18,
    argmax = 19,
    head_comm = 20,
    // DFly2 drafter buckets
    dfly_fuse = 21,
    dfly_qkv = 22,
    dfly_mla = 23,
    dfly_attn_mm = 24,
    dfly_mlp_mm = 25,
    dfly_comm = 26,
    dfly_bnd = 27,
    dfly_hc = 28,
    dfly_hc_comm = 29,
    dfly_lmhead = 30,
    dfly_argmax_comm = 31,

    pub const count = 32;
};

pub const Kind = enum(u8) { none = 0, target = 1, dfly = 2 };

pub const Key = struct {
    kind: Kind = .none,
    rows: usize = 0,
    parts: usize = 0,
    mode: usize = 0, // target: greedy (0/1); dfly: depth (1..5)

    pub fn eql(a: Key, b: Key) bool {
        return a.kind == b.kind and a.rows == b.rows and a.parts == b.parts and a.mode == b.mode;
    }
};

pub fn nowNs() i64 {
    var ts: sys.c.timespec = undefined;
    if (sys.c.clock_gettime(sys.c.CLOCK.MONOTONIC, &ts) != 0) return 0;
    return @as(i64, @intCast(ts.sec)) * std.time.ns_per_s + @as(i64, @intCast(ts.nsec));
}

pub fn enabled() bool {
    const v = @import("profile").lookup("TF_HY4_PROF") orelse return false;
    return v.len > 0 and v[0] != '0';
}

pub const Prof = struct {
    evs: [max_marks]Event = undefined,
    n_evs: usize = 0,
    tags: [max_marks]Bucket = undefined,
    cursor: usize = 0,
    active: bool = false,
    cur_key: Key = .{},
    t0_ns: i64 = 0,
    submit_ns: i64 = 0,
    cur_eng_cpu_ns: i64 = 0,
    moe_items_dev: ?cuda.DeviceBuffer = null,
    moe_items_n: usize = 0,

    // Running accumulator for the current shape key
    acc_key: Key = .{},
    acc_n: usize = 0,
    acc_wall_ms: f64 = 0,
    acc_submit_ms: f64 = 0,
    acc_sync_ms: f64 = 0,
    acc_span_ms: f64 = 0,
    acc_eng_cpu_ms: f64 = 0,
    acc_b: [Bucket.count]f64 = @splat(0),
    acc_moe_items_sum: u64 = 0,
    acc_moe_items_cnt: u64 = 0,
    acc_moe_items_min: u32 = std.math.maxInt(u32),
    acc_moe_items_max: u32 = 0,

    pub fn init(d: *const cuda.Driver) !Prof {
        var p: Prof = .{};
        errdefer p.deinit();
        for (0..max_marks) |i| {
            p.evs[i] = try Event.init(d, true);
            p.n_evs = i + 1;
        }
        p.moe_items_dev = try cuda.DeviceBuffer.alloc(d, 64 * 4);
        return p;
    }

    pub fn deinit(p: *Prof) void {
        p.flush();
        if (p.moe_items_dev) |*buf| buf.free();
        p.moe_items_dev = null;
        for (p.evs[0..p.n_evs]) |*e| e.deinit();
        p.n_evs = 0;
        p.* = undefined;
    }

    pub fn begin(p: *Prof, s: Stream, key: Key) void {
        if (p.n_evs == 0) return;
        p.active = true;
        p.cur_key = key;
        p.cursor = 0;
        p.cur_eng_cpu_ns = 0;
        p.moe_items_n = 0;
        p.t0_ns = nowNs();
        p.submit_ns = p.t0_ns;
        p.evs[0].record(s) catch {
            p.active = false;
        };
    }

    pub fn recordMoeItems(p: *Prof, s: Stream, counts_dev: u64) void {
        if (!p.active or p.moe_items_n >= 64) return;
        const buf = p.moe_items_dev orelse return;
        buf.copyFrom(p.moe_items_n * 4, counts_dev, 4, s.handle) catch return;
        p.moe_items_n += 1;
    }

    pub fn mark(p: *Prof, s: Stream, b: Bucket) void {
        if (!p.active or p.cursor + 1 >= p.n_evs) return;
        p.tags[p.cursor] = b;
        p.cursor += 1;
        p.evs[p.cursor].record(s) catch {};
    }

    pub fn addEngCpuNs(p: *Prof, dt_ns: i64) void {
        if (p.active and dt_ns > 0) p.cur_eng_cpu_ns += dt_ns;
    }

    pub fn endSubmit(p: *Prof) void {
        if (p.active) p.submit_ns = nowNs();
    }

    /// Synchronizes stream if needed, resolves all recorded CUDA events, and accumulates/prints.
    pub fn finish(p: *Prof, s: Stream) void {
        if (!p.active) return;
        p.active = false;
        s.synchronize() catch return;
        const end_ns = nowNs();
        const wall_ms = @as(f64, @floatFromInt(end_ns - p.t0_ns)) / 1e6;
        const submit_ms = @as(f64, @floatFromInt(p.submit_ns - p.t0_ns)) / 1e6;
        const sync_ms = @as(f64, @floatFromInt(end_ns - p.submit_ns)) / 1e6;
        const eng_cpu_ms = @as(f64, @floatFromInt(p.cur_eng_cpu_ns)) / 1e6;
        const span_ms: f64 = if (p.cursor > 0) @floatCast(Event.elapsedMs(p.evs[0], p.evs[p.cursor]) catch 0) else 0;

        if (p.acc_n > 0 and !p.acc_key.eql(p.cur_key)) p.flush();
        p.acc_key = p.cur_key;
        p.acc_n += 1;
        p.acc_wall_ms += wall_ms;
        p.acc_submit_ms += submit_ms;
        p.acc_sync_ms += sync_ms;
        p.acc_span_ms += span_ms;
        p.acc_eng_cpu_ms += eng_cpu_ms;
        for (0..p.cursor) |i| {
            const dt: f64 = @floatCast(Event.elapsedMs(p.evs[i], p.evs[i + 1]) catch 0);
            p.acc_b[@backingInt(p.tags[i])] += dt;
        }
        if (p.moe_items_n > 0) {
            if (p.moe_items_dev) |buf| {
                var host_items: [64]u32 = undefined;
                if (buf.download(0, std.mem.sliceAsBytes(host_items[0..p.moe_items_n]))) |_| {
                    for (host_items[0..p.moe_items_n]) |cnt| {
                        p.acc_moe_items_sum += cnt;
                        p.acc_moe_items_cnt += 1;
                        if (cnt < p.acc_moe_items_min) p.acc_moe_items_min = cnt;
                        if (cnt > p.acc_moe_items_max) p.acc_moe_items_max = cnt;
                    }
                } else |_| {}
            }
        }

        // Flush immediately for prefill (rows > 64) or every `flush_every` decode/draft calls
        if (p.cur_key.rows > 64 or p.acc_n >= flush_every) p.flush();
    }

    pub fn flush(p: *Prof) void {
        if (p.acc_n == 0) return;
        const inv = 1.0 / @as(f64, @floatFromInt(p.acc_n));
        const wall = p.acc_wall_ms * inv;
        const submit = p.acc_submit_ms * inv;
        const sync_dl = p.acc_sync_ms * inv;
        const span = p.acc_span_ms * inv;
        const eng_cpu = p.acc_eng_cpu_ms * inv;
        var b: [Bucket.count]f64 = undefined;
        var sum: f64 = 0;
        for (0..Bucket.count) |i| {
            b[i] = p.acc_b[i] * inv;
            sum += b[i];
        }
        const k = p.acc_key;
        if (k.kind == .target) {
            std.log.info(
                "[HY4P:STEP] n={d} rows={d} parts={d} greedy={d} wall={d:.2} submit={d:.2} sync={d:.2} gpu_span={d:.2} sum={d:.2} gap={d:.2}",
                .{ p.acc_n, k.rows, k.parts, k.mode, wall, submit, sync_dl, span, sum, wall - span },
            );
            std.log.info(
                "[HY4P:ATTN] embed={d:.2} qkv={d:.2} idx={d:.2} attn_qmmf={d:.2} attn_mla={d:.2} attn_comm={d:.2} ihc={d:.2}",
                .{
                    b[@backingInt(Bucket.embed)],     b[@backingInt(Bucket.qkv)],      b[@backingInt(Bucket.idx)],
                    b[@backingInt(Bucket.attn_qmmf)], b[@backingInt(Bucket.attn_mla)], b[@backingInt(Bucket.attn_comm)],
                    b[@backingInt(Bucket.ihc)],
                },
            );
            const avg_items: f64 = if (p.acc_moe_items_cnt > 0)
                @as(f64, @floatFromInt(p.acc_moe_items_sum)) / @as(f64, @floatFromInt(p.acc_moe_items_cnt))
            else
                0.0;
            const min_items: u32 = if (p.acc_moe_items_cnt > 0) p.acc_moe_items_min else 0;
            std.log.info(
                "[HY4P:MOE] l0_ffn={d:.2} route={d:.2} ds={d:.2} exp_up={d:.2} exp_dn={d:.2} comb={d:.2} lup={d:.2} mlp_comm={d:.2} exp_items={d:.1}({d}..{d})",
                .{
                    b[@backingInt(Bucket.l0_ffn)],     b[@backingInt(Bucket.moe_route)],  b[@backingInt(Bucket.moe_ds)],
                    b[@backingInt(Bucket.moe_exp_up)], b[@backingInt(Bucket.moe_exp_dn)], b[@backingInt(Bucket.moe_comb)],
                    b[@backingInt(Bucket.moe_lup)],    b[@backingInt(Bucket.mlp_comm)],   avg_items,
                    min_items,                         p.acc_moe_items_max,
                },
            );
            std.log.info(
                "[HY4P:MISC] eng_cpu={d:.2} eng_gpu={d:.2} eng_comm={d:.2} lmhead={d:.2} save_hid={d:.2} argmax={d:.2} head_comm={d:.2}",
                .{
                    eng_cpu,                          b[@backingInt(Bucket.eng_gpu)],  b[@backingInt(Bucket.eng_comm)],
                    b[@backingInt(Bucket.lmhead)],    b[@backingInt(Bucket.save_hid)], b[@backingInt(Bucket.argmax)],
                    b[@backingInt(Bucket.head_comm)],
                },
            );
        } else if (k.kind == .dfly) {
            std.log.info(
                "[HY4P:DFLY] n={d} B={d} K={d} wall={d:.2} submit={d:.2} sync={d:.2} gpu_span={d:.2} sum={d:.2} | fuse={d:.2} qkv={d:.2} mla={d:.2} attn_mm={d:.2} mlp_mm={d:.2} comm={d:.2} bnd={d:.2} | hc={d:.2} hc_comm={d:.2} lmhead={d:.2} argmax_comm={d:.2}",
                .{
                    p.acc_n,                                 k.parts,                             k.mode,
                    wall,                                    submit,                              sync_dl,
                    span,                                    sum,                                 b[@backingInt(Bucket.dfly_fuse)],
                    b[@backingInt(Bucket.dfly_qkv)],         b[@backingInt(Bucket.dfly_mla)],     b[@backingInt(Bucket.dfly_attn_mm)],
                    b[@backingInt(Bucket.dfly_mlp_mm)],      b[@backingInt(Bucket.dfly_comm)],    b[@backingInt(Bucket.dfly_bnd)],
                    b[@backingInt(Bucket.dfly_hc)],          b[@backingInt(Bucket.dfly_hc_comm)], b[@backingInt(Bucket.dfly_lmhead)],
                    b[@backingInt(Bucket.dfly_argmax_comm)],
                },
            );
        }
        p.acc_n = 0;
        p.acc_wall_ms = 0;
        p.acc_submit_ms = 0;
        p.acc_sync_ms = 0;
        p.acc_span_ms = 0;
        p.acc_eng_cpu_ms = 0;
        p.acc_b = @splat(0);
        p.acc_moe_items_sum = 0;
        p.acc_moe_items_cnt = 0;
        p.acc_moe_items_min = std.math.maxInt(u32);
        p.acc_moe_items_max = 0;
    }
};
