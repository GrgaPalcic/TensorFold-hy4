//! Multi-rank TP session for Hy4-flash (`world` 2 or 4): `fabric/tp` RoCE + NCCL collectives, identity, and `PlanLink`.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const core = @import("core");
const tp = @import("tp");
const tp_kernel = @import("tp_kernel");
const config = @import("config.zig");
const cuda_layer = @import("cuda_layer.zig");

pub const max_world: usize = 4;

pub const Op = enum(i64) {
    stop = 0,
    open_seq = 1,
    release_seq = 2,
    forward = 3,
    verify_round = 4,
    keep_round = 5,
    snap_save = 6,
    snap_restore = 7,
    snap_drop = 8,
    forward_greedy = 9,
    verify_round_greedy = 10,
    draft_round = 11,
    verify_round_greedy_plain = 12,
    forward_prompt = 13,
    snap_load = 14,
    snap_trim = 15,
    forward_sample = 16,
    verify_round_sample = 17,
    verify_round_sample_plain = 18,
    set_media = 19,
    clear_media = 20,
    map_pages = 21,
    release_pages = 22,
    snap_save_share = 23,
    snap_load_share = 24,
};

pub const Identity = extern struct {
    config: [32]u8,
    index: [32]u8,
    program: [32]u8,
    checkpoint: [32]u8,
    context: u64,
    chunk: u64,
    world: u64,
    layers: u64,
    engram_mode: u64,
    kv8: u64 = 0,
    paged: u64 = 0,
};

fn fileHash(a: std.mem.Allocator, io: std.Io, path: []const u8) ![32]u8 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 28));
    defer a.free(bytes);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return out;
}

const profile = @import("profile");

/// Computes rank identity; honors `TENSORFOLD_CHECKPOINT_SHA256` to skip hashing 267.4 GiB on cold unified memory.
pub fn identity(a: std.mem.Allocator, io: std.Io, dir: []const u8, context: usize, chunk: usize, world: usize, layers: usize, engram_mode: u64, kv8: bool, paged: bool) !Identity {
    var checkpoint: [32]u8 = undefined;
    if (profile.lookup("TENSORFOLD_CHECKPOINT_SHA256")) |given| {
        const digest = std.mem.trim(u8, given, " \t\r\n");
        if (digest.len != 64) return error.BadCheckpointDigest;
        _ = std.fmt.hexToBytes(&checkpoint, digest) catch return error.BadCheckpointDigest;
    } else checkpoint = try core.checkpoint_identity.fullIdentity(a, io, dir);
    const cfg_path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(cfg_path);
    const idx_path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    defer a.free(idx_path);
    const prog_path = try std.process.executablePathAlloc(io, a);
    defer a.free(prog_path);
    const idx_hash = fileHash(a, io, idx_path) catch @as([32]u8, @splat(0));
    return .{
        .config = try fileHash(a, io, cfg_path),
        .index = idx_hash,
        .program = try fileHash(a, io, prog_path),
        .checkpoint = checkpoint,
        .context = context,
        .chunk = chunk,
        .world = world,
        .layers = layers,
        .engram_mode = engram_mode,
        .kv8 = @intFromBool(kv8),
        .paged = @intFromBool(paged),
    };
}

pub const PeerRoom = struct { bytes: u64, rank: usize };

pub const OpenOptions = struct {
    world: usize,
    rank: usize,
    master: ?[]const u8 = null,
    master_port: u16 = 29550,
    device: u32 = 0,
};

pub const Session = struct {
    gpa: std.mem.Allocator,
    rank: usize,
    world: usize,
    inner: tp.Session,
    image_owned: ?[]align(16) u8 = null,
    sum_mod: cuda.rank_scatter.RankScatter,
    full_mesh: bool = true,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, opts: OpenOptions, mine: Identity) !*Session {
        if (opts.world != 2 and opts.world != 4) return error.Hy4UnsupportedWorld;
        if (opts.rank >= opts.world) return error.BadRank;
        var cfg = try tp.Config.fromEnvFor(@intCast(opts.world), @intCast(opts.rank));
        cfg.device = opts.device;
        if (opts.master) |m| cfg.master = m;
        if (profile.lookup("TF_TP_PORT") == null) {
            cfg.port = opts.master_port;
            if (profile.lookup("TF_TP_PLAN_PORT") == null) cfg.link_port = opts.master_port +| 7;
        }
        var owned_img: ?[]align(16) u8 = null;
        errdefer if (owned_img) |b| gpa.free(b);
        const image: []const u8 = if (profile.lookup("TF_TP_KERNEL_IMAGE")) |p| blk: {
            const raw = try std.Io.Dir.cwd().readFileAllocOptions(io, p, gpa, .limited(1 << 26), .@"16", null);
            owned_img = raw;
            break :blk raw;
        } else tp_kernel.mailbox;
        if (profile.lookup("TF_COMM_BACKEND") == null and image.len > 0 and (opts.world == 2 or cfg.roce_relay)) {
            cfg.backend = .roce;
        }
        const s = try gpa.create(Session);
        errdefer gpa.destroy(s);
        const sum_mod = try cuda.rank_scatter.RankScatter.load(d);
        s.* = .{ .gpa = gpa, .rank = opts.rank, .world = opts.world, .inner = undefined, .image_owned = owned_img, .sum_mod = sum_mod };
        owned_img = null;
        errdefer s.sum_mod.unload();
        errdefer if (s.image_owned) |b| gpa.free(b);
        try s.inner.open(d, cfg, image);
        errdefer s.inner.close();
        s.full_mesh = !cfg.roce_relay and s.inner.roce_multi == null and profile.lookup("NCCL_GRAPH_FILE") == null;
        try s.verifyIdentity(d, mine);
        return s;
    }

    pub fn close(s: *Session) void {
        if (s.rank == 0) s.send(&.{@backingInt(Op.stop)}) catch {};
        s.inner.close();
        s.sum_mod.unload();
        if (s.image_owned) |b| s.gpa.free(b);
        const gpa = s.gpa;
        gpa.destroy(s);
    }

    pub fn comm(s: *Session) tp.Collective {
        return s.inner.comm();
    }

    fn verifyIdentity(s: *Session, d: *const cuda.Driver, mine: Identity) !void {
        const sz = @sizeOf(Identity);
        var stream = try cuda.Stream.init(d, true);
        defer stream.deinit();
        var dev = try cuda.DeviceBuffer.alloc(d, (1 + s.world) * sz);
        defer dev.free();
        try dev.uploadAsync(0, std.mem.asBytes(&mine), stream.handle);
        const c = s.comm();
        try c.allGather(dev.ptr, dev.ptr + sz, sz, .u8, stream.handle);
        try stream.synchronize();
        try c.check();
        var all: [max_world]Identity = undefined;
        try dev.download(sz, std.mem.sliceAsBytes(all[0..s.world]));
        for (all[0..s.world]) |peer| {
            if (!std.mem.eql(u8, std.mem.asBytes(&mine), std.mem.asBytes(&peer))) return error.RankIdentityMismatch;
        }
    }

    /// Exchanges every rank's post-load memory room and returns the smallest peer room (plus agreed minimum).
    pub fn agreeRoom(s: *Session, d: *const cuda.Driver, local: u64) !struct { min: u64, peer: ?PeerRoom } {
        var stream = try cuda.Stream.init(d, true);
        defer stream.deinit();
        var dev = try cuda.DeviceBuffer.alloc(d, (1 + s.world) * 8);
        defer dev.free();
        try dev.uploadAsync(0, std.mem.asBytes(&local), stream.handle);
        const c = s.comm();
        try c.allGather(dev.ptr, dev.ptr + 8, 1, .i64, stream.handle);
        try stream.synchronize();
        try c.check();
        var rooms: [max_world]u64 = undefined;
        try dev.download(8, std.mem.sliceAsBytes(rooms[0..s.world]));
        var min_val = local;
        var peer: ?PeerRoom = null;
        for (rooms[0..s.world], 0..) |r, idx| {
            if (r < min_val) min_val = r;
            if (idx != s.rank and (peer == null or r < peer.?.bytes)) peer = .{ .bytes = r, .rank = idx };
        }
        return .{ .min = min_val, .peer = peer };
    }

    /// Votes across all ranks whether an allocation succeeded so no rank mutates state on partial failure.
    pub fn agreeOk(s: *Session, d: *const cuda.Driver, stream: cuda.Stream, scratch_ptr: u64, ok: bool) !bool {
        const word: u32 = @intFromBool(ok);
        const src = scratch_ptr;
        const dst = scratch_ptr + 16;
        try cuda.copy_source.enqueue(d, src, std.mem.asBytes(&word), stream);
        const c = s.comm();
        try c.allGather(src, dst, 1, .i32, stream.handle);
        try stream.synchronize();
        try c.check();
        var votes: [max_world]u32 = undefined;
        try (cuda.DeviceBuffer{ .d = d, .ptr = dst, .len = s.world * 4 }).download(0, std.mem.sliceAsBytes(votes[0..s.world]));
        for (votes[0..s.world]) |v| if (v != 1) return false;
        return true;
    }

    /// Scratch bytes `sumRankOrder` needs in `gather_buf` for calls up to `max_count` floats (`0` at `world == 2`).
    pub fn gatherBytes(s: *const Session, max_count: usize) usize {
        if (s.world <= 2) return 0;
        if (!s.full_mesh) return s.world * max_count * 4;
        return max_count * 4;
    }

    /// Rank-order fp32 sum: `allReduce` at world 2, `RankScatter` + `allGather` on wide world-4 seams, else gather.
    pub fn sumRankOrder(s: *Session, stream: cuda.Stream, src_ptr: u64, gather_buf: u64, dst_ptr: u64, count: usize) !void {
        if (count == 0) return;
        const c = s.comm();
        if (s.world == 2) return c.allReduce(src_ptr, dst_ptr, count, .f32, .sum, stream.handle);
        if (s.scatterSum(count)) {
            const sub = count / s.world;
            const mine = dst_ptr + s.rank * sub * 4;
            try s.sum_mod.run(&s.inner.lib, s.inner.nccl.comm, s.world, s.rank, stream, src_ptr, s.world, sub, gather_buf, mine);
            return s.inner.nccl.iface().allGather(mine, dst_ptr, sub, .f32, stream.handle);
        }
        try c.allGather(src_ptr, gather_buf, count, .f32, stream.handle);
        try s.sum_mod.rankSum(stream, gather_buf, gather_buf, s.world, 0, count, dst_ptr);
    }

    /// Full-mesh world 4: point-to-point `RankScatter` + `allGather`; switchless rings use ring `allGather` instead.
    fn scatterSum(s: *const Session, count: usize) bool {
        return s.world >= 4 and s.full_mesh and count % s.world == 0;
    }

    /// Gathers each rank's `[rows, vocab_local]` fp32 logits into `[rows, vocab]` on `stream`.
    pub fn gatherLogits(s: *Session, stream: cuda.Stream, local_ptr: u64, vocab_local: usize, full_ptr: u64, rows: usize) !void {
        const c = s.comm();
        const vocab = vocab_local * s.world;
        for (0..rows) |r| {
            try c.allGather(local_ptr + r * vocab_local * 4, full_ptr + r * vocab * 4, vocab_local, .f32, stream.handle);
        }
    }

    /// Gathers each rank's `[rows][2]` u32 local argmax `{max_bits, global_id}` into `[world][rows][2]` in one call.
    pub fn gatherArgmax(s: *Session, stream: cuda.Stream, local_ptr: u64, full_ptr: u64, rows: usize) !void {
        if (rows == 0) return;
        try s.comm().allGather(local_ptr, full_ptr, rows * 2, .i32, stream.handle);
    }

    /// Gathers `count_bf16` (even) bf16 elements from each rank into `[world, count_bf16]` via i32 allGather.
    pub fn gatherBf16(s: *Session, stream: cuda.Stream, local_ptr: u64, full_ptr: u64, count_bf16: usize) !void {
        if (count_bf16 == 0) return;
        try s.comm().allGather(local_ptr, full_ptr, count_bf16 / 2, .i32, stream.handle);
    }

    /// Exchanges `count_f32` fp32 elements directly with the peer rank (`world == 2`) via `tp_roce` / NCCL exchange.
    pub fn exchangeF32(s: *Session, stream: cuda.Stream, send_ptr: u64, recv_ptr: u64, count_f32: usize) !void {
        if (count_f32 == 0) return;
        const peer: u32 = @intCast(1 - s.rank);
        try s.comm().exchange(send_ptr, recv_ptr, count_f32, .f32, peer, stream.handle);
    }

    /// Streams `byte_len` device bytes from rank 0 (`ptr`) into every follower rank's `ptr` via ring `ncclBroadcast`.
    pub fn broadcastFrom0(s: *Session, stream: cuda.Stream, ptr: u64, byte_len: usize) !void {
        if (byte_len == 0 or s.world <= 1) return;
        try s.inner.lib.check(s.inner.lib.api.ncclBroadcast(ptr, ptr, byte_len, .u8, 0, s.inner.nccl.comm, stream.handle), "ncclBroadcast");
    }

    pub fn send(s: *Session, words: []const i64) !void {
        try s.inner.plan.send(words);
    }

    pub fn recvGrow(s: *Session, buf: *std.ArrayList(i64)) ![]i64 {
        return s.inner.plan.recvGrow(s.gpa, buf);
    }

    pub fn check(s: *Session) !void {
        try s.comm().check();
    }
};

test "cuda_tp declarations compile without GPU" {
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest;
    std.testing.refAllDecls(@This());
}
