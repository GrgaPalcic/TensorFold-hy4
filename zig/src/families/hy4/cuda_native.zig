//! Hy4-flash (`hy_v4_flash`) native CUDA family entry (`TP` 1, 2, 4 on `sm_121` at `--precision full`).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const api = @import("engine_api");
const kv = core.kv_pool;
const config = @import("config.zig");
const memory = @import("memory.zig");
const fwd = @import("cuda_model.zig");
const be = @import("cuda_backend.zig");
const snap = @import("cuda_snap.zig");
const cuda_tp = @import("cuda_tp.zig");
const cvis = @import("cuda_vision.zig");
const cuda_pages = @import("cuda_pages.zig");
const kp = core.kv_prompts;

pub const model_type = "hy_v4_flash";
pub const formats: []const []const u8 = &.{"fp8-block128"};
pub const default_context: i64 = 32768;
pub const default_image_tokens: u32 = 4224;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = be.prompt_rows;
pub const snap_grid: u32 = be.prompt_rows;
pub const snap_lookahead: u32 = 0;
pub const share_grid: u32 = cuda_pages.page_tokens;
pub const warm_quiet: u64 = 150 * std.time.ns_per_ms;
pub const warm_usable = true;
pub const kept_view = @import("cuda_disk.zig");
pub const request_starts = true;
pub const request_starts_switch = "TF_HY4_REQUEST_STARTS";
pub const window_rows: u32 = be.max_window;
pub const takes_drafter = false;
pub const drafter_refusal = "--drafter: Hy4-flash on CUDA loads no external drafter; serve without --drafter";
pub const explains_open = true;
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };

pub const Options = struct {
    context: usize,
    drafts: bool,
    segments: usize = 1,
    slide: bool = false,
    vision: ?cvis.Limits = null,
    kv8: bool = false,
    tp: usize = 1,
    rank: usize = 0,
    master: ?[]const u8 = null,
    master_port: u16 = 29550,
    precision: ?enum { full } = null,
    room: ?Room = null,
};

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 0,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    target: *be.Backend,
    vision: ?*cvis.Vision = null,
    layout: ?kv.Layout = null,
    engine_paged: bool = false,
    per_request: ?kv.PerRequest = null,
    peer_room: ?cuda_tp.PeerRoom = null,
    follow: ?*const fn (*anyopaque) anyerror!void = null,
};

const Owned = struct {
    gpa: std.mem.Allocator,
    tp_sess: ?*cuda_tp.Session = null,
    k: fwd.AllKernels,
    s: cuda.Stream,
    m: *fwd.Model,
    b: be.Backend,
    vis: ?*cvis.Vision = null,
};

pub fn admit(sm: u32, world: usize, rank: usize) !void {
    if (sm != 121) return error.Hy4RequiresSm121;
    if (world != 1 and world != 2 and world != 4) return error.Hy4UnsupportedWorld;
    if (rank >= world) return error.BadRank;
}

/// Rank-sliced GPU weight bytes (`74.80 GiB` @ TP=2, `38.40 GiB` @ TP=4; adds `15.37 GiB` when Engram is resident).
pub fn placementWeightBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, world: usize, rank: usize, _: bool) !u64 {
    if (world != 1 and world != 2 and world != 4) return error.Hy4UnsupportedWorld;
    if (rank >= world) return error.BadRank;
    const c = config.Config.read(gpa, io, dir) catch return 0;
    const w_bytes: u64 = memory.weightBytes(c, rank, world, .packed_) catch return 0;
    const mode = fwd.parseEngramMode() catch .nvme;
    const e_bytes: u64 = if (mode == .resident) (memory.engramTableBytes(c, rank, world, .packed_) catch 0) else 0;
    return w_bytes + e_bytes;
}

/// Rank-0 device bytes reserved for `vit.*` weights + scratch when `--vision` is active.
pub fn visionBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
    const vc = (try cvis.Config.read(gpa, io, dir)) orelse return 0;
    return @intCast(cvis.weightArenaBytes(vc) + cvis.scratchArenaBytes(vc, @import("vision_config.zig").max_patches));
}

pub fn benchmark(ptr: *anyopaque) struct { ranks: ?u32, group: ?[]const u8 } {
    const own: *const Owned = @ptrCast(@alignCast(ptr));
    return .{ .ranks = @intCast(own.m.at.world), .group = null };
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (kernels != null) return error.CapturedKernels;
    const sm = try ctx.capability();
    try admit(sm, o.tp, o.rank);
    const priority_copies = try cuda.priority_copies.read("TF_HY4_COPIES", o.drafts);
    const engram_mode = try fwd.parseEngramMode();
    const engram_cache = try fwd.parseEngramCache();
    const c = try config.Config.read(gpa, io, dir);
    const active_layers = try fwd.parseLayersLimit(c.layers);
    const on_pages = try cuda_pages.pagedWanted();
    if (o.context > c.max_positions) return error.PastContext;

    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .k = undefined, .s = undefined, .m = undefined, .b = undefined };

    if (o.tp > 1) {
        const mine = try cuda_tp.identity(gpa, io, dir, o.context, prompt_rows, o.tp, active_layers, @backingInt(engram_mode), o.kv8, on_pages);
        own.tp_sess = try cuda_tp.Session.open(gpa, io, ctx.d, .{
            .world = o.tp,
            .rank = o.rank,
            .master = o.master,
            .master_port = o.master_port,
            .device = @intCast(ctx.device),
        }, mine);
    }
    errdefer if (own.tp_sess) |ts| ts.close();

    own.k = try fwd.AllKernels.load(ctx);
    errdefer own.k.deinit();
    own.s = try cuda.Stream.init(ctx.d, false);
    errdefer own.s.deinit();
    own.m = try fwd.Model.init(gpa, io, ctx, &own.k, dir, .{
        .at = .{ .rank = o.rank, .world = o.tp },
        .context = o.context,
        .layers_limit = active_layers,
        .engram_mode = engram_mode,
        .engram_cache = engram_cache,
        .tp_sess = own.tp_sess,
        .kv8 = o.kv8,
    });
    errdefer own.m.deinit();
    if (o.vision) |vlim| {
        own.vis = try cvis.Vision.open(gpa, io, ctx.d, dir, o.rank, vlim);
    }
    errdefer if (own.vis) |v| v.deinit();
    own.b = try be.Backend.init(gpa, own.m, &own.k, own.s, priority_copies);
    own.b.vis = own.vis;
    errdefer own.b.deinit();
    if (on_pages) {
        try own.m.openSeqPaged(0, own.m.context);
    }
    var peer_room: ?cuda_tp.PeerRoom = null;
    if (own.tp_sess) |ts| {
        const local_room: u64 = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
        const agreed = try ts.agreeRoom(ctx.d, local_room);
        peer_room = agreed.peer;
    }

    const fixed = own.m.fixedSeqBytes();
    const tok = own.m.tokenSeqBytes();
    const one_full: usize = @intCast(fixed + tok * (o.context + fwd.max_window));

    if (o.rank == 0) {
        std.log.info("hy4: world={d}, layers={d}/{d}, context={d}, kv={s}, paged={}, vision={}, engram={s} (fixed {d:.1} MiB/seq + {d} B/tok)", .{
            o.tp,
            active_layers,
            c.layers,
            o.context,
            if (o.kv8) "fp8" else "bf16",
            on_pages,
            own.vis != null,
            @tagName(engram_mode),
            @as(f64, @floatFromInt(fixed)) / (1 << 20),
            tok,
        });
    }

    return .{
        .backend = own.b.backend(),
        .facts = own.b.facts(),
        .rows = be.max_window,
        .stream_bytes = if (on_pages) @intCast(cuda_pages.layout(own.m).fixed_bytes) else one_full,
        .free_streams = if (on_pages) 1 else 0,
        .ctx = own,
        .deinit = release,
        .target = &own.b,
        .vision = own.vis,
        .layout = if (on_pages) cuda_pages.layout(own.m) else null,
        .engine_paged = on_pages,
        .per_request = if (on_pages) null else .{
            .layout = .{ .fixed_bytes = fixed, .token_bytes = tok, .page_tokens = 0, .snap_bytes = 0 },
            .lookahead = be.max_window,
            .scratch = 0,
        },
        .peer_room = peer_room,
        .follow = if (o.rank > 0) followRank else null,
    };
}

pub fn attach(ptr: *anyopaque, extra_streams: u32, pages: usize) !void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    own.m.pages = try cuda_pages.Pages.init(own.gpa, own.m, own.s, 1 + @as(usize, extra_streams), pages, own.m.at.rank == 0);
}

pub fn fits(ptr: *anyopaque, prompt: u32, max_new: u32) bool {
    const own: *const Owned = @ptrCast(@alignCast(ptr));
    if (!be.fits(own.m.context, prompt, max_new)) return true;
    const pg = own.m.pages orelse return std.mem.indexOfScalar(bool, &own.b.free, true) != null;
    const has_slot = std.mem.indexOfScalar(bool, own.b.free[0..pg.streams], true) != null;
    return has_slot and pg.pool.fits(cuda_pages.needPages(prompt, max_new));
}

pub fn kvPool(ptr: *anyopaque) ?kv.Source {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    const pg = own.m.pages orelse return null;
    return .{ .pool = &pg.pool, .page_tokens = cuda_pages.page_tokens };
}

pub fn share_pool(ptr: *anyopaque) ?*kv.Pool {
    const b = backendOf(ptr);
    const pg = b.m.pages orelse return null;
    return &pg.pool;
}

pub fn share_attach(ptr: *anyopaque, prompts: ?kp.Prompts) void {
    backendOf(ptr).prompts = prompts;
}

pub fn share_disk(ptr: *anyopaque) api.prompt_radix_disk.Hooks {
    return @import("cuda_disk_pages.zig").hooksFor(backendOf(ptr));
}

fn followRank(ptr: *anyopaque) anyerror!void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    return own.b.followLoop();
}

fn release(ptr: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    const gpa = own.gpa;
    own.b.deinit();
    if (own.vis) |v| v.deinit();
    own.m.deinit();
    own.s.deinit();
    own.k.deinit();
    if (own.tp_sess) |ts| ts.close();
    gpa.destroy(own);
}

fn backendOf(ptr: *anyopaque) *be.Backend {
    return @ptrCast(@alignCast(ptr));
}

pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    const b = backendOf(ptr);
    if (b.prompts != null) {
        const pg = b.m.pages orelse return snap.bytes(b.m, at);
        const from: u32 = (at / cuda_pages.page_tokens) * cuda_pages.page_tokens;
        return @as(u64, snap.blobPages(b.m, at, from)) * pg.page_bytes;
    }
    return snap.bytes(b.m, at);
}

pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
    return @ptrCast(try backendOf(ptr).saveState(owner, at));
}

pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: api.prompt_cache.Saved) anyerror!void {
    return backendOf(ptr).restoreState(owner, @ptrCast(@alignCast(saved)));
}

pub fn snap_drop(ptr: *anyopaque, saved: api.prompt_cache.Saved) void {
    backendOf(ptr).dropState(@ptrCast(@alignCast(saved)));
}

pub fn snap_spare(ptr: *anyopaque) u64 {
    return backendOf(ptr).spares.bytes();
}

pub fn snap_trim(ptr: *anyopaque, room: u64) void {
    backendOf(ptr).trimSpares(room);
}

pub fn snap_reuses(ptr: *anyopaque, at: u32) bool {
    const b = backendOf(ptr);
    return b.spares.fits(@intCast(snap.bytes(b.m, at)));
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.Hy4RequiresSm121 => "Hy4-flash on CUDA targets GB10 (sm_121) only",
        error.Hy4UnsupportedWorld => "--tp: Hy4-flash on CUDA supports 1, 2 or 4 tensor-parallel ranks (--tp 2 or --tp 4)",
        error.MissingEngramPackDir => "Hy4-flash requires packed Engram files (engram_layer{1,11,21,31}.hy4e): pack with tf-hy4-engram-pack and set TF_HY4_ENGRAM_DIR",
        error.BadCheckpointDigest => "TENSORFOLD_CHECKPOINT_SHA256 must be a 64-character hex SHA-256 digest",
        error.RankIdentityMismatch => "Hy4-flash TP ranks disagree on checkpoint digest, config, binary, context, or Engram mode",
        error.CapturedKernels => "TENSORFOLD_CUDA_KERNELS: Hy4-flash runs its own CUDA kernels only; serve with native or unset",
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.OutOfDeviceMemory => "insufficient device memory across ranks for this request's KV cache; retry when a stream finishes",
        error.NoHy4Vision => "this Hy4-flash checkpoint carries no vision_config (or server was started without --vision)",
        error.UnsupportedHy4Vision => "--vision: this checkpoint's vision_config differs from Hy4-flash's 27-layer 1152-wide ViT",
        error.Hy4VideoUnsupported => "video inputs are not served on Hy4-flash CUDA yet; send images or text",
        error.PastContext => "--context exceeds Hy4-flash's max_position_embeddings (1,048,576)",
        error.KvShareWithoutPages => "TF_KV_SHARE keeps Hy4-flash's prompts as pool pages, and TF_HY4_KV_PAGES=0 turns pages off: unset one of the two",
        error.BadKvShare => "TF_KV_SHARE takes a nonnegative count of kept prompt states (0 or unset: copies)",
        error.InvalidPagesEnv => "TF_HY4_KV_PAGES must be 0 or 1",
        else => null,
    };
}

test "admit enforces sm_121 and world in {1, 2, 4}" {
    try admit(121, 1, 0);
    try admit(121, 2, 1);
    try admit(121, 4, 3);
    try std.testing.expectError(error.Hy4RequiresSm121, admit(86, 2, 0));
    try std.testing.expectError(error.Hy4UnsupportedWorld, admit(121, 3, 0));
    try std.testing.expectError(error.BadRank, admit(121, 2, 2));
    std.testing.refAllDecls(@This());
}
