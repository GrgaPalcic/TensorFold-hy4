//! Hy4-flash kept states for the NVMe disk store (`native/cuda_kept.zig`): block-deduplicated layout, blank, and key.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const config = @import("config.zig");
const cuda_cache = @import("cuda_cache.zig");
const cuda_index = @import("cuda_index.zig");
const fwd = @import("cuda_model.zig");
const snap = @import("cuda_snap.zig");
const be = @import("cuda_backend.zig");

const Allocator = std.mem.Allocator;

pub const Target = be.Backend;
pub const State = snap.Saved;
pub const Words = extern struct { at: u64, active_layers: u64, kv8: u32 = 0, mrope_delta: i32 = 0 };
pub const probe_seed: u64 = 0x687934;

pub fn bind(t: *Target, ctx: *const cuda.Context) void {
    t.ctx = ctx;
}

pub fn bindCurrent(t: *const Target) void {
    if (t.ctx) |c| c.makeCurrent() catch {};
}

pub fn driver(t: *Target) *const cuda.Driver {
    return t.m.d;
}

pub fn stream(t: *Target) cuda.Stream {
    return t.s;
}

pub fn buffers(st: *const State) [2]?cuda.DeviceBuffer {
    return .{ st.buf, null };
}
pub fn heldBytes(t: *const Target, st: *const State) usize {
    if (st.pages.len > 0 or st.blob.len > 0) return @intCast(snap.bytes(t.m, st.at));
    return st.buf.len;
}

pub fn copyCustom(t: *Target, st: *State, ranges: []const api.prompt_cache.rows.image.Range, host: []u8, out: bool) !bool {
    if (st.pages.len == 0 and st.blob.len == 0) return false;
    if (t.ctx) |c| c.makeCurrent() catch {};
    try @import("cuda_disk_pages.zig").copyShareRanges(t, st, ranges, host, out);
    return true;
}

pub fn drafts(_: *const Target) bool {
    return false;
}

pub fn context(t: *const Target) usize {
    return t.m.context;
}

pub fn staging(_: *const Target) u64 {
    return 256 << 20;
}

pub fn wordsOf(st: *const State) Words {
    return .{ .at = st.at, .active_layers = st.active_layers, .kv8 = @intFromBool(st.kv8), .mrope_delta = st.mrope_delta };
}

pub fn setWords(st: *State, w: Words) !void {
    if (w.at != st.at or w.active_layers != st.active_layers or (w.kv8 != 0) != st.kv8) return error.KeptWordsDiffer;
    st.mrope_delta = w.mrope_delta;
    try snap.syncHistory(st);
}

/// Token history (`4 B/tok`), each indexed layer's MLA (`1152`), indexer (`256`) and 4:1 pool (`256`), sliding tails.
pub fn parts(t: *const Target, at: u32, out: anytype, a: Allocator) !void {
    try partsForModel(t.m, at, out, a);
}

pub fn partsForModel(m: *const fwd.Model, at: u32, out: anytype, a: Allocator) !void {
    if (at == 0 or m.active_layers == 0) {
        try out.append(a, .{ .bytes = snap.bytes(m, at) });
        return;
    }
    const pos: u64 = at;
    const rb: u32 = @intCast(cuda_cache.mlaRowBytes(m.kv8));
    try out.append(a, .{ .bytes = pos * 4, .row = 4 });
    for (m.layer_ids) |l| {
        if (m.c.kind[l] == .indexed) {
            try out.append(a, .{ .bytes = pos * rb, .row = rb });
            try out.append(a, .{ .bytes = (pos % cuda_index.ratio) * cuda_cache.index_row_bytes });
            const g_row: u32 = cuda_index.dim * 2;
            try out.append(a, .{ .bytes = (pos / cuda_index.ratio) * g_row, .row = g_row, .per = cuda_index.ratio });
        } else {
            try out.append(a, .{ .bytes = @min(pos, m.c.window) * rb });
        }
    }
}

pub fn blank(t: *Target, at: u32) !*State {
    return t.blankState(at);
}

pub fn save(t: *Target, s: *lanes.Stream, at: u32) !*State {
    return t.saveState(s, at);
}

pub fn drop(t: *Target, st: *State) void {
    t.dropState(st);
}

/// KV layout facts beyond the probe: active layers, indexed count, sliding window, and TP world size.
pub fn facts(t: *const Target) [5]u64 {
    return .{ t.m.active_layers, t.m.n_indexed, t.m.c.window, t.m.at.world, @intFromBool(t.m.kv8) };
}

pub const policy_domain = "hy4-cuda-policy-v2";
pub fn policy(target: anytype) [4]u64 {
    const t: *const Target = @ptrCast(@alignCast(target));
    return .{ t.m.at.world, t.m.active_layers, @intFromBool(t.m.engram_reader == null), @intFromBool(t.m.kv8) };
}

const testing = std.testing;
fn release() !config.Config {
    var why: config.Why = .{};
    return config.parse(testing.allocator, @embedFile("fixtures/config.json"), &why);
}

test "hy4 disk: parts are valid 256-token block images and match snap.bytes at every position" {
    const rows = api.prompt_cache.rows;
    var m: fwd.Model = undefined;
    m.c = try release();
    m.active_layers = m.c.layers;
    m.kv8 = false;
    var lids: [config.max_layers]usize = undefined;
    var n_idx: usize = 0;
    for (0..m.c.layers) |l| {
        lids[l] = l;
        if (m.c.kind[l] == .indexed) n_idx += 1;
    }
    m.layer_ids = lids[0..m.c.layers];
    m.n_indexed = n_idx;

    for ([_]bool{ false, true }) |kv8| {
        m.kv8 = kv8;
        for ([_]u32{ 0, 1, 3, 4, 255, 256, 511, 512, 1000, 4136, 65536, 131072, 524288, 1048576 }) |at| {
            var list: std.ArrayList(rows.image.Part) = .empty;
            defer list.deinit(testing.allocator);
            try list.append(testing.allocator, .{ .bytes = @sizeOf(Words) });
            try partsForModel(&m, at, &list, testing.allocator);
            try testing.expect(rows.image.valid(list.items, at, rows.block_tokens));
            try testing.expectEqual(@sizeOf(Words) + snap.bytes(&m, at), rows.image.imageBytes(list.items));
        }
    }
    m.kv8 = false;
    try testing.expect(snap.bytes(&m, 131072) > (1 << 30)); // > 1 GiB past ~74K tokens (#14 item 2)
    m.kv8 = true;
    try testing.expect(snap.bytes(&m, 131072) > (1 << 30)); // > 1 GiB past ~125K tokens under --kv-cache fp8
}

test "hy4 disk: changing TP world, active layers or Engram residency changes the policy key" {
    var m: fwd.Model = undefined;
    m.c = try release();
    m.active_layers = 42;
    m.n_indexed = 12;
    m.kv8 = false;
    m.at = .{ .rank = 0, .world = 2 };
    m.engram_reader = null;
    var b: Target = undefined;
    b.m = &m;
    const base_facts = facts(&b);
    const base_pol = policy(&b);
    m.at.world = 4;
    try testing.expect(!std.mem.eql(u64, &base_facts, &facts(&b)));
    try testing.expect(!std.mem.eql(u64, &base_pol, &policy(&b)));
    m.at.world = 2;
    m.active_layers = 12;
    try testing.expect(!std.mem.eql(u64, &base_pol, &policy(&b)));
}
