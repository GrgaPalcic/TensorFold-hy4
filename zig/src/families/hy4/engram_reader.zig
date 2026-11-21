//! Hy4-flash Engram row reader: NVMe O_DIRECT (`engram_rows`, LRU) and resident RAM slice (`15.37 GiB` @ TP=4).
const std = @import("std");
const builtin = @import("builtin");
const er = @import("engram_rows");
const config = @import("config.zig");
const engram_hash = @import("engram_hash.zig");
const partition = @import("partition.zig");
const ep = @import("engram_pack.zig");

const Config = config.Config;
const Io = std.Io;

pub const Mode = enum { nvme, resident };
pub const default_cache_rows: u32 = 262144;

pub const Rank = struct { rank: usize = 0, world: usize = 1 };

/// Hashes `tokens` at `positions` into this rank's `part.heads` global row ids and active mask.
pub fn hashRank(c: Config, slot: usize, part: partition.Engram, map: []const u8, tokens: []const u32, positions: []const u32, out_idx: []i64, out_mask: []u8) !void {
    if (tokens.len != positions.len or out_mask.len != tokens.len or out_idx.len != tokens.len * part.heads.len) return error.BadEngramArgs;
    const l = &engram_hash.layers[slot];
    for (tokens, 0..) |tok, i| {
        out_mask[i] = @intFromBool(!c.isEngramMasked(tok));
        const all = engram_hash.rows(l, map, tokens, positions, i, c.engram_pad);
        const dst = out_idx[i * part.heads.len ..][0..part.heads.len];
        for (dst, all[part.heads.lo..][0..part.heads.len]) |*o, id| o.* = id;
    }
}

fn loadResidentSlice(gpa: std.mem.Allocator, io: Io, path: []const u8, share: partition.Share) ![]u8 {
    const total_bytes = try std.math.mul(usize, share.len, ep.row_bytes);
    const buf = try gpa.alloc(u8, total_bytes);
    errdefer gpa.free(buf);
    const file = try Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const start_off = ep.header_bytes + @as(u64, share.lo) * ep.row_bytes;
    const step: usize = 64 << 20;
    var done: usize = 0;
    while (done < total_bytes) {
        const k = @min(step, total_bytes - done);
        const off = start_off + done;
        if (try file.readPositionalAll(io, buf[done..][0..k], off) != k) return error.ShortRead;
        if (comptime builtin.os.tag == .linux) {
            _ = std.os.linux.fadvise(file.handle, @intCast(off & ~@as(u64, 4095)), @intCast(k), std.os.linux.POSIX_FADV.DONTNEED);
        }
        done += k;
    }
    return buf;
}

pub const Reader = struct {
    gpa: std.mem.Allocator,
    c: Config,
    at: Rank,
    part: partition.Engram,
    mode: Mode,
    slots: []usize,
    shares: []partition.Share,
    nvme: ?*er.Rows = null,
    resident: [][]u8 = &.{},

    /// Opens `slots` (indices in `0..4` for layers `1, 11, 21, 31`) of rank `at` from `<dir>/engram_layer<L>.hy4e`.
    pub fn open(gpa: std.mem.Allocator, io: Io, dir: []const u8, c: Config, at: Rank, slots_in: []const usize, mode: Mode, cache_rows: u32) !Reader {
        if (slots_in.len == 0) return error.NoEngramLayers;
        var why: config.Why = .{};
        try engram_hash.check(c, &why);
        const part = try partition.Engram.of(c, at.rank, at.world);
        const slots = try gpa.dupe(usize, slots_in);
        errdefer gpa.free(slots);
        const shares = try gpa.alloc(partition.Share, slots.len);
        errdefer gpa.free(shares);

        var paths = try gpa.alloc([]u8, slots.len);
        var paths_made: usize = 0;
        defer {
            for (paths[0..paths_made]) |p| gpa.free(p);
            gpa.free(paths);
        }
        var specs = try gpa.alloc(er.OpenSpec, slots.len);
        defer gpa.free(specs);

        for (slots, 0..) |slot, i| {
            if (slot >= config.engram_layers) return error.BadEngramSlot;
            const layer: usize = c.engram_ids[slot] - 1;
            var nbuf: [64]u8 = undefined;
            const p = try std.fs.path.join(gpa, &.{ dir, try ep.fileName(&nbuf, layer) });
            paths[i] = p;
            paths_made += 1;

            const file = try Io.Dir.cwd().openFile(io, p, .{});
            defer file.close(io);
            var hb: [ep.header_bytes]u8 = undefined;
            if (try file.readPositionalAll(io, &hb, 0) != ep.header_bytes) return error.BadHy4EngramPack;
            const h = try ep.Header.parse(&hb, try file.length(io));
            try h.checkConfig(c, slot, &why);
            const share = part.rows(c, slot);
            shares[i] = share;
            specs[i] = .{
                .path = p,
                .h = .{ .layer = layer, .lo = share.lo, .hi = share.lo + share.len, .total = h.rows, .row_bytes = ep.row_bytes },
                .file_lo = 0,
                .file_hi = h.rows,
                .align_bytes = er.page,
            };
        }

        var r: Reader = .{ .gpa = gpa, .c = c, .at = at, .part = part, .mode = mode, .slots = slots, .shares = shares };
        switch (mode) {
            .nvme => {
                const cap = @max(cache_rows, 4096);
                r.nvme = if (@hasDecl(er.Rows, "openSpecsWithIo") and builtin.os.tag != .linux and builtin.os.tag != .macos)
                    try er.Rows.openSpecsWithIo(gpa, io, specs, cap)
                else
                    try er.Rows.openSpecs(gpa, specs, cap);
            },
            .resident => {
                const res = try gpa.alloc([]u8, slots.len);
                var made: usize = 0;
                errdefer {
                    for (res[0..made]) |s| gpa.free(s);
                    gpa.free(res);
                }
                for (paths, shares, res) |p, share, *dst| {
                    dst.* = try loadResidentSlice(gpa, io, p, share);
                    made += 1;
                }
                r.resident = res;
            },
        }
        return r;
    }

    pub fn close(r: *Reader) void {
        if (r.nvme) |n| n.close();
        for (r.resident) |s| r.gpa.free(s);
        if (r.resident.len > 0) r.gpa.free(r.resident);
        r.gpa.free(r.shares);
        r.gpa.free(r.slots);
        r.* = undefined;
    }

    pub fn indexOfSlot(r: *const Reader, slot: usize) ?usize {
        return std.mem.indexOfScalar(usize, r.slots, slot);
    }

    /// Starts async NVMe prefetch of `idx` for `slot_idx` (no-op in resident mode after range check).
    pub fn issue(r: *Reader, slot_idx: usize, idx: []const i64) !void {
        if (slot_idx >= r.slots.len) return error.BadEngramSlot;
        switch (r.mode) {
            .nvme => try r.nvme.?.issue(slot_idx, idx),
            .resident => {
                const sh = r.shares[slot_idx];
                for (idx) |id| if (id < 0 or @as(usize, @intCast(id)) < sh.lo or @as(usize, @intCast(id)) >= sh.lo + sh.len)
                    return error.EngramRowOutsideShard;
            },
        }
    }

    /// Fetches `idx` (`[idx.len]` global row ids in this rank's share) as contiguous 100-byte records into `out`.
    pub fn fetch(r: *Reader, slot_idx: usize, idx: []const i64, out: []u8) !void {
        if (slot_idx >= r.slots.len) return error.BadEngramSlot;
        if (out.len != try std.math.mul(usize, idx.len, ep.row_bytes)) return error.BadEngramRows;
        switch (r.mode) {
            .nvme => try r.nvme.?.rawRows(slot_idx, idx, out),
            .resident => {
                const sh = r.shares[slot_idx];
                const src = r.resident[slot_idx];
                for (idx, 0..) |id, k| {
                    if (id < 0) return error.EngramRowOutsideShard;
                    const u: usize = @intCast(id);
                    if (u < sh.lo or u >= sh.lo + sh.len) return error.EngramRowOutsideShard;
                    const off = (u - sh.lo) * ep.row_bytes;
                    @memcpy(out[k * ep.row_bytes ..][0..ep.row_bytes], src[off..][0..ep.row_bytes]);
                }
            },
        }
    }
};
