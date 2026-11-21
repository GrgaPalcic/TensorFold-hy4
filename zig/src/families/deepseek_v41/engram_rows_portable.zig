//! Portable synchronous NVMe reader and bounded LRU. Linux retains its async physical reader.
const std = @import("std");
const f = @import("engram_rows.zig");
const eh = @import("engram_host.zig");
const Entry = struct { id: u64 = std.math.maxInt(u64), age: u64 = 0 };
const Table = struct { file: std.Io.File, h: f.Header, file_lo: u64, entries: []Entry, recs: []u8 };
pub const Stats = struct { hits: u64 = 0, misses: u64 = 0, issued: u64 = 0, reads: u64 = 0, bytes: u64 = 0 };
pub const Rows = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    tables: []Table,
    scratch: []u8,
    clock: u64 = 0,
    stats: Stats = .{},
    pub fn open(a: std.mem.Allocator, dir: []const u8, rank: u32, world: u32, layers: []const u32, cap: u32) !*Rows {
        return openWithIo(a, std.Io.Threaded.global_single_threaded.io(), dir, rank, world, layers, cap);
    }
    pub fn openWithIo(a: std.mem.Allocator, io: std.Io, dir: []const u8, rank: u32, world: u32, layers: []const u32, cap: u32) !*Rows {
        if (world == 0 or rank >= world or cap == 0 or layers.len == 0) return error.BadEngramShard;
        const r = try a.create(Rows);
        errdefer a.destroy(r);
        const tables = try a.alloc(Table, layers.len);
        errdefer a.free(tables);
        const scratch = try a.alloc(u8, @intCast(f.row_bytes));
        errdefer a.free(scratch);
        var made: usize = 0;
        errdefer for (tables[0..made]) |t| {
            t.file.close(io);
            a.free(t.entries);
            a.free(t.recs);
        };
        for (layers, tables) |layer, *t| {
            const path = try std.fmt.allocPrint(a, "{s}/engram-l{d}-r{d}of{d}.bin", .{ dir, layer, rank, world });
            defer a.free(path);
            const file = try std.Io.Dir.cwd().openFile(io, path, .{});
            errdefer file.close(io);
            var bytes: [48]u8 = undefined;
            if (try file.readPositionalAll(io, &bytes, 0) != 48) return error.BadEngramShard;
            const h = try f.Header.parse(&bytes, try file.length(io));
            if (h.layer != layer) return error.BadEngramShard;
            const entries = try a.alloc(Entry, cap);
            errdefer a.free(entries);
            @memset(entries, .{});
            const recs = try a.alloc(u8, try std.math.mul(usize, cap, @intCast(h.row_bytes)));
            t.* = .{ .file = file, .h = h, .file_lo = h.lo, .entries = entries, .recs = recs };
            made += 1;
        }
        r.* = .{ .gpa = a, .io = io, .tables = tables, .scratch = scratch };
        return r;
    }
    pub fn openSpecs(a: std.mem.Allocator, specs: []const f.OpenSpec, cap: u32) !*Rows {
        return openSpecsWithIo(a, std.Io.Threaded.global_single_threaded.io(), specs, cap);
    }
    pub fn openSpecsWithIo(a: std.mem.Allocator, io: std.Io, specs: []const f.OpenSpec, cap: u32) !*Rows {
        if (cap < 4096 or specs.len == 0) return error.EngramCacheTooSmall;
        const r = try a.create(Rows);
        errdefer a.destroy(r);
        const tables = try a.alloc(Table, specs.len);
        errdefer a.free(tables);
        var max_rb: usize = 0;
        for (specs) |s| max_rb = @max(max_rb, @as(usize, @intCast(s.h.row_bytes)));
        const scratch = try a.alloc(u8, max_rb);
        errdefer a.free(scratch);
        var made: usize = 0;
        errdefer for (tables[0..made]) |t| {
            t.file.close(io);
            a.free(t.entries);
            a.free(t.recs);
        };
        for (specs, tables) |spec, *t| {
            const file = try std.Io.Dir.cwd().openFile(io, spec.path, .{});
            errdefer file.close(io);
            try spec.h.checkRange(spec.file_lo, spec.file_hi, try file.length(io));
            const entries = try a.alloc(Entry, cap);
            errdefer a.free(entries);
            @memset(entries, .{});
            const recs = try a.alloc(u8, try std.math.mul(usize, cap, @intCast(spec.h.row_bytes)));
            t.* = .{ .file = file, .h = spec.h, .file_lo = spec.file_lo, .entries = entries, .recs = recs };
            made += 1;
        }
        r.* = .{ .gpa = a, .io = io, .tables = tables, .scratch = scratch };
        return r;
    }
    pub fn close(r: *Rows) void {
        for (r.tables) |t| {
            t.file.close(r.io);
            r.gpa.free(t.entries);
            r.gpa.free(t.recs);
        }
        r.gpa.free(r.scratch);
        r.gpa.free(r.tables);
        r.gpa.destroy(r);
    }
    fn validate(r: *Rows, li: usize, ids: []const i64) !void {
        if (li >= r.tables.len) return error.BadEngramShard;
        const h = r.tables[li].h;
        for (ids) |id| if (id < 0 or @as(u64, @intCast(id)) < h.lo or @as(u64, @intCast(id)) >= h.hi)
            return error.EngramRowOutsideShard;
    }
    fn record(r: *Rows, li: usize, id: u64) ![]const u8 {
        const table = &r.tables[li];
        const rb: usize = @intCast(table.h.row_bytes);
        r.clock +%= 1;
        var slot_idx: usize = 0;
        for (table.entries, 0..) |*entry, idx| {
            if (entry.id == id) {
                entry.age = r.clock;
                r.stats.hits += 1;
                return table.recs[idx * rb ..][0..rb];
            }
            if (entry.age < table.entries[slot_idx].age) slot_idx = idx;
        }
        // Publish the cache identity only after the full positional read succeeds.
        const buf = r.scratch[0..rb];
        const offset = f.header_bytes + (id - table.file_lo) * table.h.row_bytes;
        if (try table.file.readPositionalAll(r.io, buf, offset) != rb) return error.ShortRead;
        table.entries[slot_idx] = .{ .id = id, .age = r.clock };
        @memcpy(table.recs[slot_idx * rb ..][0..rb], buf);
        r.stats.misses += 1;
        r.stats.reads += 1;
        r.stats.bytes += table.h.row_bytes;
        return table.recs[slot_idx * rb ..][0..rb];
    }
    /// Portable prefetch is synchronous. The Source contract permits completion before return.
    pub fn issue(r: *Rows, li: usize, ids: []const i64) !void {
        try r.validate(li, ids);
        for (ids) |id| {
            _ = try r.record(li, @intCast(id));
            r.stats.issued += 1;
        }
    }
    pub fn rawRows(r: *Rows, li: usize, ids: []const i64, out: []u8) !void {
        try r.validate(li, ids);
        const rb: usize = @intCast(r.tables[li].h.row_bytes);
        if (out.len != try std.math.mul(usize, ids.len, rb)) return error.BadEngramRows;
        for (ids, 0..) |id, i| @memcpy(out[i * rb ..][0..rb], try r.record(li, @intCast(id)));
    }
    pub fn rows(r: *Rows, li: usize, ids: []const i64, dim: usize, out: []u16) !void {
        try r.validate(li, ids);
        if (r.tables[li].h.row_bytes != f.row_bytes or dim != 256 or out.len != try std.math.mul(usize, ids.len, 256)) return error.BadEngramShard;
        for (ids, 0..) |id, i| {
            const b = try r.record(li, @intCast(id));
            for (0..256) |j| out[i * 256 + j] = f.bf16(f.e4m3(b[j]) * f.scale(b[256 + j / 32]));
        }
    }
    pub fn source(r: *Rows) eh.Source {
        return .{ .ptr = r, .vtable = &.{ .issue = issueFn, .rows = rowsFn } };
    }
    fn issueFn(p: *anyopaque, li: usize, ids: []const i64) anyerror!void {
        const r: *Rows = @ptrCast(@alignCast(p));
        return r.issue(li, ids);
    }
    fn rowsFn(p: *anyopaque, li: usize, ids: []const i64, dim: usize, out: []u16) anyerror!void {
        const r: *Rows = @ptrCast(@alignCast(p));
        return r.rows(li, ids, dim, out);
    }
};
