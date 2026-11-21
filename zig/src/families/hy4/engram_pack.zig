//! Hy4-flash single-file Engram FP8 row packer (`100` B/row: `96` e4m3 codes from fp64 + `fp32` scale, 4 KiB header).
const std = @import("std");
const builtin = @import("builtin");
const er = @import("engram_rows");
const config = @import("config.zig");
const engram_hash = @import("engram_hash.zig");

const Config = config.Config;
const Io = std.Io;

pub const magic: u64 = 0x3152474E45345948; // "HY4ENGR1" little-endian
pub const header_bytes: u64 = 4096;
pub const dim: usize = 96;
pub const row_bytes: usize = dim + 4;
pub const fp8_max: f64 = 448.0;
pub const default_chunk: usize = 1 << 20;

pub fn fileName(buf: []u8, layer: usize) ![]u8 {
    return std.fmt.bufPrint(buf, "engram_layer{d}.hy4e", .{layer});
}

pub const Header = struct {
    layer: u32,
    layer_id: u32,
    seed: u64,
    rows: u64,
    sizes: [config.engram_heads]u64,

    pub fn of(c: Config, slot: usize) Header {
        var total: u64 = 0;
        for (c.engram_sizes[slot]) |s| total += s;
        return .{
            .layer = c.engram_ids[slot] - 1,
            .layer_id = c.engram_ids[slot],
            .seed = c.engram_seed,
            .rows = total,
            .sizes = c.engram_sizes[slot],
        };
    }

    pub fn encode(h: Header) [header_bytes]u8 {
        var b: [header_bytes]u8 = @splat(0);
        const words = [8]u64{ magic, h.layer, h.layer_id, h.seed, h.rows, dim, row_bytes, config.engram_heads };
        for (words, 0..) |w, i| std.mem.writeInt(u64, b[8 * i ..][0..8], w, .little);
        for (h.sizes, 0..) |s, i| std.mem.writeInt(u64, b[64 + 8 * i ..][0..8], s, .little);
        return b;
    }

    pub fn parse(b: *const [header_bytes]u8, file_size: u64) !Header {
        var w: [8]u64 = undefined;
        for (&w, 0..) |*x, i| x.* = std.mem.readInt(u64, b[8 * i ..][0..8], .little);
        if (w[0] != magic) return error.NotHy4EngramPack;
        if (w[5] != dim or w[6] != row_bytes or w[7] != config.engram_heads) return error.BadHy4EngramPack;
        if (w[1] + 1 != w[2] or w[2] > std.math.maxInt(u32) or w[4] == 0) return error.BadHy4EngramPack;
        var sizes: [config.engram_heads]u64 = undefined;
        var sum: u64 = 0;
        for (&sizes, 0..) |*s, i| {
            s.* = std.mem.readInt(u64, b[64 + 8 * i ..][0..8], .little);
            if (s.* == 0) return error.BadHy4EngramPack;
            sum = std.math.add(u64, sum, s.*) catch return error.BadHy4EngramPack;
        }
        if (sum != w[4]) return error.BadHy4EngramPack;
        const payload = std.math.mul(u64, w[4], row_bytes) catch return error.BadHy4EngramPack;
        const want = std.math.add(u64, header_bytes, payload) catch return error.BadHy4EngramPack;
        if (file_size != want) return error.BadHy4EngramPack;
        return .{ .layer = @intCast(w[1]), .layer_id = @intCast(w[2]), .seed = w[3], .rows = w[4], .sizes = sizes };
    }

    pub fn checkConfig(h: Header, c: Config, slot: usize, why: *config.Why) !void {
        if (h.layer_id != c.engram_ids[slot] or h.seed != c.engram_seed) {
            why.set("Engram pack layer_id {d} seed {d} != config layer_id {d} seed {d}", .{ h.layer_id, h.seed, c.engram_ids[slot], c.engram_seed });
            return error.UnsupportedEngram;
        }
        for (h.sizes, c.engram_sizes[slot], 0..) |got, want, i| if (got != want) {
            why.set("Engram pack layer {d} head {d}: {d} rows != config {d}", .{ h.layer, i, got, want });
            return error.UnsupportedEngram;
        };
    }
};

/// Exact float32 -> float8_e4m3fn RNE, matching PyTorch `.float().to(torch.float8_e4m3fn)` bit-for-bit.
pub fn e4m3Rne(x: f32) u8 {
    const bits: u32 = @bitCast(x);
    const sign: u8 = @intCast((bits >> 31) << 7);
    const ab: u32 = bits & 0x7fff_ffff;
    if (ab > 0x43e8_0000) return sign | 0x7f;
    if (ab < 0x3a80_0000) return sign;
    if (ab < 0x3c80_0000) {
        const exp = ab >> 23;
        const sig: u32 = 0x80_0000 | (ab & 0x7f_ffff);
        const shift: u5 = @intCast(141 - exp);
        var q: u32 = sig >> shift;
        const rem: u32 = sig & ((@as(u32, 1) << shift) - 1);
        const half: u32 = @as(u32, 1) << (shift - 1);
        if (rem > half or (rem == half and (q & 1) == 1)) q += 1;
        return sign | @as(u8, @intCast(q));
    }
    const exp = ab >> 23;
    const mant = ab & 0x7f_ffff;
    var q: u32 = ((exp - 120) << 3) | (mant >> 20);
    const rem = mant & 0x0f_ffff;
    if (rem > 0x08_0000 or (rem == 0x08_0000 and (q & 1) == 1)) q += 1;
    return sign | @as(u8, @intCast(q));
}

/// Packs one 96-wide bf16 row into 96 e4m3 codes + fp32(amax / 448) using fp64 arithmetic (hy4_engram_check.py).
pub fn packRow(bf16_bytes: *const [dim * 2]u8, out: *[row_bytes]u8) void {
    var x64: [dim]f64 = undefined;
    var amax: f64 = 1e-12;
    for (0..dim) |j| {
        const u = std.mem.readInt(u16, bf16_bytes[2 * j ..][0..2], .little);
        const v: f64 = @as(f32, @bitCast(@as(u32, u) << 16));
        x64[j] = v;
        const a = @abs(v);
        if (a > amax) amax = a;
    }
    const s64: f64 = amax / fp8_max;
    for (0..dim) |j| {
        const clamped = std.math.clamp(x64[j] / s64, -fp8_max, fp8_max);
        out[j] = e4m3Rne(@floatCast(clamped));
    }
    const s32: f32 = @floatCast(s64);
    std.mem.writeInt(u32, out[dim..][0..4], @bitCast(s32), .little);
}

/// Decodes one 100-byte packed row as `code[d] * scale` in fp32, matching the GPU kernel's arithmetic.
pub fn unpackRowF32(rec: *const [row_bytes]u8, out: *[dim]f32) void {
    const scale: f32 = @bitCast(std.mem.readInt(u32, rec[dim..][0..4], .little));
    for (0..dim) |j| out[j] = er.e4m3(rec[j]) * scale;
}

pub const TableLoc = struct {
    shard_path: []u8,
    data_offset: u64,
    rows: u64,

    pub fn deinit(t: *TableLoc, gpa: std.mem.Allocator) void {
        gpa.free(t.shard_path);
        t.* = undefined;
    }
};

/// Locates `model.layers.{layer}.engram_layer.multi_head_embedding.embedding.weight` without mapping every shard.
pub fn locateTable(gpa: std.mem.Allocator, io: Io, model_dir: []const u8, layer: usize) !TableLoc {
    var tbuf: [128]u8 = undefined;
    const tname = try std.fmt.bufPrint(&tbuf, "model.layers.{d}.engram_layer.multi_head_embedding.embedding.weight", .{layer});
    const idx_path = try std.fs.path.join(gpa, &.{ model_dir, "model.safetensors.index.json" });
    defer gpa.free(idx_path);
    const shard_path = blk: {
        if (Io.Dir.cwd().readFileAlloc(io, idx_path, gpa, .limited(1 << 26))) |text| {
            defer gpa.free(text);
            var parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
            defer parsed.deinit();
            if (parsed.value != .object) return error.BadIndex;
            const wm = parsed.value.object.get("weight_map") orelse return error.BadIndex;
            if (wm != .object) return error.BadIndex;
            const sv = wm.object.get(tname) orelse return error.MissingEngramTable;
            if (sv != .string) return error.BadIndex;
            break :blk try std.fs.path.join(gpa, &.{ model_dir, sv.string });
        } else |e| switch (e) {
            error.FileNotFound => break :blk try std.fs.path.join(gpa, &.{ model_dir, "model.safetensors" }),
            else => return e,
        }
    };
    errdefer gpa.free(shard_path);
    const f = try Io.Dir.cwd().openFile(io, shard_path, .{});
    defer f.close(io);
    var len_bytes: [8]u8 = undefined;
    if (try f.readPositionalAll(io, &len_bytes, 0) != 8) return error.BadSafetensors;
    const hlen = std.mem.readInt(u64, &len_bytes, .little);
    if (hlen == 0 or hlen > (1 << 26)) return error.BadSafetensors;
    const hjson = try gpa.alloc(u8, @intCast(hlen));
    defer gpa.free(hjson);
    if (try f.readPositionalAll(io, hjson, 8) != hjson.len) return error.BadSafetensors;
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, hjson, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadSafetensors;
    const entry = parsed.value.object.get(tname) orelse return error.MissingEngramTable;
    if (entry != .object) return error.BadSafetensors;
    const dtype = entry.object.get("dtype") orelse return error.BadSafetensors;
    if (dtype != .string or !std.mem.eql(u8, dtype.string, "BF16")) return error.UnexpectedTensor;
    const shape = entry.object.get("shape") orelse return error.BadSafetensors;
    if (shape != .array or shape.array.items.len != 2) return error.UnexpectedTensor;
    if (shape.array.items[0] != .integer or shape.array.items[0].integer <= 0) return error.UnexpectedTensor;
    if (shape.array.items[1] != .integer or shape.array.items[1].integer != dim) return error.UnexpectedTensor;
    const rows: u64 = @intCast(shape.array.items[0].integer);
    const offs = entry.object.get("data_offsets") orelse return error.BadSafetensors;
    if (offs != .array or offs.array.items.len != 2) return error.BadSafetensors;
    if (offs.array.items[0] != .integer or offs.array.items[1] != .integer or offs.array.items[0].integer < 0) return error.BadSafetensors;
    const off0: u64 = @intCast(offs.array.items[0].integer);
    const off1: u64 = @intCast(offs.array.items[1].integer);
    if (off1 < off0 or off1 - off0 != rows * dim * 2) return error.BadSafetensors;
    const base = 8 + hlen + off0;
    if (base + (off1 - off0) > try f.length(io)) return error.BadSafetensors;
    return .{ .shard_path = shard_path, .data_offset = base, .rows = rows };
}

fn packSlice(in_bytes: []const u8, out_bytes: []u8, r0: usize, r1: usize) void {
    for (r0..r1) |i| packRow(in_bytes[i * dim * 2 ..][0 .. dim * 2], out_bytes[i * row_bytes ..][0..row_bytes]);
}

fn dropCache(file: Io.File, off: u64, len: u64) void {
    if (comptime builtin.os.tag == .linux) {
        _ = std.os.linux.fadvise(file.handle, @intCast(off), @intCast(len), std.os.linux.POSIX_FADV.DONTNEED);
    }
}

pub const Progress = *const fn (layer: usize, done: u64, total: u64) void;

/// Streams layer `layer`'s bf16 Engram table into `<out_dir>/engram_layer<layer>.hy4e` across `threads` workers.
pub fn packLayer(gpa: std.mem.Allocator, io: Io, model_dir: []const u8, out_dir: Io.Dir, c: Config, layer: usize, chunk_rows: usize, threads: usize, progress: ?Progress) !u64 {
    if (chunk_rows == 0) return error.InvalidChunk;
    const slot = c.engramSlot(layer) orelse return error.NotEngramLayer;
    var why: config.Why = .{};
    try engram_hash.check(c, &why);
    const head = Header.of(c, slot);
    var loc = try locateTable(gpa, io, model_dir, layer);
    defer loc.deinit(gpa);
    if (loc.rows != head.rows) return error.UnexpectedTensor;

    const shard = try Io.Dir.cwd().openFile(io, loc.shard_path, .{});
    defer shard.close(io);
    var nbuf: [64]u8 = undefined;
    const name = try fileName(&nbuf, layer);
    var pbuf: [72]u8 = undefined;
    const part = try std.fmt.bufPrint(&pbuf, "{s}.part", .{name});
    const out_file = try out_dir.createFile(io, part, .{});
    var closed = false;
    errdefer {
        if (!closed) out_file.close(io);
        out_dir.deleteFile(io, part) catch {};
    }
    const hbytes = head.encode();
    try out_file.writePositionalAll(io, &hbytes, 0);

    const step: usize = @intCast(@min(@as(u64, chunk_rows), head.rows));
    const in_buf = try gpa.alloc(u8, step * dim * 2);
    defer gpa.free(in_buf);
    const out_buf = try gpa.alloc(u8, step * row_bytes);
    defer gpa.free(out_buf);
    const max_workers = @max(@min(threads, 64), 1);
    const pool = try gpa.alloc(std.Thread, max_workers);
    defer gpa.free(pool);

    var done: u64 = 0;
    while (done < head.rows) {
        const k: usize = @intCast(@min(@as(u64, step), head.rows - done));
        const in_off = loc.data_offset + done * dim * 2;
        if (try shard.readPositionalAll(io, in_buf[0 .. k * dim * 2], in_off) != k * dim * 2) return error.ShortRead;
        dropCache(shard, in_off & ~@as(u64, 4095), k * dim * 2);

        const workers = @min(max_workers, @max((k + 4095) / 4096, 1));
        if (workers <= 1) {
            packSlice(in_buf, out_buf, 0, k);
        } else {
            const share = (k + workers - 1) / workers;
            var spawned: usize = 0;
            for (0..workers) |w| {
                const r0 = w * share;
                if (r0 >= k) break;
                const r1 = @min(r0 + share, k);
                pool[spawned] = try std.Thread.spawn(.{ .stack_size = 1 << 19 }, packSlice, .{ in_buf, out_buf, r0, r1 });
                spawned += 1;
            }
            for (pool[0..spawned]) |th| th.join();
        }
        try out_file.writePositionalAll(io, out_buf[0 .. k * row_bytes], header_bytes + done * row_bytes);
        done += k;
        if ((done / step) % 8 == 0) {
            try out_file.sync(io);
            dropCache(out_file, 0, header_bytes + done * row_bytes);
        }
        if (progress) |cb| cb(layer, done, head.rows);
    }
    try out_file.sync(io);
    dropCache(out_file, 0, 0);
    out_file.close(io);
    closed = true;
    try out_dir.rename(part, out_dir, name, io);
    return head.rows;
}

pub const Verified = struct { rows: u64 = 0, bad: u64 = 0 };

/// Verifies `samples` random rows plus boundary rows of `<engram_dir>/engram_layer<layer>.hy4e` against the shard.
pub fn verifyLayer(gpa: std.mem.Allocator, io: Io, model_dir: []const u8, engram_dir: []const u8, c: Config, layer: usize, samples: usize, seed: u64) !Verified {
    const slot = c.engramSlot(layer) orelse return error.NotEngramLayer;
    var loc = try locateTable(gpa, io, model_dir, layer);
    defer loc.deinit(gpa);
    var nbuf: [64]u8 = undefined;
    const ppath = try std.fs.path.join(gpa, &.{ engram_dir, try fileName(&nbuf, layer) });
    defer gpa.free(ppath);
    const pfile = try Io.Dir.cwd().openFile(io, ppath, .{});
    defer pfile.close(io);
    var hb: [header_bytes]u8 = undefined;
    if (try pfile.readPositionalAll(io, &hb, 0) != header_bytes) return error.BadHy4EngramPack;
    const h = try Header.parse(&hb, try pfile.length(io));
    var why: config.Why = .{};
    try h.checkConfig(c, slot, &why);

    const shard = try Io.Dir.cwd().openFile(io, loc.shard_path, .{});
    defer shard.close(io);
    var ids: std.ArrayList(u64) = .empty;
    defer ids.deinit(gpa);
    try ids.append(gpa, 0);
    try ids.append(gpa, h.rows - 1);
    var acc: u64 = 0;
    for (h.sizes) |s| {
        try ids.append(gpa, acc);
        try ids.append(gpa, acc + s - 1);
        acc += s;
    }
    var prng = std.Random.DefaultPrng.init(seed ^ layer);
    const rng = prng.random();
    for (0..samples) |_| try ids.append(gpa, rng.uintLessThan(u64, h.rows));

    var v: Verified = .{};
    var bf_row: [dim * 2]u8 = undefined;
    var want_rec: [row_bytes]u8 = undefined;
    var got_rec: [row_bytes]u8 = undefined;
    for (ids.items) |r| {
        if (try shard.readPositionalAll(io, &bf_row, loc.data_offset + r * dim * 2) != bf_row.len) return error.ShortRead;
        if (try pfile.readPositionalAll(io, &got_rec, header_bytes + r * row_bytes) != got_rec.len) return error.ShortRead;
        packRow(&bf_row, &want_rec);
        v.rows += 1;
        if (!std.mem.eql(u8, &want_rec, &got_rec)) v.bad += 1;
    }
    return v;
}

test {
    _ = @import("engram_pack_test.zig");
}
