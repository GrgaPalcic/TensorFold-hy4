//! `tf-hy4-engram-pack`: offline multithreaded fp64 packer for Hy4-flash's four Engram tables (`100` B/row, `.hy4e`).
const std = @import("std");
const hy4 = @import("hy4");

const ep = hy4.engram_pack;
const reader = hy4.engram_reader;
const Config = hy4.Config;

const usage =
    \\usage: tf-hy4-engram-pack pack   RELEASE_DIR OUT_DIR [--layers 1,11,21,31] [--chunk-rows N] [--threads T]
    \\       tf-hy4-engram-pack verify RELEASE_DIR OUT_DIR [--layers 1,11,21,31] [--samples N] [--seed S]
    \\Packs each Engram layer's bf16 [rows, 96] table in fp64 arithmetic into OUT_DIR/engram_layer<L>.hy4e
    \\(4,096-byte O_DIRECT header + 100 B/row: 96 e4m3 codes + fp32(amax / 448)), serving world 1, 2 and 4.
    \\
;

fn progress(layer: usize, done: u64, total: u64) void {
    if (done == total or (done / ep.default_chunk) % 16 == 0) {
        std.debug.print("layer {d}: {d} / {d} rows\n", .{ layer, done, total });
    }
}

fn parseLayers(a: std.mem.Allocator, text: []const u8) ![]usize {
    var out: std.ArrayList(usize) = .empty;
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |tok| try out.append(a, try std.fmt.parseInt(usize, tok, 10));
    return out.toOwnedSlice(a);
}

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 4 or !(std.mem.eql(u8, args[1], "pack") or std.mem.eql(u8, args[1], "verify"))) {
        std.debug.print("{s}", .{usage});
        return error.Usage;
    }
    const c = try Config.read(a, io, args[2]);
    var default_layers: [hy4.config.engram_layers]usize = undefined;
    for (c.engram_ids, 0..) |id, i| default_layers[i] = id - 1;
    var layers: []const usize = &default_layers;
    var chunk: usize = ep.default_chunk;
    var threads: usize = std.Thread.getCpuCount() catch 8;
    var samples: usize = 10_000;
    var seed: u64 = 1;
    var i: usize = 4;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) return error.Usage;
        const v = args[i + 1];
        if (std.mem.eql(u8, args[i], "--layers")) layers = try parseLayers(arena, v) else if (std.mem.eql(u8, args[i], "--chunk-rows")) chunk = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, args[i], "--threads")) threads = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, args[i], "--samples")) samples = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, args[i], "--seed")) seed = try std.fmt.parseInt(u64, v, 10) else return error.Usage;
    }
    if (std.mem.eql(u8, args[1], "pack")) {
        var dir = try std.Io.Dir.cwd().createDirPathOpen(io, args[3], .{});
        defer dir.close(io);
        for (layers) |l| {
            const rows = try ep.packLayer(a, io, args[2], dir, c, l, chunk, threads, progress);
            std.debug.print("layer {d}: packed {d} rows ({d} bytes)\n", .{ l, rows, ep.header_bytes + rows * ep.row_bytes });
        }
    }
    var total_rows: u64 = 0;
    var total_bad: u64 = 0;
    for (layers) |l| {
        const v = try ep.verifyLayer(a, io, args[2], args[3], c, l, samples, seed);
        total_rows += v.rows;
        total_bad += v.bad;
        const slot = c.engramSlot(l).?;
        // Also check NVMe reader at world 1, 2 and 4 on the first and last row of each rank's slice.
        for ([_]usize{ 1, 2, 4 }) |world| for (0..world) |rk| {
            var r = try reader.Reader.open(a, io, args[3], c, .{ .rank = rk, .world = world }, &.{slot}, .nvme, 4096);
            defer r.close();
            const sh = r.shares[0];
            var buf: [2 * ep.row_bytes]u8 = undefined;
            try r.fetch(0, &.{ @intCast(sh.lo), @intCast(sh.lo + sh.len - 1) }, &buf);
        };
    }
    std.debug.print("{s}: verified {d} rows across {d} layer(s), {d} differ\n", .{ if (total_bad == 0) "PASS" else "FAIL", total_rows, layers.len, total_bad });
    if (total_bad != 0) return error.VerifyFailed;
}
