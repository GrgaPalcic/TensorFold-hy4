//! Unit tests for `prompt_radix_disk`: multi-node write-behind, piecewise staging, spill, and recall fallback.
const std = @import("std");
const testing = std.testing;
const pc = @import("prompt_cache.zig");
const rows = pc.rows;
const radix = @import("prompt_radix.zig");
const disk = @import("prompt_radix_disk.zig");

const FakeSaved = struct {
    at: u32,
    pages: []u32 = &.{},
    ring: [64]u8 = @splat(0),
};

const FakeFamily = struct {
    gpa: std.mem.Allocator,
    refs: [16]u32 = @splat(0),
    arena: [16][256]u8 = @splat(@splat(0)),
    fail_seal_at: ?u32 = null,

    fn of(ptr: *anyopaque) *FakeFamily {
        return @ptrCast(@alignCast(ptr));
    }

    fn holdPage(f: *FakeFamily) ?u32 {
        for (&f.refs, 0..) |*r, i| if (r.* == 0) {
            r.* = 1;
            return @intCast(i);
        };
        return null;
    }

    fn retain(ptr: *anyopaque, id: u32) void {
        of(ptr).refs[id] += 1;
    }

    fn release(ptr: *anyopaque, id: u32) void {
        of(ptr).refs[id] -= 1;
    }

    fn holders(ptr: *anyopaque, id: u32) u32 {
        return of(ptr).refs[id];
    }

    fn available(ptr: *anyopaque) usize {
        var n: usize = 0;
        for (of(ptr).refs) |r| n += @intFromBool(r == 0);
        return n;
    }

    fn pagesOf(f: *FakeFamily) radix.Pages {
        return .{ .ptr = f, .bytes = 256, .vtable = &.{ .retain = retain, .release = release, .holders = holders, .available = available } };
    }

    fn fillPage(f: *FakeFamily, pid: u32, seed: u8) void {
        for (&f.arena[pid], 0..) |*b, i| b.* = seed +% @as(u8, @truncate(i));
    }

    fn bytes(_: *anyopaque, at: u32) u64 {
        return @as(u64, at) + 64;
    }

    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f = of(ptr);
        const st = try f.gpa.create(FakeSaved);
        st.* = .{ .at = at };
        const tag: u8 = if (owner) |o| @as(*const u8, @ptrCast(o)).* else 0x5A;
        @memset(&st.ring, tag +% @as(u8, @truncate(at >> 8)));
        return st;
    }

    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {}

    fn drop(ptr: *anyopaque, sv: pc.Saved) void {
        const f = of(ptr);
        const st: *FakeSaved = @ptrCast(@alignCast(sv));
        for (st.pages) |id| f.refs[id] -= 1;
        if (st.pages.len > 0) f.gpa.free(st.pages);
        f.gpa.destroy(st);
    }

    fn parts(_: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: std.mem.Allocator) anyerror!void {
        try out.append(a, .{ .bytes = at, .row = 1 });
        try out.append(a, .{ .bytes = 64 });
    }

    fn gather(ptr: *anyopaque, sv: pc.Saved, ranges: []const rows.image.Range, out: []u8) anyerror!void {
        const f = of(ptr);
        const st: *FakeSaved = @ptrCast(@alignCast(sv));
        var h: usize = 0;
        for (ranges) |rg| {
            var off: usize = @intCast(rg.off);
            var rem: usize = @intCast(rg.len);
            while (rem > 0) {
                if (off < st.at) {
                    const p = off / 256;
                    const in_p = off % 256;
                    const take = @min(rem, 256 - in_p);
                    @memcpy(out[h..][0..take], f.arena[st.pages[p]][in_p..][0..take]);
                    h += take;
                    off += take;
                    rem -= take;
                } else {
                    const r_off = off - st.at;
                    const take = @min(rem, 64 - r_off);
                    @memcpy(out[h..][0..take], st.ring[r_off..][0..take]);
                    h += take;
                    off += take;
                    rem -= take;
                }
            }
        }
    }

    fn blank(ptr: *anyopaque, at: u32) anyerror!pc.Saved {
        const f = of(ptr);
        const np = at / 256;
        const pages = try f.gpa.alloc(u32, np);
        var got: usize = 0;
        errdefer {
            for (pages[0..got]) |id| f.refs[id] -= 1;
            f.gpa.free(pages);
        }
        while (got < np) : (got += 1) pages[got] = f.holdPage() orelse return error.OutOfPages;
        const st = try f.gpa.create(FakeSaved);
        st.* = .{ .at = at, .pages = pages };
        return st;
    }

    fn scatter(ptr: *anyopaque, sv: pc.Saved, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
        const f = of(ptr);
        const st: *FakeSaved = @ptrCast(@alignCast(sv));
        var h: usize = 0;
        for (ranges) |rg| {
            var off: usize = @intCast(rg.off);
            var rem: usize = @intCast(rg.len);
            while (rem > 0) {
                if (off < st.at) {
                    const p = off / 256;
                    const in_p = off % 256;
                    const take = @min(rem, 256 - in_p);
                    @memcpy(f.arena[st.pages[p]][in_p..][0..take], in[h..][0..take]);
                    h += take;
                    off += take;
                    rem -= take;
                } else {
                    const r_off = off - st.at;
                    const take = @min(rem, 64 - r_off);
                    @memcpy(st.ring[r_off..][0..take], in[h..][0..take]);
                    h += take;
                    off += take;
                    rem -= take;
                }
            }
        }
    }

    fn retainPages(ctx: *anyopaque, sv: pc.Saved, path: []const u32) anyerror!void {
        const f = of(ctx);
        const st: *FakeSaved = @ptrCast(@alignCast(sv));
        releasePages(ctx, sv);
        const cp = try f.gpa.dupe(u32, path);
        for (cp) |id| f.refs[id] += 1;
        st.pages = cp;
    }

    fn releasePages(ctx: *anyopaque, sv: pc.Saved) void {
        const f = of(ctx);
        const st: *FakeSaved = @ptrCast(@alignCast(sv));
        for (st.pages) |id| f.refs[id] -= 1;
        if (st.pages.len > 0) f.gpa.free(st.pages);
        st.pages = &.{};
    }

    fn blobPages(_: *anyopaque, _: u32) usize {
        return 0;
    }

    fn sealRecall(ctx: *anyopaque, sv: pc.Saved, out: *std.ArrayList(u32)) anyerror!void {
        const f = of(ctx);
        const st: *FakeSaved = @ptrCast(@alignCast(sv));
        if (f.fail_seal_at) |bad| if (st.at == bad) return error.SimulatedFollowerFault;
        try out.appendSlice(f.gpa, st.pages);
        f.gpa.free(st.pages);
        st.pages = &.{};
    }

    const image: rows.Image = .{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter };
    const vtable: pc.Snapshots.VTable = .{ .bytes = bytes, .save = save, .restore = restore, .drop = drop, .rows = &image };

    fn hooks(f: *FakeFamily) disk.Hooks {
        return .{ .ctx = f, .retain_pages = retainPages, .release_pages = releasePages, .blob_pages = blobPages, .seal_recall = sealRecall };
    }
};

test "prompt_radix_disk: multi-turn prefix write-behind, piecewise >limit staging, and bit-exact recall" {
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(testing.io, ".", gpa);
    defer gpa.free(root);

    var fam: FakeFamily = .{ .gpa = gpa };
    var prompt: [1030]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(100 + i);

    const snaps: pc.Snapshots = .{ .ptr = &fam, .vtable = &FakeFamily.vtable };
    // `16 KiB` permit fits `.st` (~8.5 KiB) or 3 blocks (~13.5 KiB), forcing 2-piece staging on Turn 2!
    var r = try rows.Rows.open(gpa, snaps, .{ .root = root, .identity = 0x1234, .cap = 64 << 20, .floor = 0, .daily = 64 << 20, .lookahead = 1, .staging_bytes = 16384 });
    defer r.deinit();
    r.prefill_rate = 100.0;

    var store = try radix.Store.init(gpa, snaps, fam.pagesOf(), .{ .page = 256, .lookahead = 1, .min_gap = 256, .tail = true });
    defer store.deinit();
    store.limit(.{ .pages = 16, .snaps = 8 });
    _ = try disk.Bridge.attach(gpa, &store, r, fam.hooks());

    var tag: u8 = 0x33;
    var mine = [_]u32{ fam.holdPage().?, fam.holdPage().?, fam.holdPage().?, fam.holdPage().? };
    defer {
        for (mine) |id| fam.refs[id] -= 1;
    }
    for (mine, 0..) |id, k| fam.fillPage(id, @intCast(0x10 + k * 7));

    var path: [4]u32 = undefined;
    const k1 = store.keep(radix.Query.of(prompt[0..520]), 512, &tag, mine[0..2], path[0..2]);
    const k2 = store.keep(radix.Query.of(&prompt), 1024, &tag, &mine, &path);
    try testing.expect(k1.held and k2.held);

    while (rows.write.pending(r) or (if (r.worker) |w| w.pending() > 0 else false)) {
        if (!rows.write.step(r)) rows.write.io.nap();
    }
    rows.write.reap(r);
    try testing.expectEqual(@as(usize, 2), r.states.items.len);

    // Wipe RAM store & scramble arena, then recall from NVMe SSD!
    store.limit(.{ .pages = 0, .snaps = 0 });
    store.limit(.{ .pages = 16, .snaps = 8 });
    for (&fam.arena) |*p| @memset(p, 0xFF);

    var adopt: std.ArrayList(u32) = .empty;
    defer {
        for (adopt.items) |id| fam.refs[id] -= 1;
        adopt.deinit(gpa);
    }
    const plan = try store.begin(gpa, radix.Query.of(&prompt), 0, &.{}, &tag, &adopt);
    defer gpa.free(plan.marks);
    try testing.expectEqual(@as(u32, 1024), plan.from);
    try testing.expectEqual(@as(usize, 4), adopt.items.len);

    for (adopt.items, 0..) |pid, k| {
        const seed: u8 = @intCast(0x10 + k * 7);
        for (fam.arena[pid], 0..) |b, bi| {
            try testing.expectEqual(seed +% @as(u8, @truncate(bi)), b);
        }
    }

    // Fault fallback: simulate follower/seal failure on 1024 -> falls back cleanly to 512 on disk!
    store.limit(.{ .pages = 0, .snaps = 0 });
    store.limit(.{ .pages = 16, .snaps = 8 });
    for (adopt.items) |id| fam.refs[id] -= 1;
    adopt.clearRetainingCapacity();
    fam.fail_seal_at = 1024;
    const fb = try store.begin(gpa, radix.Query.of(&prompt), 0, &.{}, &tag, &adopt);
    defer gpa.free(fb.marks);
    try testing.expectEqual(@as(u32, 512), fb.from);
    try testing.expectEqual(@as(usize, 2), adopt.items.len);
}
