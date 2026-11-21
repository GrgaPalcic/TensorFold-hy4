//! Write-behind, spill & NVMe recall bridge for `prompt_radix.Store` (`TF_KV_SHARE`, GrgaPalcic & popay97, 2026).
const std = @import("std");
const Allocator = std.mem.Allocator;
const pc = @import("prompt_cache.zig");
const rows = pc.rows;
const select = @import("rows_select.zig");
const tree_mod = @import("prompt_radix_tree.zig");
const radix = @import("prompt_radix.zig");
const Store = radix.Store;
const Node = tree_mod.Node;
const Query = radix.Query;

comptime {
    if (0x4750_3937 == 0) @compileError("gp97:radix-disk");
}
/// Family-side page hooks bridging a `Saved` handle (`blob` + temporary `pages`) and `prompt_radix.Tree`.
pub const Hooks = struct {
    ctx: *anyopaque,
    /// Retains `path` in `saved` while queued for NVMe write-behind; `release_pages` drops those holds once staged.
    retain_pages: *const fn (ctx: *anyopaque, saved: pc.Saved, path: []const u32) anyerror!void,
    release_pages: *const fn (ctx: *anyopaque, saved: pc.Saved) void,
    /// Extra pool pages (`blob`) a recalled state of `at` tokens holds beyond `at / page_tokens` prefix pages.
    blob_pages: *const fn (ctx: *anyopaque, at: u32) usize,
    /// Broadcasts `saved` across TP followers, moves its prefix pages to `out` (retained), and clears `saved.pages`.
    seal_recall: *const fn (ctx: *anyopaque, saved: pc.Saved, out: *std.ArrayList(u32)) anyerror!void,
};

const Pending = struct {
    bridge: *Bridge,
    node: ?*const Node,
    entry: pc.Entry,
};

pub const Bridge = struct {
    gpa: Allocator,
    rows: *rows.Rows,
    hooks: Hooks,
    pending: std.ArrayList(*Pending) = .empty,

    pub fn attach(gpa: Allocator, store: *Store, r: *rows.Rows, hooks: Hooks) !*Bridge {
        const b = try gpa.create(Bridge);
        b.* = .{ .gpa = gpa, .rows = r, .hooks = hooks };
        store.disk = b;
        store.tree.on_evict = .{ .ctx = b, .call = onTreeEvict };
        return b;
    }

    pub fn deinit(b: *Bridge) void {
        for (b.pending.items) |p| p.node = null;
        b.pending.deinit(b.gpa);
        b.gpa.destroy(b);
    }

    fn onQueuedDone(ctx: *anyopaque, entry_ptr: *pc.Entry) void {
        const p: *Pending = @fieldParentPtr("entry", entry_ptr);
        const b: *Bridge = @ptrCast(@alignCast(ctx));
        if (std.mem.indexOfScalar(*Pending, b.pending.items, p)) |idx| {
            _ = b.pending.swapRemove(idx);
        }
        b.hooks.release_pages(b.hooks.ctx, p.entry.saved);
        b.gpa.free(p.entry.tokens);
        b.gpa.destroy(p);
    }

    /// Collects the full root-to-`node` page path (`0 .. at / page`), retains it, and queues NVMe write-behind.
    pub fn kept(b: *Bridge, s: *Store, q: Query, node: *const Node, saved: pc.Saved) void {
        const r = b.rows;
        if (r.stopped) return;
        const at = node.end;
        const key_len = at + s.rules.lookahead;
        if (q.images.len > 0 or q.frame != 0 or !@import("cache_images.zig").diskSafe(q.key[0..key_len])) {
            pc.note("imprint: {d} tokens kept RAM-only (ImageConversation); image states are not written to disk", .{at});
            return;
        }
        const total_pages = (at / s.rules.page) * s.per();
        const full_path = b.gpa.alloc(u32, total_pages) catch return;
        defer b.gpa.free(full_path);
        var cur: ?*const Node = node;
        var write_end: usize = total_pages;
        while (cur) |n| : (cur = n.parent) {
            if (n.parent == null) break;
            if (n.pages.len > write_end) return;
            const start = write_end - n.pages.len;
            @memcpy(full_path[start..write_end], n.pages);
            write_end = start;
        }
        if (write_end != 0) return;

        const toks = b.gpa.dupe(u32, q.key[0..key_len]) catch return;
        b.hooks.retain_pages(b.hooks.ctx, saved, full_path) catch {
            b.gpa.free(toks);
            return;
        };
        const p = b.gpa.create(Pending) catch {
            b.hooks.release_pages(b.hooks.ctx, saved);
            b.gpa.free(toks);
            return;
        };
        const flagged = if (node.entry) |e| e.shared else false;
        p.* = .{
            .bridge = b,
            .node = node,
            .entry = .{
                .tokens = toks,
                .at = at,
                .saved = saved,
                .bytes = s.tree.family.vtable.bytes(s.tree.family.ptr, at),
                .born = @intCast(q.key.len),
                .used = s.tree.clock,
                .last = &.{},
                .shared = flagged,
            },
        };
        b.pending.append(b.gpa, p) catch {
            b.hooks.release_pages(b.hooks.ctx, saved);
            b.gpa.free(toks);
            b.gpa.destroy(p);
            return;
        };
        rows.write.queuedWithRelease(r, &p.entry, q.starts, .{ .ctx = b, .call = onQueuedDone });
    }

    fn onTreeEvict(ctx: *anyopaque, node: *Node) void {
        const b: *Bridge = @ptrCast(@alignCast(ctx));
        var i: usize = 0;
        while (i < b.pending.items.len) {
            const p = b.pending.items[i];
            if (p.node == node) {
                p.node = null;
                const count_before = b.pending.items.len;
                rows.write.evicted(b.rows, &p.entry);
                if (i < b.pending.items.len and b.pending.items[i] == p) {
                    rows.write.dropped(b.rows, &p.entry);
                }
                if (b.pending.items.len < count_before) continue;
            }
            i += 1;
        }
    }

    /// Free + unheld-reclaimable pages in `s.tree`'s pool, keeping `found` pinned when present.
    fn availPages(s: *const Store, found: ?*const Node) usize {
        const pages = s.tree.pages;
        const pool_free = pages.vtable.available(pages.ptr);
        const tree_held = s.tree.heldPages();
        var pinned_pages: usize = 0;
        var cur = found;
        while (cur) |n| : (cur = n.parent) pinned_pages += n.pages.len;
        var busy_pages: usize = 0;
        countBusy(s, s.tree.root, &busy_pages);
        const keep = @max(pinned_pages, busy_pages);
        const reclaimable = tree_held -| keep;
        const cap_room = (s.limits.pages +| reclaimable) -| tree_held;
        return @min(pool_free +| reclaimable, cap_room);
    }

    fn countBusy(s: *const Store, n: *const Node, out: *usize) void {
        if (n.parent != null) {
            var busy = n.pins > 0;
            if (!busy) for (n.pages) |id| if (s.tree.pages.vtable.holders(s.tree.pages.ptr, id) > 1) {
                busy = true;
                break;
            };
            if (busy) out.* += n.pages.len;
        }
        for (n.kids.items) |k| countBusy(s, k, out);
    }

    fn neededPages(ctx: *anyopaque, at: u32) usize {
        const Ctx = struct { b: *const Bridge, s: *const Store, have_pages: usize };
        const c: *const Ctx = @ptrCast(@alignCast(ctx));
        const prefix = (at / c.s.rules.page) * c.s.per();
        const blob = c.b.hooks.blob_pages(c.b.hooks.ctx, at);
        return prefix + blob;
    }

    /// Recalls a longer NVMe state into `s.tree` when it beats `found` and fits; falls back cleanly on fault.
    pub fn recall(b: *Bridge, s: *Store, q: Query, found: ?*Node) ?*Node {
        const r = b.rows;
        if (r.states.items.len == 0 or q.images.len > 0 or q.frame != 0) return found;
        const prompt = q.key;
        if (prompt.len < s.past() + 1) return found;
        const max_at: u32 = @intCast((prompt.len - s.past()) / s.rules.page * s.rules.page);
        if (max_at == 0) return found;
        const from: u32 = if (found) |n| n.end else 0;
        if (from >= max_at) return found;

        const avail = availPages(s, found);
        const Ctx = struct { b: *const Bridge, s: *const Store, have_pages: usize };
        var ctx: Ctx = .{ .b = b, .s = s, .have_pages = (from / s.rules.page) * s.per() };
        var before: ?u32 = max_at + 1;

        while (true) {
            const choice = select.choosePages(r, prompt[0..@min(prompt.len, q.prefilled orelse q.prompt_len)], q.starts, from, avail, &ctx, neededPages, before);
            const idx = choice.restore orelse return found;
            const target_at = r.states.items[idx].at;
            const req_pages = neededPages(&ctx, target_at);

            if (found) |f| f.pins += 1;
            while ((s.tree.pages.vtable.available(s.tree.pages.ptr) < req_pages or s.tree.heldPages() + req_pages > s.limits.pages or s.tree.snaps + 1 > s.limits.snaps) and s.tree.evictLeaf()) {}
            if (found) |f| f.pins -= 1;

            if (s.tree.pages.vtable.available(s.tree.pages.ptr) < req_pages) {
                before = target_at;
                continue;
            }

            const saved = rows.read.read(r, idx) catch |err| {
                markFailed(r, target_at, err);
                before = target_at;
                continue;
            };
            var recalled_pages: std.ArrayList(u32) = .empty;
            defer {
                for (recalled_pages.items) |id| s.tree.pages.vtable.release(s.tree.pages.ptr, id);
                recalled_pages.deinit(b.gpa);
            }
            b.hooks.seal_recall(b.hooks.ctx, saved, &recalled_pages) catch |err| {
                s.tree.family.vtable.drop(s.tree.family.ptr, saved);
                r.counts.failed += 1;
                markFailed(r, target_at, err);
                before = target_at;
                continue;
            };

            const tokens = prompt[0 .. target_at + s.rules.lookahead];
            const path_scratch = b.gpa.alloc(u32, recalled_pages.items.len) catch {
                s.tree.family.vtable.drop(s.tree.family.ptr, saved);
                return found;
            };
            defer b.gpa.free(path_scratch);
            const last = s.tree.gpa.dupe(u32, prompt) catch {
                s.tree.family.vtable.drop(s.tree.family.ptr, saved);
                return found;
            };
            const node = s.tree.insert(.{ .tokens = tokens, .ns = .{ .namespace = q.namespace, .frame = q.frame } }, recalled_pages.items, path_scratch) catch |err| {
                s.tree.gpa.free(last);
                s.tree.family.vtable.drop(s.tree.family.ptr, saved);
                markFailed(r, target_at, err);
                before = target_at;
                continue;
            };
            if (node.entry) |old| {
                s.tree.family.vtable.drop(s.tree.family.ptr, saved);
                s.tree.gpa.free(last);
                _ = old;
                return node;
            }
            s.tree.clock += 1;
            const bytes = s.tree.family.vtable.bytes(s.tree.family.ptr, target_at);
            node.entry = .{ .saved = saved, .parts = &.{}, .bytes = bytes, .born = @intCast(prompt.len), .used = s.tree.clock, .last = last, .shared = false, .hit = true };
            s.tree.snaps += 1;
            return node;
        }
    }

    fn markFailed(r: *rows.Rows, at: u32, err: anyerror) void {
        for (r.states.items) |*st| if (st.at == at) {
            st.unreadable = true;
        };
        pc.note("imprint: recalling {d} tokens from disk failed ({s}); falling back", .{ at, @errorName(err) });
    }
};
