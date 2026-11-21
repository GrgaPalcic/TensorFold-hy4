//! Incremental SSE argument streaming for Hy4-flash `<tool_calls:HYTK>` and `<tool_call:HYTK>` envelopes.
const std = @import("std");
const json = @import("json");
const reply_text = @import("reply_text.zig");
const tool_hy4 = @import("tool_hy4.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;
const Tags = tool_hy4.Tags;

pub const Next = enum { more, wait, prose, rejected, outside };
pub const Stage = enum { calls, name, pairs, key, value };

/// Resumed streaming state for a `<tool_calls{suf}>` or `<tool_call:suf>` block; stores offsets only.
pub const Stream = struct {
    start: usize = 0,
    suf_at: usize = 0,
    suf_len: usize = 0,
    open_len: usize = 0,
    outer: bool = true,
    stage: Stage = .calls,
    at: usize = 0,
    from: usize = 0,
    val_sent: usize = 0,
    sent: usize = 0,

    pub fn initFrom(h: tool_hy4.Head, start: usize) Stream {
        const first = start + h.open_len;
        return .{
            .start = start,
            .suf_at = start + h.suf_off,
            .suf_len = h.suf_len,
            .open_len = h.open_len,
            .outer = h.outer,
            .stage = if (h.outer) .calls else .name,
            .at = first,
            .from = first,
        };
    }

    pub fn blockClose(h: *const Stream, a: Allocator, text: []const u8) Allocator.Error![]const u8 {
        const suf = text[h.suf_at .. h.suf_at + h.suf_len];
        return if (h.outer) std.fmt.allocPrint(a, "</tool_calls{s}>", .{suf}) else std.fmt.allocPrint(a, "</tool_call{s}>", .{suf});
    }
};

/// Advances the open Hy4 block on `text`, emitting incremental header/argument deltas into `out`.
pub fn feed(s: anytype, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!Next {
    const h = &s.hy4;
    const tags = Tags.init(text[h.suf_at .. h.suf_at + h.suf_len]);
    while (true) {
        const next = switch (h.stage) {
            .calls => stepCalls(s, text, &tags),
            .name => try stepName(s, text, &tags, out),
            .pairs => try stepPairs(s, text, &tags, out),
            .key => try stepKey(s, text, &tags, out),
            .value => try stepValue(s, text, &tags, out),
        };
        if (next != .more) return next;
    }
}

/// Flushes a value cut at `max_tokens` (`finish_reason == "length"`).
pub fn cut(s: anytype, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!void {
    const h = &s.hy4;
    if (h.stage != .value) return;
    const tags = Tags.init(text[h.suf_at .. h.suf_at + h.suf_len]);
    const end = text.len - closerHold(text[@min(h.from, text.len)..], &tags, h.outer);
    try sendVal(s, text, end, out);
    if (s.typed) try out.append(s.a, try s.args(try s.a.dupe(u8, s.buffer.items)));
    s.pos = end;
}

fn stepCalls(s: anytype, text: []const u8, tags: *const Tags) Next {
    const h = &s.hy4;
    const i = skipSpace(text, &h.at) orelse return .wait;
    const co = tagAt(text, i, tags.callOpen());
    const cc = tagAt(text, i, tags.callsClose());
    if (co == .whole) {
        h.from = i + tags.callOpen().len;
        h.at = h.from;
        h.stage = .name;
        return .more;
    }
    if (cc == .whole) {
        if (h.sent == 0) return .rejected;
        s.pos = i + tags.callsClose().len;
        s.searched = s.pos;
        s.state = .outside;
        return .outside;
    }
    if (co == .part or cc == .part) return .wait;
    return if (h.sent == 0) .prose else .rejected;
}

fn stepName(s: anytype, text: []const u8, tags: *const Tags, out: *std.ArrayList(Value)) Allocator.Error!Next {
    const h = &s.hy4;
    const ko = tags.keyOpen();
    const cc = tags.callClose();
    var at = h.at;
    while (std.mem.indexOfScalarPos(u8, text, at, '<')) |i| : (at = i + 1) {
        const t_ko = tagAt(text, i, ko);
        const t_cc = tagAt(text, i, cc);
        if (h.outer and tagAt(text, i, tags.callsClose()) == .whole) return .rejected;
        if (t_ko == .whole or t_cc == .whole) {
            const raw = reply_text.pyStrip(text[h.from..i]);
            if (raw.len == 0 or std.mem.indexOfAny(u8, raw, "<>\n\r") != null) return .rejected;
            const offered = s.known.get(try std.ascii.allocLowerString(s.a, raw)) orelse return .rejected;
            try s.begin(offered, out);
            h.sent += 1;
            if (t_cc == .whole) {
                try out.append(s.a, try s.args("{}"));
                const next_pos = i + cc.len;
                if (h.outer) {
                    h.at = next_pos;
                    h.stage = .calls;
                    return .more;
                }
                s.pos = next_pos;
                s.searched = next_pos;
                s.state = .outside;
                return .outside;
            }
            h.from = i + ko.len;
            h.at = h.from;
            h.stage = .key;
            return .more;
        }
        if (t_ko == .part or t_cc == .part or (h.outer and tagAt(text, i, tags.callsClose()) == .part)) {
            h.at = i;
            return .wait;
        }
        return .rejected;
    }
    h.at = text.len;
    if (reply_text.charCount(text[h.from..]) > 256) return .rejected;
    return .wait;
}

fn stepPairs(s: anytype, text: []const u8, tags: *const Tags, out: *std.ArrayList(Value)) Allocator.Error!Next {
    const h = &s.hy4;
    const i = skipSpace(text, &h.at) orelse return .wait;
    const ko = tagAt(text, i, tags.keyOpen());
    const cc = tagAt(text, i, tags.callClose());
    if (ko == .whole) {
        h.from = i + tags.keyOpen().len;
        h.at = h.from;
        h.stage = .key;
        return .more;
    }
    if (cc == .whole) {
        try out.append(s.a, try s.args(if (s.args_open) "}" else "{}"));
        const next_pos = i + tags.callClose().len;
        if (h.outer) {
            h.at = next_pos;
            h.stage = .calls;
            return .more;
        }
        s.pos = next_pos;
        s.searched = next_pos;
        s.state = .outside;
        return .outside;
    }
    return if (ko == .part or cc == .part) .wait else .rejected;
}

fn stepKey(s: anytype, text: []const u8, tags: *const Tags, out: *std.ArrayList(Value)) Allocator.Error!Next {
    const h = &s.hy4;
    const kc = tags.keyClose();
    const vo = tags.valOpen();
    const from = @max(h.from, h.at -| (kc.len + 1));
    const ke_opt = std.mem.indexOfPos(u8, text, from, kc);
    const limit = ke_opt orelse text.len;
    if (hasTag(text, h.from, from, limit, tags.callClose()) or (h.outer and hasTag(text, h.from, from, limit, tags.callsClose()))) return .rejected;
    const ke = ke_opt orelse {
        h.at = text.len;
        return .wait;
    };
    if (hasTag(text, h.from, h.from, ke, tags.keyOpen())) return .rejected;
    var after = ke + kc.len;
    const j = skipSpace(text, &after) orelse {
        h.at = ke;
        return .wait;
    };
    switch (tagAt(text, j, vo)) {
        .part => {
            h.at = ke;
            return .wait;
        },
        .no => return .rejected,
        .whole => {},
    }
    const key = text[h.from..ke];
    s.value_schema = try tool_hy4.schemaForKey(s.a, &s.schemas, s.current, key);
    s.typed = !(try tool_hy4.isOnlyStringType(s.a, s.value_schema));
    s.buffer.clearRetainingCapacity();
    const quoted = try json.quote(s.a, key, .{ .ascii = false });
    try out.append(s.a, try s.args(try std.mem.concat(s.a, u8, &.{ if (s.args_open) "," else "{", quoted, if (s.typed) ":" else ":\"" })));
    s.args_open = true;
    h.from = j + vo.len;
    h.at = h.from;
    h.val_sent = h.from;
    h.stage = .value;
    return .more;
}

fn stepValue(s: anytype, text: []const u8, tags: *const Tags, out: *std.ArrayList(Value)) Allocator.Error!Next {
    const h = &s.hy4;
    const vc = tags.valClose();
    const max_len = @max(vc.len, @max(tags.callClose().len, tags.callsClose().len));
    const from = @max(h.from, h.at -| (max_len - 1));
    const ve_opt = std.mem.indexOfPos(u8, text, from, vc);
    const limit = ve_opt orelse text.len;
    if (hasTag(text, h.from, from, limit, tags.callClose()) or (h.outer and hasTag(text, h.from, from, limit, tags.callsClose())) or hasTag(text, h.from, from, limit, tags.keyOpen())) return .rejected;
    const ve = ve_opt orelse {
        try sendVal(s, text, text.len - closerHold(text[@min(h.from, text.len)..], tags, h.outer), out);
        h.at = text.len;
        return .wait;
    };
    try sendVal(s, text, ve, out);
    if (s.typed) {
        const v = try tool_hy4.parseValue(s.a, s.buffer.items, s.value_schema);
        try out.append(s.a, try s.args(try json.stringify(s.a, v, .{ .ascii = false, .compact = true })));
    } else try out.append(s.a, try s.args("\""));
    h.at = ve + vc.len;
    h.stage = .pairs;
    return .more;
}

fn sendVal(s: anytype, text: []const u8, end: usize, out: *std.ArrayList(Value)) Allocator.Error!void {
    const h = &s.hy4;
    if (end <= h.val_sent) return;
    if (s.typed) try s.buffer.appendSlice(s.a, text[h.val_sent..end]) else try out.append(s.a, try s.args(try s.esc(text[h.val_sent..end])));
    h.val_sent = end;
}

fn skipSpace(text: []const u8, at: *usize) ?usize {
    var i = at.*;
    defer at.* = i;
    while (i < text.len) {
        const n = reply_text.spaceAt(text, i);
        if (n > 0) {
            i += n;
            continue;
        }
        const full = if (text[i] < 0x80) 1 else std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
        return if (i + full > text.len) null else i;
    }
    return null;
}

const Match = enum { whole, part, no };
fn tagAt(text: []const u8, i: usize, tag: []const u8) Match {
    const rest = text[i..];
    if (std.mem.startsWith(u8, rest, tag)) return .whole;
    return if (rest.len < tag.len and std.mem.startsWith(u8, tag, rest)) .part else .no;
}

fn hasTag(text: []const u8, base: usize, at: usize, end: usize, tag: []const u8) bool {
    const lo = @max(base, at -| (tag.len - 1));
    return lo < end and std.mem.indexOf(u8, text[lo..end], tag) != null;
}

fn closerHold(written: []const u8, tags: *const Tags, outer: bool) usize {
    var hold = @max(reply_text.partialTag(written, tags.valClose()), reply_text.partialTag(written, tags.callClose()));
    if (outer) hold = @max(hold, reply_text.partialTag(written, tags.callsClose()));
    return hold;
}
