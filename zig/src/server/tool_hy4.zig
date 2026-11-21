//! Hy4-flash's HYTK-suffixed thinking switches, tool-call parsing, typed schema coercion and envelope helpers.
const std = @import("std");
const json = @import("json");
const fields = @import("fields.zig");
const reply_code = @import("reply_code.zig");
const reply_text = @import("reply_text.zig");
const tool_dsml = @import("tool_dsml.zig");
const tool_params = @import("tool_params.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;
const Call = tool_dsml.Call;
const Error = tool_dsml.Error;

pub const max_suffix = 64;

/// Whether `suf` is an admissible HYTK suffix (`""` or `:[^\s<>/'"｜]{1,63}`).
pub fn validSuffix(suf: []const u8) bool {
    if (suf.len == 0) return true;
    return suf[0] == ':' and suf.len >= 2 and suf.len <= max_suffix and validSuffixBody(suf[1..]);
}

/// Whether `body` (the bytes after `:`) can belong to an HYTK suffix.
pub fn validSuffixBody(body: []const u8) bool {
    if (body.len > max_suffix - 1) return false;
    for (body) |b| {
        if (reply_text.isSpaceByte(b) or b == '<' or b == '>' or b == '/' or b == '"' or b == '\'') return false;
    }
    return std.mem.indexOf(u8, body, "\u{ff5c}") == null;
}

/// Stack-formatted HYTK tag set for one suffix (`suf.len <= max_suffix`), avoiding per-token heap allocations.
pub const Tags = struct {
    b: [7][96]u8 = undefined,
    n: [7]u8 = undefined,

    pub fn init(suf: []const u8) Tags {
        var t: Tags = .{};
        const heads = [7][]const u8{ "</tool_calls", "<tool_call", "</tool_call", "<arg_key", "</arg_key", "<arg_value", "</arg_value" };
        for (heads, 0..) |h, i| {
            @memcpy(t.b[i][0..h.len], h);
            @memcpy(t.b[i][h.len .. h.len + suf.len], suf);
            t.b[i][h.len + suf.len] = '>';
            t.n[i] = @intCast(h.len + suf.len + 1);
        }
        return t;
    }

    pub fn callsClose(t: *const Tags) []const u8 {
        return t.b[0][0..t.n[0]];
    }
    pub fn callOpen(t: *const Tags) []const u8 {
        return t.b[1][0..t.n[1]];
    }
    pub fn callClose(t: *const Tags) []const u8 {
        return t.b[2][0..t.n[2]];
    }
    pub fn keyOpen(t: *const Tags) []const u8 {
        return t.b[3][0..t.n[3]];
    }
    pub fn keyClose(t: *const Tags) []const u8 {
        return t.b[4][0..t.n[4]];
    }
    pub fn valOpen(t: *const Tags) []const u8 {
        return t.b[5][0..t.n[5]];
    }
    pub fn valClose(t: *const Tags) []const u8 {
        return t.b[6][0..t.n[6]];
    }
};

/// Whether `source` is Hy4's chat template (`reasoning_mode` with `no_think`).
pub fn hasNoThink(source: []const u8) bool {
    return std.mem.indexOf(u8, source, "reasoning_mode") != null and std.mem.indexOf(u8, source, "no_think") != null;
}

/// Maps server/request thinking flags onto Hy4's `no_think`/`high`/`max` `reasoning_effort`; pass-through elsewhere.
pub fn templateEffort(source: []const u8, thinking: bool, effort: ?[]const u8) ?[]const u8 {
    if (!hasNoThink(source)) return effort;
    if (!thinking) return "no_think";
    const e = effort orelse return null;
    if (std.mem.eql(u8, e, "none") or std.mem.eql(u8, e, "no_think")) return "no_think";
    if (std.mem.eql(u8, e, "max") or std.mem.eql(u8, e, "xhigh")) return "max";
    return "high";
}

/// Extracts `HYTK` from a Hy4 chat template (`set HYTK = ':...'` or literal `</think:...>`), or null.
pub fn templateSuffix(source: []const u8) ?[]const u8 {
    if (std.mem.indexOf(u8, source, "HYTK")) |at| {
        const window = source[at..@min(source.len, at + 128)];
        if (std.mem.indexOfScalar(u8, window, '=')) |eq| {
            const rhs = std.mem.trimStart(u8, window[eq + 1 ..], " \t");
            if (rhs.len >= 2 and (rhs[0] == '\'' or rhs[0] == '"')) {
                if (std.mem.indexOfScalarPos(u8, rhs, 1, rhs[0])) |end| {
                    const suf = rhs[1..end];
                    if (validSuffix(suf)) return suf;
                }
            }
        }
    }
    for ([_][]const u8{ "</think:", "<tool_calls:", "<tool_call:" }) |prefix| {
        if (std.mem.indexOf(u8, source, prefix)) |p| {
            const colon = p + prefix.len - 1;
            if (std.mem.indexOfScalarPos(u8, source, colon, '>')) |gt| {
                const suf = source[colon..gt];
                if (validSuffix(suf)) return suf;
            }
        }
    }
    return null;
}

/// Builds Hy4 `reply_text.Markers` (and its suffixed call `Openers`) when the template or tokenizer declares HYTK.
pub fn markers(a: Allocator, source: []const u8, text: anytype) Allocator.Error!?reply_text.Markers {
    const from_tpl = templateSuffix(source);
    const suf: []const u8 = if (from_tpl) |s| s else if (text.tokenId("</think:opensource>") != null) ":opensource" else return null;
    if (suf.len == 0 and !hasNoThink(source)) return null;
    const close = if (suf.len == 0) "</think>" else try std.fmt.allocPrint(a, "</think{s}>", .{suf});
    const self_open: ?[]const u8 = if (suf.len == 0) null else try std.fmt.allocPrint(a, "<think{s}>", .{suf});
    return .{ .open = "", .close = close, .self_open = self_open, .openers = try openersForSuffix(a, suf) };
}

/// Configures `srv.markers` and `srv.thinking_switch` for MiniMax-M3, Hy4 (`HYTK`), and standard templates.
pub fn initServerMarkers(srv: anytype, a: Allocator, source: []const u8, text: anytype) Allocator.Error!void {
    if (text.tokenId(reply_text.channel_markers.close) != null) srv.markers = reply_text.channel_markers;
    const minimax = text.tokenId(reply_text.mm_markers.close) != null;
    if (minimax) srv.markers = reply_text.mm_markers else if (try markers(a, source, text)) |m| srv.markers = m;
    srv.thinking_switch = minimax or hasNoThink(source) or std.mem.indexOf(u8, source, "enable_thinking") != null or std.mem.indexOf(u8, source, "thinking_mode") != null;
}

/// `reply_text.Openers` extended with `<tool_calls{suf}>` (lead `<tool_call{suf}>`) and `<tool_call{suf}>`.
pub fn openersForSuffix(a: Allocator, suf: []const u8) Allocator.Error!reply_text.Openers {
    const extra: usize = if (suf.len > 0) 2 else 1;
    const call_list = try a.alloc([2][]const u8, reply_text.calls.len + extra);
    const lead_list = try a.alloc([]const []const u8, reply_text.calls.len + extra);
    @memcpy(call_list[0..reply_text.calls.len], &reply_text.calls);
    for (&reply_text.calls, 0..) |_, i| lead_list[i] = reply_text.shared_openers.leads[i];
    const co = try std.fmt.allocPrint(a, "<tool_calls{s}>", .{suf});
    const cc = try std.fmt.allocPrint(a, "</tool_calls{s}>", .{suf});
    const inner_open = try std.fmt.allocPrint(a, "<tool_call{s}>", .{suf});
    const one_lead = try a.alloc([]const u8, 1);
    one_lead[0] = inner_open;
    call_list[reply_text.calls.len] = .{ co, cc };
    lead_list[reply_text.calls.len] = one_lead;
    if (suf.len > 0) {
        call_list[reply_text.calls.len + 1] = .{ inner_open, try std.fmt.allocPrint(a, "</tool_call{s}>", .{suf}) };
        lead_list[reply_text.calls.len + 1] = &.{};
    }
    return .{ .calls = call_list, .leads = lead_list };
}

/// Returns `shared_openers` unless `text` contains `<tool_calls...` or `<tool_call:...` (whole or trailing partial).
pub fn openersIn(a: Allocator, text: []const u8) Allocator.Error!reply_text.Openers {
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, text, pos, "<tool_call")) |at| : (pos = at + 10) {
        const rest = text[at..];
        if (openedBy(rest)) |op| switch (op) {
            .open => |h| return openersForSuffix(a, rest[h.suf_off .. h.suf_off + h.suf_len]),
            .wait => if (at + rest.len == text.len) {
                const call_list = try a.alloc([2][]const u8, reply_text.calls.len + 1);
                const lead_list = try a.alloc([]const []const u8, reply_text.calls.len + 1);
                @memcpy(call_list[0..reply_text.calls.len], &reply_text.calls);
                for (&reply_text.calls, 0..) |_, i| lead_list[i] = reply_text.shared_openers.leads[i];
                call_list[reply_text.calls.len] = .{ try std.fmt.allocPrint(a, "{s}>", .{rest}), "</tool_calls>" };
                lead_list[reply_text.calls.len] = &.{};
                return .{ .calls = call_list, .leads = lead_list };
            },
        };
    }
    return reply_text.shared_openers;
}

pub const Head = struct { outer: bool, suf_off: usize, suf_len: usize, open_len: usize };
pub const Open = union(enum) { wait, open: Head };

/// Parses `<tool_calls{suf}>` or `<tool_call:suf>` at the start of `rest`, or `.wait` if cut before `>`.
pub fn openedBy(rest: []const u8) ?Open {
    for ([_]struct { p: []const u8, outer: bool }{ .{ .p = "<tool_calls", .outer = true }, .{ .p = "<tool_call", .outer = false } }) |k| {
        if (!std.mem.startsWith(u8, rest, k.p)) continue;
        const after = rest[k.p.len..];
        if (after.len == 0) return if (k.outer) .wait else null;
        if (after[0] == '>') return if (k.outer) .{ .open = .{ .outer = true, .suf_off = k.p.len, .suf_len = 0, .open_len = k.p.len + 1 } } else null;
        if (after[0] != ':') continue;
        const gt = std.mem.indexOfScalar(u8, after, '>') orelse return if (validSuffixBody(after[1..])) .wait else null;
        const suf = after[0..gt];
        if (!validSuffix(suf)) return null;
        return .{ .open = .{ .outer = k.outer, .suf_off = k.p.len, .suf_len = suf.len, .open_len = k.p.len + gt + 1 } };
    }
    return null;
}

/// Matches a Hy4 `<tool_calls{suf}>` or `<tool_call:suf>` envelope at `s` for `tool_parse.envelopes`.
pub fn envelopeAt(text: []const u8, s: usize) ?struct { start: usize, end: usize, payload: []const u8, complete: bool } {
    const h = switch (openedBy(text[s..]) orelse return null) {
        .open => |v| v,
        .wait => return null,
    };
    const tags = Tags.init(text[s + h.suf_off .. s + h.suf_off + h.suf_len]);
    const body_start = s + h.open_len;
    if (h.outer) {
        const trimmed = reply_text.pyLstrip(text[body_start..]);
        if (trimmed.len == 0) return .{ .start = s, .end = text.len, .payload = text[s..], .complete = false };
        if (!std.mem.startsWith(u8, trimmed, tags.callOpen()) and !std.mem.startsWith(u8, trimmed, tags.callsClose())) return null;
        const close = tags.callsClose();
        if (std.mem.indexOfPos(u8, text, body_start, close)) |e| {
            return .{ .start = s, .end = e + close.len, .payload = text[s .. e + close.len], .complete = true };
        }
        return .{ .start = s, .end = text.len, .payload = text[s..], .complete = false };
    }
    const close = tags.callClose();
    const e = std.mem.indexOfPos(u8, text, body_start, close) orelse return null;
    return .{ .start = s, .end = e + close.len, .payload = text[s .. e + close.len], .complete = true };
}

pub const Block = struct { inner: []const u8, suf: []const u8, outer: bool };

/// Unwraps a complete Hy4 `<tool_calls{suf}>...</tool_calls{suf}>` or `<tool_call:suf>...</tool_call:suf>` payload.
pub fn whole(payload: []const u8) ?Block {
    const h = switch (openedBy(payload) orelse return null) {
        .open => |v| v,
        .wait => return null,
    };
    const suf = payload[h.suf_off .. h.suf_off + h.suf_len];
    const tags = Tags.init(suf);
    const close = if (h.outer) tags.callsClose() else tags.callClose();
    if (payload.len < h.open_len + close.len or !std.mem.endsWith(u8, payload, close)) return null;
    return .{ .inner = payload[h.open_len .. payload.len - close.len], .suf = suf, .outer = h.outer };
}

/// Extracts all calls in `b` strictly, coercing `<arg_value>` text against `schemas` as `hy_v4_tool_parser.py` does.
pub fn calls(a: Allocator, b: Block, schemas: ?*const tool_params.Schemas) Error![]Call {
    const tags = Tags.init(b.suf);
    if (!b.outer) {
        const list = try a.alloc(Call, 1);
        list[0] = try parseOneCall(a, b.inner, &tags, schemas);
        return list;
    }
    var out: std.ArrayList(Call) = .empty;
    var pos: usize = 0;
    const co = tags.callOpen();
    const cc = tags.callClose();
    while (true) {
        const rest = reply_text.pyLstrip(b.inner[pos..]);
        pos = b.inner.len - rest.len;
        if (pos == b.inner.len) break;
        if (!std.mem.startsWith(u8, rest, co)) return error.Invalid;
        const body_start = pos + co.len;
        const end = std.mem.indexOfPos(u8, b.inner, body_start, cc) orelse return error.Invalid;
        try out.append(a, try parseOneCall(a, b.inner[body_start..end], &tags, schemas));
        pos = end + cc.len;
    }
    if (out.items.len == 0) return error.Invalid;
    return out.items;
}

fn parseOneCall(a: Allocator, body: []const u8, tags: *const Tags, schemas: ?*const tool_params.Schemas) Error!Call {
    const ko = tags.keyOpen();
    const kc = tags.keyClose();
    const vo = tags.valOpen();
    const vc = tags.valClose();
    const arg_start = std.mem.indexOf(u8, body, ko);
    const name = reply_text.pyStrip(if (arg_start) |s| body[0..s] else body);
    if (name.len == 0 or std.mem.indexOfAny(u8, name, "<>\n\r") != null) return error.Invalid;
    const arguments = try json.newObject(a);
    const first = arg_start orelse {
        for ([_][]const u8{ kc, vo, vc, tags.callOpen() }) |bad| if (std.mem.indexOf(u8, body, bad) != null) return error.Invalid;
        return .{ .name = name, .arguments = .{ .object = arguments } };
    };
    var pos = first;
    while (true) {
        const rest = reply_text.pyLstrip(body[pos..]);
        pos = body.len - rest.len;
        if (pos == body.len) break;
        if (!std.mem.startsWith(u8, rest, ko)) return error.Invalid;
        const ks = pos + ko.len;
        const ke = std.mem.indexOfPos(u8, body, ks, kc) orelse return error.Invalid;
        if (std.mem.indexOfPos(u8, body, ks, ko)) |nk| if (nk < ke) return error.Invalid;
        const after_k = reply_text.pyLstrip(body[ke + kc.len ..]);
        if (!std.mem.startsWith(u8, after_k, vo)) return error.Invalid;
        const vs = body.len - after_k.len + vo.len;
        const ve = std.mem.indexOfPos(u8, body, vs, vc) orelse return error.Invalid;
        if (std.mem.indexOfPos(u8, body, vs, ko)) |nk| if (nk < ve) return error.Invalid;
        const key = body[ks..ke];
        const schema = if (schemas) |s| try schemaForKey(a, s, name, key) else try tool_params.emptySchema(a);
        try arguments.put(a, key, try parseValue(a, body[vs..ve], schema));
        pos = ve + vc.len;
    }
    return .{ .name = name, .arguments = .{ .object = arguments } };
}

/// Looks up exact `key` first, then `pyStrip(key)`, matching `hy_v4_tool_parser.py` property resolution.
pub fn schemaForKey(a: Allocator, s: *const tool_params.Schemas, tool_name: []const u8, key: []const u8) Allocator.Error!tool_params.Schema {
    const lower = try std.ascii.allocLowerString(a, tool_name);
    const tool = s.get(lower) orelse return tool_params.emptySchema(a);
    if (tool.properties.get(key)) |v| return .{ .value = v, .root = tool.root };
    if (tool.properties.get(reply_text.pyStrip(key))) |v| return .{ .value = v, .root = tool.root };
    return tool_params.emptySchema(a);
}

const Kind = enum { string, boolean, integer, number, array, object, null, other };
const KindSet = std.EnumSet(Kind);
const SchemaTypes = struct { types: KindSet = .empty, allows_null: bool = false };

fn normalizeType(raw: []const u8) Kind {
    const aliases = [_]struct { k: []const u8, v: Kind }{
        .{ .k = "string", .v = .string },   .{ .k = "str", .v = .string },    .{ .k = "text", .v = .string },
        .{ .k = "varchar", .v = .string },  .{ .k = "char", .v = .string },   .{ .k = "enum", .v = .string },
        .{ .k = "boolean", .v = .boolean }, .{ .k = "bool", .v = .boolean },  .{ .k = "binary", .v = .boolean },
        .{ .k = "integer", .v = .integer }, .{ .k = "int", .v = .integer },   .{ .k = "number", .v = .number },
        .{ .k = "float", .v = .number },    .{ .k = "double", .v = .number }, .{ .k = "array", .v = .array },
        .{ .k = "list", .v = .array },      .{ .k = "object", .v = .object }, .{ .k = "dict", .v = .object },
        .{ .k = "map", .v = .object },      .{ .k = "null", .v = .null },
    };
    for (aliases) |e| if (std.mem.eql(u8, raw, e.k)) return e.v;
    for ([_][]const u8{ "int", "uint", "long", "short", "unsigned" }) |p| if (std.ascii.startsWithIgnoreCase(raw, p)) return .integer;
    for ([_][]const u8{ "num", "float" }) |p| if (std.ascii.startsWithIgnoreCase(raw, p)) return .number;
    return .other;
}

fn isRefOnly(v: Value) bool {
    if (v != .object) return false;
    const r = v.get("$ref") orelse return false;
    return r == .string and v.get("type") == null and v.get("anyOf") == null and v.get("oneOf") == null;
}

fn findAnchor(node: Value, anchor: []const u8, depth: usize) ?Value {
    if (depth > 8) return null;
    switch (node) {
        .object => |o| {
            if (node.strField("$anchor")) |a| if (std.mem.eql(u8, a, anchor)) return node;
            for (o.values()) |child| if (findAnchor(child, anchor, depth + 1)) |hit| return hit;
        },
        .array => |items| for (items) |child| if (findAnchor(child, anchor, depth + 1)) |hit| return hit,
        else => {},
    }
    return null;
}

fn resolveLocalRef(a: Allocator, root: Value, ref: []const u8) Allocator.Error!?Value {
    if (ref.len == 0 or ref[0] != '#') return null;
    const frag = ref[1..];
    if (frag.len == 0) return if (root == .object) root else null;
    if (frag[0] != '/') return findAnchor(root, frag, 0);
    var cur = root;
    var it = std.mem.splitScalar(u8, frag[1..], '/');
    while (it.next()) |tok| {
        const unescaped = try unescapePtr(a, tok) orelse return null;
        switch (cur) {
            .object => cur = cur.object.get(unescaped) orelse return null,
            .array => {
                const idx = std.fmt.parseInt(usize, unescaped, 10) catch return null;
                if (idx >= cur.array.len) return null;
                cur = cur.array[idx];
            },
            else => return null,
        }
    }
    return if (cur == .object) cur else null;
}

fn unescapePtr(a: Allocator, tok: []const u8) Allocator.Error!?[]const u8 {
    if (std.mem.indexOfAny(u8, tok, "%~") == null) return tok;
    var pct: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < tok.len) : (i += 1) {
        if (tok[i] == '%') {
            if (i + 2 >= tok.len) return null;
            const hi = std.fmt.charToDigit(tok[i + 1], 16) catch return null;
            const lo = std.fmt.charToDigit(tok[i + 2], 16) catch return null;
            try pct.append(a, hi * 16 + lo);
            i += 2;
        } else try pct.append(a, tok[i]);
    }
    var out: std.ArrayList(u8) = .empty;
    var j: usize = 0;
    while (j < pct.items.len) : (j += 1) {
        if (pct.items[j] != '~') {
            try out.append(a, pct.items[j]);
            continue;
        }
        if (j + 1 >= pct.items.len) return null;
        j += 1;
        try out.append(a, if (pct.items[j] == '1') '/' else if (pct.items[j] == '0') '~' else return null);
    }
    return out.items;
}

fn collectOptions(a: Allocator, node: Value, root: Value, seen: []const []const u8, hops: usize, out: *SchemaTypes) Allocator.Error!void {
    if (node != .object) {
        out.types.insert(.string);
        return;
    }
    if (isRefOnly(node)) {
        const ref = node.get("$ref").?.string;
        for (seen) |prev| if (std.mem.eql(u8, prev, ref)) {
            out.types.insert(.string);
            return;
        };
        if (hops >= 3) {
            out.types.insert(.string);
            return;
        }
        const target = (try resolveLocalRef(a, root, ref)) orelse {
            out.types.insert(.string);
            return;
        };
        const next_seen = try std.mem.concat(a, []const u8, &.{ seen, &.{ref} });
        if (target.get("type") != null or target.get("anyOf") != null or target.get("oneOf") != null or isRefOnly(target)) {
            return collectOptions(a, target, root, next_seen, hops + 1, out);
        }
        out.types.insert(.string);
        return;
    }
    if (node.get("type")) |tv| {
        switch (tv) {
            .string => |s| addKind(normalizeType(s), out),
            .array => |arr| for (arr) |item| addKind(if (item == .string) normalizeType(item.string) else .string, out),
            else => out.types.insert(.string),
        }
        return;
    }
    for ([_][]const u8{ "anyOf", "oneOf" }) |uk| if (node.get(uk)) |branches| {
        if (branches == .array) {
            for (branches.array) |br| try collectOptions(a, br, root, seen, hops, out);
            return;
        }
    };
    out.types.insert(.string);
}

fn addKind(k: Kind, out: *SchemaTypes) void {
    if (k == .null) out.allows_null = true else out.types.insert(k);
}

/// True only for non-nullable pure string schemas (`_is_only_string_type` in `hy_v4_tool_parser.py`).
pub fn isOnlyStringType(a: Allocator, schema: tool_params.Schema) Allocator.Error!bool {
    var st: SchemaTypes = .{};
    try collectOptions(a, schema.value, schema.root, &.{}, 0, &st);
    return !st.allows_null and st.types.count() == 1 and st.types.contains(.string);
}

/// Coerces raw `<arg_value>` text using `hy_v4_tool_parser.py`'s `_parse_value` chain.
pub fn parseValue(a: Allocator, raw: []const u8, schema: tool_params.Schema) Allocator.Error!Value {
    var st: SchemaTypes = .{};
    try collectOptions(a, schema.value, schema.root, &.{}, 0, &st);
    if (st.allows_null and std.mem.eql(u8, raw, "null")) return .null;
    if (st.types.contains(.boolean)) {
        if (std.ascii.eqlIgnoreCase(raw, "true")) return .{ .bool = true };
        if (std.ascii.eqlIgnoreCase(raw, "false")) return .{ .bool = false };
    }
    if (st.types.contains(.integer)) {
        if (try fields.pyIntText(a, raw)) |it| return .{ .int = it };
    }
    if (st.types.contains(.number)) {
        if (std.mem.indexOfAny(u8, raw, ".eE") != null) {
            if (fields.pyFloat(raw)) |f| if (std.math.isFinite(f)) return .{ .float = f };
        } else if (try fields.pyIntText(a, raw)) |it| return .{ .int = it };
    }
    if (st.types.contains(.array) or st.types.contains(.object) or st.types.contains(.other)) {
        switch (try json.parseText(a, raw)) {
            .ok => |v| if (tool_params.finite(v)) return v,
            .err => {},
        }
    }
    if (st.types.contains(.string)) return .{ .string = raw };
    switch (try json.parseText(a, raw)) {
        .ok => |v| if (tool_params.finite(v)) return v,
        .err => {},
    }
    if (try tool_params.pythonLiteral(a, raw)) |v| if (tool_params.finite(v)) return v;
    return .{ .string = raw };
}

/// Closes an open Hy4 tool envelope when the model ended on its EOS token before the closing tag(s).
pub fn closeCall(a: Allocator, text: []const u8, tools: []const Value) Allocator.Error!?[]const u8 {
    var pos: usize = 0;
    var code: reply_code.Scanner = .{};
    while (std.mem.indexOfScalarPos(u8, text, pos, '<')) |s| {
        if (code.check(text, s, true) == .skip) {
            pos = code.skip_end;
            continue;
        }
        const h = switch (openedBy(text[s..]) orelse {
            pos = s + 1;
            continue;
        }) {
            .open => |v| v,
            .wait => break,
        };
        const tags = Tags.init(text[s + h.suf_off .. s + h.suf_off + h.suf_len]);
        const body_start = s + h.open_len;
        const close = if (h.outer) tags.callsClose() else tags.callClose();
        if (h.outer) {
            const trimmed = reply_text.pyLstrip(text[body_start..]);
            if (trimmed.len > 0 and !std.mem.startsWith(u8, trimmed, tags.callOpen())) {
                pos = body_start;
                continue;
            }
        }
        if (std.mem.indexOfPos(u8, text, body_start, close)) |e| {
            pos = e + close.len;
            code.jumpTo(pos);
            continue;
        }
        const tail = text[body_start..];
        var add: std.ArrayList(u8) = .empty;
        if (unclosedLast(tail, tags.valOpen(), tags.valClose())) try add.appendSlice(a, tags.valClose());
        if (!h.outer or unclosedLast(tail, tags.callOpen(), tags.callClose())) try add.appendSlice(a, tags.callClose());
        if (h.outer) try add.appendSlice(a, tags.callsClose());
        const closed = try std.mem.concat(a, u8, &.{ text[s..], add.items });
        const parsed = try @import("tool_parse.zig").parse(a, closed, tools, null);
        return if (parsed.calls != null and parsed.rejected_envelopes == 0) add.items else "";
    }
    return null;
}

fn unclosedLast(text: []const u8, open: []const u8, close: []const u8) bool {
    const o = std.mem.lastIndexOf(u8, text, open) orelse return false;
    const c = std.mem.lastIndexOf(u8, text, close) orelse return true;
    return c < o;
}

/// Reconstructs calls cut at `max_tokens` when `Streamer.open()` was false (e.g. closed inner call, open outer block).
pub fn cutCalls(a: Allocator, text: []const u8, tools: []const Value) Allocator.Error!?[]Value {
    const suffix = (try closeCall(a, text, tools)) orelse return null;
    if (suffix.len == 0) return null;
    const closed = try std.mem.concat(a, u8, &.{ text, suffix });
    return (try @import("tool_parse.zig").parse(a, closed, tools, null)).calls;
}

/// Whether `<NAME>` is a Hy4 envelope opener (`tool_calls`, `tool_calls:suf`, `tool_call:suf`) for `tool_policy.zig`.
pub fn policyTag(name: []const u8) bool {
    if (std.ascii.eqlIgnoreCase(name, "tool_calls")) return true;
    for ([_][]const u8{ "tool_calls:", "tool_call:" }) |p| {
        if (std.ascii.startsWithIgnoreCase(name, p)) return validSuffix(name[p.len - 1 ..]);
    }
    return false;
}

/// Whether partial `<NAME` (no `>` yet) could still become a Hy4 envelope opener for `tool_policy.zig`.
pub fn policyPartial(name: []const u8) bool {
    if (std.ascii.startsWithIgnoreCase("tool_calls:", name)) return true;
    for ([_][]const u8{ "tool_calls:", "tool_call:" }) |p| {
        if (std.ascii.startsWithIgnoreCase(name, p)) return validSuffixBody(name[p.len..]);
    }
    return false;
}
