//! End-to-end and byte-boundary tests for Hy4-flash HYTK reasoning switches, tool-call parsing and SSE streaming.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const HfText = @import("hf_text.zig").HfText;
const http_conn = @import("http_conn.zig");
const openai = @import("openai.zig");
const reply_text = @import("reply_text.zig");
const Server = @import("server.zig").Server;
const tool_parse = @import("tool_parse.zig");
const tool_stream = @import("tool_stream.zig");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const Value = json.Value;

const corpus_tools_json =
    \\[
    \\  {"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{
    \\    "city":{"type":"str"},"days":{"type":"int"},"metric":{"type":"bool"},
    \\    "threshold":{"type":"float"},"tags":{"type":"list"},"meta":{"type":"dict"}
    \\  }}}},
    \\  {"type":"function","function":{"name":"search_docs","parameters":{
    \\    "$defs":{"Limit":{"$anchor":"lim_anchor","type":"uint32"},"Mid":{"$ref":"#/$defs/Limit"}},
    \\    "type":"object","properties":{
    \\      "query":{"type":"string"},
    \\      "note":{"type":["string","null"]},
    \\      "limit":{"anyOf":[{"$ref":"#/$defs/Mid"},{"type":"null"}]},
    \\      "cursor":{"$ref":"#lim_anchor"},
    \\      "weight":{"type":"number"}
    \\    }
    \\  }}},
    \\  {"type":"function","function":{"name":"get_clock","parameters":{"type":"object","properties":{}}}}
    \\]
;

fn toolsOf(a: Allocator) ![]const Value {
    return (try json.parse(a, corpus_tools_json)).ok.array;
}

fn joinedStreamArgs(a: Allocator, deltas: []const Value, call_idx: i64) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (deltas) |d| for (d.get("tool_calls").?.array) |one| {
        if (one.get("index").?.int64().? == call_idx) {
            try out.appendSlice(a, one.get("function").?.get("arguments").?.string);
        }
    };
    return out.items;
}

fn countHeaders(deltas: []const Value) usize {
    var n: usize = 0;
    for (deltas) |d| for (d.get("tool_calls").?.array) |one| {
        n += @intFromBool(one.get("id") != null);
    };
    return n;
}

test "hy4 markup: parity corpus matches vLLM HYV4ToolExtractor across single, parallel, nested, aliases, refs and bare calls" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try toolsOf(a);

    const Case = struct {
        reply: []const u8,
        want_content: []const u8,
        want_names: []const []const u8,
        want_args: []const []const u8,
        want_unoffered: usize = 0,
        want_malformed: usize = 0,
    };
    const cases = [_]Case{
        .{
            .reply = "Checking weather.\n<tool_calls:opensource><tool_call:opensource>get_weather" ++
                "<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Tokyo</arg_value:opensource>" ++
                "<arg_key:opensource>days</arg_key:opensource><arg_value:opensource>007</arg_value:opensource>" ++
                "<arg_key:opensource>metric</arg_key:opensource><arg_value:opensource>True</arg_value:opensource>" ++
                "<arg_key:opensource>threshold</arg_key:opensource><arg_value:opensource>1.5e2</arg_value:opensource>" ++
                "<arg_key:opensource>tags</arg_key:opensource><arg_value:opensource>[\"rain\",\"wind\"]</arg_value:opensource>" ++
                "<arg_key:opensource>meta</arg_key:opensource><arg_value:opensource>{\" nested \":[1,{\"ok\":false}]}</arg_value:opensource>" ++
                "</tool_call:opensource></tool_calls:opensource>",
            .want_content = "Checking weather.",
            .want_names = &.{"get_weather"},
            .want_args = &.{"{\"city\":\"Tokyo\",\"days\":7,\"metric\":true,\"threshold\":150.0,\"tags\":[\"rain\",\"wind\"],\"meta\":{\" nested \":[1,{\"ok\":false}]}}"},
        },
        .{
            .reply = "<tool_calls:opensource>" ++
                "<tool_call:opensource>search_docs" ++
                "<arg_key:opensource>query</arg_key:opensource><arg_value:opensource>a < b & c > d</arg_value:opensource>" ++
                "<arg_key:opensource>note</arg_key:opensource><arg_value:opensource>null</arg_value:opensource>" ++
                "<arg_key:opensource>limit</arg_key:opensource><arg_value:opensource>25</arg_value:opensource>" ++
                "<arg_key:opensource>cursor</arg_key:opensource><arg_value:opensource>4</arg_value:opensource>" ++
                "<arg_key:opensource>weight</arg_key:opensource><arg_value:opensource>5</arg_value:opensource>" ++
                "</tool_call:opensource>" ++
                "<tool_call:opensource>get_clock</tool_call:opensource>" ++
                "</tool_calls:opensource>",
            .want_content = "",
            .want_names = &.{ "search_docs", "get_clock" },
            .want_args = &.{ "{\"query\":\"a < b & c > d\",\"note\":null,\"limit\":25,\"cursor\":4,\"weight\":5}", "{}" },
        },
        .{
            .reply = "Custom suffix: <tool_calls:6124c78e><tool_call:6124c78e>get_weather" ++
                "<arg_key:6124c78e>city</arg_key:6124c78e><arg_value:6124c78e>null</arg_value:6124c78e>" ++
                "<arg_key:6124c78e>tags</arg_key:6124c78e><arg_value:6124c78e>['py', 'tuple']</arg_value:6124c78e>" ++
                "</tool_call:6124c78e></tool_calls:6124c78e>",
            .want_content = "Custom suffix:",
            .want_names = &.{"get_weather"},
            .want_args = &.{"{\"city\":\"null\",\"tags\":[\"py\",\"tuple\"]}"},
        },
        .{
            .reply = "Bare call: <tool_call:opensource>search_docs<arg_key:opensource>query</arg_key:opensource>" ++
                "<arg_value:opensource>mla</arg_value:opensource><arg_key:opensource>note</arg_key:opensource>" ++
                "<arg_value:opensource>kept</arg_value:opensource></tool_call:opensource>",
            .want_content = "Bare call:",
            .want_names = &.{"search_docs"},
            .want_args = &.{"{\"query\":\"mla\",\"note\":\"kept\"}"},
        },
        .{
            .reply = "<tool_calls><tool_call>get_clock</tool_call></tool_calls>",
            .want_content = "",
            .want_names = &.{"get_clock"},
            .want_args = &.{"{}"},
        },
        .{
            .reply = "Hy4 templates open blocks with <tool_calls:opensource> in system instructions.",
            .want_content = "Hy4 templates open blocks with <tool_calls:opensource> in system instructions.",
            .want_names = &.{},
            .want_args = &.{},
        },
        .{
            .reply = "Quoted: `<tool_calls:opensource><tool_call:opensource>get_clock</tool_call:opensource></tool_calls:opensource>`",
            .want_content = "Quoted: `<tool_calls:opensource><tool_call:opensource>get_clock</tool_call:opensource></tool_calls:opensource>`",
            .want_names = &.{},
            .want_args = &.{},
        },
        .{
            .reply = "Bad tool <tool_calls:opensource><tool_call:opensource>not_offered</tool_call:opensource></tool_calls:opensource>",
            .want_content = "Bad tool <tool_calls:opensource><tool_call:opensource>not_offered</tool_call:opensource></tool_calls:opensource>",
            .want_names = &.{},
            .want_args = &.{},
            .want_unoffered = 1,
        },
        .{
            .reply = "Stray text <tool_calls:opensource><tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource>" ++
                "<arg_value:opensource>Paris</arg_value:opensource>junk</tool_call:opensource></tool_calls:opensource>",
            .want_content = "Stray text <tool_calls:opensource><tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource>" ++
                "<arg_value:opensource>Paris</arg_value:opensource>junk</tool_call:opensource></tool_calls:opensource>",
            .want_names = &.{},
            .want_args = &.{},
            .want_malformed = 1,
        },
    };
    for (cases) |c| {
        const parsed = try tool_parse.parse(a, c.reply, tools, null);
        try testing.expectEqualStrings(c.want_content, parsed.content);
        try testing.expectEqual(c.want_unoffered, parsed.kept_unoffered);
        try testing.expectEqual(c.want_malformed, parsed.kept_malformed);
        if (c.want_names.len == 0) {
            try testing.expect(parsed.calls == null);
            continue;
        }
        const calls = parsed.calls orelse return error.ExpectedCalls;
        try testing.expectEqual(c.want_names.len, calls.len);
        for (calls, c.want_names, c.want_args) |call, wn, wa| {
            try testing.expectEqualStrings(wn, call.get("function").?.get("name").?.string);
            try testing.expectEqualStrings(wa, call.get("function").?.get("arguments").?.string);
        }
    }

    const parallel = cases[1].reply;
    const single = try tool_parse.parse(a, parallel, tools, 1);
    try testing.expectEqual(@as(usize, 1), single.calls.?.len);
    var pol: tool_stream.Policy = .{ .single = true };
    const filtered = try pol.text(a, try std.mem.concat(a, u8, &.{ "Before ", parallel, " After" }));
    try testing.expectEqualStrings("Before ", filtered);
    try testing.expectEqualStrings("", pol.flush());
}

test "hy4 markup: every byte boundary streams incremental deltas that join byte-for-byte to whole parse and hide markup" {
    const samples = [_][]const u8{
        "Lead prose.\n<tool_calls:opensource>" ++
            "<tool_call:opensource>search_docs" ++
            "<arg_key:opensource>query</arg_key:opensource><arg_value:opensource>h\u{e9}llo < 2 \"q\"</arg_value:opensource>" ++
            "<arg_key:opensource>note</arg_key:opensource><arg_value:opensource>null</arg_value:opensource>" ++
            "<arg_key:opensource>limit</arg_key:opensource><arg_value:opensource>12</arg_value:opensource>" ++
            "</tool_call:opensource>" ++
            "<tool_call:opensource>get_clock</tool_call:opensource>" ++
            "</tool_calls:opensource>",
        "Note <tool_calls:opensource> in prose, then <tool_call:opensource>get_weather" ++
            "<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Z\u{fc}rich</arg_value:opensource>" ++
            "<arg_key:opensource>tags</arg_key:opensource><arg_value:opensource>[\"a\",\"b\"]</arg_value:opensource>" ++
            "</tool_call:opensource>",
    };
    for (samples) |sample| {
        var base_arena: std.heap.ArenaAllocator = .init(testing.allocator);
        defer base_arena.deinit();
        const ba = base_arena.allocator();
        const tools = try toolsOf(ba);
        const whole = try tool_parse.parse(ba, sample, tools, null);
        const want_calls = whole.calls.?;
        const whole_prose = try reply_text.hideToolCalls(ba, sample, true);

        for (0..sample.len + 1) |cut_at| {
            var arena: std.heap.ArenaAllocator = .init(testing.allocator);
            defer arena.deinit();
            const a = arena.allocator();
            var s = try tool_stream.Streamer.init(a, tools);
            var out: std.ArrayList(Value) = .empty;
            try s.feed(sample[0..cut_at], &out);
            try s.feed(sample, &out);
            try s.finish(sample, &out);
            try testing.expectEqual(want_calls.len, countHeaders(out.items));
            for (want_calls, 0..) |wc, idx| {
                const want_arg = wc.get("function").?.get("arguments").?.string;
                try testing.expectEqualStrings(want_arg, try joinedStreamArgs(a, out.items, @intCast(idx)));
            }
            const hidden = try reply_text.hideToolCalls(a, sample[0..cut_at], false);
            try testing.expect(std.mem.startsWith(u8, whole_prose, hidden));
            try testing.expect(std.mem.indexOf(u8, hidden, "opensource>") == null or std.mem.indexOf(u8, whole_prose, "opensource>") != null);
        }

        var step_arena: std.heap.ArenaAllocator = .init(testing.allocator);
        defer step_arena.deinit();
        const sa = step_arena.allocator();
        var s3 = try tool_stream.Streamer.init(sa, tools);
        var out3: std.ArrayList(Value) = .empty;
        var early_deltas: usize = 0;
        const val_close = std.mem.indexOf(u8, sample, "</arg_value:opensource>").?;
        var k: usize = 0;
        while (k < sample.len) : (k += 3) {
            const end = @min(k + 3, sample.len);
            try s3.feed(sample[0..end], &out3);
            if (end < val_close) early_deltas = out3.items.len;
        }
        try s3.finish(sample, &out3);
        try testing.expect(early_deltas >= 3);
        for (want_calls, 0..) |wc, idx| {
            try testing.expectEqualStrings(wc.get("function").?.get("arguments").?.string, try joinedStreamArgs(sa, out3.items, @intCast(idx)));
        }
    }
}

test "hy4 markup: EOS closeCall and max_tokens cutCalls recover open HYTK envelopes safely" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try toolsOf(a);

    const open_val = "<tool_calls:opensource><tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Oslo";
    const tail = try tool_parse.closeCall(a, open_val, tools);
    try testing.expectEqualStrings("</arg_value:opensource></tool_call:opensource></tool_calls:opensource>", tail);
    const closed = try std.mem.concat(a, u8, &.{ open_val, tail });
    const parsed = try tool_parse.parse(a, closed, tools, null);
    try testing.expectEqualStrings("{\"city\":\"Oslo\"}", parsed.calls.?[0].get("function").?.get("arguments").?.string);

    const cut_mid = try tool_stream.cutCalls(a, open_val, tools);
    try testing.expectEqualStrings("{\"city\":\"Oslo", cut_mid.?[0].get("function").?.get("arguments").?.string);

    const open_outer = "<tool_calls:opensource><tool_call:opensource>get_clock</tool_call:opensource>";
    const cut_outer = try tool_stream.cutCalls(a, open_outer, tools);
    try testing.expectEqualStrings("{}", cut_outer.?[0].get("function").?.get("arguments").?.string);

    const bad = "<tool_calls:opensource><tool_call:opensource>unknown_fn<arg_key:opensource>x</arg_key:opensource><arg_value:opensource>1";
    try testing.expectEqualStrings("", try tool_parse.closeCall(a, bad, tools));
}

const hy4_jinja =
    \\{%- set HYTK = ':opensource' -%}
    \\{%- set hy_start_token = '<｜hy_start{}｜>'.format(HYTK) -%}
    \\{%- set hy_middle_token = '<｜hy_middle{}｜>'.format(HYTK) -%}
    \\{%- set hy_end_token = '<｜hy_end{}｜>'.format(HYTK) -%}
    \\{%- set think_begin_token = '<think{}>'.format(HYTK) -%}
    \\{%- set think_end_token = '</think{}>'.format(HYTK) -%}
    \\{%- set toolcalls_begin_token = '<tool_calls{}>'.format(HYTK) -%}
    \\{%- set toolcalls_end_token = '</tool_calls{}>'.format(HYTK) -%}
    \\{%- set toolcall_begin_token = '<tool_call{}>'.format(HYTK) -%}
    \\{%- set toolcall_end_token = '</tool_call{}>'.format(HYTK) -%}
    \\{%- set argkey_begin_token = '<arg_key{}>'.format(HYTK) -%}
    \\{%- set argkey_end_token = '</arg_key{}>'.format(HYTK) -%}
    \\{%- set argvalue_begin_token = '<arg_value{}>'.format(HYTK) -%}
    \\{%- set argvalue_end_token = '</arg_value{}>'.format(HYTK) -%}
    \\{%- set reasoning_mode_token = '<｜reasoning_mode{}｜>'.format(HYTK) -%}
    \\{%- if not reasoning_effort is defined -%}
    \\    {%- set reasoning_effort = 'high' -%}
    \\{%- elif reasoning_effort not in ['max', 'high', 'no_think'] -%}
    \\    {{- raise_exception('reasoning_effort error : ' + reasoning_effort + ', should be no_think/high/max') -}}
    \\{%- endif -%}
    \\{{- hy_start_token ~ 'system' ~ hy_middle_token ~ reasoning_mode_token ~ 'reasoning_effort:' ~ reasoning_effort ~ hy_end_token -}}
    \\{%- for m in messages -%}
    \\    {{- hy_start_token ~ m['role'] ~ hy_middle_token ~ m['content'] ~ hy_end_token -}}
    \\{%- endfor -%}
    \\{%- if add_generation_prompt -%}
    \\    {%- if reasoning_effort == 'no_think' -%}
    \\        {{- hy_start_token ~ 'assistant' ~ hy_middle_token ~ think_begin_token ~ think_end_token -}}
    \\    {%- else -%}
    \\        {{- hy_start_token ~ 'assistant' ~ hy_middle_token ~ think_begin_token -}}
    \\    {%- endif -%}
    \\{%- endif -%}
;

const Hy4Script = struct {
    reply_tokens: []const u32 = &.{},
    last_prompt: []const u32 = &.{},

    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 32768 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, req: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        s.last_prompt = req.prompt;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        var i: usize = 0;
        while (i < s.reply_tokens.len) : (i += 2) {
            const end = @min(i + 2, s.reply_tokens.len);
            sink.event(sink.ctx, id, &.{ .tokens = s.reply_tokens[i..end] });
        }
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const CaptureOut = struct {
    a: Allocator,
    status: u16 = 0,
    whole: ?Value = null,
    events: usize = 0,
    sse_reason: std.ArrayList(u8) = .empty,
    sse_content: std.ArrayList(u8) = .empty,
    sse_args: std.ArrayList(u8) = .empty,
    sse_finish: []const u8 = "",

    fn out(c: *@This()) openai.Out {
        return .{ .ctx = c, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(_: *anyopaque, _: usize) error{Closed}!void {}
    fn event(ctx: *anyopaque, payload: ?Value) error{Closed}!void {
        const c: *@This() = @ptrCast(@alignCast(ctx));
        const p = payload orelse return;
        c.events += 1;
        const choices = p.get("choices") orelse return;
        if (choices != .array or choices.array.len == 0) return;
        const ch = choices.array[0];
        if (ch.strField("finish_reason")) |fr| c.sse_finish = c.a.dupe(u8, fr) catch return error.Closed;
        const delta = ch.get("delta") orelse return;
        if (delta.strField("reasoning_content")) |r| c.sse_reason.appendSlice(c.a, r) catch return error.Closed;
        if (delta.strField("content")) |t| c.sse_content.appendSlice(c.a, t) catch return error.Closed;
        if (delta.get("tool_calls")) |arr| if (arr == .array) for (arr.array) |one| {
            if (one.get("function")) |fn_obj| if (fn_obj.strField("arguments")) |piece| {
                c.sse_args.appendSlice(c.a, piece) catch return error.Closed;
            };
        };
    }
    fn reply(ctx: *anyopaque, status: u16, payload: Value) void {
        const c: *@This() = @ptrCast(@alignCast(ctx));
        c.status = status;
        const text = json.stringify(c.a, payload, .{ .ascii = false }) catch return;
        c.whole = (json.parse(c.a, text) catch return).ok;
    }
};

test "hy4 markup: HTTP chat JSON and SSE split HYTK reasoning, honor high/max/no_think and stream tool calls" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});

    var vocab_buf: std.ArrayList(u8) = .empty;
    try vocab_buf.appendSlice(a, "{\"model\":{\"type\":\"BPE\",\"vocab\":{");
    for (32..127) |b| {
        if (b > 32) try vocab_buf.append(a, ',');
        const ch = [_]u8{@intCast(b)};
        try vocab_buf.print(a, "{s}:{d}", .{ try json.quote(a, &ch, .{}), b - 32 });
    }
    try vocab_buf.appendSlice(a, ",\"\\n\":95},\"merges\":[]},\"decoder\":{\"type\":\"Fuse\"},\"added_tokens\":[");
    const specials = [_]struct { id: u32, s: []const u8, sp: bool }{
        .{ .id = 96, .s = "<｜hy_start:opensource｜>", .sp = true },
        .{ .id = 97, .s = "<｜hy_middle:opensource｜>", .sp = true },
        .{ .id = 98, .s = "<｜hy_end:opensource｜>", .sp = true },
        .{ .id = 99, .s = "<｜reasoning_mode:opensource｜>", .sp = true },
        .{ .id = 100, .s = "<think:opensource>", .sp = false },
        .{ .id = 101, .s = "</think:opensource>", .sp = false },
        .{ .id = 102, .s = "<tool_calls:opensource>", .sp = false },
        .{ .id = 103, .s = "</tool_calls:opensource>", .sp = false },
        .{ .id = 104, .s = "<tool_call:opensource>", .sp = false },
        .{ .id = 105, .s = "</tool_call:opensource>", .sp = false },
        .{ .id = 106, .s = "<arg_key:opensource>", .sp = false },
        .{ .id = 107, .s = "</arg_key:opensource>", .sp = false },
        .{ .id = 108, .s = "<arg_value:opensource>", .sp = false },
        .{ .id = 109, .s = "</arg_value:opensource>", .sp = false },
    };
    for (specials, 0..) |tok, idx| {
        if (idx > 0) try vocab_buf.append(a, ',');
        try vocab_buf.print(a, "{{\"id\":{d},\"content\":{s},\"special\":{s}}}", .{ tok.id, try json.quote(a, tok.s, .{ .ascii = false }), if (tok.sp) "true" else "false" });
    }
    try vocab_buf.appendSlice(a, "]}");
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "tokenizer.json", .data = vocab_buf.items });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "tokenizer_config.json", .data = "{\"eos_token\":\"<｜hy_end:opensource｜>\"}" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"hy_v4_flash\",\"vocab_size\":110,\"eos_token_id\":98}" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "chat_template.jinja", .data = hy4_jinja });

    var problem: []const u8 = "";
    const hf = try HfText.load(testing.allocator, testing.io, dir_path, null, a, &problem);
    defer hf.deinit();
    var script: Hy4Script = .{};
    const srv = try Server.init(testing.allocator, testing.io, script.engine(), hf.text(), .{
        .served_name = "hy4-flash",
        .model_ids = &.{"hy4-flash"},
        .enable_thinking = true,
        .use_drafts = false,
    }, null);
    defer srv.deinit();

    try testing.expect(srv.info.capability(.thinking_applicable));
    try testing.expect(srv.info.capability(.thinking_switch));

    const generated = "Plan carefully.</think:opensource>\nLet me check.<tool_calls:opensource><tool_call:opensource>get_weather" ++
        "<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Berlin</arg_value:opensource>" ++
        "<arg_key:opensource>days</arg_key:opensource><arg_value:opensource>3</arg_value:opensource>" ++
        "</tool_call:opensource></tool_calls:opensource><｜hy_end:opensource｜>";
    script.reply_tokens = try hf.text().encode(a, generated, false);

    const req_base = "{\"model\":\"hy4-flash\",\"max_tokens\":256,\"messages\":[{\"role\":\"user\",\"content\":\"Weather?\"}]," ++
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\"," ++
        "\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}}}}}]";

    var conn: http_conn.Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var whole_cap: CaptureOut = .{ .a = a };
    openai.run(srv, a, whole_cap.out(), .{ .conn = &conn }, true, (try json.parse(a, try std.mem.concat(a, u8, &.{ req_base, "}" }))).ok);
    try testing.expectEqual(@as(u16, 200), whole_cap.status);
    const prompt_str = try hf.text().decode(a, script.last_prompt);
    try testing.expect(std.mem.indexOf(u8, prompt_str, "reasoning_effort:high") != null);
    try testing.expect(std.mem.endsWith(u8, prompt_str, "<think:opensource>"));
    const msg = whole_cap.whole.?.get("choices").?.array[0].get("message").?;
    try testing.expectEqualStrings("Plan carefully.", msg.get("reasoning_content").?.string);
    try testing.expectEqualStrings("Let me check.", msg.get("content").?.string);
    const tc = msg.get("tool_calls").?.array[0];
    try testing.expectEqualStrings("get_weather", tc.get("function").?.get("name").?.string);
    try testing.expectEqualStrings("{\"city\":\"Berlin\",\"days\":3}", tc.get("function").?.get("arguments").?.string);
    try testing.expectEqualStrings("tool_calls", whole_cap.whole.?.get("choices").?.array[0].get("finish_reason").?.string);

    var sse_cap: CaptureOut = .{ .a = a };
    openai.run(srv, a, sse_cap.out(), .{ .conn = &conn }, true, (try json.parse(a, try std.mem.concat(a, u8, &.{ req_base, ",\"stream\":true}" }))).ok);
    try testing.expect(sse_cap.events > 10);
    try testing.expectEqualStrings("Plan carefully.", sse_cap.sse_reason.items);
    try testing.expectEqualStrings("Let me check.", sse_cap.sse_content.items);
    try testing.expectEqualStrings("{\"city\":\"Berlin\",\"days\":3}", sse_cap.sse_args.items);
    try testing.expectEqualStrings("tool_calls", sse_cap.sse_finish);

    script.reply_tokens = try hf.text().encode(a, "Direct answer.<｜hy_end:opensource｜>", false);
    for ([_][]const u8{
        ",\"chat_template_kwargs\":{\"enable_thinking\":false}}",
        ",\"reasoning_effort\":\"none\"}",
        ",\"reasoning_effort\":\"no_think\"}",
    }) |off_tail| {
        var off_cap: CaptureOut = .{ .a = a };
        openai.run(srv, a, off_cap.out(), .{ .conn = &conn }, true, (try json.parse(a, try std.mem.concat(a, u8, &.{ req_base, off_tail }))).ok);
        try testing.expectEqual(@as(u16, 200), off_cap.status);
        const off_prompt = try hf.text().decode(a, script.last_prompt);
        try testing.expect(std.mem.indexOf(u8, off_prompt, "reasoning_effort:no_think") != null);
        try testing.expect(std.mem.endsWith(u8, off_prompt, "<think:opensource></think:opensource>"));
        const off_msg = off_cap.whole.?.get("choices").?.array[0].get("message").?;
        try testing.expect(off_msg.get("reasoning_content") == null);
        try testing.expectEqualStrings("Direct answer.", off_msg.get("content").?.string);
    }

    var max_cap: CaptureOut = .{ .a = a };
    openai.run(srv, a, max_cap.out(), .{ .conn = &conn }, true, (try json.parse(a, try std.mem.concat(a, u8, &.{ req_base, ",\"reasoning_effort\":\"xhigh\"}" }))).ok);
    try testing.expectEqual(@as(u16, 200), max_cap.status);
    try testing.expect(std.mem.indexOf(u8, try hf.text().decode(a, script.last_prompt), "reasoning_effort:max") != null);

    script.reply_tokens = try hf.text().encode(a, "Unclosed thought at EOS.<｜hy_end:opensource｜>", false);
    var rec_cap: CaptureOut = .{ .a = a };
    openai.run(srv, a, rec_cap.out(), .{ .conn = &conn }, true, (try json.parse(a, try std.mem.concat(a, u8, &.{ req_base, "}" }))).ok);
    const rec_msg = rec_cap.whole.?.get("choices").?.array[0].get("message").?;
    try testing.expectEqualStrings("Unclosed thought at EOS.", rec_msg.get("content").?.string);
    try testing.expect(rec_cap.whole.?.get("tensorfold").?.get("reasoning_recovered").?.bool);
}

test "hy4 markup: inline code and fenced code blocks quoting HYTK tool tags remain prose in whole and streaming" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try toolsOf(a);
    const quoted = "Use `<tool_calls:opensource><tool_call:opensource>get_clock</tool_call:opensource></tool_calls:opensource>` or:\n" ++
        "```xml\n<tool_calls:opensource>\n<tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Oslo</arg_value:opensource></tool_call:opensource>\n</tool_calls:opensource>\n```\nDone.";
    const parsed = try tool_parse.parse(a, quoted, tools, null);
    try testing.expect(parsed.calls == null);
    try testing.expectEqualStrings(quoted, try reply_text.finishedProse(a, quoted));
    var streamer: tool_stream.Streamer = try .init(a, tools);
    var deltas: std.ArrayList(Value) = .empty;
    var i: usize = 0;
    while (i < quoted.len) : (i += 3) {
        const next = @min(i + 3, quoted.len);
        try streamer.feed(quoted[0..next], &deltas);
    }
    try streamer.finish(quoted, &deltas);
    try testing.expectEqual(@as(usize, 0), deltas.items.len);
    try testing.expectEqualStrings(quoted, try reply_text.hideToolCalls(a, quoted, true));
}

test "hy4 markup: real TF_HY4_DIR checkpoint tokenizer and chat_template.jinja render and parse end to end" {
    const dir = if (std.c.getenv("TF_HY4_DIR")) |p| std.mem.span(p) else return;
    if (dir.len == 0) return;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const hf = try HfText.load(testing.allocator, testing.io, dir, null, a, &problem);
    defer hf.deinit();
    var script: Hy4Script = .{};
    const srv = try Server.init(testing.allocator, testing.io, script.engine(), hf.text(), .{
        .served_name = "hy4-flash",
        .model_ids = &.{"hy4-flash"},
        .enable_thinking = true,
        .use_drafts = false,
    }, null);
    defer srv.deinit();

    try testing.expectEqual(@as(?u32, 120030), srv.close_id);
    try testing.expect(srv.info.capability(.thinking_applicable));
    try testing.expect(srv.info.capability(.thinking_switch));

    const out_text = "Step 1: inspect tools.\n</think:opensource>Calling both.<tool_calls:opensource>" ++
        "<tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Seoul</arg_value:opensource>" ++
        "<arg_key:opensource>days</arg_key:opensource><arg_value:opensource>2</arg_value:opensource></tool_call:opensource>" ++
        "<tool_call:opensource>get_clock</tool_call:opensource></tool_calls:opensource><｜hy_end:opensource｜>";
    script.reply_tokens = try hf.text().encode(a, out_text, false);

    const body_text = "{\"model\":\"hy4-flash\",\"max_tokens\":256,\"messages\":[" ++
        "{\"role\":\"user\",\"content\":\"Check Seoul weather and clock\"}," ++
        "{\"role\":\"assistant\",\"content\":\"\",\"reasoning_content\":\"Earlier thought\",\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_clock\",\"arguments\":\"{}\"}}]}," ++
        "{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"12:00\"}," ++
        "{\"role\":\"user\",\"content\":\"Now Seoul too\"}]," ++
        "\"tools\":" ++ corpus_tools_json ++ "}";
    var conn: http_conn.Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var cap: CaptureOut = .{ .a = a };
    openai.run(srv, a, cap.out(), .{ .conn = &conn }, true, (try json.parse(a, body_text)).ok);
    try testing.expectEqual(@as(u16, 200), cap.status);
    const rendered = try hf.text().decode(a, script.last_prompt);
    try testing.expect(std.mem.indexOf(u8, rendered, "<think:opensource>Earlier thought</think:opensource>") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "<tool_response:opensource>12:00</tool_response:opensource>") != null);
    try testing.expect(std.mem.endsWith(u8, rendered, "<think:opensource>"));
    const msg = cap.whole.?.get("choices").?.array[0].get("message").?;
    try testing.expectEqualStrings("Step 1: inspect tools.", msg.get("reasoning_content").?.string);
    try testing.expectEqualStrings("Calling both.", msg.get("content").?.string);
    try testing.expectEqual(@as(usize, 2), msg.get("tool_calls").?.array.len);

    const ba_body = "{\"model\":\"hy4-flash\",\"max_tokens\":64,\"messages\":[" ++
        "{\"role\":\"user\",\"content\":\"Both?\"}," ++
        "{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[" ++
        "{\"id\":\"call_a\",\"type\":\"function\",\"function\":{\"name\":\"get_clock\",\"arguments\":\"{}\"}}," ++
        "{\"id\":\"call_b\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Rome\\\"}\"}}]}," ++
        "{\"role\":\"tool\",\"tool_call_id\":\"call_b\",\"content\":\"SECOND\"}," ++
        "{\"role\":\"tool\",\"tool_call_id\":\"call_a\",\"content\":\"FIRST\"}," ++
        "{\"role\":\"user\",\"content\":\"Summarize\"}],\"tools\":" ++ corpus_tools_json ++ "}";
    var ba_cap: CaptureOut = .{ .a = a };
    openai.run(srv, a, ba_cap.out(), .{ .conn = &conn }, true, (try json.parse(a, ba_body)).ok);
    try testing.expectEqual(@as(u16, 200), ba_cap.status);
    const ba_rendered = try hf.text().decode(a, script.last_prompt);
    try testing.expect(std.mem.indexOf(u8, ba_rendered, "<tool_response:opensource>FIRST</tool_response:opensource><tool_response:opensource>SECOND</tool_response:opensource>") != null);
}
