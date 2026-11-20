//! Pull's requests follow each redirect here, so every hop's host takes its own route: its proxy or NO_PROXY's direct.
const std = @import("std");
const route = @import("pull_route.zig");
const Client = std.http.Client;

pub const max_hops = 3;

/// A bodiless `method` to `url` into `req`, following up to three redirects; `buf` holds the redirect targets.
pub fn open(client: *Client, req: *Client.Request, method: std.http.Method, url: []const u8, extra: []const std.http.Header, buf: []u8) !Client.Response {
    var uri = std.Uri.parse(url) catch return error.BadUrl;
    var rest = buf;
    var hops: usize = 0;
    while (true) : (hops += 1) {
        try route.note(client.io, uri);
        req.* = try client.request(method, uri, .{ .redirect_behavior = .unhandled, .extra_headers = extra, .headers = .{ .accept_encoding = .{ .override = "identity" } } });
        errdefer req.deinit();
        try req.sendBodiless();
        const response = try req.receiveHead(&.{});
        const location = response.head.location orelse return response;
        if (response.head.status.class() != .redirect or hops == max_hops) return response;
        if (location.len > rest.len) return error.HttpRedirectLocationOversize;
        @memcpy(rest[0..location.len], location);
        uri = uri.resolveInPlace(location.len, &rest) catch return error.HttpRedirectLocationInvalid;
        req.deinit();
    }
}

/// A small body whole, after its redirects, with its status.
pub fn body(a: std.mem.Allocator, client: *Client, url: []const u8) !struct { std.http.Status, []u8 } {
    var req: Client.Request = undefined;
    var buf: [8 << 10]u8 = undefined;
    var response = try open(client, &req, .GET, url, &.{}, &buf);
    defer req.deinit();
    const status = response.head.status;
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    var transfer: [8 << 10]u8 = undefined;
    _ = response.reader(&transfer).streamRemaining(&out.writer) catch return error.ReadFailed;
    return .{ status, try out.toOwnedSlice() };
}
