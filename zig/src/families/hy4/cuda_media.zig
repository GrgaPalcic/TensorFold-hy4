//! Per-sequence multimodal prompt state for Hy4-flash: ViT features, 4-stream splice map, and 3D M-RoPE positions.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const vcfg = @import("vision_config.zig");
const cvis = @import("cuda_vision.zig");
const cuda_rows = @import("cuda_rows.zig");
const cuda_index = @import("cuda_index.zig");
const cuda_tp = @import("cuda_tp.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const Stream = cuda.Stream;

/// Active prompt media attached to a `Seq` slot during multimodal prefill (freed once prefill completes or resets).
pub const SeqMedia = struct {
    gpa: std.mem.Allocator,
    prompt_len: usize,
    feat_rows: usize,
    delta: i32,
    features: DeviceBuffer, // bf16 [feat_rows, out_hidden]
    src_idx: DeviceBuffer, // i32 [prompt_len]: -1 on text, 0..feat_rows-1 on image_token
    pos3: DeviceBuffer, // i32 [3, prompt_len]: T, H, W positions for prompt tokens
    mrope_cs: DeviceBuffer, // float2 [prompt_cap, 32]: 3D M-RoPE sin/cos for indexer pool/query in prompt
    src_host: []i32,
    pos3_host: []i32,

    pub fn allocate(gpa: std.mem.Allocator, d: *const cuda.Driver, prompt_len: usize, feat_rows: usize, out_hidden: usize, delta: i32) !SeqMedia {
        const prompt_cap = std.mem.alignForward(usize, @max(prompt_len, 4), 4);
        var feat = try DeviceBuffer.alloc(d, @max(feat_rows * out_hidden * 2, 16));
        errdefer feat.free();
        var sidx = try DeviceBuffer.alloc(d, @max(prompt_len * 4, 16));
        errdefer sidx.free();
        var p3 = try DeviceBuffer.alloc(d, @max(3 * prompt_len * 4, 16));
        errdefer p3.free();
        var mcs = try DeviceBuffer.alloc(d, prompt_cap * cuda_index.rope_pairs * 8);
        errdefer mcs.free();
        const sh = try gpa.alloc(i32, prompt_len);
        errdefer gpa.free(sh);
        const ph = try gpa.alloc(i32, 3 * prompt_len);
        return .{
            .gpa = gpa,
            .prompt_len = prompt_len,
            .feat_rows = feat_rows,
            .delta = delta,
            .features = feat,
            .src_idx = sidx,
            .pos3 = p3,
            .mrope_cs = mcs,
            .src_host = sh,
            .pos3_host = ph,
        };
    }

    pub fn deinit(sm: *SeqMedia) void {
        sm.gpa.free(sm.pos3_host);
        sm.gpa.free(sm.src_host);
        sm.mrope_cs.free();
        sm.pos3.free();
        sm.src_idx.free();
        sm.features.free();
        sm.* = undefined;
    }

    /// Splices image features into `[rows, 4, hidden]` iHC `streams` for prompt chunk `[first, first + rows)`.
    pub fn spliceChunk(sm: *const SeqMedia, rows_op: *const cuda_rows.Rows, s: Stream, streams: u64, first: usize, rows: usize, hidden: usize) !void {
        if (first >= sm.prompt_len or rows == 0) return;
        const n = @min(rows, sm.prompt_len - first);
        var has_img = false;
        for (sm.src_host[first .. first + n]) |idx| if (idx >= 0) {
            has_img = true;
            break;
        };
        if (!has_img) return;
        try rows_op.spliceStreams(s, sm.features.ptr, sm.src_idx.ptr + first * 4, streams, hidden, n);
    }
};

pub fn clearSeq(media: *?SeqMedia, mrope_delta: *i32) void {
    if (media.*) |*sm| sm.deinit();
    media.* = null;
    mrope_delta.* = 0;
}

pub fn shiftSkippedSrc(src: []i32, skip_feat_rows: usize) void {
    if (skip_feat_rows == 0) return;
    const skip_i32: i32 = @intCast(skip_feat_rows);
    for (src) |*idx| {
        if (idx.* >= 0) idx.* = if (idx.* < skip_i32) -1 else idx.* - skip_i32;
    }
}

/// Validates placeholders on rank 0, runs ViT `encode`, populates `mrope_cs`, and broadcasts to TP followers.
pub fn prepareRank0(
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    s: Stream,
    vis: *const cvis.Vision,
    rows_op: *const cuda_rows.Rows,
    norms: *const cuda.norms.Norms,
    rope_table_fn: cuda.Function,
    idx_freqs: u64,
    tp_sess: ?*cuda_tp.Session,
    slot: usize,
    ids: []const u32,
    images: []const lanes.stream.Image,
    resumed_at: usize,
) !SeqMedia {
    const c = vis.c;
    if (images.len == 0 or images.len > vis.limits.max_images) return error.BadImageCount;
    const tower = vis.tower orelse return error.NoHy4Vision;

    var runs: [64]vcfg.Run = undefined;
    var total_patches: usize = 0;
    var skip_imgs: usize = 0;
    var skip_feat_rows: usize = 0;
    var enc_feat_rows: usize = 0;
    var prev_end: usize = 0;
    for (images, 0..) |im, i| {
        const r: vcfg.Run = .{ .at = im.at, .gh = im.gh, .gw = im.gw };
        if (r.gh == 0 or r.gw == 0 or r.gh % vcfg.merge != 0 or r.gw % vcfg.merge != 0) return error.BadImageGrid;
        if (im.tokens != r.tokens() or im.pixels.len != r.patchesCount() * vcfg.patch_width) return error.BadImagePatches;
        if (r.at == 0 or r.at <= prev_end or r.at + r.tokens() >= ids.len) return error.BadImagePlaceholders;
        if (ids[r.at - 1] != c.image_start or ids[r.at + r.tokens()] != c.image_end) return error.BadImagePlaceholders;
        for (ids[r.at..][0..r.tokens()]) |tok| if (tok != c.image_token) return error.BadImagePlaceholders;
        prev_end = r.at + r.tokens();
        total_patches += r.patchesCount();
        runs[i] = r;
        if (skip_imgs == i and prev_end <= resumed_at) {
            skip_imgs += 1;
            skip_feat_rows += r.tokens();
        } else {
            enc_feat_rows += r.tokens();
        }
    }
    if (total_patches > vcfg.max_patches) return error.BadImageGrid;

    var sm = try SeqMedia.allocate(gpa, d, ids.len, enc_feat_rows, c.out_hidden, 0);
    errdefer sm.deinit();
    sm.delta = vcfg.layout(ids.len, runs[0..images.len], sm.src_host, sm.pos3_host);
    for (ids, 0..) |tok, at| {
        if (tok == c.video_token) return error.Hy4VideoUnsupported;
        if (tok == c.image_token and sm.src_host[at] < 0) return error.BadImagePlaceholders;
    }
    shiftSkippedSrc(sm.src_host, skip_feat_rows);

    var feat_off: usize = 0;
    for (images[skip_imgs..], runs[skip_imgs..images.len]) |im, r| {
        try tower.encode(s, rows_op, norms, im.pixels, &.{.{ .h = r.gh, .w = r.gw }}, sm.features.ptr + feat_off * c.out_hidden * 2);
        feat_off += r.tokens();
    }
    try cuda.copy_source.enqueue(d, sm.src_idx.ptr, std.mem.sliceAsBytes(sm.src_host), s);
    try cuda.copy_source.enqueue(d, sm.pos3.ptr, std.mem.sliceAsBytes(sm.pos3_host), s);
    if (idx_freqs != 0) {
        try cuda_index.ropeTable(rope_table_fn, s, sm.mrope_cs.ptr, 0, ids.len, idx_freqs, sm.pos3.ptr, ids.len, 10, 10, false);
    }
    try s.synchronize();

    if (tp_sess) |ts| {
        try ts.send(&.{
            @backingInt(cuda_tp.Op.set_media),
            @intCast(slot),
            @intCast(ids.len),
            @intCast(enc_feat_rows),
            @intCast(c.out_hidden),
            sm.delta,
        });
        try ts.broadcastFrom0(s, sm.features.ptr, enc_feat_rows * c.out_hidden * 2);
        try ts.broadcastFrom0(s, sm.src_idx.ptr, ids.len * 4);
        try ts.broadcastFrom0(s, sm.pos3.ptr, 3 * ids.len * 4);
        try s.synchronize();
        try ts.check();
    }
    return sm;
}

/// Receives encoded ViT features and 3D M-RoPE buffers on follower ranks (`rank > 0`).
pub fn receiveFollower(
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    s: Stream,
    rope_table_fn: cuda.Function,
    idx_freqs: u64,
    ts: *cuda_tp.Session,
    prompt_len: usize,
    feat_rows: usize,
    out_hidden: usize,
    delta: i32,
) !SeqMedia {
    var sm = try SeqMedia.allocate(gpa, d, prompt_len, feat_rows, out_hidden, delta);
    errdefer sm.deinit();
    try ts.broadcastFrom0(s, sm.features.ptr, feat_rows * out_hidden * 2);
    try ts.broadcastFrom0(s, sm.src_idx.ptr, prompt_len * 4);
    try ts.broadcastFrom0(s, sm.pos3.ptr, 3 * prompt_len * 4);
    try s.synchronize();
    try ts.check();
    try sm.src_idx.download(0, std.mem.sliceAsBytes(sm.src_host));
    try sm.pos3.download(0, std.mem.sliceAsBytes(sm.pos3_host));
    if (idx_freqs != 0) {
        try cuda_index.ropeTable(rope_table_fn, s, sm.mrope_cs.ptr, 0, prompt_len, idx_freqs, sm.pos3.ptr, prompt_len, 10, 10, false);
        try s.synchronize();
    }
    return sm;
}

test "cuda_media partial ViT skip shifts uncached src_idx while preserving 3D M-RoPE" {
    const runs = [2]vcfg.Run{ .{ .at = 1, .gh = 4, .gw = 4 }, .{ .at = 9, .gh = 4, .gw = 4 } };
    var src: [16]i32 = undefined;
    var pos3: [48]i32 = undefined;
    const delta = vcfg.layout(16, &runs, &src, &pos3);
    try std.testing.expectEqual(@as(i32, -6), delta);
    shiftSkippedSrc(&src, runs[0].tokens());
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, -1 }, &src);
    std.testing.refAllDecls(@This());
}
