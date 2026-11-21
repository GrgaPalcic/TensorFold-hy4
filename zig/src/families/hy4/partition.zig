//! The tensor-parallel split (docs/recipes/hy4-flash.md, decisions 5 and 6): what each rank holds and computes.
const config = @import("config.zig");

const Config = config.Config;

pub const max_world = 4;
pub const block = 128; // an FP8 weight block's side

pub const Share = struct {
    lo: usize,
    len: usize,

    pub fn scaled(s: Share, by: usize) Share {
        return .{ .lo = s.lo * by, .len = s.len * by };
    }
};

/// `count` items in `world` equal shares (world 1, 2 or 4), each a whole number of `unit`s; anything else is refused.
pub fn even(count: usize, unit: usize, rank: usize, world: usize) !Share {
    if ((world != 1 and world != 2 and world != 4) or rank >= world) return error.UnsupportedWorld;
    if (unit == 0 or count % (world * unit) != 0) return error.SplitNotOnBlocks;
    const len = count / world;
    return .{ .lo = rank * len, .len = len };
}

/// A layer's attention on one rank: whole heads; `q_b`, `linear_gate`, `kv_b` rows and `o_proj` columns follow them.
pub const Attn = struct {
    heads: Share,
    q_rows: Share, // q_b: nope + rope rows a head
    gate_rows: Share, // linear_gate: v rows a head
    kv_rows: Share, // kv_b: nope + v rows a head
    o_cols: Share, // o_proj: v columns a head, partial sums all-reduced

    pub fn of(c: Config, layer: usize, rank: usize, world: usize) !Attn {
        const h = c.heads(layer);
        const heads = try even(h.count, 1, rank, world);
        const a: Attn = .{
            .heads = heads,
            .q_rows = heads.scaled(h.nope + h.rope),
            .gate_rows = heads.scaled(h.v),
            .kv_rows = heads.scaled(h.nope + h.v),
            .o_cols = heads.scaled(h.v),
        };
        for ([_]Share{ a.q_rows, a.gate_rows, a.o_cols }) |s| if (s.lo % block != 0 or s.len % block != 0) return error.SplitNotOnBlocks;
        return a;
    }
};

/// An FFN's intermediate width on one rank: gate and up rows, down columns; the down sums are all-reduced.
pub fn ffn(width: usize, rank: usize, world: usize) !Share {
    return even(width, block, rank, world);
}

/// Each routed expert split inside (every rank holds every expert's slice): routing stays replicated.
pub fn expert(c: Config, rank: usize, world: usize) !Share {
    return ffn(c.expert_width, rank, world);
}

/// Engram on one rank: whole hash heads (rank 0 the 2-gram heads at world 2), their rows and projection columns.
pub const Engram = struct {
    heads: Share,
    cols: Share, // value_proj and key_projs input columns: the heads' concatenated rows
    bias: bool, // the projections' biases join on one rank, before the all-reduce

    pub fn of(c: Config, rank: usize, world: usize) !Engram {
        const heads = try even(config.engram_heads, 1, rank, world);
        return .{ .heads = heads, .cols = heads.scaled(c.engram_dim), .bias = rank == 0 };
    }

    /// The table rows a rank's heads own in Engram layer `slot`: the heads lie back to back.
    pub fn rows(e: Engram, c: Config, slot: usize) Share {
        var lo: usize = 0;
        for (c.engram_sizes[slot][0..e.heads.lo]) |p| lo += p;
        var len: usize = 0;
        for (c.engram_sizes[slot][e.heads.lo..][0..e.heads.len]) |p| len += p;
        return .{ .lo = lo, .len = len };
    }
};

/// lm_head rows by vocabulary; the logits are gathered.
pub fn vocab(c: Config, rank: usize, world: usize) !Share {
    return even(c.vocab, 1, rank, world);
}

/// Every split the model needs at `world`, for every rank: refused by name before anything loads.
pub fn check(c: Config, world: usize, why: *config.Why) !void {
    if (world != 1 and world != 2 and world != 4) {
        why.set("world {d} is not supported (expected 1, 2 or 4)", .{world});
        return error.UnsupportedWorld;
    }
    for (0..world) |r| {
        for (0..c.layers) |l| _ = Attn.of(c, l, r, world) catch |e| {
            why.set("layer {d}: {d} heads do not split over {d} ranks on {d}-row blocks", .{ l, c.heads(l).count, world, block });
            return e;
        };
        _ = expert(c, r, world) catch |e| return refuse(why, "routed expert width", c.expert_width, world, e);
        _ = ffn(c.shared_width, r, world) catch |e| return refuse(why, "shared expert width", c.shared_width, world, e);
        _ = ffn(c.dense_width, r, world) catch |e| return refuse(why, "dense FFN width", c.dense_width, world, e);
        _ = Engram.of(c, r, world) catch |e| return refuse(why, "Engram heads", config.engram_heads, world, e);
        _ = vocab(c, r, world) catch |e| return refuse(why, "vocabulary", c.vocab, world, e);
    }
}

fn refuse(why: *config.Why, what: []const u8, n: usize, world: usize, e: anyerror) anyerror {
    why.set("{s} {d} does not split over {d} ranks", .{ what, n, world });
    return e;
}
