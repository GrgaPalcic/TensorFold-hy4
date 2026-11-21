# Hy4-flash per-layer reference fixtures (fp64, bf16) from transformers' hy_v4_flash; docs/recipes/hy4-flash.md.
import argparse, json, os, struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.hy_v4_flash import modeling_hy_v4_flash as M
from transformers.models.hy_v4_flash.configuration_hy_v4_flash import HYV4FlashConfig

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("out")
ap.add_argument("--layers", default="0,1,2,5")
ap.add_argument("--tokens", type=int, default=256)
ap.add_argument("--text", required=True, help="a UTF-8 text file; its first --tokens tokens are the prompt")
ap.add_argument("--threads", type=int, default=0)
a = ap.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(0)

cfg = HYV4FlashConfig.from_pretrained(a.model)
tc = cfg.text_config
tc._attn_implementation = "eager"  # sinks and the indexer's mask go through eager_attention_forward
index = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
handles = {}


def tensor(name):
    shard = index[name]
    if shard not in handles:
        handles[shard] = safe_open(os.path.join(a.model, shard), "pt")
    return handles[shard].get_tensor(name)


def has(name):
    return name in index


def dequant(codes, scale):
    """e4m3 codes [..., n, k] times their 128 x 128 blocks' scale_inv, exactly (fp64)."""
    n, k = codes.shape[-2:]
    s = scale.to(torch.float64).repeat_interleave(128, -2).repeat_interleave(128, -1)[..., :n, :k]
    return codes.to(torch.float64) * s


# checkpoint names -> module names (transformers' conversion_mapping for hy_v4_flash, text tower)
RENAMES = [(".self_attn.learnable_sink_param", ".self_attn.sinks"), (".self_attn.linear_gate.weight", ".self_attn.gate_proj.weight"),
           (".hc_attn_layer.hc_pre.hc_", ".attn_hc."), (".hc_mlp_layer.hc_pre.hc_", ".ffn_hc.")]
KEEP32 = M.HYV4FlashPreTrainedModel._keep_in_fp32_modules_strict


class LazyExperts(nn.Module):
    """HYV4FlashExperts.forward, each picked expert dequantized from the stored e4m3 stacks when first needed."""

    def __init__(self, prefix, limit, dtype):
        super().__init__()
        self.gu, self.gus = tensor(prefix + "gate_up_proj"), tensor(prefix + "gate_up_proj_scale_inv")
        self.dn, self.dns = tensor(prefix + "down_proj"), tensor(prefix + "down_proj_scale_inv")
        self.num_experts, self.swiglu_limit, self.dtype = self.gu.shape[0], limit, dtype

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            e = int(expert_idx[0])
            if e == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[e])
            gu = dequant(self.gu[e], self.gus[e]).to(self.dtype)
            dn = dequant(self.dn[e], self.dns[e]).to(self.dtype)
            gate, up = F.linear(hidden_states[token_idx], gu).chunk(2, dim=-1)
            gate = gate.clamp(min=None, max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            current = F.linear(F.silu(gate) * up, dn) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final


def bf16_rows(shard, name):
    """A bf16 tensor's rows by memory map (the Engram tables are ~30 GiB a layer)."""
    path = os.path.join(a.model, shard)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))[name]
    rows, dim = h["shape"]
    return np.memmap(path, dtype=np.uint16, mode="r", offset=8 + n + h["data_offsets"][0], shape=(rows, dim))


class LazyTable(nn.Module):
    """HYV4FlashEngramEmbedding.forward over the memory-mapped table."""

    def __init__(self, emb, rows):
        super().__init__()
        self.offsets, self.rows = emb.offsets, rows

    def forward(self, hash_ids):
        at = (hash_ids.clamp_min(0) + self.offsets).numpy()
        bits = torch.from_numpy(self.rows[at.reshape(-1)].astype(np.int16)).view(torch.bfloat16)
        out = bits.reshape(*at.shape, self.rows.shape[1])
        return out.masked_fill(hash_ids.unsqueeze(-1) < 0, 0.0)


def build_layer(i, dtype):
    with torch.device("meta"):
        layer = M.HYV4FlashDecoderLayer(tc, i)
    moe = tc.mlp_layer_types[i] != "dense"
    if moe:  # replaced below by the lazy stacks: never allocated
        layer.mlp.experts = nn.Identity()
    if layer.engram_layer is not None:
        sizes = layer.engram_layer.config_head_vocab_sizes
        layer.engram_layer.multi_head_embedding = nn.Identity()
    layer = layer.to_empty(device="cpu")
    if layer.self_attn.compressed_indexer is not None:  # its rotary tables are derived, not stored
        layer.self_attn.compressed_indexer.rotary_emb = M.HYV4FlashRotaryEmbedding(tc)
    state = {}
    p = f"model.layers.{i}."
    for name in [k for k in index if k.startswith(p)]:
        short = name[len(p) - 1:]
        if "engram_layer.multi_head_embedding" in short or ".mlp.experts." in short or short.endswith("_scale_inv"):
            continue
        t = tensor(name)
        if t.dtype == torch.float8_e4m3fn:
            t = dequant(t, tensor(name + "_scale_inv"))
        for old, new in RENAMES:
            short = short.replace(old, new)
        state[short[1:]] = t
    if layer.engram_layer is not None:  # buffers the checkpoint carries or the module derives
        state["engram_layer.lookup_table"] = tensor(p + "engram_layer.lookup_table")
    missing, unexpected = layer.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise SystemExit(f"layer {i}: missing {sorted(missing)[:8]}, unexpected {sorted(unexpected)[:8]}")
    # the run's dtype from each stored value; transformers' fp32 modules stay fp32 in the bf16 run
    for n, prm in list(layer.named_parameters()) + list(layer.named_buffers()):
        if prm.is_floating_point() and not n.endswith("inv_freq"):
            keep = dtype != torch.float64 and any(k in n for k in KEEP32)
            prm.data = prm.data.to(torch.float32 if keep else dtype)
    if layer.engram_layer is not None:
        g = layer.engram_layer
        g.head_vocab_sizes = torch.tensor(sizes)
        g.hash_multipliers = g.draw_hash_multipliers(int(g.lookup_table.max()) + 1)
        name = p + "engram_layer.multi_head_embedding.embedding.weight"
        table = nn.Module()
        table.offsets = torch.tensor([0] + sizes[:-1]).cumsum(0)
        g.multi_head_embedding = LazyTable(table, bf16_rows(index[name], name))
    if moe:
        layer.mlp.experts = LazyExperts(p + "mlp.experts.", tc.swiglu_limit, dtype)
    return layer.eval()


def frame(ids, dtype):
    """What the text model hands each layer: masks, rotary tables, positions, n-gram inputs (one document)."""
    T = ids.shape[1]
    position_ids = torch.arange(T).view(1, 1, -1).expand(3, 1, -1)
    dummy = torch.zeros(1, T, tc.hidden_size, dtype=dtype)
    kw = dict(config=tc, inputs_embeds=dummy, attention_mask=None, cache=None, position_ids=position_ids[0], allow_is_causal_skip=False)
    masks = {"indexed_attention": create_causal_mask(**kw), "sliding_attention": create_sliding_window_causal_mask(**kw)}
    rotary = M.HYV4FlashRotaryEmbedding(tc)
    rope = {lt: rotary(dummy, position_ids=position_ids, layer_type=lt) for lt in rotary.layer_types}
    pad = tc.engram_pad_id
    shifted = [ids]
    for off in range(1, tc.engram_max_ngram_size):
        prev = F.pad(ids, (off, 0), value=pad)[:, :T]
        shifted.append(torch.where(position_ids[0] < off, pad, prev))
    ngram = torch.stack(shifted, dim=-1)
    emask = (ids != cfg.image_token_id) & (ids != cfg.video_token_id) & (ids != pad)
    return dict(masks=masks, rope=rope, position_ids=position_ids, ngram=ngram, emask=emask)


def run(layer, x, fr, taps, tag):
    hooks = []

    def keep(name, fn):
        def hook(mod, args, kwargs, out):
            taps[f"{name}{tag}"] = fn(args, kwargs, out).detach().contiguous()
        return hook

    sa = layer.self_attn
    for name, hc in (("attn_hc", layer.attn_hc), ("ffn_hc", layer.ffn_hc)):  # iHC: (post, mixed stream)
        hooks.append(hc.register_forward_hook(keep(name + "_post", lambda a_, k, o: o[0]), with_kwargs=True))
        hooks.append(hc.register_forward_hook(keep(name + "_out", lambda a_, k, o: o[1]), with_kwargs=True))
    hooks.append(layer.attn_hc.register_forward_hook(keep("attn_hc_in", lambda a_, k, o: a_[0]), with_kwargs=True))
    hooks.append(layer.ffn_hc.register_forward_hook(keep("mid", lambda a_, k, o: a_[0]), with_kwargs=True))  # streams
    hooks.append(sa.register_forward_hook(keep("attn_in", lambda a_, k, o: k["hidden_states"]), with_kwargs=True))
    hooks.append(sa.register_forward_hook(keep("attn_out", lambda a_, k, o: o[0]), with_kwargs=True))
    hooks.append(layer.mlp.register_forward_hook(keep("mlp_out", lambda a_, k, o: o), with_kwargs=True))
    if hasattr(layer.mlp, "gate"):
        hooks.append(layer.mlp.gate.register_forward_hook(keep("router_ids", lambda a_, k, o: o[2].to(torch.int32)), with_kwargs=True))
        hooks.append(layer.mlp.gate.register_forward_hook(keep("router_w", lambda a_, k, o: o[1]), with_kwargs=True))
    if sa.compressed_indexer is not None:
        hooks.append(sa.compressed_indexer.register_forward_hook(keep("recall", lambda a_, k, o: o.to(torch.int32)), with_kwargs=True))
    if layer.engram_layer is not None:
        hooks.append(layer.engram_layer.register_forward_hook(keep("engram_out", lambda a_, k, o: o), with_kwargs=True))
    lt = layer.attention_type
    with torch.no_grad():
        y = layer(x, attention_mask=fr["masks"][lt], position_ids=fr["position_ids"], cache=None, use_cache=False,
                  position_embeddings=fr["rope"][lt], indexer_position_ids=fr["position_ids"], ngram_tokens=fr["ngram"],
                  engram_mask=fr["emask"])
    for h in hooks:
        h.remove()
    return y


def self_check():
    """The two substitutions against the PR's own modules, bit for bit, on the checkpoint's first MoE layer."""
    import copy
    i = next(j for j, t in enumerate(tc.mlp_layer_types) if t != "dense")
    small = copy.deepcopy(tc)
    small.n_routed_experts = 8
    ref = M.HYV4FlashExperts(small).to(torch.float64)
    lazy = LazyExperts(f"model.layers.{i}.mlp.experts.", tc.swiglu_limit, torch.float64)
    lazy.gu, lazy.gus, lazy.dn, lazy.dns, lazy.num_experts = lazy.gu[:8], lazy.gus[:8], lazy.dn[:8], lazy.dns[:8], 8
    ref.gate_up_proj.data, ref.down_proj.data = dequant(lazy.gu, lazy.gus), dequant(lazy.dn, lazy.dns)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(6, tc.routed_expert_hidden_size, generator=g, dtype=torch.float64)
    idx = torch.stack([torch.randperm(8, generator=g)[:4] for _ in range(6)])
    w = torch.rand(6, 4, generator=g, dtype=torch.float64)
    assert torch.equal(ref(x, idx, w), lazy(x, idx, w)), "lazy experts differ from HYV4FlashExperts"
    sizes = [3, 5, 7]
    emb = M.HYV4FlashEngramEmbedding(sizes, 4)
    emb.embedding.weight.data = torch.randn(15, 4, generator=g).to(torch.bfloat16)
    rows = emb.embedding.weight.data.view(torch.int16).numpy().view(np.uint16)
    hid = torch.tensor([[[0, 4, -1], [2, 0, 6]]])
    holder = nn.Module()
    holder.offsets = emb.offsets
    assert torch.equal(emb(hid), LazyTable(holder, rows)(hid)), "lazy table differs from HYV4FlashEngramEmbedding"
    print("self-check: lazy experts and table equal the PR's modules", flush=True)


self_check()
tok = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
ids = tok.encode(open(a.text, encoding="utf-8").read(), add_special_tokens=False).ids
if len(ids) < a.tokens:
    raise SystemExit(f"{a.text} holds {len(ids)} tokens, fewer than --tokens {a.tokens}")
ids = torch.tensor([[cfg.text_config.bos_token_id] + ids[: a.tokens - 1]])
T = ids.shape[1]
json.dump({"ids": ids[0].tolist(), "positions": list(range(T)), "text": a.text}, open(os.path.join(a.out, f"ids_t{T}.json"), "w"))
fr = {torch.float64: frame(ids, torch.float64), torch.bfloat16: frame(ids, torch.bfloat16)}

embed = tensor("model.embed_tokens.weight")[ids[0]]  # bf16 [T, H]
x = embed.unsqueeze(0).unsqueeze(2).expand(-1, -1, tc.hc_mult, -1).contiguous()  # [1, T, 4, H] bf16
for i in [int(s) for s in a.layers.split(",")]:
    taps = {"in": x.clone()}
    y64 = run(build_layer(i, torch.float64), x.to(torch.float64), fr[torch.float64], taps, "64")
    y16 = run(build_layer(i, torch.bfloat16), x, fr[torch.bfloat16], taps, "16")
    taps["out64"], taps["out16"] = y64.contiguous(), y16.contiguous()
    err = (y16.double() - y64).abs()
    print(f"layer {i} ({tc.layer_types[i]}, {tc.mlp_layer_types[i]}): bf16 vs fp64 max {err.max():.3e} mean {err.mean():.3e}", flush=True)
    save_file(taps, os.path.join(a.out, f"layer{i}_t{T}.safetensors"), metadata={"layer": str(i), "tokens": str(T)})
    x = y64.to(torch.bfloat16)

# the head on the last streams: hc_head, the final norm, lm_head in fp32 (as the model keeps it)
head = {"in": x.clone()}
for dtype, tag in ((torch.float64, "64"), (torch.bfloat16, "16")):
    with torch.device("meta"):
        hc, norm = M.HYV4FlashHyperHead(tc), M.HYV4FlashRMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)
    hc, norm = hc.to_empty(device="cpu"), norm.to_empty(device="cpu")
    hc.load_state_dict({"hc_fn": tensor("model.hc_head.hc_head_fn"), "hc_base": tensor("model.hc_head.hc_head_base"), "hc_scale": tensor("model.hc_head.hc_head_scale")})
    norm.load_state_dict({"weight": tensor("model.norm.weight")})
    hc, norm = hc.to(dtype), norm.to(dtype)
    if dtype != torch.float64:
        hc = hc.float()
    w = tensor("lm_head.weight").to(torch.float64 if dtype == torch.float64 else torch.float32)
    with torch.no_grad():
        h = norm(hc(x.to(dtype)))
        head[f"hidden{tag}"] = h
        head[f"logits{tag}"] = F.linear(h.to(w.dtype), w)
save_file(head, os.path.join(a.out, f"head_t{T}.safetensors"), metadata={"tokens": str(T)})
e = (head["logits16"].double() - head["logits64"]).abs()
print(f"head: bf16 vs fp64 logits max {e.max():.3e}; top-1 equal {(head['logits16'].argmax(-1) == head['logits64'].argmax(-1)).float().mean():.4f}", flush=True)
