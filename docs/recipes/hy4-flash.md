# Hy4-flash on CUDA

Native CUDA serving for `tencent/Hy4-flash-FP8` (`hy_v4_flash`) in full checkpoint precision (W8A16) across two or
four GB10 GPUs (`sm_121`) over ConnectX-7 RoCE + NCCL, with built-in 5-layer `DFly2` speculative drafting, `--vision`
multimodal encoder (`model.vit.*` + 3D M-RoPE), `--kv-cache fp8` up to `1,048,576` context, CUDA Graphs, `HYTK`
reasoning/tool streaming, `xgrammar` structured output, and block-deduplicated NVMe conversation spill/recall.

## Contract

| | |
|---|---|
| Behaviour added | Native CUDA serving of `tencent/Hy4-flash-FP8` (model_type `hy_v4_flash`): text & `--vision` image inputs, `--kv-cache bf16\|fp8` up to `1M` context, built-in 5-layer `DFly2` + copy speculative decoding, CUDA Graphs, `xgrammar` structured output, `HYTK` tools/reasoning, NVMe conversation spill/recall |
| Checkpoint | `tencent/Hy4-flash-FP8` @ `003107fa248f5892064ab1c43af0a7003dd68c9b`: 148 shards, 2,199 tensors, 267.4 GiB |
| Reference | transformers PR #50903 at `c2ec850510a37978ea83df226589623d5f2be776` (`models/hy_v4_flash`), run in fp32 and fp64 with the FP8 weights dequantized exactly |
| Precision | **full W8A16 weights**: e4m3 codes expanded to bf16 in the kernels, bf16 activations, fp32 sums; the ops the reference keeps in fp32 stay fp32 (below). `DFly2` optionally packs its draft-only `lm_head` view as 1x64 `FP8G` while target verification keeps exact `bf16`. KV cache defaults to `bf16` (`14,596 B/tok`) with opt-in `--kv-cache fp8` (`8,644 B/tok`: e4m3 `kv_lora` + fp32 scale, bf16 `k_rope` & indexer `group_keys`) |
| Backend, GPUs | CUDA on sm_121 (GB10, CUDA 13.0) |
| Ranks | 1 for per-layer checks; 2 (TP=2 over one dual-port CX7 link) and 4 (TP=4 over a switchless CX7 ring) to serve the whole checkpoint |
| Refused by name | Video parts, image parts when served without `--vision`, external `--drafter` (built-in `DFly2` is loaded from `model.dfly_layers.*`), other quantizations, other Engram seeds or maps, unknown layer kinds, `--precision checkpoint`, world sizes other than 1, 2 or 4 |
| Not covered yet | Video inputs, world sizes other than 1, 2 or 4 |
## The model, as the engine sees it

- 42 layers, hidden 3,072, vocab 120,832, context 1,048,576. Layer 0 has a dense FFN (13,824); layers 1-41 are MoE.
- Attention by `swa_pattern`: 12 indexed layers (0, 1, 5, 9, …, 41) and 30 sliding-window layers (window 512, the
  row's own key included).
- Gated MLA on every layer: q_lora 1,024, kv_lora 512, query/key 192 nope + 64 rope, value 256. Indexed layers have 32
  heads, sliding layers 64. A learned sink logit a head (fp32). An elementwise sigmoid gate on the attention output
  (`linear_gate`, 256 a head), then o_proj and `post_proj_norm`.
- RoPE interleaved, rope theta 1e7 on indexed layers and 1e4 on sliding ones; 3D interleaved MRoPE sections 12/10/10
  (`hy4_rows_kv_write_ex`, `hy4_rows_q_rope_mrope`, `tf_hy4_idx_q_mrope`: `(t, h, w)` over image spans and `pos + delta`
  on trailing text; text-only turns keep all three axes equal).
- Vision (`--vision`, `vision_config.zig`, `cuda_vision.zig`, `cuda_media.zig`): `HYV4FlashImageProcessor` bicubic
  `smart_resize` (`factor = 32`), 27-layer `1,152`-dim 16-head SigLIP ViT (`16x16` patch conv + bilinear `128x128`
  learned pos embed, `LayerNorm`, `fc1` GELU `4,304` MLP) + `HYV4FlashVisionPatchMerger` (`RMSNorm -> 2x2 merge -> GELU -> 3,072`
  + `image_newline` per row, `891.7 MiB` on rank 0, `0 B` on followers, broadcast via `Op.set_media`).
- The compressed indexer on indexed layers: a per-token key (3,072 → 128, RMSNorm); only the trailing `<= 3` incomplete
  tail rows (`[4, 128]` bf16 per sequence, `cuda_index.runTail`) stay buffered between calls while completed 4-token
  groups pool in fp32 by softmax over learned gates plus `intra_group_ape`, then `key_norm`; a group key takes the RoPE
  phase of its first token. Queries from `q_resid` (16 heads × 128), scores relu(q·k) weighted per head, a query keeps
  its best 511 complete groups plus its own: 2,048 tokens. A query at slot t < 2,048 recalls every earlier group, so it
  attends densely; the reference skips the indexer when the whole key length is ≤ 2,048.
- LatentMoE: 512 experts, top 16, sigmoid scores plus `e_score_correction_bias` (fp32), the picked scores normalized
  and scaled by 3.998. The routed experts work at 1,536: `routed_expert_down_proj` 3,072 → 1,536 before them,
  `routed_expert_up_proj` 1,536 → 3,072 after. SwiGLU clamp 10 (gate ≤ 10, up in ±10). One shared expert at full
  width. `combine_output_norm` after the sum; the dense MLP has `post_fc2_norm`.
- iHC: 4 residual streams. Each sublayer: `mixes = (streams · fn) · rsqrt(mean(streams²) + eps)`, pre =
  sigmoid(mix · scale + base) + eps, post = 2 · sigmoid(…) + eps, all fp32; the sublayer sees Σ pre·stream, and its
  output is added back as post · out + stream (fp32, stored bf16). DeepSeek-V4's mHC without the Sinkhorn
  combination. `hc_head` collapses the streams before the final norm.
- Engram on layers 1, 11, 21, 31 (0-based; config ids 2, 12, 22, 32), added to the streams before attention (next
  section).
- lm_head in fp32 (bf16 weights widened, the hidden row cast to fp32).

## Engram

| | |
|---|---|
| Orders | 2-grams and 3-grams of the current token and the ones before it; never across a document start |
| Heads | 8 an order, 16 a layer; each a prime-sized table of ~10.31M rows, stored as one table a layer: bf16 [rows, 96], rows the sum of its 16 primes (164,994,380 to 165,008,942) |
| Size | 63.4B parameters, 118.0 GiB bf16 over the four layers |
| Token map | `lookup_table` int32 [120,818], identical in the four layers: 92,620 normalized ids; pad 120002; ids past the map read as pad |
| Hash | `c = map[token]`, `m_i = c_i × mult_i` (int64), order n: XOR of `m_0 … m_{n-1}`, then `mod` each head's prime |
| Multipliers | NumPy `default_rng(seed + 10007 × layer_id)`, `integers(0, int64_max // 92,620 // 2, 3) × 2 + 1`, seed 0 |
| Read | each of a row's 16 vectors through an FP8 e4m3 round trip with its own scale, RMSNorm a head, concatenated (1,536), `value_proj` and 4 `key_projs`; a stream's gate is sigmoid of a signed square root of a normed key-query dot product; masked on pad, image and video positions |

**Storage: FP8 rows, exact against the fp64 round trip.** The reference passes each looked-up 96-vector through an
FP8 e4m3 round trip with its own scale (`_fp8_round_trip`: scale = amax / 448, codes = e4m3(x / scale) clamped to
±448, value = code × scale, returned as x + (value − x)). The engines disagree on its arithmetic:

| | What a row reads as |
|---|---|
| transformers, bf16 | every step in bf16; x + (value − x) is rounded twice and misses `value` in 0.041% of rows (66,953 of layer 1's 164,994,380) |
| transformers, fp64 (our oracle) | `value` with the scale and the codes computed exactly |
| vLLM (PR #68296), this release's bf16 table | the bf16 row itself, no round trip; its FP8 path reads code × scale in fp32 |

The engine follows the oracle. The pack computes each row's codes in fp64 offline, then stores the 96 codes and the
fp32 scale fp32(amax / 448): 100 bytes a row, 61.5 GiB in all. The kernel reads code × scale in fp32, which matches the
fp64 round trip to the scale's fp32 rounding. Codes computed in fp32 instead pick a different code than fp64 in about
0.4% of rows (near-ties; one e4m3 step, up to 7.1% of the row's amax), so the pack does not compute them in fp32.
`tools/zig/hy4_engram_check.py` checks a whole table against the fp64 round trip.

**Multipliers** are hard-coded for this release (as `deepseek_v41/engram_hash.zig` does); the loader checks the seed,
the map (max + 1 = 92,620) and every head size, and refuses anything else by name.

| Layer (config id) | Multipliers |
|---|---|
| 2 | 81853575799317, 39690085502883, 24229915018165 |
| 12 | 19457948045719, 44729556510419, 84779352922315 |
| 22 | 72974230057307, 21530602121221, 55998057243097 |
| 32 | 29128509562787, 16745155128523, 22474588302733 |

**Reads.** 64 rows a token (16 heads × 4 layers), random. The rows come from per-rank files on local NVMe through
DeepSeek-V4.1's reader (`families/deepseek_v41/engram_rows_linux.zig`: io_uring, AIO or threads, LRU), whose row
format is fixed at 264-byte FP8 rows of 256 values today. It is generalized to a row geometry read from the file
header rather than copied.

## Design decisions

1. **Attention: absorbed MLA over the latent on every layer.** The cache holds what the reference caches: the normed
   512-wide latent and the 64 roped columns, bf16 (1,152 bytes a token and layer), plus each indexed layer's 128-wide
   token key. A query's 192 nope columns are absorbed through `kv_b`'s key half into the latent; scores add the roped
   part, scale by 256^-0.5, and the head's sink joins the softmax denominator; the weighted latent goes through
   `kv_b`'s value half to 256 a head. One new kernel family serves both kinds, choosing keys by window (512) or by the
   indexer's list (2,048). Reasons: the shared `mla.zig` is fixed at 128-wide heads and `window_attention` has no
   sinks, and expanding K/V for the sliding layers would cost ~1 GiB a stream for no gain.
2. **Sinks** live in that kernel (fp32 logits).
3. **The indexer** is replicated on both ranks (its keys are small; splitting its heads would all-reduce a score for
   every group). Its scoring and selection follow GLM's pooled indexer (`families/glm/cuda_index.zig`,
   `kernels/cuda/glm_index*.cu*`) with Hy4's pooling, group RoPE and own-group rule.
4. **Ties** in a top-k (router: 16 of 512; indexer: 511 groups) go to the higher score, then the lower index, as
   DeepSeek-V4.1's `topk_cuda.cu` already orders them. The fixtures report any position where the reference picked a
   different member of a tie.
5. **Engram at TP=2 and TP=4: whole heads split evenly (8 a rank at TP=2, 4 a rank at TP=4).** Each rank reads its
   heads' rows, norms them, and applies its input columns of `value_proj` and the `key_projs` (`768` at TP=2, `384` at
   TP=4); the partial sums are all-reduced (rank 0 adds the projections' biases once). One contiguous 16-head FP8 pack
   file a layer serves world 1, 2 and 4 by head-range offset (`30.73 GiB` on NVMe at TP=2; `15.37 GiB` in memory or on
   NVMe at TP=4).
6. **TP=2 and TP=4 split elsewhere:** attention heads split evenly (`16` / `32` a rank at TP=2, `8` / `16` a rank at
   TP=4), `q_b` and `linear_gate` by heads (on 128-row blocks), `kv_b` by heads, `o_proj` by input columns with an
   all-reduce, the latent cache whole on every rank; routing and the two `1,536`-latent projections replicated (avoiding
   a third collective per layer); routed experts and the shared expert split inside each expert (`1,536` = 12 blocks of
   128: `768` = 6 blocks at TP=2, `384` = 3 blocks at TP=4) with one all-reduce after the MoE; layer 0's FFN (`13,824`)
   split the same way (`6,912` / `3,456`); embeddings replicated; `lm_head` split by vocabulary (`60,416` / `30,208`).
7. **Precision by op.** fp32: router logits and sigmoid, iHC and `hc_head`, RMSNorm interiors, RoPE, sinks, the
   indexer's pooling and scores, the Engram gate's dot products, lm_head. bf16 storage wherever the reference stores
   bf16 between ops. Matmuls: bf16 activations, e4m3 weights expanded exactly, fp32 sums, the block scale on each
   128-column partial (`cuda/fp8.zig`'s `fp8g` and `cuda/experts.zig`'s fp8g form). The checkpoint's scales are bf16;
   they are widened to fp32 at pack time, which is exact.
8. **Weights as stored.** FP8 codes are repacked for the shared kernels (`fp8.packCodes`, `packScalesAt`); the fused
   expert tensors ([512, 3,072, 1,536] and [512, 1,536, 1,536]) are sliced by expert. Nothing is requantized. A
   projection's format comes from its own tensors, not its role: the checkpoint leaves some projections bf16 per layer
   (layer 0's `q_b_proj` is bf16 [8,192, 1,024], with no scale), so the loader admits bf16 or e4m3 + scale for each.

## Memory at TP=2 and TP=4

Counted by `memory.zig` from the config and the split in `partition.zig`; at one rank its byte totals equal the
release's safetensors headers exactly (the test holds them).

| | A rank at TP=4 | A rank at TP=2 | One rank |
|---|---|---|---|
| Weights without Engram's tables (e4m3 codes, fp32 block scales) | 38.40 GiB | 74.80 GiB | 147.58 GiB |
| Engram rows (e4m3 codes, fp32 row scale: 100 bytes a row) | 15.37 GiB (memory or NVMe) | 30.73 GiB on NVMe | 61.47 GiB |
| Vision encoder (`--vision`, rank 0 only: weights + `16,384`-patch scratch) | 0.87 GiB (rank 0) / 0 B (1..3) | 0.87 GiB (rank 0) / 0 B (rank 1) | 0.87 GiB |
| KV a token (`--kv-cache bf16`: 12 indexed layers × (1,152 B MLA + 64 B group key) + 4 B id) | 14,596 bytes | the same | the same |
| KV a token (`--kv-cache fp8`: 12 indexed layers × (656 B MLA + 64 B group key) + 4 B id) | 8,644 bytes | the same | the same |
| A stream's fixed sliding rings (`30 × 4,608` rows + `32`-row indexer tails) | 152.0 MiB (`bf16`) / 86.6 MiB (`fp8`) | the same | the same |

By default (`TF_HY4_KV_PAGES=1`), all `12` indexed layers' MLA rows and `4:1` `group_keys` share one unified `kv_pool.Pool`
of `256`-token pages (`2.11 MiB/page` in `fp8`, `3.56 MiB/page` in `bf16`, `0.0%` padding; `91 MiB` fixed sliding rings per
stream). With `TF_KV_SHARE=32` (`share_grid = 256`) and `--context 1048576 --parallel 8 --kv-cache fp8 --vision` at safe
`TENSORFOLD_MEMORY_RESERVE_GIB=8`: `TP=2` allocates `10,879` pages (`22.41 GiB`, `2.79M` tokens — up to `12,435` pages /
`3.18M` tokens at `=4`), dynamically serving `2 × 1M` full windows, `5 × 550K` unshared deep sessions, or `8 × 816K`
(`6.53M` logical tokens) shared-prefix sessions; `TP=4` allocates `22,414` pages (`46.17 GiB`, `5.74M` tokens: `5 × 1M` or
`8 × 717K` unshared) with Resident Engram and `29,790` pages (`61.37 GiB`, `7.63M` tokens: `7 × 1M` or `8 × 953K` unshared,
`8 × 1M` shared) with NVMe Engram.

## Reused, extended, new

| Piece | From |
|---|---|
| Lanes engine and backend, lane host, kv_pool, kv_ledger, prompt cache | core (`zig/src/core/lanes`, `kv_pool.zig`, `prompt_cache.zig`) |
| Block-FP8 lane matmuls and experts | `cuda/fp8.zig`, `cuda/qmmf.zig` (`FP8G` `64x128` 2-CTA/SM TMA prompt + `sk=1`-pinned decode + fused `shared_down`), `cuda/experts.zig` (`sm_120+` 1D TMA decode + `K64` 3-stage 1D TMA full-grid prompt `fp8_experts_k64.cuh`) |
| Sigmoid top-k with bias, combine | `hy4_route.cuh` (`tf_hy4_route_m4_plan1` 1-CTA fused router, bit-equal to `cuda/moe_glue.zig`) + `hy4_rows.cu` (`comb_swiglu_bf16`) |
| Norms, embeddings, dense rows, GPU argmax & sampling | `cuda/norms.zig`, `cuda/embed_rows.zig`, `cuda/dense_rows.zig`, `hy4_rows.cu` (`hy4_rows_argmax`, `hy4_rows_f32_to_bf16`, `hy4_rows_interleave_bf16`), `cuda_sample.zig` (1-collective BF16 gather + `sample.cu` GPU draw for `temperature > 0`) |
| Engram reader | `families/deepseek_v41/engram_rows*.zig` (hybrid `1024`-SQE `io_uring` for decode + `262K`-row/layer LRU + merged multi-stream `prefetchWindows` + lazy background thread pool for `>512`-extent bulk prefill batches + early slot 0..3 prefetch) |
| iHC | `hy4_ihc.cu` (`R <= 64` 1-launch ticket/coop boundary + 1-pass tiled prefill, exact stock single-rounding) |
| Indexer | new kernels after GLM's pooled indexer (`cuda_index.zig` lossless 32-row rewind-safe `runTail` ring + `512`-row `Scratch` sub-tiling + batched multi-stream `preproject` + `sm_90+` PDL + `tf_hy4_idx_q_mrope` 3D M-RoPE + `256`-token `_paged` pool/score) |
| Absorbed MLA with window or list and sinks | `hy4_mla_attn.cu` + `hy4_mla_x16_qreg.cuh` (low-spill 2-CTA/SM FP32-exact `x16_qreg` / `x16_qreg_rows` / `x16_qreg_paged` across both `bf16` `1,152 B` and `--kv-cache fp8` `656 B` rows via `hy4_rows_kv_write_ex` / `_paged`) + `hy4_mla_c2.cu` |
| Vision encoder (`--vision`) | `core/picture` (`decode` + bicubic `smartResize`), `cuda/vit.zig`, `hy4/vision_config.zig`, `hy4/cuda_vision.zig`, `hy4/cuda_media.zig`, `hy4_rows.cu` (`vit_qkv`, `vit_merge_gather`, `vit_append_newline`, `splice_streams`, `q_rope_mrope`) |
| Built-in 5-layer `DFly2` drafter & CUDA Graphs | `hy4_dfly2_glue.cu` + `cuda_dfly*.zig` (reuses target paged/unpaged MLA caches `[25, 29, 33, 37, 41]` + saved sparse top-k; dense `<2K` & sparse `>=2K` `DFly2` CUDA Graphs, `2043..2048` seam guard, and merged post-attn/SWA CUDA Graphs) |
| Paged `TF_KV_SHARE` & NVMe conversation spill | `hy4/cuda_pages.zig`, `core/prompt_radix_disk.zig`, `hy4/cuda_disk_pages.zig` (`256`-token zero-copy page sharing + `Blob` ring tails, `256 MiB` piecewise NVMe write-behind, eviction spill, `Op.snap_load` / `Op.snap_load_share` ring `ncclBroadcast` recall), `TF_HY4_REQUEST_STARTS`, `TF_CACHE_EVICT=decayed_resume` |
| Template, tokenizer, tool calls, grammar | `core/template`, `core/tokenizer`, `server/tool_hy4.zig` (`HYTK`), `libtfgrammar.so` (`xgrammar` + `vtable.held`) |

## Reference fixtures

`tools/zig/hy4_reference.py` writes the oracle each layer kind is checked against. It runs transformers' own
`HYV4FlashDecoderLayer` (PR #50903 at the commit above) one layer at a time on the CPU:

```sh
python -I tools/zig/hy4_reference.py MODEL OUT --layers 0,1,2,3,4,5 --tokens 2560 --text FILE --threads 12
```

- Each chosen layer gets one input: the previous chosen layer's fp64 output rounded to bf16 (the embeddings for the
  first), as the engine's bf16 streams hold it. It runs twice on it: in fp64 (the reference keeps its own fp32 casts in
  the norms, iHC and router) and in bf16 with transformers' fp32 modules kept fp32.
- FP8 weights are dequantized exactly (code × scale_inv in fp64, then the run's dtype). Routed experts are dequantized
  only when picked and Engram rows are read by memory map, so a layer fits in ~16 GB of memory. A self-check at start
  holds both substitutions bit-equal to the PR's modules.
- `MODEL` may be a part of the checkpoint holding just the chosen layers. `OUT/layer<i>_t<T>.safetensors` holds `in`,
  `out64`, `out16` and taps with the same suffixes: `attn_hc_in`, `attn_hc_post`, `attn_hc_out`, `attn_in`, `attn_out`,
  `mid`, `ffn_hc_post`, `ffn_hc_out`, `mlp_out`, `router_ids`, `router_w`, `recall` (indexed layers past 2,048 keys),
  `engram_out`; `OUT/head_t<T>.safetensors` the final streams through `hc_head`, the norm and the fp32 lm_head;
  `OUT/ids_t<T>.json` the prompt.
- 2,560 tokens put a sliding layer past its 512-key window and send rows 2,048 and later through the indexer's
  selection.

## Build and multi-rank transport

- **Recommended DGX Spark build (`nvcc >= 13.2`):**
  `zig build -Dgpu=spark -Dnvcc=/usr/local/cuda/bin/nvcc -Dnative-fp-decode`.
  Compiles the three FP8 kernels (`fp8_experts`, `qmmf`, `prompt16`) with Blackwell hardware FP8 pair-decode instead of
  the software LUT (`tf-cuda-test native-fp` verifies bit-identical outputs across finite pairs). Code default remains
  off for compatibility; refused on HIP and with `-Dfatbins`.
- **Recommended serve config (1M context):**
  Set `TF_KV_SHARE=32`, `TENSORFOLD_MEMORY_RESERVE_GIB=8`, ConnectX-7 `NCCL_*` / `TF_TP_ROCE_*` vars below, and on 4×
  Spark `TF_HY4_ENGRAM_MODE=resident` (`5.74M` KV pool; omit for NVMe / `7.63M`; 2× always NVMe / `2.79M`–`3.18M`).
  Launch each rank with `--tp 4 --rank <0..3> --master <RANK0_IP>` (or `--tp 2 --rank <0|1>`) plus:
  `--precision full --context 1048576 --parallel 8 --kv-cache fp8 --vision --learn-conversations`.
- **Fallback build for `nvcc 13.0` / `13.1` (or fast fatbin iteration):**
  Omit `-Dnative-fp-decode` (`zig build -Dgpu=spark -Dnvcc=/usr/local/cuda/bin/nvcc`), or prebuild kernel fatbins once
  (`zig build fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121`) plus `tp_mailbox.fatbin`, then iterate with
  `zig build native -Dfatbins=/path/to/fatbins -Doptimize=ReleaseFast` and set
  `TF_TP_KERNEL_IMAGE=/path/to/fatbins/tp_mailbox.fatbin` so `tp_roce` stays active without recompiling `.cu` units.
- Set identical `TENSORFOLD_CHECKPOINT_SHA256` across ranks to skip hashing all 267.4 GiB on cold unified memory.
- On dual-function ConnectX-7 links, assign both PCIe x8 RDMA functions on shared `/24` subnets and set
  `NCCL_IB_HCA==rocep1s0f1:1,roceP2p1s0f1:1`, `NCCL_IB_ADDR_RANGE=<SUBNET>/23` (covering both fabric `/24` subnets),
  `NCCL_PROTO=Simple`, `TF_TP_ROCE_FAST=1`, and `TF_TP_MAILBOX_MAX_KB=1024` (covering batched/DFly2 windows; add
  `TF_TP_ROCE_RELAY=1` and a ring `NCCL_GRAPH_FILE` on switchless 4-rank CX7 rings; Hy4 defaults the relay ceiling
  to `512 KiB` (`TF_TP_ROCE_RELAY_MAX_KB`) so verify windows of 6 rows and more stay off NCCL's ~290 us path).
- Seam reductions (`cuda_tp.zig` `sumRankOrder`) use one-step `allReduce` at `world = 2` (`0 B` `gather_buf`),
  `RankScatter.run` + `ncclAllGather` on full-mesh `world = 4`, and `tp_roce` multi-rank relay (up to the relay
  ceiling) plus ring `ncclAllGather` + `tf_rank_sum` (above it, 0-bit-exact rank-0-first `fp32` summation) on
  switchless 4-rank rings; NVMe snapshot recall (`Op.snap_load`) streams from rank 0 via ring `ncclBroadcast`.
- Decode (`R <= 64`) row-splits the replicated `qkv_a` and `latent_down` over the ranks at any world whose shapes
  divide and gathers the parts (`TF_HY4_SHARD_REP=0`: whole on every rank). At `world = 4` that is 168 collectives a
  token instead of 85 and 0.96 ms less (26.56 -> 25.60 ms, one stream). The scheduler prices windows with a `world = 4`
  cost table there (verify 25.9 ms at 1 row, 70.1 at 16; `world = 2`: 38 and 128).

## Checks

- Host: `zig build test-hy4-host` (config, tensor map against the real headers, Engram hash against golden ids,
  partition and memory floor at world 1, 2 and 4), then the repository's no-GPU set and `tools/zig/lean_check.py` with
  no new findings.
- Each layer kind against the fp64 reference on a GB10 (`tf-cuda-test hy4-<kind> MODEL FIXTURE`, `-Dsm=121`): error no
  worse than the bf16 reference's, picked experts and recalled groups equal, five byte-identical repeats. Every new
  kernel goes through the CUDA contract runner.
- Row, chunk, sparse-seam and multi-stream batch invariance (`tf-cuda-test hy4-invariance MODEL FIXTURE_DIR`): zero bit
  differences across `R ∈ {1, 4, 5, 6, 16, 17, 24, 512, 1024, 2560}`, across the `2,048`-key live indexer seam, and
  between 4-stream `24`-row batched `segs` (`tf_hy4_mla_attn_rows`), `6`-row windows, and `1`-row plain decode at
  `world ∈ {1, 2, 4}`.
- Template, tokenizer and tool-call parity (`test-tool-hy4`) with the Python reference.
- Whole-model 42-layer fidelity and engine contracts at TP=2 and TP=4: note that `hy4_reference.py` resets
  `x = y64.to(bf16)` after each layer (so `head_t<T>.safetensors`'s `logits16` measures head-only bf16 on `layer41`'s
  fp64-chained state); chained 42-layer serving must achieve top-5 logprob `max_abs` mean no worse than the 42-layer
  chained bf16 reference (`~0.63`), `1-GPU virtual world == multi-process TP` bit parity, `drafted == plain`,
  `resumed == fresh`, `concurrent == solo`, clean cancel/reset, and reproducible `bench_openai.py` /
  `bench_concurrent.py` / `bench_cache.py` receipts.

## Measured throughput and fidelity on GB10 (`sm_121`)

42-layer `tencent/Hy4-flash-FP8` (`--precision full`, `--context 32768`, `NCCL_PROTO=Simple`, `TF_TP_ROCE_FAST=1`,
`TF_TP_MAILBOX_MAX_KB=1024`), dual-port ConnectX-7 RoCE (direct link at `TP=2`, switchless 4-node ring at `TP=4`):

| Configuration | Weights + Engram / rank | Free after reserve | 42L `ids_t64` top-1 (`max_abs` mean) | 1s Decode `DFly2` / Plain (tok/s) | 4s Decode `DFly2` / Plain (tok/s) | 8s Decode `DFly2` / Plain (tok/s) | `bench_concurrent` 8s `code` / `chat` (steady) | Cold Prefill `2K` / `8K` / `32K` (tok/s) |
|---|---|---|---|---|---|---|---|---|
| `TP=2` (`2x GB10`, NVMe Engram) | `79.94 GiB` + `30.73 GiB` NVMe | `24.7 GiB` (`4.0 GiB` cache, `8` streams) | `62/64` (`acc_multi` `0.6089`, `372/383`) | `43.62` (`0.141 s` TTFT) / `27.42` | `70.01` (`32.48 r/s`) / `62.26` | `92.15` (`47.51 r/s`) / `89.59` | `207.5` (`243.5`) / `205.6` (`229.6`) | `1,936` / `2,088` / `2,090` |
| `TP=4` (`4x GB10`, NVMe Engram) | `42.98 GiB` + `15.37 GiB` NVMe | `61.4 GiB` (`29,790` pages, `7.63M` tok) | `62/64` (`0.8196`) | `55.66` (`0.099 s` TTFT) / `37.19` | `107.15` / `95.02` | `118.14` / `116.93` | `216.6` (`255.2`) / `225.8` (`252.3`) | `1,876` / `1,964` / `1,946` |
| `TP=4` (`4x GB10`, Resident Engram) | `42.98 GiB` + `15.37 GiB` RAM | `46.2 GiB` (`22,414` pages, `5.74M` tok) | `62/64` (`0.8196`) | `57.96` (`0.093 s` TTFT) / `37.53` | `110.39` / `95.20` | `123.49` / `119.84` | `207.5` / `221.7` | `1,910` / `1,969` / `1,945` |

- `bench_openai.py` (128 tok, 3-rep median): at `--context 32768`, `TP=2` `t=0` hits `34.70 tok/s` (`0.105 s` TTFT,
  `fibonacci-raw`) and `36.18 tok/s` (`0.075 s` TTFT, `gpu-chat-no-think`), `t=0.9` `31.55` and `34.94 tok/s`; under full
  `1M` paged serving (`--context 1048576 --kv-cache fp8 --vision --learn-conversations`, `TF_KV_SHARE=32`, MTP + copy drafts),
  `TP=2` hits `82.60 tok/s` (`0.166 s` TTFT, `fibonacci-raw` `t=0`), `34.50 tok/s` (`gpu-chat-no-think` `t=0`), and `34.74` /
  `35.20 tok/s` at `t=0.9`, while `TP=4` (`nvme`) hits `55.93` / `49.29 tok/s` (`t=0`, `37.3 ms` p50 gap) and `47.34` /
  `46.48 tok/s` (`t=0.9`, `38.2 ms` p50 gap).
- `bench_concurrent.py` (`--levels 1,4,8 --alone --serial`): `0` failures and `100%` `alone` (`156/156`) and `serial`
  (`54/54`, `drafted == plain`) token-for-token equality across all `12/12` cells at `t=0` and `t=0.9` on both `TP=2` and
  `TP=4`. Note that default `t=0` cells send one shared prompt (`16` MoE experts across all `8` streams: `TP=2` `code`
  `326.7`, `chat` `175.3`; `TP=4` `code` `216.6`, `chat` `225.8`), whereas `t=0.9` (`seed + i` diverging trajectories:
  `TP=2` `code` `90.0`–`92.8`, `chat` `99.1`–`107.9`; `TP=4` `nvme` `code` `123.7`, `chat` `141.5`) and the `8`-essay
  distinct prose benchmark (`92.2` `TP=2`, `118.1` `TP=4` `nvme`, `123.5` `TP=4` `resident`) measure true distinct-expert
  routing.
- Long-context & sparse `>= 2K` `DFly2` (`TP=2`, `--context 524288`, `512`-row `cuda_index.Scratch` sub-tiling frees
  `2.28 GiB/rank`): `2,043..2,048` seam crossings (solo, mixed 2-stream, and short-suffix `2043 -> 2052` resumed turns)
  pass `drafted == plain` and `concurrent == solo` at `t=0` and `t=0.9`; sparse `DFly2` CUDA Graphs bring `13.7K`-ctx
  rounds down to `75.3 ms/r` (`t=0`) and `78.5 ms/r` (`t=0.9`).
- `--kv-cache fp8` fidelity & `8 × ~500K` (`3.99M`-token) uncached prose residency: across `218` matched-prefix steps on
  `6` classic novels (`2.6K`–`46.1K` tokens), `fp8` KV matches `bf16` greedy top-1 on `214/218` (`98.17%`) with mean
  `KL = 0.0089` nats/tok (`+0.90%` PPL, median `0.000089`). On `TP=4` (`--parallel 8 --vision --kv-cache fp8 --context 524288`),
  `8` disjoint classic-novel manuscripts (`471K`–`509K` cold tokens each, `3,985,219` simultaneous KV tokens, `cached = 0`
  on all `8`) complete concurrently at `81.50 GiB` peak device memory/rank with `24/24` (`100%`) multi-depth marker recall.
- `--vision` multimodal serving (`TP=2` and `TP=4`): `4,104`-token high-res chart OCR (`pretrained-model-bpb.png`) and
  multi-turn visual Q&A produce identical outputs across `TP=2` and `TP=4` (`bf16` and `fp8` KV).
- Paged `TF_KV_SHARE=32` + NVMe SSD write-behind/recall (`--learn-conversations`, `share_grid = 256`): warm multi-turn
  resume, 4-way shared-prefix branching, cold-restart `Op.snap_load_share` NVMe recall (`24`–`35 ms` disk read for `81 MiB` /
  `8,704` tok, `0.903 s` e2e on `TP=2`, `0.703 s` 4-rank ring `ncclBroadcast` recall on `TP=4`), and unpaged
  `TF_HY4_KV_PAGES=0` produce bit-for-bit identical greedy replies. On `2× GB10` (`TP=2`, `10,879` pages), `8` concurrent
  `523,867`-token deep sessions (`4.19M` logical KV tokens) complete in `33.70 s` (`16/16` needles), and `8` concurrent
  `816,058`-token deep sessions (`6,528,146` logical KV tokens, `810,752` shared + private tails) complete in `48.29 s` using
  `7.60 GiB` (`3,626` pages, `14.8 GiB` still free) with `16/16` (`100%`) multi-depth needle recall and `13` deduplicated
  `816K` states (`7,805 MiB`) persisted to NVMe SSD (alongside `#14` sub-2048 `warm_turns`, `TF_HY4_REQUEST_STARTS=1`, and
  `TF_CACHE_EVICT=decayed_resume`).
- CUDA kernel contract runner (`tf-cuda-test hy4-wave1-contract` + `hy4-index-contract` + `hy4-invariance` + `hy4-tail-kv8`
  + `hy4-vit`): `12/12` `hy4-wave1` units + `9/9` planted controls caught, `10/10` `hy4-index` units + `41/41` planted
  controls caught, `0` live `sm_121` Gate 5 limit problems (`nvcc 13.0`, including all `13` `_paged` launches), `9/9`
  `hy4-invariance` checks pass with `0` bit differences across `world ∈ {1, 2, 4}`, scrambled `256`-token `page_table` matches
  contiguous `runTail` + `bf16`/`fp8` MLA with `0` bit diffs, `hy4-tail-kv8` layer cosine `0.99998`, and `hy4-vit` matches
  PyTorch `fp32` (`cos = 1.00000`, `rmse = 0.00304`).

### 2026-11-23 update (recommended 1M Spark setup, distinct-prompt matrix & 128K/256K prose)

Measured in the recommended `1M` Spark serving config (`--precision full --context 1048576 --parallel 8 --kv-cache fp8 --vision --learn-conversations`,
`TF_KV_SHARE=32`, `512 KiB` default Hy4 relay ceiling, `TF_HY4_SHARD_REP=1`, `world = 4` draft cost table, and the
recommended `-Dnative-fp-decode` Spark build on GB10 `sm_121`). All multi-stream throughput numbers below use distinct prompts (`cached = 0`):

| Workload (`cached = 0`, recommended `1M` FP8 setup) | `TP=4` (`v0.1.0-hy4` -> today, Resident Engram, `5.74M` KV pool) | `TP=2` (today, NVMe Engram, `2.79M` KV pool) | Notes |
|---|---|---|---|
| `tools/bench_prose.py` (`350tok_prose`, 8 distinct novels) | `116.6` -> **`139.5 tok/s`** (`+19.6%`; plain `113.7` -> `134.7`, 4s `86.3` -> `97.2`, 1s plain `38.0` -> `40.6`) | `83.8` (`84.3` plain, 4s `59.8`, 1s `32.1` / `26.6` plain) | `32K` BF16 launch config (`-Dnative-fp-decode`, 3-rep median): `scoreboard.py` 1s `58.0` -> **`64.5 tok/s`** (`+11.2%`, `64.57/64.47/63.94`; plain `37.5` -> `41.9`), `tools/bench_prose.py` 8s `123.5` -> **`139.7 tok/s`** (`+13.1%`, `139.66/144.00/139.13`) |
| `tools/bench_prose.py` (`2.5Ktok_chapter`, 8 distinct novels) | `100.5` -> **`120.7 tok/s`** (`+20.1%`; 4s `79.8` -> `91.5`, 1s `48.5` -> `49.8`) | — | `32K` BF16 launch config: `102.1` -> **`121.0 tok/s`** (`+18.6%`) |
| `tools/bench_concurrent.py --prompts tools/bench_prompts.json` (`t=1.0`, distinct; 1s is `quicksort-raw` Python code) | 1s `80.7` -> **`85.8`**, 4s `103.6` -> **`114.7`** (`145.9` steady), 8s `128.0` -> **`142.9 tok/s`** (`+11.6%`, `187.4` steady) | — | `32K` BF16 launch config: 1s `74.0` -> `80.9`, 4s `103.0` -> `122.0`, 8s `122.5` -> `133.7` |
| `tools/bench_concurrent.py --prompts tools/bench_prompts.json` (`t=0.0`, distinct) | 1s `54.5` -> **`55.7`**, 4s `100.3` -> **`115.1`** (`+14.8%`), 8s `122.5` -> **`137.4 tok/s`** (`+12.2%`, `177.1` steady) | — | `32K` BF16 launch config: 1s `82.5` -> `89.7`, 4s `104.4` -> `117.3`, 8s `121.8` -> `136.8` |
| Cold `32K` prefill (recommended `1M` FP8 setup) | `1,863–1,864 tok/s` | **`1,893.5–1,902.5 tok/s`** | `TP=2` `1M` FP8 `32K`: `1,847–1,855` -> `1,893.5–1,902.5` (`+2.4%`) |
| Real prose `128K` (first `134,276` tok of *War and Peace*) | `1,824.6 tok/s` (`73.59 s`) -> `49.54` drafted / `38.80` plain | `1,853.5 tok/s` (`72.45 s`) -> `32.70` drafted / `26.20` plain | Plain decode retains `93.5%` (`TP=4`) and `95.4%` (`TP=2`) of `0K` plain |
| Real prose `256K` (first `268,252` tok of *Les Misérables*) | `1,780.1 tok/s` (`150.70 s`) -> `42.30` drafted / `38.17` plain | `1,804.8 tok/s` (`148.63 s`) -> `28.52` drafted / `26.22` plain | Plain decode retains `92.0%` (`TP=4`) and `95.5%` (`TP=2`) of `0K` plain; drafted acceptance tapers with context length |

- **4-rank ring relay (`tf-tp-test suite`):** eager half-forwarding cuts 4 KB all-gather `21.10 -> 14.10 us` (`-33.2%`),
  8 KB `22.08 -> 17.58 us` (`-20.4%`), and `decode_window` `22.12 -> 17.70 us` (`-20.0%`), `328/328` checks pass.
- **Exactness:** bit-identical greedy outputs and 42-layer FP64 `ids_t64` logits across SW vs `-Dnative-fp-decode` and
  `TF_HY4_SHARD_REP=0` vs `1`; `bench_concurrent.py` `12/12` cells, `0` fail, `0` unequal (`156/156` alone, `54/54`
  serial) at both `TP=2` and `TP=4`; all five Hy4 CUDA contracts plus `hy4-tail-kv8` and `native-fp` pass.
- **Fallback build check (`nvcc 13.0/13.1`, without `-Dnative-fp-decode`, exact Nov 21 `--context 32768` `bf16` KV launch config on
  `TP=4` Resident Engram):** `scoreboard.py` solo `57.96 -> 62.64 tok/s` (`+8.1%`) drafted / `37.53 -> 41.71 tok/s`
  (`+11.1%`) plain, `4`-stream `110.39 -> 145.83 tok/s`, cold `32K` prefill `1,945 -> 1,967 tok/s` (`2K` `1,853`, `8K`
  `1,956`); `real_prose_bench.py` `8` distinct novels `123.49 -> 138.18 tok/s` (`+11.9%`, `138.34` plain). Against
  `v0.1.0-hy4`, `3/3` greedy prompts and a `3,849`-token multi-turn prose session are bit-identical and share the exact
  same on-disk NVMe conversation directory format.
