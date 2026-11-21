# Hy4-flash Engram rows: e4m3 codes (fp64 arithmetic) + an fp32 scale against the exact FP8 round trip, every row.
import argparse, json, os, struct, time

import numpy as np
import torch

ap = argparse.ArgumentParser(description="Check a checkpoint's Engram table against its packed FP8 row form, every row.")
ap.add_argument("model", help="checkpoint folder (whole or a part holding the table)")
ap.add_argument("--layer", type=int, default=1, help="0-based layer of the table")
ap.add_argument("--chunk", type=int, default=1 << 21, help="rows a step")
a = ap.parse_args()

name = f"model.layers.{a.layer}.engram_layer.multi_head_embedding.embedding.weight"
shard = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"][name]
path = os.path.join(a.model, shard)
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    h = json.loads(f.read(n))[name]
rows, dim = h["shape"]
table = np.memmap(path, dtype=np.int16, mode="r", offset=8 + n + h["data_offsets"][0], shape=(rows, dim))
fp8max = torch.finfo(torch.float8_e4m3fn).max
e4m3 = torch.float8_e4m3fn


def codes(x, s):
    """e4m3 of x / s, clamped: fp64 (or fp32) to fp32 to e4m3 rounds once (24 bits >= 2 x 4 + 2)."""
    return (x / s).clamp(-fp8max, fp8max).float().to(e4m3)


fp32_codes_differ = bf16_off = zero = 0
worst = 0.0  # largest |packed - exact| / amax
t0 = time.time()
for lo in range(0, rows, a.chunk):
    x = torch.from_numpy(np.array(table[lo:lo + a.chunk])).view(torch.bfloat16)
    # exact: transformers' _fp8_round_trip in fp64 (x + (q - x) is q there, to fp64 rounding)
    x64 = x.double()
    amax = x64.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    s64 = amax / fp8max
    c64 = codes(x64, s64)
    q64 = c64.double() * s64
    # the pack: fp64 codes and the scale rounded to fp32; the kernel reads code x scale in fp32
    s32 = s64.float()
    q = c64.float() * s32
    worst = max(worst, float(((q.double() - q64).abs() / amax).max()))
    # for the record: codes computed in fp32 instead, and transformers' bf16 arithmetic
    x32 = x.float()
    c32 = codes(x32, x32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / fp8max)
    fp32_codes_differ += int((c32.view(torch.uint8) != c64.view(torch.uint8)).any(dim=-1).sum())
    sb = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / fp8max
    qb = (x / sb).clamp(-fp8max, fp8max).to(e4m3).to(x.dtype) * sb
    bf16_off += int(((x + (qb - x)).view(torch.int16) != qb.view(torch.int16)).any(dim=-1).sum())
    zero += int((x.view(torch.int16) == 0).all(dim=-1).sum())
    if lo // a.chunk % 16 == 0:
        print(f"{lo + len(x):>11,} / {rows:,} rows, {time.time() - t0:.0f} s, worst {worst:.3e}", flush=True)
print(f"layer {a.layer}: {rows:,} rows of {dim}; packed vs exact: worst |difference| / amax {worst:.3e}; fp32-arithmetic "
      f"codes differ in {fp32_codes_differ} rows; bf16 x + (q - x) != q in {bf16_off} rows; all-zero rows {zero}", flush=True)
