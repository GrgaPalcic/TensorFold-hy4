// Hy4-flash's lm_head and router at 5 or more rows: tf_dense_rows' sums, each x load shared by a warp's C weight rows.

// A warp C output columns, RT rows a launch tile; each (row, column) sums as tf_dense_rows_1 does, bit for bit.

#include <cuda_bf16.h>
#include <stdint.h>
#include "kv_fp8.cuh"

namespace tf_hy4_rows {

constexpr int WARPS = 8;
constexpr int VEC = 8; // bf16 a 16-byte load

__device__ __forceinline__ void unpack(const uint4& u, float* f) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __bfloat1622float2(h[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

// A weight row is read once: no L1 line for it, so the tile's x rows stay in L1 for the CTA's other warps.
__device__ __forceinline__ uint4 stream(const __nv_bfloat16* p) {
    uint4 v;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(p));
    return v;
}

// lane l adds k = 8 l + 256 j in order, then the fixed shuffle tree; CONTROL 1-4: drop NaN, drift, skip a row, fp16.
template <int RT, int C, int CONTROL>
__device__ __forceinline__ void tile(const __nv_bfloat16* x, int64_t x_stride, const __nv_bfloat16* w, float* out,
                                     int64_t out_stride, int rows, int n, int k, int repeat) {
    const int lane = threadIdx.x & 31;
    const int col0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * C;
    const int r0 = blockIdx.y * RT;
    if (col0 >= n) return;
    const int nr = min(RT, rows - r0) - (CONTROL == 3 && r0 + RT >= rows ? 1 : 0);
    const __nv_bfloat16* wr[C];
#pragma unroll
    for (int c = 0; c < C; ++c) wr[c] = w + static_cast<int64_t>(min(col0 + c, n - 1)) * k;
    float acc[RT][C];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) acc[r][c] = 0.f;
#pragma unroll 2
    for (int k0 = lane * VEC; k0 < k; k0 += 32 * VEC) {
        float wf[C][VEC];
#pragma unroll
        for (int c = 0; c < C; ++c) unpack(stream(wr[c] + k0), wf[c]);
#pragma unroll
        for (int r = 0; r < RT; ++r) {
            if (r >= nr) break;
            float xf[VEC];
            unpack(*reinterpret_cast<const uint4*>(x + (r0 + r) * x_stride + k0), xf);
#pragma unroll
            for (int c = 0; c < C; ++c)
#pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    acc[r][c] = __fmaf_rn(xf[i], wf[c][i], acc[r][c]);
                    if constexpr (CONTROL == 4) acc[r][c] = __half2float(__float2half_rn(acc[r][c]));
                }
        }
    }
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float v = acc[r][c];
            for (int m = 16; m > 0; m >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, m));
            if constexpr (CONTROL == 1) v = v != v ? 0.f : v;
            if constexpr (CONTROL == 2) v = __fadd_rn(v, static_cast<float>(repeat));
            if (lane == 0 && r < nr && col0 + c < n) out[(r0 + r) * out_stride + col0 + c] = v;
        }
}

}  // namespace tf_hy4_rows

using namespace tf_hy4_rows;

#define HY4_ROWS(NAME, RT, C, CONTROL)                                                                                 \
    extern "C" __global__ void __launch_bounds__(WARPS * 32) NAME(const __nv_bfloat16* x, long long x_stride,          \
                                                                  const __nv_bfloat16* w, float* out,                  \
                                                                  long long out_stride, int rows, int n, int k,        \
                                                                  int repeat) {                                        \
        tile<RT, C, CONTROL>(x, x_stride, w, out, out_stride, rows, n, k, repeat);                                     \
    }

// wide: 16 rows by 4 columns a warp (lm_head, prompt routers); narrow: 4 rows by 2 columns (a decode round's router)
HY4_ROWS(hy4_rows_wide, 16, 4, 0)
HY4_ROWS(hy4_rows_wide_bad_nan, 16, 4, 1)
HY4_ROWS(hy4_rows_wide_bad_repeat, 16, 4, 2)
HY4_ROWS(hy4_rows_wide_bad_tail, 16, 4, 3)
HY4_ROWS(hy4_rows_wide_bad_sums, 16, 4, 4)
HY4_ROWS(hy4_rows_narrow, 4, 2, 0)
HY4_ROWS(hy4_rows_narrow_bad_nan, 4, 2, 1)
HY4_ROWS(hy4_rows_narrow_bad_repeat, 4, 2, 2)
HY4_ROWS(hy4_rows_narrow_bad_tail, 4, 2, 3)
HY4_ROWS(hy4_rows_narrow_bad_sums, 4, 2, 4)

// Row-wise fp32 argmax `{float_bits(max_v), global_id}` (lowest `global_id` tie-break; non-finite sets NaN).
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_argmax(const float* __restrict__ logits, int vocab_local, int vocab_lo, int rows,
                    unsigned int* __restrict__ out) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float* row = logits + static_cast<size_t>(r) * vocab_local;
    float best_v = -INFINITY;
    int best_id = 0x7fffffff;
    int bad = 0;
    auto consider = [&](float v, int id) {
        if (!isfinite(v)) bad = 1;
        else if (v > best_v || (v == best_v && id < best_id)) {
            best_v = v;
            best_id = id;
        }
    };
    if ((vocab_local & 3) == 0 && (reinterpret_cast<uintptr_t>(row) & 15u) == 0) {
        const int n4 = vocab_local >> 2;
        const float4* row4 = reinterpret_cast<const float4*>(row);
        for (int i = tid; i < n4; i += 256) {
            const float4 v = row4[i];
            const int base = i << 2;
            consider(v.x, base);
            consider(v.y, base + 1);
            consider(v.z, base + 2);
            consider(v.w, base + 3);
        }
    } else {
        for (int i = tid; i < vocab_local; i += 256) consider(row[i], i);
    }
    bad = __any_sync(0xFFFFFFFFu, bad);
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFFu, best_v, m);
        const int oid = __shfl_xor_sync(0xFFFFFFFFu, best_id, m);
        if (ov > best_v || (ov == best_v && oid < best_id)) {
            best_v = ov;
            best_id = oid;
        }
    }
    __shared__ float sv[8];
    __shared__ int sid[8];
    __shared__ int sbad[8];
    if (lane == 0) {
        sv[warp] = best_v;
        sid[warp] = best_id;
        sbad[warp] = bad;
    }
    __syncthreads();
    if (tid == 0) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        int any_bad = 0;
        for (int w = 0; w < 8; ++w) {
            any_bad |= sbad[w];
            if (sv[w] > bv || (sv[w] == bv && sid[w] < bi)) {
                bv = sv[w];
                bi = sid[w];
            }
        }
        out[2 * r] = any_bad ? 0x7fc00000u : __float_as_uint(bv);
        out[2 * r + 1] = static_cast<unsigned int>(vocab_lo + bi);
    }
}

// Extract leading `dst_u4` uint4s per row from strided `src` [rows, src_u4] into contiguous `dst` [rows, dst_u4].
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_slice_bf16(const uint4* __restrict__ src, int src_u4, uint4* __restrict__ dst, int dst_u4, int rows) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(rows) * dst_u4;
    if (idx >= total) return;
    const int r = static_cast<int>(idx / dst_u4);
    const int c = static_cast<int>(idx % dst_u4);
    dst[idx] = src[static_cast<long long>(r) * src_u4 + c];
}

// Interleave world=2 allGather output `src` [2, rows, half_u4] into `dst` [rows, 2 * half_u4].
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_interleave2_bf16(const uint4* __restrict__ src, uint4* __restrict__ dst, int half_u4, int rows) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long plane = static_cast<long long>(rows) * half_u4;
    if (idx >= plane * 2) return;
    const int r = static_cast<int>(idx / (2 * half_u4));
    const int c = static_cast<int>(idx % (2 * half_u4));
    const int rk = c >= half_u4 ? 1 : 0;
    const int lc = rk ? (c - half_u4) : c;
    dst[idx] = src[static_cast<long long>(rk) * plane + static_cast<long long>(r) * half_u4 + lc];
}

// Extract `dst_f4` float4s per row at `col_off_f4` from strided `src` `[rows, src_f4]` into `dst` `[rows, dst_f4]`.
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_slice_f32(const float4* __restrict__ src, int src_f4, int col_off_f4, float4* __restrict__ dst, int dst_f4,
                       int rows) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(rows) * dst_f4;
    if (idx >= total) return;
    const int r = static_cast<int>(idx / dst_f4);
    const int c = static_cast<int>(idx % dst_f4);
    dst[idx] = src[static_cast<long long>(r) * src_f4 + col_off_f4 + c];
}

// Fused world=2 `latent_up` combine reduction (`[0, half)`) + shared SwiGLU (`[half, +sw)`) into bf16 `out`.
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_comb_swiglu_bf16(const float* __restrict__ comb_f32, int comb_stride, int my_col,
                              const float* __restrict__ peer_f32, const __nv_bfloat16* __restrict__ gu, int gu_stride,
                              int gate_off, int up_off, __nv_bfloat16* __restrict__ out, int half, int sw, int rows,
                              int rank) {
    const int width = half + sw;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(rows) * width;
    if (idx >= total) return;
    const int r = static_cast<int>(idx / width);
    const int col = static_cast<int>(idx % width);
    if (col < half) {
        const float mine = comb_f32[static_cast<long long>(r) * comb_stride + my_col + col];
        const float peer = peer_f32[static_cast<long long>(r) * half + col];
        const float sum = (rank == 0) ? __fadd_rn(mine, peer) : __fadd_rn(peer, mine);
        out[idx] = __float2bfloat16_rn(sum);
    } else {
        const int c = col - half;
        const float g = __bfloat162float(gu[static_cast<long long>(r) * gu_stride + gate_off + c]);
        const float u = __bfloat162float(gu[static_cast<long long>(r) * gu_stride + up_off + c]);
        const float sig = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g)));
        out[idx] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(g, sig), u));
    }
}

// Fused batched `kv_a_norm` (latent=512) + `k_rope` (64 cols) for single-stream or `caches[r]` with ring wrap.
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_kv_write(const __nv_bfloat16* __restrict__ qa, int qa_stride, int q_lora,
                      const __nv_bfloat16* __restrict__ kv_norm_w, const float* __restrict__ freqs,
                      const int* __restrict__ pos, const unsigned long long* __restrict__ caches,
                      __nv_bfloat16* __restrict__ single_cache, int ring, float eps) {
    __shared__ float red[8];
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    const int lane = t & 31, wid = t >> 5;
    const int p = pos[r];
    const size_t slot = (ring > 0) ? static_cast<size_t>(p % ring) : static_cast<size_t>(p);
    __nv_bfloat16* dst = (caches != nullptr ? reinterpret_cast<__nv_bfloat16*>(caches[r]) : single_cache) + slot * 576;
    const __nv_bfloat16* src = qa + static_cast<size_t>(r) * qa_stride + q_lora;

    const float v0 = __bfloat162float(src[t]);
    const float v1 = __bfloat162float(src[t + 256]);
    float acc = __fmaf_rn(v0, v0, 0.0f);
    acc = __fmaf_rn(v1, v1, acc);
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xFFFFFFFFu, acc, m));
    if (lane == 0) red[wid] = acc;
    __syncthreads();
    float s = red[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) s = __fadd_rn(s, red[w]);
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(s, 512.0f), eps)));
    dst[t] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(v0, inv), __bfloat162float(kv_norm_w[t])));
    dst[t + 256] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(v1, inv), __bfloat162float(kv_norm_w[t + 256])));

    if (t < 32) {
        const float x0 = __bfloat162float(src[512 + 2 * t]);
        const float x1 = __bfloat162float(src[512 + 2 * t + 1]);
        float sv, cv;
        sincosf(__fmul_rn(freqs[t], static_cast<float>(static_cast<long long>(p))), &sv, &cv);
        dst[512 + 2 * t] = __float2bfloat16_rn(__fsub_rn(__fmul_rn(x0, cv), __fmul_rn(x1, sv)));
        dst[512 + 2 * t + 1] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(x1, cv), __fmul_rn(x0, sv)));
    }
}

// Batched per-row GPT-J RoPE (64 cols, `tf_mla::rope` order reading `pos[r]`); grid `(rows, heads, 1)`, block `32`.
extern "C" __global__ void __launch_bounds__(32)
    hy4_rows_q_rope(const __nv_bfloat16* __restrict__ x, long long xs, long long xh, __nv_bfloat16* __restrict__ out,
                    long long os, long long oh, const int* __restrict__ pos, const float* __restrict__ inv_freq) {
    const long long r = blockIdx.x, h = blockIdx.y;
    const int i = threadIdx.x;
    const __nv_bfloat16* src = x + r * xs + h * xh + 2 * i;
    const float x0 = __bfloat162float(src[0]), x1 = __bfloat162float(src[1]);
    float sv, cv;
    sincosf(__fmul_rn(inv_freq[i], static_cast<float>(static_cast<long long>(pos[r]))), &sv, &cv);
    __nv_bfloat16* dst = out + r * os + h * oh + 2 * i;
    dst[0] = __float2bfloat16_rn(__fsub_rn(__fmul_rn(x0, cv), __fmul_rn(x1, sv)));
    dst[1] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(x1, cv), __fmul_rn(x0, sv)));
}

// Convert fp32 logits [total] -> bf16 [total] (RNE); sets *nonfinite = 1 on NaN/Inf if non-null.
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_f32_to_bf16(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, int total,
                         unsigned int* __restrict__ nonfinite) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= total) return;
    if (i + 4 <= total) {
        const float4 v = *reinterpret_cast<const float4*>(src + i);
        if (nonfinite != nullptr && (!isfinite(v.x) || !isfinite(v.y) || !isfinite(v.z) || !isfinite(v.w))) {
            *nonfinite = 1u;
        }
        __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y);
        __nv_bfloat162 b = __floats2bfloat162_rn(v.z, v.w);
        uint2 out;
        out.x = *reinterpret_cast<unsigned int*>(&a);
        out.y = *reinterpret_cast<unsigned int*>(&b);
        *reinterpret_cast<uint2*>(dst + i) = out;
    } else {
        for (int j = i; j < total; ++j) {
            const float x = src[j];
            if (nonfinite != nullptr && !isfinite(x)) *nonfinite = 1u;
            dst[j] = __float2bfloat16_rn(x);
        }
    }
}

// Interleave allGather output [world, rows, local_u4] -> [rows, world * local_u4] bf16; flags Inf/NaN.
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_interleave_bf16(const uint4* __restrict__ src, uint4* __restrict__ dst, int local_u4, int rows, int world,
                             unsigned int* __restrict__ nonfinite) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long plane = static_cast<long long>(rows) * local_u4;
    const long long total = plane * world;
    if (idx >= total) return;
    const int row_u4 = world * local_u4;
    const int r = static_cast<int>(idx / row_u4);
    const int c = static_cast<int>(idx % row_u4);
    const int rk = c / local_u4;
    const int lc = c - rk * local_u4;
    const uint4 v = src[static_cast<long long>(rk) * plane + static_cast<long long>(r) * local_u4 + lc];
    if (nonfinite != nullptr) {
        const unsigned int w[4] = {v.x, v.y, v.z, v.w};
        bool bad = false;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (((w[k] & 0x7f80u) == 0x7f80u) || (((w[k] >> 16) & 0x7f80u) == 0x7f80u)) bad = true;
        }
        if (bad) *nonfinite = 1u;
    }
    dst[idx] = v;
}

__device__ __forceinline__ int mrope_axis(int pair) { return (pair < 30 && (pair % 3) != 0) ? (pair % 3) : 0; }

__device__ __forceinline__ void kv_write_body(const __nv_bfloat16* __restrict__ qa, int qa_stride, int q_lora,
                                              const __nv_bfloat16* __restrict__ kv_norm_w,
                                              const float* __restrict__ freqs, int r, int sp,
                                              const int* __restrict__ rot_pos, const int* __restrict__ pos3, int ld3,
                                              unsigned char* base, size_t slot, float eps, int kv8, float* red) {
    const int t = threadIdx.x;
    const int lane = t & 31, wid = t >> 5;
    const __nv_bfloat16* src = qa + static_cast<size_t>(r) * qa_stride + q_lora;
    if (t < 256) {
        const float v0 = __bfloat162float(src[t]);
        const float v1 = __bfloat162float(src[t + 256]);
        float acc = __fmaf_rn(v0, v0, 0.0f);
        acc = __fmaf_rn(v1, v1, acc);
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xFFFFFFFFu, acc, m));
        if (lane == 0) red[wid] = acc;
    }
    __syncthreads();
    float s = red[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) s = __fadd_rn(s, red[w]);
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(s, 512.0f), eps)));
    const __nv_bfloat16 normed = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(src[t]), inv),
                                                               __bfloat162float(kv_norm_w[t])));
    __syncthreads();

    __nv_bfloat16* rope_dst;
    if (!kv8) {
        __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(base) + slot * 576;
        dst[t] = normed;
        rope_dst = dst + 512;
    } else {
        unsigned char* dst = base + slot * 656;
        tf_kv_fp8::quant_row<512, true>(__bfloat162float(normed), t, dst, reinterpret_cast<float*>(dst + 640), red);
        if (t >= 32 && t < 35) reinterpret_cast<unsigned int*>(dst + 644)[t - 32] = 0u;
        rope_dst = reinterpret_cast<__nv_bfloat16*>(dst + 512);
    }
    if (t < 32) {
        const int rp = pos3 != nullptr ? pos3[static_cast<size_t>(mrope_axis(t)) * ld3 + r]
                                       : (rot_pos != nullptr ? rot_pos[r] : sp);
        const float x0 = __bfloat162float(src[512 + 2 * t]);
        const float x1 = __bfloat162float(src[512 + 2 * t + 1]);
        float sv, cv;
        sincosf(__fmul_rn(freqs[t], static_cast<float>(static_cast<long long>(rp))), &sv, &cv);
        rope_dst[2 * t] = __float2bfloat16_rn(__fsub_rn(__fmul_rn(x0, cv), __fmul_rn(x1, sv)));
        rope_dst[2 * t + 1] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(x1, cv), __fmul_rn(x0, sv)));
    }
}

// Extended `kv_a_norm` + `k_rope` write supporting `--kv-cache fp8` (656B row) and 3D M-RoPE (`pos3` / `rot_pos`).
extern "C" __global__ void __launch_bounds__(512)
    hy4_rows_kv_write_ex(const __nv_bfloat16* __restrict__ qa, int qa_stride, int q_lora,
                         const __nv_bfloat16* __restrict__ kv_norm_w, const float* __restrict__ freqs,
                         const int* __restrict__ slot_pos, const int* __restrict__ rot_pos,
                         const int* __restrict__ pos3, int ld3, const unsigned long long* __restrict__ caches,
                         void* __restrict__ single_cache, int ring, float eps, int kv8) {
    __shared__ float red[16];
    const int r = blockIdx.x;
    const int sp = slot_pos[r];
    const size_t slot = (ring > 0) ? static_cast<size_t>(sp % ring) : static_cast<size_t>(sp);
    unsigned char* base = caches != nullptr ? reinterpret_cast<unsigned char*>(caches[r])
                                            : static_cast<unsigned char*>(single_cache);
    kv_write_body(qa, qa_stride, q_lora, kv_norm_w, freqs, r, sp, rot_pos, pos3, ld3, base, slot, eps, kv8, red);
}

// Paged `kv_a_norm` + `k_rope` write (`256`-token pages via `page_table[s * pt_stride + (sp >> 8)]`).
extern "C" __global__ void __launch_bounds__(512)
    hy4_rows_kv_write_paged(const __nv_bfloat16* __restrict__ qa, int qa_stride, int q_lora,
                            const __nv_bfloat16* __restrict__ kv_norm_w, const float* __restrict__ freqs,
                            const int* __restrict__ slot_pos, const int* __restrict__ rot_pos,
                            const int* __restrict__ pos3, int ld3, const unsigned int* __restrict__ page_table,
                            const int* __restrict__ row_stream, int pt_stride, void* __restrict__ arena_base,
                            float eps, int kv8) {
    __shared__ float red[16];
    const int r = blockIdx.x;
    const int sp = slot_pos[r];
    const int s = row_stream != nullptr ? row_stream[r] : 0;
    const unsigned up = static_cast<unsigned>(sp);
    const size_t pg = page_table[static_cast<size_t>(s) * pt_stride + (up >> 8)];
    const size_t slot = pg * 256u + (up & 255u);
    kv_write_body(qa, qa_stride, q_lora, kv_norm_w, freqs, r, sp, rot_pos, pos3, ld3,
                  static_cast<unsigned char*>(arena_base), slot, eps, kv8, red);
}

// Batched per-row 3D interleaved M-RoPE (`mrope_section = [12, 10, 10]`) on query heads (`pos3` [3, ld3]).
extern "C" __global__ void __launch_bounds__(32)
    hy4_rows_q_rope_mrope(const __nv_bfloat16* __restrict__ x, long long xs, long long xh, __nv_bfloat16* __restrict__ out,
                          long long os, long long oh, const int* __restrict__ pos, const int* __restrict__ pos3, int ld3,
                          const float* __restrict__ inv_freq) {
    const long long r = blockIdx.x, h = blockIdx.y;
    const int i = threadIdx.x;
    const int rp = pos3 != nullptr ? pos3[static_cast<size_t>(mrope_axis(i)) * ld3 + r] : pos[r];
    const __nv_bfloat16* src = x + r * xs + h * xh + 2 * i;
    const float x0 = __bfloat162float(src[0]), x1 = __bfloat162float(src[1]);
    float sv, cv;
    sincosf(__fmul_rn(inv_freq[i], static_cast<float>(static_cast<long long>(rp))), &sv, &cv);
    __nv_bfloat16* dst = out + r * os + h * oh + 2 * i;
    dst[0] = __float2bfloat16_rn(__fsub_rn(__fmul_rn(x0, cv), __fmul_rn(x1, sv)));
    dst[1] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(x1, cv), __fmul_rn(x0, sv)));
}

// Splices encoded ViT image features `feat[src_idx[r]]` across all 4 iHC residual streams `streams[r, 0..3]`.
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_splice_streams(const uint4* __restrict__ feat, const int* __restrict__ src_idx,
                            uint4* __restrict__ streams, int hidden_u4, int rows) {
    const int r = blockIdx.x;
    const int c = static_cast<int>(blockIdx.y) * 256 + static_cast<int>(threadIdx.x);
    if (r >= rows || c >= hidden_u4) return;
    const int s = src_idx[r];
    if (s < 0) return;
    const uint4 v = feat[static_cast<size_t>(s) * hidden_u4 + c];
    const size_t base = static_cast<size_t>(r) * 4 * hidden_u4 + c;
    streams[base] = v;
    streams[base + hidden_u4] = v;
    streams[base + 2 * hidden_u4] = v;
    streams[base + 3 * hidden_u4] = v;
}

// ViT QKV bias + transpose `[rows, 3, heads, hd]` fp32 -> `q, k, v [heads, rows, hd]` bf16 (no 2D RoPE).
extern "C" __global__ void __launch_bounds__(128)
    hy4_rows_vit_qkv(const float* __restrict__ acc, const __nv_bfloat16* __restrict__ bias,
                     __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v,
                     int rows, int heads, int hd) {
    const long long r = blockIdx.x;
    const int h = blockIdx.y;
    const int D = heads * hd;
    const float* a = acc + r * 3 * D;
    const long long o = (static_cast<long long>(h) * rows + r) * hd;
    for (int j = threadIdx.x; j < hd; j += blockDim.x) {
        const int c = h * hd + j;
        q[o + j] = __float2bfloat16_rn(a[c] + __bfloat162float(bias[c]));
        k[o + j] = __float2bfloat16_rn(a[D + c] + __bfloat162float(bias[D + c]));
        v[o + j] = __float2bfloat16_rn(a[2 * D + c] + __bfloat162float(bias[2 * D + c]));
    }
}

// Gathers raster `[gh, gw, d_u4]` into 2x2 merged blocks `[gh/2 * gw/2, 2, 2, d_u4]` for `perceive.proj.0`.
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_vit_merge_gather(const uint4* __restrict__ src, uint4* __restrict__ dst, int gh, int gw, int d_u4) {
    const int mh = gh >> 1, mw = gw >> 1;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(mh) * mw * 4 * d_u4;
    if (idx >= total) return;
    const int c = static_cast<int>(idx % d_u4);
    const int sub = static_cast<int>((idx / d_u4) & 3);
    const int blk = static_cast<int>(idx / (4 * d_u4));
    const int oh = blk / mw, ow = blk % mw;
    const int r = (oh * 2 + (sub >> 1)) * gw + (ow * 2 + (sub & 1));
    dst[idx] = src[static_cast<long long>(r) * d_u4 + c];
}

// Appends `image_newline` `[out_u4]` after each merged row `[mh, mw, out_u4]` -> `[mh, mw + 1, out_u4]`.
extern "C" __global__ void __launch_bounds__(256)
    hy4_rows_vit_append_newline(const uint4* __restrict__ merged, const uint4* __restrict__ newline,
                                uint4* __restrict__ out, int mh, int mw, int out_u4) {
    const int out_w = mw + 1;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(mh) * out_w * out_u4;
    if (idx >= total) return;
    const int c = static_cast<int>(idx % out_u4);
    const int tok = static_cast<int>(idx / out_u4);
    const int oh = tok / out_w, ow = tok % out_w;
    out[idx] = (ow < mw) ? merged[static_cast<long long>(oh * mw + ow) * out_u4 + c] : newline[c];
}
