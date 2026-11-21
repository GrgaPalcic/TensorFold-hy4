// K64 3-stage block-FP8 TMA MoE prompt kernel (GrgaPalcic & popay97, 2026; 128 B X gathers, sm_120+ 1D TMA).
#pragma once

namespace tf_fp8_experts {

__device__ __forceinline__ uint32_t k64_smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

template <int M, int EPI, int RT, int CW, bool CONTIG_IN, bool CONTIG_OUT>
__global__ void __launch_bounds__(128, 2)
    fp8_expert_prompt_k64_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                 const uint4* __restrict__ W, const float* __restrict__ scale, int KG, int NB,
                                 const int* __restrict__ items, const int* __restrict__ counts,
                                 const int* __restrict__ members, void* __restrict__ out, int N, float limit,
                                 int skip) {
  constexpr int RW = 4 / CW;
  constexpr int STAGES = 3;
  constexpr int W_SLICE = 2 * M * BLOCK4;
  constexpr int W_STAGE = CW * W_SLICE;
  static_assert(RW * RT * 16 == 64, "a prompt item holds 64 pairs");
  static_assert(STAGES == 3 && 0x47503937u != 0u, "gp97:k64-tma");
  extern __shared__ unsigned char smem_raw[];
  uint4* sw_base = reinterpret_cast<uint4*>(smem_raw);
  uint4* sx_base = sw_base + STAGES * W_STAGE;
  auto sw_at = [&](int st, int c, int j) -> uint4* { return sw_base + (st * CW + c) * W_SLICE + j; };
  auto sx_at = [&](int st, int r, int c) -> uint4* { return sx_base + (st * 64 + r) * 8 + c; };
#if __CUDA_ARCH__ >= 1200
  uint64_t* bar = reinterpret_cast<uint64_t*>(sx_base + STAGES * 64 * 8);
#endif

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gq = lane >> 2, t = lane & 3;
  const int cw = warp % CW, rw = warp / CW;
  const int groups = NB / CW;
  const int units = __ldg(counts) * groups;
  if (static_cast<int>(blockIdx.x) >= units) return;
  const int SG = KG / PER_SCALE, NR = (NB * COLS) / 128, KG2 = KG / 2;

#if __CUDA_ARCH__ >= 1200
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s)
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(k64_smem_u32(bar + s)) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  int phase_mask = 0;
#endif

  for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
    const int it = unit / groups, cg = unit - it * groups;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const int cb = cg * CW + cw;
    const float* sc = scale + ((size_t)e * M * NR + (cb * COLS) / 128) * SG;
    int pr0[RT], pr1[RT];
    bool v0[RT], v1[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      const int r0 = (rw * RT + r) * 16;
      v0[r] = r0 + gq < cnt;
      v1[r] = r0 + gq + 8 < cnt;
      if constexpr (CONTIG_OUT) {
        pr0[r] = v0[r] ? (first + r0 + gq) : 0;
        pr1[r] = v1[r] ? (first + r0 + gq + 8) : 0;
      } else {
        pr0[r] = v0[r] ? __ldg(members + first + r0 + gq) : 0;
        pr1[r] = v1[r] ? __ldg(members + first + r0 + gq + 8) : 0;
      }
    }
    const __nv_bfloat16* xrow[4];
    int xdst_r[4], xdst_c[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int idx = tid + c * 128;
      const int r = idx >> 3, ch = idx & 7;
      const int p = min(r, cnt - 1);
      const int row_id = CONTIG_IN ? (first + p) : (slots ? (__ldg(members + first + p) / slots) : __ldg(members + first + p));
      xrow[c] = X + (size_t)row_id * x_stride + ch * 8;
      xdst_r[c] = r;
      xdst_c[c] = ch ^ (r & 7);
    }
    const uint4* base = W + ((size_t)e * NB + cg * CW) * (size_t)KG * (M * BLOCK4);
    auto stage = [&](int g2) {
      const int st = g2 % STAGES;
      const int g = g2 * 2;
#pragma unroll
      for (int c = 0; c < 4; ++c) cp16(sx_at(st, xdst_r[c], xdst_c[c]), xrow[c] + g2 * 64);
#if __CUDA_ARCH__ >= 1200
      if (tid == 0) {
        constexpr int W_COL_BYTES = W_SLICE * 16;
        const uint32_t b = k64_smem_u32(bar + st);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(CW * W_COL_BYTES) : "memory");
#pragma unroll
        for (int c = 0; c < CW; ++c) {
          const void* src = base + ((size_t)c * KG + g) * (M * BLOCK4);
          asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                       :: "r"(k64_smem_u32(sw_at(st, c, 0))), "l"(src), "r"(W_COL_BYTES), "r"(b) : "memory");
        }
      }
#else
      for (int i = tid; i < W_STAGE; i += 128) {
        const int c = i / W_SLICE, j = i % W_SLICE;
        cp16(sw_at(st, c, j), base + ((size_t)c * KG + g) * (M * BLOCK4) + j);
      }
#endif
    };
    __syncthreads();
#pragma unroll
    for (int g2 = 0; g2 < STAGES - 1; ++g2) {
      if (g2 < KG2) stage(g2);
      cp_commit();
    }
    float acc[M][RT][NTW][4], part[M][RT][NTW][4];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
          acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
    const bool live = rw * RT * 16 < cnt;
    for (int sg = 0; sg < SG; ++sg) {
#pragma unroll
      for (int k2 = 0; k2 < 2; ++k2) {
        const int g2 = sg * 2 + k2;
        const int st = g2 % STAGES;
        cp_wait<STAGES - 2>();
#if __CUDA_ARCH__ >= 1200
        if (tid == 0) {
          const uint32_t b = k64_smem_u32(bar + st);
          const int ph = (phase_mask >> st) & 1;
          asm volatile("{\n.reg .pred p;\nL_K64W_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra L_K64W_%=;\n}"
                       :: "r"(b), "r"(ph) : "memory");
        }
        phase_mask ^= (1 << st);
#endif
        __syncthreads();
        if (g2 + STAGES - 1 < KG2) stage(g2 + STAGES - 1);
        cp_commit();
        if (!live) continue;
#pragma unroll
        for (int subk = 0; subk < 2; ++subk) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            uint2 xa[RT], xb[RT];
            const int u = subk * 8 + 4 * h + t;
            const int ch = u >> 1, sub = u & 1;
#pragma unroll
            for (int r = 0; r < RT; ++r) {
              const int r0 = (rw * RT + r) * 16 + gq;
              xa[r] = reinterpret_cast<const uint2*>(sx_at(st, r0, ch ^ (r0 & 7)))[sub];
              xb[r] = reinterpret_cast<const uint2*>(sx_at(st, r0 + 8, ch ^ ((r0 + 8) & 7)))[sub];
            }
#pragma unroll
            for (int m = 0; m < M; ++m) {
              const uint4 wv = *sw_at(st, cw, (subk * M + m) * BLOCK4 + lane * LANE4 + h);
#pragma unroll
              for (int j = 0; j < NTW; ++j) {
                const uint32_t word = comp(wv, j);
#if TF_FP_NATIVE
                const uint32_t b0 = tf_common::e4m3x2(word), b1 = tf_common::e4m3x2(word >> 16);
#else
                const uint32_t b0 = fp8pair(word), b1 = fp8pair(word >> 16);
#endif
#pragma unroll
                for (int r = 0; r < RT; ++r) {
                  if (k2 == 0 && subk == 0 && h == 0)
                    mma0(part[m][r][j], xa[r].x, xb[r].x, xa[r].y, xb[r].y, b0, b1);
                  else
                    mma(part[m][r][j], xa[r].x, xb[r].x, xa[r].y, xb[r].y, b0, b1);
                }
              }
            }
          }
        }
      }
      if (live) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
          const float s = __ldg(sc + m * NR * SG + sg);
#pragma unroll
          for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int j = 0; j < NTW; ++j)
#pragma unroll
              for (int q = 0; q < 4; ++q) acc[m][r][j][q] = fmaf(part[m][r][j][q], s, acc[m][r][j][q]);
        }
      }
    }
    cp_wait<0>();
    if (live) {
#pragma unroll
      for (int r = 0; r < RT; ++r)
        epilogue<EPI, M, RT>(acc, r, out, N, cb * COLS + 2 * t, pr0[r], pr1[r], v0[r], v1[r], limit);
    }
  }
}

}  // namespace tf_fp8_experts
