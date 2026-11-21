// Extern "C" entry points for Hy4-flash MLA query absorb (`W_UK_T`), value expand + gate (`W_UV`), and `kv_b` repack.
#include "hy4_mla_absorb_expand_gate.cuh"

#include <math.h>

using namespace hy4c2;

//                         K   MT NT WARPS KC STAGES D  PLANES STREAM
#define C2_AD(NT) Cfg<ND, 1, NT, 4, ND, 2, 6, 1, true>
#define C2_AP Cfg<ND, 2, 8, 8, 64, 3, 2, 1, false>
#define C2_ED(NT) Cfg<KL, 1, NT, 4, KL, 2, 8, 1, true>
#define C2_EP Cfg<KL, 2, 8, 8, 64, 3, 2, 1, false>
#define C2_EDF(NT) Cfg<KL, 1, NT, 4, KL, 2, 8, 3, true>
#define C2_EPF Cfg<KL, 2, 4, 8, 64, 3, 2, 3, false>

typedef C2_AD(1) AD1;
typedef C2_AD(2) AD2;
typedef C2_AD(4) AD4;
typedef C2_AP AP;
typedef C2_ED(1) ED1;
typedef C2_ED(2) ED2;
typedef C2_ED(4) ED4;
typedef C2_EP EP;
typedef C2_EDF(1) EDF1;
typedef C2_EDF(2) EDF2;
typedef C2_EDF(4) EDF4;
typedef Cfg<KL, 1, 4, 4, 64, 2, 2, 3, true> EDF4_K64_S2;
typedef C2_EPF EPF;

#define C2_ABSORB(NAME, CFG, OUT32, MINB)                                                                              \
    extern "C" __global__ void __launch_bounds__(CFG::THREADS, MINB)                                                   \
        NAME(const bf16* QB, int ldq, int R, int Hl, const bf16* WUKT, const float* CS, const int* CSI, void* QA) {    \
        absorb_body<CFG, OUT32, false>(QB, ldq, R, Hl, WUKT, CS, CSI, QA);                                             \
    }                                                                                                                  \
    extern "C" __global__ void __launch_bounds__(CFG::THREADS, MINB)                                                   \
        NAME##_cs2(const bf16* QB, int ldq, int R, int Hl, const bf16* WUKT, const float2* CS, const int* CSI,         \
                   void* QA) {                                                                                         \
        absorb_body<CFG, OUT32, true>(QB, ldq, R, Hl, WUKT, CS, CSI, QA);                                              \
    }
// e2e: bf16 QA plus c1's exact lo plane QLO elements after QA (no rope: CS null)
#define C2_ABSORB_X(NAME, CFG, MINB)                                                                                   \
    extern "C" __global__ void __launch_bounds__(CFG::THREADS, MINB)                                                   \
        NAME(const bf16* QB, int ldq, int R, int Hl, const bf16* WUKT, void* QA, long long QLO) {                      \
        absorb_body<CFG, false, false>(QB, ldq, R, Hl, WUKT, nullptr, nullptr, QA, QLO);                               \
    }
C2_ABSORB_X(hy4c2_absorb_d1_x, AD1, 4)
C2_ABSORB_X(hy4c2_absorb_d2_x, AD2, 4)
C2_ABSORB_X(hy4c2_absorb_p_x, AP, 2)
C2_ABSORB(hy4c2_absorb_d1, AD1, false, 4)
C2_ABSORB(hy4c2_absorb_d2, AD2, false, 4)
C2_ABSORB(hy4c2_absorb_d4, AD4, false, 4)
C2_ABSORB(hy4c2_absorb_p, AP, false, 2)
C2_ABSORB(hy4c2_absorb_d1_f32, AD1, true, 4)
C2_ABSORB(hy4c2_absorb_d2_f32, AD2, true, 4)
C2_ABSORB(hy4c2_absorb_d4_f32, AD4, true, 4)
C2_ABSORB(hy4c2_absorb_p_f32, AP, true, 2)

#define C2_EXPAND(NAME, CFG, LT, MINB)                                                                                 \
    extern "C" __global__ void __launch_bounds__(CFG::THREADS, MINB)                                                   \
        NAME(const LT* LAT, int R, int Hl, const bf16* WUV, const bf16* GATE, int ldg, bf16* ATT) {                    \
        expand_body<CFG, LT>(LAT, R, Hl, WUV, GATE, ldg, ATT);                                                         \
    }
C2_EXPAND(hy4c2_expand_d1, ED1, bf16, 4)
C2_EXPAND(hy4c2_expand_d2, ED2, bf16, 4)
C2_EXPAND(hy4c2_expand_d4, ED4, bf16, 2)
C2_EXPAND(hy4c2_expand_p, EP, bf16, 2)
C2_EXPAND(hy4c2_expand_d1_f32lat, EDF1, float, 4)
C2_EXPAND(hy4c2_expand_d2_f32lat, EDF2, float, 2)
C2_EXPAND(hy4c2_expand_d4_f32lat, EDF4, float, 1)
C2_EXPAND(hy4c2_expand_d4k64s2_f32lat, EDF4_K64_S2, float, 3)
C2_EXPAND(hy4c2_expand_p_f32lat, EPF, float, 2)

// Repack `kv_b_proj [Hl * 448, 512]` into TILED `W_UK_T [Hl, 512, 192]` and `W_UV [Hl, 256, 512]` (`grid (16, Hl)`).
extern "C" __global__ void __launch_bounds__(256) hy4c2_repack_kv_b(const bf16* __restrict__ kvb, int Hl,
                                                                   bf16* __restrict__ WUKT, bf16* __restrict__ WUV) {
    __shared__ bf16 t[ND][34];
    const int h = blockIdx.y, c0 = blockIdx.x * 32;
    const bf16* src = kvb + static_cast<size_t>(h) * KVB * KL;
    for (int e = threadIdx.x; e < ND * 32; e += blockDim.x) {
        const int d = e / 32, c = e % 32;
        t[d][c] = src[static_cast<size_t>(d) * KL + c0 + c];
    }
    __syncthreads();
    constexpr int CHK = ND / 32, CHV = KL / 32;
    uint4* dk = reinterpret_cast<uint4*>(WUKT + static_cast<size_t>(h) * KL * ND);
    for (int e = threadIdx.x; e < 2 * CHK * 64; e += blockDim.x) {  // two 16-row tiles x 6 blocks x 64 units
        const int l = e & 31, j = (e >> 5) & 1, b = (e >> 6) % CHK, tt = (e >> 6) / CHK;
        const int row = 16 * tt + (l >> 2) + 8 * j, k0 = 32 * b + 8 * (l & 3);
        __align__(16) bf16 v8[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) v8[i] = t[k0 + i][row];
        const int tg = c0 / 16 + tt;
        dk[((static_cast<size_t>(tg) * CHK + b) * 2 + j) * 32 + l] = *reinterpret_cast<const uint4*>(v8);
    }
    if (blockIdx.x * 32 < VD) {
        uint4* dv = reinterpret_cast<uint4*>(WUV + static_cast<size_t>(h) * VD * KL);
        for (int e = threadIdx.x; e < 2 * CHV * 64; e += blockDim.x) {
            const int l = e & 31, j = (e >> 5) & 1, b = (e >> 6) % CHV, tt = (e >> 6) / CHV;
            const int v = c0 + 16 * tt + (l >> 2) + 8 * j, k0 = 32 * b + 8 * (l & 3);
            const int tg = c0 / 16 + tt;
            dv[((static_cast<size_t>(tg) * CHV + b) * 2 + j) * 32 + l] =
                *reinterpret_cast<const uint4*>(src + static_cast<size_t>(ND + v) * KL + k0);
        }
    }
}

// Precompute inv_freq[j] = 1 / theta^(2j / 64) once at load (separating powf from sincosf keeps 0 B stack everywhere).
extern "C" __global__ void __launch_bounds__(32) hy4c2_inv_freq(float theta, float* __restrict__ INV) {
    const int j = threadIdx.x;
    if (j < 32) INV[j] = __fdiv_rn(1.0f, powf(theta, __fdiv_rn(static_cast<float>(2 * j), 64.0f)));
}

// Per-row M-RoPE `cos`/`sin` table (`CS2 == false`: split `[n, 64]`; `CS2 == true`: interleaved `float2 [n * 32]`).
template <bool CS2>
__device__ __forceinline__ void mrope_cs_write(const int* __restrict__ POS3, int n, int ld3, float inv_j, int i, int j,
                                               int sec_h, int sec_w, int round_bf16, void* __restrict__ CSv) {
    const int axis = (j % 3 == 1 && j < 3 * sec_h) ? 1 : (j % 3 == 2 && j < 3 * sec_w) ? 2 : 0;
    const int pos = __ldg(POS3 + axis * ld3 + i);
    const float f = __fmul_rn(static_cast<float>(pos), inv_j);
    float s, c;
    sincosf(f, &s, &c);
    if (round_bf16) {
        c = rbf(c);
        s = rbf(s);
    }
    if constexpr (CS2) {
        reinterpret_cast<float2*>(CSv)[static_cast<size_t>(i) * 32 + j] = make_float2(c, s);
    } else {
        float* CS = reinterpret_cast<float*>(CSv);
        CS[static_cast<size_t>(i) * 64 + j] = c;
        CS[static_cast<size_t>(i) * 64 + 32 + j] = s;
    }
}

extern "C" __global__ void __launch_bounds__(256) hy4c2_mrope_cs(const int* __restrict__ POS3, int n, int ld3,
                                                                const float* __restrict__ INV, int sec_h, int sec_w,
                                                                int round_bf16, float* __restrict__ CS) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n * 32) return;
    const int i = e / 32, j = e % 32;
    mrope_cs_write<false>(POS3, n, ld3, __ldg(INV + j), i, j, sec_h, sec_w, round_bf16, CS);
}

extern "C" __global__ void __launch_bounds__(256) hy4c2_mrope_cs2(const int* __restrict__ POS3, int n, int ld3,
                                                                 const float* __restrict__ INV, int sec_h, int sec_w,
                                                                 int round_bf16, float2* __restrict__ CS) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n * 32) return;
    const int i = e / 32, j = e % 32;
    mrope_cs_write<true>(POS3, n, ld3, __ldg(INV + j), i, j, sec_h, sec_w, round_bf16, CS);
}
