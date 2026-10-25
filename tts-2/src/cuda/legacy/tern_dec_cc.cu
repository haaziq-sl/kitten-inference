// CUDA-core ternary decode projections for GPUs without bf16 tensor-core instructions (SM 6.x Pascal, SM 7.5
// Turing), bit-identical to the decode kernel (tern_dec_kernel in src/cuda/kernels/tern_proj.cu) with its software
// MMA (mma_16816_emu_lo / mma_16816_emu in tern_proj.cuh).
//
// Why: on these GPUs every m16n8k16 MMA of the decode kernel is emulated with 24 warp shuffles and 32 FMAs per lane
// computing 8 rows x 8 columns, of which a 1-row decode step uses one row. The projections are then bound by
// shuffle and instruction issue (~8% of DRAM bandwidth on a GTX 1080 Ti), not by the 2-bit weight stream.
// These kernels compute only the rows that exist, with plain FMAs and no inline PTX.
//
// Arithmetic (the family contract of include/kt2n/kernels/tern_proj.h, operation for operation): K = 256 C;
// partial w (0 .. WK-1) covers chunks [w CPW, (w+1) CPW); MMA step j (0..15) of chunk c feeds accumulator j % ACC
// the 16 products of k = 256c + 64q + 4j + e (quarter q = 0..3, e = 0..3) in the emulation's order
//     (q0 e0, q0 e1, q1 e0, q1 e1, q2 e0, q2 e1, q3 e0, q3 e1, q0 e2, q0 e3, q1 e2, ..., q3 e3)
// each as fmaf(bf16 x, bf16 w, acc) (exact product, one rounding). Then partial = ((acc0 + acc1) + acc2) + ...,
// output = ((p0 + p1) + p2) + ... in fp32, and the decode kernel's epilogue arithmetic. Every weight is decoded by
// the same PRMT tables (scale_tables, and decode_step's selectors), so the fp32 weight bits are those the emulation
// reads. Each chain lives in one lane (or is handed between two lanes unrounded, by a shuffle), so the order holds.
//
// Two kernels (both persistent: warps loop over weight rows; lane = accumulator chain(s) of one row):
//   dec_cc_kernel      any family, 1..4 decode rows. X staged once per CTA as fp32 in shared memory. A row has
//                      L = WK * LPP lanes (lane (w, p) runs chains p, p + LPP of partial w: two FMA chains per lane),
//                      RPW = 32 / L rows per warp, and RPL rows per lane (SM 7.5, 1 decode row: 2) so that one
//                      shared-memory activation load feeds RPL rows. Partials are added in order with shuffles on
//                      the row's first lane.
//   dec_cc_m1_kernel   SM 7.5, one decode row, the 2048-wide families (qkv, o; gate_up also row by row for 2..4):
//                      lane = (chain, chunk) of a row, its 64 activations in registers; gate_up's two chunks per
//                      partial run as a two-lane pipeline. Two weight rows per lane.
// Only intrinsics HIP also has (__byte_perm, __shfl_sync / __shfl, fmaf, __fadd_rn).
#include "tern_dec_cc.h"

#if __has_include("kt2n/kernels/tern_proj.h")
#include "kt2n/kernels/tern_proj.h"
#else
#include "../../../include/kt2n/kernels/tern_proj.h"
#endif
#include "tern_proj.cuh"
#include "kt2n/cuda_caps.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <atomic>

namespace kt2n_tern_cc {
namespace {

using kt2n_tern::bf16_t;
using kt2n_tern::kChunk;

// bf16 halves of a packed pair as exact fp32 (low half = lower k), as the MMA emulation reads them.
__device__ __forceinline__ float lo16(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi16(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

#ifndef KT2N_TERN_CC_WARPS
#define KT2N_TERN_CC_WARPS 8  // warps per CTA
#endif
#ifndef KT2N_TERN_CC_REGS
#define KT2N_TERN_CC_REGS 96  // register budget per thread (sets the minimum CTAs per SM)
#endif
constexpr int kWarps = KT2N_TERN_CC_WARPS;
constexpr int kThreads = 32 * kWarps;
constexpr int kMinBlocks = 65536 / (KT2N_TERN_CC_REGS * kThreads) > 0 ? 65536 / (KT2N_TERN_CC_REGS * kThreads) : 1;

// Lane-to-lane copy inside a 32-lane group (HIP: also inside each half of a 64-wide wavefront).
__device__ __forceinline__ float shfl_idx(float v, int src) {
#if defined(__HIP_PLATFORM_AMD__)
    return __shfl(v, src, 32);
#else
    return __shfl_sync(0xffffffffu, v, src, 32);
#endif
}

// PRMT selectors of 4 weights (decode_step's, bit for bit): code field f (0..3) of a "sel" word, as y * 0x11 +
// 0x40404040 = (y << 4) | y | 0x40404040 for y = the field's four 2-bit codes, one per byte. Written with ORs so the
// compiler sees the selector nibbles' top bits are zero (no extra masking for __byte_perm's 3-bit selectors).
#ifndef KT2N_TERN_CC_SEL_MUL
#define KT2N_TERN_CC_SEL_MUL 1
#endif
__device__ __forceinline__ uint32_t sel_of(uint32_t y) {
#if KT2N_TERN_CC_SEL_MUL
    return y * 0x11u + 0x40404040u;  // one IMAD (Turing: on the FP32 pipe, beside the decode's integer work)
#else
    return (y << 4) | y | 0x40404040u;
#endif
}
// bf16 pairs {w(4f), w(4f+1)}, {w(4f+2), w(4f+3)} of a selector, as decode_step.
__device__ __forceinline__ void pairs_of(uint32_t sel, uint32_t ta, uint32_t tb, uint32_t& b0, uint32_t& b1) {
    b0 = __byte_perm(ta, tb, sel);
    b1 = __byte_perm(ta, tb, sel >> 16);
}

// One step of one accumulator chain for RPL weight rows at once: 16 products of k = 256c + 64q + 4j + e in the
// emulation's order, the activations loaded once (per decode row mm) for all RPL rows.
template <int MB, int RPL>
__device__ __forceinline__ void step_rows(float (&acc)[RPL][MB], const float* xr, int xs_row,
                                          const uint32_t (&b0)[RPL][4], const uint32_t (&b1)[RPL][4]) {
#pragma unroll
    for (int mm = 0; mm < MB; ++mm) {
        float4 xv[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) xv[q] = *reinterpret_cast<const float4*>(xr + mm * xs_row + 64 * q);
#pragma unroll
        for (int r = 0; r < RPL; ++r) {
            float s = acc[r][mm];
#pragma unroll
            for (int q = 0; q < 4; ++q) {  // k + 0, + 1 of every quarter
                s = fmaf(xv[q].x, lo16(b0[r][q]), s);
                s = fmaf(xv[q].y, hi16(b0[r][q]), s);
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) {  // then k + 2, + 3
                s = fmaf(xv[q].z, lo16(b1[r][q]), s);
                s = fmaf(xv[q].w, hi16(b1[r][q]), s);
            }
            acc[r][mm] = s;
        }
    }
}

// Warps loop over groups of RPW x RPL weight rows; X is staged once per CTA and pass. Each row has L = WK * LPP
// lanes: lane (w, p) of a row runs the CH = ACC / LPP chains a = p, p + LPP, ... of partial w, i.e. steps
// j = a, a + ACC, ... of each of the partial's CPW chunks (two independent FMA chains per lane). Lane `sub` of the
// warp's RPW row slots runs rows n0 + sub + r RPW (r < RPL) together, so one shared-memory load of activations
// feeds RPL rows (the loads are the limit: shared memory delivers one 16-byte load per lane per 4 cycles per SM).
template <int C, int WK, int ACC, int MB, int EPI, int RPL>
__global__ void __launch_bounds__(kThreads, kMinBlocks)
dec_cc_kernel(const bf16_t* X, int ldx, int M, const uint4* __restrict__ Wt, const uint16_t* __restrict__ St,
              bf16_t* Y, int ldy, const bf16_t* R, int ldr, int N) {
    constexpr int CH = ACC >= 2 ? 2 : 1;        // chains per lane
    constexpr int LPP = ACC / CH;               // lanes per partial
    constexpr int L = WK * LPP;                 // lanes per row
    constexpr int RPW = 32 / L;                 // row slots per warp
    constexpr int G = RPW * RPL;                // rows per warp and iteration
    static_assert(C % WK == 0 && (ACC == 1 || ACC == 2 || ACC == 4) && L <= 32, "family shape");
    static_assert(EPI != KT2N_TERN_EPI_SILU_MUL || RPW % 2 == 0, "SILU_MUL pairs rows 2i, 2i + 1 in one warp");
    constexpr int CPW = C / WK;                 // chunks per partial
    constexpr int K = C * kChunk;
    // Floats per staged chunk: 256 + pad, chosen so that the distinct 16-byte activation reads of one warp
    // instruction (partial w, first chain p: offset w XC + 4p) spread over the 8 bank groups of 16 bytes
    // (bank group (w XC / 4 + p) mod 8): 2w + p for two lanes per partial, w for one.
    constexpr int XC = kChunk + (LPP == 2 ? 8 : 4);
    constexpr int XS = C * XC;                  // floats per staged row
    __shared__ __align__(16) float xs[MB * XS];

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int sub = lane / L;                   // row slot of this lane (>= RPW: idle lane)
    const int cl = lane - sub * L;              // lane within the row
    const int w = cl / LPP, p = cl % LPP;       // partial, first chain
    const bool active = sub < RPW;
    const int gwarp = blockIdx.x * kWarps + warp, nwarps = gridDim.x * kWarps, stride = nwarps * G;

    for (int m0 = 0; m0 < M; m0 += MB) {
        if (m0) __syncthreads();  // the previous pass is done with xs
        // Stage rows m0 .. m0 + MB - 1 of X as fp32 (rows >= M as zeros: computed, never stored).
        for (int v = threadIdx.x; v < MB * (K / 8); v += kThreads) {
            const int mm = v / (K / 8), k = (v % (K / 8)) * 8;
            uint4 x8 = make_uint4(0u, 0u, 0u, 0u);
            if (m0 + mm < M) x8 = *reinterpret_cast<const uint4*>(X + (size_t)(m0 + mm) * ldx + k);
            float* dst = xs + mm * XS + (k / kChunk) * XC + (k % kChunk);
            *reinterpret_cast<float4*>(dst) = make_float4(lo16(x8.x), hi16(x8.x), lo16(x8.y), hi16(x8.y));
            *reinterpret_cast<float4*>(dst + 4) = make_float4(lo16(x8.z), hi16(x8.z), lo16(x8.w), hi16(x8.w));
        }
        __syncthreads();

        // The lane's weights of its rows: chunk c = w CPW + i, quarter q = 16 bytes = k 256c + 64q + [0, 64) (lanes
        // of one partial load the same addresses in one instruction), and the chunk's two group scales.
        auto load = [&](int n0, uint4 (&W)[RPL][CPW][4], uint32_t (&S)[RPL][CPW][2]) {
            if (!active || n0 >= N) return;
#pragma unroll
            for (int r = 0; r < RPL; ++r) {
                const int n = n0 + sub + r * RPW, tile = n >> 3, g = n & 7;
#pragma unroll
                for (int i = 0; i < CPW; ++i) {
                    const size_t tc = (size_t)tile * C + w * CPW + i;
#pragma unroll
                    for (int q = 0; q < 4; ++q) W[r][i][q] = __ldg(Wt + tc * 32 + g * 4 + q);
#pragma unroll
                    for (int h = 0; h < 2; ++h) S[r][i][h] = __ldg(St + tc * 16 + g * 2 + h);
                }
            }
        };
        auto body = [&](int n0, const uint4 (&W)[RPL][CPW][4], const uint32_t (&S)[RPL][CPW][2]) {
            float acc[CH][RPL][MB];
#pragma unroll
            for (int h = 0; h < CH; ++h)
#pragma unroll
                for (int r = 0; r < RPL; ++r)
#pragma unroll
                    for (int mm = 0; mm < MB; ++mm) acc[h][r][mm] = 0.f;
            if (active) {
#pragma unroll
                for (int i = 0; i < CPW; ++i) {
                    uint32_t ta[RPL][2], tb[RPL][2];  // PRMT tables of the chunk's two 128-groups (quarters 0-1, 2-3)
#pragma unroll
                    for (int r = 0; r < RPL; ++r)
#pragma unroll
                        for (int h = 0; h < 2; ++h) kt2n_tern::scale_tables(S[r][i][h], ta[r][h], tb[r][h]);
                    const float* xc = xs + (w * CPW + i) * XC;
#if defined(KT2N_HIP)
                    // AMD: one step group per iteration. Unrolled, this loop needs more than the 256 VGPRs of a wave
                    // and spills ~290 of them (gate_up, 2 decode rows); with ROCm 7.2's compiler that spilled code
                    // gave wrong, run-to-run different outputs when many waves were resident (ROCm 7.1's was
                    // correct). Not unrolled: no spills in any instance; the same FMAs in the same order.
#pragma unroll 1
                    for (int jo = 0; jo < 16 / ACC; ++jo) {
#else
#line 200
#pragma unroll
                    for (int jo = 0; jo < 16 / ACC; ++jo) {
#endif
#line 202
                        // Chain h of this lane: a = p + h LPP, step j = jo ACC + a; its codes are field j & 3 of word
                        // j >> 2 (the same word for both chains when ACC = 4).
#pragma unroll
                        for (int h = 0; h < CH; ++h) {
                            const int jstat = jo * ACC + h * LPP;  // j - p
                            uint32_t b0[RPL][4], b1[RPL][4];
#pragma unroll
                            for (int r = 0; r < RPL; ++r)
#pragma unroll
                                for (int q = 0; q < 4; ++q) {
                                    const uint32_t word = kt2n_tern::word_of(W[r][i][q], jstat >> 2);
                                    const uint32_t y = (word >> (2 * ((jstat & 3) + p))) & 0x03030303u;
                                    pairs_of(sel_of(y), ta[r][q >> 1], tb[r][q >> 1], b0[r][q], b1[r][q]);
                                }
                            step_rows<MB, RPL>(acc[h], xc + 4 * (jstat + p), XS, b0, b1);
                        }
                    }
                }
            }
            // Per row: partial w = ((acc0 + acc1) + acc2) + ... in chain order (chain a on lane w LPP + a % LPP,
            // slot a / LPP), then ((p0 + p1) + p2) + ... on the row's first lane, then the decode kernel's epilogue
            // (store_tile) per element.
#pragma unroll
            for (int r = 0; r < RPL; ++r) {
                const int n = n0 + sub + r * RPW;   // weight row (output column) of this lane
#pragma unroll
                for (int mm = 0; mm < MB; ++mm) {
                    float part = acc[0][r][mm];
#pragma unroll
                    for (int a = 1; a < ACC; ++a) {
                        const float own = acc[a / LPP][r][mm];
                        part = __fadd_rn(part, (a % LPP) ? shfl_idx(own, (lane + a % LPP) & 31) : own);
                    }
                    float s = part;
#pragma unroll
                    for (int ww = 1; ww < WK; ++ww) s = __fadd_rn(s, shfl_idx(part, (lane + ww * LPP) & 31));
                    const int m = m0 + mm;
                    if constexpr (EPI == KT2N_TERN_EPI_SILU_MUL) {
                        const float su = shfl_idx(s, (lane + L) & 31);  // the up row's sum (odd row), on the gate's lane
                        if (active && cl == 0 && (sub & 1) == 0 && m < M && n < N)
                            Y[(size_t)m * ldy + (n >> 1)] = kt2n_tern::silu_mul_bf16(s, su);
                    } else if (active && cl == 0 && m < M && n < N) {
                        if constexpr (EPI == KT2N_TERN_EPI_RESIDUAL) {
                            const float res = __bfloat162float(R[(size_t)m * ldr + n]);
                            Y[(size_t)m * ldy + n] = __float2bfloat16_rn(__fadd_rn(res, kt2n_tern::bf16_round(s)));
                        } else {
                            Y[(size_t)m * ldy + n] = __float2bfloat16_rn(s);
                        }
                    }
                }
            }
        };
        uint4 wq[RPL][CPW][4];
        uint32_t sc[RPL][CPW][2];
        for (int n0 = gwarp * G; n0 < N; n0 += stride) {
            load(n0, wq, sc);
            body(n0, wq, sc);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// One decode row (M = 1) of the families whose chains x chunks fill a warp: qkv / o (8 partials x 4 chains, one chunk
// each) and gate_up (4 partials x 4 chains, two chunks each). Lane = (chunk half h, partial w, chain a); the lane's
// 64 activations (k = 256c + 64q + 4j + e for its chunk c = w CPW + h and steps j = a, a + 4, a + 8, a + 12) stay in
// registers for the whole kernel, so a row costs no shared-memory traffic. Warps loop over rows (gate_up: over
// gate/up row pairs). With two chunks per partial the halves form a pipeline over the warp's rows: half 0 runs the
// first chunk of row t while half 1 runs the second chunk of row t - 1, continuing that chain from the value half 0
// produced one iteration earlier (passed by a shuffle: the same unrounded fp32 chain).
#ifndef KT2N_TERN_CC_M1_WARPS
#define KT2N_TERN_CC_M1_WARPS 4
#endif
#ifndef KT2N_TERN_CC_M1_REGS
#define KT2N_TERN_CC_M1_REGS 128
#endif
constexpr int kWarps1 = KT2N_TERN_CC_M1_WARPS;
constexpr int kThreads1 = 32 * kWarps1;
constexpr int kMinBlocks1 =
    65536 / (KT2N_TERN_CC_M1_REGS * kThreads1) > 0 ? 65536 / (KT2N_TERN_CC_M1_REGS * kThreads1) : 1;

// RPL weight rows per lane and iteration (each activation register then feeds RPL FMA chains): gate_up always runs
// its gate and up rows together (RPL 2); qkv / o run RPL rows n, n + 1, ... of the warp's group.
// PF1: the next iteration's weights are loaded before the current one runs (two register sets).
#ifndef KT2N_TERN_CC_M1_RPL
#define KT2N_TERN_CC_M1_RPL 2
#endif
#ifndef KT2N_TERN_CC_M1_PF
#define KT2N_TERN_CC_M1_PF 0
#endif
template <int WK, int EPI, int RPL, bool PF1>
__global__ void __launch_bounds__(kThreads1, kMinBlocks1)
dec_cc_m1_kernel(const bf16_t* X, const uint4* __restrict__ Wt, const uint16_t* __restrict__ St, bf16_t* Y,
                 const bf16_t* R, int N) {
    constexpr int C = 8;                        // K = 2048
    constexpr int CPW = C / WK;                 // chunks per partial: 1 (qkv, o) or 2 (gate_up)
    constexpr int CL = 32 / CPW;                // chain lanes per half
    static_assert(WK * 4 * CPW == 32, "one lane per (chain, chunk)");
    static_assert(CPW == 1 || (EPI == KT2N_TERN_EPI_SILU_MUL && RPL == 2), "gate_up: gate and up rows together");
    static_assert(EPI != KT2N_TERN_EPI_SILU_MUL || RPL == 2, "SILU_MUL pairs rows 2i, 2i + 1");
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int half = CPW == 2 ? lane / CL : 0;
    const int chain = lane % CL;
    const int a = chain & 3;
    const int c = (chain >> 2) * CPW + half;    // this lane's chunk
    const int gwarp = blockIdx.x * kWarps1 + warp, nwarps = gridDim.x * kWarps1;

    // X (one row, K = 2048) is converted to fp32 once per CTA in shared memory; each lane then loads its 64 values
    // into registers (loaded values stay in registers: the compiler does not re-derive them from bf16 per use).
    __shared__ __align__(16) float xs1[C * kChunk];
    for (int v = threadIdx.x; v < C * kChunk / 8; v += kThreads1) {
        const uint4 x8 = *reinterpret_cast<const uint4*>(X + 8 * v);
        *reinterpret_cast<float4*>(xs1 + 8 * v) = make_float4(lo16(x8.x), hi16(x8.x), lo16(x8.y), hi16(x8.y));
        *reinterpret_cast<float4*>(xs1 + 8 * v + 4) = make_float4(lo16(x8.z), hi16(x8.z), lo16(x8.w), hi16(x8.w));
    }
    __syncthreads();
    float xa[4][4][4];                          // [jo][q][e]: x[256c + 64q + 4(4jo + a) + e]
#pragma unroll
    for (int jo = 0; jo < 4; ++jo)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float4 v = *reinterpret_cast<const float4*>(xs1 + kChunk * c + 64 * q + 16 * jo + 4 * a);
            xa[jo][q][0] = v.x;
            xa[jo][q][1] = v.y;
            xa[jo][q][2] = v.z;
            xa[jo][q][3] = v.w;
        }

    // Row group of iteration t: rows base .. base + RPL - 1 (half 1 runs one group behind); -1 = none.
    const int groups = N / RPL;
    const int T = gwarp < groups ? (groups - gwarp + nwarps - 1) / nwarps : 0;
    auto base_of = [&](int t) -> int { return t < 0 || t >= T ? -1 : (gwarp + t * nwarps) * RPL; };
    auto load = [&](int base, uint4 (&W)[RPL][4], uint32_t (&S)[RPL][2]) {
        if (base < 0) return;
#pragma unroll
        for (int r = 0; r < RPL; ++r) {
            const int n = base + r;
            const size_t tc = (size_t)(n >> 3) * C + c;
            const int g = n & 7;
#pragma unroll
            for (int q = 0; q < 4; ++q) W[r][q] = __ldg(Wt + tc * 32 + g * 4 + q);
#pragma unroll
            for (int h = 0; h < 2; ++h) S[r][h] = __ldg(St + tc * 16 + g * 2 + h);
        }
    };
    // The chain's 4 steps in this lane's chunk for each of the RPL rows, continuing from acc[r].
    auto run = [&](float (&acc)[RPL], const uint4 (&W)[RPL][4], const uint32_t (&S)[RPL][2]) {
        uint32_t ta[RPL][2], tb[RPL][2];
#pragma unroll
        for (int r = 0; r < RPL; ++r)
#pragma unroll
            for (int h = 0; h < 2; ++h) kt2n_tern::scale_tables(S[r][h], ta[r][h], tb[r][h]);
#pragma unroll
        for (int jo = 0; jo < 4; ++jo) {
            uint32_t b0[RPL][4], b1[RPL][4];
#pragma unroll
            for (int r = 0; r < RPL; ++r)
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const uint32_t y = (kt2n_tern::word_of(W[r][q], jo) >> (2 * a)) & 0x03030303u;
                    pairs_of(sel_of(y), ta[r][q >> 1], tb[r][q >> 1], b0[r][q], b1[r][q]);
                }
#pragma unroll
            for (int q = 0; q < 4; ++q)  // k + 0, + 1 of every quarter
#pragma unroll
                for (int r = 0; r < RPL; ++r) {
                    acc[r] = fmaf(xa[jo][q][0], lo16(b0[r][q]), acc[r]);
                    acc[r] = fmaf(xa[jo][q][1], hi16(b0[r][q]), acc[r]);
                }
#pragma unroll
            for (int q = 0; q < 4; ++q)  // then k + 2, + 3
#pragma unroll
                for (int r = 0; r < RPL; ++r) {
                    acc[r] = fmaf(xa[jo][q][2], lo16(b1[r][q]), acc[r]);
                    acc[r] = fmaf(xa[jo][q][3], hi16(b1[r][q]), acc[r]);
                }
        }
    };
    float carry[RPL];  // half 1: the first-chunk values of the rows it runs next
#pragma unroll
    for (int r = 0; r < RPL; ++r) carry[r] = 0.f;
    auto iteration = [&](int t, const uint4 (&W)[RPL][4], const uint32_t (&S)[RPL][2]) {
        const int base = base_of(t - half);
        float acc[RPL];
#pragma unroll
        for (int r = 0; r < RPL; ++r) acc[r] = half ? carry[r] : 0.f;
        if (base >= 0) run(acc, W, S);
        // The rows finished in this iteration live on the last half's lanes. Partial w = ((c0 + c1) + c2) + c3 on
        // lane 4w (of that half), then ((p0 + p1) + p2) + ... on the half's first lane, then the decode kernel's
        // epilogue.
        float sum[RPL];
#pragma unroll
        for (int r = 0; r < RPL; ++r) {
            float part = acc[r];
#pragma unroll
            for (int aa = 1; aa < 4; ++aa) part = __fadd_rn(part, shfl_idx(acc[r], (lane + aa) & 31));
            float s = part;
#pragma unroll
            for (int ww = 1; ww < WK; ++ww) s = __fadd_rn(s, shfl_idx(part, (lane + 4 * ww) & 31));
            sum[r] = s;
        }
        if (half == CPW - 1 && chain == 0 && base >= 0) {
            if constexpr (EPI == KT2N_TERN_EPI_SILU_MUL) {
                Y[base >> 1] = kt2n_tern::silu_mul_bf16(sum[0], sum[1]);
            } else {
#pragma unroll
                for (int r = 0; r < RPL; ++r) {
                    const int n = base + r;
                    if constexpr (EPI == KT2N_TERN_EPI_RESIDUAL)
                        Y[n] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(R[n]), kt2n_tern::bf16_round(sum[r])));
                    else
                        Y[n] = __float2bfloat16_rn(sum[r]);
                }
            }
        }
        if constexpr (CPW == 2) {
#pragma unroll
            for (int r = 0; r < RPL; ++r) carry[r] = shfl_idx(acc[r], lane ^ CL);  // half 0's chains -> half 1
        }
    };
    const int iters = T + CPW - 1;
    uint4 wa[RPL][4];
    uint32_t sa[RPL][2];
    if constexpr (PF1) {
        uint4 wb[RPL][4];
        uint32_t sb[RPL][2];
        load(base_of(0 - half), wa, sa);
        for (int t = 0; t < iters; t += 2) {
            load(base_of(t + 1 - half), wb, sb);
            iteration(t, wa, sa);
            if (t + 1 >= iters) break;
            load(base_of(t + 2 - half), wa, sa);
            iteration(t + 1, wb, sb);
        }
    } else {
        for (int t = 0; t < iters; ++t) {
            load(base_of(t - half), wa, sa);
            iteration(t, wa, sa);
        }
    }
}

// The kernel has no image for the current GPU (a pre-8.0 GPU outside the build's 6.1 / 7.5 targets, e.g. SM 6.0 /
// 7.0 / 7.2 in an `all` build, whose main library does cover it).
inline bool no_image(cudaError_t e) {
#if defined(KT2N_HIP)
    return e == hipErrorNoBinaryForGpu || e == hipErrorInvalidDeviceFunction;
#else
    return e == cudaErrorNoKernelImageForDevice || e == cudaErrorInvalidDeviceFunction;
#endif
}

// Persistent grid size: CTAs of `kern` resident on the current GPU at once, queried once per kernel and device
// (cache in the caller's thread_local pair). ctas = 0: no image for this GPU, cached too; the caller declines.
template <typename Kern>
cudaError_t resident_ctas(Kern kern, int threads, int& cached_dev, int& cached_ctas, int& ctas) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    if (dev != cached_dev) {
        int per_sm = 0, sms = 0;
        e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, threads, 0);
        if (no_image(e)) {
            (void)cudaGetLastError();  // not sticky; cleared so later error checks of the caller start clean
            cached_dev = dev;
            cached_ctas = 0;
        } else {
            if (e == cudaSuccess) e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
            if (e != cudaSuccess) return e;
            cached_ctas = (per_sm > 0 ? per_sm : 1) * sms;
            cached_dev = dev;
        }
    }
    ctas = cached_ctas;
    return cudaSuccess;
}

template <int WK, int EPI>
cudaError_t go_m1(const void* x, const void* codes, const void* scales, int n, void* y, const void* residual,
                  cudaStream_t stream) {
    constexpr int RPL = (WK == 4 || EPI == KT2N_TERN_EPI_SILU_MUL) ? 2 : KT2N_TERN_CC_M1_RPL;
    constexpr auto kern = dec_cc_m1_kernel<WK, EPI, RPL, KT2N_TERN_CC_M1_PF != 0>;
    if (n % (8 * RPL)) return cudaErrorInvalidConfiguration;
    static thread_local int cached_dev = -1, cached_ctas = 0;
    int ctas = 0;
    const cudaError_t e = resident_ctas(kern, kThreads1, cached_dev, cached_ctas, ctas);
    if (e != cudaSuccess) return e;
    if (ctas == 0) return cudaErrorInvalidConfiguration;  // no image for this GPU: the decode kernel runs
    const int groups = (n / RPL + kWarps1 - 1) / kWarps1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(groups < ctas ? groups : ctas);
    cfg.blockDim = dim3(kThreads1);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    return cudaLaunchKernelEx(&cfg, kern, static_cast<const bf16_t*>(x), static_cast<const uint4*>(codes),
                              static_cast<const uint16_t*>(scales), static_cast<bf16_t*>(y),
                              static_cast<const bf16_t*>(residual), n);
}

// Rows per pass: the shared-memory budget (fp32 rows of K) keeps several CTAs per SM.
constexpr int max_rows_per_pass(int c) { return c * kChunk <= 2048 ? 4 : 1; }

template <int C, int WK, int ACC, int MB, int EPI, int RPL = 1>
cudaError_t go(const void* x, int ldx, int m, const void* codes, const void* scales, int n, void* y, int ldy,
               const void* residual, int ldr, cudaStream_t stream) {
    constexpr auto kern = dec_cc_kernel<C, WK, ACC, MB, EPI, RPL>;
    constexpr int L = WK * (ACC >= 2 ? ACC / 2 : 1), G = (32 / L) * RPL;
    if (n % G) return cudaErrorInvalidConfiguration;
    // Persistent grid: as many CTAs as fit on the GPU at once (occupancy queried once per kernel and device), never
    // more than there are row groups.
    static thread_local int cached_dev = -1, cached_ctas = 0;
    int ctas = 0;
    const cudaError_t e = resident_ctas(kern, kThreads, cached_dev, cached_ctas, ctas);
    if (e != cudaSuccess) return e;
    if (ctas == 0) return cudaErrorInvalidConfiguration;  // no image for this GPU: the decode kernel runs
    const int groups = (n / G + kWarps - 1) / kWarps;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(groups < ctas ? groups : ctas);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    return cudaLaunchKernelEx(&cfg, kern, static_cast<const bf16_t*>(x), ldx, m, static_cast<const uint4*>(codes),
                              static_cast<const uint16_t*>(scales), static_cast<bf16_t*>(y), ldy,
                              static_cast<const bf16_t*>(residual), ldr, n);
}

// One decode row: two weight rows per lane on SM 7.5 (halves the shared-memory loads, the limit there); one on
// SM 6.x, where the larger loop body measured slower (GTX 1080 Ti: down 35 -> 82 us).
#ifndef KT2N_TERN_CC_RPL_ARCH
#define KT2N_TERN_CC_RPL_ARCH 7
#endif
inline bool two_rows_per_lane() { return kt2n::cuda::current_caps().major == KT2N_TERN_CC_RPL_ARCH; }

template <int C, int WK, int ACC, int EPI>
cudaError_t go_rows(int mb, const void* x, int ldx, int m, const void* codes, const void* scales, int n, void* y,
                    int ldy, const void* residual, int ldr, cudaStream_t stream) {
    if constexpr (max_rows_per_pass(C) >= 4) {
        switch (mb) {
            case 1:
                if (two_rows_per_lane())
                    return go<C, WK, ACC, 1, EPI, 2>(x, ldx, m, codes, scales, n, y, ldy, residual, ldr, stream);
                return go<C, WK, ACC, 1, EPI>(x, ldx, m, codes, scales, n, y, ldy, residual, ldr, stream);
            case 2: return go<C, WK, ACC, 2, EPI>(x, ldx, m, codes, scales, n, y, ldy, residual, ldr, stream);
            case 3: return go<C, WK, ACC, 3, EPI>(x, ldx, m, codes, scales, n, y, ldy, residual, ldr, stream);
            case 4: return go<C, WK, ACC, 4, EPI>(x, ldx, m, codes, scales, n, y, ldy, residual, ldr, stream);
            default: return cudaErrorInvalidConfiguration;
        }
    } else {
        if (mb != 1) return cudaErrorInvalidConfiguration;
        if (two_rows_per_lane())
            return go<C, WK, ACC, 1, EPI, 2>(x, ldx, m, codes, scales, n, y, ldy, residual, ldr, stream);
        return go<C, WK, ACC, 1, EPI>(x, ldx, m, codes, scales, n, y, ldy, residual, ldr, stream);
    }
}

template <int C, int WK, int ACC>
cudaError_t go_epi(int epi, int mb, const void* x, int ldx, int m, const void* codes, const void* scales, int n,
                   void* y, int ldy, const void* residual, int ldr, cudaStream_t stream) {
    constexpr int L = WK * (ACC >= 2 ? ACC / 2 : 1), RPW = 32 / L;
    if (n % 8 || n % RPW) return cudaErrorInvalidConfiguration;
    switch (epi) {
        case KT2N_TERN_EPI_STORE:
            return go_rows<C, WK, ACC, KT2N_TERN_EPI_STORE>(mb, x, ldx, m, codes, scales, n, y, ldy, residual, ldr,
                                                             stream);
        case KT2N_TERN_EPI_SILU_MUL:
            if constexpr (RPW % 2 == 0)
                return go_rows<C, WK, ACC, KT2N_TERN_EPI_SILU_MUL>(mb, x, ldx, m, codes, scales, n, y, ldy,
                                                                    residual, ldr, stream);
            return cudaErrorInvalidConfiguration;
        case KT2N_TERN_EPI_RESIDUAL:
            return go_rows<C, WK, ACC, KT2N_TERN_EPI_RESIDUAL>(mb, x, ldx, m, codes, scales, n, y, ldy, residual,
                                                                ldr, stream);
        default: return cudaErrorInvalidConfiguration;
    }
}

std::atomic<unsigned long> g_launches{0};

cudaError_t launch_impl(int c, int wk, int acc, int epi, const void* x, int ldx, int m, const void* codes,
                        const void* scales, int n, void* y, int ldy, const void* residual, int ldr,
                        cudaStream_t stream) {
    // Above kMaxRows decode rows the caller keeps the old kernel (tern_dec_cc.h).
    const kt2n::cuda::DeviceCaps caps = kt2n::cuda::current_caps();
    if (m < 1 || m > kMaxRows || n <= 0) return cudaErrorInvalidConfiguration;
    const int cap = max_rows_per_pass(c);
    const int passes = (m + cap - 1) / cap;
    const int mb = (m + passes - 1) / passes;  // rows per pass, fewest padded rows
#ifndef KT2N_TERN_CC_M1
#define KT2N_TERN_CC_M1 1
#endif
    // SM 7.5: the register-resident one-row kernel for the 2048-wide families (gate_up with its SILU_MUL epilogue);
    // gate_up also for 2..4 rows, one launch per row (the shared-memory kernel is load-bound there: 2 rows of
    // gate_up 151 -> 2 x 44 us on the RTX 2060). Measured slower than the shared-memory kernel on SM 6.x.
#ifndef KT2N_TERN_CC_M1_ALL
#define KT2N_TERN_CC_M1_ALL 0
#endif
    if (KT2N_TERN_CC_M1 && (KT2N_TERN_CC_M1_ALL || caps.major == 7) && c == 8 && acc == 4 && n % 16 == 0) {
        if (m == 1 && wk == 8 && epi == KT2N_TERN_EPI_STORE)
            return go_m1<8, KT2N_TERN_EPI_STORE>(x, codes, scales, n, y, residual, stream);
        if (m == 1 && wk == 8 && epi == KT2N_TERN_EPI_RESIDUAL)
            return go_m1<8, KT2N_TERN_EPI_RESIDUAL>(x, codes, scales, n, y, residual, stream);
        if (wk == 4 && epi == KT2N_TERN_EPI_SILU_MUL) {
            for (int r = 0; r < m; ++r) {
                const cudaError_t e = go_m1<4, KT2N_TERN_EPI_SILU_MUL>(
                    static_cast<const bf16_t*>(x) + (size_t)r * ldx, codes, scales, n,
                    static_cast<bf16_t*>(y) + (size_t)r * ldy, residual, stream);
                if (e != cudaSuccess) return e;
            }
            return cudaSuccess;
        }
    }
    // The decode step's families with their default splits (tern_proj.cu family table).
    if (c == 8 && wk == 8 && acc == 4)
        return go_epi<8, 8, 4>(epi, mb, x, ldx, m, codes, scales, n, y, ldy, residual, ldr, stream);
    if (c == 8 && wk == 4 && acc == 4)
        return go_epi<8, 4, 4>(epi, mb, x, ldx, m, codes, scales, n, y, ldy, residual, ldr, stream);
    if (c == 24 && wk == 12 && acc == 2)
        return go_epi<24, 12, 2>(epi, mb, x, ldx, m, codes, scales, n, y, ldy, residual, ldr, stream);
    return cudaErrorInvalidConfiguration;
}

}  // namespace

cudaError_t launch(int c, int wk, int acc, int epi, const void* x, int ldx, int m, const void* codes,
                   const void* scales, int n, void* y, int ldy, const void* residual, int ldr, cudaStream_t stream) {
    const cudaError_t e = launch_impl(c, wk, acc, epi, x, ldx, m, codes, scales, n, y, ldy, residual, ldr, stream);
    if (e == cudaSuccess) g_launches.fetch_add(1, std::memory_order_relaxed);
    return e;
}

bool available() {
    cudaFuncAttributes attr{};
    const cudaError_t e = cudaFuncGetAttributes(&attr, dec_cc_kernel<8, 8, 4, 1, KT2N_TERN_EPI_STORE, 1>);
    if (e == cudaSuccess) return true;
    (void)cudaGetLastError();
    return false;
}

unsigned long launches() { return g_launches.load(std::memory_order_relaxed); }

}  // namespace kt2n_tern_cc
