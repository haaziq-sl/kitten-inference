#include "../ops_neon.hpp"
#include "profile_internal.hpp"
#include "work_split.hpp"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

// ──────────────────────────────────────────────────────────────
// FP32 GEMM kernels — weight packing + SGEMM + gemm_fp32.
// Weight packing: [C_out, K] → [C_out_pad/TILE, K, TILE]
// ──────────────────────────────────────────────────────────────

// Tile = number of output channels per co_blk (= SIMD width for floats)
#if defined(__AVX512F__)
static constexpr int TILE = 16;
#elif defined(__AVX2__)
static constexpr int TILE = 8;
#else
static constexpr int TILE = 4;   // NEON / scalar
#endif

// Number of M-rows in main tile
static constexpr int MR = 8;


// ──────────────────────────────────────────────────────────────
// pack_weights_f32
//   w:     [C_out, K]  (K = C_in * kH * kW, NHWC-reordered)
//   →      [Co_t, K, TILE]  where Co_t = ceil(C_out / TILE)
// Zero-padded for C_out not a multiple of TILE.
// ──────────────────────────────────────────────────────────────
float* pack_weights_f32(const float* w, int C_out, int K)
{
    const int Co_t = (C_out + TILE - 1) / TILE;
    const size_t sz = (size_t)Co_t * K * TILE;
    float* packed = new float[sz]();   // zero-initialised
    for (int co_blk = 0; co_blk < Co_t; ++co_blk) {
        for (int k = 0; k < K; ++k) {
            for (int oc = 0; oc < TILE; ++oc) {
                const int co = co_blk * TILE + oc;
                if (co < C_out)
                    packed[(size_t)co_blk * K * TILE + k * TILE + oc] = w[co * K + k];
            }
        }
    }
    return packed;
}

void free_packed_f32(float* p)
{
    delete[] p;
}

size_t packed_f32_elems(int C_out, int K)
{
    const int Co_t = (C_out + TILE - 1) / TILE;
    return (size_t)Co_t * K * TILE;
}

// ──────────────────────────────────────────────────────────────
// pack_merged_weights_f32
//   w: [C_out, kernel_size, C_in_g]  (NHWC layout after reorder)
//   → B_merged: [Co_t, kernel_size, C_in_g, TILE]
//   Access: B_merged[(co_blk * kernel_size + kp) * C_in_g * TILE + k * TILE + oc]
// ──────────────────────────────────────────────────────────────
float* pack_merged_weights_f32(const float* w, int C_out, int C_in_g, int kernel_size)
{
    const int Co_t = (C_out + TILE - 1) / TILE;
    const size_t sz = (size_t)Co_t * kernel_size * C_in_g * TILE;
    float* packed = new float[sz]();
    for (int co_blk = 0; co_blk < Co_t; ++co_blk) {
        for (int kp = 0; kp < kernel_size; ++kp) {
            float* dst = packed + ((size_t)co_blk * kernel_size + kp) * C_in_g * TILE;
            for (int k = 0; k < C_in_g; ++k) {
                for (int oc = 0; oc < TILE; ++oc) {
                    const int co = co_blk * TILE + oc;
                    if (co < C_out)
                        dst[k * TILE + oc] = w[(size_t)co * kernel_size * C_in_g + (size_t)kp * C_in_g + k];
                }
            }
        }
    }
    return packed;
}

// ──────────────────────────────────────────────────────────────
// sgemm_f32
//   A:        [M, K]          row-major float
//   B_packed: [Co_t, K, TILE] packed floats
//   bias:     [N]  or nullptr
//   C:        [M, N]          row-major float (output)
//   relu:     apply max(0, ·) after bias add
//
// Recursive OMP dispatch: call with in_parallel=false from outside
// a parallel region; the function forks a team and calls itself
// with in_parallel=true inside.  Inside the team each thread
// independently computes its share of output tiles.
//
// Hybrid M/N partition (mirrors INT8 gemm_int8_neon):
//   M >= N  →  M-partition: thread tid owns m-tiles [mi_s, mi_e)
//   M <  N  →  N-partition: thread tid owns co_blks [co_s, co_e)
// ──────────────────────────────────────────────────────────────
void sgemm_f32(
    const float* A,
    const float* B_packed,
    const float* bias,
    float*       C,
    bool         relu,
    int M, int K, int N,
    bool in_parallel,
    StreamHandle /* stream */)
{
    const bool _profile_sgemm = g_ops_profile_enabled && !in_parallel;
    const double _t0_sgemm = _profile_sgemm ? now_ms() : 0.0;

    if (!in_parallel) {
#ifdef _OPENMP
        // Only fork a new team if NOT already inside a parallel region.
        // When called inside an existing parallel region (e.g. from the Winograd
        // pos-parallel loop), we compute the full GEMM single-threaded on the
        // calling thread (tid=0, nT=1 override below).
        //
        // Also skip forking for tiny GEMMs (M=1 LSTM scan steps): fork overhead
        // (~5-10 µs) dominates over the compute benefit for < 2M FLOPs.
        if (!omp_in_parallel()) {
            const int nthreads = omp_get_max_threads();
            if (nthreads > 1 && (size_t)M * K * N >= 2000000UL) {
#pragma omp parallel num_threads(nthreads)
                sgemm_f32(A, B_packed, bias, C, relu, M, K, N, /*in_parallel=*/true, nullptr);
                if (_profile_sgemm) g_sgemm_total_ms += now_ms() - _t0_sgemm;
                return;
            }
        }
        // Inside a parallel region or single-threaded → run as tid=0, nT=1
#endif
    }

#ifdef _OPENMP
    const int tid = in_parallel ? omp_get_thread_num() : 0;
    const int nT  = in_parallel ? omp_get_num_threads() : 1;
#else
    const int tid = 0, nT = 1;
#endif

    const int Co_t    = (N + TILE - 1) / TILE;
    const int m_tiles = M / MR;

    const bool n_part = (nT > 1) && (M < N);
    kt_split_tiles(n_part, m_tiles, Co_t, tid, nT, in_parallel,
                   [&](const int mi_s, const int mi_e, const int co_s, const int co_e, const bool tail) {

#ifdef __ARM_NEON
    // ── NEON path: mi-outer, co_blk-inner (MR=8 rows) ────────────────────
    for (int mi = mi_s; mi < mi_e; ++mi) {
        const int m = mi * MR;
        const float* a0 = A + (size_t)(m+0) * K;
        const float* a1 = A + (size_t)(m+1) * K;
        const float* a2 = A + (size_t)(m+2) * K;
        const float* a3 = A + (size_t)(m+3) * K;
        const float* a4 = A + (size_t)(m+4) * K;
        const float* a5 = A + (size_t)(m+5) * K;
        const float* a6 = A + (size_t)(m+6) * K;
        const float* a7 = A + (size_t)(m+7) * K;

        int co_blk = co_s;

        // ── NEON: 8 rows × 8 cols (dual co_blk) ──────────────────────
        {
            const float32x4_t zero_v = vdupq_n_f32(0.f);
            for (; co_blk + 1 < co_e && (co_blk + 2) * TILE <= N; co_blk += 2) {
                const float* bp0 = B_packed + (size_t)co_blk * K * 4;
                const float* bp1 = B_packed + (size_t)(co_blk + 1) * K * 4;
                const int n0a = co_blk * 4, n0b = (co_blk + 1) * 4;

                float32x4_t acc0a=vdupq_n_f32(0), acc0b=vdupq_n_f32(0);
                float32x4_t acc1a=vdupq_n_f32(0), acc1b=vdupq_n_f32(0);
                float32x4_t acc2a=vdupq_n_f32(0), acc2b=vdupq_n_f32(0);
                float32x4_t acc3a=vdupq_n_f32(0), acc3b=vdupq_n_f32(0);
                float32x4_t acc4a=vdupq_n_f32(0), acc4b=vdupq_n_f32(0);
                float32x4_t acc5a=vdupq_n_f32(0), acc5b=vdupq_n_f32(0);
                float32x4_t acc6a=vdupq_n_f32(0), acc6b=vdupq_n_f32(0);
                float32x4_t acc7a=vdupq_n_f32(0), acc7b=vdupq_n_f32(0);

                int k = 0;
                for (; k + 3 < K; k += 4) {
                    float32x4_t ba0 = vld1q_f32(bp0 + k*4 +  0);
                    float32x4_t ba1 = vld1q_f32(bp0 + k*4 +  4);
                    float32x4_t ba2 = vld1q_f32(bp0 + k*4 +  8);
                    float32x4_t ba3 = vld1q_f32(bp0 + k*4 + 12);
                    float32x4_t bb0 = vld1q_f32(bp1 + k*4 +  0);
                    float32x4_t bb1 = vld1q_f32(bp1 + k*4 +  4);
                    float32x4_t bb2 = vld1q_f32(bp1 + k*4 +  8);
                    float32x4_t bb3 = vld1q_f32(bp1 + k*4 + 12);
                    float32x4_t av0 = vld1q_f32(a0 + k);
                    float32x4_t av1 = vld1q_f32(a1 + k);
                    float32x4_t av2 = vld1q_f32(a2 + k);
                    float32x4_t av3 = vld1q_f32(a3 + k);
                    float32x4_t av4 = vld1q_f32(a4 + k);
                    float32x4_t av5 = vld1q_f32(a5 + k);
                    float32x4_t av6 = vld1q_f32(a6 + k);
                    float32x4_t av7 = vld1q_f32(a7 + k);
                    // Row 0
                    acc0a = vfmaq_laneq_f32(acc0a, ba0, av0, 0);
                    acc0a = vfmaq_laneq_f32(acc0a, ba1, av0, 1);
                    acc0a = vfmaq_laneq_f32(acc0a, ba2, av0, 2);
                    acc0a = vfmaq_laneq_f32(acc0a, ba3, av0, 3);
                    acc0b = vfmaq_laneq_f32(acc0b, bb0, av0, 0);
                    acc0b = vfmaq_laneq_f32(acc0b, bb1, av0, 1);
                    acc0b = vfmaq_laneq_f32(acc0b, bb2, av0, 2);
                    acc0b = vfmaq_laneq_f32(acc0b, bb3, av0, 3);
                    // Row 1
                    acc1a = vfmaq_laneq_f32(acc1a, ba0, av1, 0);
                    acc1a = vfmaq_laneq_f32(acc1a, ba1, av1, 1);
                    acc1a = vfmaq_laneq_f32(acc1a, ba2, av1, 2);
                    acc1a = vfmaq_laneq_f32(acc1a, ba3, av1, 3);
                    acc1b = vfmaq_laneq_f32(acc1b, bb0, av1, 0);
                    acc1b = vfmaq_laneq_f32(acc1b, bb1, av1, 1);
                    acc1b = vfmaq_laneq_f32(acc1b, bb2, av1, 2);
                    acc1b = vfmaq_laneq_f32(acc1b, bb3, av1, 3);
                    // Row 2
                    acc2a = vfmaq_laneq_f32(acc2a, ba0, av2, 0);
                    acc2a = vfmaq_laneq_f32(acc2a, ba1, av2, 1);
                    acc2a = vfmaq_laneq_f32(acc2a, ba2, av2, 2);
                    acc2a = vfmaq_laneq_f32(acc2a, ba3, av2, 3);
                    acc2b = vfmaq_laneq_f32(acc2b, bb0, av2, 0);
                    acc2b = vfmaq_laneq_f32(acc2b, bb1, av2, 1);
                    acc2b = vfmaq_laneq_f32(acc2b, bb2, av2, 2);
                    acc2b = vfmaq_laneq_f32(acc2b, bb3, av2, 3);
                    // Row 3
                    acc3a = vfmaq_laneq_f32(acc3a, ba0, av3, 0);
                    acc3a = vfmaq_laneq_f32(acc3a, ba1, av3, 1);
                    acc3a = vfmaq_laneq_f32(acc3a, ba2, av3, 2);
                    acc3a = vfmaq_laneq_f32(acc3a, ba3, av3, 3);
                    acc3b = vfmaq_laneq_f32(acc3b, bb0, av3, 0);
                    acc3b = vfmaq_laneq_f32(acc3b, bb1, av3, 1);
                    acc3b = vfmaq_laneq_f32(acc3b, bb2, av3, 2);
                    acc3b = vfmaq_laneq_f32(acc3b, bb3, av3, 3);
                    // Row 4
                    acc4a = vfmaq_laneq_f32(acc4a, ba0, av4, 0);
                    acc4a = vfmaq_laneq_f32(acc4a, ba1, av4, 1);
                    acc4a = vfmaq_laneq_f32(acc4a, ba2, av4, 2);
                    acc4a = vfmaq_laneq_f32(acc4a, ba3, av4, 3);
                    acc4b = vfmaq_laneq_f32(acc4b, bb0, av4, 0);
                    acc4b = vfmaq_laneq_f32(acc4b, bb1, av4, 1);
                    acc4b = vfmaq_laneq_f32(acc4b, bb2, av4, 2);
                    acc4b = vfmaq_laneq_f32(acc4b, bb3, av4, 3);
                    // Row 5
                    acc5a = vfmaq_laneq_f32(acc5a, ba0, av5, 0);
                    acc5a = vfmaq_laneq_f32(acc5a, ba1, av5, 1);
                    acc5a = vfmaq_laneq_f32(acc5a, ba2, av5, 2);
                    acc5a = vfmaq_laneq_f32(acc5a, ba3, av5, 3);
                    acc5b = vfmaq_laneq_f32(acc5b, bb0, av5, 0);
                    acc5b = vfmaq_laneq_f32(acc5b, bb1, av5, 1);
                    acc5b = vfmaq_laneq_f32(acc5b, bb2, av5, 2);
                    acc5b = vfmaq_laneq_f32(acc5b, bb3, av5, 3);
                    // Row 6
                    acc6a = vfmaq_laneq_f32(acc6a, ba0, av6, 0);
                    acc6a = vfmaq_laneq_f32(acc6a, ba1, av6, 1);
                    acc6a = vfmaq_laneq_f32(acc6a, ba2, av6, 2);
                    acc6a = vfmaq_laneq_f32(acc6a, ba3, av6, 3);
                    acc6b = vfmaq_laneq_f32(acc6b, bb0, av6, 0);
                    acc6b = vfmaq_laneq_f32(acc6b, bb1, av6, 1);
                    acc6b = vfmaq_laneq_f32(acc6b, bb2, av6, 2);
                    acc6b = vfmaq_laneq_f32(acc6b, bb3, av6, 3);
                    // Row 7
                    acc7a = vfmaq_laneq_f32(acc7a, ba0, av7, 0);
                    acc7a = vfmaq_laneq_f32(acc7a, ba1, av7, 1);
                    acc7a = vfmaq_laneq_f32(acc7a, ba2, av7, 2);
                    acc7a = vfmaq_laneq_f32(acc7a, ba3, av7, 3);
                    acc7b = vfmaq_laneq_f32(acc7b, bb0, av7, 0);
                    acc7b = vfmaq_laneq_f32(acc7b, bb1, av7, 1);
                    acc7b = vfmaq_laneq_f32(acc7b, bb2, av7, 2);
                    acc7b = vfmaq_laneq_f32(acc7b, bb3, av7, 3);
                }
                for (; k < K; ++k) {
                    float32x4_t bav = vld1q_f32(bp0 + k * 4);
                    float32x4_t bbv = vld1q_f32(bp1 + k * 4);
                    acc0a = vfmaq_n_f32(acc0a, bav, a0[k]); acc0b = vfmaq_n_f32(acc0b, bbv, a0[k]);
                    acc1a = vfmaq_n_f32(acc1a, bav, a1[k]); acc1b = vfmaq_n_f32(acc1b, bbv, a1[k]);
                    acc2a = vfmaq_n_f32(acc2a, bav, a2[k]); acc2b = vfmaq_n_f32(acc2b, bbv, a2[k]);
                    acc3a = vfmaq_n_f32(acc3a, bav, a3[k]); acc3b = vfmaq_n_f32(acc3b, bbv, a3[k]);
                    acc4a = vfmaq_n_f32(acc4a, bav, a4[k]); acc4b = vfmaq_n_f32(acc4b, bbv, a4[k]);
                    acc5a = vfmaq_n_f32(acc5a, bav, a5[k]); acc5b = vfmaq_n_f32(acc5b, bbv, a5[k]);
                    acc6a = vfmaq_n_f32(acc6a, bav, a6[k]); acc6b = vfmaq_n_f32(acc6b, bbv, a6[k]);
                    acc7a = vfmaq_n_f32(acc7a, bav, a7[k]); acc7b = vfmaq_n_f32(acc7b, bbv, a7[k]);
                }

                float32x4_t bva = bias ? vld1q_f32(bias + n0a) : vdupq_n_f32(0.f);
                float32x4_t bvb = bias ? vld1q_f32(bias + n0b) : vdupq_n_f32(0.f);
                auto sv2 = [&](float32x4_t acca, float32x4_t accb, int row)
                    __attribute__((always_inline)) {
                    acca = vaddq_f32(acca, bva);
                    accb = vaddq_f32(accb, bvb);
                    if (relu) { acca = vmaxq_f32(acca, zero_v); accb = vmaxq_f32(accb, zero_v); }
                    vst1q_f32(C + (size_t)(m + row) * N + n0a, acca);
                    vst1q_f32(C + (size_t)(m + row) * N + n0b, accb);
                };
                sv2(acc0a, acc0b, 0); sv2(acc1a, acc1b, 1);
                sv2(acc2a, acc2b, 2); sv2(acc3a, acc3b, 3);
                sv2(acc4a, acc4b, 4); sv2(acc5a, acc5b, 5);
                sv2(acc6a, acc6b, 6); sv2(acc7a, acc7b, 7);
            }   // end dual co_blk loop
        }

        // ── NEON: 8 rows × 4 cols (single co_blk, handles odd remainder) ─
        for (; co_blk < co_e; ++co_blk) {
            const float* bptr   = B_packed + (size_t)co_blk * K * 4;
            const int    n0     = co_blk * 4;
            const int    n_valid = std::min(4, N - n0);

            float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f);
            float32x4_t acc2 = vdupq_n_f32(0.f), acc3 = vdupq_n_f32(0.f);
            float32x4_t acc4 = vdupq_n_f32(0.f), acc5 = vdupq_n_f32(0.f);
            float32x4_t acc6 = vdupq_n_f32(0.f), acc7 = vdupq_n_f32(0.f);

            int k = 0;
            for (; k + 3 < K; k += 4) {
                const float* bp = bptr + k * 4;
                float32x4_t bv0 = vld1q_f32(bp +  0);
                float32x4_t bv1 = vld1q_f32(bp +  4);
                float32x4_t bv2 = vld1q_f32(bp +  8);
                float32x4_t bv3 = vld1q_f32(bp + 12);
                float32x4_t av0 = vld1q_f32(a0 + k);
                float32x4_t av1 = vld1q_f32(a1 + k);
                float32x4_t av2 = vld1q_f32(a2 + k);
                float32x4_t av3 = vld1q_f32(a3 + k);
                float32x4_t av4 = vld1q_f32(a4 + k);
                float32x4_t av5 = vld1q_f32(a5 + k);
                float32x4_t av6 = vld1q_f32(a6 + k);
                float32x4_t av7 = vld1q_f32(a7 + k);
                acc0 = vfmaq_laneq_f32(acc0, bv0, av0, 0);
                acc0 = vfmaq_laneq_f32(acc0, bv1, av0, 1);
                acc0 = vfmaq_laneq_f32(acc0, bv2, av0, 2);
                acc0 = vfmaq_laneq_f32(acc0, bv3, av0, 3);
                acc1 = vfmaq_laneq_f32(acc1, bv0, av1, 0);
                acc1 = vfmaq_laneq_f32(acc1, bv1, av1, 1);
                acc1 = vfmaq_laneq_f32(acc1, bv2, av1, 2);
                acc1 = vfmaq_laneq_f32(acc1, bv3, av1, 3);
                acc2 = vfmaq_laneq_f32(acc2, bv0, av2, 0);
                acc2 = vfmaq_laneq_f32(acc2, bv1, av2, 1);
                acc2 = vfmaq_laneq_f32(acc2, bv2, av2, 2);
                acc2 = vfmaq_laneq_f32(acc2, bv3, av2, 3);
                acc3 = vfmaq_laneq_f32(acc3, bv0, av3, 0);
                acc3 = vfmaq_laneq_f32(acc3, bv1, av3, 1);
                acc3 = vfmaq_laneq_f32(acc3, bv2, av3, 2);
                acc3 = vfmaq_laneq_f32(acc3, bv3, av3, 3);
                acc4 = vfmaq_laneq_f32(acc4, bv0, av4, 0);
                acc4 = vfmaq_laneq_f32(acc4, bv1, av4, 1);
                acc4 = vfmaq_laneq_f32(acc4, bv2, av4, 2);
                acc4 = vfmaq_laneq_f32(acc4, bv3, av4, 3);
                acc5 = vfmaq_laneq_f32(acc5, bv0, av5, 0);
                acc5 = vfmaq_laneq_f32(acc5, bv1, av5, 1);
                acc5 = vfmaq_laneq_f32(acc5, bv2, av5, 2);
                acc5 = vfmaq_laneq_f32(acc5, bv3, av5, 3);
                acc6 = vfmaq_laneq_f32(acc6, bv0, av6, 0);
                acc6 = vfmaq_laneq_f32(acc6, bv1, av6, 1);
                acc6 = vfmaq_laneq_f32(acc6, bv2, av6, 2);
                acc6 = vfmaq_laneq_f32(acc6, bv3, av6, 3);
                acc7 = vfmaq_laneq_f32(acc7, bv0, av7, 0);
                acc7 = vfmaq_laneq_f32(acc7, bv1, av7, 1);
                acc7 = vfmaq_laneq_f32(acc7, bv2, av7, 2);
                acc7 = vfmaq_laneq_f32(acc7, bv3, av7, 3);
            }
            for (; k < K; ++k) {
                float32x4_t bv = vld1q_f32(bptr + k * 4);
                acc0 = vfmaq_n_f32(acc0, bv, a0[k]);
                acc1 = vfmaq_n_f32(acc1, bv, a1[k]);
                acc2 = vfmaq_n_f32(acc2, bv, a2[k]);
                acc3 = vfmaq_n_f32(acc3, bv, a3[k]);
                acc4 = vfmaq_n_f32(acc4, bv, a4[k]);
                acc5 = vfmaq_n_f32(acc5, bv, a5[k]);
                acc6 = vfmaq_n_f32(acc6, bv, a6[k]);
                acc7 = vfmaq_n_f32(acc7, bv, a7[k]);
            }

            if (n_valid == 4) {
                float32x4_t bias_v = bias ? vld1q_f32(bias + n0) : vdupq_n_f32(0.f);
                float32x4_t zero_v = vdupq_n_f32(0.f);
                auto sv = [&](float32x4_t acc, int row) __attribute__((always_inline)) {
                    acc = vaddq_f32(acc, bias_v);
                    if (relu) acc = vmaxq_f32(acc, zero_v);
                    vst1q_f32(C + (size_t)(m + row) * N + n0, acc);
                };
                sv(acc0, 0); sv(acc1, 1); sv(acc2, 2); sv(acc3, 3);
                sv(acc4, 4); sv(acc5, 5); sv(acc6, 6); sv(acc7, 7);
            } else {
                alignas(16) float tmp[MR][4];
                vst1q_f32(tmp[0], acc0); vst1q_f32(tmp[1], acc1);
                vst1q_f32(tmp[2], acc2); vst1q_f32(tmp[3], acc3);
                vst1q_f32(tmp[4], acc4); vst1q_f32(tmp[5], acc5);
                vst1q_f32(tmp[6], acc6); vst1q_f32(tmp[7], acc7);
                for (int r = 0; r < MR; ++r) {
                    float* cptr = C + (size_t)(m + r) * N + n0;
                    for (int oc = 0; oc < n_valid; ++oc) {
                        float v = tmp[r][oc] + (bias ? bias[n0 + oc] : 0.f);
                        cptr[oc] = relu ? std::max(0.f, v) : v;
                    }
                }
            }
        }   // end single co_blk loop

        // NEON path fully handled above.
    }   // end mi loop (NEON)

#else
    // ── Non-NEON: co_blk-outer, mi-inner (loop interchange for B cache reuse) ──

    // Determine dual_end: largest co_blk index (aligned to pairs) we can safely
    // process in the dual loop (both co_blks must be fully populated).
    int dual_end = co_s;
    while (dual_end + 1 < co_e && (dual_end + 2) * TILE <= N) dual_end += 2;

#if defined(__AVX512F__)
    // ── AVX-512: M-cache-blocked, co_blk-inner ─────────────────────────────
    // MC=32 m-tiles (256 rows) keeps A panel (~512KB) in L2 per core while
    // sweeping all co_blks; avoids re-fetching A from DRAM per co_blk.
    constexpr int MC = 32;
    for (int mi_blk = mi_s; mi_blk < mi_e; mi_blk += MC) {
    const int mi_blk_end = std::min(mi_blk + MC, mi_e);
    for (int co_blk = co_s; co_blk < dual_end; co_blk += 2) {
        const float* bpa = B_packed + (size_t)co_blk       * K * 16;
        const float* bpb = B_packed + (size_t)(co_blk + 1) * K * 16;
        const int n0a = co_blk * 16, n0b = (co_blk + 1) * 16;
        const __m512 bias_va = bias ? _mm512_loadu_ps(bias + n0a) : _mm512_setzero_ps();
        const __m512 bias_vb = bias ? _mm512_loadu_ps(bias + n0b) : _mm512_setzero_ps();
        const __m512 zero_v  = _mm512_setzero_ps();

        for (int mi = mi_blk; mi < mi_blk_end; ++mi) {
            const int m = mi * MR;
            const float* a0 = A + (size_t)(m+0) * K;
            const float* a1 = A + (size_t)(m+1) * K;
            const float* a2 = A + (size_t)(m+2) * K;
            const float* a3 = A + (size_t)(m+3) * K;
            const float* a4 = A + (size_t)(m+4) * K;
            const float* a5 = A + (size_t)(m+5) * K;
            const float* a6 = A + (size_t)(m+6) * K;
            const float* a7 = A + (size_t)(m+7) * K;

            __m512 acc0a=_mm512_setzero_ps(), acc0b=_mm512_setzero_ps();
            __m512 acc1a=_mm512_setzero_ps(), acc1b=_mm512_setzero_ps();
            __m512 acc2a=_mm512_setzero_ps(), acc2b=_mm512_setzero_ps();
            __m512 acc3a=_mm512_setzero_ps(), acc3b=_mm512_setzero_ps();
            __m512 acc4a=_mm512_setzero_ps(), acc4b=_mm512_setzero_ps();
            __m512 acc5a=_mm512_setzero_ps(), acc5b=_mm512_setzero_ps();
            __m512 acc6a=_mm512_setzero_ps(), acc6b=_mm512_setzero_ps();
            __m512 acc7a=_mm512_setzero_ps(), acc7b=_mm512_setzero_ps();

            int k = 0;
            for (; k + 3 < K; k += 4) {
                __m512 bva0=_mm512_loadu_ps(bpa+k*16+ 0), bva1=_mm512_loadu_ps(bpa+k*16+16);
                __m512 bva2=_mm512_loadu_ps(bpa+k*16+32), bva3=_mm512_loadu_ps(bpa+k*16+48);
                __m512 bvb0=_mm512_loadu_ps(bpb+k*16+ 0), bvb1=_mm512_loadu_ps(bpb+k*16+16);
                __m512 bvb2=_mm512_loadu_ps(bpb+k*16+32), bvb3=_mm512_loadu_ps(bpb+k*16+48);
#define FMA2_512(aa,ab,ar,b0,b1,b2,b3) do{ \
    __m512 _a0=_mm512_set1_ps((ar)[k+0]),_a1=_mm512_set1_ps((ar)[k+1]); \
    __m512 _a2=_mm512_set1_ps((ar)[k+2]),_a3=_mm512_set1_ps((ar)[k+3]); \
    aa=_mm512_fmadd_ps(_a0,b0,aa); ab=_mm512_fmadd_ps(_a0,bvb0,ab); \
    aa=_mm512_fmadd_ps(_a1,b1,aa); ab=_mm512_fmadd_ps(_a1,bvb1,ab); \
    aa=_mm512_fmadd_ps(_a2,b2,aa); ab=_mm512_fmadd_ps(_a2,bvb2,ab); \
    aa=_mm512_fmadd_ps(_a3,b3,aa); ab=_mm512_fmadd_ps(_a3,bvb3,ab); }while(0)
                FMA2_512(acc0a,acc0b,a0,bva0,bva1,bva2,bva3);
                FMA2_512(acc1a,acc1b,a1,bva0,bva1,bva2,bva3);
                FMA2_512(acc2a,acc2b,a2,bva0,bva1,bva2,bva3);
                FMA2_512(acc3a,acc3b,a3,bva0,bva1,bva2,bva3);
                FMA2_512(acc4a,acc4b,a4,bva0,bva1,bva2,bva3);
                FMA2_512(acc5a,acc5b,a5,bva0,bva1,bva2,bva3);
                FMA2_512(acc6a,acc6b,a6,bva0,bva1,bva2,bva3);
                FMA2_512(acc7a,acc7b,a7,bva0,bva1,bva2,bva3);
#undef FMA2_512
            }
            for (; k < K; ++k) {
                __m512 bva=_mm512_loadu_ps(bpa+k*16), bvb=_mm512_loadu_ps(bpb+k*16);
                __m512 _a0=_mm512_set1_ps(a0[k]), _a1=_mm512_set1_ps(a1[k]);
                __m512 _a2=_mm512_set1_ps(a2[k]), _a3=_mm512_set1_ps(a3[k]);
                __m512 _a4=_mm512_set1_ps(a4[k]), _a5=_mm512_set1_ps(a5[k]);
                __m512 _a6=_mm512_set1_ps(a6[k]), _a7=_mm512_set1_ps(a7[k]);
                acc0a=_mm512_fmadd_ps(_a0,bva,acc0a); acc0b=_mm512_fmadd_ps(_a0,bvb,acc0b);
                acc1a=_mm512_fmadd_ps(_a1,bva,acc1a); acc1b=_mm512_fmadd_ps(_a1,bvb,acc1b);
                acc2a=_mm512_fmadd_ps(_a2,bva,acc2a); acc2b=_mm512_fmadd_ps(_a2,bvb,acc2b);
                acc3a=_mm512_fmadd_ps(_a3,bva,acc3a); acc3b=_mm512_fmadd_ps(_a3,bvb,acc3b);
                acc4a=_mm512_fmadd_ps(_a4,bva,acc4a); acc4b=_mm512_fmadd_ps(_a4,bvb,acc4b);
                acc5a=_mm512_fmadd_ps(_a5,bva,acc5a); acc5b=_mm512_fmadd_ps(_a5,bvb,acc5b);
                acc6a=_mm512_fmadd_ps(_a6,bva,acc6a); acc6b=_mm512_fmadd_ps(_a6,bvb,acc6b);
                acc7a=_mm512_fmadd_ps(_a7,bva,acc7a); acc7b=_mm512_fmadd_ps(_a7,bvb,acc7b);
            }
            {
                auto sv2 = [&](__m512 acca, __m512 accb, int row) __attribute__((always_inline)) {
                    acca = _mm512_add_ps(acca, bias_va);
                    accb = _mm512_add_ps(accb, bias_vb);
                    if (relu) { acca = _mm512_max_ps(acca, zero_v); accb = _mm512_max_ps(accb, zero_v); }
                    _mm512_storeu_ps(C + (size_t)(m + row) * N + n0a, acca);
                    _mm512_storeu_ps(C + (size_t)(m + row) * N + n0b, accb);
                };
                sv2(acc0a,acc0b,0); sv2(acc1a,acc1b,1); sv2(acc2a,acc2b,2); sv2(acc3a,acc3b,3);
                sv2(acc4a,acc4b,4); sv2(acc5a,acc5b,5); sv2(acc6a,acc6b,6); sv2(acc7a,acc7b,7);
            }
        }   // end mi loop (dual AVX-512)
    }   // end dual co_blk loop (AVX-512)

    // ── AVX-512 single co_blk (8 rows × 16 cols), co_blk-outer mi-inner ──
    for (int co_blk = dual_end; co_blk < co_e; ++co_blk) {
        const float* bptr    = B_packed + (size_t)co_blk * K * 16;
        const int    n0      = co_blk * 16;
        const int    n_valid = std::min(16, N - n0);
        const __mmask16 mask = (n_valid == 16) ? 0xFFFF : ((__mmask16)1 << n_valid) - 1;
        const __m512 bias_v  = (bias && n_valid == 16) ? _mm512_loadu_ps(bias + n0) : _mm512_setzero_ps();
        const __m512 zero_v  = _mm512_setzero_ps();

        for (int mi = mi_blk; mi < mi_blk_end; ++mi) {
            const int m = mi * MR;
            const float* a0 = A + (size_t)(m+0) * K;
            const float* a1 = A + (size_t)(m+1) * K;
            const float* a2 = A + (size_t)(m+2) * K;
            const float* a3 = A + (size_t)(m+3) * K;
            const float* a4 = A + (size_t)(m+4) * K;
            const float* a5 = A + (size_t)(m+5) * K;
            const float* a6 = A + (size_t)(m+6) * K;
            const float* a7 = A + (size_t)(m+7) * K;

            // Dual-accumulator per row to break k-dependency chain
            __m512 acc0a=_mm512_setzero_ps(), acc0b=_mm512_setzero_ps();
            __m512 acc1a=_mm512_setzero_ps(), acc1b=_mm512_setzero_ps();
            __m512 acc2a=_mm512_setzero_ps(), acc2b=_mm512_setzero_ps();
            __m512 acc3a=_mm512_setzero_ps(), acc3b=_mm512_setzero_ps();
            __m512 acc4a=_mm512_setzero_ps(), acc4b=_mm512_setzero_ps();
            __m512 acc5a=_mm512_setzero_ps(), acc5b=_mm512_setzero_ps();
            __m512 acc6a=_mm512_setzero_ps(), acc6b=_mm512_setzero_ps();
            __m512 acc7a=_mm512_setzero_ps(), acc7b=_mm512_setzero_ps();

            int k = 0;
            for (; k + 3 < K; k += 4) {
                const float* bp = bptr + k * 16;
                __m512 bv0 = _mm512_loadu_ps(bp +  0);
                __m512 bv1 = _mm512_loadu_ps(bp + 16);
                __m512 bv2 = _mm512_loadu_ps(bp + 32);
                __m512 bv3 = _mm512_loadu_ps(bp + 48);
                acc0a=_mm512_fmadd_ps(_mm512_set1_ps(a0[k+0]),bv0,acc0a); acc0b=_mm512_fmadd_ps(_mm512_set1_ps(a0[k+1]),bv1,acc0b);
                acc0a=_mm512_fmadd_ps(_mm512_set1_ps(a0[k+2]),bv2,acc0a); acc0b=_mm512_fmadd_ps(_mm512_set1_ps(a0[k+3]),bv3,acc0b);
                acc1a=_mm512_fmadd_ps(_mm512_set1_ps(a1[k+0]),bv0,acc1a); acc1b=_mm512_fmadd_ps(_mm512_set1_ps(a1[k+1]),bv1,acc1b);
                acc1a=_mm512_fmadd_ps(_mm512_set1_ps(a1[k+2]),bv2,acc1a); acc1b=_mm512_fmadd_ps(_mm512_set1_ps(a1[k+3]),bv3,acc1b);
                acc2a=_mm512_fmadd_ps(_mm512_set1_ps(a2[k+0]),bv0,acc2a); acc2b=_mm512_fmadd_ps(_mm512_set1_ps(a2[k+1]),bv1,acc2b);
                acc2a=_mm512_fmadd_ps(_mm512_set1_ps(a2[k+2]),bv2,acc2a); acc2b=_mm512_fmadd_ps(_mm512_set1_ps(a2[k+3]),bv3,acc2b);
                acc3a=_mm512_fmadd_ps(_mm512_set1_ps(a3[k+0]),bv0,acc3a); acc3b=_mm512_fmadd_ps(_mm512_set1_ps(a3[k+1]),bv1,acc3b);
                acc3a=_mm512_fmadd_ps(_mm512_set1_ps(a3[k+2]),bv2,acc3a); acc3b=_mm512_fmadd_ps(_mm512_set1_ps(a3[k+3]),bv3,acc3b);
                acc4a=_mm512_fmadd_ps(_mm512_set1_ps(a4[k+0]),bv0,acc4a); acc4b=_mm512_fmadd_ps(_mm512_set1_ps(a4[k+1]),bv1,acc4b);
                acc4a=_mm512_fmadd_ps(_mm512_set1_ps(a4[k+2]),bv2,acc4a); acc4b=_mm512_fmadd_ps(_mm512_set1_ps(a4[k+3]),bv3,acc4b);
                acc5a=_mm512_fmadd_ps(_mm512_set1_ps(a5[k+0]),bv0,acc5a); acc5b=_mm512_fmadd_ps(_mm512_set1_ps(a5[k+1]),bv1,acc5b);
                acc5a=_mm512_fmadd_ps(_mm512_set1_ps(a5[k+2]),bv2,acc5a); acc5b=_mm512_fmadd_ps(_mm512_set1_ps(a5[k+3]),bv3,acc5b);
                acc6a=_mm512_fmadd_ps(_mm512_set1_ps(a6[k+0]),bv0,acc6a); acc6b=_mm512_fmadd_ps(_mm512_set1_ps(a6[k+1]),bv1,acc6b);
                acc6a=_mm512_fmadd_ps(_mm512_set1_ps(a6[k+2]),bv2,acc6a); acc6b=_mm512_fmadd_ps(_mm512_set1_ps(a6[k+3]),bv3,acc6b);
                acc7a=_mm512_fmadd_ps(_mm512_set1_ps(a7[k+0]),bv0,acc7a); acc7b=_mm512_fmadd_ps(_mm512_set1_ps(a7[k+1]),bv1,acc7b);
                acc7a=_mm512_fmadd_ps(_mm512_set1_ps(a7[k+2]),bv2,acc7a); acc7b=_mm512_fmadd_ps(_mm512_set1_ps(a7[k+3]),bv3,acc7b);
            }
            for (; k < K; ++k) {
                __m512 bv = _mm512_loadu_ps(bptr + k * 16);
                acc0a=_mm512_fmadd_ps(_mm512_set1_ps(a0[k]),bv,acc0a);
                acc1a=_mm512_fmadd_ps(_mm512_set1_ps(a1[k]),bv,acc1a);
                acc2a=_mm512_fmadd_ps(_mm512_set1_ps(a2[k]),bv,acc2a);
                acc3a=_mm512_fmadd_ps(_mm512_set1_ps(a3[k]),bv,acc3a);
                acc4a=_mm512_fmadd_ps(_mm512_set1_ps(a4[k]),bv,acc4a);
                acc5a=_mm512_fmadd_ps(_mm512_set1_ps(a5[k]),bv,acc5a);
                acc6a=_mm512_fmadd_ps(_mm512_set1_ps(a6[k]),bv,acc6a);
                acc7a=_mm512_fmadd_ps(_mm512_set1_ps(a7[k]),bv,acc7a);
            }
            {
                auto sv16 = [&](__m512 acca, __m512 accb, int row) __attribute__((always_inline)) {
                    __m512 acc = _mm512_add_ps(acca, accb);
                    if (n_valid == 16) {
                        acc = _mm512_add_ps(acc, bias_v);
                        if (relu) acc = _mm512_max_ps(acc, zero_v);
                        _mm512_storeu_ps(C + (size_t)(m + row) * N + n0, acc);
                    } else {
                        if (bias) acc = _mm512_add_ps(acc, _mm512_loadu_ps(bias + n0));
                        if (relu) acc = _mm512_max_ps(acc, zero_v);
                        _mm512_mask_storeu_ps(C + (size_t)(m + row) * N + n0, mask, acc);
                    }
                };
                sv16(acc0a,acc0b,0); sv16(acc1a,acc1b,1); sv16(acc2a,acc2b,2); sv16(acc3a,acc3b,3);
                sv16(acc4a,acc4b,4); sv16(acc5a,acc5b,5); sv16(acc6a,acc6b,6); sv16(acc7a,acc7b,7);
            }
        }   // end mi loop (single AVX-512)
    }   // end single co_blk loop (AVX-512)
    } // end MC block (AVX-512)

#elif defined(__AVX2__)
    // ── AVX2: M-cache-blocked, co_blk-inner ────────────────────────────────
    constexpr int MC = 32;
    for (int mi_blk = mi_s; mi_blk < mi_e; mi_blk += MC) {
    const int mi_blk_end = std::min(mi_blk + MC, mi_e);
    for (int co_blk = co_s; co_blk < dual_end; co_blk += 2) {
        const float* bpa = B_packed + (size_t)co_blk       * K * 8;
        const float* bpb = B_packed + (size_t)(co_blk + 1) * K * 8;
        const int n0a = co_blk * 8, n0b = (co_blk + 1) * 8;
        const __m256 bias_va = bias ? _mm256_loadu_ps(bias + n0a) : _mm256_setzero_ps();
        const __m256 bias_vb = bias ? _mm256_loadu_ps(bias + n0b) : _mm256_setzero_ps();
        const __m256 zero_v  = _mm256_setzero_ps();

        for (int mi = mi_blk; mi < mi_blk_end; ++mi) {
            const int m = mi * MR;
            const float* a0 = A + (size_t)(m+0) * K;
            const float* a1 = A + (size_t)(m+1) * K;
            const float* a2 = A + (size_t)(m+2) * K;
            const float* a3 = A + (size_t)(m+3) * K;
            const float* a4 = A + (size_t)(m+4) * K;
            const float* a5 = A + (size_t)(m+5) * K;
            const float* a6 = A + (size_t)(m+6) * K;
            const float* a7 = A + (size_t)(m+7) * K;

            __m256 acc0a=_mm256_setzero_ps(), acc0b=_mm256_setzero_ps();
            __m256 acc1a=_mm256_setzero_ps(), acc1b=_mm256_setzero_ps();
            __m256 acc2a=_mm256_setzero_ps(), acc2b=_mm256_setzero_ps();
            __m256 acc3a=_mm256_setzero_ps(), acc3b=_mm256_setzero_ps();
            __m256 acc4a=_mm256_setzero_ps(), acc4b=_mm256_setzero_ps();
            __m256 acc5a=_mm256_setzero_ps(), acc5b=_mm256_setzero_ps();
            __m256 acc6a=_mm256_setzero_ps(), acc6b=_mm256_setzero_ps();
            __m256 acc7a=_mm256_setzero_ps(), acc7b=_mm256_setzero_ps();

            int k = 0;
            for (; k + 3 < K; k += 4) {
                __builtin_prefetch(bpa + (k+8)*8,     0, 0);
                __builtin_prefetch(bpa + (k+8)*8 + 16, 0, 0);
                __builtin_prefetch(bpb + (k+8)*8,     0, 0);
                __builtin_prefetch(bpb + (k+8)*8 + 16, 0, 0);
                __m256 bva0=_mm256_loadu_ps(bpa+k*8+ 0), bva1=_mm256_loadu_ps(bpa+k*8+ 8);
                __m256 bva2=_mm256_loadu_ps(bpa+k*8+16), bva3=_mm256_loadu_ps(bpa+k*8+24);
                __m256 bvb0=_mm256_loadu_ps(bpb+k*8+ 0), bvb1=_mm256_loadu_ps(bpb+k*8+ 8);
                __m256 bvb2=_mm256_loadu_ps(bpb+k*8+16), bvb3=_mm256_loadu_ps(bpb+k*8+24);
#define FMA2_256(aa,ab,ar,b0,b1,b2,b3) do{ \
    __m256 _a0=_mm256_set1_ps((ar)[k+0]),_a1=_mm256_set1_ps((ar)[k+1]); \
    __m256 _a2=_mm256_set1_ps((ar)[k+2]),_a3=_mm256_set1_ps((ar)[k+3]); \
    aa=_mm256_fmadd_ps(_a0,b0,aa); ab=_mm256_fmadd_ps(_a0,bvb0,ab); \
    aa=_mm256_fmadd_ps(_a1,b1,aa); ab=_mm256_fmadd_ps(_a1,bvb1,ab); \
    aa=_mm256_fmadd_ps(_a2,b2,aa); ab=_mm256_fmadd_ps(_a2,bvb2,ab); \
    aa=_mm256_fmadd_ps(_a3,b3,aa); ab=_mm256_fmadd_ps(_a3,bvb3,ab); }while(0)
                FMA2_256(acc0a,acc0b,a0,bva0,bva1,bva2,bva3);
                FMA2_256(acc1a,acc1b,a1,bva0,bva1,bva2,bva3);
                FMA2_256(acc2a,acc2b,a2,bva0,bva1,bva2,bva3);
                FMA2_256(acc3a,acc3b,a3,bva0,bva1,bva2,bva3);
                FMA2_256(acc4a,acc4b,a4,bva0,bva1,bva2,bva3);
                FMA2_256(acc5a,acc5b,a5,bva0,bva1,bva2,bva3);
                FMA2_256(acc6a,acc6b,a6,bva0,bva1,bva2,bva3);
                FMA2_256(acc7a,acc7b,a7,bva0,bva1,bva2,bva3);
#undef FMA2_256
            }
            for (; k < K; ++k) {
                __m256 bva=_mm256_loadu_ps(bpa+k*8), bvb=_mm256_loadu_ps(bpb+k*8);
                __m256 _a0=_mm256_set1_ps(a0[k]), _a1=_mm256_set1_ps(a1[k]);
                __m256 _a2=_mm256_set1_ps(a2[k]), _a3=_mm256_set1_ps(a3[k]);
                __m256 _a4=_mm256_set1_ps(a4[k]), _a5=_mm256_set1_ps(a5[k]);
                __m256 _a6=_mm256_set1_ps(a6[k]), _a7=_mm256_set1_ps(a7[k]);
                acc0a=_mm256_fmadd_ps(_a0,bva,acc0a); acc0b=_mm256_fmadd_ps(_a0,bvb,acc0b);
                acc1a=_mm256_fmadd_ps(_a1,bva,acc1a); acc1b=_mm256_fmadd_ps(_a1,bvb,acc1b);
                acc2a=_mm256_fmadd_ps(_a2,bva,acc2a); acc2b=_mm256_fmadd_ps(_a2,bvb,acc2b);
                acc3a=_mm256_fmadd_ps(_a3,bva,acc3a); acc3b=_mm256_fmadd_ps(_a3,bvb,acc3b);
                acc4a=_mm256_fmadd_ps(_a4,bva,acc4a); acc4b=_mm256_fmadd_ps(_a4,bvb,acc4b);
                acc5a=_mm256_fmadd_ps(_a5,bva,acc5a); acc5b=_mm256_fmadd_ps(_a5,bvb,acc5b);
                acc6a=_mm256_fmadd_ps(_a6,bva,acc6a); acc6b=_mm256_fmadd_ps(_a6,bvb,acc6b);
                acc7a=_mm256_fmadd_ps(_a7,bva,acc7a); acc7b=_mm256_fmadd_ps(_a7,bvb,acc7b);
            }
            {
                auto sv2 = [&](__m256 acca, __m256 accb, int row) __attribute__((always_inline)) {
                    acca = _mm256_add_ps(acca, bias_va);
                    accb = _mm256_add_ps(accb, bias_vb);
                    if (relu) { acca = _mm256_max_ps(acca, zero_v); accb = _mm256_max_ps(accb, zero_v); }
                    _mm256_storeu_ps(C + (size_t)(m + row) * N + n0a, acca);
                    _mm256_storeu_ps(C + (size_t)(m + row) * N + n0b, accb);
                };
                sv2(acc0a,acc0b,0); sv2(acc1a,acc1b,1); sv2(acc2a,acc2b,2); sv2(acc3a,acc3b,3);
                sv2(acc4a,acc4b,4); sv2(acc5a,acc5b,5); sv2(acc6a,acc6b,6); sv2(acc7a,acc7b,7);
            }
        }   // end mi loop (dual AVX2)
    }   // end dual co_blk loop (AVX2)

    // ── AVX2 single co_blk (8 rows × 8 cols), co_blk-outer mi-inner ──────
    for (int co_blk = dual_end; co_blk < co_e; ++co_blk) {
        const float* bptr    = B_packed + (size_t)co_blk * K * 8;
        const int    n0      = co_blk * 8;
        const int    n_valid = std::min(8, N - n0);
        const __m256 zero_v  = _mm256_setzero_ps();

        for (int mi = mi_blk; mi < mi_blk_end; ++mi) {
            const int m = mi * MR;
            const float* a0 = A + (size_t)(m+0) * K;
            const float* a1 = A + (size_t)(m+1) * K;
            const float* a2 = A + (size_t)(m+2) * K;
            const float* a3 = A + (size_t)(m+3) * K;
            const float* a4 = A + (size_t)(m+4) * K;
            const float* a5 = A + (size_t)(m+5) * K;
            const float* a6 = A + (size_t)(m+6) * K;
            const float* a7 = A + (size_t)(m+7) * K;

            __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
            __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();

            int k = 0;
            for (; k + 3 < K; k += 4) {
                __builtin_prefetch(bptr + (k+8)*8,      0, 0);
                __builtin_prefetch(bptr + (k+8)*8 + 16, 0, 0);
                const float* bp = bptr + k * 8;
                __m256 bv0 = _mm256_loadu_ps(bp +  0);
                __m256 bv1 = _mm256_loadu_ps(bp +  8);
                __m256 bv2 = _mm256_loadu_ps(bp + 16);
                __m256 bv3 = _mm256_loadu_ps(bp + 24);
                acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[k+0]), bv0, acc0);
                acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[k+1]), bv1, acc0);
                acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[k+2]), bv2, acc0);
                acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[k+3]), bv3, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[k+0]), bv0, acc1);
                acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[k+1]), bv1, acc1);
                acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[k+2]), bv2, acc1);
                acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[k+3]), bv3, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[k+0]), bv0, acc2);
                acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[k+1]), bv1, acc2);
                acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[k+2]), bv2, acc2);
                acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[k+3]), bv3, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[k+0]), bv0, acc3);
                acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[k+1]), bv1, acc3);
                acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[k+2]), bv2, acc3);
                acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[k+3]), bv3, acc3);
                acc4 = _mm256_fmadd_ps(_mm256_set1_ps(a4[k+0]), bv0, acc4);
                acc4 = _mm256_fmadd_ps(_mm256_set1_ps(a4[k+1]), bv1, acc4);
                acc4 = _mm256_fmadd_ps(_mm256_set1_ps(a4[k+2]), bv2, acc4);
                acc4 = _mm256_fmadd_ps(_mm256_set1_ps(a4[k+3]), bv3, acc4);
                acc5 = _mm256_fmadd_ps(_mm256_set1_ps(a5[k+0]), bv0, acc5);
                acc5 = _mm256_fmadd_ps(_mm256_set1_ps(a5[k+1]), bv1, acc5);
                acc5 = _mm256_fmadd_ps(_mm256_set1_ps(a5[k+2]), bv2, acc5);
                acc5 = _mm256_fmadd_ps(_mm256_set1_ps(a5[k+3]), bv3, acc5);
                acc6 = _mm256_fmadd_ps(_mm256_set1_ps(a6[k+0]), bv0, acc6);
                acc6 = _mm256_fmadd_ps(_mm256_set1_ps(a6[k+1]), bv1, acc6);
                acc6 = _mm256_fmadd_ps(_mm256_set1_ps(a6[k+2]), bv2, acc6);
                acc6 = _mm256_fmadd_ps(_mm256_set1_ps(a6[k+3]), bv3, acc6);
                acc7 = _mm256_fmadd_ps(_mm256_set1_ps(a7[k+0]), bv0, acc7);
                acc7 = _mm256_fmadd_ps(_mm256_set1_ps(a7[k+1]), bv1, acc7);
                acc7 = _mm256_fmadd_ps(_mm256_set1_ps(a7[k+2]), bv2, acc7);
                acc7 = _mm256_fmadd_ps(_mm256_set1_ps(a7[k+3]), bv3, acc7);
            }
            for (; k < K; ++k) {
                __m256 bv = _mm256_loadu_ps(bptr + k * 8);
                acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[k]), bv, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[k]), bv, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[k]), bv, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[k]), bv, acc3);
                acc4 = _mm256_fmadd_ps(_mm256_set1_ps(a4[k]), bv, acc4);
                acc5 = _mm256_fmadd_ps(_mm256_set1_ps(a5[k]), bv, acc5);
                acc6 = _mm256_fmadd_ps(_mm256_set1_ps(a6[k]), bv, acc6);
                acc7 = _mm256_fmadd_ps(_mm256_set1_ps(a7[k]), bv, acc7);
            }
            if (n_valid == 8) {
                const __m256 bias_v = bias ? _mm256_loadu_ps(bias + n0) : _mm256_setzero_ps();
                auto sv8 = [&](__m256 acc, int row) __attribute__((always_inline)) {
                    acc = _mm256_add_ps(acc, bias_v);
                    if (relu) acc = _mm256_max_ps(acc, zero_v);
                    _mm256_storeu_ps(C + (size_t)(m + row) * N + n0, acc);
                };
                sv8(acc0, 0); sv8(acc1, 1); sv8(acc2, 2); sv8(acc3, 3);
                sv8(acc4, 4); sv8(acc5, 5); sv8(acc6, 6); sv8(acc7, 7);
            } else {
                alignas(32) float tmp[MR][8];
                _mm256_store_ps(tmp[0], acc0); _mm256_store_ps(tmp[1], acc1);
                _mm256_store_ps(tmp[2], acc2); _mm256_store_ps(tmp[3], acc3);
                _mm256_store_ps(tmp[4], acc4); _mm256_store_ps(tmp[5], acc5);
                _mm256_store_ps(tmp[6], acc6); _mm256_store_ps(tmp[7], acc7);
                for (int r = 0; r < MR; ++r) {
                    float* cptr = C + (size_t)(m + r) * N + n0;
                    for (int oc = 0; oc < n_valid; ++oc) {
                        float v = tmp[r][oc] + (bias ? bias[n0 + oc] : 0.f);
                        cptr[oc] = relu ? std::max(0.f, v) : v;
                    }
                }
            }
        }   // end mi loop (single AVX2)
    }   // end single co_blk loop (AVX2)
    } // end MC block (AVX2)

#else
    // ── Scalar fallback: M-cache-blocked, co_blk-inner ────────────────────
    constexpr int MC = 32;
    for (int mi_blk = mi_s; mi_blk < mi_e; mi_blk += MC) {
    const int mi_blk_end = std::min(mi_blk + MC, mi_e);
    for (int co_blk = co_s; co_blk < co_e; ++co_blk) {
        const float* bptr    = B_packed + (size_t)co_blk * K * TILE;
        const int    n0      = co_blk * TILE;
        const int    n_valid = std::min(TILE, N - n0);
        for (int mi = mi_blk; mi < mi_blk_end; ++mi) {
            const int m = mi * MR;
            const float* a0 = A + (size_t)(m+0) * K;
            const float* a1 = A + (size_t)(m+1) * K;
            const float* a2 = A + (size_t)(m+2) * K;
            const float* a3 = A + (size_t)(m+3) * K;
            const float* a4 = A + (size_t)(m+4) * K;
            const float* a5 = A + (size_t)(m+5) * K;
            const float* a6 = A + (size_t)(m+6) * K;
            const float* a7 = A + (size_t)(m+7) * K;
            float acc[MR][TILE] = {};
            for (int k = 0; k < K; ++k) {
                const float* bp = bptr + k * TILE;
                for (int oc = 0; oc < n_valid; ++oc) {
                    acc[0][oc] += a0[k] * bp[oc];
                    acc[1][oc] += a1[k] * bp[oc];
                    acc[2][oc] += a2[k] * bp[oc];
                    acc[3][oc] += a3[k] * bp[oc];
                    acc[4][oc] += a4[k] * bp[oc];
                    acc[5][oc] += a5[k] * bp[oc];
                    acc[6][oc] += a6[k] * bp[oc];
                    acc[7][oc] += a7[k] * bp[oc];
                }
            }
            for (int r = 0; r < MR; ++r) {
                float* cptr = C + (size_t)(m + r) * N + n0;
                for (int oc = 0; oc < n_valid; ++oc) {
                    float v = acc[r][oc] + (bias ? bias[n0 + oc] : 0.f);
                    cptr[oc] = relu ? std::max(0.f, v) : v;
                }
            }
        }
    }
    } // end MC block (scalar)
#endif  // AVX512/AVX2/scalar

#endif  // __ARM_NEON

    // ── Tail rows (M % MR residual, at most MR-1 rows) ────────────────────
    // M-partition: one share (thread 0's) handles the tail over all co_blks.
    // N-partition: every share handles the tail with its co_blk range.
    if (tail) {
        const int co_s2 = co_s;
        const int co_e2 = co_e;
        for (int m = m_tiles * MR; m < M; ++m) {
            const float* a_row = A + (size_t)m * K;
            for (int co_blk = co_s2; co_blk < co_e2; ++co_blk) {
                const float* bptr   = B_packed + (size_t)co_blk * K * TILE;
                const int    n0     = co_blk * TILE;
                const int    n_valid = std::min(TILE, N - n0);
#ifdef __ARM_NEON
                float32x4_t acc = vdupq_n_f32(0.f);
                int k = 0;
                for (; k + 3 < K; k += 4) {
                    const float* bp = bptr + k * 4;
                    acc = vfmaq_n_f32(acc, vld1q_f32(bp +  0), a_row[k+0]);
                    acc = vfmaq_n_f32(acc, vld1q_f32(bp +  4), a_row[k+1]);
                    acc = vfmaq_n_f32(acc, vld1q_f32(bp +  8), a_row[k+2]);
                    acc = vfmaq_n_f32(acc, vld1q_f32(bp + 12), a_row[k+3]);
                }
                for (; k < K; ++k)
                    acc = vfmaq_n_f32(acc, vld1q_f32(bptr + k * 4), a_row[k]);
                if (n_valid == TILE) {
                    if (bias) acc = vaddq_f32(acc, vld1q_f32(bias + n0));
                    if (relu) acc = vmaxq_f32(acc, vdupq_n_f32(0.f));
                    vst1q_f32(C + (size_t)m * N + n0, acc);
                } else {
                    alignas(16) float tmp[4];
                    vst1q_f32(tmp, acc);
                    for (int oc = 0; oc < n_valid; ++oc) {
                        float v = tmp[oc] + (bias ? bias[n0 + oc] : 0.f);
                        C[(size_t)m * N + n0 + oc] = relu ? std::max(0.f, v) : v;
                    }
                }
#elif defined(__AVX512F__)
                {
                    __m512 acc = _mm512_setzero_ps();
                    int k = 0;
                    for (; k + 3 < K; k += 4) {
                        const float* bp = bptr + k * 16;
                        acc = _mm512_fmadd_ps(_mm512_set1_ps(a_row[k+0]), _mm512_loadu_ps(bp +  0), acc);
                        acc = _mm512_fmadd_ps(_mm512_set1_ps(a_row[k+1]), _mm512_loadu_ps(bp + 16), acc);
                        acc = _mm512_fmadd_ps(_mm512_set1_ps(a_row[k+2]), _mm512_loadu_ps(bp + 32), acc);
                        acc = _mm512_fmadd_ps(_mm512_set1_ps(a_row[k+3]), _mm512_loadu_ps(bp + 48), acc);
                    }
                    for (; k < K; ++k)
                        acc = _mm512_fmadd_ps(_mm512_set1_ps(a_row[k]), _mm512_loadu_ps(bptr + k*16), acc);
                    __mmask16 mask = (n_valid == TILE) ? 0xFFFF : ((__mmask16)1 << n_valid) - 1;
                    if (bias) acc = _mm512_add_ps(acc, _mm512_loadu_ps(bias + n0));
                    if (relu) acc = _mm512_max_ps(acc, _mm512_setzero_ps());
                    _mm512_mask_storeu_ps(C + (size_t)m * N + n0, mask, acc);
                }
#elif defined(__AVX2__)
                {
                    __m256 acc = _mm256_setzero_ps();
                    int k = 0;
                    for (; k + 3 < K; k += 4) {
                        const float* bp = bptr + k * 8;
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[k+0]), _mm256_loadu_ps(bp +  0), acc);
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[k+1]), _mm256_loadu_ps(bp +  8), acc);
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[k+2]), _mm256_loadu_ps(bp + 16), acc);
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[k+3]), _mm256_loadu_ps(bp + 24), acc);
                    }
                    for (; k < K; ++k)
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(a_row[k]), _mm256_loadu_ps(bptr + k*8), acc);
                    if (n_valid == TILE) {
                        if (bias) acc = _mm256_add_ps(acc, _mm256_loadu_ps(bias + n0));
                        if (relu) acc = _mm256_max_ps(acc, _mm256_setzero_ps());
                        _mm256_storeu_ps(C + (size_t)m * N + n0, acc);
                    } else {
                        alignas(32) float tmp[8];
                        _mm256_store_ps(tmp, acc);
                        for (int oc = 0; oc < n_valid; ++oc) {
                            float v = tmp[oc] + (bias ? bias[n0 + oc] : 0.f);
                            C[(size_t)m * N + n0 + oc] = relu ? std::max(0.f, v) : v;
                        }
                    }
                }
#else
                float acc[TILE] = {};
                for (int k = 0; k < K; ++k)
                    for (int oc = 0; oc < n_valid; ++oc)
                        acc[oc] += a_row[k] * bptr[k * TILE + oc];
                for (int oc = 0; oc < n_valid; ++oc) {
                    float v = acc[oc] + (bias ? bias[n0 + oc] : 0.f);
                    C[(size_t)m * N + n0 + oc] = relu ? std::max(0.f, v) : v;
                }
#endif
            }
        }
    }
    });
}


// ──────────────────────────────────────────────────────────────
// gemm_fp32
//   input:    [C_in]
//   weight:   [C_out, C_in]
//   bias:     [C_out]
//   output:   [C_out]
//   w_packed: optional pre-packed weights [Co_t, K, TILE] where K = C_in
// ──────────────────────────────────────────────────────────────
void gemm_fp32_vec(
    const float* input,
    const float* weight,
    const float* bias,
    float*       output,
    int C_in, int C_out,
    const float* w_packed,
    StreamHandle /* stream */)
{
    if (w_packed) {
        sgemm_f32(input, w_packed, bias, output,
                  /*relu=*/false, /*M=*/1, /*K=*/C_in, /*N=*/C_out,
                  /*in_parallel=*/false, nullptr);
    } else {
        float* p = pack_weights_f32(weight, C_out, C_in);
        sgemm_f32(input, p, bias, output,
                  /*relu=*/false, /*M=*/1, /*K=*/C_in, /*N=*/C_out,
                  /*in_parallel=*/false, nullptr);
        free_packed_f32(p);
    }
}


// ──────────────────────────────────────────────────────────────
// conv1d_kfused_sgemm_f32
//   Fused K-split conv1d SGEMM: accumulates all kernel_size positions
//   in a single pass, keeping partial sums in SIMD registers.
//   Result: C is written ONCE per output element (not kernel_size times).
//   A_base:   padded input [T_padded, C_in]
//   B_merged: [Co_t, kernel_size, C_in, TILE]  (pack_merged_weights_f32)
//   C:        output [M, C_out]
// ──────────────────────────────────────────────────────────────
void conv1d_kfused_sgemm_f32(
    const float*         A_base,
    const float*         B_merged,
    const float*         bias,
    float*               C,
    bool                 relu,
    int M, int C_in, int C_out,
    int kernel_size, int dilation,
    bool in_parallel)
{
    if (!in_parallel) {
#ifdef _OPENMP
        if (!omp_in_parallel()) {
            const int nthreads = omp_get_max_threads();
            if (nthreads > 1 && (size_t)M * C_in * C_out * kernel_size >= 2000000UL) {
#pragma omp parallel num_threads(nthreads)
                conv1d_kfused_sgemm_f32(A_base, B_merged, bias, C, relu,
                                        M, C_in, C_out, kernel_size, dilation,
                                        /*in_parallel=*/true);
                return;
            }
        }
#endif
    }

#ifdef _OPENMP
    const int tid = in_parallel ? omp_get_thread_num() : 0;
    const int nT  = in_parallel ? omp_get_num_threads() : 1;
#else
    const int tid = 0, nT = 1;
#endif

    const int Co_t    = (C_out + TILE - 1) / TILE;
    const int m_tiles = M / MR;

    const bool n_part = (nT > 1) && (M < C_out);
    kt_split_tiles(n_part, m_tiles, Co_t, tid, nT, in_parallel,
                   [&](const int mi_s, const int mi_e, const int co_s, const int co_e, const bool tail) {

    {
#if defined(__AVX512F__)
    // ── AVX-512: MC-blocked, single N-tile (16 cols × 8 rows), kp inner ─────
    // MC blocking: for each A row block (MC_TILES*8 rows), sweep all co_blks.
    // A[MC×C_in] stays in L2 and is reused Co_t times; cuts L3 traffic for A.
    // Each _mm512_set1_ps(scalar) used once → vfmadd231ps {1to16} embedded broadcast.
    constexpr int MC_TILES = 4;  // 4 × 8 = 32 A-rows, ~64KB per kp in L2
    for (int mc_base = mi_s; mc_base < mi_e; mc_base += MC_TILES) {
    const int mc_end = std::min(mc_base + MC_TILES, mi_e);
    for (int co_blk = co_s; co_blk < co_e; ++co_blk) {
        const int n0      = co_blk * 16;
        const int n_valid = std::min(16, C_out - n0);
        const __mmask16 mask = (n_valid == 16) ? 0xFFFF : ((__mmask16)1 << n_valid) - 1;
        const __m512 bias_v  = (bias && n_valid == 16) ? _mm512_loadu_ps(bias + n0) : _mm512_setzero_ps();
        const __m512 zero_v  = _mm512_setzero_ps();
        for (int mi = mc_base; mi < mc_end; ++mi) {
            const int m = mi * MR;
            __m512 acc0=_mm512_setzero_ps(), acc1=_mm512_setzero_ps();
            __m512 acc2=_mm512_setzero_ps(), acc3=_mm512_setzero_ps();
            __m512 acc4=_mm512_setzero_ps(), acc5=_mm512_setzero_ps();
            __m512 acc6=_mm512_setzero_ps(), acc7=_mm512_setzero_ps();
            for (int kp = 0; kp < kernel_size; ++kp) {
                const size_t kp_off  = ((size_t)co_blk * kernel_size + kp) * C_in * 16;
                const int    row_off = kp * dilation;
                const float* bptr = B_merged + kp_off;
                const float* a0 = A_base + (size_t)(m+0 + row_off) * C_in;
                const float* a1 = A_base + (size_t)(m+1 + row_off) * C_in;
                const float* a2 = A_base + (size_t)(m+2 + row_off) * C_in;
                const float* a3 = A_base + (size_t)(m+3 + row_off) * C_in;
                const float* a4 = A_base + (size_t)(m+4 + row_off) * C_in;
                const float* a5 = A_base + (size_t)(m+5 + row_off) * C_in;
                const float* a6 = A_base + (size_t)(m+6 + row_off) * C_in;
                const float* a7 = A_base + (size_t)(m+7 + row_off) * C_in;
                int k = 0;
                for (; k + 3 < C_in; k += 4) {
                    __m512 bv0=_mm512_loadu_ps(bptr+k*16+ 0), bv1=_mm512_loadu_ps(bptr+k*16+16);
                    __m512 bv2=_mm512_loadu_ps(bptr+k*16+32), bv3=_mm512_loadu_ps(bptr+k*16+48);
                    acc0=_mm512_fmadd_ps(_mm512_set1_ps(a0[k+0]),bv0,acc0);
                    acc0=_mm512_fmadd_ps(_mm512_set1_ps(a0[k+1]),bv1,acc0);
                    acc0=_mm512_fmadd_ps(_mm512_set1_ps(a0[k+2]),bv2,acc0);
                    acc0=_mm512_fmadd_ps(_mm512_set1_ps(a0[k+3]),bv3,acc0);
                    acc1=_mm512_fmadd_ps(_mm512_set1_ps(a1[k+0]),bv0,acc1);
                    acc1=_mm512_fmadd_ps(_mm512_set1_ps(a1[k+1]),bv1,acc1);
                    acc1=_mm512_fmadd_ps(_mm512_set1_ps(a1[k+2]),bv2,acc1);
                    acc1=_mm512_fmadd_ps(_mm512_set1_ps(a1[k+3]),bv3,acc1);
                    acc2=_mm512_fmadd_ps(_mm512_set1_ps(a2[k+0]),bv0,acc2);
                    acc2=_mm512_fmadd_ps(_mm512_set1_ps(a2[k+1]),bv1,acc2);
                    acc2=_mm512_fmadd_ps(_mm512_set1_ps(a2[k+2]),bv2,acc2);
                    acc2=_mm512_fmadd_ps(_mm512_set1_ps(a2[k+3]),bv3,acc2);
                    acc3=_mm512_fmadd_ps(_mm512_set1_ps(a3[k+0]),bv0,acc3);
                    acc3=_mm512_fmadd_ps(_mm512_set1_ps(a3[k+1]),bv1,acc3);
                    acc3=_mm512_fmadd_ps(_mm512_set1_ps(a3[k+2]),bv2,acc3);
                    acc3=_mm512_fmadd_ps(_mm512_set1_ps(a3[k+3]),bv3,acc3);
                    acc4=_mm512_fmadd_ps(_mm512_set1_ps(a4[k+0]),bv0,acc4);
                    acc4=_mm512_fmadd_ps(_mm512_set1_ps(a4[k+1]),bv1,acc4);
                    acc4=_mm512_fmadd_ps(_mm512_set1_ps(a4[k+2]),bv2,acc4);
                    acc4=_mm512_fmadd_ps(_mm512_set1_ps(a4[k+3]),bv3,acc4);
                    acc5=_mm512_fmadd_ps(_mm512_set1_ps(a5[k+0]),bv0,acc5);
                    acc5=_mm512_fmadd_ps(_mm512_set1_ps(a5[k+1]),bv1,acc5);
                    acc5=_mm512_fmadd_ps(_mm512_set1_ps(a5[k+2]),bv2,acc5);
                    acc5=_mm512_fmadd_ps(_mm512_set1_ps(a5[k+3]),bv3,acc5);
                    acc6=_mm512_fmadd_ps(_mm512_set1_ps(a6[k+0]),bv0,acc6);
                    acc6=_mm512_fmadd_ps(_mm512_set1_ps(a6[k+1]),bv1,acc6);
                    acc6=_mm512_fmadd_ps(_mm512_set1_ps(a6[k+2]),bv2,acc6);
                    acc6=_mm512_fmadd_ps(_mm512_set1_ps(a6[k+3]),bv3,acc6);
                    acc7=_mm512_fmadd_ps(_mm512_set1_ps(a7[k+0]),bv0,acc7);
                    acc7=_mm512_fmadd_ps(_mm512_set1_ps(a7[k+1]),bv1,acc7);
                    acc7=_mm512_fmadd_ps(_mm512_set1_ps(a7[k+2]),bv2,acc7);
                    acc7=_mm512_fmadd_ps(_mm512_set1_ps(a7[k+3]),bv3,acc7);
                }
                for (; k < C_in; ++k) {
                    __m512 bv=_mm512_loadu_ps(bptr+k*16);
                    acc0=_mm512_fmadd_ps(_mm512_set1_ps(a0[k]),bv,acc0);
                    acc1=_mm512_fmadd_ps(_mm512_set1_ps(a1[k]),bv,acc1);
                    acc2=_mm512_fmadd_ps(_mm512_set1_ps(a2[k]),bv,acc2);
                    acc3=_mm512_fmadd_ps(_mm512_set1_ps(a3[k]),bv,acc3);
                    acc4=_mm512_fmadd_ps(_mm512_set1_ps(a4[k]),bv,acc4);
                    acc5=_mm512_fmadd_ps(_mm512_set1_ps(a5[k]),bv,acc5);
                    acc6=_mm512_fmadd_ps(_mm512_set1_ps(a6[k]),bv,acc6);
                    acc7=_mm512_fmadd_ps(_mm512_set1_ps(a7[k]),bv,acc7);
                }
            } // kp
            {
                auto sv = [&](__m512 acc, int row) __attribute__((always_inline)) {
                    if (n_valid == 16) {
                        acc = _mm512_add_ps(acc, bias_v);
                        if (relu) acc = _mm512_max_ps(acc, zero_v);
                        _mm512_storeu_ps(C + (size_t)(m + row) * C_out + n0, acc);
                    } else {
                        if (bias) acc = _mm512_add_ps(acc, _mm512_loadu_ps(bias + n0));
                        if (relu) acc = _mm512_max_ps(acc, zero_v);
                        _mm512_mask_storeu_ps(C + (size_t)(m + row) * C_out + n0, mask, acc);
                    }
                };
                sv(acc0,0); sv(acc1,1); sv(acc2,2); sv(acc3,3);
                sv(acc4,4); sv(acc5,5); sv(acc6,6); sv(acc7,7);
            }
        } // mi
    } // co_blk
    } // mc_base
#elif defined(__AVX2__)
    // ── AVX2 dual co_blk (8 rows × 16 cols), MC-blocked ─────────────────────
    // VEX encoding has no embedded-broadcast FMA → dual N-tile remains optimal
    // MC blocking: for each A row block sweep all co_blks → A stays in L2.
    {
    constexpr int MC_TILES = 4;
    int dual_end = co_s;
    while (dual_end + 1 < co_e && (dual_end + 2) * 8 <= C_out) dual_end += 2;
    for (int mc_base = mi_s; mc_base < mi_e; mc_base += MC_TILES) {
    const int mc_end = std::min(mc_base + MC_TILES, mi_e);
    for (int co_blk = co_s; co_blk < dual_end; co_blk += 2) {
        const int n0a = co_blk * 8, n0b = (co_blk + 1) * 8;
        const __m256 bias_va = bias ? _mm256_loadu_ps(bias + n0a) : _mm256_setzero_ps();
        const __m256 bias_vb = bias ? _mm256_loadu_ps(bias + n0b) : _mm256_setzero_ps();
        const __m256 zero_v  = _mm256_setzero_ps();
        for (int mi = mc_base; mi < mc_end; ++mi) {
            const int m = mi * MR;
            __m256 acc0a=_mm256_setzero_ps(), acc0b=_mm256_setzero_ps();
            __m256 acc1a=_mm256_setzero_ps(), acc1b=_mm256_setzero_ps();
            __m256 acc2a=_mm256_setzero_ps(), acc2b=_mm256_setzero_ps();
            __m256 acc3a=_mm256_setzero_ps(), acc3b=_mm256_setzero_ps();
            __m256 acc4a=_mm256_setzero_ps(), acc4b=_mm256_setzero_ps();
            __m256 acc5a=_mm256_setzero_ps(), acc5b=_mm256_setzero_ps();
            __m256 acc6a=_mm256_setzero_ps(), acc6b=_mm256_setzero_ps();
            __m256 acc7a=_mm256_setzero_ps(), acc7b=_mm256_setzero_ps();
            for (int kp = 0; kp < kernel_size; ++kp) {
                const size_t kp_off_a = ((size_t)co_blk       * kernel_size + kp) * C_in * 8;
                const size_t kp_off_b = ((size_t)(co_blk + 1) * kernel_size + kp) * C_in * 8;
                const int    row_off  = kp * dilation;
                const float* bpa = B_merged + kp_off_a;
                const float* bpb = B_merged + kp_off_b;
                const float* a0  = A_base + (size_t)(m+0 + row_off) * C_in;
                const float* a1  = A_base + (size_t)(m+1 + row_off) * C_in;
                const float* a2  = A_base + (size_t)(m+2 + row_off) * C_in;
                const float* a3  = A_base + (size_t)(m+3 + row_off) * C_in;
                const float* a4  = A_base + (size_t)(m+4 + row_off) * C_in;
                const float* a5  = A_base + (size_t)(m+5 + row_off) * C_in;
                const float* a6  = A_base + (size_t)(m+6 + row_off) * C_in;
                const float* a7  = A_base + (size_t)(m+7 + row_off) * C_in;
                int k = 0;
                for (; k + 3 < C_in; k += 4) {
                    __m256 bva0=_mm256_loadu_ps(bpa+k*8+ 0), bva1=_mm256_loadu_ps(bpa+k*8+ 8);
                    __m256 bva2=_mm256_loadu_ps(bpa+k*8+16), bva3=_mm256_loadu_ps(bpa+k*8+24);
                    __m256 bvb0=_mm256_loadu_ps(bpb+k*8+ 0), bvb1=_mm256_loadu_ps(bpb+k*8+ 8);
                    __m256 bvb2=_mm256_loadu_ps(bpb+k*8+16), bvb3=_mm256_loadu_ps(bpb+k*8+24);
#define KF8X2_256(acca,accb,ar) do{ \
    __m256 _a0=_mm256_set1_ps((ar)[k+0]),_a1=_mm256_set1_ps((ar)[k+1]); \
    __m256 _a2=_mm256_set1_ps((ar)[k+2]),_a3=_mm256_set1_ps((ar)[k+3]); \
    acca=_mm256_fmadd_ps(_a0,bva0,acca); accb=_mm256_fmadd_ps(_a0,bvb0,accb); \
    acca=_mm256_fmadd_ps(_a1,bva1,acca); accb=_mm256_fmadd_ps(_a1,bvb1,accb); \
    acca=_mm256_fmadd_ps(_a2,bva2,acca); accb=_mm256_fmadd_ps(_a2,bvb2,accb); \
    acca=_mm256_fmadd_ps(_a3,bva3,acca); accb=_mm256_fmadd_ps(_a3,bvb3,accb); }while(0)
                    KF8X2_256(acc0a,acc0b,a0); KF8X2_256(acc1a,acc1b,a1);
                    KF8X2_256(acc2a,acc2b,a2); KF8X2_256(acc3a,acc3b,a3);
                    KF8X2_256(acc4a,acc4b,a4); KF8X2_256(acc5a,acc5b,a5);
                    KF8X2_256(acc6a,acc6b,a6); KF8X2_256(acc7a,acc7b,a7);
#undef KF8X2_256
                }
                for (; k < C_in; ++k) {
                    __m256 bva=_mm256_loadu_ps(bpa+k*8), bvb=_mm256_loadu_ps(bpb+k*8);
                    __m256 _a0=_mm256_set1_ps(a0[k]), _a1=_mm256_set1_ps(a1[k]);
                    __m256 _a2=_mm256_set1_ps(a2[k]), _a3=_mm256_set1_ps(a3[k]);
                    __m256 _a4=_mm256_set1_ps(a4[k]), _a5=_mm256_set1_ps(a5[k]);
                    __m256 _a6=_mm256_set1_ps(a6[k]), _a7=_mm256_set1_ps(a7[k]);
                    acc0a=_mm256_fmadd_ps(_a0,bva,acc0a); acc0b=_mm256_fmadd_ps(_a0,bvb,acc0b);
                    acc1a=_mm256_fmadd_ps(_a1,bva,acc1a); acc1b=_mm256_fmadd_ps(_a1,bvb,acc1b);
                    acc2a=_mm256_fmadd_ps(_a2,bva,acc2a); acc2b=_mm256_fmadd_ps(_a2,bvb,acc2b);
                    acc3a=_mm256_fmadd_ps(_a3,bva,acc3a); acc3b=_mm256_fmadd_ps(_a3,bvb,acc3b);
                    acc4a=_mm256_fmadd_ps(_a4,bva,acc4a); acc4b=_mm256_fmadd_ps(_a4,bvb,acc4b);
                    acc5a=_mm256_fmadd_ps(_a5,bva,acc5a); acc5b=_mm256_fmadd_ps(_a5,bvb,acc5b);
                    acc6a=_mm256_fmadd_ps(_a6,bva,acc6a); acc6b=_mm256_fmadd_ps(_a6,bvb,acc6b);
                    acc7a=_mm256_fmadd_ps(_a7,bva,acc7a); acc7b=_mm256_fmadd_ps(_a7,bvb,acc7b);
                }
            } // kp
            {
                auto sv2 = [&](__m256 acca, __m256 accb, int row) __attribute__((always_inline)) {
                    acca = _mm256_add_ps(acca, bias_va);
                    accb = _mm256_add_ps(accb, bias_vb);
                    if (relu) { acca = _mm256_max_ps(acca, zero_v); accb = _mm256_max_ps(accb, zero_v); }
                    _mm256_storeu_ps(C + (size_t)(m + row) * C_out + n0a, acca);
                    _mm256_storeu_ps(C + (size_t)(m + row) * C_out + n0b, accb);
                };
                sv2(acc0a,acc0b,0); sv2(acc1a,acc1b,1); sv2(acc2a,acc2b,2);
                sv2(acc3a,acc3b,3); sv2(acc4a,acc4b,4); sv2(acc5a,acc5b,5);
                sv2(acc6a,acc6b,6); sv2(acc7a,acc7b,7);
            }
        } // mi
    } // co_blk dual
    // ── AVX2 single co_blk remainder ─────────────────────────────────────
    for (int co_blk = dual_end; co_blk < co_e; ++co_blk) {
        const int n0      = co_blk * 8;
        const int n_valid = std::min(8, C_out - n0);
        const __m256 zero_v = _mm256_setzero_ps();
        for (int mi = mc_base; mi < mc_end; ++mi) {
            const int m = mi * MR;
            __m256 acc0=_mm256_setzero_ps(), acc1=_mm256_setzero_ps();
            __m256 acc2=_mm256_setzero_ps(), acc3=_mm256_setzero_ps();
            __m256 acc4=_mm256_setzero_ps(), acc5=_mm256_setzero_ps();
            __m256 acc6=_mm256_setzero_ps(), acc7=_mm256_setzero_ps();
            for (int kp = 0; kp < kernel_size; ++kp) {
                const size_t kp_off = ((size_t)co_blk * kernel_size + kp) * C_in * 8;
                const int    row_off = kp * dilation;
                const float* bptr = B_merged + kp_off;
                const float* a0  = A_base + (size_t)(m+0 + row_off) * C_in;
                const float* a1  = A_base + (size_t)(m+1 + row_off) * C_in;
                const float* a2  = A_base + (size_t)(m+2 + row_off) * C_in;
                const float* a3  = A_base + (size_t)(m+3 + row_off) * C_in;
                const float* a4  = A_base + (size_t)(m+4 + row_off) * C_in;
                const float* a5  = A_base + (size_t)(m+5 + row_off) * C_in;
                const float* a6  = A_base + (size_t)(m+6 + row_off) * C_in;
                const float* a7  = A_base + (size_t)(m+7 + row_off) * C_in;
                int k = 0;
                for (; k + 3 < C_in; k += 4) {
                    const float* bp = bptr + k * 8;
                    __m256 bv0=_mm256_loadu_ps(bp+ 0), bv1=_mm256_loadu_ps(bp+ 8);
                    __m256 bv2=_mm256_loadu_ps(bp+16), bv3=_mm256_loadu_ps(bp+24);
                    acc0=_mm256_fmadd_ps(_mm256_set1_ps(a0[k+0]),bv0,acc0);
                    acc0=_mm256_fmadd_ps(_mm256_set1_ps(a0[k+1]),bv1,acc0);
                    acc0=_mm256_fmadd_ps(_mm256_set1_ps(a0[k+2]),bv2,acc0);
                    acc0=_mm256_fmadd_ps(_mm256_set1_ps(a0[k+3]),bv3,acc0);
                    acc1=_mm256_fmadd_ps(_mm256_set1_ps(a1[k+0]),bv0,acc1);
                    acc1=_mm256_fmadd_ps(_mm256_set1_ps(a1[k+1]),bv1,acc1);
                    acc1=_mm256_fmadd_ps(_mm256_set1_ps(a1[k+2]),bv2,acc1);
                    acc1=_mm256_fmadd_ps(_mm256_set1_ps(a1[k+3]),bv3,acc1);
                    acc2=_mm256_fmadd_ps(_mm256_set1_ps(a2[k+0]),bv0,acc2);
                    acc2=_mm256_fmadd_ps(_mm256_set1_ps(a2[k+1]),bv1,acc2);
                    acc2=_mm256_fmadd_ps(_mm256_set1_ps(a2[k+2]),bv2,acc2);
                    acc2=_mm256_fmadd_ps(_mm256_set1_ps(a2[k+3]),bv3,acc2);
                    acc3=_mm256_fmadd_ps(_mm256_set1_ps(a3[k+0]),bv0,acc3);
                    acc3=_mm256_fmadd_ps(_mm256_set1_ps(a3[k+1]),bv1,acc3);
                    acc3=_mm256_fmadd_ps(_mm256_set1_ps(a3[k+2]),bv2,acc3);
                    acc3=_mm256_fmadd_ps(_mm256_set1_ps(a3[k+3]),bv3,acc3);
                    acc4=_mm256_fmadd_ps(_mm256_set1_ps(a4[k+0]),bv0,acc4);
                    acc4=_mm256_fmadd_ps(_mm256_set1_ps(a4[k+1]),bv1,acc4);
                    acc4=_mm256_fmadd_ps(_mm256_set1_ps(a4[k+2]),bv2,acc4);
                    acc4=_mm256_fmadd_ps(_mm256_set1_ps(a4[k+3]),bv3,acc4);
                    acc5=_mm256_fmadd_ps(_mm256_set1_ps(a5[k+0]),bv0,acc5);
                    acc5=_mm256_fmadd_ps(_mm256_set1_ps(a5[k+1]),bv1,acc5);
                    acc5=_mm256_fmadd_ps(_mm256_set1_ps(a5[k+2]),bv2,acc5);
                    acc5=_mm256_fmadd_ps(_mm256_set1_ps(a5[k+3]),bv3,acc5);
                    acc6=_mm256_fmadd_ps(_mm256_set1_ps(a6[k+0]),bv0,acc6);
                    acc6=_mm256_fmadd_ps(_mm256_set1_ps(a6[k+1]),bv1,acc6);
                    acc6=_mm256_fmadd_ps(_mm256_set1_ps(a6[k+2]),bv2,acc6);
                    acc6=_mm256_fmadd_ps(_mm256_set1_ps(a6[k+3]),bv3,acc6);
                    acc7=_mm256_fmadd_ps(_mm256_set1_ps(a7[k+0]),bv0,acc7);
                    acc7=_mm256_fmadd_ps(_mm256_set1_ps(a7[k+1]),bv1,acc7);
                    acc7=_mm256_fmadd_ps(_mm256_set1_ps(a7[k+2]),bv2,acc7);
                    acc7=_mm256_fmadd_ps(_mm256_set1_ps(a7[k+3]),bv3,acc7);
                }
                for (; k < C_in; ++k) {
                    __m256 bv=_mm256_loadu_ps(bptr+k*8);
                    acc0=_mm256_fmadd_ps(_mm256_set1_ps(a0[k]),bv,acc0);
                    acc1=_mm256_fmadd_ps(_mm256_set1_ps(a1[k]),bv,acc1);
                    acc2=_mm256_fmadd_ps(_mm256_set1_ps(a2[k]),bv,acc2);
                    acc3=_mm256_fmadd_ps(_mm256_set1_ps(a3[k]),bv,acc3);
                    acc4=_mm256_fmadd_ps(_mm256_set1_ps(a4[k]),bv,acc4);
                    acc5=_mm256_fmadd_ps(_mm256_set1_ps(a5[k]),bv,acc5);
                    acc6=_mm256_fmadd_ps(_mm256_set1_ps(a6[k]),bv,acc6);
                    acc7=_mm256_fmadd_ps(_mm256_set1_ps(a7[k]),bv,acc7);
                }
            } // kp
            if (n_valid == 8) {
                auto sv8 = [&](__m256 acc, int row) __attribute__((always_inline)) {
                    if (bias) acc = _mm256_add_ps(acc, _mm256_loadu_ps(bias + n0));
                    if (relu) acc = _mm256_max_ps(acc, zero_v);
                    _mm256_storeu_ps(C + (size_t)(m + row) * C_out + n0, acc);
                };
                sv8(acc0,0); sv8(acc1,1); sv8(acc2,2);
                sv8(acc3,3); sv8(acc4,4); sv8(acc5,5);
                sv8(acc6,6); sv8(acc7,7);
            } else {
                alignas(32) float tmp8[MR][8];
                _mm256_store_ps(tmp8[0], acc0); _mm256_store_ps(tmp8[1], acc1);
                _mm256_store_ps(tmp8[2], acc2); _mm256_store_ps(tmp8[3], acc3);
                _mm256_store_ps(tmp8[4], acc4); _mm256_store_ps(tmp8[5], acc5);
                _mm256_store_ps(tmp8[6], acc6); _mm256_store_ps(tmp8[7], acc7);
                for (int r = 0; r < MR; ++r) {
                    float* cp = C + (size_t)(m + r) * C_out + n0;
                    for (int oc = 0; oc < n_valid; ++oc) {
                        float v = tmp8[r][oc] + (bias ? bias[n0 + oc] : 0.f);
                        cp[oc] = relu ? std::max(0.f, v) : v;
                    }
                }
            }
        } // mi
    } // co_blk single
    } // mc_base
    } // AVX2 dual_end block
#elif defined(__ARM_NEON)
    // ── NEON: MC-blocked, dual co_blk (8 rows × 8 cols), kp inner ────────────
    {
    constexpr int MC_TILES = 4;
    int dual_end = co_s;
    while (dual_end + 1 < co_e && (dual_end + 2) * 4 <= C_out) dual_end += 2;
    for (int mc_base = mi_s; mc_base < mi_e; mc_base += MC_TILES) {
    const int mc_end = std::min(mc_base + MC_TILES, mi_e);
    // ── Dual co_blk: 8 rows × 8 cols ─────────────────────────────────────
    for (int co_blk = co_s; co_blk < dual_end; co_blk += 2) {
        const int n0a = co_blk * 4, n0b = (co_blk + 1) * 4;
        const float32x4_t zero_v = vdupq_n_f32(0.f);
        for (int mi = mc_base; mi < mc_end; ++mi) {
            const int m = mi * MR;
            float32x4_t acc0a=vdupq_n_f32(0), acc0b=vdupq_n_f32(0);
            float32x4_t acc1a=vdupq_n_f32(0), acc1b=vdupq_n_f32(0);
            float32x4_t acc2a=vdupq_n_f32(0), acc2b=vdupq_n_f32(0);
            float32x4_t acc3a=vdupq_n_f32(0), acc3b=vdupq_n_f32(0);
            float32x4_t acc4a=vdupq_n_f32(0), acc4b=vdupq_n_f32(0);
            float32x4_t acc5a=vdupq_n_f32(0), acc5b=vdupq_n_f32(0);
            float32x4_t acc6a=vdupq_n_f32(0), acc6b=vdupq_n_f32(0);
            float32x4_t acc7a=vdupq_n_f32(0), acc7b=vdupq_n_f32(0);
            for (int kp = 0; kp < kernel_size; ++kp) {
                const float* bp0 = B_merged + ((size_t)co_blk       * kernel_size + kp) * C_in * 4;
                const float* bp1 = B_merged + ((size_t)(co_blk + 1) * kernel_size + kp) * C_in * 4;
                const int row_off = kp * dilation;
                const float* a0 = A_base + (size_t)(m+0 + row_off) * C_in;
                const float* a1 = A_base + (size_t)(m+1 + row_off) * C_in;
                const float* a2 = A_base + (size_t)(m+2 + row_off) * C_in;
                const float* a3 = A_base + (size_t)(m+3 + row_off) * C_in;
                const float* a4 = A_base + (size_t)(m+4 + row_off) * C_in;
                const float* a5 = A_base + (size_t)(m+5 + row_off) * C_in;
                const float* a6 = A_base + (size_t)(m+6 + row_off) * C_in;
                const float* a7 = A_base + (size_t)(m+7 + row_off) * C_in;
                int k = 0;
                for (; k + 3 < C_in; k += 4) {
                    float32x4_t ba0 = vld1q_f32(bp0 + k*4 +  0);
                    float32x4_t ba1 = vld1q_f32(bp0 + k*4 +  4);
                    float32x4_t ba2 = vld1q_f32(bp0 + k*4 +  8);
                    float32x4_t ba3 = vld1q_f32(bp0 + k*4 + 12);
                    float32x4_t bb0 = vld1q_f32(bp1 + k*4 +  0);
                    float32x4_t bb1 = vld1q_f32(bp1 + k*4 +  4);
                    float32x4_t bb2 = vld1q_f32(bp1 + k*4 +  8);
                    float32x4_t bb3 = vld1q_f32(bp1 + k*4 + 12);
                    float32x4_t av0 = vld1q_f32(a0 + k);
                    float32x4_t av1 = vld1q_f32(a1 + k);
                    float32x4_t av2 = vld1q_f32(a2 + k);
                    float32x4_t av3 = vld1q_f32(a3 + k);
                    float32x4_t av4 = vld1q_f32(a4 + k);
                    float32x4_t av5 = vld1q_f32(a5 + k);
                    float32x4_t av6 = vld1q_f32(a6 + k);
                    float32x4_t av7 = vld1q_f32(a7 + k);
                    acc0a = vfmaq_laneq_f32(acc0a, ba0, av0, 0); acc0b = vfmaq_laneq_f32(acc0b, bb0, av0, 0);
                    acc0a = vfmaq_laneq_f32(acc0a, ba1, av0, 1); acc0b = vfmaq_laneq_f32(acc0b, bb1, av0, 1);
                    acc0a = vfmaq_laneq_f32(acc0a, ba2, av0, 2); acc0b = vfmaq_laneq_f32(acc0b, bb2, av0, 2);
                    acc0a = vfmaq_laneq_f32(acc0a, ba3, av0, 3); acc0b = vfmaq_laneq_f32(acc0b, bb3, av0, 3);
                    acc1a = vfmaq_laneq_f32(acc1a, ba0, av1, 0); acc1b = vfmaq_laneq_f32(acc1b, bb0, av1, 0);
                    acc1a = vfmaq_laneq_f32(acc1a, ba1, av1, 1); acc1b = vfmaq_laneq_f32(acc1b, bb1, av1, 1);
                    acc1a = vfmaq_laneq_f32(acc1a, ba2, av1, 2); acc1b = vfmaq_laneq_f32(acc1b, bb2, av1, 2);
                    acc1a = vfmaq_laneq_f32(acc1a, ba3, av1, 3); acc1b = vfmaq_laneq_f32(acc1b, bb3, av1, 3);
                    acc2a = vfmaq_laneq_f32(acc2a, ba0, av2, 0); acc2b = vfmaq_laneq_f32(acc2b, bb0, av2, 0);
                    acc2a = vfmaq_laneq_f32(acc2a, ba1, av2, 1); acc2b = vfmaq_laneq_f32(acc2b, bb1, av2, 1);
                    acc2a = vfmaq_laneq_f32(acc2a, ba2, av2, 2); acc2b = vfmaq_laneq_f32(acc2b, bb2, av2, 2);
                    acc2a = vfmaq_laneq_f32(acc2a, ba3, av2, 3); acc2b = vfmaq_laneq_f32(acc2b, bb3, av2, 3);
                    acc3a = vfmaq_laneq_f32(acc3a, ba0, av3, 0); acc3b = vfmaq_laneq_f32(acc3b, bb0, av3, 0);
                    acc3a = vfmaq_laneq_f32(acc3a, ba1, av3, 1); acc3b = vfmaq_laneq_f32(acc3b, bb1, av3, 1);
                    acc3a = vfmaq_laneq_f32(acc3a, ba2, av3, 2); acc3b = vfmaq_laneq_f32(acc3b, bb2, av3, 2);
                    acc3a = vfmaq_laneq_f32(acc3a, ba3, av3, 3); acc3b = vfmaq_laneq_f32(acc3b, bb3, av3, 3);
                    acc4a = vfmaq_laneq_f32(acc4a, ba0, av4, 0); acc4b = vfmaq_laneq_f32(acc4b, bb0, av4, 0);
                    acc4a = vfmaq_laneq_f32(acc4a, ba1, av4, 1); acc4b = vfmaq_laneq_f32(acc4b, bb1, av4, 1);
                    acc4a = vfmaq_laneq_f32(acc4a, ba2, av4, 2); acc4b = vfmaq_laneq_f32(acc4b, bb2, av4, 2);
                    acc4a = vfmaq_laneq_f32(acc4a, ba3, av4, 3); acc4b = vfmaq_laneq_f32(acc4b, bb3, av4, 3);
                    acc5a = vfmaq_laneq_f32(acc5a, ba0, av5, 0); acc5b = vfmaq_laneq_f32(acc5b, bb0, av5, 0);
                    acc5a = vfmaq_laneq_f32(acc5a, ba1, av5, 1); acc5b = vfmaq_laneq_f32(acc5b, bb1, av5, 1);
                    acc5a = vfmaq_laneq_f32(acc5a, ba2, av5, 2); acc5b = vfmaq_laneq_f32(acc5b, bb2, av5, 2);
                    acc5a = vfmaq_laneq_f32(acc5a, ba3, av5, 3); acc5b = vfmaq_laneq_f32(acc5b, bb3, av5, 3);
                    acc6a = vfmaq_laneq_f32(acc6a, ba0, av6, 0); acc6b = vfmaq_laneq_f32(acc6b, bb0, av6, 0);
                    acc6a = vfmaq_laneq_f32(acc6a, ba1, av6, 1); acc6b = vfmaq_laneq_f32(acc6b, bb1, av6, 1);
                    acc6a = vfmaq_laneq_f32(acc6a, ba2, av6, 2); acc6b = vfmaq_laneq_f32(acc6b, bb2, av6, 2);
                    acc6a = vfmaq_laneq_f32(acc6a, ba3, av6, 3); acc6b = vfmaq_laneq_f32(acc6b, bb3, av6, 3);
                    acc7a = vfmaq_laneq_f32(acc7a, ba0, av7, 0); acc7b = vfmaq_laneq_f32(acc7b, bb0, av7, 0);
                    acc7a = vfmaq_laneq_f32(acc7a, ba1, av7, 1); acc7b = vfmaq_laneq_f32(acc7b, bb1, av7, 1);
                    acc7a = vfmaq_laneq_f32(acc7a, ba2, av7, 2); acc7b = vfmaq_laneq_f32(acc7b, bb2, av7, 2);
                    acc7a = vfmaq_laneq_f32(acc7a, ba3, av7, 3); acc7b = vfmaq_laneq_f32(acc7b, bb3, av7, 3);
                }
                for (; k < C_in; ++k) {
                    float32x4_t bav = vld1q_f32(bp0 + k * 4);
                    float32x4_t bbv = vld1q_f32(bp1 + k * 4);
                    acc0a = vfmaq_n_f32(acc0a, bav, a0[k]); acc0b = vfmaq_n_f32(acc0b, bbv, a0[k]);
                    acc1a = vfmaq_n_f32(acc1a, bav, a1[k]); acc1b = vfmaq_n_f32(acc1b, bbv, a1[k]);
                    acc2a = vfmaq_n_f32(acc2a, bav, a2[k]); acc2b = vfmaq_n_f32(acc2b, bbv, a2[k]);
                    acc3a = vfmaq_n_f32(acc3a, bav, a3[k]); acc3b = vfmaq_n_f32(acc3b, bbv, a3[k]);
                    acc4a = vfmaq_n_f32(acc4a, bav, a4[k]); acc4b = vfmaq_n_f32(acc4b, bbv, a4[k]);
                    acc5a = vfmaq_n_f32(acc5a, bav, a5[k]); acc5b = vfmaq_n_f32(acc5b, bbv, a5[k]);
                    acc6a = vfmaq_n_f32(acc6a, bav, a6[k]); acc6b = vfmaq_n_f32(acc6b, bbv, a6[k]);
                    acc7a = vfmaq_n_f32(acc7a, bav, a7[k]); acc7b = vfmaq_n_f32(acc7b, bbv, a7[k]);
                }
            } // kp
            float32x4_t bva = bias ? vld1q_f32(bias + n0a) : vdupq_n_f32(0.f);
            float32x4_t bvb = bias ? vld1q_f32(bias + n0b) : vdupq_n_f32(0.f);
            auto sv2 = [&](float32x4_t acca, float32x4_t accb, int row)
                __attribute__((always_inline)) {
                acca = vaddq_f32(acca, bva); accb = vaddq_f32(accb, bvb);
                if (relu) { acca = vmaxq_f32(acca, zero_v); accb = vmaxq_f32(accb, zero_v); }
                vst1q_f32(C + (size_t)(m + row) * C_out + n0a, acca);
                vst1q_f32(C + (size_t)(m + row) * C_out + n0b, accb);
            };
            sv2(acc0a,acc0b,0); sv2(acc1a,acc1b,1); sv2(acc2a,acc2b,2); sv2(acc3a,acc3b,3);
            sv2(acc4a,acc4b,4); sv2(acc5a,acc5b,5); sv2(acc6a,acc6b,6); sv2(acc7a,acc7b,7);
        } // mi
    } // co_blk dual
    // ── Single co_blk remainder: 8 rows × 4 cols ──────────────────────────
    for (int co_blk = dual_end; co_blk < co_e; ++co_blk) {
        const int n0      = co_blk * 4;
        const int n_valid = std::min(4, C_out - n0);
        const float32x4_t zero_v = vdupq_n_f32(0.f);
        for (int mi = mc_base; mi < mc_end; ++mi) {
            const int m = mi * MR;
            float32x4_t acc0=vdupq_n_f32(0), acc1=vdupq_n_f32(0);
            float32x4_t acc2=vdupq_n_f32(0), acc3=vdupq_n_f32(0);
            float32x4_t acc4=vdupq_n_f32(0), acc5=vdupq_n_f32(0);
            float32x4_t acc6=vdupq_n_f32(0), acc7=vdupq_n_f32(0);
            for (int kp = 0; kp < kernel_size; ++kp) {
                const float* bptr = B_merged + ((size_t)co_blk * kernel_size + kp) * C_in * 4;
                const int row_off = kp * dilation;
                const float* a0 = A_base + (size_t)(m+0 + row_off) * C_in;
                const float* a1 = A_base + (size_t)(m+1 + row_off) * C_in;
                const float* a2 = A_base + (size_t)(m+2 + row_off) * C_in;
                const float* a3 = A_base + (size_t)(m+3 + row_off) * C_in;
                const float* a4 = A_base + (size_t)(m+4 + row_off) * C_in;
                const float* a5 = A_base + (size_t)(m+5 + row_off) * C_in;
                const float* a6 = A_base + (size_t)(m+6 + row_off) * C_in;
                const float* a7 = A_base + (size_t)(m+7 + row_off) * C_in;
                int k = 0;
                for (; k + 3 < C_in; k += 4) {
                    const float* bp = bptr + k * 4;
                    float32x4_t bv0 = vld1q_f32(bp +  0);
                    float32x4_t bv1 = vld1q_f32(bp +  4);
                    float32x4_t bv2 = vld1q_f32(bp +  8);
                    float32x4_t bv3 = vld1q_f32(bp + 12);
                    float32x4_t av0 = vld1q_f32(a0 + k);
                    float32x4_t av1 = vld1q_f32(a1 + k);
                    float32x4_t av2 = vld1q_f32(a2 + k);
                    float32x4_t av3 = vld1q_f32(a3 + k);
                    float32x4_t av4 = vld1q_f32(a4 + k);
                    float32x4_t av5 = vld1q_f32(a5 + k);
                    float32x4_t av6 = vld1q_f32(a6 + k);
                    float32x4_t av7 = vld1q_f32(a7 + k);
                    acc0 = vfmaq_laneq_f32(acc0, bv0, av0, 0); acc0 = vfmaq_laneq_f32(acc0, bv1, av0, 1);
                    acc0 = vfmaq_laneq_f32(acc0, bv2, av0, 2); acc0 = vfmaq_laneq_f32(acc0, bv3, av0, 3);
                    acc1 = vfmaq_laneq_f32(acc1, bv0, av1, 0); acc1 = vfmaq_laneq_f32(acc1, bv1, av1, 1);
                    acc1 = vfmaq_laneq_f32(acc1, bv2, av1, 2); acc1 = vfmaq_laneq_f32(acc1, bv3, av1, 3);
                    acc2 = vfmaq_laneq_f32(acc2, bv0, av2, 0); acc2 = vfmaq_laneq_f32(acc2, bv1, av2, 1);
                    acc2 = vfmaq_laneq_f32(acc2, bv2, av2, 2); acc2 = vfmaq_laneq_f32(acc2, bv3, av2, 3);
                    acc3 = vfmaq_laneq_f32(acc3, bv0, av3, 0); acc3 = vfmaq_laneq_f32(acc3, bv1, av3, 1);
                    acc3 = vfmaq_laneq_f32(acc3, bv2, av3, 2); acc3 = vfmaq_laneq_f32(acc3, bv3, av3, 3);
                    acc4 = vfmaq_laneq_f32(acc4, bv0, av4, 0); acc4 = vfmaq_laneq_f32(acc4, bv1, av4, 1);
                    acc4 = vfmaq_laneq_f32(acc4, bv2, av4, 2); acc4 = vfmaq_laneq_f32(acc4, bv3, av4, 3);
                    acc5 = vfmaq_laneq_f32(acc5, bv0, av5, 0); acc5 = vfmaq_laneq_f32(acc5, bv1, av5, 1);
                    acc5 = vfmaq_laneq_f32(acc5, bv2, av5, 2); acc5 = vfmaq_laneq_f32(acc5, bv3, av5, 3);
                    acc6 = vfmaq_laneq_f32(acc6, bv0, av6, 0); acc6 = vfmaq_laneq_f32(acc6, bv1, av6, 1);
                    acc6 = vfmaq_laneq_f32(acc6, bv2, av6, 2); acc6 = vfmaq_laneq_f32(acc6, bv3, av6, 3);
                    acc7 = vfmaq_laneq_f32(acc7, bv0, av7, 0); acc7 = vfmaq_laneq_f32(acc7, bv1, av7, 1);
                    acc7 = vfmaq_laneq_f32(acc7, bv2, av7, 2); acc7 = vfmaq_laneq_f32(acc7, bv3, av7, 3);
                }
                for (; k < C_in; ++k) {
                    float32x4_t bv = vld1q_f32(bptr + k * 4);
                    acc0 = vfmaq_n_f32(acc0, bv, a0[k]); acc1 = vfmaq_n_f32(acc1, bv, a1[k]);
                    acc2 = vfmaq_n_f32(acc2, bv, a2[k]); acc3 = vfmaq_n_f32(acc3, bv, a3[k]);
                    acc4 = vfmaq_n_f32(acc4, bv, a4[k]); acc5 = vfmaq_n_f32(acc5, bv, a5[k]);
                    acc6 = vfmaq_n_f32(acc6, bv, a6[k]); acc7 = vfmaq_n_f32(acc7, bv, a7[k]);
                }
            } // kp
            if (n_valid == 4) {
                float32x4_t bias_v = bias ? vld1q_f32(bias + n0) : vdupq_n_f32(0.f);
                auto sv = [&](float32x4_t acc, int row) __attribute__((always_inline)) {
                    acc = vaddq_f32(acc, bias_v);
                    if (relu) acc = vmaxq_f32(acc, zero_v);
                    vst1q_f32(C + (size_t)(m + row) * C_out + n0, acc);
                };
                sv(acc0,0); sv(acc1,1); sv(acc2,2); sv(acc3,3);
                sv(acc4,4); sv(acc5,5); sv(acc6,6); sv(acc7,7);
            } else {
                alignas(16) float tmp[MR][4];
                vst1q_f32(tmp[0],acc0); vst1q_f32(tmp[1],acc1);
                vst1q_f32(tmp[2],acc2); vst1q_f32(tmp[3],acc3);
                vst1q_f32(tmp[4],acc4); vst1q_f32(tmp[5],acc5);
                vst1q_f32(tmp[6],acc6); vst1q_f32(tmp[7],acc7);
                for (int r = 0; r < MR; ++r) {
                    float* cp = C + (size_t)(m + r) * C_out + n0;
                    for (int oc = 0; oc < n_valid; ++oc) {
                        float v = tmp[r][oc] + (bias ? bias[n0 + oc] : 0.f);
                        cp[oc] = relu ? std::max(0.f, v) : v;
                    }
                }
            }
        } // mi
    } // co_blk single
    } // mc_base
    } // NEON block
#else
    // ── Scalar fallback ──────────────────────────────────────────────────
    for (int mi = mi_s; mi < mi_e; ++mi) {
        const int m = mi * MR;
        for (int co_blk = co_s; co_blk < co_e; ++co_blk) {
        const int n0      = co_blk * TILE;
        const int n_valid = std::min(TILE, C_out - n0);
        float acc[MR][TILE] = {};
        for (int kp = 0; kp < kernel_size; ++kp) {
            const float* bptr = B_merged + ((size_t)co_blk * kernel_size + kp) * C_in * TILE;
            for (int r = 0; r < MR; ++r) {
                const float* ar = A_base + (size_t)(m + r + (size_t)kp*dilation) * C_in;
                for (int k = 0; k < C_in; ++k)
                    for (int oc = 0; oc < n_valid; ++oc)
                        acc[r][oc] += ar[k] * bptr[k * TILE + oc];
            }
        }
        for (int r = 0; r < MR; ++r) {
            float* cp = C + (size_t)(m + r) * C_out + n0;
            for (int oc = 0; oc < n_valid; ++oc) {
                float v = acc[r][oc] + (bias ? bias[n0 + oc] : 0.f);
                cp[oc] = relu ? std::max(0.f, v) : v;
            }
        }
        } // co_blk
    } // mi
#endif  // AVX512/AVX2/scalar
    } // end block

    // ── Tail rows (M % MR residual) ────────────────────────────────────────
    if (tail) {
        const int co_s2 = co_s;
        const int co_e2 = co_e;
        for (int m = m_tiles * MR; m < M; ++m) {
            for (int co_blk = co_s2; co_blk < co_e2; ++co_blk) {
                const int n0      = co_blk * TILE;
                const int n_valid = std::min(TILE, C_out - n0);
#if defined(__AVX512F__)
                __m512 acc = _mm512_setzero_ps();
                for (int kp = 0; kp < kernel_size; ++kp) {
                    const float* bptr = B_merged + ((size_t)co_blk * kernel_size + kp) * C_in * 16;
                    const float* ar   = A_base + (size_t)(m + kp*dilation) * C_in;
                    int k = 0;
                    for (; k + 3 < C_in; k += 4) {
                        const float* bp = bptr + k * 16;
                        acc=_mm512_fmadd_ps(_mm512_set1_ps(ar[k+0]),_mm512_loadu_ps(bp+ 0),acc);
                        acc=_mm512_fmadd_ps(_mm512_set1_ps(ar[k+1]),_mm512_loadu_ps(bp+16),acc);
                        acc=_mm512_fmadd_ps(_mm512_set1_ps(ar[k+2]),_mm512_loadu_ps(bp+32),acc);
                        acc=_mm512_fmadd_ps(_mm512_set1_ps(ar[k+3]),_mm512_loadu_ps(bp+48),acc);
                    }
                    for (; k < C_in; ++k)
                        acc=_mm512_fmadd_ps(_mm512_set1_ps(ar[k]),_mm512_loadu_ps(bptr+k*16),acc);
                }
                {
                    __mmask16 mask = (n_valid == 16) ? 0xFFFF : ((__mmask16)1 << n_valid) - 1;
                    if (bias) acc = _mm512_add_ps(acc, _mm512_loadu_ps(bias + n0));
                    if (relu) acc = _mm512_max_ps(acc, _mm512_setzero_ps());
                    _mm512_mask_storeu_ps(C + (size_t)m * C_out + n0, mask, acc);
                }
#elif defined(__AVX2__)
                __m256 acc = _mm256_setzero_ps();
                for (int kp = 0; kp < kernel_size; ++kp) {
                    const float* bptr = B_merged + ((size_t)co_blk * kernel_size + kp) * C_in * 8;
                    const float* ar   = A_base + (size_t)(m + kp*dilation) * C_in;
                    int k = 0;
                    for (; k + 3 < C_in; k += 4) {
                        const float* bp = bptr + k * 8;
                        acc=_mm256_fmadd_ps(_mm256_set1_ps(ar[k+0]),_mm256_loadu_ps(bp+ 0),acc);
                        acc=_mm256_fmadd_ps(_mm256_set1_ps(ar[k+1]),_mm256_loadu_ps(bp+ 8),acc);
                        acc=_mm256_fmadd_ps(_mm256_set1_ps(ar[k+2]),_mm256_loadu_ps(bp+16),acc);
                        acc=_mm256_fmadd_ps(_mm256_set1_ps(ar[k+3]),_mm256_loadu_ps(bp+24),acc);
                    }
                    for (; k < C_in; ++k)
                        acc=_mm256_fmadd_ps(_mm256_set1_ps(ar[k]),_mm256_loadu_ps(bptr+k*8),acc);
                }
                if (n_valid == 8) {
                    if (bias) acc = _mm256_add_ps(acc, _mm256_loadu_ps(bias + n0));
                    if (relu) acc = _mm256_max_ps(acc, _mm256_setzero_ps());
                    _mm256_storeu_ps(C + (size_t)m * C_out + n0, acc);
                } else {
                    alignas(32) float t8[8]; _mm256_store_ps(t8, acc);
                    float* cp = C + (size_t)m * C_out + n0;
                    for (int oc = 0; oc < n_valid; ++oc) {
                        float v = t8[oc] + (bias ? bias[n0 + oc] : 0.f);
                        cp[oc] = relu ? std::max(0.f, v) : v;
                    }
                }
#elif defined(__ARM_NEON)
                {
                float32x4_t acc = vdupq_n_f32(0.f);
                for (int kp = 0; kp < kernel_size; ++kp) {
                    const float* bptr = B_merged + ((size_t)co_blk * kernel_size + kp) * C_in * 4;
                    const float* ar   = A_base + (size_t)(m + kp*dilation) * C_in;
                    int k = 0;
                    for (; k + 3 < C_in; k += 4) {
                        const float* bp = bptr + k * 4;
                        acc = vfmaq_n_f32(acc, vld1q_f32(bp +  0), ar[k+0]);
                        acc = vfmaq_n_f32(acc, vld1q_f32(bp +  4), ar[k+1]);
                        acc = vfmaq_n_f32(acc, vld1q_f32(bp +  8), ar[k+2]);
                        acc = vfmaq_n_f32(acc, vld1q_f32(bp + 12), ar[k+3]);
                    }
                    for (; k < C_in; ++k)
                        acc = vfmaq_n_f32(acc, vld1q_f32(bptr + k * 4), ar[k]);
                }
                if (n_valid == 4) {
                    if (bias) acc = vaddq_f32(acc, vld1q_f32(bias + n0));
                    if (relu) acc = vmaxq_f32(acc, vdupq_n_f32(0.f));
                    vst1q_f32(C + (size_t)m * C_out + n0, acc);
                } else {
                    alignas(16) float t4[4]; vst1q_f32(t4, acc);
                    float* cp = C + (size_t)m * C_out + n0;
                    for (int oc = 0; oc < n_valid; ++oc) {
                        float v = t4[oc] + (bias ? bias[n0 + oc] : 0.f);
                        cp[oc] = relu ? std::max(0.f, v) : v;
                    }
                }
                }
#else
                float acc[TILE] = {};
                for (int kp = 0; kp < kernel_size; ++kp) {
                    const float* bptr = B_merged + ((size_t)co_blk * kernel_size + kp) * C_in * TILE;
                    const float* ar   = A_base + (size_t)(m + kp*dilation) * C_in;
                    for (int k = 0; k < C_in; ++k)
                        for (int oc = 0; oc < n_valid; ++oc)
                            acc[oc] += ar[k] * bptr[k * TILE + oc];
                }
                float* cp = C + (size_t)m * C_out + n0;
                for (int oc = 0; oc < n_valid; ++oc) {
                    float v = acc[oc] + (bias ? bias[n0 + oc] : 0.f);
                    cp[oc] = relu ? std::max(0.f, v) : v;
                }
#endif
            }
        }
    }
    });
}
