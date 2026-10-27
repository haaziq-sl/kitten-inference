#include "../ops_neon.hpp"
#include "profile_internal.hpp"
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

void add_requant_int8(
    const int8_t* in1,
    const int8_t* in2,
    float in1_scale, int in1_zp,
    float in2_scale, int in2_zp,
    float out_scale, int out_zp,
    int8_t* output,
    int N,
    StreamHandle /* stream */)
{
    double _t0_add = now_ms();
    float m1 = in1_scale / out_scale;
    float m2 = in2_scale / out_scale;

    // Pre-compute Q31 params for the general fallback path (used when Q12 doesn't apply,
    // and for all scalar tails). One-time cost before any loops.
    int32_t mm1, me1, mm2, me2;
    QuantizeMultiplier(m1, &mm1, &me1);
    QuantizeMultiplier(m2, &mm2, &me2);

#ifdef __ARM_NEON
    // Q12 fixed-point fast path: safe when |m1|, |m2| ≤ 7.99 (always true for ResNet-101).
    // Uses int16 multipliers and pre-fused zero-point bias — fewer NEON instructions.
    if (std::abs(m1) <= 7.99f && std::abs(m2) <= 7.99f) {
        const int16_t M1 = (int16_t)std::roundf(m1 * 4096.0f);
        const int16_t M2 = (int16_t)std::roundf(m2 * 4096.0f);
        const int32_t pre_bias = (int32_t)out_zp * 4096
                               - (int32_t)in1_zp * M1
                               - (int32_t)in2_zp * M2;
        const int32x4_t vbias = vdupq_n_s32(pre_bias);

        const int nblk = N / 16;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(omp_get_max_threads() > 1)
#endif
        for (int bi = 0; bi < nblk; ++bi) {
            const int i = bi * 16;
            int8x16_t a = vld1q_s8(in1 + i);
            int8x16_t b = vld1q_s8(in2 + i);
            int16x8_t a_lo = vmovl_s8(vget_low_s8(a));
            int16x8_t a_hi = vmovl_s8(vget_high_s8(a));
            int16x8_t b_lo = vmovl_s8(vget_low_s8(b));
            int16x8_t b_hi = vmovl_s8(vget_high_s8(b));
            int32x4_t acc0 = vaddq_s32(vmlal_n_s16(vmull_n_s16(vget_low_s16(a_lo),  M1), vget_low_s16(b_lo),  M2), vbias);
            int32x4_t acc1 = vaddq_s32(vmlal_n_s16(vmull_n_s16(vget_high_s16(a_lo), M1), vget_high_s16(b_lo), M2), vbias);
            int32x4_t acc2 = vaddq_s32(vmlal_n_s16(vmull_n_s16(vget_low_s16(a_hi),  M1), vget_low_s16(b_hi),  M2), vbias);
            int32x4_t acc3 = vaddq_s32(vmlal_n_s16(vmull_n_s16(vget_high_s16(a_hi), M1), vget_high_s16(b_hi), M2), vbias);
            int8x8_t o0 = vqmovn_s16(vcombine_s16(vrshrn_n_s32(acc0, 12), vrshrn_n_s32(acc1, 12)));
            int8x8_t o1 = vqmovn_s16(vcombine_s16(vrshrn_n_s32(acc2, 12), vrshrn_n_s32(acc3, 12)));
            vst1q_s8(output + i, vcombine_s8(o0, o1));
        }
        // Integer scalar tail — no float
        for (int i = nblk * 16; i < N; ++i) {
            int32_t v = apply_q31_scalar((int32_t)in1[i] - in1_zp, mm1, me1)
                      + apply_q31_scalar((int32_t)in2[i] - in2_zp, mm2, me2)
                      + out_zp;
            output[i] = static_cast<int8_t>(std::clamp(v, -128, 127));
        }
        g_add_req_ms += now_ms() - _t0_add;
        return;
    }
    // Q31 NEON fallback (|m| > 7.99 — should never occur for ResNet-101)
    {
        const int32x4_t vm1  = vdupq_n_s32(mm1), ve1 = vdupq_n_s32(me1);
        const int32x4_t vm2  = vdupq_n_s32(mm2), ve2 = vdupq_n_s32(me2);
        const int16x8_t vzp1 = vdupq_n_s16((int16_t)in1_zp);
        const int16x8_t vzp2 = vdupq_n_s16((int16_t)in2_zp);
        const int16x8_t vout = vdupq_n_s16((int16_t)out_zp);
        const int nblk2 = N / 8;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(omp_get_max_threads() > 1)
#endif
        for (int bi = 0; bi < nblk2; ++bi) {
            const int i = bi * 8;
            int16x8_t a = vsubq_s16(vmovl_s8(vld1_s8(in1 + i)), vzp1);
            int16x8_t b = vsubq_s16(vmovl_s8(vld1_s8(in2 + i)), vzp2);
            int32x4_t a_lo = vmovl_s16(vget_low_s16(a));
            int32x4_t a_hi = vmovl_s16(vget_high_s16(a));
            int32x4_t b_lo = vmovl_s16(vget_low_s16(b));
            int32x4_t b_hi = vmovl_s16(vget_high_s16(b));
            int32x4_t r1_lo = vrshlq_s32(vqrdmulhq_s32(a_lo, vm1), ve1);
            int32x4_t r1_hi = vrshlq_s32(vqrdmulhq_s32(a_hi, vm1), ve1);
            int32x4_t r2_lo = vrshlq_s32(vqrdmulhq_s32(b_lo, vm2), ve2);
            int32x4_t r2_hi = vrshlq_s32(vqrdmulhq_s32(b_hi, vm2), ve2);
            int16x8_t sum = vaddq_s16(
                vcombine_s16(vqmovn_s32(vaddq_s32(r1_lo, r2_lo)),
                             vqmovn_s32(vaddq_s32(r1_hi, r2_hi))),
                vout);
            vst1_s8(output + i, vqmovn_s16(sum));
        }
        for (int i = nblk2 * 8; i < N; ++i) {
            int32_t v = apply_q31_scalar((int32_t)in1[i] - in1_zp, mm1, me1)
                      + apply_q31_scalar((int32_t)in2[i] - in2_zp, mm2, me2)
                      + out_zp;
            output[i] = static_cast<int8_t>(std::clamp(v, -128, 127));
        }
    }
    g_add_req_ms += now_ms() - _t0_add;
    return;
#else
#ifdef __AVX512BW__
    // Q12 int32 fast path: safe when |m1|, |m2| ≤ 7.99 (always true for ResNet-101).
    // Rounds each term separately (matches Q31 semantics): term1 = round((a-zp1)*m1),
    // term2 = round((b-zp2)*m2), out = term1 + term2 + out_zp.
    // Uses int32 arithmetic only — ~1.7× faster than the int64 path.
    if (std::abs(m1) <= 7.99f && std::abs(m2) <= 7.99f) {
        const int32_t M1 = (int32_t)std::roundf(m1 * 4096.0f);
        const int32_t M2 = (int32_t)std::roundf(m2 * 4096.0f);
        // Fold zp into per-term biases (each term rounded separately, matching Q31)
        const int32_t bias1 = -(int32_t)in1_zp * M1 + 2048;  // +2048 = round-half-up
        const int32_t bias2 = -(int32_t)in2_zp * M2 + 2048;
        const __m512i vm1    = _mm512_set1_epi32(M1);
        const __m512i vm2    = _mm512_set1_epi32(M2);
        const __m512i vbias1 = _mm512_set1_epi32(bias1);
        const __m512i vbias2 = _mm512_set1_epi32(bias2);
        const __m512i vozp   = _mm512_set1_epi32((int32_t)out_zp);
        const int nblk = N / 16;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(omp_get_max_threads() > 1)
#endif
        for (int bi = 0; bi < nblk; ++bi) {
            const int i = bi * 16;
            __m128i a8 = _mm_loadu_si128((const __m128i*)(in1 + i));
            __m128i b8 = _mm_loadu_si128((const __m128i*)(in2 + i));
            __m512i a32 = _mm512_cvtepi8_epi32(a8);
            __m512i b32 = _mm512_cvtepi8_epi32(b8);
            // Round each term separately then sum (matches Q31 rounding)
            __m512i t1 = _mm512_srai_epi32(_mm512_add_epi32(_mm512_mullo_epi32(a32, vm1), vbias1), 12);
            __m512i t2 = _mm512_srai_epi32(_mm512_add_epi32(_mm512_mullo_epi32(b32, vm2), vbias2), 12);
            __m512i acc = _mm512_add_epi32(_mm512_add_epi32(t1, t2), vozp);
            __m128i r8 = _mm512_cvtsepi32_epi8(acc);
            _mm_storeu_si128((__m128i*)(output + i), r8);
        }
        for (int i = nblk * 16; i < N; ++i) {
            int32_t v = apply_q31_scalar((int32_t)in1[i] - in1_zp, mm1, me1)
                      + apply_q31_scalar((int32_t)in2[i] - in2_zp, mm2, me2)
                      + out_zp;
            output[i] = static_cast<int8_t>(std::clamp(v, -128, 127));
        }
        g_add_req_ms += now_ms() - _t0_add;
        return;
    }
    // Q31 exact path using 64-bit arithmetic (fallback when |m| > 7.99).
    // Processes 16 int8s per iteration: extend to int32, split into two groups of 8,
    // sign-extend each group to int64, multiply with _mm512_mul_epi32, round-shift.
    {
        const int rshift1 = 31 - me1;
        const int rshift2 = 31 - me2;
        const int64_t rnd1 = 1LL << (rshift1 - 1);
        const int64_t rnd2 = 1LL << (rshift2 - 1);
        const __m512i vmm1   = _mm512_set1_epi64((int64_t)mm1);
        const __m512i vmm2   = _mm512_set1_epi64((int64_t)mm2);
        const __m512i vrnd1  = _mm512_set1_epi64(rnd1);
        const __m512i vrnd2  = _mm512_set1_epi64(rnd2);
        const __m512i vrs1   = _mm512_set1_epi64(rshift1);
        const __m512i vrs2   = _mm512_set1_epi64(rshift2);
        const __m512i vzp1v  = _mm512_set1_epi32(in1_zp);
        const __m512i vzp2v  = _mm512_set1_epi32(in2_zp);
        const __m512i voutzp = _mm512_set1_epi64((int64_t)out_zp);
        const int nblk = N / 16;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(omp_get_max_threads() > 1)
#endif
        for (int bi = 0; bi < nblk; ++bi) {
            const int i = bi * 16;
            __m128i a8 = _mm_loadu_si128((const __m128i*)(in1 + i));
            __m128i b8 = _mm_loadu_si128((const __m128i*)(in2 + i));
            // int8 -> int32, subtract zero points
            __m512i a32 = _mm512_sub_epi32(_mm512_cvtepi8_epi32(a8), vzp1v);
            __m512i b32 = _mm512_sub_epi32(_mm512_cvtepi8_epi32(b8), vzp2v);
            // Split 16 int32s into two groups of 8, sign-extend each to 8 int64
            // _mm512_cvtepi32_epi64 sign-extends 8 int32 (__m256i) -> 8 int64 (__m512i).
            // The int32 values land in the lower 32 bits of each int64 lane, which is
            // exactly where _mm512_mul_epi32 reads from (even int32 lanes = lower halves).
            __m512i a64_lo = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(a32));
            __m512i a64_hi = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(a32, 1));
            __m512i b64_lo = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(b32));
            __m512i b64_hi = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(b32, 1));
            // 64-bit products: (a - zp) * mult
            __m512i p1_lo = _mm512_srav_epi64(_mm512_add_epi64(_mm512_mul_epi32(a64_lo, vmm1), vrnd1), vrs1);
            __m512i p1_hi = _mm512_srav_epi64(_mm512_add_epi64(_mm512_mul_epi32(a64_hi, vmm1), vrnd1), vrs1);
            __m512i p2_lo = _mm512_srav_epi64(_mm512_add_epi64(_mm512_mul_epi32(b64_lo, vmm2), vrnd2), vrs2);
            __m512i p2_hi = _mm512_srav_epi64(_mm512_add_epi64(_mm512_mul_epi32(b64_hi, vmm2), vrnd2), vrs2);
            // sum = round(a*m1) + round(b*m2) + out_zp
            __m512i sum_lo = _mm512_add_epi64(_mm512_add_epi64(p1_lo, p2_lo), voutzp);
            __m512i sum_hi = _mm512_add_epi64(_mm512_add_epi64(p1_hi, p2_hi), voutzp);
            // Narrow: int64 -[sat]-> int32, combine, then int32 -[sat]-> int16 -[sat]-> int8
            __m256i r32_lo = _mm512_cvtsepi64_epi32(sum_lo);
            __m256i r32_hi = _mm512_cvtsepi64_epi32(sum_hi);
            __m512i r32 = _mm512_inserti64x4(_mm512_castsi256_si512(r32_lo), r32_hi, 1);
            __m256i r16 = _mm512_cvtsepi32_epi16(r32);
            __m128i r8  = _mm_packs_epi16(_mm256_castsi256_si128(r16),
                                           _mm256_extracti128_si256(r16, 1));
            _mm_storeu_si128((__m128i*)(output + i), r8);
        }
        for (int i = nblk * 16; i < N; ++i) {
            int32_t v = apply_q31_scalar((int32_t)in1[i] - in1_zp, mm1, me1)
                      + apply_q31_scalar((int32_t)in2[i] - in2_zp, mm2, me2)
                      + out_zp;
            output[i] = static_cast<int8_t>(std::clamp(v, -128, 127));
        }
        g_add_req_ms += now_ms() - _t0_add;
        return;
    }
#endif
    // Scalar Q31 fallback (non-AVX512 x86)
    for (int i = 0; i < N; ++i) {
        int32_t v = apply_q31_scalar((int32_t)in1[i] - in1_zp, mm1, me1)
                  + apply_q31_scalar((int32_t)in2[i] - in2_zp, mm2, me2)
                  + out_zp;
        output[i] = static_cast<int8_t>(std::clamp(v, -128, 127));
    }
#endif
    g_add_req_ms += now_ms() - _t0_add;
}


// ──────────────────────────────────────────────────────────────
// MaxPool int8 (NHWC layout) — [H, W, C] → [oH, oW, C]
// Vectorizes across C channels using NEON vmaxq_s8
// ──────────────────────────────────────────────────────────────
