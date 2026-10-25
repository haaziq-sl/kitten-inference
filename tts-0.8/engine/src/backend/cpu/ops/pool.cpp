#include "../ops_neon.hpp"
#include "profile_internal.hpp"
#include <cstdlib>
#include <limits>
#include <vector>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

void maxpool_int8(
    const int8_t* input,
    int8_t*       output,
    int C, int H, int W,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    StreamHandle /* stream */)
{
    double _t0_mp = now_ms();
    int oH = (H + 2 * pad_h - kH) / stride_h + 1;
    int oW = (W + 2 * pad_w - kW) / stride_w + 1;

    // Compute interior range (no padding needed)
    int oh_lo = (pad_h + stride_h - 1) / stride_h;
    int oh_hi = (H - kH + pad_h) / stride_h;
    int ow_lo = (pad_w + stride_w - 1) / stride_w;
    int ow_hi = (W - kW + pad_w) / stride_w;

    for (int c = 0; c < C; ++c) {
        const int8_t* in_c = input + c * H * W;
        int8_t*       out_c = output + c * oH * oW;

        // ── Interior: no boundary checks ────────────────────────
        for (int oh = oh_lo; oh <= oh_hi; ++oh) {
            const int ih0 = oh * stride_h - pad_h;
#ifdef __ARM_NEON
            // NEON fast path for kH=kW=3, stride_h=stride_w=2:
            // Process 8 output columns per iteration.
            // For output col ow (stride=2), input at kw=0,1,2 is at iw0, iw0+1, iw0+2
            // where iw0 = ow*2 - pad_w.
            // Load 16 bytes per row starting at iw0-1 (= ow*2-1); take 3-col max;
            // then stride-2 subsample via vuzp1_s8 to get 8 output values.
            if (kH == 3 && kW == 3 && stride_h == 2 && stride_w == 2) {
                int ow = ow_lo;
                for (; ow + 7 <= ow_hi; ow += 8) {
                    const int iw0 = ow * 2 - pad_w;  // = ow*2 - 1 for pad=1
                    const int8_t* r0 = in_c + (ih0 + 0) * W + iw0;
                    const int8_t* r1 = in_c + (ih0 + 1) * W + iw0;
                    const int8_t* r2 = in_c + (ih0 + 2) * W + iw0;

                    // Row 0: max of (col, col+1, col+2) for 16 positions
                    int8x16_t row0_a = vld1q_s8(r0);
                    int8x16_t row0_b = vld1q_s8(r0 + 1);
                    int8x16_t row0_c = vld1q_s8(r0 + 2);
                    int8x16_t rmax0 = vmaxq_s8(vmaxq_s8(row0_a, row0_b), row0_c);

                    // Row 1
                    int8x16_t row1_a = vld1q_s8(r1);
                    int8x16_t row1_b = vld1q_s8(r1 + 1);
                    int8x16_t row1_c = vld1q_s8(r1 + 2);
                    int8x16_t rmax1 = vmaxq_s8(vmaxq_s8(row1_a, row1_b), row1_c);

                    // Row 2
                    int8x16_t row2_a = vld1q_s8(r2);
                    int8x16_t row2_b = vld1q_s8(r2 + 1);
                    int8x16_t row2_c = vld1q_s8(r2 + 2);
                    int8x16_t rmax2 = vmaxq_s8(vmaxq_s8(row2_a, row2_b), row2_c);

                    // Pixel max across 3 rows
                    int8x16_t pmax = vmaxq_s8(vmaxq_s8(rmax0, rmax1), rmax2);

                    // Stride-2 subsample: extract even-indexed elements (0,2,4,...,14)
                    // → max values for output cols ow+0 .. ow+7
                    int8x8_t result = vuzp1_s8(vget_low_s8(pmax), vget_high_s8(pmax));
                    vst1_s8(out_c + oh * oW + ow, result);
                }
                // Scalar tail for remaining output columns
                for (; ow <= ow_hi; ++ow) {
                    const int iw0 = ow * stride_w - pad_w;
                    int8_t mx = -128;
                    for (int kh = 0; kh < 3; ++kh) {
                        const int8_t* row = in_c + (ih0 + kh) * W + iw0;
                        if (row[0] > mx) mx = row[0];
                        if (row[1] > mx) mx = row[1];
                        if (row[2] > mx) mx = row[2];
                    }
                    out_c[oh * oW + ow] = mx;
                }
                continue;  // skip generic interior loop for this oh
            }
#endif
            for (int ow = ow_lo; ow <= ow_hi; ++ow) {
                const int iw0 = ow * stride_w - pad_w;
                int8_t mx = -128;
                for (int kh = 0; kh < kH; ++kh) {
                    const int8_t* row = in_c + (ih0 + kh) * W + iw0;
                    for (int kw = 0; kw < kW; ++kw) {
                        if (row[kw] > mx) mx = row[kw];
                    }
                }
                out_c[oh * oW + ow] = mx;
            }
        }

        // ── Border: boundary-checked ──────────────────────────
        for (int oh = 0; oh < oH; ++oh) {
            for (int ow = 0; ow < oW; ++ow) {
                if (oh >= oh_lo && oh <= oh_hi && ow >= ow_lo && ow <= ow_hi) continue;
                int8_t mx = -128;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = oh * stride_h - pad_h + kh;
                    if (ih < 0 || ih >= H) continue;
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = ow * stride_w - pad_w + kw;
                        if (iw < 0 || iw >= W) continue;
                        int8_t v = in_c[ih * W + iw];
                        if (v > mx) mx = v;
                    }
                }
                out_c[oh * oW + ow] = mx;
            }
        }
    }
    g_maxpool_ms += now_ms() - _t0_mp;
}


// ──────────────────────────────────────────────────────────────
// Global average pool with requantization
// ──────────────────────────────────────────────────────────────
void avgpool_global_int8(
    const int8_t* input,
    float  in_scale,
    int    in_zp,
    float  out_scale,
    int    out_zp,
    int8_t* output,
    int C, int H, int W,
    StreamHandle /* stream */)
{
    int HW = H * W;

    // Fold HW into the scale: combined = (in_scale / out_scale) / HW
    // Precompute once in Q31 — no float inside the channel loop.
    int32_t mult, exp;
    QuantizeMultiplier(in_scale / (out_scale * static_cast<float>(HW)), &mult, &exp);

    for (int c = 0; c < C; ++c) {
        int32_t sum = 0;
        const int8_t* in_c = input + c * HW;
        for (int i = 0; i < HW; ++i)
            sum += static_cast<int32_t>(in_c[i]) - in_zp;
        // apply_q31_scalar: round(sum * combined_scale) + out_zp
        int32_t q = apply_q31_scalar(sum, mult, exp) + out_zp;
        output[c] = static_cast<int8_t>(std::clamp(q, -128, 127));
    }
}


// ──────────────────────────────────────────────────────────────
// Residual Add with requantization — NEON vectorized (16 elements/iter)

void maxpool_int8_nhwc(
    const int8_t* input,   // [H, W, C]
    int8_t*       output,  // [oH, oW, C]
    int C, int H, int W,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    StreamHandle /* stream */)
{
    double _t0_mp = now_ms();
    int oH = (H + 2 * pad_h - kH) / stride_h + 1;
    int oW = (W + 2 * pad_w - kW) / stride_w + 1;

    for (int oh = 0; oh < oH; ++oh) {
        for (int ow = 0; ow < oW; ++ow) {
            int8_t* out_ptr = output + (oh * oW + ow) * C;
            // Initialize to -128
#ifdef __ARM_NEON
            {
                int c = 0;
                int8x16_t vmin = vdupq_n_s8(-128);
                for (; c + 15 < C; c += 16)
                    vst1q_s8(out_ptr + c, vmin);
                for (; c < C; ++c)
                    out_ptr[c] = -128;
            }
#else
            memset(out_ptr, 0x80, C);  // -128
#endif
            for (int kh = 0; kh < kH; ++kh) {
                int ih = oh * stride_h - pad_h + kh;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < kW; ++kw) {
                    int iw = ow * stride_w - pad_w + kw;
                    if (iw < 0 || iw >= W) continue;
                    const int8_t* in_ptr = input + (ih * W + iw) * C;
#ifdef __ARM_NEON
                    int c = 0;
                    for (; c + 15 < C; c += 16) {
                        int8x16_t cur = vld1q_s8(out_ptr + c);
                        int8x16_t val = vld1q_s8(in_ptr + c);
                        vst1q_s8(out_ptr + c, vmaxq_s8(cur, val));
                    }
                    for (; c < C; ++c)
                        out_ptr[c] = std::max(out_ptr[c], in_ptr[c]);
#else
                    for (int c = 0; c < C; ++c)
                        out_ptr[c] = std::max(out_ptr[c], in_ptr[c]);
#endif
                }
            }
        }
    }
    g_maxpool_ms += now_ms() - _t0_mp;
}


// ──────────────────────────────────────────────────────────────
// Global average pool int8 → int8 (NHWC layout)
// Input: [H, W, C] → output: [C]
// Vectorizes accumulation across C channels using NEON
// ──────────────────────────────────────────────────────────────
void avgpool_global_int8_nhwc(
    const int8_t* input,       // [H, W, C]
    float  in_scale,
    int    in_zp,
    float  out_scale,
    int    out_zp,
    int8_t* output,            // [C]
    int C, int H, int W,
    StreamHandle /* stream */)
{
    int HW = H * W;

    // Fold HW into the scale
    int32_t mult, exp;
    QuantizeMultiplier(in_scale / (out_scale * static_cast<float>(HW)), &mult, &exp);

    // Accumulate sums across spatial dimensions
    // Use int32 accumulators per channel
    std::vector<int32_t> sums(C, 0);

    for (int hw = 0; hw < HW; ++hw) {
        const int8_t* row = input + hw * C;
#ifdef __ARM_NEON
        int c = 0;
        int16x8_t vzp = vdupq_n_s16((int16_t)in_zp);
        for (; c + 7 < C; c += 8) {
            int8x8_t v = vld1_s8(row + c);
            int16x8_t v16 = vsubl_s8(v, vdup_n_s8((int8_t)in_zp));
            int32x4_t lo = vld1q_s32(sums.data() + c);
            int32x4_t hi = vld1q_s32(sums.data() + c + 4);
            lo = vaddw_s16(lo, vget_low_s16(v16));
            hi = vaddw_s16(hi, vget_high_s16(v16));
            vst1q_s32(sums.data() + c, lo);
            vst1q_s32(sums.data() + c + 4, hi);
        }
        for (; c < C; ++c)
            sums[c] += (int32_t)row[c] - in_zp;
#else
        for (int c = 0; c < C; ++c)
            sums[c] += (int32_t)row[c] - in_zp;
#endif
    }

    for (int c = 0; c < C; ++c) {
        int32_t q = apply_q31_scalar(sums[c], mult, exp) + out_zp;
        output[c] = static_cast<int8_t>(std::clamp(q, -128, 127));
    }
}
