#include "work_split.hpp"
#include "../ops_neon.hpp"
#include "profile_internal.hpp"
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <cmath>
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
// Fast im2col for 1×1 convolutions (pad=0, any stride).
// For kH=kW=1, pad=0: col[m, c] = input[c, oh*sh, ow*sw].
// This is a (possibly subsampled) NCHW→NHWC transpose.
// Uses NEON 8×8 block transpose to avoid strided-gather overhead.
// ──────────────────────────────────────────────────────────────
#ifdef __ARM_NEON
// 8×8 int8 block transpose using 3-stage vzip on 64-bit vectors.
// Loads 8 rows of 8 bytes (one per channel), stores 8 columns of 8 bytes (one per pixel).
// src[i*stride .. +7]: 8 pixels for channel c_base+i
// dst[p*C_in .. +7]:   8 channels for pixel pixel_base+p
static inline __attribute__((always_inline)) void transpose8x8_store(
    const int8_t* src, int8_t* dst, int stride, int C_in)
{
    // Load 8 channel-rows, each of 8 consecutive spatial pixels.
    int8x8_t r0 = vld1_s8(src + 0*stride);
    int8x8_t r1 = vld1_s8(src + 1*stride);
    int8x8_t r2 = vld1_s8(src + 2*stride);
    int8x8_t r3 = vld1_s8(src + 3*stride);
    int8x8_t r4 = vld1_s8(src + 4*stride);
    int8x8_t r5 = vld1_s8(src + 5*stride);
    int8x8_t r6 = vld1_s8(src + 6*stride);
    int8x8_t r7 = vld1_s8(src + 7*stride);

    // Stage 1: interleave adjacent rows at byte level.
    // vzip_s8(rA, rB): val[0]=[A0,B0,A1,B1,A2,B2,A3,B3], val[1]=[A4,B4,A5,B5,A6,B6,A7,B7]
    int8x8x2_t p0 = vzip_s8(r0, r1);
    int8x8x2_t p1 = vzip_s8(r2, r3);
    int8x8x2_t p2 = vzip_s8(r4, r5);
    int8x8x2_t p3 = vzip_s8(r6, r7);

    // Stage 2: interleave at 16-bit level (2-byte pairs).
    // q0.val[0] as bytes: [a0,b0,c0,d0, a1,b1,c1,d1]
    // q0.val[1] as bytes: [a2,b2,c2,d2, a3,b3,c3,d3]
    int16x4x2_t q0 = vzip_s16(vreinterpret_s16_s8(p0.val[0]), vreinterpret_s16_s8(p1.val[0]));
    int16x4x2_t q1 = vzip_s16(vreinterpret_s16_s8(p2.val[0]), vreinterpret_s16_s8(p3.val[0]));
    int16x4x2_t q2 = vzip_s16(vreinterpret_s16_s8(p0.val[1]), vreinterpret_s16_s8(p1.val[1]));
    int16x4x2_t q3 = vzip_s16(vreinterpret_s16_s8(p2.val[1]), vreinterpret_s16_s8(p3.val[1]));

    // Stage 3: interleave at 32-bit level (4-byte groups).
    // v0.val[0] as bytes: [a0,b0,c0,d0,e0,f0,g0,h0] = pixel 0's 8 channels ✓
    // v0.val[1] as bytes: [a2,b2,c2,d2,e2,f2,g2,h2] = pixel 2's 8 channels ✓
    int32x2x2_t v0 = vzip_s32(vreinterpret_s32_s16(q0.val[0]), vreinterpret_s32_s16(q1.val[0]));
    int32x2x2_t v1 = vzip_s32(vreinterpret_s32_s16(q0.val[1]), vreinterpret_s32_s16(q1.val[1]));
    int32x2x2_t v2 = vzip_s32(vreinterpret_s32_s16(q2.val[0]), vreinterpret_s32_s16(q3.val[0]));
    int32x2x2_t v3 = vzip_s32(vreinterpret_s32_s16(q2.val[1]), vreinterpret_s32_s16(q3.val[1]));

    // After 3-stage vzip: v0=(px0,px1), v1=(px2,px3), v2=(px4,px5), v3=(px6,px7)
    vst1_s8(dst + 0*C_in, vreinterpret_s8_s32(v0.val[0]));  // pixel 0
    vst1_s8(dst + 1*C_in, vreinterpret_s8_s32(v0.val[1]));  // pixel 1
    vst1_s8(dst + 2*C_in, vreinterpret_s8_s32(v1.val[0]));  // pixel 2
    vst1_s8(dst + 3*C_in, vreinterpret_s8_s32(v1.val[1]));  // pixel 3
    vst1_s8(dst + 4*C_in, vreinterpret_s8_s32(v2.val[0]));  // pixel 4
    vst1_s8(dst + 5*C_in, vreinterpret_s8_s32(v2.val[1]));  // pixel 5
    vst1_s8(dst + 6*C_in, vreinterpret_s8_s32(v3.val[0]));  // pixel 6
    vst1_s8(dst + 7*C_in, vreinterpret_s8_s32(v3.val[1]));  // pixel 7
}
#endif

// ── im2col_1x1 row helpers — separate functions for clean register allocation ──
// The __restrict__ pointers let GCC keep int params in registers across int8* writes.
#ifdef __ARM_NEON
static void im2col_1x1_neon_oh(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ col,
    int oh, int C_in, int HW, int oW, int stride_h, int W)
{
    const int row_base = oh * stride_h * W;
    int ow = 0;
    for (; ow + 7 < oW; ow += 8) {
        int8_t* dst = col + (oh * oW + ow) * C_in;
        const int8_t* src_pix = input + row_base + ow;
        int c = 0;
        for (; c + 7 < C_in; c += 8)
            transpose8x8_store(src_pix + c * HW, dst + c, HW, C_in);
        for (; c < C_in; ++c) {
            const int8_t* src_c = input + c * HW + row_base;
            for (int p = 0; p < 8; ++p)
                dst[p * C_in + c] = src_c[ow + p];
        }
    }
    for (; ow < oW; ++ow) {
        int8_t* dst = col + (oh * oW + ow) * C_in;
        for (int c = 0; c < C_in; ++c)
            dst[c] = input[c * HW + row_base + ow];
    }
}
#endif

static void im2col_1x1_gen_oh(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ col,
    int oh, int C_in, int HW, int oW, int stride_h, int stride_w, int W)
{
    const int ih = oh * stride_h;
    for (int ow = 0; ow < oW; ++ow) {
        const int iw = ow * stride_w;
        int8_t* dst = col + (oh * oW + ow) * C_in;
        for (int c = 0; c < C_in; ++c)
            dst[c] = input[c * HW + ih * W + iw];
    }
}

static void im2col_1x1(
    const int8_t* input,
    int C_in, int H, int W,
    int stride_h, int stride_w,
    int oH, int oW,
    int8_t* col,
    bool in_parallel = false)   // true → use #pragma omp for (existing team)
{
    // col[oh*oW + ow, c] = input[c, oh*stride_h, ow*stride_w]
    const int HW = H * W;

#ifdef __ARM_NEON
    if (stride_w == 1) {
#ifdef _OPENMP
        if (in_parallel) {
#pragma omp for schedule(static)
            for (int oh = 0; oh < oH; ++oh)
                im2col_1x1_neon_oh(input, col, oh, C_in, HW, oW, stride_h, W);
        } else if (omp_get_max_threads() > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static)
            for (int oh = 0; oh < oH; ++oh)
                im2col_1x1_neon_oh(input, col, oh, C_in, HW, oW, stride_h, W);
        } else {
            for (int oh = 0; oh < oH; ++oh)
                im2col_1x1_neon_oh(input, col, oh, C_in, HW, oW, stride_h, W);
        }
#else
        for (int oh = 0; oh < oH; ++oh)
            im2col_1x1_neon_oh(input, col, oh, C_in, HW, oW, stride_h, W);
#endif
        return;
    }
#endif
    // General path: stride != 1 or no NEON
#ifdef _OPENMP
    if (in_parallel) {
#pragma omp for schedule(static)
        for (int oh = 0; oh < oH; ++oh)
            im2col_1x1_gen_oh(input, col, oh, C_in, HW, oW, stride_h, stride_w, W);
    } else if (omp_get_max_threads() > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static)
        for (int oh = 0; oh < oH; ++oh)
            im2col_1x1_gen_oh(input, col, oh, C_in, HW, oW, stride_h, stride_w, W);
    } else {
        for (int oh = 0; oh < oH; ++oh)
            im2col_1x1_gen_oh(input, col, oh, C_in, HW, oW, stride_h, stride_w, W);
    }
#else
    for (int oh = 0; oh < oH; ++oh)
        im2col_1x1_gen_oh(input, col, oh, C_in, HW, oW, stride_h, stride_w, W);
#endif
}


// ──────────────────────────────────────────────────────────────
// Specialised im2col for 3×3 conv, stride=2, pad=1.
// Same 4-byte load + uint64 pack trick as the stride=1 version.
// Used for the 3 downsampling bottleneck 3×3 convs in ResNet101.
// ──────────────────────────────────────────────────────────────
// ── im2col_3x3s2p1 row helpers ────────────────────────────────────────────
static void im2col_3x3s2p1_interior_oh(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ col,
    int oh, int C_in, int HW, int K, int W, int oW, int ow_lo, int ow_hi)
{
    const int row0_off = (oh * 2 - 1) * W;
    for (int ow = ow_lo; ow <= ow_hi; ++ow) {
        int8_t* dst = col + (oh * oW + ow) * K;
        const int base_col = row0_off + (ow * 2 - 1);
        for (int c = 0; c < C_in; ++c) {
            const int8_t* base = input + (size_t)c * HW + base_col;
            uint32_t v0, v1, v2;
            memcpy(&v0, base,         4);
            memcpy(&v1, base + W,     4);
            memcpy(&v2, base + 2 * W, 4);
            uint64_t q = ((uint64_t)(v0 & 0xFFFFFFu))
                       | ((uint64_t)(v1 & 0xFFFFFFu) << 24)
                       | ((uint64_t)(v2 & 0xFFFFu)   << 48);
            memcpy(dst, &q, 8);
            dst[8] = (uint8_t)(v2 >> 16);
            dst += 9;
        }
    }
}

static void im2col_3x3s2p1_border_oh(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ col,
    int oh, int C_in, int HW, int K, int H, int W, int oW,
    int oh_lo, int oh_hi, int ow_lo, int ow_hi, int8_t pad_val)
{
    for (int ow = 0; ow < oW; ++ow) {
        if (oh >= oh_lo && oh <= oh_hi && ow >= ow_lo && ow <= ow_hi) continue;
        int8_t* dst = col + (oh * oW + ow) * K;
        for (int c = 0; c < C_in; ++c) {
            const int8_t* in_c = input + (size_t)c * HW;
            for (int kh = 0; kh < 3; ++kh) {
                int ih = oh * 2 - 1 + kh;
                if (ih < 0 || ih >= H) {
                    dst[0] = dst[1] = dst[2] = pad_val;
                } else {
                    const int8_t* row = in_c + ih * W;
                    for (int kw = 0; kw < 3; ++kw) {
                        int iw = ow * 2 - 1 + kw;
                        dst[kw] = (iw >= 0 && iw < W) ? row[iw] : pad_val;
                    }
                }
                dst += 3;
            }
        }
    }
}

static void im2col_3x3s2p1(
    const int8_t* input,
    int C_in, int H, int W,
    int8_t pad_val,
    int oH, int oW,
    int8_t* col,                // [oH*oW, 9*C_in]
    bool in_parallel = false)   // true → use #pragma omp for (existing team)
{
    const int K  = 9 * C_in;
    const int HW = H * W;

    const int oh_lo = 1, oh_hi = (H - 2) / 2;
    const int ow_lo = 1, ow_hi = (W - 2) / 2;

#ifdef _OPENMP
    if (in_parallel) {
#pragma omp for schedule(static) nowait
        for (int oh = oh_lo; oh <= oh_hi; ++oh)
            im2col_3x3s2p1_interior_oh(input, col, oh, C_in, HW, K, W, oW, ow_lo, ow_hi);
    } else if (omp_get_max_threads() > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static)
        for (int oh = oh_lo; oh <= oh_hi; ++oh)
            im2col_3x3s2p1_interior_oh(input, col, oh, C_in, HW, K, W, oW, ow_lo, ow_hi);
    } else {
        for (int oh = oh_lo; oh <= oh_hi; ++oh)
            im2col_3x3s2p1_interior_oh(input, col, oh, C_in, HW, K, W, oW, ow_lo, ow_hi);
    }
#else
    for (int oh = oh_lo; oh <= oh_hi; ++oh)
        im2col_3x3s2p1_interior_oh(input, col, oh, C_in, HW, K, W, oW, ow_lo, ow_hi);
#endif

#ifdef _OPENMP
    if (in_parallel) {
#pragma omp for schedule(static)
        for (int oh = 0; oh < oH; ++oh)
            im2col_3x3s2p1_border_oh(input, col, oh, C_in, HW, K, H, W, oW,
                                     oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
    } else if (omp_get_max_threads() > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static)
        for (int oh = 0; oh < oH; ++oh)
            im2col_3x3s2p1_border_oh(input, col, oh, C_in, HW, K, H, W, oW,
                                     oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
    } else {
        for (int oh = 0; oh < oH; ++oh)
            im2col_3x3s2p1_border_oh(input, col, oh, C_in, HW, K, H, W, oW,
                                     oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
    }
#else
    for (int oh = 0; oh < oH; ++oh)
        im2col_3x3s2p1_border_oh(input, col, oh, C_in, HW, K, H, W, oW,
                                 oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
#endif
}


// ──────────────────────────────────────────────────────────────
// Specialised im2col for 3×3 conv, stride=1, pad=1 (oH=H, oW=W).
// Splits the spatial domain into border rows/cols (with boundary
// checks, a small minority) and the interior (fully branch-free).
// For ResNet101 all 3×3 convs use exactly these parameters.
// ──────────────────────────────────────────────────────────────
// ── im2col_3x3s1p1 row helpers ────────────────────────────────────────────
static void im2col_3x3s1p1_interior_oh(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ col,
    int oh, int C_in, int HW, int K, int W)
{
    const int row0_off = (oh - 1) * W;
    for (int ow = 1; ow < W - 1; ++ow) {
        int8_t* dst = col + (oh * W + ow) * K;
        const int base_col = row0_off + (ow - 1);
        for (int c = 0; c < C_in; ++c) {
            const int8_t* base = input + (size_t)c * HW + base_col;
            uint32_t v0, v1, v2;
            memcpy(&v0, base,         4);
            memcpy(&v1, base + W,     4);
            memcpy(&v2, base + 2 * W, 4);
            uint64_t q = ((uint64_t)(v0 & 0xFFFFFFu))
                       | ((uint64_t)(v1 & 0xFFFFFFu) << 24)
                       | ((uint64_t)(v2 & 0xFFFFu)   << 48);
            memcpy(dst, &q, 8);
            dst[8] = (uint8_t)(v2 >> 16);
            dst += 9;
        }
    }
}

static void im2col_3x3s1p1_border_oh(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ col,
    int oh, int C_in, int HW, int K, int H, int W, int8_t pad_val)
{
    for (int ow = 0; ow < W; ++ow) {
        if (oh > 0 && oh < H - 1 && ow > 0 && ow < W - 1) continue;
        int8_t* dst = col + (oh * W + ow) * K;
        for (int c = 0; c < C_in; ++c) {
            const int8_t* in_c = input + (size_t)c * HW;
            for (int kh = 0; kh < 3; ++kh) {
                int ih = oh - 1 + kh;
                if (ih < 0 || ih >= H) {
                    dst[0] = dst[1] = dst[2] = pad_val;
                } else {
                    const int8_t* row = in_c + ih * W;
                    dst[0] = (ow > 0)     ? row[ow - 1] : pad_val;
                    dst[1] =                row[ow];
                    dst[2] = (ow < W - 1) ? row[ow + 1] : pad_val;
                }
                dst += 3;
            }
        }
    }
}

static void im2col_3x3s1p1(
    const int8_t* input,
    int C_in, int H, int W,
    int8_t pad_val,
    int8_t* col,                // [H*W, 9*C_in]
    bool in_parallel = false)   // true → use #pragma omp for (existing team)
{
    const int K  = 9 * C_in;
    const int HW = H * W;

#ifdef _OPENMP
    if (in_parallel) {
#pragma omp for schedule(static) nowait
        for (int oh = 1; oh < H - 1; ++oh)
            im2col_3x3s1p1_interior_oh(input, col, oh, C_in, HW, K, W);
    } else if (omp_get_max_threads() > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static)
        for (int oh = 1; oh < H - 1; ++oh)
            im2col_3x3s1p1_interior_oh(input, col, oh, C_in, HW, K, W);
    } else {
        for (int oh = 1; oh < H - 1; ++oh)
            im2col_3x3s1p1_interior_oh(input, col, oh, C_in, HW, K, W);
    }
#else
    for (int oh = 1; oh < H - 1; ++oh)
        im2col_3x3s1p1_interior_oh(input, col, oh, C_in, HW, K, W);
#endif

#ifdef _OPENMP
    if (in_parallel) {
#pragma omp for schedule(static)
        for (int oh = 0; oh < H; ++oh)
            im2col_3x3s1p1_border_oh(input, col, oh, C_in, HW, K, H, W, pad_val);
    } else if (omp_get_max_threads() > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static)
        for (int oh = 0; oh < H; ++oh)
            im2col_3x3s1p1_border_oh(input, col, oh, C_in, HW, K, H, W, pad_val);
    } else {
        for (int oh = 0; oh < H; ++oh)
            im2col_3x3s1p1_border_oh(input, col, oh, C_in, HW, K, H, W, pad_val);
    }
#else
    for (int oh = 0; oh < H; ++oh)
        im2col_3x3s1p1_border_oh(input, col, oh, C_in, HW, K, H, W, pad_val);
#endif
}

// ──────────────────────────────────────────────────────────────
// Specialized im2col for 7×7 stride-2 pad-3 (ResNet101 stem conv only).
// Interior pixels use 7 unrolled memcpy(7) calls per channel so the compiler
// can merge them into a single ~49-byte store, eliminating the kH loop overhead.
// ──────────────────────────────────────────────────────────────
// ── im2col_7x7s2p3 row helpers ────────────────────────────────────────────
static void im2col_7x7s2p3_interior_oh(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ col,
    int oh, int C_in, int HW, int K, int W, int oW, int ow_lo, int ow_hi)
{
    for (int ow = ow_lo; ow <= ow_hi; ++ow) {
        int8_t* dst = col + (oh * oW + ow) * K;
        const int ih0 = oh * 2 - 3;
        const int iw0 = ow * 2 - 3;
        for (int c = 0; c < C_in; ++c) {
            const int8_t* base = input + (size_t)c * HW + ih0 * W + iw0;
            memcpy(dst,    base,           7);
            memcpy(dst+ 7, base + W,       7);
            memcpy(dst+14, base + 2 * W,   7);
            memcpy(dst+21, base + 3 * W,   7);
            memcpy(dst+28, base + 4 * W,   7);
            memcpy(dst+35, base + 5 * W,   7);
            memcpy(dst+42, base + 6 * W,   7);
            dst += 49;
        }
    }
}

static void im2col_7x7s2p3_border_oh(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ col,
    int oh, int C_in, int HW, int K, int H, int W, int oW,
    int oh_lo, int oh_hi, int ow_lo, int ow_hi, int8_t pad_val)
{
    for (int ow = 0; ow < oW; ++ow) {
        if (oh >= oh_lo && oh <= oh_hi && ow >= ow_lo && ow <= ow_hi) continue;
        int8_t* dst = col + (oh * oW + ow) * K;
        int idx = 0;
        for (int c = 0; c < C_in; ++c) {
            const int8_t* in_c = input + (size_t)c * HW;
            for (int kh = 0; kh < 7; ++kh) {
                int ih = oh * 2 - 3 + kh;
                if (ih < 0 || ih >= H) {
                    memset(dst + idx, pad_val, 7); idx += 7;
                } else {
                    const int8_t* row = in_c + ih * W;
                    for (int kw = 0; kw < 7; ++kw) {
                        int iw = ow * 2 - 3 + kw;
                        dst[idx++] = (iw >= 0 && iw < W) ? row[iw] : pad_val;
                    }
                }
            }
        }
    }
}

static void im2col_7x7s2p3(
    const int8_t* input,
    int C_in, int H, int W,
    int8_t pad_val,
    int oH, int oW,
    int8_t* col,                // [oH*oW, 49*C_in]
    bool in_parallel = false)   // true → use #pragma omp for (existing team)
{
    const int K  = 49 * C_in;
    const int HW = H * W;

    const int oh_lo = 2, oh_hi = (H - 4) / 2;
    const int ow_lo = 2, ow_hi = (W - 4) / 2;

#ifdef _OPENMP
    if (in_parallel) {
#pragma omp for schedule(static) nowait
        for (int oh = oh_lo; oh <= oh_hi; ++oh)
            im2col_7x7s2p3_interior_oh(input, col, oh, C_in, HW, K, W, oW, ow_lo, ow_hi);
    } else if (omp_get_max_threads() > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static)
        for (int oh = oh_lo; oh <= oh_hi; ++oh)
            im2col_7x7s2p3_interior_oh(input, col, oh, C_in, HW, K, W, oW, ow_lo, ow_hi);
    } else {
        for (int oh = oh_lo; oh <= oh_hi; ++oh)
            im2col_7x7s2p3_interior_oh(input, col, oh, C_in, HW, K, W, oW, ow_lo, ow_hi);
    }
#else
    for (int oh = oh_lo; oh <= oh_hi; ++oh)
        im2col_7x7s2p3_interior_oh(input, col, oh, C_in, HW, K, W, oW, ow_lo, ow_hi);
#endif

#ifdef _OPENMP
    if (in_parallel) {
#pragma omp for schedule(static)
        for (int oh = 0; oh < oH; ++oh)
            im2col_7x7s2p3_border_oh(input, col, oh, C_in, HW, K, H, W, oW,
                                     oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
    } else if (omp_get_max_threads() > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static)
        for (int oh = 0; oh < oH; ++oh)
            im2col_7x7s2p3_border_oh(input, col, oh, C_in, HW, K, H, W, oW,
                                     oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
    } else {
        for (int oh = 0; oh < oH; ++oh)
            im2col_7x7s2p3_border_oh(input, col, oh, C_in, HW, K, H, W, oW,
                                     oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
    }
#else
    for (int oh = 0; oh < oH; ++oh)
        im2col_7x7s2p3_border_oh(input, col, oh, C_in, HW, K, H, W, oW,
                                 oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
#endif
}

// ──────────────────────────────────────────────────────────────
// im2col: expand input patches into rows
// Input:  [1, C_in, H, W]  NCHW
// Output: [oH*oW, kH*kW*C_in]  — one row per output pixel (GEMM-friendly)
// ──────────────────────────────────────────────────────────────
static void im2col(
    const int8_t* input,
    int C_in, int H, int W,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int oH, int oW,
    int8_t* col,      // [oH*oW, kH*kW*C_in]
    int8_t  pad_val)
{
    const int K  = kH * kW * C_in;
    const int HW = H * W;

    // Compute interior range where no boundary checking is needed.
    // oh_lo..oh_hi / ow_lo..ow_hi are the output positions whose
    // entire kH×kW receptive field lies inside the input.
    int oh_lo = (pad_h + stride_h - 1) / stride_h;
    int oh_hi = (H - kH + pad_h) / stride_h;
    int ow_lo = (pad_w + stride_w - 1) / stride_w;
    int ow_hi = (W - kW + pad_w) / stride_w;

    // ── Interior: no bounds check, use memcpy per kernel row ──
    for (int oh = oh_lo; oh <= oh_hi; ++oh) {
        for (int ow = ow_lo; ow <= ow_hi; ++ow) {
            int8_t* dst = col + (oh * oW + ow) * K;
            int ih0 = oh * stride_h - pad_h;
            int iw0 = ow * stride_w - pad_w;
            for (int c = 0; c < C_in; ++c) {
                const int8_t* in_c = input + (size_t)c * HW;
                for (int kh = 0; kh < kH; ++kh) {
                    memcpy(dst, in_c + (ih0 + kh) * W + iw0, kW);
                    dst += kW;
                }
            }
        }
    }

    // ── Border: output positions touching the padding boundary ──
    for (int oh = 0; oh < oH; ++oh) {
        for (int ow = 0; ow < oW; ++ow) {
            if (oh >= oh_lo && oh <= oh_hi && ow >= ow_lo && ow <= ow_hi) continue;
            int8_t* dst = col + (oh * oW + ow) * K;
            int idx = 0;
            for (int c = 0; c < C_in; ++c) {
                const int8_t* in_c = input + (size_t)c * HW;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = oh * stride_h - pad_h + kh;
                    if (ih < 0 || ih >= H) {
                        memset(dst + idx, pad_val, kW);
                        idx += kW;
                    } else {
                        const int8_t* in_row = in_c + ih * W;
                        for (int kw = 0; kw < kW; ++kw) {
                            int iw = ow * stride_w - pad_w + kw;
                            dst[idx++] = (iw >= 0 && iw < W) ? in_row[iw] : pad_val;
                        }
                    }
                }
            }
        }
    }
}


// ──────────────────────────────────────────────────────────────
// NHWC im2col for 3×3 conv, stride=1, pad=1
// Input: [H, W, C_in] NHWC → col: [H*W, 9*C_in]
// ──────────────────────────────────────────────────────────────
static void im2col_3x3_nhwc_s1p1_interior_oh(
    const int8_t* __restrict__ input, int8_t* __restrict__ col,
    int oh, int C_in, int W, int K, int C3)
{
    for (int ow = 1; ow < W - 1; ++ow) {
        int8_t* dst = col + (oh * W + ow) * K;
        for (int kh = 0; kh < 3; ++kh) {
            const int8_t* src = input + ((oh - 1 + kh) * W + (ow - 1)) * C_in;
            memcpy(dst + kh * C3, src, C3);
        }
    }
}

static void im2col_3x3_nhwc_s1p1_border_oh(
    const int8_t* __restrict__ input, int8_t* __restrict__ col,
    int oh, int C_in, int H, int W, int K, int C3, int8_t pad_val)
{
    for (int ow = 0; ow < W; ++ow) {
        if (oh > 0 && oh < H - 1 && ow > 0 && ow < W - 1) continue;
        int8_t* dst = col + (oh * W + ow) * K;
        for (int kh = 0; kh < 3; ++kh) {
            int ih = oh - 1 + kh;
            if (ih < 0 || ih >= H) {
                memset(dst + kh * C3, pad_val, C3);
            } else {
                for (int kw = 0; kw < 3; ++kw) {
                    int iw = ow - 1 + kw;
                    if (iw < 0 || iw >= W) {
                        memset(dst + (kh * 3 + kw) * C_in, pad_val, C_in);
                    } else {
                        memcpy(dst + (kh * 3 + kw) * C_in, input + (ih * W + iw) * C_in, C_in);
                    }
                }
            }
        }
    }
}

static void im2col_3x3_nhwc_s1p1(
    const int8_t* input,   // [H, W, C_in] NHWC
    int C_in, int H, int W,
    int8_t pad_val,
    int8_t* col,           // [H*W, 9*C_in]
    bool in_parallel = false)
{
    const int K = 9 * C_in;
    const int C3 = 3 * C_in;

#ifdef _OPENMP
    if (in_parallel) {
#pragma omp for schedule(static) nowait
        for (int oh = 1; oh < H - 1; ++oh)
            im2col_3x3_nhwc_s1p1_interior_oh(input, col, oh, C_in, W, K, C3);
    } else {
        for (int oh = 1; oh < H - 1; ++oh)
            im2col_3x3_nhwc_s1p1_interior_oh(input, col, oh, C_in, W, K, C3);
    }
#else
    for (int oh = 1; oh < H - 1; ++oh)
        im2col_3x3_nhwc_s1p1_interior_oh(input, col, oh, C_in, W, K, C3);
#endif

#ifdef _OPENMP
    if (in_parallel) {
#pragma omp for schedule(static)
        for (int oh = 0; oh < H; ++oh)
            im2col_3x3_nhwc_s1p1_border_oh(input, col, oh, C_in, H, W, K, C3, pad_val);
    } else {
        for (int oh = 0; oh < H; ++oh)
            im2col_3x3_nhwc_s1p1_border_oh(input, col, oh, C_in, H, W, K, C3, pad_val);
    }
#else
    for (int oh = 0; oh < H; ++oh)
        im2col_3x3_nhwc_s1p1_border_oh(input, col, oh, C_in, H, W, K, C3, pad_val);
#endif
}

// ──────────────────────────────────────────────────────────────
// NHWC im2col for 3×3 conv, stride=2, pad=1
// Input: [H, W, C_in] NHWC → col: [oH*oW, 9*C_in]
// ──────────────────────────────────────────────────────────────
static void im2col_3x3_nhwc_s2p1_interior_oh(
    const int8_t* __restrict__ input, int8_t* __restrict__ col,
    int oh, int C_in, int W, int oW, int K, int C3, int ow_lo, int ow_hi)
{
    for (int ow = ow_lo; ow <= ow_hi; ++ow) {
        int8_t* dst = col + (oh * oW + ow) * K;
        for (int kh = 0; kh < 3; ++kh) {
            const int8_t* src = input + ((oh * 2 - 1 + kh) * W + (ow * 2 - 1)) * C_in;
            memcpy(dst + kh * C3, src, C3);
        }
    }
}

static void im2col_3x3_nhwc_s2p1_border_oh(
    const int8_t* __restrict__ input, int8_t* __restrict__ col,
    int oh, int C_in, int H, int W, int oH, int oW, int K, int C3,
    int oh_lo, int oh_hi, int ow_lo, int ow_hi, int8_t pad_val)
{
    for (int ow = 0; ow < oW; ++ow) {
        if (oh >= oh_lo && oh <= oh_hi && ow >= ow_lo && ow <= ow_hi) continue;
        int8_t* dst = col + (oh * oW + ow) * K;
        for (int kh = 0; kh < 3; ++kh) {
            int ih = oh * 2 - 1 + kh;
            if (ih < 0 || ih >= H) {
                memset(dst + kh * C3, pad_val, C3);
            } else {
                for (int kw = 0; kw < 3; ++kw) {
                    int iw = ow * 2 - 1 + kw;
                    if (iw < 0 || iw >= W) {
                        memset(dst + (kh * 3 + kw) * C_in, pad_val, C_in);
                    } else {
                        memcpy(dst + (kh * 3 + kw) * C_in, input + (ih * W + iw) * C_in, C_in);
                    }
                }
            }
        }
    }
}

static void im2col_3x3_nhwc_s2p1(
    const int8_t* input,   // [H, W, C_in] NHWC
    int C_in, int H, int W,
    int8_t pad_val,
    int oH, int oW,
    int8_t* col,           // [oH*oW, 9*C_in]
    bool in_parallel = false)
{
    const int K = 9 * C_in;
    const int C3 = 3 * C_in;

    const int oh_lo = 1, oh_hi = (H - 2) / 2;
    const int ow_lo = 1, ow_hi = (W - 2) / 2;

#ifdef _OPENMP
    if (in_parallel) {
#pragma omp for schedule(static) nowait
        for (int oh = oh_lo; oh <= oh_hi; ++oh)
            im2col_3x3_nhwc_s2p1_interior_oh(input, col, oh, C_in, W, oW, K, C3, ow_lo, ow_hi);
    } else {
        for (int oh = oh_lo; oh <= oh_hi; ++oh)
            im2col_3x3_nhwc_s2p1_interior_oh(input, col, oh, C_in, W, oW, K, C3, ow_lo, ow_hi);
    }
#else
    for (int oh = oh_lo; oh <= oh_hi; ++oh)
        im2col_3x3_nhwc_s2p1_interior_oh(input, col, oh, C_in, W, oW, K, C3, ow_lo, ow_hi);
#endif

#ifdef _OPENMP
    if (in_parallel) {
#pragma omp for schedule(static)
        for (int oh = 0; oh < oH; ++oh)
            im2col_3x3_nhwc_s2p1_border_oh(input, col, oh, C_in, H, W, oH, oW, K, C3, oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
    } else {
        for (int oh = 0; oh < oH; ++oh)
            im2col_3x3_nhwc_s2p1_border_oh(input, col, oh, C_in, H, W, oH, oW, K, C3, oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
    }
#else
    for (int oh = 0; oh < oH; ++oh)
        im2col_3x3_nhwc_s2p1_border_oh(input, col, oh, C_in, H, W, oH, oW, K, C3, oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
#endif
}

// ──────────────────────────────────────────────────────────────
// NHWC im2col for 7×7 conv, stride=2, pad=3 (stem conv)
// Input: [H, W, C_in] NHWC → col: [oH*oW, 49*C_in]
// ──────────────────────────────────────────────────────────────
static void im2col_7x7_nhwc_s2p3_interior_oh(
    const int8_t* __restrict__ input, int8_t* __restrict__ col,
    int oh, int C_in, int W, int oW, int K, int C7, int ow_lo, int ow_hi)
{
    for (int ow = ow_lo; ow <= ow_hi; ++ow) {
        int8_t* dst = col + (oh * oW + ow) * K;
        const int ih0 = oh * 2 - 3;
        const int iw0 = ow * 2 - 3;
        for (int kh = 0; kh < 7; ++kh) {
            memcpy(dst + kh * C7, input + ((ih0 + kh) * W + iw0) * C_in, C7);
        }
    }
}

static void im2col_7x7_nhwc_s2p3_border_oh(
    const int8_t* __restrict__ input, int8_t* __restrict__ col,
    int oh, int C_in, int H, int W, int oH, int oW, int K, int C7,
    int oh_lo, int oh_hi, int ow_lo, int ow_hi, int8_t pad_val)
{
    for (int ow = 0; ow < oW; ++ow) {
        if (oh >= oh_lo && oh <= oh_hi && ow >= ow_lo && ow <= ow_hi) continue;
        int8_t* dst = col + (oh * oW + ow) * K;
        for (int kh = 0; kh < 7; ++kh) {
            int ih = oh * 2 - 3 + kh;
            if (ih < 0 || ih >= H) {
                memset(dst + kh * C7, pad_val, C7);
            } else {
                for (int kw = 0; kw < 7; ++kw) {
                    int iw = ow * 2 - 3 + kw;
                    if (iw < 0 || iw >= W) {
                        memset(dst + (kh * 7 + kw) * C_in, pad_val, C_in);
                    } else {
                        memcpy(dst + (kh * 7 + kw) * C_in, input + (ih * W + iw) * C_in, C_in);
                    }
                }
            }
        }
    }
}

static void im2col_7x7_nhwc_s2p3(
    const int8_t* input,   // [H, W, C_in] NHWC
    int C_in, int H, int W,
    int8_t pad_val,
    int oH, int oW,
    int8_t* col,           // [oH*oW, 49*C_in]
    bool in_parallel = false)
{
    const int K = 49 * C_in;
    const int C7 = 7 * C_in;

    const int oh_lo = 2, oh_hi = (H - 4) / 2;
    const int ow_lo = 2, ow_hi = (W - 4) / 2;

#ifdef _OPENMP
    if (in_parallel) {
#pragma omp for schedule(static) nowait
        for (int oh = oh_lo; oh <= oh_hi; ++oh)
            im2col_7x7_nhwc_s2p3_interior_oh(input, col, oh, C_in, W, oW, K, C7, ow_lo, ow_hi);
    } else {
        for (int oh = oh_lo; oh <= oh_hi; ++oh)
            im2col_7x7_nhwc_s2p3_interior_oh(input, col, oh, C_in, W, oW, K, C7, ow_lo, ow_hi);
    }
#else
    for (int oh = oh_lo; oh <= oh_hi; ++oh)
        im2col_7x7_nhwc_s2p3_interior_oh(input, col, oh, C_in, W, oW, K, C7, ow_lo, ow_hi);
#endif

#ifdef _OPENMP
    if (in_parallel) {
#pragma omp for schedule(static)
        for (int oh = 0; oh < oH; ++oh)
            im2col_7x7_nhwc_s2p3_border_oh(input, col, oh, C_in, H, W, oH, oW, K, C7, oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
    } else {
        for (int oh = 0; oh < oH; ++oh)
            im2col_7x7_nhwc_s2p3_border_oh(input, col, oh, C_in, H, W, oH, oW, K, C7, oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
    }
#else
    for (int oh = 0; oh < oH; ++oh)
        im2col_7x7_nhwc_s2p3_border_oh(input, col, oh, C_in, H, W, oH, oW, K, C7, oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
#endif
}


// ──────────────────────────────────────────────────────────────
// INT8 GEMM -> INT32 accumulators (no requantization).
// Fast 16-row SDOT micro-kernel version (matches gemm_int8_neon).
// Outputs raw INT32 dot products: C[m*N + n] = sum_k A[m,k]*B[n,k].
// Used for Winograd domain GEMMs.
// ──────────────────────────────────────────────────────────────

void conv2d_winograd_nhwc_int8(
    const int8_t*  input,
    const int8_t*  const* w_wino_packed,
    const float*   wino_weight_scale,
    float          wino_input_scale,
    const int64_t* eff_bias,
    const int32_t* req_mult,
    const int32_t* req_exp,
    const float*   req_scale,
    int8_t         in_zp,
    int8_t         out_zp,
    int8_t*        output,
    int C_in, int H, int W,
    int C_out,
    int8_t*  scratch_input_hat,
    int32_t* scratch_output_hat,
    const int32_t* const* w_wino_row_sums,
    StreamHandle /* stream */)
{
    const int oH = H, oW = W;
    const int tH = (oH + 1) / 2;
    const int tW = (oW + 1) / 2;
    const int num_tiles = tH * tW;

    // ── Step 1: Input transform ───────────────────────────────────
    // NHWC input: input[(ih*W + iw)*C_in + c]
    double t0 = now_ms();

    // Use pre-allocated scratch or fall back to local allocation
    std::vector<int8_t> _ih_local;
    int8_t* input_hat_base;
    if (scratch_input_hat) {
        input_hat_base = scratch_input_hat;
    } else {
        _ih_local.resize((size_t)16 * num_tiles * C_in);
        input_hat_base = _ih_local.data();
    }
    int8_t* input_hat[16];
    for (int p = 0; p < 16; ++p)
        input_hat[p] = input_hat_base + (size_t)p * num_tiles * C_in;

    // Classify tiles as interior (no boundary checks) or border.
    // Interior condition: ih0 >= 0 AND ih0+3 < H  →  1 <= th < (H-1)/2
    // (H-1)/2 works for both even and odd H.
    const int th_inner_start = 1;
    const int th_inner_end   = (H - 1) / 2;
    const int tw_inner_start = 1;
    const int tw_inner_end   = (W - 1) / 2;

#ifdef __ARM_NEON
// ── Fused tile-parallel Winograd (all 3 phases, single OMP region) ────────
// Motivation: pos-parallel GEMM distributes output_hat rows across cores'
// L2 caches. The following tile-parallel output transform then suffers
// cross-core cache traffic, causing a regression at 4T vs 1T.
// Fix: each thread owns tiles [tile_s, tile_e) and runs all 3 phases
// sequentially so input_hat/output_hat stay in that thread's L2 cache.
#ifdef _OPENMP
    if (omp_get_max_threads() > 1) {
        // Set up output_hat scratch (mirror of the setup in Phase 2 below)
        std::vector<int32_t> _oh_fused;
        int32_t* output_hat_base_f;
        if (scratch_output_hat) {
            output_hat_base_f = scratch_output_hat;
        } else {
            _oh_fused.resize((size_t)16 * num_tiles * C_out);
            output_hat_base_f = _oh_fused.data();
        }
        int32_t* output_hat_f[16];
        for (int p = 0; p < 16; ++p)
            output_hat_f[p] = output_hat_base_f + (size_t)p * num_tiles * C_out;

        // Pre-compute Winograd combined-scale factors (used in Phase 3)
        int32_t cs_mult_f[16], cs_exp_f[16];
        for (int p = 0; p < 16; ++p) {
            float cs = wino_weight_scale[p] / wino_input_scale;
            QuantizeMultiplier(cs, &cs_mult_f[p], &cs_exp_f[p]);
        }

        const double t_fused = now_ms();

        #pragma omp parallel
        {
            const int tid  = omp_get_thread_num();
            const int nt   = omp_get_num_threads();
            const int tile_s = (int)((int64_t)num_tiles * tid / nt);
            const int tile_e = (int)((int64_t)num_tiles * (tid + 1) / nt);

            // ── Phase 1: input transform for tiles [tile_s, tile_e) ──────
            for (int tile_idx = tile_s; tile_idx < tile_e; ++tile_idx) {
                const int th2 = tile_idx / tW;
                const int tw2 = tile_idx % tW;
                const int ih0 = th2 * 2 - 1;
                const int iw0 = tw2 * 2 - 1;
                const bool interior = (th2 >= th_inner_start && th2 < th_inner_end &&
                                       tw2 >= tw_inner_start && tw2 < tw_inner_end);
                if (interior) {
                    int c = 0;
                    for (; c + 7 < C_in; c += 8) {
                        int16x8_t d00,d01,d02,d03, d10,d11,d12,d13,
                                  d20,d21,d22,d23, d30,d31,d32,d33;
                        {
                            const int8_t* r0 = input + ((ih0+0)*W + iw0)*C_in + c;
                            const int8_t* r1 = input + ((ih0+1)*W + iw0)*C_in + c;
                            const int8_t* r2 = input + ((ih0+2)*W + iw0)*C_in + c;
                            const int8_t* r3 = input + ((ih0+3)*W + iw0)*C_in + c;
                            const int step = C_in;
                            d00=vmovl_s8(vld1_s8(r0));         d01=vmovl_s8(vld1_s8(r0+step));
                            d02=vmovl_s8(vld1_s8(r0+2*step));  d03=vmovl_s8(vld1_s8(r0+3*step));
                            d10=vmovl_s8(vld1_s8(r1));         d11=vmovl_s8(vld1_s8(r1+step));
                            d12=vmovl_s8(vld1_s8(r1+2*step));  d13=vmovl_s8(vld1_s8(r1+3*step));
                            d20=vmovl_s8(vld1_s8(r2));         d21=vmovl_s8(vld1_s8(r2+step));
                            d22=vmovl_s8(vld1_s8(r2+2*step));  d23=vmovl_s8(vld1_s8(r2+3*step));
                            d30=vmovl_s8(vld1_s8(r3));         d31=vmovl_s8(vld1_s8(r3+step));
                            d32=vmovl_s8(vld1_s8(r3+2*step));  d33=vmovl_s8(vld1_s8(r3+3*step));
                        }
                        int16x8_t t00=vsubq_s16(d00,d20),t01=vsubq_s16(d01,d21),t02=vsubq_s16(d02,d22),t03=vsubq_s16(d03,d23);
                        int16x8_t t10=vaddq_s16(d10,d20),t11=vaddq_s16(d11,d21),t12=vaddq_s16(d12,d22),t13=vaddq_s16(d13,d23);
                        int16x8_t t20=vsubq_s16(d20,d10),t21=vsubq_s16(d21,d11),t22=vsubq_s16(d22,d12),t23=vsubq_s16(d23,d13);
                        int16x8_t t30=vsubq_s16(d10,d30),t31=vsubq_s16(d11,d31),t32=vsubq_s16(d12,d32),t33=vsubq_s16(d13,d33);
                        int16x8_t v00=vsubq_s16(t00,t02),v01=vaddq_s16(t01,t02),v02=vsubq_s16(t02,t01),v03=vsubq_s16(t01,t03);
                        int16x8_t v10=vsubq_s16(t10,t12),v11=vaddq_s16(t11,t12),v12=vsubq_s16(t12,t11),v13=vsubq_s16(t11,t13);
                        int16x8_t v20=vsubq_s16(t20,t22),v21=vaddq_s16(t21,t22),v22=vsubq_s16(t22,t21),v23=vsubq_s16(t21,t23);
                        int16x8_t v30=vsubq_s16(t30,t32),v31=vaddq_s16(t31,t32),v32=vsubq_s16(t32,t31),v33=vsubq_s16(t31,t33);
                        int8x8_t o00=vqrshrn_n_s16(v00,2),o01=vqrshrn_n_s16(v01,2),o02=vqrshrn_n_s16(v02,2),o03=vqrshrn_n_s16(v03,2);
                        int8x8_t o10=vqrshrn_n_s16(v10,2),o11=vqrshrn_n_s16(v11,2),o12=vqrshrn_n_s16(v12,2),o13=vqrshrn_n_s16(v13,2);
                        int8x8_t o20=vqrshrn_n_s16(v20,2),o21=vqrshrn_n_s16(v21,2),o22=vqrshrn_n_s16(v22,2),o23=vqrshrn_n_s16(v23,2);
                        int8x8_t o30=vqrshrn_n_s16(v30,2),o31=vqrshrn_n_s16(v31,2),o32=vqrshrn_n_s16(v32,2),o33=vqrshrn_n_s16(v33,2);
                        const size_t off = (size_t)tile_idx * C_in + c;
                        vst1_s8(input_hat[ 0]+off,o00); vst1_s8(input_hat[ 1]+off,o01);
                        vst1_s8(input_hat[ 2]+off,o02); vst1_s8(input_hat[ 3]+off,o03);
                        vst1_s8(input_hat[ 4]+off,o10); vst1_s8(input_hat[ 5]+off,o11);
                        vst1_s8(input_hat[ 6]+off,o12); vst1_s8(input_hat[ 7]+off,o13);
                        vst1_s8(input_hat[ 8]+off,o20); vst1_s8(input_hat[ 9]+off,o21);
                        vst1_s8(input_hat[10]+off,o22); vst1_s8(input_hat[11]+off,o23);
                        vst1_s8(input_hat[12]+off,o30); vst1_s8(input_hat[13]+off,o31);
                        vst1_s8(input_hat[14]+off,o32); vst1_s8(input_hat[15]+off,o33);
                    }
                    for (; c < C_in; ++c) {
                        int16_t d[4][4];
                        for (int i=0;i<4;++i) for (int j=0;j<4;++j)
                            d[i][j]=(int16_t)input[((ih0+i)*W+(iw0+j))*C_in+c];
                        int16_t tmp[4][4];
                        for (int j=0;j<4;++j){tmp[0][j]=d[0][j]-d[2][j];tmp[1][j]=d[1][j]+d[2][j];tmp[2][j]=-d[1][j]+d[2][j];tmp[3][j]=d[1][j]-d[3][j];}
                        int16_t v[4][4];
                        for (int i=0;i<4;++i){v[i][0]=tmp[i][0]-tmp[i][2];v[i][1]=tmp[i][1]+tmp[i][2];v[i][2]=-tmp[i][1]+tmp[i][2];v[i][3]=tmp[i][1]-tmp[i][3];}
                        for (int i=0;i<4;++i) for (int j=0;j<4;++j){
                            int16_t s=(int16_t)((v[i][j]+2)>>2);
                            if(s>127)s=127; if(s<-128)s=-128;
                            input_hat[i*4+j][tile_idx*C_in+c]=(int8_t)s;
                        }
                    }
                } else {
                    for (int c=0;c<C_in;++c){
                        int16_t d[4][4];
                        for (int i=0;i<4;++i){ int ih=ih0+i;
                            for (int j=0;j<4;++j){ int iw=iw0+j;
                                d[i][j]=(ih>=0&&ih<H&&iw>=0&&iw<W)?(int16_t)input[(ih*W+iw)*C_in+c]:(int16_t)in_zp;
                            }
                        }
                        int16_t tmp[4][4];
                        for (int j=0;j<4;++j){tmp[0][j]=d[0][j]-d[2][j];tmp[1][j]=d[1][j]+d[2][j];tmp[2][j]=-d[1][j]+d[2][j];tmp[3][j]=d[1][j]-d[3][j];}
                        int16_t v[4][4];
                        for (int i=0;i<4;++i){v[i][0]=tmp[i][0]-tmp[i][2];v[i][1]=tmp[i][1]+tmp[i][2];v[i][2]=-tmp[i][1]+tmp[i][2];v[i][3]=tmp[i][1]-tmp[i][3];}
                        for (int i=0;i<4;++i) for (int j=0;j<4;++j){
                            int16_t s=(int16_t)((v[i][j]+2)>>2);
                            if(s>127)s=127; if(s<-128)s=-128;
                            input_hat[i*4+j][tile_idx*C_in+c]=(int8_t)s;
                        }
                    }
                }
            } // end Phase 1 tile loop

            // ── Phase 2: 16 serial mini-GEMMs for tiles [tile_s, tile_e) ─
            {
                const int T = tile_e - tile_s;
                if (T > 0) {
                    for (int pos = 0; pos < 16; ++pos) {
                        gemm_int8_int32(
                            input_hat[pos]    + (size_t)tile_s * C_in,
                            w_wino_packed[pos],
                            output_hat_f[pos] + (size_t)tile_s * C_out,
                            T, C_in, C_out,
                            w_wino_row_sums ? w_wino_row_sums[pos] : nullptr, nullptr);
                    }
                }
            }

            // ── Phase 3: inverse transform + requantize for [tile_s, tile_e) ─
            {
                const int C_out_al = (C_out / 4) * 4;
                const int32x4_t vzp_f = vdupq_n_s32((int32_t)out_zp);
                for (int tile_idx = tile_s; tile_idx < tile_e; ++tile_idx) {
                    const int th2 = tile_idx / tW;
                    const int tw2 = tile_idx % tW;
                    for (int n = 0; n < C_out_al; n += 4) {
                        int32_t bias4[4] = {(int32_t)eff_bias[n],(int32_t)eff_bias[n+1],
                                            (int32_t)eff_bias[n+2],(int32_t)eff_bias[n+3]};
                        const int32x4_t vbias4 = vld1q_s32(bias4);
                        const int32x4_t vqm = vld1q_s32(req_mult + n);
                        const int32x4_t vqe = vld1q_s32(req_exp  + n);
                        int32x4_t m00,m01,m02,m03, m10,m11,m12,m13,
                                  m20,m21,m22,m23, m30,m31,m32,m33;
                        #define LSP(mi,mj,pos) { \
                            int32x4_t raw=vld1q_s32(output_hat_f[pos]+(size_t)tile_idx*C_out+n); \
                            int32x4_t sc=vqrdmulhq_s32(raw,vdupq_n_s32(cs_mult_f[pos])); \
                            m##mi##mj=vrshlq_s32(sc,vdupq_n_s32(cs_exp_f[pos])); }
                        LSP(0,0, 0) LSP(0,1, 1) LSP(0,2, 2) LSP(0,3, 3)
                        LSP(1,0, 4) LSP(1,1, 5) LSP(1,2, 6) LSP(1,3, 7)
                        LSP(2,0, 8) LSP(2,1, 9) LSP(2,2,10) LSP(2,3,11)
                        LSP(3,0,12) LSP(3,1,13) LSP(3,2,14) LSP(3,3,15)
                        #undef LSP
                        int32x4_t tmp00=vaddq_s32(vaddq_s32(m00,m10),m20);
                        int32x4_t tmp01=vaddq_s32(vaddq_s32(m01,m11),m21);
                        int32x4_t tmp02=vaddq_s32(vaddq_s32(m02,m12),m22);
                        int32x4_t tmp03=vaddq_s32(vaddq_s32(m03,m13),m23);
                        int32x4_t tmp10=vsubq_s32(vsubq_s32(m10,m20),m30);
                        int32x4_t tmp11=vsubq_s32(vsubq_s32(m11,m21),m31);
                        int32x4_t tmp12=vsubq_s32(vsubq_s32(m12,m22),m32);
                        int32x4_t tmp13=vsubq_s32(vsubq_s32(m13,m23),m33);
                        int32x4_t out00=vaddq_s32(vaddq_s32(tmp00,tmp01),tmp02);
                        int32x4_t out01=vsubq_s32(vsubq_s32(tmp01,tmp02),tmp03);
                        int32x4_t out10=vaddq_s32(vaddq_s32(tmp10,tmp11),tmp12);
                        int32x4_t out11=vsubq_s32(vsubq_s32(tmp11,tmp12),tmp13);
                        auto rq4 = [&](int32x4_t acc) -> int32x4_t {
                            int32x4_t s=vaddq_s32(acc,vbias4);
                            return vaddq_s32(vrshlq_s32(vqrdmulhq_s32(s,vqm),vqe),vzp_f);
                        };
                        int32x4_t rq00=rq4(out00),rq01=rq4(out01),rq10=rq4(out10),rq11=rq4(out11);
                        int16x4_t h00=vqmovn_s32(rq00),h01=vqmovn_s32(rq01);
                        int16x4_t h10=vqmovn_s32(rq10),h11=vqmovn_s32(rq11);
                        int8x8_t b00=vqmovn_s16(vcombine_s16(h00,h00));
                        int8x8_t b01=vqmovn_s16(vcombine_s16(h01,h01));
                        int8x8_t b10=vqmovn_s16(vcombine_s16(h10,h10));
                        int8x8_t b11=vqmovn_s16(vcombine_s16(h11,h11));
                        int8_t r00v[4],r01v[4],r10v[4],r11v[4];
                        vst1_lane_s32((int32_t*)r00v,vreinterpret_s32_s8(b00),0);
                        vst1_lane_s32((int32_t*)r01v,vreinterpret_s32_s8(b01),0);
                        vst1_lane_s32((int32_t*)r10v,vreinterpret_s32_s8(b10),0);
                        vst1_lane_s32((int32_t*)r11v,vreinterpret_s32_s8(b11),0);
                        for (int di=0;di<2;++di){ int oh=th2*2+di; if(oh>=oH) continue;
                            for (int dj=0;dj<2;++dj){ int ow=tw2*2+dj; if(ow>=oW) continue;
                                const int8_t* src=(di==0)?((dj==0)?r00v:r01v):((dj==0)?r10v:r11v);
                                memcpy(output+(oh*oW+ow)*C_out+n, src, 4);
                            }
                        }
                    }
                    // Scalar tail (C_out not a multiple of 4)
                    for (int n = C_out_al; n < C_out; ++n) {
                        const int32_t bias=(int32_t)eff_bias[n];
                        int32_t mm[4][4];
                        for (int i=0;i<4;++i) for (int j=0;j<4;++j){
                            int pos=i*4+j;
                            mm[i][j]=apply_q31_scalar(output_hat_f[pos][tile_idx*C_out+n],cs_mult_f[pos],cs_exp_f[pos]);
                        }
                        int32_t tmp[2][4];
                        for (int j=0;j<4;++j){tmp[0][j]=mm[0][j]+mm[1][j]+mm[2][j];tmp[1][j]=mm[1][j]-mm[2][j]-mm[3][j];}
                        int32_t out2x2[2][2];
                        for (int i=0;i<2;++i){out2x2[i][0]=tmp[i][0]+tmp[i][1]+tmp[i][2];out2x2[i][1]=tmp[i][1]-tmp[i][2]-tmp[i][3];}
                        for (int di=0;di<2;++di){ int oh=th2*2+di; if(oh>=oH) continue;
                            for (int dj=0;dj<2;++dj){ int ow=tw2*2+dj; if(ow>=oW) continue;
                                output[(oh*oW+ow)*C_out+n]=requant_fixedpoint(out2x2[di][dj]+bias,req_mult[n],req_exp[n],out_zp);
                            }
                        }
                    }
                } // end Phase 3 tile loop
            }
        } // end omp parallel

        const double dt = now_ms() - t_fused;
        g_wino_transform_ms += dt * 0.15;
        g_wino_gemm_ms      += dt * 0.70;
        g_wino_output_ms    += dt * 0.15;
        return;
    } // end if (omp_get_max_threads() > 1)
#endif // _OPENMP

// 1-thread fallback: original 3-phase code below
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(omp_get_max_threads() > 1)
#endif
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const int th = tile_idx / tW;
        const int tw = tile_idx % tW;
        const int ih0 = th * 2 - 1;
        const int iw0 = tw * 2 - 1;

        const bool is_interior = (th >= th_inner_start && th < th_inner_end &&
                                   tw >= tw_inner_start && tw < tw_inner_end);

        if (is_interior) {
            // Interior tile: vectorized NEON path (no boundary checks needed)
            // NHWC: for 8 channels starting at c, we can load 8 consecutive bytes
            // from input[(ih*W + iw)*C_in + c] for each (ih, iw).

            int c = 0;
            for (; c + 7 < C_in; c += 8) {
                // Load 4x4 patch for 8 channels: d[i][j] as int16x8_t
                int16x8_t d00, d01, d02, d03;
                int16x8_t d10, d11, d12, d13;
                int16x8_t d20, d21, d22, d23;
                int16x8_t d30, d31, d32, d33;

                // NHWC: 8 consecutive channels at each spatial position
                // input + ((ih0+i)*W + (iw0+j)) * C_in + c
                {
                    const int8_t* r0 = input + ((ih0+0)*W + iw0) * C_in + c;
                    const int8_t* r1 = input + ((ih0+1)*W + iw0) * C_in + c;
                    const int8_t* r2 = input + ((ih0+2)*W + iw0) * C_in + c;
                    const int8_t* r3 = input + ((ih0+3)*W + iw0) * C_in + c;
                    const int step = C_in;  // stride between adjacent w positions
                    d00 = vmovl_s8(vld1_s8(r0));          d01 = vmovl_s8(vld1_s8(r0 + step));
                    d02 = vmovl_s8(vld1_s8(r0 + 2*step)); d03 = vmovl_s8(vld1_s8(r0 + 3*step));
                    d10 = vmovl_s8(vld1_s8(r1));          d11 = vmovl_s8(vld1_s8(r1 + step));
                    d12 = vmovl_s8(vld1_s8(r1 + 2*step)); d13 = vmovl_s8(vld1_s8(r1 + 3*step));
                    d20 = vmovl_s8(vld1_s8(r2));          d21 = vmovl_s8(vld1_s8(r2 + step));
                    d22 = vmovl_s8(vld1_s8(r2 + 2*step)); d23 = vmovl_s8(vld1_s8(r2 + 3*step));
                    d30 = vmovl_s8(vld1_s8(r3));          d31 = vmovl_s8(vld1_s8(r3 + step));
                    d32 = vmovl_s8(vld1_s8(r3 + 2*step)); d33 = vmovl_s8(vld1_s8(r3 + 3*step));
                }

                // Column transform: BT x d
                int16x8_t t00 = vsubq_s16(d00, d20), t01 = vsubq_s16(d01, d21), t02 = vsubq_s16(d02, d22), t03 = vsubq_s16(d03, d23);
                int16x8_t t10 = vaddq_s16(d10, d20), t11 = vaddq_s16(d11, d21), t12 = vaddq_s16(d12, d22), t13 = vaddq_s16(d13, d23);
                int16x8_t t20 = vsubq_s16(d20, d10), t21 = vsubq_s16(d21, d11), t22 = vsubq_s16(d22, d12), t23 = vsubq_s16(d23, d13);
                int16x8_t t30 = vsubq_s16(d10, d30), t31 = vsubq_s16(d11, d31), t32 = vsubq_s16(d12, d32), t33 = vsubq_s16(d13, d33);

                // Row transform: (BT x d) x BT^T
                int16x8_t v00 = vsubq_s16(t00, t02), v01 = vaddq_s16(t01, t02), v02 = vsubq_s16(t02, t01), v03 = vsubq_s16(t01, t03);
                int16x8_t v10 = vsubq_s16(t10, t12), v11 = vaddq_s16(t11, t12), v12 = vsubq_s16(t12, t11), v13 = vsubq_s16(t11, t13);
                int16x8_t v20 = vsubq_s16(t20, t22), v21 = vaddq_s16(t21, t22), v22 = vsubq_s16(t22, t21), v23 = vsubq_s16(t21, t23);
                int16x8_t v30 = vsubq_s16(t30, t32), v31 = vaddq_s16(t31, t32), v32 = vsubq_s16(t32, t31), v33 = vsubq_s16(t31, t33);

                // Scale by 1/4
                int8x8_t o00 = vqrshrn_n_s16(v00,2), o01 = vqrshrn_n_s16(v01,2), o02 = vqrshrn_n_s16(v02,2), o03 = vqrshrn_n_s16(v03,2);
                int8x8_t o10 = vqrshrn_n_s16(v10,2), o11 = vqrshrn_n_s16(v11,2), o12 = vqrshrn_n_s16(v12,2), o13 = vqrshrn_n_s16(v13,2);
                int8x8_t o20 = vqrshrn_n_s16(v20,2), o21 = vqrshrn_n_s16(v21,2), o22 = vqrshrn_n_s16(v22,2), o23 = vqrshrn_n_s16(v23,2);
                int8x8_t o30 = vqrshrn_n_s16(v30,2), o31 = vqrshrn_n_s16(v31,2), o32 = vqrshrn_n_s16(v32,2), o33 = vqrshrn_n_s16(v33,2);

                // Store to input_hat[pos][tile_idx * C_in + c]
                const size_t off = (size_t)tile_idx * C_in + c;
                vst1_s8(input_hat[ 0] + off, o00); vst1_s8(input_hat[ 1] + off, o01);
                vst1_s8(input_hat[ 2] + off, o02); vst1_s8(input_hat[ 3] + off, o03);
                vst1_s8(input_hat[ 4] + off, o10); vst1_s8(input_hat[ 5] + off, o11);
                vst1_s8(input_hat[ 6] + off, o12); vst1_s8(input_hat[ 7] + off, o13);
                vst1_s8(input_hat[ 8] + off, o20); vst1_s8(input_hat[ 9] + off, o21);
                vst1_s8(input_hat[10] + off, o22); vst1_s8(input_hat[11] + off, o23);
                vst1_s8(input_hat[12] + off, o30); vst1_s8(input_hat[13] + off, o31);
                vst1_s8(input_hat[14] + off, o32); vst1_s8(input_hat[15] + off, o33);
            }

            // Scalar tail for remaining channels
            for (; c < C_in; ++c) {
                int16_t d[4][4];
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        // NHWC: input[((ih0+i)*W + (iw0+j)) * C_in + c]
                        d[i][j] = (int16_t)input[((ih0+i)*W + (iw0+j)) * C_in + c];
                    }
                }
                int16_t tmp[4][4];
                for (int j = 0; j < 4; ++j) {
                    tmp[0][j] = d[0][j] - d[2][j];
                    tmp[1][j] = d[1][j] + d[2][j];
                    tmp[2][j] = -d[1][j] + d[2][j];
                    tmp[3][j] = d[1][j] - d[3][j];
                }
                int16_t v[4][4];
                for (int i = 0; i < 4; ++i) {
                    v[i][0] = tmp[i][0] - tmp[i][2];
                    v[i][1] = tmp[i][1] + tmp[i][2];
                    v[i][2] = -tmp[i][1] + tmp[i][2];
                    v[i][3] = tmp[i][1] - tmp[i][3];
                }
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j) {
                        int16_t val = v[i][j];
                        int16_t shifted = (val + 2) >> 2;
                        if (shifted > 127) shifted = 127;
                        if (shifted < -128) shifted = -128;
                        input_hat[i * 4 + j][tile_idx * C_in + c] = (int8_t)shifted;
                    }
            }
        } else {
            // Border tile: scalar path with boundary checks (NHWC)
            for (int c = 0; c < C_in; ++c) {
                int16_t d[4][4];
                for (int i = 0; i < 4; ++i) {
                    int ih = ih0 + i;
                    for (int j = 0; j < 4; ++j) {
                        int iw = iw0 + j;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            d[i][j] = (int16_t)input[(ih * W + iw) * C_in + c];
                        else
                            d[i][j] = (int16_t)in_zp;
                    }
                }

                int16_t tmp[4][4];
                for (int j = 0; j < 4; ++j) {
                    tmp[0][j] = d[0][j] - d[2][j];
                    tmp[1][j] = d[1][j] + d[2][j];
                    tmp[2][j] = -d[1][j] + d[2][j];
                    tmp[3][j] = d[1][j] - d[3][j];
                }
                int16_t v[4][4];
                for (int i = 0; i < 4; ++i) {
                    v[i][0] = tmp[i][0] - tmp[i][2];
                    v[i][1] = tmp[i][1] + tmp[i][2];
                    v[i][2] = -tmp[i][1] + tmp[i][2];
                    v[i][3] = tmp[i][1] - tmp[i][3];
                }
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j) {
                        int16_t val = v[i][j];
                        int16_t shifted = (val + 2) >> 2;
                        if (shifted > 127) shifted = 127;
                        if (shifted < -128) shifted = -128;
                        input_hat[i * 4 + j][tile_idx * C_in + c] = (int8_t)shifted;
                    }
            }
        }
    }
#else
    // AVX-512BW: process 32 channels per iteration
#ifdef __AVX512BW__
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(omp_get_max_threads() > 1)
#endif
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const int th = tile_idx / tW, tw = tile_idx % tW;
        const int ih0 = th * 2 - 1, iw0 = tw * 2 - 1;
        const bool interior = (ih0 >= 0 && ih0 + 3 < H && iw0 >= 0 && iw0 + 3 < W);

        // Set up per-position source pointers.
        // Interior: point directly into `input`; no copy needed.
        // Border: build a padded 4×4×C_in scratch buffer (max C_in=512 → 8 KB stack)
        // then point src_ij into that buffer.  Both paths then share one vectorized loop.
        const int8_t* src_ij[4][4];
        if (interior) {
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    src_ij[i][j] = input + ((ih0+i)*W + (iw0+j)) * C_in;
        } else {
            // Padded scratch: fill with in_zp, then overwrite valid positions
            alignas(32) int8_t pad[16 * 512];  // 8 KB; fine on the stack
            std::memset(pad, (unsigned char)(uint8_t)in_zp, 16 * C_in);
            for (int i = 0; i < 4; ++i) {
                const int ih = ih0 + i;
                for (int j = 0; j < 4; ++j) {
                    const int iw = iw0 + j;
                    int8_t* dst = pad + (i*4+j) * C_in;
                    src_ij[i][j] = dst;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        std::memcpy(dst, input + (size_t)(ih*W+iw)*C_in, C_in);
                }
            }
        }

        int c = 0;
        {
            // Unified vectorized path for both interior and border tiles
            const __m512i vrnd = _mm512_set1_epi16(2);
            for (; c + 31 < C_in; c += 32) {
                __m512i d[4][4];
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        d[i][j] = _mm512_cvtepi8_epi16(_mm256_loadu_si256(
                            (const __m256i*)(src_ij[i][j] + c)));

                // Row transform: B^T applied along rows (i dimension)
                __m512i tmp[4][4];
                for (int j = 0; j < 4; ++j) {
                    tmp[0][j] = _mm512_sub_epi16(d[0][j], d[2][j]);
                    tmp[1][j] = _mm512_add_epi16(d[1][j], d[2][j]);
                    tmp[2][j] = _mm512_sub_epi16(d[2][j], d[1][j]);
                    tmp[3][j] = _mm512_sub_epi16(d[1][j], d[3][j]);
                }

                // Col transform + shift + store
                for (int i = 0; i < 4; ++i) {
                    __m512i v0 = _mm512_sub_epi16(tmp[i][0], tmp[i][2]);
                    __m512i v1 = _mm512_add_epi16(tmp[i][1], tmp[i][2]);
                    __m512i v2 = _mm512_sub_epi16(tmp[i][2], tmp[i][1]);
                    __m512i v3 = _mm512_sub_epi16(tmp[i][1], tmp[i][3]);
                    __m256i o0 = _mm512_cvtsepi16_epi8(_mm512_srai_epi16(_mm512_add_epi16(v0, vrnd), 2));
                    __m256i o1 = _mm512_cvtsepi16_epi8(_mm512_srai_epi16(_mm512_add_epi16(v1, vrnd), 2));
                    __m256i o2 = _mm512_cvtsepi16_epi8(_mm512_srai_epi16(_mm512_add_epi16(v2, vrnd), 2));
                    __m256i o3 = _mm512_cvtsepi16_epi8(_mm512_srai_epi16(_mm512_add_epi16(v3, vrnd), 2));
                    _mm256_storeu_si256((__m256i*)(input_hat[i*4+0] + (size_t)tile_idx * C_in + c), o0);
                    _mm256_storeu_si256((__m256i*)(input_hat[i*4+1] + (size_t)tile_idx * C_in + c), o1);
                    _mm256_storeu_si256((__m256i*)(input_hat[i*4+2] + (size_t)tile_idx * C_in + c), o2);
                    _mm256_storeu_si256((__m256i*)(input_hat[i*4+3] + (size_t)tile_idx * C_in + c), o3);
                }
            }
        }
        // Scalar tail: only runs when C_in is not a multiple of 32.
        // For border tiles with C_in ≤ 512 the vectorized path above handles all
        // channels, so this loop is empty in the common case.
        for (int ch = c; ch < C_in; ++ch) {
            int16_t d[4][4];
            for (int i = 0; i < 4; ++i) {
                int ih = ih0 + i;
                for (int j = 0; j < 4; ++j) {
                    int iw = iw0 + j;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        d[i][j] = (int16_t)input[(ih * W + iw) * C_in + ch];
                    else
                        d[i][j] = (int16_t)in_zp;
                }
            }
            int16_t tmp[4][4], v[4][4];
            for (int j = 0; j < 4; ++j) {
                tmp[0][j]=d[0][j]-d[2][j]; tmp[1][j]=d[1][j]+d[2][j];
                tmp[2][j]=-d[1][j]+d[2][j]; tmp[3][j]=d[1][j]-d[3][j];
            }
            for (int i = 0; i < 4; ++i) {
                v[i][0]=tmp[i][0]-tmp[i][2]; v[i][1]=tmp[i][1]+tmp[i][2];
                v[i][2]=-tmp[i][1]+tmp[i][2]; v[i][3]=tmp[i][1]-tmp[i][3];
            }
            for (int i=0;i<4;++i) for (int j=0;j<4;++j) {
                int16_t s=(v[i][j]+2)>>2;
                if(s>127)s=127; if(s<-128)s=-128;
                input_hat[i*4+j][(size_t)tile_idx*C_in+ch]=(int8_t)s;
            }
        }
    }
#else
    // AVX2 fallback or scalar
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(omp_get_max_threads() > 1)
#endif
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const int th = tile_idx / tW, tw = tile_idx % tW;
        const int ih0 = th * 2 - 1;
        const int iw0 = tw * 2 - 1;
        for (int c = 0; c < C_in; ++c) {
            int16_t d[4][4];
            for (int i = 0; i < 4; ++i) {
                int ih = ih0 + i;
                for (int j = 0; j < 4; ++j) {
                    int iw = iw0 + j;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        d[i][j] = (int16_t)input[(ih * W + iw) * C_in + c];
                    else
                        d[i][j] = (int16_t)in_zp;
                }
            }
            int16_t tmp[4][4];
            for (int j = 0; j < 4; ++j) {
                tmp[0][j]=d[0][j]-d[2][j]; tmp[1][j]=d[1][j]+d[2][j];
                tmp[2][j]=-d[1][j]+d[2][j]; tmp[3][j]=d[1][j]-d[3][j];
            }
            int16_t v[4][4];
            for (int i = 0; i < 4; ++i) {
                v[i][0]=tmp[i][0]-tmp[i][2]; v[i][1]=tmp[i][1]+tmp[i][2];
                v[i][2]=-tmp[i][1]+tmp[i][2]; v[i][3]=tmp[i][1]-tmp[i][3];
            }
            for (int i=0;i<4;++i) for (int j=0;j<4;++j) {
                int16_t s=(v[i][j]+2)>>2;
                if(s>127)s=127; if(s<-128)s=-128;
                input_hat[i*4+j][(size_t)tile_idx*C_in+c]=(int8_t)s;
            }
        }
    }
#endif
#endif

    g_wino_transform_ms += now_ms() - t0;

    // ── Step 2: 16 domain GEMMs ──────────────────────────────────
    t0 = now_ms();

    std::vector<int32_t> _oh_local;
    int32_t* output_hat_base;
    if (scratch_output_hat) {
        output_hat_base = scratch_output_hat;
    } else {
        _oh_local.resize((size_t)16 * num_tiles * C_out);
        output_hat_base = _oh_local.data();
    }
    int32_t* output_hat[16];
    for (int p = 0; p < 16; ++p)
        output_hat[p] = output_hat_base + (size_t)p * num_tiles * C_out;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(omp_get_max_threads() > 1)
#endif
    for (int pos = 0; pos < 16; ++pos) {
        gemm_int8_int32(
            input_hat[pos],
            w_wino_packed[pos],
            output_hat[pos],
            num_tiles, C_in, C_out,
            w_wino_row_sums ? w_wino_row_sums[pos] : nullptr, nullptr);
    }

    g_wino_gemm_ms += now_ms() - t0;

    // ── Step 3: Inverse transform (INT32 NEON) + requantization ──
    // Output is NHWC: output[(oh*oW + ow)*C_out + n]
    t0 = now_ms();

    // Pre-compute Q31 fixed-point combined_scale for each position.
    int32_t cs_mult[16], cs_exp[16];
    for (int p = 0; p < 16; ++p) {
        float cs = wino_weight_scale[p] / wino_input_scale;
        QuantizeMultiplier(cs, &cs_mult[p], &cs_exp[p]);
    }

#ifdef __ARM_NEON
    // Process 4 output channels at once using NEON int32x4_t.
    // Parallelise over tile_idx (spatial tiles) so threads write to different
    // output rows → no false sharing (vs. channel-parallel which causes 8 threads
    // to write different 4-byte chunks of the same 64-byte output cache line).
    const int oHW = oH * oW;
    const int C_out_aligned = (C_out / 4) * 4;
    const int32x4_t vzp = vdupq_n_s32((int32_t)out_zp);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(omp_get_max_threads() > 1)
#endif
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const int th = tile_idx / tW;
        const int tw = tile_idx % tW;

        for (int n = 0; n < C_out_aligned; n += 4) {
            int32_t bias4[4] = { (int32_t)eff_bias[n], (int32_t)eff_bias[n+1],
                                 (int32_t)eff_bias[n+2], (int32_t)eff_bias[n+3] };
            const int32x4_t vbias4 = vld1q_s32(bias4);
            const int32x4_t vqm = vld1q_s32(req_mult + n);
            const int32x4_t vqe = vld1q_s32(req_exp + n);

            int32x4_t m00, m01, m02, m03;
            int32x4_t m10, m11, m12, m13;
            int32x4_t m20, m21, m22, m23;
            int32x4_t m30, m31, m32, m33;

            #define LOAD_SCALE_POS(mi, mj, pos) { \
                int32x4_t raw = vld1q_s32(output_hat[pos] + (size_t)tile_idx * C_out + n); \
                int32x4_t scaled = vqrdmulhq_s32(raw, vdupq_n_s32(cs_mult[pos])); \
                m##mi##mj = vrshlq_s32(scaled, vdupq_n_s32(cs_exp[pos])); \
            }
            LOAD_SCALE_POS(0, 0, 0);  LOAD_SCALE_POS(0, 1, 1);
            LOAD_SCALE_POS(0, 2, 2);  LOAD_SCALE_POS(0, 3, 3);
            LOAD_SCALE_POS(1, 0, 4);  LOAD_SCALE_POS(1, 1, 5);
            LOAD_SCALE_POS(1, 2, 6);  LOAD_SCALE_POS(1, 3, 7);
            LOAD_SCALE_POS(2, 0, 8);  LOAD_SCALE_POS(2, 1, 9);
            LOAD_SCALE_POS(2, 2, 10); LOAD_SCALE_POS(2, 3, 11);
            LOAD_SCALE_POS(3, 0, 12); LOAD_SCALE_POS(3, 1, 13);
            LOAD_SCALE_POS(3, 2, 14); LOAD_SCALE_POS(3, 3, 15);
            #undef LOAD_SCALE_POS

            // Inverse column transform: AT x m
            int32x4_t tmp00 = vaddq_s32(vaddq_s32(m00, m10), m20);
            int32x4_t tmp01 = vaddq_s32(vaddq_s32(m01, m11), m21);
            int32x4_t tmp02 = vaddq_s32(vaddq_s32(m02, m12), m22);
            int32x4_t tmp03 = vaddq_s32(vaddq_s32(m03, m13), m23);
            int32x4_t tmp10 = vsubq_s32(vsubq_s32(m10, m20), m30);
            int32x4_t tmp11 = vsubq_s32(vsubq_s32(m11, m21), m31);
            int32x4_t tmp12 = vsubq_s32(vsubq_s32(m12, m22), m32);
            int32x4_t tmp13 = vsubq_s32(vsubq_s32(m13, m23), m33);

            // Inverse row transform
            int32x4_t out00 = vaddq_s32(vaddq_s32(tmp00, tmp01), tmp02);
            int32x4_t out01 = vsubq_s32(vsubq_s32(tmp01, tmp02), tmp03);
            int32x4_t out10 = vaddq_s32(vaddq_s32(tmp10, tmp11), tmp12);
            int32x4_t out11 = vsubq_s32(vsubq_s32(tmp11, tmp12), tmp13);

            // Requantize
            auto requant4 = [&](int32x4_t acc) -> int32x4_t {
                int32x4_t s = vaddq_s32(acc, vbias4);
                int32x4_t mq = vqrdmulhq_s32(s, vqm);
                return vaddq_s32(vrshlq_s32(mq, vqe), vzp);
            };

            int32x4_t rq00 = requant4(out00);
            int32x4_t rq01 = requant4(out01);
            int32x4_t rq10 = requant4(out10);
            int32x4_t rq11 = requant4(out11);

            // Clamp and extract
            int16x4_t h00 = vqmovn_s32(rq00), h01 = vqmovn_s32(rq01);
            int16x4_t h10 = vqmovn_s32(rq10), h11 = vqmovn_s32(rq11);
            int8x8_t b00 = vqmovn_s16(vcombine_s16(h00, h00));
            int8x8_t b01 = vqmovn_s16(vcombine_s16(h01, h01));
            int8x8_t b10 = vqmovn_s16(vcombine_s16(h10, h10));
            int8x8_t b11 = vqmovn_s16(vcombine_s16(h11, h11));

            int8_t r00_v[4], r01_v[4], r10_v[4], r11_v[4];
            vst1_lane_s32((int32_t*)r00_v, vreinterpret_s32_s8(b00), 0);
            vst1_lane_s32((int32_t*)r01_v, vreinterpret_s32_s8(b01), 0);
            vst1_lane_s32((int32_t*)r10_v, vreinterpret_s32_s8(b10), 0);
            vst1_lane_s32((int32_t*)r11_v, vreinterpret_s32_s8(b11), 0);

            // Write to NHWC output: output[(oh*oW + ow)*C_out + n]
            for (int di = 0; di < 2; ++di) {
                int oh = th * 2 + di;
                if (oh >= oH) continue;
                for (int dj = 0; dj < 2; ++dj) {
                    int ow = tw * 2 + dj;
                    if (ow >= oW) continue;
                    const int8_t* src = (di == 0) ? ((dj == 0) ? r00_v : r01_v)
                                                  : ((dj == 0) ? r10_v : r11_v);
                    // NHWC: write 4 bytes at output[(oh*oW + ow)*C_out + n]
                    memcpy(output + (oh * oW + ow) * C_out + n, src, 4);
                }
            }
        }
    }

    // Scalar tail for remaining channels (at most 3)
    for (int n = C_out_aligned; n < C_out; ++n) {
        const int32_t bias = (int32_t)eff_bias[n];
        const int32_t qm = req_mult[n];
        const int32_t qe = req_exp[n];

        for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            const int th = tile_idx / tW;
            const int tw = tile_idx % tW;

            int32_t mm[4][4];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) {
                    int pos = i * 4 + j;
                    int32_t raw = output_hat[pos][tile_idx * C_out + n];
                    mm[i][j] = apply_q31_scalar(raw, cs_mult[pos], cs_exp[pos]);
                }

            int32_t tmp[2][4];
            for (int j = 0; j < 4; ++j) {
                tmp[0][j] = mm[0][j] + mm[1][j] + mm[2][j];
                tmp[1][j] = mm[1][j] - mm[2][j] - mm[3][j];
            }

            int32_t out2x2[2][2];
            for (int i = 0; i < 2; ++i) {
                out2x2[i][0] = tmp[i][0] + tmp[i][1] + tmp[i][2];
                out2x2[i][1] = tmp[i][1] - tmp[i][2] - tmp[i][3];
            }

            for (int di = 0; di < 2; ++di) {
                int oh = th * 2 + di;
                if (oh >= oH) continue;
                for (int dj = 0; dj < 2; ++dj) {
                    int ow = tw * 2 + dj;
                    if (ow >= oW) continue;
                    int32_t total = out2x2[di][dj] + bias;
                    // NHWC output
                    output[(oh * oW + ow) * C_out + n] =
                        requant_fixedpoint(total, qm, qe, out_zp);
                }
            }
        }
    }
#elif defined(__AVX512BW__)
{
    // AVX-512BW: process 16 output channels at once.
    // Parallelise over tile_idx for better utilisation when C_out < 16*num_threads
    // (e.g. C_out=64 gives only 4 work-units with the old n-major scheme).
    const int C_out16 = (C_out / 16) * 16;
    // Pre-compute aligned bias/mult/exp for all channel blocks once (read-only inside parallel loop)
    alignas(64) int32_t bias_all[2048]={}, mult_all[2048]={}, nexp_all[2048]={};
    for (int n = 0; n < C_out16; ++n) {
        bias_all[n] = (int32_t)eff_bias[n];
        mult_all[n] = req_mult[n];
        nexp_all[n] = -req_exp[n];
    }
    const __m512i vzp = _mm512_set1_epi32((int32_t)out_zp);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(omp_get_max_threads() > 1)
#endif
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const int th = tile_idx / tW, tw = tile_idx % tW;
        for (int n = 0; n < C_out16; n += 16) {
            __m512i m[4][4];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) {
                    const int pos = i * 4 + j;
                    __m512i raw = _mm512_loadu_si512(
                        output_hat[pos] + (size_t)tile_idx * C_out + n);
                    __m512i s = qrdmulh_epi32_512(raw, _mm512_set1_epi32(cs_mult[pos]));
                    if (cs_exp[pos] < 0)
                        m[i][j] = _mm512_srav_epi32(s, _mm512_set1_epi32(-cs_exp[pos]));
                    else if (cs_exp[pos] > 0)
                        m[i][j] = _mm512_sllv_epi32(s, _mm512_set1_epi32(cs_exp[pos]));
                    else
                        m[i][j] = s;
                }
            __m512i tmp[2][4];
            for (int j = 0; j < 4; ++j) {
                tmp[0][j] = _mm512_add_epi32(_mm512_add_epi32(m[0][j], m[1][j]), m[2][j]);
                tmp[1][j] = _mm512_sub_epi32(_mm512_sub_epi32(m[1][j], m[2][j]), m[3][j]);
            }
            __m512i out2x2[2][2];
            for (int i = 0; i < 2; ++i) {
                out2x2[i][0] = _mm512_add_epi32(_mm512_add_epi32(tmp[i][0], tmp[i][1]), tmp[i][2]);
                out2x2[i][1] = _mm512_sub_epi32(_mm512_sub_epi32(tmp[i][1], tmp[i][2]), tmp[i][3]);
            }
            const __m512i vbias = _mm512_load_si512(bias_all + n);
            const __m512i vqm   = _mm512_load_si512(mult_all + n);
            const __m512i vqe   = _mm512_load_si512(nexp_all + n);
            for (int di = 0; di < 2; ++di) {
                int oh = th * 2 + di; if (oh >= oH) continue;
                for (int dj = 0; dj < 2; ++dj) {
                    int ow = tw * 2 + dj; if (ow >= oW) continue;
                    __m512i acc = _mm512_add_epi32(out2x2[di][dj], vbias);
                    __m512i qm2 = qrdmulh_epi32_512(acc, vqm);
                    __m512i q   = _mm512_add_epi32(_mm512_srav_epi32(qm2, vqe), vzp);
                    __m128i out8 = _mm512_cvtsepi32_epi8(q);
                    memcpy(output + (oh * oW + ow) * C_out + n, &out8, 16);
                }
            }
        }
        // Scalar tail for remaining < 16 channels
        for (int n = C_out16; n < C_out; ++n) {
            const int32_t bias = (int32_t)eff_bias[n];
            const int32_t qm   = req_mult[n];
            const int32_t qe   = req_exp[n];
            int32_t mm[4][4];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) {
                    int pos = i*4+j;
                    mm[i][j] = apply_q31_scalar(output_hat[pos][tile_idx*C_out+n], cs_mult[pos], cs_exp[pos]);
                }
            int32_t tmps[2][4];
            for (int j = 0; j < 4; ++j) {
                tmps[0][j] = mm[0][j] + mm[1][j] + mm[2][j];
                tmps[1][j] = mm[1][j] - mm[2][j] - mm[3][j];
            }
            for (int di = 0; di < 2; ++di) {
                int oh = th*2+di; if (oh >= oH) continue;
                for (int dj = 0; dj < 2; ++dj) {
                    int ow = tw*2+dj; if (ow >= oW) continue;
                    int32_t r = (dj == 0) ? (tmps[di][0]+tmps[di][1]+tmps[di][2])
                                          : (tmps[di][1]-tmps[di][2]-tmps[di][3]);
                    output[(oh*oW+ow)*C_out+n] = requant_fixedpoint(r+bias, qm, qe, out_zp);
                }
            }
        }
    }
}
#elif defined(__AVX2__)
{
    // AVX2: process 8 output channels at once.
    // Parallelise over tile_idx for better utilisation when C_out < 8*num_threads.
    const int C_out8 = (C_out / 8) * 8;
    alignas(32) int32_t bias_all[2048]={}, mult_all[2048]={}, nexp_all[2048]={};
    for (int n = 0; n < C_out8; ++n) {
        bias_all[n] = (int32_t)eff_bias[n];
        mult_all[n] = req_mult[n];
        nexp_all[n] = -req_exp[n];
    }
    const __m256i vzp = _mm256_set1_epi32((int32_t)out_zp);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(omp_get_max_threads() > 1)
#endif
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const int th = tile_idx / tW, tw = tile_idx % tW;
        for (int n = 0; n < C_out8; n += 8) {
            __m256i m[4][4];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) {
                    const int pos = i * 4 + j;
                    __m256i raw = _mm256_loadu_si256(
                        (const __m256i*)(output_hat[pos] + (size_t)tile_idx * C_out + n));
                    __m256i s = qrdmulh_epi32_256(raw, _mm256_set1_epi32(cs_mult[pos]));
                    if (cs_exp[pos] < 0)
                        m[i][j] = _mm256_srav_epi32(s, _mm256_set1_epi32(-cs_exp[pos]));
                    else if (cs_exp[pos] > 0)
                        m[i][j] = _mm256_sllv_epi32(s, _mm256_set1_epi32(cs_exp[pos]));
                    else
                        m[i][j] = s;
                }
            __m256i tmp[2][4];
            for (int j = 0; j < 4; ++j) {
                tmp[0][j] = _mm256_add_epi32(_mm256_add_epi32(m[0][j], m[1][j]), m[2][j]);
                tmp[1][j] = _mm256_sub_epi32(_mm256_sub_epi32(m[1][j], m[2][j]), m[3][j]);
            }
            __m256i out2x2[2][2];
            for (int i = 0; i < 2; ++i) {
                out2x2[i][0] = _mm256_add_epi32(_mm256_add_epi32(tmp[i][0], tmp[i][1]), tmp[i][2]);
                out2x2[i][1] = _mm256_sub_epi32(_mm256_sub_epi32(tmp[i][1], tmp[i][2]), tmp[i][3]);
            }
            const __m256i vbias = _mm256_loadu_si256((const __m256i*)(bias_all + n));
            const __m256i vqm   = _mm256_loadu_si256((const __m256i*)(mult_all + n));
            const __m256i vqe   = _mm256_loadu_si256((const __m256i*)(nexp_all + n));
            for (int di = 0; di < 2; ++di) {
                int oh = th * 2 + di; if (oh >= oH) continue;
                for (int dj = 0; dj < 2; ++dj) {
                    int ow = tw * 2 + dj; if (ow >= oW) continue;
                    __m256i acc = _mm256_add_epi32(out2x2[di][dj], vbias);
                    __m256i qm2 = qrdmulh_epi32_256(acc, vqm);
                    __m256i q   = _mm256_add_epi32(_mm256_srav_epi32(qm2, vqe), vzp);
                    q = _mm256_min_epi32(_mm256_max_epi32(q, _mm256_set1_epi32(-128)),
                                         _mm256_set1_epi32(127));
                    __m128i q16 = _mm_packs_epi32(_mm256_castsi256_si128(q),
                                                   _mm256_extracti128_si256(q, 1));
                    __m128i out8 = _mm_packs_epi16(q16, _mm_setzero_si128());
                    _mm_storel_epi64((__m128i*)(output + (oh * oW + ow) * C_out + n), out8);
                }
            }
        }
        // Scalar tail for remaining < 8 channels
        for (int n = C_out8; n < C_out; ++n) {
            const int32_t bias = (int32_t)eff_bias[n];
            const int32_t qm   = req_mult[n];
            const int32_t qe   = req_exp[n];
            int32_t mm[4][4];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) {
                    int pos = i*4+j;
                    mm[i][j] = apply_q31_scalar(output_hat[pos][tile_idx*C_out+n], cs_mult[pos], cs_exp[pos]);
                }
            int32_t tmps[2][4];
            for (int j = 0; j < 4; ++j) {
                tmps[0][j] = mm[0][j]+mm[1][j]+mm[2][j];
                tmps[1][j] = mm[1][j]-mm[2][j]-mm[3][j];
            }
            for (int di = 0; di < 2; ++di) {
                int oh = th*2+di; if (oh >= oH) continue;
                for (int dj = 0; dj < 2; ++dj) {
                    int ow = tw*2+dj; if (ow >= oW) continue;
                    int32_t r = (dj==0) ? (tmps[di][0]+tmps[di][1]+tmps[di][2])
                                        : (tmps[di][1]-tmps[di][2]-tmps[di][3]);
                    output[(oh*oW+ow)*C_out+n] = requant_fixedpoint(r+bias, qm, qe, out_zp);
                }
            }
        }
    }  // end tile_idx loop
}
#else
    // Scalar fallback (no SIMD)
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(omp_get_max_threads() > 1)
#endif
    for (int n = 0; n < C_out; ++n) {
        const int32_t bias = (int32_t)eff_bias[n];
        const int32_t qm = req_mult[n];
        const int32_t qe = req_exp[n];

        for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            const int th = tile_idx / tW;
            const int tw = tile_idx % tW;

            int32_t mm[4][4];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) {
                    int pos = i * 4 + j;
                    int32_t raw = output_hat[pos][tile_idx * C_out + n];
                    mm[i][j] = apply_q31_scalar(raw, cs_mult[pos], cs_exp[pos]);
                }

            int32_t tmp[2][4];
            for (int j = 0; j < 4; ++j) {
                tmp[0][j] = mm[0][j] + mm[1][j] + mm[2][j];
                tmp[1][j] = mm[1][j] - mm[2][j] - mm[3][j];
            }

            int32_t out2x2[2][2];
            for (int i = 0; i < 2; ++i) {
                out2x2[i][0] = tmp[i][0] + tmp[i][1] + tmp[i][2];
                out2x2[i][1] = tmp[i][1] - tmp[i][2] - tmp[i][3];
            }

            for (int di = 0; di < 2; ++di) {
                int oh = th * 2 + di;
                if (oh >= oH) continue;
                for (int dj = 0; dj < 2; ++dj) {
                    int ow = tw * 2 + dj;
                    if (ow >= oW) continue;
                    int32_t total = out2x2[di][dj] + bias;
                    // NHWC output
                    output[(oh * oW + ow) * C_out + n] =
                        requant_fixedpoint(total, qm, qe, out_zp);
                }
            }
        }
    }
#endif

    g_wino_output_ms += now_ms() - t0;
}


#ifdef __ARM_NEON
// ──────────────────────────────────────────────────────────────
// Direct 3×3 depthwise INT8 NEON kernels (no im2col, no GEMM).
// Used when groups == C_in, kH=kW=3, pad=1, stride=1 or 2.
// Per-channel: input [H,W], weights [9], output [oH,oW] (NCHW layout).
// ──────────────────────────────────────────────────────────────

// stride=1, pad=1: oH=H, oW=W.
// Dual-row sliding window: processes 2 output rows per iteration, sharing
// 2 of the 4 needed input rows → 33% fewer loads in the hot 16-output block.
// 4-slot ring buffer, stepped by 2 each pair.
static void dw3x3_s1_int8_ch(
    const int8_t* inp, const int8_t* wt, int8_t* out,
    int H, int W, int8_t in_zp,
    int64_t bias, int32_t mult, int32_t exp_val, int8_t ozp)
{
    const int oH = H, oW = W;
    const int32_t bias32 = (int32_t)bias;
    const int32x4_t vbias = vdupq_n_s32(bias32);
    const int32x4_t vmult = vdupq_n_s32(mult);
    const int32x4_t vexp  = vdupq_n_s32(exp_val);
    const int32x4_t vozp  = vdupq_n_s32((int32_t)ozp);

    auto rq4 = [&](int32x4_t v) __attribute__((always_inline)) {
        return vaddq_s32(vrshlq_s32(vqrdmulhq_s32(vaddq_s32(v, vbias), vmult), vexp), vozp);
    };

    // Hoist 9 weight scalars (constant for the whole channel).
    const int8x8_t wv0=vdup_n_s8(wt[0]), wv1=vdup_n_s8(wt[1]), wv2=vdup_n_s8(wt[2]);
    const int8x8_t wv3=vdup_n_s8(wt[3]), wv4=vdup_n_s8(wt[4]), wv5=vdup_n_s8(wt[5]);
    const int8x8_t wv6=vdup_n_s8(wt[6]), wv7=vdup_n_s8(wt[7]), wv8=vdup_n_s8(wt[8]);

    // Inline MAC helpers: multiply 16-lane input by scalar weight, accumulate into 4 int32x4.
    // Low half → a0,a1 ; high half → a2,a3.
#define MAC16(a0,a1,a2,a3, iv, wk) do { \
        int16x8_t _plo = vmull_s8(vget_low_s8(iv), wk); \
        int16x8_t _phi = vmull_s8(vget_high_s8(iv), wk); \
        a0 = vaddw_s16(a0, vget_low_s16(_plo));  \
        a1 = vaddw_s16(a1, vget_high_s16(_plo)); \
        a2 = vaddw_s16(a2, vget_low_s16(_phi));  \
        a3 = vaddw_s16(a3, vget_high_s16(_phi)); \
    } while(0)
#define MAC8(a0,a1, iv8, wk) do { \
        int16x8_t _p = vmull_s8(iv8, wk); \
        a0 = vaddw_s16(a0, vget_low_s16(_p)); \
        a1 = vaddw_s16(a1, vget_high_s16(_p)); \
    } while(0)
#define STORE16(dst, q0,q1,q2,q3) \
        vst1q_s8(dst, vcombine_s8( \
            vqmovn_s16(vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1))), \
            vqmovn_s16(vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3)))))
#define STORE8(dst, q0,q1) \
        vst1_s8(dst, vqmovn_s16(vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1))))

    // Pre-compute all H+2 padded rows (incl. top/bottom padding) into a flat buffer.
    // Eliminates fill_slot from the hot output-row loop; output loop is pure compute.
    // Buffer: (H+2) rows × (W+2) bytes each.  For W=112→114×114=12996 bytes (fits L1).
    static thread_local std::vector<int8_t> _tls_rows;
    const int row_stride = W + 2;
    const int buf_rows   = H + 2;
    if ((int)_tls_rows.size() < buf_rows * row_stride)
        _tls_rows.resize(buf_rows * row_stride);
    int8_t* buf = _tls_rows.data();

    // Row 0 in buf = padded row(ih=-1) = all in_zp.
    memset(buf, (uint8_t)in_zp, row_stride);
    // Rows 1..d1 = padded data rows.
    for (int ih = 0; ih < H; ++ih) {
        int8_t* s = buf + (ih + 1) * row_stride;
        s[0] = in_zp;  s[row_stride - 1] = in_zp;
        memcpy(s + 1, inp + (size_t)ih * W, W);
    }
    // Row H+1 in buf = padded row(ih=H) = all in_zp.
    memset(buf + (H + 1) * row_stride, (uint8_t)in_zp, row_stride);

    // ── Dual-row loop ──────────────────────────────────────────────────────────
    // Each iteration processes output rows oh and oh+1 together.
    // Needs padded rows r0=row(oh-1), r1=row(oh), r2=row(oh+1), r3=row(oh+2).
    // r1 and r2 are shared between the two output rows → 12 loads vs 18.
    // In buf: padded row(ih) is at buf + (ih+1)*row_stride.
    int oh = 0;
    for (; oh + 1 < oH; oh += 2) {
        const int8_t* r0 = buf + oh * row_stride;        // row(oh-1)
        const int8_t* r1 = buf + (oh + 1) * row_stride;  // row(oh)
        const int8_t* r2 = buf + (oh + 2) * row_stride;  // row(oh+1) — SHARED
        const int8_t* r3 = buf + (oh + 3) * row_stride;  // row(oh+2)

        int8_t* out0 = out + oh * oW;
        int8_t* out1 = out + (oh + 1) * oW;
        int ow = 0;

        for (; ow + 16 <= oW; ow += 16) {
            // Issue all 12 loads first (r1 and r2 reused for both output rows).
            int8x16_t i00=vld1q_s8(r0+ow  ), i01=vld1q_s8(r0+ow+1), i02=vld1q_s8(r0+ow+2);
            int8x16_t i10=vld1q_s8(r1+ow  ), i11=vld1q_s8(r1+ow+1), i12=vld1q_s8(r1+ow+2);
            int8x16_t i20=vld1q_s8(r2+ow  ), i21=vld1q_s8(r2+ow+1), i22=vld1q_s8(r2+ow+2);
            int8x16_t i30=vld1q_s8(r3+ow  ), i31=vld1q_s8(r3+ow+1), i32=vld1q_s8(r3+ow+2);

            // Row oh: kh=0→r0, kh=1→r1, kh=2→r2
            int32x4_t a0=vdupq_n_s32(0),a1=vdupq_n_s32(0),a2=vdupq_n_s32(0),a3=vdupq_n_s32(0);
            MAC16(a0,a1,a2,a3, i00,wv0); MAC16(a0,a1,a2,a3, i01,wv1); MAC16(a0,a1,a2,a3, i02,wv2);
            MAC16(a0,a1,a2,a3, i10,wv3); MAC16(a0,a1,a2,a3, i11,wv4); MAC16(a0,a1,a2,a3, i12,wv5);
            MAC16(a0,a1,a2,a3, i20,wv6); MAC16(a0,a1,a2,a3, i21,wv7); MAC16(a0,a1,a2,a3, i22,wv8);
            STORE16(out0+ow, rq4(a0),rq4(a1),rq4(a2),rq4(a3));

            // Row oh+1: kh=0→r1, kh=1→r2, kh=2→r3  (r1,r2 reused from above)
            int32x4_t b0=vdupq_n_s32(0),b1=vdupq_n_s32(0),b2=vdupq_n_s32(0),b3=vdupq_n_s32(0);
            MAC16(b0,b1,b2,b3, i10,wv0); MAC16(b0,b1,b2,b3, i11,wv1); MAC16(b0,b1,b2,b3, i12,wv2);
            MAC16(b0,b1,b2,b3, i20,wv3); MAC16(b0,b1,b2,b3, i21,wv4); MAC16(b0,b1,b2,b3, i22,wv5);
            MAC16(b0,b1,b2,b3, i30,wv6); MAC16(b0,b1,b2,b3, i31,wv7); MAC16(b0,b1,b2,b3, i32,wv8);
            STORE16(out1+ow, rq4(b0),rq4(b1),rq4(b2),rq4(b3));
        }
        // 8-output tail (handles W divisible by 8 but not 16, e.g. W=56, 28)
        for (; ow + 8 <= oW; ow += 8) {
            int8x8_t j00=vld1_s8(r0+ow  ),j01=vld1_s8(r0+ow+1),j02=vld1_s8(r0+ow+2);
            int8x8_t j10=vld1_s8(r1+ow  ),j11=vld1_s8(r1+ow+1),j12=vld1_s8(r1+ow+2);
            int8x8_t j20=vld1_s8(r2+ow  ),j21=vld1_s8(r2+ow+1),j22=vld1_s8(r2+ow+2);
            int8x8_t j30=vld1_s8(r3+ow  ),j31=vld1_s8(r3+ow+1),j32=vld1_s8(r3+ow+2);
            int32x4_t a0=vdupq_n_s32(0),a1=vdupq_n_s32(0);
            MAC8(a0,a1,j00,wv0); MAC8(a0,a1,j01,wv1); MAC8(a0,a1,j02,wv2);
            MAC8(a0,a1,j10,wv3); MAC8(a0,a1,j11,wv4); MAC8(a0,a1,j12,wv5);
            MAC8(a0,a1,j20,wv6); MAC8(a0,a1,j21,wv7); MAC8(a0,a1,j22,wv8);
            STORE8(out0+ow, rq4(a0),rq4(a1));
            int32x4_t b0=vdupq_n_s32(0),b1=vdupq_n_s32(0);
            MAC8(b0,b1,j10,wv0); MAC8(b0,b1,j11,wv1); MAC8(b0,b1,j12,wv2);
            MAC8(b0,b1,j20,wv3); MAC8(b0,b1,j21,wv4); MAC8(b0,b1,j22,wv5);
            MAC8(b0,b1,j30,wv6); MAC8(b0,b1,j31,wv7); MAC8(b0,b1,j32,wv8);
            STORE8(out1+ow, rq4(b0),rq4(b1));
        }
        // Scalar tail (both rows)
        for (int ox = ow; ox < oW; ++ox) {
            int32_t a=bias32, b=bias32;
            for (int kh=0;kh<3;++kh) for (int kw=0;kw<3;++kw) {
                const int8_t* rows[4]={r0,r1,r2,r3};
                int w=(int32_t)wt[kh*3+kw];
                a += w*(int32_t)rows[kh  ][ox+kw];
                b += w*(int32_t)rows[kh+1][ox+kw];
            }
            out0[ox] = requant_fixedpoint(a, mult, exp_val, ozp);
            out1[ox] = requant_fixedpoint(b, mult, exp_val, ozp);
        }
    }

    // ── Single-row tail (when oH is odd) ──────────────────────────────────────
    if (oh < oH) {
        const int8_t* r0 = buf + oh * row_stride;
        const int8_t* r1 = buf + (oh + 1) * row_stride;
        const int8_t* r2 = buf + (oh + 2) * row_stride;
        int8_t* out_row  = out + oh * oW;
        int ow = 0;
        for (; ow + 16 <= oW; ow += 16) {
            int8x16_t i00=vld1q_s8(r0+ow  ),i01=vld1q_s8(r0+ow+1),i02=vld1q_s8(r0+ow+2);
            int8x16_t i10=vld1q_s8(r1+ow  ),i11=vld1q_s8(r1+ow+1),i12=vld1q_s8(r1+ow+2);
            int8x16_t i20=vld1q_s8(r2+ow  ),i21=vld1q_s8(r2+ow+1),i22=vld1q_s8(r2+ow+2);
            int32x4_t a0=vdupq_n_s32(0),a1=vdupq_n_s32(0),a2=vdupq_n_s32(0),a3=vdupq_n_s32(0);
            MAC16(a0,a1,a2,a3,i00,wv0); MAC16(a0,a1,a2,a3,i01,wv1); MAC16(a0,a1,a2,a3,i02,wv2);
            MAC16(a0,a1,a2,a3,i10,wv3); MAC16(a0,a1,a2,a3,i11,wv4); MAC16(a0,a1,a2,a3,i12,wv5);
            MAC16(a0,a1,a2,a3,i20,wv6); MAC16(a0,a1,a2,a3,i21,wv7); MAC16(a0,a1,a2,a3,i22,wv8);
            STORE16(out_row+ow, rq4(a0),rq4(a1),rq4(a2),rq4(a3));
        }
        for (; ow + 8 <= oW; ow += 8) {
            int8x8_t j00=vld1_s8(r0+ow),j01=vld1_s8(r0+ow+1),j02=vld1_s8(r0+ow+2);
            int8x8_t j10=vld1_s8(r1+ow),j11=vld1_s8(r1+ow+1),j12=vld1_s8(r1+ow+2);
            int8x8_t j20=vld1_s8(r2+ow),j21=vld1_s8(r2+ow+1),j22=vld1_s8(r2+ow+2);
            int32x4_t a0=vdupq_n_s32(0),a1=vdupq_n_s32(0);
            MAC8(a0,a1,j00,wv0); MAC8(a0,a1,j01,wv1); MAC8(a0,a1,j02,wv2);
            MAC8(a0,a1,j10,wv3); MAC8(a0,a1,j11,wv4); MAC8(a0,a1,j12,wv5);
            MAC8(a0,a1,j20,wv6); MAC8(a0,a1,j21,wv7); MAC8(a0,a1,j22,wv8);
            STORE8(out_row+ow, rq4(a0),rq4(a1));
        }
        for (; ow < oW; ++ow) {
            int32_t acc = bias32;
            for (int kh=0;kh<3;++kh) for (int kw=0;kw<3;++kw) {
                const int8_t* rows[3]={r0,r1,r2};
                acc += (int32_t)wt[kh*3+kw]*(int32_t)rows[kh][ow+kw];
            }
            out_row[ow] = requant_fixedpoint(acc, mult, exp_val, ozp);
        }
    }

#undef MAC16
#undef MAC8
#undef STORE16
#undef STORE8
}

// stride=2, pad=1: oH=ceil(H/2), oW=ceil(W/2).
static void dw3x3_s2_int8_ch(
    const int8_t* inp, const int8_t* wt, int8_t* out,
    int H, int W, int oH, int oW, int8_t in_zp,
    int64_t bias, int32_t mult, int32_t exp_val, int8_t ozp)
{
    const int32_t bias32 = (int32_t)bias;
    const int32x4_t vbias = vdupq_n_s32(bias32);
    const int32x4_t vmult = vdupq_n_s32(mult);
    const int32x4_t vexp  = vdupq_n_s32(exp_val);
    const int32x4_t vozp  = vdupq_n_s32((int32_t)ozp);
    auto rq4 = [&](int32x4_t v) {
        return vaddq_s32(vrshlq_s32(vqrdmulhq_s32(vaddq_s32(v, vbias), vmult), vexp), vozp);
    };

    static thread_local std::vector<int8_t> _tls_rows2;
    const int rs = W + 2;
    if ((int)_tls_rows2.size() < 3 * rs) _tls_rows2.resize(3 * rs);
    int8_t* row[3];
    for (int i = 0; i < 3; ++i) row[i] = _tls_rows2.data() + i * rs;

    auto fill_row = [&](int ih, int ri) {
        std::fill(row[ri], row[ri] + rs, in_zp);
        if (ih >= 0 && ih < H) memcpy(row[ri] + 1, inp + (size_t)ih * W, W);
    };

    for (int oh = 0; oh < oH; ++oh) {
        int ih0 = oh * 2;
        fill_row(ih0 - 1, 0);
        fill_row(ih0,     1);
        fill_row(ih0 + 1, 2);
        const int8_t* r[3] = {row[0], row[1], row[2]};
        int8_t* out_row = out + oh * oW;
        int ow = 0;
        // Hoist 9 weight vectors.
        int8x8_t wv[9];
        for (int k = 0; k < 9; ++k) wv[k] = vdup_n_s8(wt[k]);
        // NEON: 8 outputs; stride=2 → load 16 bytes, deinterleave even positions.
        for (; ow + 8 <= oW; ow += 8) {
            int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    int8x16_t  in_v = vld1q_s8(r[kh] + 2 * ow + kw);
                    int8x8x2_t uz   = vuzp_s8(vget_low_s8(in_v), vget_high_s8(in_v));
                    int16x8_t  p    = vmull_s8(uz.val[0], wv[kh*3+kw]);
                    a0 = vaddw_s16(a0, vget_low_s16(p));
                    a1 = vaddw_s16(a1, vget_high_s16(p));
                }
            }
            int32x4_t q0 = rq4(a0), q1 = rq4(a1);
            vst1_s8(out_row + ow, vqmovn_s16(vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1))));
        }
        // Scalar tail
        for (; ow < oW; ++ow) {
            int32_t acc = bias32;
            for (int kh = 0; kh < 3; ++kh)
                for (int kw = 0; kw < 3; ++kw)
                    acc += (int32_t)wt[kh*3+kw] * (int32_t)r[kh][2*ow+kw];
            out_row[ow] = requant_fixedpoint(acc, mult, exp_val, ozp);
        }
    }
}

// Dispatcher: parallel over C_in channels.
static void conv2d_depthwise_int8(
    const int8_t*  input,  const int8_t*  weight,
    const int64_t* eff_bias, const int32_t* req_mult, const int32_t* req_exp,
    int8_t in_zp, int8_t out_zp, int8_t* output,
    int C_in, int H, int W, int oH, int oW, int stride, int nthreads)
{
    double _t0 = now_ms();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads)
#endif
    for (int g = 0; g < C_in; ++g) {
        const int8_t* in_g  = input  + (size_t)g * H  * W;
        const int8_t* wt_g  = weight + (size_t)g * 9;
        int8_t*       out_g = output + (size_t)g * oH * oW;
        if (stride == 1)
            dw3x3_s1_int8_ch(in_g, wt_g, out_g, H, W, in_zp,
                              eff_bias[g], req_mult[g], req_exp[g], out_zp);
        else
            dw3x3_s2_int8_ch(in_g, wt_g, out_g, H, W, oH, oW, in_zp,
                              eff_bias[g], req_mult[g], req_exp[g], out_zp);
    }
    g_depthwise_ms += now_ms() - _t0;
}
#endif // __ARM_NEON

// ──────────────────────────────────────────────────────────────
// conv2d_int8 — NCHW or NHWC, uses im2col + SDOT GEMM
// im2col produces [oHW, K] directly (no intermediate transpose).
// scratch_col: caller-supplied buffer of at least oHW*K bytes (or nullptr).
// nhwc: if true, input is [H, W, C_in] and output is [oH, oW, C_out]
// ──────────────────────────────────────────────────────────────
void conv2d_int8(
    const int8_t*  input,
    const int8_t*  weight,
    const int8_t*  w_pre_packed,
    const int64_t* eff_bias,
    const int32_t* req_mult,
    const int32_t* req_exp,
    int8_t         in_zp,
    int8_t         out_zp,
    int8_t*        output,
    int C_in, int H, int W,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int groups,
    int8_t* scratch_col,
    bool nhwc,
    const int32_t* b_row_sums,
    StreamHandle /* stream */)
{
    int oH = (H + 2 * pad_h - kH) / stride_h + 1;
    int oW = (W + 2 * pad_w - kW) / stride_w + 1;

    int C_in_g  = C_in  / groups;
    int C_out_g = C_out / groups;
    int K = kH * kW * C_in_g;
    int oHW = oH * oW;

    // ── NHWC path ───────────────────────────────────────────────
    if (nhwc) {
        assert(groups == 1);  // NHWC only supports groups=1

        const int8_t* w_pack_ptr;
        int8_t* w_packed_local = nullptr;
        if (w_pre_packed) {
            w_pack_ptr = w_pre_packed;
        } else {
            w_packed_local = pack_weights_sdot(weight, C_out, K);
            w_pack_ptr = w_packed_local;
        }

        bool is_1x1 = (kH == 1 && kW == 1 && pad_h == 0 && pad_w == 0);

        // Allocate col buffer for non-1x1 cases
        std::vector<int8_t> _col_local_nhwc;
        int8_t* col = scratch_col;
        bool needs_col = !(is_1x1 && stride_h == 1 && stride_w == 1);
        if (needs_col && !col) {
            _col_local_nhwc.resize((size_t)oHW * K);
            col = _col_local_nhwc.data();
        }

#ifdef _OPENMP
        const int nthreads = omp_get_max_threads();
        if (nthreads > 1) {
            // ── Fused NHWC path: ONE parallel region for im2col + GEMM ──────
            double t0 = now_ms();
#pragma omp parallel num_threads(nthreads)
            {
                if (is_1x1 && stride_h == 1 && stride_w == 1) {
                    // No im2col needed — input IS the A matrix
                    // no barrier needed
                } else if (is_1x1 && stride_h == 2) {
                    // 1×1 stride=2: subsample input rows
#pragma omp for schedule(static)
                    for (int oh = 0; oh < oH; ++oh) {
                        for (int ow = 0; ow < oW; ++ow) {
                            const int ih = oh * stride_h;
                            const int iw = ow * stride_w;
                            memcpy(col + (oh * oW + ow) * C_in, input + (ih * W + iw) * C_in, C_in);
                        }
                    }
                    // implicit barrier after omp for
                } else if (kH == 3 && kW == 3 && stride_h == 1 && stride_w == 1
                           && pad_h == 1 && pad_w == 1) {
                    im2col_3x3_nhwc_s1p1(input, C_in, H, W, in_zp, col, /*in_parallel=*/true);
                } else if (kH == 3 && kW == 3 && stride_h == 2 && stride_w == 2
                           && pad_h == 1 && pad_w == 1) {
                    im2col_3x3_nhwc_s2p1(input, C_in, H, W, in_zp, oH, oW, col, /*in_parallel=*/true);
                } else if (kH == 7 && kW == 7 && stride_h == 2 && stride_w == 2
                           && pad_h == 3 && pad_w == 3) {
                    im2col_7x7_nhwc_s2p3(input, C_in, H, W, in_zp, oH, oW, col, /*in_parallel=*/true);
                } else {
                    // General kernel: do im2col serially from one thread
#pragma omp single
                    {
                        for (int oh = 0; oh < oH; ++oh) {
                            for (int ow = 0; ow < oW; ++ow) {
                                int8_t* dst = col + (oh * oW + ow) * K;
                                int idx = 0;
                                for (int kh_i = 0; kh_i < kH; ++kh_i) {
                                    int ih = oh * stride_h - pad_h + kh_i;
                                    for (int kw_i = 0; kw_i < kW; ++kw_i) {
                                        int iw = ow * stride_w - pad_w + kw_i;
                                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                            memcpy(dst + idx, input + (ih * W + iw) * C_in, C_in);
                                        } else {
                                            memset(dst + idx, in_zp, C_in);
                                        }
                                        idx += C_in;
                                    }
                                }
                            }
                        }
                    }
                    // implicit barrier after omp single
                }
                // All threads run GEMM with in_parallel=true
                const int8_t* a_ptr = (is_1x1 && stride_h == 1 && stride_w == 1) ? input : col;
                gemm_int8(a_ptr, w_pack_ptr,
                          eff_bias, req_mult, req_exp, nullptr,
                          out_zp, output, false,
                          oHW, K, C_out,
                          /*nchw_out=*/false, /*in_parallel=*/true,
                          b_row_sums, nullptr);
            }
            g_gemm_ms += now_ms() - t0;  // im2col + GEMM combined (fused)
        } else
#endif
        {
            // ── Single-threaded NHWC path ────────────────────────────────
            if (is_1x1 && stride_h == 1 && stride_w == 1) {
                double t0 = now_ms();
                gemm_int8(input, w_pack_ptr,
                          eff_bias, req_mult, req_exp, nullptr,
                          out_zp, output, false,
                          oHW, K, C_out,
                          /*nchw_out=*/false, /*in_parallel=*/false,
                          b_row_sums, nullptr);
                g_gemm_ms += now_ms() - t0;
            } else if (is_1x1 && stride_h == 2) {
                double t0 = now_ms();
                for (int oh = 0; oh < oH; ++oh) {
                    for (int ow = 0; ow < oW; ++ow) {
                        const int ih = oh * stride_h;
                        const int iw = ow * stride_w;
                        memcpy(col + (oh * oW + ow) * C_in, input + (ih * W + iw) * C_in, C_in);
                    }
                }
                g_im2col_1x1_ms += now_ms() - t0;
                t0 = now_ms();
                gemm_int8(col, w_pack_ptr,
                          eff_bias, req_mult, req_exp, nullptr,
                          out_zp, output, false,
                          oHW, K, C_out,
                          /*nchw_out=*/false, /*in_parallel=*/false,
                          b_row_sums, nullptr);
                g_gemm_ms += now_ms() - t0;
            } else {
                {
                    double t0 = now_ms();
                    if (kH == 3 && kW == 3 && stride_h == 1 && stride_w == 1
                        && pad_h == 1 && pad_w == 1) {
                        im2col_3x3_nhwc_s1p1(input, C_in, H, W, in_zp, col);
                        g_im2col_3x3_ms += now_ms() - t0;
                    } else if (kH == 3 && kW == 3 && stride_h == 2 && stride_w == 2
                               && pad_h == 1 && pad_w == 1) {
                        im2col_3x3_nhwc_s2p1(input, C_in, H, W, in_zp, oH, oW, col);
                        g_im2col_3x3_ms += now_ms() - t0;
                    } else if (kH == 7 && kW == 7 && stride_h == 2 && stride_w == 2
                               && pad_h == 3 && pad_w == 3) {
                        im2col_7x7_nhwc_s2p3(input, C_in, H, W, in_zp, oH, oW, col);
                        g_im2col_gen_ms += now_ms() - t0;
                    } else {
                        for (int oh = 0; oh < oH; ++oh) {
                            for (int ow = 0; ow < oW; ++ow) {
                                int8_t* dst = col + (oh * oW + ow) * K;
                                int idx = 0;
                                for (int kh_i = 0; kh_i < kH; ++kh_i) {
                                    int ih = oh * stride_h - pad_h + kh_i;
                                    for (int kw_i = 0; kw_i < kW; ++kw_i) {
                                        int iw = ow * stride_w - pad_w + kw_i;
                                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                            memcpy(dst + idx, input + (ih * W + iw) * C_in, C_in);
                                        } else {
                                            memset(dst + idx, in_zp, C_in);
                                        }
                                        idx += C_in;
                                    }
                                }
                            }
                        }
                        g_im2col_gen_ms += now_ms() - t0;
                    }
                }
                {
                    double t0 = now_ms();
                    gemm_int8(col, w_pack_ptr,
                              eff_bias, req_mult, req_exp, nullptr,
                              out_zp, output, false,
                              oHW, K, C_out,
                              /*nchw_out=*/false, /*in_parallel=*/false,
                              b_row_sums, nullptr);
                    g_gemm_ms += now_ms() - t0;
                }
            }
        }

        if (w_packed_local) free_packed(w_packed_local);
        return;
    }

    // ── NCHW path ─────────────────────────────────────────────────
#ifdef __ARM_NEON
    // Fast path: direct 3×3 depthwise NEON (groups == C_in, C_out_g=1, pad=1, stride=1/2).
    // Bypasses im2col + GEMM entirely; ~2× faster for depthwise layers.
    if (groups == C_in && C_out_g == 1 &&
        kH == 3 && kW == 3 && pad_h == 1 && pad_w == 1 &&
        (stride_h == 1 || stride_h == 2) && stride_w == stride_h) {
        const int nthreads_dw = omp_get_max_threads();
        conv2d_depthwise_int8(input, weight, eff_bias, req_mult, req_exp,
                              in_zp, out_zp, output,
                              C_in, H, W, oH, oW, stride_h, nthreads_dw);
        return;
    }
#endif

    // For groups=1: single group, gemm_int8 will spawn its own parallel region.
    // For groups>1 (depthwise): parallelize over groups — each thread owns a subset
    //   of groups and runs im2col+GEMM serially for each.  This keeps all threads
    //   busy for depthwise layers (where per-group work is tiny) without the
    //   per-group OMP-spawn overhead that previously caused large regressions.
    //
    // Pre-pack all group weights serially to avoid concurrent heap allocation.

#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
#endif

    // Pre-pack weights for all groups (serial)
    std::vector<int8_t*> w_packs_owned(groups, nullptr);
    std::vector<const int8_t*> w_packs(groups, nullptr);
    if (w_pre_packed && groups == 1) {
        w_packs[0] = w_pre_packed;
    } else {
        for (int g = 0; g < groups; ++g) {
            w_packs_owned[g] = pack_weights_sdot(weight + g * C_out_g * K, C_out_g, K);
            w_packs[g] = w_packs_owned[g];
        }
    }

    // For groups=1: single-group col buffer (caller scratch or local)
    std::vector<int8_t> _col_local;
    int8_t* col1 = scratch_col;
    if (groups == 1 && !col1) {
        _col_local.resize((size_t)oHW * K);
        col1 = _col_local.data();
    }

#ifdef _OPENMP
    // groups>1: parallelize over groups; groups=1: serial (gemm_int8 spawns internally)
#pragma omp parallel for schedule(static) num_threads(nthreads) if(groups > 1)
#endif
    for (int g = 0; g < groups; ++g) {
        const int8_t* in_g  = input  + (size_t)g * C_in_g * H * W;
        int8_t*       out_g = output + (size_t)g * C_out_g * oHW;
        const int8_t* w_pack_ptr = w_packs[g];

        // For groups>1: use thread-local col buffer (safe for concurrent group loops).
        // For groups=1: use col1 (single-threaded, no TLS needed).
        int8_t* col_g;
        if (groups > 1) {
            static thread_local std::vector<int8_t> _tls_col;
            if ((int)_tls_col.size() < oHW * K) _tls_col.resize((size_t)oHW * K);
            col_g = _tls_col.data();
        } else {
            col_g = col1;
        }

        // im2col (single-threaded per group)
        if (kH == 1 && kW == 1 && pad_h == 0 && pad_w == 0)
            im2col_1x1(in_g, C_in_g, H, W, stride_h, stride_w, oH, oW, col_g);
        else if (kH == 3 && kW == 3 && stride_h == 1 && stride_w == 1 && pad_h == 1 && pad_w == 1)
            im2col_3x3s1p1(in_g, C_in_g, H, W, in_zp, col_g);
        else if (kH == 3 && kW == 3 && stride_h == 2 && stride_w == 2 && pad_h == 1 && pad_w == 1)
            im2col_3x3s2p1(in_g, C_in_g, H, W, in_zp, oH, oW, col_g);
        else if (kH == 7 && kW == 7 && stride_h == 2 && stride_w == 2 && pad_h == 3 && pad_w == 3)
            im2col_7x7s2p3(in_g, C_in_g, H, W, in_zp, oH, oW, col_g);
        else
            im2col(in_g, C_in_g, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w, oH, oW, col_g, in_zp);

        // GEMM: in_parallel=false.
        //   groups=1: not in a parallel region → gemm_int8 spawns its own (8T).
        //   groups>1: inside #pragma omp parallel for → omp_in_parallel()=true
        //             → gemm_int8 skips spawn, uses nT=1/tid=0 → all m8-tiles on this thread.
        gemm_int8(col_g, w_pack_ptr,
                  eff_bias + g * C_out_g,
                  req_mult  + g * C_out_g,
                  req_exp   + g * C_out_g,
                  nullptr,
                  out_zp, out_g, false,
                  oHW, K, C_out_g,
                  /*nchw_out=*/true, /*in_parallel=*/false, nullptr);
    }

    for (auto p : w_packs_owned) if (p) free_packed(p);
}


// ──────────────────────────────────────────────────────────────
// Native NHWC 3×3 depthwise INT8 convolution (stride=1 and stride=2).
// Operates end-to-end in NHWC layout — no NHWC↔NCHW transpositions.
// w_hwc:   weights in [9, C] tap-major order (tap index is outermost).
// eff_b32: eff_bias truncated to int32.
// OMP-parallel over output rows; 16-channel SIMD blocks (NEON only).
// ──────────────────────────────────────────────────────────────
#ifdef __ARM_NEON

// Inline NHWC MAC: element-wise multiply 16 input channels by 16 weight
// channels, widen to int32, accumulate into 4 int32x4 registers.
#define DW_NHWC_MAC(a0,a1,a2,a3, iv, wv) do { \
    int16x8_t _plo = vmull_s8(vget_low_s8(iv),  vget_low_s8(wv));  \
    int16x8_t _phi = vmull_s8(vget_high_s8(iv), vget_high_s8(wv)); \
    (a0) = vaddw_s16((a0), vget_low_s16(_plo));  \
    (a1) = vaddw_s16((a1), vget_high_s16(_plo)); \
    (a2) = vaddw_s16((a2), vget_low_s16(_phi));  \
    (a3) = vaddw_s16((a3), vget_high_s16(_phi)); \
} while(0)

// Requantize 4 int32x4 accumulators → int8x16.
#define DW_NHWC_RQ(a0,a1,a2,a3, m0,m1,m2,m3, e0,e1,e2,e3, vozp) \
    vcombine_s8( \
        vqmovn_s16(vcombine_s16( \
            vqmovn_s32(vaddq_s32(vrshlq_s32(vqrdmulhq_s32((a0),(m0)),(e0)),(vozp))), \
            vqmovn_s32(vaddq_s32(vrshlq_s32(vqrdmulhq_s32((a1),(m1)),(e1)),(vozp))))), \
        vqmovn_s16(vcombine_s16( \
            vqmovn_s32(vaddq_s32(vrshlq_s32(vqrdmulhq_s32((a2),(m2)),(e2)),(vozp))), \
            vqmovn_s32(vaddq_s32(vrshlq_s32(vqrdmulhq_s32((a3),(m3)),(e3)),(vozp))))))

// stride=1, pad=1: oH=H, oW=W.
static void dw3x3_nhwc_s1_int8(
    const int8_t* inp, int8_t* out,
    int H, int W, int C,
    int8_t in_zp,
    const int8_t* w_hwc, const int32_t* eff_b32,
    const int32_t* req_mult, const int32_t* req_exp,
    int8_t ozp, int nthreads)
{
    const int8x16_t vZP    = vdupq_n_s8(in_zp);
    const int32x4_t vozp_v = vdupq_n_s32((int32_t)ozp);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads)
#endif
    for (int oh = 0; oh < H; ++oh) {
        const int8_t* r0 = (oh > 0)   ? (inp + (oh-1)*(size_t)W*C) : nullptr;
        const int8_t* r1 =               inp + oh*(size_t)W*C;
        const int8_t* r2 = (oh < H-1) ? (inp + (oh+1)*(size_t)W*C) : nullptr;
        int8_t*      orow = out + oh*(size_t)W*C;

        for (int c = 0; c + 15 < C; c += 16) {
            const int8x16_t wv0=vld1q_s8(w_hwc+0*C+c), wv1=vld1q_s8(w_hwc+1*C+c);
            const int8x16_t wv2=vld1q_s8(w_hwc+2*C+c), wv3=vld1q_s8(w_hwc+3*C+c);
            const int8x16_t wv4=vld1q_s8(w_hwc+4*C+c), wv5=vld1q_s8(w_hwc+5*C+c);
            const int8x16_t wv6=vld1q_s8(w_hwc+6*C+c), wv7=vld1q_s8(w_hwc+7*C+c);
            const int8x16_t wv8=vld1q_s8(w_hwc+8*C+c);
            const int32x4_t vb0=vld1q_s32(eff_b32+c),   vb1=vld1q_s32(eff_b32+c+4);
            const int32x4_t vb2=vld1q_s32(eff_b32+c+8), vb3=vld1q_s32(eff_b32+c+12);
            const int32x4_t vm0=vld1q_s32(req_mult+c),   vm1=vld1q_s32(req_mult+c+4);
            const int32x4_t vm2=vld1q_s32(req_mult+c+8), vm3=vld1q_s32(req_mult+c+12);
            const int32x4_t ve0=vld1q_s32(req_exp+c),    ve1=vld1q_s32(req_exp+c+4);
            const int32x4_t ve2=vld1q_s32(req_exp+c+8),  ve3=vld1q_s32(req_exp+c+12);

            for (int ow = 0; ow < W; ++ow) {
                int8x16_t iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7, iv8;
                if (__builtin_expect(r0 && r2 && ow > 0 && ow < W-1, 1)) {
                    iv0=vld1q_s8(r0+(ow-1)*C+c); iv1=vld1q_s8(r0+ow*C+c); iv2=vld1q_s8(r0+(ow+1)*C+c);
                    iv3=vld1q_s8(r1+(ow-1)*C+c); iv4=vld1q_s8(r1+ow*C+c); iv5=vld1q_s8(r1+(ow+1)*C+c);
                    iv6=vld1q_s8(r2+(ow-1)*C+c); iv7=vld1q_s8(r2+ow*C+c); iv8=vld1q_s8(r2+(ow+1)*C+c);
                } else {
                    auto ld = [&](const int8_t* row, int iw) -> int8x16_t {
                        return (!row || iw < 0 || iw >= W) ? vZP : vld1q_s8(row + iw*C + c);
                    };
                    iv0=ld(r0,ow-1); iv1=ld(r0,ow); iv2=ld(r0,ow+1);
                    iv3=ld(r1,ow-1); iv4=ld(r1,ow); iv5=ld(r1,ow+1);
                    iv6=ld(r2,ow-1); iv7=ld(r2,ow); iv8=ld(r2,ow+1);
                }
                int32x4_t a0=vb0, a1=vb1, a2=vb2, a3=vb3;
                DW_NHWC_MAC(a0,a1,a2,a3, iv0,wv0); DW_NHWC_MAC(a0,a1,a2,a3, iv1,wv1);
                DW_NHWC_MAC(a0,a1,a2,a3, iv2,wv2); DW_NHWC_MAC(a0,a1,a2,a3, iv3,wv3);
                DW_NHWC_MAC(a0,a1,a2,a3, iv4,wv4); DW_NHWC_MAC(a0,a1,a2,a3, iv5,wv5);
                DW_NHWC_MAC(a0,a1,a2,a3, iv6,wv6); DW_NHWC_MAC(a0,a1,a2,a3, iv7,wv7);
                DW_NHWC_MAC(a0,a1,a2,a3, iv8,wv8);
                vst1q_s8(orow + ow*C + c,
                    DW_NHWC_RQ(a0,a1,a2,a3, vm0,vm1,vm2,vm3, ve0,ve1,ve2,ve3, vozp_v));
            }
        }
        // Scalar tail for C % 16 != 0.
        for (int c = (C>>4)<<4; c < C; ++c) {
            for (int ow = 0; ow < W; ++ow) {
                int32_t acc = eff_b32[c];
                for (int kh = 0; kh < 3; ++kh) {
                    const int8_t* row = (kh==0)?r0 : (kh==1)?r1 : r2;
                    for (int kw = 0; kw < 3; ++kw) {
                        int iw = ow + kw - 1;
                        int8_t v = (!row || iw < 0 || iw >= W) ? in_zp : row[iw*C + c];
                        acc += (int32_t)v * (int32_t)w_hwc[(kh*3+kw)*C + c];
                    }
                }
                int32_t q = apply_q31_scalar(acc, req_mult[c], req_exp[c]) + (int32_t)ozp;
                orow[ow*C + c] = (int8_t)std::max(-128, std::min(127, q));
            }
        }
    }
}

// stride=2, pad=1: oH=(H+1)/2, oW=(W+1)/2.
static void dw3x3_nhwc_s2_int8(
    const int8_t* inp, int8_t* out,
    int H, int W, int C, int oH, int oW,
    int8_t in_zp,
    const int8_t* w_hwc, const int32_t* eff_b32,
    const int32_t* req_mult, const int32_t* req_exp,
    int8_t ozp, int nthreads)
{
    const int8x16_t vZP    = vdupq_n_s8(in_zp);
    const int32x4_t vozp_v = vdupq_n_s32((int32_t)ozp);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads)
#endif
    for (int oh = 0; oh < oH; ++oh) {
        const int ih = oh * 2;
        const int8_t* r0 = (ih > 0)   ? (inp + (ih-1)*(size_t)W*C) : nullptr;
        const int8_t* r1 =               inp + ih*(size_t)W*C;
        const int8_t* r2 = (ih < H-1) ? (inp + (ih+1)*(size_t)W*C) : nullptr;
        int8_t*      orow = out + oh*(size_t)oW*C;

        for (int c = 0; c + 15 < C; c += 16) {
            const int8x16_t wv0=vld1q_s8(w_hwc+0*C+c), wv1=vld1q_s8(w_hwc+1*C+c);
            const int8x16_t wv2=vld1q_s8(w_hwc+2*C+c), wv3=vld1q_s8(w_hwc+3*C+c);
            const int8x16_t wv4=vld1q_s8(w_hwc+4*C+c), wv5=vld1q_s8(w_hwc+5*C+c);
            const int8x16_t wv6=vld1q_s8(w_hwc+6*C+c), wv7=vld1q_s8(w_hwc+7*C+c);
            const int8x16_t wv8=vld1q_s8(w_hwc+8*C+c);
            const int32x4_t vb0=vld1q_s32(eff_b32+c),   vb1=vld1q_s32(eff_b32+c+4);
            const int32x4_t vb2=vld1q_s32(eff_b32+c+8), vb3=vld1q_s32(eff_b32+c+12);
            const int32x4_t vm0=vld1q_s32(req_mult+c),   vm1=vld1q_s32(req_mult+c+4);
            const int32x4_t vm2=vld1q_s32(req_mult+c+8), vm3=vld1q_s32(req_mult+c+12);
            const int32x4_t ve0=vld1q_s32(req_exp+c),    ve1=vld1q_s32(req_exp+c+4);
            const int32x4_t ve2=vld1q_s32(req_exp+c+8),  ve3=vld1q_s32(req_exp+c+12);

            for (int ow = 0; ow < oW; ++ow) {
                const int iw = ow * 2;
                int8x16_t iv0, iv1, iv2, iv3, iv4, iv5, iv6, iv7, iv8;
                if (__builtin_expect(r0 && r2 && iw > 0 && iw < W-1, 1)) {
                    iv0=vld1q_s8(r0+(iw-1)*C+c); iv1=vld1q_s8(r0+iw*C+c); iv2=vld1q_s8(r0+(iw+1)*C+c);
                    iv3=vld1q_s8(r1+(iw-1)*C+c); iv4=vld1q_s8(r1+iw*C+c); iv5=vld1q_s8(r1+(iw+1)*C+c);
                    iv6=vld1q_s8(r2+(iw-1)*C+c); iv7=vld1q_s8(r2+iw*C+c); iv8=vld1q_s8(r2+(iw+1)*C+c);
                } else {
                    auto ld = [&](const int8_t* row, int x) -> int8x16_t {
                        return (!row || x < 0 || x >= W) ? vZP : vld1q_s8(row + x*C + c);
                    };
                    iv0=ld(r0,iw-1); iv1=ld(r0,iw); iv2=ld(r0,iw+1);
                    iv3=ld(r1,iw-1); iv4=ld(r1,iw); iv5=ld(r1,iw+1);
                    iv6=ld(r2,iw-1); iv7=ld(r2,iw); iv8=ld(r2,iw+1);
                }
                int32x4_t a0=vb0, a1=vb1, a2=vb2, a3=vb3;
                DW_NHWC_MAC(a0,a1,a2,a3, iv0,wv0); DW_NHWC_MAC(a0,a1,a2,a3, iv1,wv1);
                DW_NHWC_MAC(a0,a1,a2,a3, iv2,wv2); DW_NHWC_MAC(a0,a1,a2,a3, iv3,wv3);
                DW_NHWC_MAC(a0,a1,a2,a3, iv4,wv4); DW_NHWC_MAC(a0,a1,a2,a3, iv5,wv5);
                DW_NHWC_MAC(a0,a1,a2,a3, iv6,wv6); DW_NHWC_MAC(a0,a1,a2,a3, iv7,wv7);
                DW_NHWC_MAC(a0,a1,a2,a3, iv8,wv8);
                vst1q_s8(orow + ow*C + c,
                    DW_NHWC_RQ(a0,a1,a2,a3, vm0,vm1,vm2,vm3, ve0,ve1,ve2,ve3, vozp_v));
            }
        }
        // Scalar tail
        for (int c = (C>>4)<<4; c < C; ++c) {
            for (int ow = 0; ow < oW; ++ow) {
                const int iw = ow * 2;
                int32_t acc = eff_b32[c];
                for (int kh = 0; kh < 3; ++kh) {
                    const int8_t* row = (kh==0)?r0 : (kh==1)?r1 : r2;
                    for (int kw = 0; kw < 3; ++kw) {
                        int x = iw + kw - 1;
                        int8_t v = (!row || x < 0 || x >= W) ? in_zp : row[x*C + c];
                        acc += (int32_t)v * (int32_t)w_hwc[(kh*3+kw)*C + c];
                    }
                }
                int32_t q = apply_q31_scalar(acc, req_mult[c], req_exp[c]) + (int32_t)ozp;
                orow[ow*C + c] = (int8_t)std::max(-128, std::min(127, q));
            }
        }
    }
}

#undef DW_NHWC_MAC
#undef DW_NHWC_RQ
#endif // __ARM_NEON

void conv2d_depthwise_nhwc_int8(
    const int8_t* input, int8_t* output,
    int H, int W, int C, int oH, int oW, int stride,
    int8_t in_zp,
    const int8_t* w_hwc, const int32_t* eff_b32,
    const int32_t* req_mult, const int32_t* req_exp,
    int8_t out_zp, int nthreads, StreamHandle /*stream*/)
{
    if (nthreads <= 0) nthreads = omp_get_max_threads();
#ifdef __ARM_NEON
    if (stride == 1)
        dw3x3_nhwc_s1_int8(input, output, H, W, C, in_zp,
                            w_hwc, eff_b32, req_mult, req_exp, out_zp, nthreads);
    else
        dw3x3_nhwc_s2_int8(input, output, H, W, C, oH, oW, in_zp,
                            w_hwc, eff_b32, req_mult, req_exp, out_zp, nthreads);
#else
    // Scalar fallback — rarely hit (NEON assumed for ARM64 targets)
    for (int oh = 0; oh < oH; ++oh) {
        for (int ow = 0; ow < oW; ++ow) {
            const int ih = oh * stride, iw_base = ow * stride;
            for (int c = 0; c < C; ++c) {
                int32_t acc = eff_b32[c];
                for (int kh = 0; kh < 3; ++kh) {
                    int y = ih + kh - 1;
                    for (int kw = 0; kw < 3; ++kw) {
                        int x = iw_base + kw - 1;
                        int8_t v = (y < 0 || y >= H || x < 0 || x >= W)
                                   ? in_zp : input[(y*W + x)*C + c];
                        acc += (int32_t)v * (int32_t)w_hwc[(kh*3+kw)*C + c];
                    }
                }
                int32_t q = apply_q31_scalar(acc, req_mult[c], req_exp[c]) + (int32_t)out_zp;
                output[(oh*oW + ow)*C + c] = (int8_t)std::max(-128, std::min(127, q));
            }
        }
    }
#endif
}

// ──────────────────────────────────────────────────────────────
// FC (linear) → float32 output
// ──────────────────────────────────────────────────────────────
void linear_int8_to_float(
    const int8_t*  input,
    const int8_t*  weight,
    const int64_t* eff_bias,
    const float*   req_scale,
    float*         output,
    int C_in, int C_out)
{
    int8_t* w_packed = pack_weights_sdot(weight, C_out, C_in);
    gemm_int8(
        input, w_packed,
        eff_bias, nullptr, nullptr, req_scale, 0,
        output, /*is_float=*/true,
        1, C_in, C_out,
        /*nchw_out=*/false, /*in_parallel=*/false, nullptr);
    free_packed(w_packed);
}


// ──────────────────────────────────────────────────────────────
// MaxPool int8 (NCHW) — compare directly on int8 (monotone op)
// ──────────────────────────────────────────────────────────────

// ──────────────────────────────────────────────────────────────
// NHWC ↔ NCHW block transpose (8×8 NEON / scalar fallback)
// Internal helpers — only used by conv2d_grouped_nhwc_fallback_int8.
// ──────────────────────────────────────────────────────────────
static void nhwc_to_nchw_int8(const int8_t* nhwc, int8_t* nchw, int HW, int C);
static void nchw_to_nhwc_int8(const int8_t* nchw, int8_t* nhwc, int HW, int C);

// ──────────────────────────────────────────────────────────────
// Grouped NHWC depthwise fallback: NHWC→NCHW → conv2d_int8 → NCHW→NHWC.
// Used when the native NHWC depthwise kernel cannot handle the config
// (e.g., kernel != 3×3 or dilation != 1).  TLS scratch avoids per-call malloc.
// ──────────────────────────────────────────────────────────────
static thread_local std::vector<int8_t> _tls_dw_in_nchw;
static thread_local std::vector<int8_t> _tls_dw_out_nchw;

void conv2d_grouped_nhwc_fallback_int8(
    const int8_t*  input,
    const int8_t*  weight,
    const int64_t* eff_bias,
    const int32_t* req_mult,
    const int32_t* req_exp,
    int8_t         in_zp,
    int8_t         out_zp,
    int8_t*        output,
    int C, int H, int W,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int groups,
    int8_t*        scratch_col,
    const int32_t* w_row_sums,
    StreamHandle   stream)
{
    const int oH = (H + 2*pad_h - kH) / stride_h + 1;
    const int oW = (W + 2*pad_w - kW) / stride_w + 1;
    const size_t in_sz  = (size_t)C    * H  * W;
    const size_t out_sz = (size_t)C_out * oH * oW;
    if (_tls_dw_in_nchw.size()  < in_sz)  _tls_dw_in_nchw.resize(in_sz);
    if (_tls_dw_out_nchw.size() < out_sz) _tls_dw_out_nchw.resize(out_sz);

    nhwc_to_nchw_int8(input, _tls_dw_in_nchw.data(), H * W, C);

    conv2d_int8(_tls_dw_in_nchw.data(), weight, /*w_pre_packed=*/nullptr,
                eff_bias, req_mult, req_exp,
                in_zp, out_zp, _tls_dw_out_nchw.data(),
                C, H, W, C_out, kH, kW,
                stride_h, stride_w, pad_h, pad_w, groups,
                scratch_col, /*nhwc=*/false, w_row_sums, stream);

    nchw_to_nhwc_int8(_tls_dw_out_nchw.data(), output, oH * oW, C_out);
}

static void nhwc_to_nchw_int8(const int8_t* nhwc, int8_t* nchw, int HW, int C)
{
#ifdef __ARM_NEON
    int c = 0;
    for (; c + 7 < C; c += 8) {
        int hw = 0;
        for (; hw + 7 < HW; hw += 8) {
            int8x8_t r0=vld1_s8(nhwc+(hw+0)*C+c), r1=vld1_s8(nhwc+(hw+1)*C+c);
            int8x8_t r2=vld1_s8(nhwc+(hw+2)*C+c), r3=vld1_s8(nhwc+(hw+3)*C+c);
            int8x8_t r4=vld1_s8(nhwc+(hw+4)*C+c), r5=vld1_s8(nhwc+(hw+5)*C+c);
            int8x8_t r6=vld1_s8(nhwc+(hw+6)*C+c), r7=vld1_s8(nhwc+(hw+7)*C+c);
            int8x8x2_t p0=vzip_s8(r0,r1), p1=vzip_s8(r2,r3);
            int8x8x2_t p2=vzip_s8(r4,r5), p3=vzip_s8(r6,r7);
            int16x4x2_t q0=vzip_s16(vreinterpret_s16_s8(p0.val[0]),vreinterpret_s16_s8(p1.val[0]));
            int16x4x2_t q1=vzip_s16(vreinterpret_s16_s8(p2.val[0]),vreinterpret_s16_s8(p3.val[0]));
            int16x4x2_t q2=vzip_s16(vreinterpret_s16_s8(p0.val[1]),vreinterpret_s16_s8(p1.val[1]));
            int16x4x2_t q3=vzip_s16(vreinterpret_s16_s8(p2.val[1]),vreinterpret_s16_s8(p3.val[1]));
            int32x2x2_t v0=vzip_s32(vreinterpret_s32_s16(q0.val[0]),vreinterpret_s32_s16(q1.val[0]));
            int32x2x2_t v1=vzip_s32(vreinterpret_s32_s16(q0.val[1]),vreinterpret_s32_s16(q1.val[1]));
            int32x2x2_t v2=vzip_s32(vreinterpret_s32_s16(q2.val[0]),vreinterpret_s32_s16(q3.val[0]));
            int32x2x2_t v3=vzip_s32(vreinterpret_s32_s16(q2.val[1]),vreinterpret_s32_s16(q3.val[1]));
            vst1_s8(nchw+(c+0)*HW+hw, vreinterpret_s8_s32(v0.val[0]));
            vst1_s8(nchw+(c+1)*HW+hw, vreinterpret_s8_s32(v0.val[1]));
            vst1_s8(nchw+(c+2)*HW+hw, vreinterpret_s8_s32(v1.val[0]));
            vst1_s8(nchw+(c+3)*HW+hw, vreinterpret_s8_s32(v1.val[1]));
            vst1_s8(nchw+(c+4)*HW+hw, vreinterpret_s8_s32(v2.val[0]));
            vst1_s8(nchw+(c+5)*HW+hw, vreinterpret_s8_s32(v2.val[1]));
            vst1_s8(nchw+(c+6)*HW+hw, vreinterpret_s8_s32(v3.val[0]));
            vst1_s8(nchw+(c+7)*HW+hw, vreinterpret_s8_s32(v3.val[1]));
        }
        for (; hw < HW; ++hw)
            for (int cc = 0; cc < 8; ++cc)
                nchw[(c+cc)*HW+hw] = nhwc[hw*C+c+cc];
    }
    for (; c < C; ++c)
        for (int hw = 0; hw < HW; ++hw)
            nchw[c*HW+hw] = nhwc[hw*C+c];
#else
    for (int hw = 0; hw < HW; ++hw)
        for (int c = 0; c < C; ++c)
            nchw[c*HW+hw] = nhwc[hw*C+c];
#endif
}

static void nchw_to_nhwc_int8(const int8_t* nchw, int8_t* nhwc, int HW, int C)
{
#ifdef __ARM_NEON
    int c = 0;
    for (; c + 7 < C; c += 8) {
        int hw = 0;
        for (; hw + 7 < HW; hw += 8) {
            int8x8_t r0=vld1_s8(nchw+(c+0)*HW+hw), r1=vld1_s8(nchw+(c+1)*HW+hw);
            int8x8_t r2=vld1_s8(nchw+(c+2)*HW+hw), r3=vld1_s8(nchw+(c+3)*HW+hw);
            int8x8_t r4=vld1_s8(nchw+(c+4)*HW+hw), r5=vld1_s8(nchw+(c+5)*HW+hw);
            int8x8_t r6=vld1_s8(nchw+(c+6)*HW+hw), r7=vld1_s8(nchw+(c+7)*HW+hw);
            int8x8x2_t p0=vzip_s8(r0,r1), p1=vzip_s8(r2,r3);
            int8x8x2_t p2=vzip_s8(r4,r5), p3=vzip_s8(r6,r7);
            int16x4x2_t q0=vzip_s16(vreinterpret_s16_s8(p0.val[0]),vreinterpret_s16_s8(p1.val[0]));
            int16x4x2_t q1=vzip_s16(vreinterpret_s16_s8(p2.val[0]),vreinterpret_s16_s8(p3.val[0]));
            int16x4x2_t q2=vzip_s16(vreinterpret_s16_s8(p0.val[1]),vreinterpret_s16_s8(p1.val[1]));
            int16x4x2_t q3=vzip_s16(vreinterpret_s16_s8(p2.val[1]),vreinterpret_s16_s8(p3.val[1]));
            int32x2x2_t v0=vzip_s32(vreinterpret_s32_s16(q0.val[0]),vreinterpret_s32_s16(q1.val[0]));
            int32x2x2_t v1=vzip_s32(vreinterpret_s32_s16(q0.val[1]),vreinterpret_s32_s16(q1.val[1]));
            int32x2x2_t v2=vzip_s32(vreinterpret_s32_s16(q2.val[0]),vreinterpret_s32_s16(q3.val[0]));
            int32x2x2_t v3=vzip_s32(vreinterpret_s32_s16(q2.val[1]),vreinterpret_s32_s16(q3.val[1]));
            vst1_s8(nhwc+(hw+0)*C+c, vreinterpret_s8_s32(v0.val[0]));
            vst1_s8(nhwc+(hw+1)*C+c, vreinterpret_s8_s32(v0.val[1]));
            vst1_s8(nhwc+(hw+2)*C+c, vreinterpret_s8_s32(v1.val[0]));
            vst1_s8(nhwc+(hw+3)*C+c, vreinterpret_s8_s32(v1.val[1]));
            vst1_s8(nhwc+(hw+4)*C+c, vreinterpret_s8_s32(v2.val[0]));
            vst1_s8(nhwc+(hw+5)*C+c, vreinterpret_s8_s32(v2.val[1]));
            vst1_s8(nhwc+(hw+6)*C+c, vreinterpret_s8_s32(v3.val[0]));
            vst1_s8(nhwc+(hw+7)*C+c, vreinterpret_s8_s32(v3.val[1]));
        }
        for (; hw < HW; ++hw)
            for (int cc = 0; cc < 8; ++cc)
                nhwc[hw*C+c+cc] = nchw[(c+cc)*HW+hw];
    }
    for (; c < C; ++c)
        for (int hw = 0; hw < HW; ++hw)
            nhwc[hw*C+c] = nchw[c*HW+hw];
#else
    for (int hw = 0; hw < HW; ++hw)
        for (int c = 0; c < C; ++c)
            nhwc[hw*C+c] = nchw[c*HW+hw];
#endif
}

// ──────────────────────────────────────────────────────────────
// INT8 conv1d K-split multi-thread path.
// Each thread quantizes its own row slice, then all cooperate on the
// INT8 GEMM, then each thread dequantizes its own slice.
// ──────────────────────────────────────────────────────────────
void conv1d_int8_ksplit(
    const float*         A_base,          // padded input [T_padded, C_in_g]
    const int8_t* const* w_per_k_packed,  // [kernel_size] packed INT8 weights
    const int64_t*       eff_zeros,       // [C_out]
    const float*         req_scale,       // [C_out] dequant scales
    const float*         bias,            // [C_out] or nullptr
    float*               out_buf,         // [T_out, C_out]
    float*               tmp_buf,         // [T_out, C_out] scratch
    int8_t*              inp_scratch,     // [T_out, C_in_g]
    float*               row_scale_buf,   // [T_out]
    int T_out, int C_in_g, int C_out,
    int kernel_size, int dilation,
    int nthreads)
{
#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
#endif
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
        const int nT  = omp_get_num_threads();
#else
        const int tid = 0; const int nT = 1;
#endif
        const int t_s = (T_out * tid) / nT;
        const int t_e = (T_out * (tid + 1)) / nT;
        // Rows are independent: a dynamic split (row chunks handed out as threads finish) gives the same values.
        const bool dyn = kt_dynamic_chunks() > 0 && nT > 1;
        const int row_chunk = std::max(1, T_out / (nT * std::max(1, kt_dynamic_chunks())));

        for (int k = 0; k < kernel_size; ++k) {
            const float* src = A_base + (size_t)k * dilation * C_in_g;
            float*       dst = (k == 0) ? out_buf : tmp_buf;

            if (dyn) {
#ifdef _OPENMP
#pragma omp for schedule(dynamic, row_chunk)
#endif
                for (int t = 0; t < T_out; ++t)
                    quantize_row_fp32_to_int8(src + (size_t)t * C_in_g,
                                             inp_scratch + (size_t)t * C_in_g,
                                             row_scale_buf[t], C_in_g);
            } else {
            for (int t = t_s; t < t_e; ++t)
                quantize_row_fp32_to_int8(src + (size_t)t * C_in_g,
                                         inp_scratch + (size_t)t * C_in_g,
                                         row_scale_buf[t], C_in_g);
#ifdef _OPENMP
#pragma omp barrier
#endif
            }

            gemm_int8(inp_scratch, w_per_k_packed[k],
                      eff_zeros, nullptr, nullptr,
                      req_scale, 0, dst, /*is_float=*/true,
                      T_out, C_in_g, C_out,
                      /*nchw_out=*/false, /*in_parallel=*/true,
                      /*b_row_sums=*/nullptr, nullptr);
#ifdef _OPENMP
#pragma omp barrier
#endif

            if (dyn) {
#ifdef _OPENMP
#pragma omp for schedule(dynamic, row_chunk)
#endif
                for (int t = 0; t < T_out; ++t) {
                    if (k == 0)
                        dequant_bias_row_fp32(out_buf + (size_t)t * C_out, row_scale_buf[t], bias, C_out);
                    else
                        dequant_accum_row_fp32(out_buf + (size_t)t * C_out, dst + (size_t)t * C_out,
                                               row_scale_buf[t], C_out);
                }
            } else {
            if (k == 0) {
                for (int t = t_s; t < t_e; ++t)
                    dequant_bias_row_fp32(out_buf + (size_t)t * C_out,
                                         row_scale_buf[t], bias, C_out);
            } else {
                for (int t = t_s; t < t_e; ++t)
                    dequant_accum_row_fp32(out_buf + (size_t)t * C_out,
                                          dst     + (size_t)t * C_out,
                                          row_scale_buf[t], C_out);
            }
#ifdef _OPENMP
#pragma omp barrier
#endif
            }
        }
    }
}
