#include "../ops_neon.hpp"
#include "profile_internal.hpp"
#include <cstring>
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
// FP32 convolution kernels — conv2d_fp32_nhwc + Winograd F(2,3).
// ──────────────────────────────────────────────────────────────

// ──────────────────────────────────────────────────────────────
// conv2d_fp32_nhwc
//   input:    [H, W, C_in]          NHWC
//   weight:   [C_out, C_in, kH, kW] NCHW (only used when w_packed=nullptr)
//   bias:     [C_out]  (may be nullptr)
//   output:   [oH, oW, C_out]       NHWC
//   w_packed: pre-packed weights [Co_t, K, TILE] where K = kH*kW*C_in (NHWC-ordered)
//             If nullptr: pack on-the-fly from weight (fallback, not used in normal flow)
//   scratch_col: pre-allocated [oHW * K] floats  (nullptr → alloc internally)
// ──────────────────────────────────────────────────────────────
void conv2d_fp32_nhwc(
    const float* input,
    const float* weight,
    const float* bias,
    float*       output,
    int C_in, int H, int W,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    bool relu,
    const float* w_packed,
    float* scratch_col,
    int dilation_h, int dilation_w,
    StreamHandle /* stream */)
{
    const int oH  = (H + 2*pad_h - dilation_h*(kH-1) - 1) / stride_h + 1;
    const int oW  = (W + 2*pad_w - dilation_w*(kW-1) - 1) / stride_w + 1;
    const int oHW = oH * oW;
    const int K_col = kH * kW * C_in;   // im2col row width

    // 1×1 s=1 p=0: input NHWC [H, W, C_in] is already A [H*W, C_in] row-major
    const bool is_1x1_s1 = (kH == 1 && kW == 1 &&
                             pad_h == 0 && pad_w == 0 &&
                             stride_h == 1 && stride_w == 1 &&
                             dilation_h == 1 && dilation_w == 1);

    // ── Pack weights on-the-fly if not pre-packed ──────────────────────────
    float* w_packed_local = nullptr;
    const float* w_pack_ptr = w_packed;
    if (!w_pack_ptr) {
        if (kH > 1 || kW > 1) {
            // Reorder NCHW [C_out, C_in, kH, kW] → NHWC [C_out, kH, kW, C_in]
            std::vector<float> w_nhwc((size_t)C_out * K_col);
            for (int co = 0; co < C_out; ++co)
                for (int kh = 0; kh < kH; ++kh)
                    for (int kw = 0; kw < kW; ++kw)
                        for (int ci = 0; ci < C_in; ++ci)
                            w_nhwc[co * K_col + (kh * kW + kw) * C_in + ci] =
                                weight[co * K_col + ci * kH * kW + kh * kW + kw];
            w_packed_local = pack_weights_f32(w_nhwc.data(), C_out, K_col);
        } else {
            w_packed_local = pack_weights_f32(weight, C_out, K_col);
        }
        w_pack_ptr = w_packed_local;
    }

    // ── Allocate scratch im2col buffer if needed ───────────────────────────
    // Thread-local so the buffer persists across calls and only reallocates when
    // the required size exceeds the current capacity (amortised O(1) per call).
    static thread_local std::vector<float> col_alloc;
    float* col_ptr = scratch_col;
    if (!is_1x1_s1 && !col_ptr) {
        const size_t col_need = (size_t)oHW * K_col;
        if (col_alloc.size() < col_need) col_alloc.resize(col_need);
        col_ptr = col_alloc.data();
    }

    // ── im2col helper: fills col[oh*oW+ow, kh*kW*C_in + kw*C_in + ci] ──────
    auto do_im2col = [&](bool in_par) {
#ifdef _OPENMP
        if (in_par) {
#pragma omp for schedule(static)
            for (int oh = 0; oh < oH; ++oh) {
                for (int ow = 0; ow < oW; ++ow) {
                    float* dst = col_ptr + (oh * oW + ow) * K_col;
                    for (int kh = 0; kh < kH; ++kh) {
                        const int ih = oh * stride_h + kh * dilation_h - pad_h;
                        for (int kw_i = 0; kw_i < kW; ++kw_i) {
                            const int iw = ow * stride_w + kw_i * dilation_w - pad_w;
                            float* dkw = dst + (kh * kW + kw_i) * C_in;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                memcpy(dkw, input + (ih * W + iw) * C_in,
                                       C_in * sizeof(float));
                            else
                                memset(dkw, 0, C_in * sizeof(float));
                        }
                    }
                }
            }
            return;
        }
#endif
        for (int oh = 0; oh < oH; ++oh) {
            for (int ow = 0; ow < oW; ++ow) {
                float* dst = col_ptr + (oh * oW + ow) * K_col;
                for (int kh = 0; kh < kH; ++kh) {
                    const int ih = oh * stride_h + kh * dilation_h - pad_h;
                    for (int kw_i = 0; kw_i < kW; ++kw_i) {
                        const int iw = ow * stride_w + kw_i * dilation_w - pad_w;
                        float* dkw = dst + (kh * kW + kw_i) * C_in;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            memcpy(dkw, input + (ih * W + iw) * C_in,
                                   C_in * sizeof(float));
                        else
                            memset(dkw, 0, C_in * sizeof(float));
                    }
                }
            }
        }
        (void)in_par;
    };

    const float* A = is_1x1_s1 ? input : nullptr;

#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
    if (nthreads > 1) {
#pragma omp parallel num_threads(nthreads)
        {
            if (!is_1x1_s1)
                do_im2col(/*in_par=*/true);
            const float* a_ptr = is_1x1_s1 ? input : col_ptr;
            sgemm_f32(a_ptr, w_pack_ptr, bias, output,
                      relu, oHW, K_col, C_out, /*in_parallel=*/true, nullptr);
        }
    } else
#endif
    {
        if (!is_1x1_s1)
            do_im2col(/*in_par=*/false);
        A = is_1x1_s1 ? input : col_ptr;
        sgemm_f32(A, w_pack_ptr, bias, output,
                  relu, oHW, K_col, C_out, /*in_parallel=*/false, nullptr);
    }

    if (w_packed_local) free_packed_f32(w_packed_local);
}


// ──────────────────────────────────────────────────────────────
// conv2d_3x3s1_winograd_nhwc_fp32
//
//   FP32 Winograd F(2,3) convolution for 3×3 stride=1 pad=1 (NHWC).
//   Decomposes the direct conv GEMM (large B matrix, DRAM-bound) into
//   16 smaller GEMMs whose B matrices fit in L2 cache.
//
//   Transforms: B^T d B (input), G g G^T (weight, pre-computed),
//               A^T m A + bias (output).
//
//   B^T = [[1, 0,-1, 0],   G = [[1,   0,   0],    A^T = [[1, 1, 1, 0],
//          [0, 1, 1, 0],        [1/2, 1/2, 1/2],         [0, 1,-1,-1]]
//          [0,-1, 1, 0],        [1/2,-1/2, 1/2],
//          [0,-1, 0, 1]]        [0,   0,   1  ]]
//
//   Tile layout: Ph = ceil(H/2), Pw = ceil(W/2), P = Ph*Pw tiles.
//   Tile t=(th,tw) covers output [2*th:2*th+2][2*tw:2*tw+2].
//   Input patch (4×4): rows [2*th-1:2*th+3], cols [2*tw-1:2*tw+3] (zero-padded).
//
//   scratch: >= 16 * P * (C_in + C_out) floats
//     [input_hat:  16 segments of P*C_in]  offset 0
//     [output_hat: 16 segments of P*C_out] offset 16*P*C_in
// ──────────────────────────────────────────────────────────────
void conv2d_winograd_nhwc_fp32(
    const float* input,
    const float* const* w_wino_packed,
    const float* bias,
    float*       output,
    int C_in, int H, int W, int C_out, bool relu,
    float* scratch,
    StreamHandle /* stream */,
    const float* /* raw_weight */)  // ignored on CPU
{
    const int Ph = (H + 1) / 2;
    const int Pw = (W + 1) / 2;
    const int P  = Ph * Pw;

    float* input_hat  = scratch;
    float* output_hat = scratch + (size_t)16 * P * C_in;

    // ── Helpers: input-transform one tile, output-transform one tile ───────

    // Apply B^T × d × B to the 4×4 NHWC patch at tile (th, tw),
    // scattering C_in channels into input_hat[0..15][tile*C_in..].
    auto input_xform_tile = [&](int tile) {
        const int th = tile / Pw, tw = tile % Pw;
        const int ih0 = 2*th - 1, iw0 = 2*tw - 1;
        int ci = 0;
#ifdef __ARM_NEON
        for (; ci + 3 < C_in; ci += 4) {
            float32x4_t d[4][4];
            for (int r = 0; r < 4; ++r) {
                const int ih = ih0 + r;
                for (int c2 = 0; c2 < 4; ++c2) {
                    const int iw = iw0 + c2;
                    d[r][c2] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        ? vld1q_f32(input + ((size_t)ih*W + iw)*C_in + ci)
                        : vdupq_n_f32(0.f);
                }
            }
            // B^T × d: apply to each column j → tmp[:,j] = B^T × d[:,j]
            float32x4_t t[4][4];
            for (int c2 = 0; c2 < 4; ++c2) {
                t[0][c2] = vsubq_f32(d[0][c2], d[2][c2]);
                t[1][c2] = vaddq_f32(d[1][c2], d[2][c2]);
                t[2][c2] = vsubq_f32(d[2][c2], d[1][c2]);
                t[3][c2] = vsubq_f32(d[1][c2], d[3][c2]);
            }
            // tmp × B: apply to each row i → res[i,:] = tmp[i,:] × B
            float32x4_t res[4][4];
            for (int r = 0; r < 4; ++r) {
                res[r][0] = vsubq_f32(t[r][0], t[r][2]);
                res[r][1] = vaddq_f32(t[r][1], t[r][2]);
                res[r][2] = vsubq_f32(t[r][2], t[r][1]);
                res[r][3] = vsubq_f32(t[r][1], t[r][3]);
            }
            for (int pos = 0; pos < 16; ++pos)
                vst1q_f32(input_hat + (size_t)pos*P*C_in + (size_t)tile*C_in + ci,
                          res[pos >> 2][pos & 3]);
        }
#endif
        for (; ci < C_in; ++ci) {
            float d[4][4];
            for (int r = 0; r < 4; ++r)
                for (int c2 = 0; c2 < 4; ++c2) {
                    const int ih = ih0+r, iw = iw0+c2;
                    d[r][c2] = (ih>=0&&ih<H&&iw>=0&&iw<W)
                        ? input[((size_t)ih*W+iw)*C_in+ci] : 0.f;
                }
            float t[4][4], res[4][4];
            for (int c2=0;c2<4;++c2) {
                t[0][c2]=d[0][c2]-d[2][c2]; t[1][c2]=d[1][c2]+d[2][c2];
                t[2][c2]=d[2][c2]-d[1][c2]; t[3][c2]=d[1][c2]-d[3][c2];
            }
            for (int r=0;r<4;++r) {
                res[r][0]=t[r][0]-t[r][2]; res[r][1]=t[r][1]+t[r][2];
                res[r][2]=t[r][2]-t[r][1]; res[r][3]=t[r][1]-t[r][3];
            }
            for (int pos=0;pos<16;++pos)
                input_hat[(size_t)pos*P*C_in+(size_t)tile*C_in+ci] = res[pos>>2][pos&3];
        }
    };

    // Apply A^T × m × A + bias + relu to the 16 GEMM outputs for tile,
    // writing 2×2 output pixels to output NHWC.
    auto output_xform_tile = [&](int tile) {
        const int th = tile / Pw, tw = tile % Pw;
        int co = 0;
#ifdef __ARM_NEON
        const float32x4_t zero_v = vdupq_n_f32(0.f);
        for (; co + 3 < C_out; co += 4) {
            float32x4_t m[4][4];
            for (int r = 0; r < 4; ++r)
                for (int c2 = 0; c2 < 4; ++c2)
                    m[r][c2] = vld1q_f32(output_hat + (size_t)(r*4+c2)*P*C_out
                                         + (size_t)tile*C_out + co);
            // A^T × m: apply to each column → tmp[0..1][j]
            float32x4_t tmp[2][4];
            for (int c2=0;c2<4;++c2) {
                tmp[0][c2] = vaddq_f32(vaddq_f32(m[0][c2], m[1][c2]), m[2][c2]);
                tmp[1][c2] = vsubq_f32(vsubq_f32(m[1][c2], m[2][c2]), m[3][c2]);
            }
            // tmp × A: apply to each row
            float32x4_t res[2][2];
            for (int r=0;r<2;++r) {
                res[r][0] = vaddq_f32(vaddq_f32(tmp[r][0], tmp[r][1]), tmp[r][2]);
                res[r][1] = vsubq_f32(vsubq_f32(tmp[r][1], tmp[r][2]), tmp[r][3]);
            }
            float32x4_t bv = bias ? vld1q_f32(bias+co) : vdupq_n_f32(0.f);
            for (int r=0;r<2;++r) {
                const int oh = 2*th+r; if (oh >= H) continue;
                for (int c2=0;c2<2;++c2) {
                    const int ow = 2*tw+c2; if (ow >= W) continue;
                    float32x4_t val = vaddq_f32(res[r][c2], bv);
                    if (relu) val = vmaxq_f32(val, zero_v);
                    vst1q_f32(output + ((size_t)oh*W+ow)*C_out + co, val);
                }
            }
        }
#endif
        for (; co < C_out; ++co) {
            float m[4][4];
            for (int r=0;r<4;++r)
                for (int c2=0;c2<4;++c2)
                    m[r][c2] = output_hat[(size_t)(r*4+c2)*P*C_out+(size_t)tile*C_out+co];
            float tmp[2][4], res[2][2];
            for (int c2=0;c2<4;++c2) {
                tmp[0][c2]=m[0][c2]+m[1][c2]+m[2][c2];
                tmp[1][c2]=m[1][c2]-m[2][c2]-m[3][c2];
            }
            for (int r=0;r<2;++r) {
                res[r][0]=tmp[r][0]+tmp[r][1]+tmp[r][2];
                res[r][1]=tmp[r][1]-tmp[r][2]-tmp[r][3];
            }
            const float bv = bias ? bias[co] : 0.f;
            for (int r=0;r<2;++r) {
                const int oh=2*th+r; if(oh>=H) continue;
                for (int c2=0;c2<2;++c2) {
                    const int ow=2*tw+c2; if(ow>=W) continue;
                    const float val = res[r][c2] + bv;
                    output[((size_t)oh*W+ow)*C_out+co] = relu ? std::max(0.f, val) : val;
                }
            }
        }
    };

    // ── OMP parallel path ─────────────────────────────────────────────────
#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
    if (nthreads > 1) {
        double tp0 = now_ms(), tp1, tp2;
#pragma omp parallel num_threads(nthreads)
        {
            // Step 1: input transform (parallel over tiles, no barrier at end)
#pragma omp for schedule(static) nowait
            for (int tile = 0; tile < P; ++tile)
                input_xform_tile(tile);
#pragma omp barrier
#pragma omp single nowait
            tp1 = now_ms();

            // Step 2: distribute 16 GEMMs across threads — each thread owns
            // ceil(16/nthreads) pos values and computes each GEMM independently.
            // No inter-GEMM barriers; better B-matrix cache locality per thread.
#pragma omp for schedule(static) nowait
            for (int pos = 0; pos < 16; ++pos)
                sgemm_f32(input_hat  + (size_t)pos * P * C_in,
                          w_wino_packed[pos], nullptr,
                          output_hat + (size_t)pos * P * C_out,
                          false, P, C_in, C_out, /*in_parallel=*/false, nullptr);
#pragma omp barrier
#pragma omp single nowait
            tp2 = now_ms();

            // Step 3: output transform (parallel over tiles)
#pragma omp for schedule(static)
            for (int tile = 0; tile < P; ++tile)
                output_xform_tile(tile);
            // implicit barrier at end of omp for
        }
        double tp3 = now_ms();
        g_wino_transform_ms += tp1 - tp0;
        g_wino_gemm_ms      += tp2 - tp1;
        g_wino_output_ms    += tp3 - tp2;
        return;
    }
#endif

    // ── Single-threaded path ─────────────────────────────────────────────
    {
        double t0 = now_ms();
        for (int tile = 0; tile < P; ++tile)
            input_xform_tile(tile);
        double t1 = now_ms();
        for (int pos = 0; pos < 16; ++pos)
            sgemm_f32(input_hat  + (size_t)pos * P * C_in,
                      w_wino_packed[pos], nullptr,
                      output_hat + (size_t)pos * P * C_out,
                      false, P, C_in, C_out, /*in_parallel=*/false, nullptr);
        double t2 = now_ms();
        for (int tile = 0; tile < P; ++tile)
            output_xform_tile(tile);
        double t3 = now_ms();
        g_wino_transform_ms += t1 - t0;
        g_wino_gemm_ms      += t2 - t1;
        g_wino_output_ms    += t3 - t2;
    }
}
