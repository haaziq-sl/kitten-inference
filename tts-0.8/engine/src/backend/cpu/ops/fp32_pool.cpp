#include "../ops_neon.hpp"
#include <cstring>
#include <limits>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

// ──────────────────────────────────────────────────────────────
// maxpool_fp32_nhwc — vectorised over C
//   input:  [H, W, C]  NHWC
//   output: [oH, oW, C] NHWC
// ──────────────────────────────────────────────────────────────
void maxpool_fp32_nhwc(
    const float* input,
    float*       output,
    int C, int H, int W,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    StreamHandle /* stream */)
{
    const int oH = (H + 2*pad_h - kH) / stride_h + 1;
    const int oW = (W + 2*pad_w - kW) / stride_w + 1;
    const float NEG_INF = -std::numeric_limits<float>::infinity();

    for (int oh = 0; oh < oH; ++oh) {
        for (int ow = 0; ow < oW; ++ow) {
            float* out_ptr = output + (oh * oW + ow) * C;

            // Initialise to -inf
            for (int c = 0; c < C; ++c) out_ptr[c] = NEG_INF;

            for (int kh = 0; kh < kH; ++kh) {
                const int ih = oh * stride_h + kh - pad_h;
                if (ih < 0 || ih >= H) continue;
                for (int kw_i = 0; kw_i < kW; ++kw_i) {
                    const int iw = ow * stride_w + kw_i - pad_w;
                    if (iw < 0 || iw >= W) continue;
                    const float* in_ptr = input + (ih * W + iw) * C;
                    int c = 0;
#ifdef __ARM_NEON
                    for (; c + 3 < C; c += 4)
                        vst1q_f32(out_ptr + c,
                                  vmaxq_f32(vld1q_f32(out_ptr + c),
                                            vld1q_f32(in_ptr + c)));
#elif defined(__AVX512F__)
                    for (; c + 15 < C; c += 16)
                        _mm512_storeu_ps(out_ptr + c,
                            _mm512_max_ps(_mm512_loadu_ps(out_ptr + c),
                                         _mm512_loadu_ps(in_ptr + c)));
#elif defined(__AVX2__)
                    for (; c + 7 < C; c += 8)
                        _mm256_storeu_ps(out_ptr + c,
                            _mm256_max_ps(_mm256_loadu_ps(out_ptr + c),
                                         _mm256_loadu_ps(in_ptr + c)));
#endif
                    for (; c < C; ++c)
                        if (in_ptr[c] > out_ptr[c]) out_ptr[c] = in_ptr[c];
                }
            }
        }
    }
}


// ──────────────────────────────────────────────────────────────
// avgpool_global_fp32_nhwc — global average pool, vectorised
//   input:  [H, W, C]  NHWC
//   output: [C]
// ──────────────────────────────────────────────────────────────
void avgpool_global_fp32_nhwc(
    const float* input,
    float*       output,
    int C, int H, int W,
    StreamHandle /* stream */)
{
    const float scale = 1.0f / (float)(H * W);

    memset(output, 0, (size_t)C * sizeof(float));

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            const float* in_ptr = input + (h * W + w) * C;
            int c = 0;
#ifdef __ARM_NEON
            for (; c + 3 < C; c += 4)
                vst1q_f32(output + c,
                          vaddq_f32(vld1q_f32(output + c), vld1q_f32(in_ptr + c)));
#elif defined(__AVX512F__)
            for (; c + 15 < C; c += 16)
                _mm512_storeu_ps(output + c,
                    _mm512_add_ps(_mm512_loadu_ps(output + c),
                                  _mm512_loadu_ps(in_ptr + c)));
#elif defined(__AVX2__)
            for (; c + 7 < C; c += 8)
                _mm256_storeu_ps(output + c,
                    _mm256_add_ps(_mm256_loadu_ps(output + c),
                                  _mm256_loadu_ps(in_ptr + c)));
#endif
            for (; c < C; ++c)
                output[c] += in_ptr[c];
        }
    }

    // Multiply by 1/(H*W)
    int c = 0;
#ifdef __ARM_NEON
    const float32x4_t scale_v = vdupq_n_f32(scale);
    for (; c + 3 < C; c += 4)
        vst1q_f32(output + c, vmulq_f32(vld1q_f32(output + c), scale_v));
#elif defined(__AVX512F__)
    const __m512 scale_v = _mm512_set1_ps(scale);
    for (; c + 15 < C; c += 16)
        _mm512_storeu_ps(output + c,
                         _mm512_mul_ps(_mm512_loadu_ps(output + c), scale_v));
#elif defined(__AVX2__)
    const __m256 scale_v = _mm256_set1_ps(scale);
    for (; c + 7 < C; c += 8)
        _mm256_storeu_ps(output + c,
                         _mm256_mul_ps(_mm256_loadu_ps(output + c), scale_v));
#endif
    for (; c < C; ++c)
        output[c] *= scale;
}
