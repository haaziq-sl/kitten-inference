#include "../ops_neon.hpp"
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

// ──────────────────────────────────────────────────────────────
// add_fp32 — vectorised element-wise add with optional ReLU
// ──────────────────────────────────────────────────────────────
void add_fp32(
    const float* in1,
    const float* in2,
    float*       output,
    int N,
    bool relu,
    StreamHandle /* stream */)
{
#ifdef __ARM_NEON
    const float32x4_t zero_v = vdupq_n_f32(0.f);
    int i = 0;
    for (; i + 3 < N; i += 4) {
        float32x4_t v = vaddq_f32(vld1q_f32(in1 + i), vld1q_f32(in2 + i));
        if (relu) v = vmaxq_f32(v, zero_v);
        vst1q_f32(output + i, v);
    }
    for (; i < N; ++i) {
        float v = in1[i] + in2[i];
        output[i] = (relu && v < 0.f) ? 0.f : v;
    }
#elif defined(__AVX512F__)
    const __m512 zero_v = _mm512_setzero_ps();
    int i = 0;
    for (; i + 15 < N; i += 16) {
        __m512 v = _mm512_add_ps(_mm512_loadu_ps(in1 + i), _mm512_loadu_ps(in2 + i));
        if (relu) v = _mm512_max_ps(v, zero_v);
        _mm512_storeu_ps(output + i, v);
    }
    for (; i < N; ++i) {
        float v = in1[i] + in2[i];
        output[i] = (relu && v < 0.f) ? 0.f : v;
    }
#elif defined(__AVX2__)
    const __m256 zero_v = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < N; i += 8) {
        __m256 v = _mm256_add_ps(_mm256_loadu_ps(in1 + i), _mm256_loadu_ps(in2 + i));
        if (relu) v = _mm256_max_ps(v, zero_v);
        _mm256_storeu_ps(output + i, v);
    }
    for (; i < N; ++i) {
        float v = in1[i] + in2[i];
        output[i] = (relu && v < 0.f) ? 0.f : v;
    }
#else
    for (int i = 0; i < N; ++i) {
        float v = in1[i] + in2[i];
        output[i] = (relu && v < 0.f) ? 0.f : v;
    }
#endif
}
