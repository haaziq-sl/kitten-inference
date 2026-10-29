// activations.cpp — Elementwise activation functions (FP32)

#include "../ops_neon.hpp"
#include "profile_internal.hpp"
#include <cmath>
#include <algorithm>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

// ─────────────────────────────────────────────────────────────────
// Leaky ReLU
// ─────────────────────────────────────────────────────────────────

void leaky_relu_fp32(const float* in, float* out, int N, float alpha, StreamHandle /* stream */)
{
#ifdef __ARM_NEON
    const float32x4_t va = vdupq_n_f32(alpha);
    const float32x4_t vz = vdupq_n_f32(0.f);
    int i = 0;
    for (; i <= N - 4; i += 4) {
        float32x4_t x = vld1q_f32(in + i);
        vst1q_f32(out + i, vbslq_f32(vcgeq_f32(x, vz), x, vmulq_f32(x, va)));
    }
    for (; i < N; ++i) out[i] = in[i] >= 0.f ? in[i] : alpha * in[i];
#elif defined(__AVX512F__)
    const __m512 va = _mm512_set1_ps(alpha);
    const __m512 vz = _mm512_setzero_ps();
    int i = 0;
    for (; i + 15 < N; i += 16) {
        __m512 x = _mm512_loadu_ps(in + i);
        // where x >= 0: keep x; else: alpha*x
        __mmask16 mask = _mm512_cmp_ps_mask(x, vz, _CMP_GE_OS);
        _mm512_storeu_ps(out + i, _mm512_mask_blend_ps(mask, _mm512_mul_ps(x, va), x));
    }
    for (; i < N; ++i) out[i] = in[i] >= 0.f ? in[i] : alpha * in[i];
#elif defined(__AVX2__)
    const __m256 va = _mm256_set1_ps(alpha);
    const __m256 vz = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < N; i += 8) {
        __m256 x    = _mm256_loadu_ps(in + i);
        __m256 neg  = _mm256_mul_ps(x, va);
        __m256 mask = _mm256_cmp_ps(x, vz, _CMP_GE_OS);
        _mm256_storeu_ps(out + i, _mm256_blendv_ps(neg, x, mask));
    }
    for (; i < N; ++i) out[i] = in[i] >= 0.f ? in[i] : alpha * in[i];
#else
    for (int i = 0; i < N; ++i) out[i] = in[i] >= 0.f ? in[i] : alpha * in[i];
#endif
}

// ─────────────────────────────────────────────────────────────────
// Elementwise exp / sin / sigmoid / tanh
// ─────────────────────────────────────────────────────────────────

void exp_fp32(const float* in, float* out, int N, StreamHandle /* stream */)
{
#ifdef __ARM_NEON
    int i = 0;
    for (; i + 4 <= N; i += 4)
        vst1q_f32(out + i, vexpq_f32_fast(vld1q_f32(in + i)));
    for (; i < N; ++i) out[i] = expf(in[i]);
#elif defined(__AVX512F__)
    int i = 0;
    for (; i + 16 <= N; i += 16)
        _mm512_storeu_ps(out + i, exp512_ps(_mm512_loadu_ps(in + i)));
    for (; i < N; ++i) out[i] = expf(in[i]);
#elif defined(__AVX2__)
    int i = 0;
    for (; i + 8 <= N; i += 8)
        _mm256_storeu_ps(out + i, exp256_ps(_mm256_loadu_ps(in + i)));
    for (; i < N; ++i) out[i] = expf(in[i]);
#else
    for (int i = 0; i < N; ++i) out[i] = expf(in[i]);
#endif
}

void sin_fp32(const float* in, float* out, int N, StreamHandle /* stream */)
{
#ifdef __ARM_NEON
    int i = 0;
    for (; i + 4 <= N; i += 4)
        vst1q_f32(out + i, vsinq_f32_fast(vld1q_f32(in + i)));
    for (; i < N; ++i) out[i] = sinf(in[i]);
#elif defined(__AVX512F__)
    int i = 0;
    for (; i + 16 <= N; i += 16)
        _mm512_storeu_ps(out + i, sin512_ps(_mm512_loadu_ps(in + i)));
    for (; i < N; ++i) out[i] = sinf(in[i]);
#elif defined(__AVX2__)
    int i = 0;
    for (; i + 8 <= N; i += 8)
        _mm256_storeu_ps(out + i, sin256_ps(_mm256_loadu_ps(in + i)));
    for (; i < N; ++i) out[i] = sinf(in[i]);
#else
    for (int i = 0; i < N; ++i) out[i] = sinf(in[i]);
#endif
}

void sigmoid_fp32(const float* in, float* out, int N, StreamHandle /* stream */)
{
#ifdef __ARM_NEON
    int i = 0;
    for (; i + 4 <= N; i += 4)
        vst1q_f32(out + i, vsigmoidq_f32(vld1q_f32(in + i)));
    for (; i < N; ++i) out[i] = 1.f / (1.f + expf(-in[i]));
#elif defined(__AVX512F__)
    int i = 0;
    for (; i + 16 <= N; i += 16)
        _mm512_storeu_ps(out + i, sigmoid512_ps(_mm512_loadu_ps(in + i)));
    for (; i < N; ++i) out[i] = 1.f / (1.f + expf(-in[i]));
#elif defined(__AVX2__)
    int i = 0;
    for (; i + 8 <= N; i += 8)
        _mm256_storeu_ps(out + i, sigmoid256_ps(_mm256_loadu_ps(in + i)));
    for (; i < N; ++i) out[i] = 1.f / (1.f + expf(-in[i]));
#else
    for (int i = 0; i < N; ++i) out[i] = 1.f / (1.f + expf(-in[i]));
#endif
}

void tanh_fp32(const float* in, float* out, int N, StreamHandle /* stream */)
{
#ifdef __ARM_NEON
    int i = 0;
    for (; i + 4 <= N; i += 4)
        vst1q_f32(out + i, vtanhq_f32(vld1q_f32(in + i)));
    for (; i < N; ++i) out[i] = tanhf(in[i]);
#elif defined(__AVX512F__)
    int i = 0;
    for (; i + 16 <= N; i += 16)
        _mm512_storeu_ps(out + i, tanh512_ps(_mm512_loadu_ps(in + i)));
    for (; i < N; ++i) out[i] = tanhf(in[i]);
#elif defined(__AVX2__)
    int i = 0;
    for (; i + 8 <= N; i += 8)
        _mm256_storeu_ps(out + i, tanh256_ps(_mm256_loadu_ps(in + i)));
    for (; i < N; ++i) out[i] = tanhf(in[i]);
#else
    for (int i = 0; i < N; ++i) out[i] = tanhf(in[i]);
#endif
}

// ─────────────────────────────────────────────────────────────────
// In-place ReLU
// ─────────────────────────────────────────────────────────────────

void relu_fp32(float* inout, int N, StreamHandle /* stream */)
{
#ifdef __ARM_NEON
    const float32x4_t zero = vdupq_n_f32(0.f);
    int i = 0;
    for (; i + 4 <= N; i += 4)
        vst1q_f32(inout + i, vmaxq_f32(vld1q_f32(inout + i), zero));
    for (; i < N; ++i) inout[i] = std::max(inout[i], 0.f);
#elif defined(__AVX512F__)
    const __m512 zero = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= N; i += 16)
        _mm512_storeu_ps(inout + i, _mm512_max_ps(_mm512_loadu_ps(inout + i), zero));
    for (; i < N; ++i) inout[i] = std::max(inout[i], 0.f);
#elif defined(__AVX2__)
    const __m256 zero = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= N; i += 8)
        _mm256_storeu_ps(inout + i, _mm256_max_ps(_mm256_loadu_ps(inout + i), zero));
    for (; i < N; ++i) inout[i] = std::max(inout[i], 0.f);
#else
    for (int i = 0; i < N; ++i) inout[i] = std::max(inout[i], 0.f);
#endif
}

// ─────────────────────────────────────────────────────────────────
// Scale: out[i] = in[i] * scalar
// ─────────────────────────────────────────────────────────────────

void scale_fp32(const float* in, float* out, size_t N, float scalar, StreamHandle /* stream */)
{
#ifdef __ARM_NEON
    float32x4_t vs = vdupq_n_f32(scalar);
    size_t i = 0;
    for (; i + 4 <= N; i += 4)
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(in + i), vs));
    for (; i < N; ++i) out[i] = in[i] * scalar;
#elif defined(__AVX512F__)
    const __m512 vs = _mm512_set1_ps(scalar);
    size_t i = 0;
    for (; i + 16 <= N; i += 16)
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(in + i), vs));
    for (; i < N; ++i) out[i] = in[i] * scalar;
#elif defined(__AVX2__)
    const __m256 vs = _mm256_set1_ps(scalar);
    size_t i = 0;
    for (; i + 8 <= N; i += 8)
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(in + i), vs));
    for (; i < N; ++i) out[i] = in[i] * scalar;
#else
    for (size_t i = 0; i < N; ++i) out[i] = in[i] * scalar;
#endif
}

// ─────────────────────────────────────────────────────────────────
// Snake1D: out = x + sin²(α·x) / α  (OMP + NEON)
// ─────────────────────────────────────────────────────────────────

void snake1d_fp32(const float* in, float* out, int T, int C,
                  const float* alpha, const float* inv_alpha)
{
    const bool _profile_snake = g_ops_profile_enabled;
    const double _t0_snake = _profile_snake ? now_ms() : 0.0;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
#endif
    for (int t = 0; t < T; ++t) {
        const float* xi = in  + (size_t)t * C;
        float*       yo = out + (size_t)t * C;
        int c = 0;
#ifdef __ARM_NEON
        for (; c + 4 <= C; c += 4) {
            float32x4_t vx  = vld1q_f32(xi + c);
            float32x4_t va  = alpha     ? vld1q_f32(alpha     + c) : vdupq_n_f32(1.f);
            float32x4_t via = inv_alpha ? vld1q_f32(inv_alpha + c) : vdupq_n_f32(1.f);
            float32x4_t s   = vsinq_f32_fast(vmulq_f32(va, vx));
            vst1q_f32(yo + c, vfmaq_f32(vx, vmulq_f32(s, s), via));
        }
#elif defined(__AVX512F__)
        {
            const __m512 one512 = _mm512_set1_ps(1.f);
            for (; c + 16 <= C; c += 16) {
                __m512 vx  = _mm512_loadu_ps(xi + c);
                __m512 va  = alpha     ? _mm512_loadu_ps(alpha     + c) : one512;
                __m512 via = inv_alpha ? _mm512_loadu_ps(inv_alpha + c) : one512;
                __m512 s   = sin512_ps(_mm512_mul_ps(va, vx));
                _mm512_storeu_ps(yo + c, _mm512_fmadd_ps(_mm512_mul_ps(s, s), via, vx));
            }
        }
#elif defined(__AVX2__)
        {
            const __m256 one256 = _mm256_set1_ps(1.f);
            for (; c + 8 <= C; c += 8) {
                __m256 vx  = _mm256_loadu_ps(xi + c);
                __m256 va  = alpha     ? _mm256_loadu_ps(alpha     + c) : one256;
                __m256 via = inv_alpha ? _mm256_loadu_ps(inv_alpha + c) : one256;
                __m256 s   = sin256_ps(_mm256_mul_ps(va, vx));
                _mm256_storeu_ps(yo + c, _mm256_fmadd_ps(_mm256_mul_ps(s, s), via, vx));
            }
        }
#endif
        for (; c < C; ++c) {
            float x  = xi[c];
            float ai = alpha ? alpha[c] : 1.f;
            float s  = sinf(ai * x);
            yo[c] = x + s * s * (inv_alpha ? inv_alpha[c] : 1.f / (ai + 1e-9f));
        }
    }
    if (_profile_snake) g_snake1d_ms += now_ms() - _t0_snake;
}

// ─────────────────────────────────────────────────────────────────
// GELU — tanh approximation, NEON-vectorised
//   GELU(x) = x · (1 − 1/(exp(2c·(x + k·x³)) + 1))
//   where c = sqrt(2/π), k = 0.044715
// ─────────────────────────────────────────────────────────────────

void gelu_fp32(float* inout, int N, StreamHandle /* stream */)
{
    // GELU(x) = x * sigmoid(two_c*(x + k*x^3))
    // where two_c = 2*sqrt(2/π) = 1.5957691..., k = 0.044715
    static const float two_c = 1.5957691216057308f;
    static const float k_val = 0.044715f;
#ifdef __ARM_NEON
    int i = 0;
    for (; i <= N - 16; i += 16) {
        #define GELU4(off) { \
            float* p = inout + i + (off); \
            float32x4_t v   = vld1q_f32(p); \
            float32x4_t v3  = vmulq_f32(vmulq_f32(v, v), v); \
            float32x4_t u   = vmulq_f32(vdupq_n_f32(two_c), vfmaq_f32(v, vdupq_n_f32(k_val), v3)); \
            float32x4_t e   = vexpq_f32_fast(u); \
            float32x4_t ep1 = vaddq_f32(e, vdupq_n_f32(1.0f)); \
            float32x4_t rec = vrecpeq_f32(ep1); \
            rec = vmulq_f32(rec, vrecpsq_f32(ep1, rec)); \
            rec = vmulq_f32(rec, vrecpsq_f32(ep1, rec)); \
            vst1q_f32(p, vmulq_f32(v, vsubq_f32(vdupq_n_f32(1.0f), rec))); }
        GELU4(0) GELU4(4) GELU4(8) GELU4(12)
        #undef GELU4
    }
    for (; i <= N - 4; i += 4) {
        float32x4_t v   = vld1q_f32(inout + i);
        float32x4_t v3  = vmulq_f32(vmulq_f32(v, v), v);
        float32x4_t u   = vmulq_f32(vdupq_n_f32(two_c), vfmaq_f32(v, vdupq_n_f32(k_val), v3));
        float32x4_t e   = vexpq_f32_fast(u);
        float32x4_t ep1 = vaddq_f32(e, vdupq_n_f32(1.0f));
        float32x4_t rec = vrecpeq_f32(ep1);
        rec = vmulq_f32(rec, vrecpsq_f32(ep1, rec));
        rec = vmulq_f32(rec, vrecpsq_f32(ep1, rec));
        vst1q_f32(inout + i, vmulq_f32(v, vsubq_f32(vdupq_n_f32(1.0f), rec)));
    }
    for (; i < N; ++i) {
        float x = inout[i];
        float t = tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x));
        inout[i] = 0.5f * x * (1.0f + t);
    }
#elif defined(__AVX512F__)
    {
        const __m512 vtwo_c = _mm512_set1_ps(two_c);
        const __m512 vk     = _mm512_set1_ps(k_val);
        const __m512 vone   = _mm512_set1_ps(1.0f);
        int i = 0;
        for (; i + 16 <= N; i += 16) {
            __m512 v   = _mm512_loadu_ps(inout + i);
            __m512 v3  = _mm512_mul_ps(_mm512_mul_ps(v, v), v);
            __m512 u   = _mm512_mul_ps(vtwo_c, _mm512_fmadd_ps(vk, v3, v));
            __m512 e   = exp512_ps(u);
            __m512 ep1 = _mm512_add_ps(e, vone);
            __m512 r   = _mm512_rcp14_ps(ep1);
            r = _mm512_mul_ps(r, _mm512_fnmadd_ps(ep1, r, _mm512_set1_ps(2.0f)));
            r = _mm512_mul_ps(r, _mm512_fnmadd_ps(ep1, r, _mm512_set1_ps(2.0f)));
            _mm512_storeu_ps(inout + i, _mm512_mul_ps(v, _mm512_sub_ps(vone, r)));
        }
        for (; i < N; ++i) {
            float x = inout[i];
            float t = tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x));
            inout[i] = 0.5f * x * (1.0f + t);
        }
    }
#elif defined(__AVX2__)
    {
        const __m256 vtwo_c = _mm256_set1_ps(two_c);
        const __m256 vk     = _mm256_set1_ps(k_val);
        const __m256 vone   = _mm256_set1_ps(1.0f);
        const __m256 vtwo   = _mm256_set1_ps(2.0f);
        int i = 0;
        for (; i + 8 <= N; i += 8) {
            __m256 v   = _mm256_loadu_ps(inout + i);
            __m256 v3  = _mm256_mul_ps(_mm256_mul_ps(v, v), v);
            __m256 u   = _mm256_mul_ps(vtwo_c, _mm256_fmadd_ps(vk, v3, v));
            __m256 e   = exp256_ps(u);
            __m256 ep1 = _mm256_add_ps(e, vone);
            __m256 r   = _mm256_rcp_ps(ep1);
            r = _mm256_mul_ps(r, _mm256_fnmadd_ps(ep1, r, vtwo));
            r = _mm256_mul_ps(r, _mm256_fnmadd_ps(ep1, r, vtwo));
            _mm256_storeu_ps(inout + i, _mm256_mul_ps(v, _mm256_sub_ps(vone, r)));
        }
        for (; i < N; ++i) {
            float x = inout[i];
            float t = tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x));
            inout[i] = 0.5f * x * (1.0f + t);
        }
    }
#else
    for (int i = 0; i < N; ++i) {
        float x = inout[i];
        float t = tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x));
        inout[i] = 0.5f * x * (1.0f + t);
    }
#endif
}

// ─────────────────────────────────────────────────────────────────
// SiLU: out[i] = in[i] * sigmoid(in[i])
// ─────────────────────────────────────────────────────────────────

void silu_fp32(const float* in, float* out, int N, StreamHandle /* stream */)
{
#ifdef __ARM_NEON
    int i = 0;
    for (; i + 4 <= N; i += 4) {
        float32x4_t x    = vld1q_f32(in + i);
        float32x4_t sig  = vsigmoidq_f32(x);
        vst1q_f32(out + i, vmulq_f32(x, sig));
    }
    for (; i < N; ++i) { float x = in[i]; out[i] = x / (1.f + expf(-x)); }
#elif defined(__AVX512F__)
    int i = 0;
    for (; i + 16 <= N; i += 16) {
        __m512 x    = _mm512_loadu_ps(in + i);
        __m512 sig  = sigmoid512_ps(x);
        _mm512_storeu_ps(out + i, _mm512_mul_ps(x, sig));
    }
    for (; i < N; ++i) { float x = in[i]; out[i] = x / (1.f + expf(-x)); }
#elif defined(__AVX2__)
    int i = 0;
    for (; i + 8 <= N; i += 8) {
        __m256 x    = _mm256_loadu_ps(in + i);
        __m256 sig  = sigmoid256_ps(x);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(x, sig));
    }
    for (; i < N; ++i) { float x = in[i]; out[i] = x / (1.f + expf(-x)); }
#else
    for (int i = 0; i < N; ++i) { float x = in[i]; out[i] = x / (1.f + expf(-x)); }
#endif
}

// ─────────────────────────────────────────────────────────────────
// Elementwise multiply: out[i] = a[i] * b[i]
// ─────────────────────────────────────────────────────────────────

void elemwise_mul_fp32(const float* a, const float* b, float* out, int N, StreamHandle /* stream */)
{
#ifdef __ARM_NEON
    int i = 0;
    for (; i + 4 <= N; i += 4)
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < N; ++i) out[i] = a[i] * b[i];
#elif defined(__AVX512F__)
    int i = 0;
    for (; i + 16 <= N; i += 16)
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    for (; i < N; ++i) out[i] = a[i] * b[i];
#elif defined(__AVX2__)
    int i = 0;
    for (; i + 8 <= N; i += 8)
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < N; ++i) out[i] = a[i] * b[i];
#else
    for (int i = 0; i < N; ++i) out[i] = a[i] * b[i];
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// dynamic_quantize_dequant_f32
// Simulates ONNX DynamicQuantizeLinear: quantizes to uint8 (asymmetric),
// then dequantizes back to fp32. Introduces the same rounding as ONNX.
// ─────────────────────────────────────────────────────────────────────────────
void dynamic_quantize_dequant_f32(float* x, int n) {
    if (n <= 0) return;
    float x_min = x[0], x_max = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < x_min) x_min = x[i];
        if (x[i] > x_max) x_max = x[i];
    }
    const float range = x_max - x_min;
    const float scale = (range > 0.f) ? (range / 255.f) : 1e-8f;
    // zero_point = clip(round(-x_min / scale), 0, 255)
    float zp_f = -x_min / scale;
    if (zp_f < 0.f)   zp_f = 0.f;
    if (zp_f > 255.f) zp_f = 255.f;
    const float zp = roundf(zp_f);
    // x[i] = (clip(round(x[i] / scale) + zp, 0, 255) - zp) * scale
    for (int i = 0; i < n; ++i) {
        float q = roundf(x[i] / scale) + zp;
        if (q < 0.f)   q = 0.f;
        if (q > 255.f) q = 255.f;
        x[i] = (q - zp) * scale;
    }
}
