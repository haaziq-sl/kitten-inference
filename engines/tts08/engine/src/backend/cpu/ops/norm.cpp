// norm.cpp — Normalization kernels (FP32): LayerNorm, RMSNorm, AdaIN, AdaLayerNorm

#include "../ops_neon.hpp"
#include "profile_internal.hpp"
#include <cmath>
#include <vector>
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
// RMS Normalization  [N, C] → [N, C]
//   y = x / sqrt(mean(x^2) + eps) * gamma
//   No mean subtraction (faster and equally effective for LLMs).
// ─────────────────────────────────────────────────────────────────

void rmsnorm_fp32(const float* input, const float* gamma, float* output,
                  int N, int C, float eps, StreamHandle /* stream */)
{
    const double t0_rms = now_ms();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(!omp_in_parallel())
#endif
    for (int n = 0; n < N; ++n) {
        const float* x = input  + (size_t)n * C;
        float*       y = output + (size_t)n * C;

        // Sum of squares
        float ss = 0.0f;
        int c = 0;
#ifdef __ARM_NEON
        { float32x4_t vs = vdupq_n_f32(0.0f);
          for (; c <= C - 4; c += 4) { float32x4_t v = vld1q_f32(x + c); vs = vfmaq_f32(vs, v, v); }
          ss = vaddvq_f32(vs); }
#elif defined(__AVX512F__)
        { __m512 vs = _mm512_setzero_ps();
          for (; c + 16 <= C; c += 16) { __m512 v = _mm512_loadu_ps(x + c); vs = _mm512_fmadd_ps(v, v, vs); }
          ss = hsum512(vs); }
#elif defined(__AVX2__)
        { __m256 vs = _mm256_setzero_ps();
          for (; c + 8 <= C; c += 8) { __m256 v = _mm256_loadu_ps(x + c); vs = _mm256_fmadd_ps(v, v, vs); }
          ss = hsum256(vs); }
#endif
        for (; c < C; ++c) ss += x[c] * x[c];

        float inv_rms = 1.0f / sqrtf(ss / (float)C + eps);

        c = 0;
#ifdef __ARM_NEON
        { float32x4_t vr = vdupq_n_f32(inv_rms);
          for (; c <= C - 4; c += 4)
              vst1q_f32(y + c, vmulq_f32(vmulq_f32(vld1q_f32(x + c), vr), vld1q_f32(gamma + c))); }
#elif defined(__AVX512F__)
        { __m512 vr = _mm512_set1_ps(inv_rms);
          for (; c + 16 <= C; c += 16)
              _mm512_storeu_ps(y + c, _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(x + c), vr),
                                                     _mm512_loadu_ps(gamma + c))); }
#elif defined(__AVX2__)
        { __m256 vr = _mm256_set1_ps(inv_rms);
          for (; c + 8 <= C; c += 8)
              _mm256_storeu_ps(y + c, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + c), vr),
                                                     _mm256_loadu_ps(gamma + c))); }
#endif
        for (; c < C; ++c) y[c] = x[c] * inv_rms * gamma[c];
    }
    g_lm_rmsnorm_ms += now_ms() - t0_rms;
}

// ─────────────────────────────────────────────────────────────────
// Layer Normalization  [N, C] → [N, C]
// ─────────────────────────────────────────────────────────────────

void layernorm_fp32(const float* input, const float* gamma, const float* beta,
                    float* output, int N, int C, float eps, StreamHandle /* stream */)
{
    const bool _profile_layernorm = g_ops_profile_enabled;
    const double _t0_layernorm = _profile_layernorm ? now_ms() : 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(!omp_in_parallel())
#endif
    for (int n = 0; n < N; ++n) {
        const float* x = input  + (size_t)n * C;
        float*       y = output + (size_t)n * C;

        float mean = 0.0f;
        int c = 0;
#ifdef __ARM_NEON
        { float32x4_t vs = vdupq_n_f32(0.0f);
          for (; c <= C - 4; c += 4) vs = vaddq_f32(vs, vld1q_f32(x + c));
          mean = vaddvq_f32(vs); }
#elif defined(__AVX512F__)
        { __m512 vs = _mm512_setzero_ps();
          for (; c + 16 <= C; c += 16) vs = _mm512_add_ps(vs, _mm512_loadu_ps(x + c));
          mean = hsum512(vs); }
#elif defined(__AVX2__)
        { __m256 vs = _mm256_setzero_ps();
          for (; c + 8 <= C; c += 8) vs = _mm256_add_ps(vs, _mm256_loadu_ps(x + c));
          mean = hsum256(vs); }
#endif
        for (; c < C; ++c) mean += x[c];
        mean /= (float)C;

        float var = 0.0f;
        c = 0;
#ifdef __ARM_NEON
        { float32x4_t vm = vdupq_n_f32(mean), vv = vdupq_n_f32(0.0f);
          for (; c <= C - 4; c += 4) {
              float32x4_t d = vsubq_f32(vld1q_f32(x + c), vm);
              vv = vfmaq_f32(vv, d, d); }
          var = vaddvq_f32(vv); }
#elif defined(__AVX512F__)
        { __m512 vm = _mm512_set1_ps(mean), vv = _mm512_setzero_ps();
          for (; c + 16 <= C; c += 16) {
              __m512 d = _mm512_sub_ps(_mm512_loadu_ps(x + c), vm);
              vv = _mm512_fmadd_ps(d, d, vv); }
          var = hsum512(vv); }
#elif defined(__AVX2__)
        { __m256 vm = _mm256_set1_ps(mean), vv = _mm256_setzero_ps();
          for (; c + 8 <= C; c += 8) {
              __m256 d = _mm256_sub_ps(_mm256_loadu_ps(x + c), vm);
              vv = _mm256_fmadd_ps(d, d, vv); }
          var = hsum256(vv); }
#endif
        for (; c < C; ++c) { float d = x[c] - mean; var += d * d; }
        var /= (float)C;

        float inv_std = 1.0f / sqrtf(var + eps);

        c = 0;
#ifdef __ARM_NEON
        { float32x4_t vm = vdupq_n_f32(mean), vis = vdupq_n_f32(inv_std);
          for (; c <= C - 4; c += 4) {
              float32x4_t v = vmulq_f32(vsubq_f32(vld1q_f32(x + c), vm), vis);
              vst1q_f32(y + c, vfmaq_f32(vld1q_f32(beta + c), v, vld1q_f32(gamma + c))); } }
#elif defined(__AVX512F__)
        { __m512 vm = _mm512_set1_ps(mean), vis = _mm512_set1_ps(inv_std);
          for (; c + 16 <= C; c += 16) {
              __m512 v = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(x + c), vm), vis);
              _mm512_storeu_ps(y + c, _mm512_fmadd_ps(v, _mm512_loadu_ps(gamma + c),
                                                       _mm512_loadu_ps(beta + c))); } }
#elif defined(__AVX2__)
        { __m256 vm = _mm256_set1_ps(mean), vis = _mm256_set1_ps(inv_std);
          for (; c + 8 <= C; c += 8) {
              __m256 v = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x + c), vm), vis);
              _mm256_storeu_ps(y + c, _mm256_fmadd_ps(v, _mm256_loadu_ps(gamma + c),
                                                       _mm256_loadu_ps(beta + c))); } }
#endif
        for (; c < C; ++c)
            y[c] = (x[c] - mean) * inv_std * gamma[c] + beta[c];
    }
    if (_profile_layernorm) g_layernorm_ms += now_ms() - _t0_layernorm;
}

// ─────────────────────────────────────────────────────────────────
// AdaIN1d normalization
//   Instance-norm per-channel over T, then style modulation:
//   out = (1 + gamma) * (x - mean) / std + beta
// ─────────────────────────────────────────────────────────────────

void ada_in1d_fp32(const float* feat, float* out,
                   const float* gamma, const float* beta,
                   int T, int C, float eps, StreamHandle /* stream */,
                   const float* norm_weight, const float* norm_bias)
{
    const bool _profile_ada = g_ops_profile_enabled;
    const double _t0_ada = _profile_ada ? now_ms() : 0.0;
    // Two-pass variance: pass 1 = mean, pass 2 = centered sum-of-squares.
    // Avoids catastrophic cancellation when |mean| >> std (e.g. mean≈-40, std≈0.4).
    std::vector<float> sum_c(C, 0.f);
    std::vector<float> mean_c(C), inv_std_c(C);

#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
    if (max_threads > 1 && !omp_in_parallel() && (size_t)T * C >= 32768) {
        std::vector<float> thread_sum((size_t)max_threads * C, 0.f);
#pragma omp parallel num_threads(max_threads)
        {
            const int tid = omp_get_thread_num();
            float* local = thread_sum.data() + (size_t)tid * C;
#pragma omp for schedule(static)
            for (int t = 0; t < T; ++t) {
                const float* x = feat + (size_t)t * C;
                int c = 0;
#ifdef __ARM_NEON
                for (; c + 4 <= C; c += 4) {
                    float32x4_t vs = vld1q_f32(local + c);
                    vst1q_f32(local + c, vaddq_f32(vs, vld1q_f32(x + c)));
                }
#elif defined(__AVX512F__)
                for (; c + 16 <= C; c += 16) {
                    __m512 vs = _mm512_loadu_ps(local + c);
                    _mm512_storeu_ps(local + c, _mm512_add_ps(vs, _mm512_loadu_ps(x + c)));
                }
#elif defined(__AVX2__)
                for (; c + 8 <= C; c += 8) {
                    __m256 vs = _mm256_loadu_ps(local + c);
                    _mm256_storeu_ps(local + c, _mm256_add_ps(vs, _mm256_loadu_ps(x + c)));
                }
#endif
                for (; c < C; ++c) local[c] += x[c];
            }
        }

        const float inv_T = 1.f / (float)T;
#pragma omp parallel for schedule(static) num_threads(max_threads)
        for (int c = 0; c < C; ++c) {
            float sum = 0.f;
            for (int tid = 0; tid < max_threads; ++tid)
                sum += thread_sum[(size_t)tid * C + c];
            sum_c[c] = sum;
            mean_c[c] = sum * inv_T;
        }

        std::vector<float> var_c(C, 0.f);
        std::vector<float> thread_var((size_t)max_threads * C, 0.f);
#pragma omp parallel num_threads(max_threads)
        {
            const int tid = omp_get_thread_num();
            float* local = thread_var.data() + (size_t)tid * C;
#pragma omp for schedule(static)
            for (int t = 0; t < T; ++t) {
                const float* x = feat + (size_t)t * C;
                int c = 0;
#ifdef __ARM_NEON
                for (; c + 4 <= C; c += 4) {
                    float32x4_t vd = vsubq_f32(vld1q_f32(x + c), vld1q_f32(mean_c.data() + c));
                    float32x4_t va = vld1q_f32(local + c);
                    vst1q_f32(local + c, vfmaq_f32(va, vd, vd));
                }
#elif defined(__AVX512F__)
                for (; c + 16 <= C; c += 16) {
                    __m512 vd = _mm512_sub_ps(_mm512_loadu_ps(x + c), _mm512_loadu_ps(mean_c.data() + c));
                    __m512 va = _mm512_loadu_ps(local + c);
                    _mm512_storeu_ps(local + c, _mm512_fmadd_ps(vd, vd, va));
                }
#elif defined(__AVX2__)
                for (; c + 8 <= C; c += 8) {
                    __m256 vd = _mm256_sub_ps(_mm256_loadu_ps(x + c), _mm256_loadu_ps(mean_c.data() + c));
                    __m256 va = _mm256_loadu_ps(local + c);
                    _mm256_storeu_ps(local + c, _mm256_fmadd_ps(vd, vd, va));
                }
#endif
                for (; c < C; ++c) {
                    float d = x[c] - mean_c[c];
                    local[c] += d * d;
                }
            }
        }

#pragma omp parallel for schedule(static) num_threads(max_threads)
        for (int c = 0; c < C; ++c) {
            float var = 0.f;
            for (int tid = 0; tid < max_threads; ++tid)
                var += thread_var[(size_t)tid * C + c];
            var_c[c] = var;
            inv_std_c[c] = 1.f / sqrtf(var * inv_T + eps);
        }
    } else
#endif
    {
    // Pass 1: accumulate sum per channel → mean
    for (int t = 0; t < T; ++t) {
        const float* x = feat + (size_t)t * C;
        int c = 0;
#ifdef __ARM_NEON
        for (; c + 4 <= C; c += 4) {
            float32x4_t vs = vld1q_f32(sum_c.data() + c);
            vst1q_f32(sum_c.data() + c, vaddq_f32(vs, vld1q_f32(x + c)));
        }
#elif defined(__AVX512F__)
        for (; c + 16 <= C; c += 16) {
            __m512 vs = _mm512_loadu_ps(sum_c.data() + c);
            _mm512_storeu_ps(sum_c.data() + c, _mm512_add_ps(vs, _mm512_loadu_ps(x + c)));
        }
#elif defined(__AVX2__)
        for (; c + 8 <= C; c += 8) {
            __m256 vs = _mm256_loadu_ps(sum_c.data() + c);
            _mm256_storeu_ps(sum_c.data() + c, _mm256_add_ps(vs, _mm256_loadu_ps(x + c)));
        }
#endif
        for (; c < C; ++c) sum_c[c] += x[c];
    }
    const float inv_T = 1.f / (float)T;
    for (int c = 0; c < C; ++c) mean_c[c] = sum_c[c] * inv_T;

    // Pass 2: accumulate centered sum-of-squares per channel
    std::vector<float> var_c(C, 0.f);
    for (int t = 0; t < T; ++t) {
        const float* x = feat + (size_t)t * C;
        int c = 0;
#ifdef __ARM_NEON
        for (; c + 4 <= C; c += 4) {
            float32x4_t vd  = vsubq_f32(vld1q_f32(x + c), vld1q_f32(mean_c.data() + c));
            float32x4_t va  = vld1q_f32(var_c.data() + c);
            vst1q_f32(var_c.data() + c, vfmaq_f32(va, vd, vd));
        }
#elif defined(__AVX512F__)
        for (; c + 16 <= C; c += 16) {
            __m512 vd = _mm512_sub_ps(_mm512_loadu_ps(x + c), _mm512_loadu_ps(mean_c.data() + c));
            __m512 va = _mm512_loadu_ps(var_c.data() + c);
            _mm512_storeu_ps(var_c.data() + c, _mm512_fmadd_ps(vd, vd, va));
        }
#elif defined(__AVX2__)
        for (; c + 8 <= C; c += 8) {
            __m256 vd = _mm256_sub_ps(_mm256_loadu_ps(x + c), _mm256_loadu_ps(mean_c.data() + c));
            __m256 va = _mm256_loadu_ps(var_c.data() + c);
            _mm256_storeu_ps(var_c.data() + c, _mm256_fmadd_ps(vd, vd, va));
        }
#endif
        for (; c < C; ++c) { float d = x[c] - mean_c[c]; var_c[c] += d * d; }
    }
    for (int c = 0; c < C; ++c)
        inv_std_c[c] = 1.f / sqrtf(var_c[c] * inv_T + eps);
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
#endif
    for (int t = 0; t < T; ++t) {
        const float* x = feat + (size_t)t * C;
        float*       y = out  + (size_t)t * C;
        int c = 0;
        if (norm_weight) {
            // InstanceNorm has learnable scale/shift: xnorm = raw_norm * nw + nb
#ifdef __ARM_NEON
            for (; c + 4 <= C; c += 4) {
                float32x4_t vx   = vld1q_f32(x + c);
                float32x4_t vm   = vld1q_f32(mean_c.data()    + c);
                float32x4_t vis  = vld1q_f32(inv_std_c.data() + c);
                float32x4_t vnw  = vld1q_f32(norm_weight + c);
                float32x4_t vnb  = vld1q_f32(norm_bias   + c);
                float32x4_t vg   = vld1q_f32(gamma + c);
                float32x4_t vb   = vld1q_f32(beta  + c);
                float32x4_t xraw = vmulq_f32(vsubq_f32(vx, vm), vis);
                float32x4_t xnorm = vfmaq_f32(vnb, vnw, xraw);  // xraw * nw + nb
                vst1q_f32(y + c, vfmaq_f32(vaddq_f32(xnorm, vb), vg, xnorm));
            }
#elif defined(__AVX512F__)
            for (; c + 16 <= C; c += 16) {
                __m512 vx   = _mm512_loadu_ps(x + c);
                __m512 vm   = _mm512_loadu_ps(mean_c.data()    + c);
                __m512 vis  = _mm512_loadu_ps(inv_std_c.data() + c);
                __m512 vnw  = _mm512_loadu_ps(norm_weight + c);
                __m512 vnb  = _mm512_loadu_ps(norm_bias   + c);
                __m512 vg   = _mm512_loadu_ps(gamma + c);
                __m512 vb   = _mm512_loadu_ps(beta  + c);
                __m512 xraw = _mm512_mul_ps(_mm512_sub_ps(vx, vm), vis);
                __m512 xnorm = _mm512_fmadd_ps(vnw, xraw, vnb);
                _mm512_storeu_ps(y + c, _mm512_fmadd_ps(vg, xnorm, _mm512_add_ps(xnorm, vb)));
            }
#elif defined(__AVX2__)
            for (; c + 8 <= C; c += 8) {
                __m256 vx   = _mm256_loadu_ps(x + c);
                __m256 vm   = _mm256_loadu_ps(mean_c.data()    + c);
                __m256 vis  = _mm256_loadu_ps(inv_std_c.data() + c);
                __m256 vnw  = _mm256_loadu_ps(norm_weight + c);
                __m256 vnb  = _mm256_loadu_ps(norm_bias   + c);
                __m256 vg   = _mm256_loadu_ps(gamma + c);
                __m256 vb   = _mm256_loadu_ps(beta  + c);
                __m256 xraw = _mm256_mul_ps(_mm256_sub_ps(vx, vm), vis);
                __m256 xnorm = _mm256_fmadd_ps(vnw, xraw, vnb);
                _mm256_storeu_ps(y + c, _mm256_fmadd_ps(vg, xnorm, _mm256_add_ps(xnorm, vb)));
            }
#endif
            for (; c < C; ++c) {
                float xraw = (x[c] - mean_c[c]) * inv_std_c[c];
                float xnorm = xraw * norm_weight[c] + norm_bias[c];
                y[c] = (1.f + gamma[c]) * xnorm + beta[c];
            }
        } else {
            // No learnable InstanceNorm params (default: scale=1, shift=0)
#ifdef __ARM_NEON
            for (; c + 4 <= C; c += 4) {
                float32x4_t vx    = vld1q_f32(x + c);
                float32x4_t vm    = vld1q_f32(mean_c.data()    + c);
                float32x4_t vis   = vld1q_f32(inv_std_c.data() + c);
                float32x4_t vg    = vld1q_f32(gamma + c);
                float32x4_t vb    = vld1q_f32(beta  + c);
                float32x4_t xnorm = vmulq_f32(vsubq_f32(vx, vm), vis);
                vst1q_f32(y + c, vfmaq_f32(vaddq_f32(xnorm, vb), vg, xnorm));
            }
#elif defined(__AVX512F__)
            for (; c + 16 <= C; c += 16) {
                __m512 vx    = _mm512_loadu_ps(x + c);
                __m512 vm    = _mm512_loadu_ps(mean_c.data()    + c);
                __m512 vis   = _mm512_loadu_ps(inv_std_c.data() + c);
                __m512 vg    = _mm512_loadu_ps(gamma + c);
                __m512 vb    = _mm512_loadu_ps(beta  + c);
                __m512 xnorm = _mm512_mul_ps(_mm512_sub_ps(vx, vm), vis);
                _mm512_storeu_ps(y + c, _mm512_fmadd_ps(vg, xnorm, _mm512_add_ps(xnorm, vb)));
            }
#elif defined(__AVX2__)
            for (; c + 8 <= C; c += 8) {
                __m256 vx    = _mm256_loadu_ps(x + c);
                __m256 vm    = _mm256_loadu_ps(mean_c.data()    + c);
                __m256 vis   = _mm256_loadu_ps(inv_std_c.data() + c);
                __m256 vg    = _mm256_loadu_ps(gamma + c);
                __m256 vb    = _mm256_loadu_ps(beta  + c);
                __m256 xnorm = _mm256_mul_ps(_mm256_sub_ps(vx, vm), vis);
                _mm256_storeu_ps(y + c, _mm256_fmadd_ps(vg, xnorm, _mm256_add_ps(xnorm, vb)));
            }
#endif
            for (; c < C; ++c) {
                float xn = (x[c] - mean_c[c]) * inv_std_c[c];
                y[c] = (1.f + gamma[c]) * xn + beta[c];
            }
        }
    }
    if (_profile_ada) g_ada_in1d_ms += now_ms() - _t0_ada;
}

// ─────────────────────────────────────────────────────────────────
// AdaLayerNorm1d normalization
//   Layer-norm per time-step (mean/var across C), then style modulation.
// ─────────────────────────────────────────────────────────────────

void ada_layer_norm1d_fp32(const float* feat, float* out,
                            const float* gamma, const float* beta,
                            int T, int C, float eps, StreamHandle /* stream */)
{
    const float inv_C = 1.f / (float)C;

    for (int t = 0; t < T; ++t) {
        const float* x = feat + (size_t)t * C;
        float*       y = out  + (size_t)t * C;

        float sum = 0.f;
        int c = 0;
#ifdef __ARM_NEON
        { float32x4_t vs = vdupq_n_f32(0.f);
          for (; c + 4 <= C; c += 4) vs = vaddq_f32(vs, vld1q_f32(x + c));
          sum = vaddvq_f32(vs); }
#elif defined(__AVX512F__)
        { __m512 vs = _mm512_setzero_ps();
          for (; c + 16 <= C; c += 16) vs = _mm512_add_ps(vs, _mm512_loadu_ps(x + c));
          sum = hsum512(vs); }
#elif defined(__AVX2__)
        { __m256 vs = _mm256_setzero_ps();
          for (; c + 8 <= C; c += 8) vs = _mm256_add_ps(vs, _mm256_loadu_ps(x + c));
          sum = hsum256(vs); }
#endif
        for (; c < C; ++c) sum += x[c];
        const float mean = sum * inv_C;

        float var = 0.f;
        c = 0;
#ifdef __ARM_NEON
        { float32x4_t vm = vdupq_n_f32(mean), vv = vdupq_n_f32(0.f);
          for (; c + 4 <= C; c += 4) {
              float32x4_t d = vsubq_f32(vld1q_f32(x + c), vm);
              vv = vfmaq_f32(vv, d, d); }
          var = vaddvq_f32(vv); }
#elif defined(__AVX512F__)
        { __m512 vm = _mm512_set1_ps(mean), vv = _mm512_setzero_ps();
          for (; c + 16 <= C; c += 16) {
              __m512 d = _mm512_sub_ps(_mm512_loadu_ps(x + c), vm);
              vv = _mm512_fmadd_ps(d, d, vv); }
          var = hsum512(vv); }
#elif defined(__AVX2__)
        { __m256 vm = _mm256_set1_ps(mean), vv = _mm256_setzero_ps();
          for (; c + 8 <= C; c += 8) {
              __m256 d = _mm256_sub_ps(_mm256_loadu_ps(x + c), vm);
              vv = _mm256_fmadd_ps(d, d, vv); }
          var = hsum256(vv); }
#endif
        for (; c < C; ++c) { float d = x[c] - mean; var += d * d; }
        const float inv_std = 1.f / sqrtf(var * inv_C + eps);

        c = 0;
#ifdef __ARM_NEON
        { float32x4_t vm = vdupq_n_f32(mean), vis = vdupq_n_f32(inv_std);
          for (; c + 4 <= C; c += 4) {
              float32x4_t vx    = vld1q_f32(x + c);
              float32x4_t vg    = vld1q_f32(gamma + c);
              float32x4_t vb    = vld1q_f32(beta  + c);
              float32x4_t xnorm = vmulq_f32(vsubq_f32(vx, vm), vis);
              vst1q_f32(y + c, vfmaq_f32(vaddq_f32(xnorm, vb), vg, xnorm)); } }
#elif defined(__AVX512F__)
        { __m512 vm = _mm512_set1_ps(mean), vis = _mm512_set1_ps(inv_std);
          for (; c + 16 <= C; c += 16) {
              __m512 vx    = _mm512_loadu_ps(x + c);
              __m512 vg    = _mm512_loadu_ps(gamma + c);
              __m512 vb    = _mm512_loadu_ps(beta  + c);
              __m512 xnorm = _mm512_mul_ps(_mm512_sub_ps(vx, vm), vis);
              _mm512_storeu_ps(y + c, _mm512_fmadd_ps(vg, xnorm, _mm512_add_ps(xnorm, vb))); } }
#elif defined(__AVX2__)
        { __m256 vm = _mm256_set1_ps(mean), vis = _mm256_set1_ps(inv_std);
          for (; c + 8 <= C; c += 8) {
              __m256 vx    = _mm256_loadu_ps(x + c);
              __m256 vg    = _mm256_loadu_ps(gamma + c);
              __m256 vb    = _mm256_loadu_ps(beta  + c);
              __m256 xnorm = _mm256_mul_ps(_mm256_sub_ps(vx, vm), vis);
              _mm256_storeu_ps(y + c, _mm256_fmadd_ps(vg, xnorm, _mm256_add_ps(xnorm, vb))); } }
#endif
        for (; c < C; ++c) {
            float xn = (x[c] - mean) * inv_std;
            y[c] = (1.f + gamma[c]) * xn + beta[c];
        }
    }
}
