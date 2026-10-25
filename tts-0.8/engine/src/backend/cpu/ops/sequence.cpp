// sequence.cpp — Sequential / recurrent / strided compute kernels

#include "../ops_neon.hpp"
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

// ─────────────────────────────────────────────────────────────────
// LSTM recurrence scan (FP32, NEON-vectorised)
//   proj: [T, 4*H]  pre-computed input projections
//   w_hh: [4*H, H]  hidden-to-hidden weights
// ─────────────────────────────────────────────────────────────────

void lstm_scan_fp32(
    const float* proj,
    const float* w_hh,
    const float* b_hh,
    const float* w_hh_packed,
    float*       h_out,
    int T, int H,
    bool reverse)
{
    std::vector<float> h(H, 0.f);
    std::vector<float> c(H, 0.f);
    std::vector<float> gates(4 * H);
    std::vector<float> h_gate(4 * H);

    auto step = [&](int t) {
        gemm_fp32_vec(h.data(), w_hh, b_hh, h_gate.data(), H, 4*H, w_hh_packed, nullptr);

        const float* pt = proj + (size_t)t * 4 * H;
        float* g_  = gates.data();
        float* hg  = h_gate.data();
#ifdef __ARM_NEON
        for (int j = 0; j + 4 <= 4*H; j += 4)
            vst1q_f32(g_ + j, vaddq_f32(vld1q_f32(pt + j), vld1q_f32(hg + j)));
        for (int j = (4*H) & ~3; j < 4*H; ++j) g_[j] = pt[j] + hg[j];
#elif defined(__AVX512F__)
        { int j = 0;
          for (; j + 16 <= 4*H; j += 16)
              _mm512_storeu_ps(g_+j, _mm512_add_ps(_mm512_loadu_ps(pt+j), _mm512_loadu_ps(hg+j)));
          for (; j < 4*H; ++j) g_[j] = pt[j] + hg[j]; }
#elif defined(__AVX2__)
        { int j = 0;
          for (; j + 8 <= 4*H; j += 8)
              _mm256_storeu_ps(g_+j, _mm256_add_ps(_mm256_loadu_ps(pt+j), _mm256_loadu_ps(hg+j)));
          for (; j < 4*H; ++j) g_[j] = pt[j] + hg[j]; }
#else
        for (int i = 0; i < 4*H; ++i) g_[i] = pt[i] + hg[i];
#endif

        float* cv = c.data();
        float* hv = h.data();
#ifdef __ARM_NEON
        int i = 0;
        for (; i + 4 <= H; i += 4) {
            float32x4_t gi = vsigmoidq_f32(vld1q_f32(g_        + i));
            float32x4_t gf = vsigmoidq_f32(vld1q_f32(g_ +   H  + i));
            float32x4_t gg = vtanhq_f32   (vld1q_f32(g_ + 2*H  + i));
            float32x4_t go = vsigmoidq_f32(vld1q_f32(g_ + 3*H  + i));
            float32x4_t cn = vfmaq_f32(vmulq_f32(gi, gg), gf, vld1q_f32(cv + i));
            vst1q_f32(cv + i, cn);
            vst1q_f32(hv + i, vmulq_f32(go, vtanhq_f32(cn)));
        }
        for (; i < H; ++i) {
            float gi = 1.f / (1.f + expf(-g_[i]));
            float gf = 1.f / (1.f + expf(-g_[H + i]));
            float gg = tanhf(g_[2*H + i]);
            float go = 1.f / (1.f + expf(-g_[3*H + i]));
            cv[i] = gf * cv[i] + gi * gg;
            hv[i] = go * tanhf(cv[i]);
        }
#elif defined(__AVX512F__)
        { int i = 0;
          for (; i + 16 <= H; i += 16) {
              __m512 gi = sigmoid512_ps(_mm512_loadu_ps(g_         + i));
              __m512 gf = sigmoid512_ps(_mm512_loadu_ps(g_ +     H + i));
              __m512 gg = tanh512_ps   (_mm512_loadu_ps(g_ + 2 * H + i));
              __m512 go = sigmoid512_ps(_mm512_loadu_ps(g_ + 3 * H + i));
              __m512 cn = _mm512_fmadd_ps(gi, gg, _mm512_mul_ps(gf, _mm512_loadu_ps(cv + i)));
              _mm512_storeu_ps(cv + i, cn);
              _mm512_storeu_ps(hv + i, _mm512_mul_ps(go, tanh512_ps(cn)));
          }
          for (; i < H; ++i) {
              float gi = 1.f / (1.f + expf(-g_[i]));
              float gf = 1.f / (1.f + expf(-g_[H + i]));
              float gg = tanhf(g_[2*H + i]);
              float go = 1.f / (1.f + expf(-g_[3*H + i]));
              cv[i] = gf * cv[i] + gi * gg;
              hv[i] = go * tanhf(cv[i]);
          } }
#elif defined(__AVX2__)
        { int i = 0;
          for (; i + 8 <= H; i += 8) {
              __m256 gi = sigmoid256_ps(_mm256_loadu_ps(g_         + i));
              __m256 gf = sigmoid256_ps(_mm256_loadu_ps(g_ +     H + i));
              __m256 gg = tanh256_ps   (_mm256_loadu_ps(g_ + 2 * H + i));
              __m256 go = sigmoid256_ps(_mm256_loadu_ps(g_ + 3 * H + i));
              __m256 cn = _mm256_fmadd_ps(gi, gg, _mm256_mul_ps(gf, _mm256_loadu_ps(cv + i)));
              _mm256_storeu_ps(cv + i, cn);
              _mm256_storeu_ps(hv + i, _mm256_mul_ps(go, tanh256_ps(cn)));
          }
          for (; i < H; ++i) {
              float gi = 1.f / (1.f + expf(-g_[i]));
              float gf = 1.f / (1.f + expf(-g_[H + i]));
              float gg = tanhf(g_[2*H + i]);
              float go = 1.f / (1.f + expf(-g_[3*H + i]));
              cv[i] = gf * cv[i] + gi * gg;
              hv[i] = go * tanhf(cv[i]);
          } }
#else
        for (int i = 0; i < H; ++i) {
            float gi = 1.f / (1.f + expf(-g_[i]));
            float gf = 1.f / (1.f + expf(-g_[H + i]));
            float gg = tanhf(g_[2*H + i]);
            float go = 1.f / (1.f + expf(-g_[3*H + i]));
            cv[i] = gf * cv[i] + gi * gg;
            hv[i] = go * tanhf(cv[i]);
        }
#endif
        memcpy(h_out + (size_t)t * H, hv, H * sizeof(float));
    };

    if (!reverse) {
        for (int t = 0; t < T; ++t) step(t);
    } else {
        for (int t = T - 1; t >= 0; --t) step(t);
    }
}

// ─────────────────────────────────────────────────────────────────
// ConvTranspose1D grouped scalar fallback
//   weight layout: [C_in, C_out/g, kernel_size]
// ─────────────────────────────────────────────────────────────────

void conv_transpose1d_grouped_fp32(
    const float* in,
    float*       out,
    const float* weight,
    int T, int T_out, int C_in, int C_out,
    int C_in_g, int C_out_g,
    int groups, int stride, int padding, int kernel_size)
{
    for (int g = 0; g < groups; ++g) {
        const int ci_off = g * C_in_g;
        const int co_off = g * C_out_g;
        for (int t = 0; t < T; ++t) {
            const float* x = in + t * C_in + ci_off;
            for (int k = 0; k < kernel_size; ++k) {
                int t_out = t * stride + k - padding;
                if (t_out < 0 || t_out >= T_out) continue;
                float* y = out + t_out * C_out + co_off;
                for (int co = 0; co < C_out_g; ++co) {
                    float acc = 0.f;
                    for (int ci = 0; ci < C_in_g; ++ci)
                        acc += x[ci] * weight[((size_t)(ci_off+ci)*C_out_g + co)*kernel_size + k];
                    y[co] += acc;
                }
            }
        }
    }
}

