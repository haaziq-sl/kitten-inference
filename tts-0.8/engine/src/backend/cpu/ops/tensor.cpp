// tensor.cpp — ModelIOTensor layout and manipulation kernels (FP32 / INT8)

#include "../ops_neon.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

// ─────────────────────────────────────────────────────────────────
// dst[i] += src[i]
// ─────────────────────────────────────────────────────────────────

void add_vectors_fp32(float* dst, const float* src, int N, StreamHandle /* stream */)
{
#ifdef __ARM_NEON
    int i = 0;
    for (; i + 4 <= N; i += 4)
        vst1q_f32(dst + i, vaddq_f32(vld1q_f32(dst + i), vld1q_f32(src + i)));
    for (; i < N; ++i) dst[i] += src[i];
#elif defined(__AVX512F__)
    int i = 0;
    for (; i + 16 <= N; i += 16)
        _mm512_storeu_ps(dst + i, _mm512_add_ps(_mm512_loadu_ps(dst + i), _mm512_loadu_ps(src + i)));
    for (; i < N; ++i) dst[i] += src[i];
#elif defined(__AVX2__)
    int i = 0;
    for (; i + 8 <= N; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
    for (; i < N; ++i) dst[i] += src[i];
#else
    for (int i = 0; i < N; ++i) dst[i] += src[i];
#endif
}

// ─────────────────────────────────────────────────────────────────
// Row-wise bias add: out[t, c] += bias[c]
// ─────────────────────────────────────────────────────────────────

void bias_add_rows_fp32(float* out, const float* bias, int T, int C, StreamHandle /* stream */)
{
    for (int t = 0; t < T; ++t) {
        float* y = out + (size_t)t * C;
        int c = 0;
#ifdef __ARM_NEON
        for (; c + 4 <= C; c += 4)
            vst1q_f32(y + c, vaddq_f32(vld1q_f32(y + c), vld1q_f32(bias + c)));
#elif defined(__AVX512F__)
        for (; c + 16 <= C; c += 16)
            _mm512_storeu_ps(y + c, _mm512_add_ps(_mm512_loadu_ps(y + c), _mm512_loadu_ps(bias + c)));
#elif defined(__AVX2__)
        for (; c + 8 <= C; c += 8)
            _mm256_storeu_ps(y + c, _mm256_add_ps(_mm256_loadu_ps(y + c), _mm256_loadu_ps(bias + c)));
#endif
        for (; c < C; ++c) y[c] += bias[c];
    }
}

// ─────────────────────────────────────────────────────────────────
// Sum channels: [T, C] → [T] (horizontal reduction per row)
// ─────────────────────────────────────────────────────────────────

void sum_channels_fp32(const float* in, float* out, int T, int C, StreamHandle /* stream */)
{
    for (int t = 0; t < T; ++t) {
        const float* src = in + (size_t)t * C;
        float s = 0.f;
        int c = 0;
#ifdef __ARM_NEON
        { float32x4_t vs = vdupq_n_f32(0.f);
          for (; c + 4 <= C; c += 4) vs = vaddq_f32(vs, vld1q_f32(src + c));
          s = vaddvq_f32(vs); }
#elif defined(__AVX512F__)
        { __m512 vs = _mm512_setzero_ps();
          for (; c + 16 <= C; c += 16) vs = _mm512_add_ps(vs, _mm512_loadu_ps(src + c));
          s = hsum512(vs); }
#elif defined(__AVX2__)
        { __m256 vs = _mm256_setzero_ps();
          for (; c + 8 <= C; c += 8) vs = _mm256_add_ps(vs, _mm256_loadu_ps(src + c));
          s = hsum256(vs); }
#endif
        for (; c < C; ++c) s += src[c];
        out[t] = s;
    }
}

// ─────────────────────────────────────────────────────────────────
// INT8 per-row quantization helpers
// ─────────────────────────────────────────────────────────────────

void quantize_row_fp32_to_int8(const float* row, int8_t* qrow,
                                float& out_scale, int C)
{
    float lmax = 0.f;
    {
        int c = 0;
#ifdef __ARM_NEON
        { float32x4_t vm = vdupq_n_f32(0.f);
          for (; c + 4 <= C; c += 4) vm = vmaxq_f32(vm, vabsq_f32(vld1q_f32(row + c)));
          lmax = vmaxvq_f32(vm); }
#elif defined(__AVX512F__)
        { __m512 vm = _mm512_setzero_ps();
          const __m512 sign_mask = _mm512_set1_ps(-0.0f);
          for (; c + 16 <= C; c += 16)
              vm = _mm512_max_ps(vm, _mm512_andnot_ps(sign_mask, _mm512_loadu_ps(row + c)));
          lmax = _mm512_reduce_max_ps(vm); }
#elif defined(__AVX2__)
        { __m256 vm = _mm256_setzero_ps();
          const __m256 sign_mask = _mm256_set1_ps(-0.0f);
          for (; c + 8 <= C; c += 8)
              vm = _mm256_max_ps(vm, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(row + c)));
          // horizontal max of 8 floats
          __m128 lo = _mm256_castps256_ps128(vm);
          __m128 hi = _mm256_extractf128_ps(vm, 1);
          lo = _mm_max_ps(lo, hi);
          lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
          lo = _mm_max_ss(lo, _mm_shuffle_ps(lo, lo, 1));
          lmax = _mm_cvtss_f32(lo); }
#endif
        for (; c < C; ++c) { float a = fabsf(row[c]); if (a > lmax) lmax = a; }
    }
    float scale = std::max(lmax / 127.f, 1e-8f);
    out_scale = scale;
    float inv = 1.f / scale;
    {
        int c = 0;
#ifdef __ARM_NEON
        { float32x4_t vinv = vdupq_n_f32(inv);
          for (; c + 16 <= C; c += 16) {
              int32x4_t q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(row+c   ), vinv));
              int32x4_t q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(row+c+4 ), vinv));
              int32x4_t q2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(row+c+8 ), vinv));
              int32x4_t q3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(row+c+12), vinv));
              vst1q_s8(qrow + c, vcombine_s8(
                  vqmovn_s16(vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1))),
                  vqmovn_s16(vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3))))); } }
#elif defined(__AVX2__)
        // Process 16 floats → 16 int8 per iteration using 128-bit pack path
        // (avoids AVX2 lane-interleaving issue in 256-bit packs).
        { __m128 vinv = _mm_set1_ps(inv);
          for (; c + 16 <= C; c += 16) {
              __m128i q0 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(row + c),      vinv));
              __m128i q1 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(row + c + 4),  vinv));
              __m128i q2 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(row + c + 8),  vinv));
              __m128i q3 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(row + c + 12), vinv));
              __m128i s01 = _mm_packs_epi32(q0, q1);   // 8x int16
              __m128i s23 = _mm_packs_epi32(q2, q3);   // 8x int16
              __m128i b   = _mm_packs_epi16(s01, s23); // 16x int8
              _mm_storeu_si128((__m128i*)(qrow + c), b); } }
#endif
        for (; c < C; ++c) {
            int q = (int)roundf(row[c] * inv);
            qrow[c] = (int8_t)std::clamp(q, -128, 127);
        }
    }
}

// Dequant in-place and add bias (k=0 path): row[c] = row[c]*scale + bias[c]
void dequant_bias_row_fp32(float* row, float row_scale,
                            const float* bias, int C)
{
    int co = 0;
#ifdef __ARM_NEON
    float32x4_t vrs = vdupq_n_f32(row_scale);
    if (bias) {
        for (; co + 4 <= C; co += 4)
            vst1q_f32(row+co, vaddq_f32(vmulq_f32(vld1q_f32(row+co), vrs),
                                         vld1q_f32(bias+co)));
    } else {
        for (; co + 4 <= C; co += 4)
            vst1q_f32(row+co, vmulq_f32(vld1q_f32(row+co), vrs));
    }
#elif defined(__AVX512F__)
    __m512 vrs = _mm512_set1_ps(row_scale);
    if (bias) {
        for (; co + 16 <= C; co += 16)
            _mm512_storeu_ps(row+co, _mm512_fmadd_ps(_mm512_loadu_ps(row+co), vrs,
                                                      _mm512_loadu_ps(bias+co)));
    } else {
        for (; co + 16 <= C; co += 16)
            _mm512_storeu_ps(row+co, _mm512_mul_ps(_mm512_loadu_ps(row+co), vrs));
    }
#elif defined(__AVX2__)
    __m256 vrs = _mm256_set1_ps(row_scale);
    if (bias) {
        for (; co + 8 <= C; co += 8)
            _mm256_storeu_ps(row+co, _mm256_fmadd_ps(_mm256_loadu_ps(row+co), vrs,
                                                      _mm256_loadu_ps(bias+co)));
    } else {
        for (; co + 8 <= C; co += 8)
            _mm256_storeu_ps(row+co, _mm256_mul_ps(_mm256_loadu_ps(row+co), vrs));
    }
#endif
    if (bias) {
        for (; co < C; ++co) row[co] = row[co] * row_scale + bias[co];
    } else {
        for (; co < C; ++co) row[co] *= row_scale;
    }
}

// Dequant src and accumulate into dst (k>0 path): dst[c] += src[c]*scale
void dequant_accum_row_fp32(float* dst, const float* src,
                             float row_scale, int C)
{
    int co = 0;
#ifdef __ARM_NEON
    float32x4_t vrs = vdupq_n_f32(row_scale);
    for (; co + 4 <= C; co += 4)
        vst1q_f32(dst+co, vaddq_f32(vld1q_f32(dst+co),
                                     vmulq_f32(vld1q_f32(src+co), vrs)));
#elif defined(__AVX512F__)
    __m512 vrs = _mm512_set1_ps(row_scale);
    for (; co + 16 <= C; co += 16)
        _mm512_storeu_ps(dst+co, _mm512_fmadd_ps(_mm512_loadu_ps(src+co), vrs,
                                                  _mm512_loadu_ps(dst+co)));
#elif defined(__AVX2__)
    __m256 vrs = _mm256_set1_ps(row_scale);
    for (; co + 8 <= C; co += 8)
        _mm256_storeu_ps(dst+co, _mm256_fmadd_ps(_mm256_loadu_ps(src+co), vrs,
                                                  _mm256_loadu_ps(dst+co)));
#endif
    for (; co < C; ++co) dst[co] += src[co] * row_scale;
}

// ─────────────────────────────────────────────────────────────────
// Embedding table lookup
// ─────────────────────────────────────────────────────────────────

void embedding_lookup_fp32(const float* ids, const float* weight, float* out,
                            int T, int num_embeddings, int embedding_dim, StreamHandle /* stream */)
{
    for (int t = 0; t < T; ++t) {
        int id = (int)ids[t];
        if (id < 0) id = 0;
        if (id >= num_embeddings) id = num_embeddings - 1;
        memcpy(out + (size_t)t * embedding_dim,
               weight + (size_t)id * embedding_dim,
               embedding_dim * sizeof(float));
    }
}

// ─────────────────────────────────────────────────────────────────
// Concat1D: [T, C1] ++ [T, C2] → [T, C1+C2]
// broadcast=true: in2 is [1, C2], replicated across T
// ─────────────────────────────────────────────────────────────────

void concat1d_fp32(const float* in1, const float* in2, float* out,
                   int T, int C1, int C2, bool broadcast, StreamHandle /* stream */)
{
    const int Cout = C1 + C2;
    for (int t = 0; t < T; ++t) {
        float* dst = out + (size_t)t * Cout;
        memcpy(dst,      in1 + (size_t)t * C1,                   C1 * sizeof(float));
        memcpy(dst + C1, broadcast ? in2 : in2 + (size_t)t * C2, C2 * sizeof(float));
    }
}

// ─────────────────────────────────────────────────────────────────
// Slice channels: in[T, C_in] → out[T, C_out] starting at ch_start
// ─────────────────────────────────────────────────────────────────

void slice_channels_fp32(const float* in, float* out, int T, int C_in,
                          int ch_start, int C_out, StreamHandle /* stream */)
{
    for (int t = 0; t < T; ++t)
        memcpy(out + (size_t)t * C_out,
               in  + (size_t)t * C_in + ch_start,
               C_out * sizeof(float));
}

// ─────────────────────────────────────────────────────────────────
// BERT embeddings: word + position + type_id[0] lookup + LayerNorm
// ─────────────────────────────────────────────────────────────────

void bert_embeddings_fp32(const float* ids, float* out, int T, int E,
                           const float* word_weight, const float* pos_weight,
                           const float* type_weight,
                           const float* ln_weight, const float* ln_bias,
                           float eps)
{
    const float inv_E = 1.f / (float)E;
    for (int t = 0; t < T; ++t) {
        int    word_id = (int)ids[t];
        float* y       = out + (size_t)t * E;

        const float* ww = word_weight + (size_t)word_id * E;
        const float* pw = pos_weight  + (size_t)t       * E;

        // embedding sum + horizontal mean
        float sum = 0.f;
        int c = 0;
#ifdef __ARM_NEON
        { float32x4_t vs = vdupq_n_f32(0.f);
          if (type_weight) {
              for (; c + 4 <= E; c += 4) {
                  float32x4_t v = vaddq_f32(vaddq_f32(vld1q_f32(ww+c), vld1q_f32(pw+c)),
                                            vld1q_f32(type_weight+c));
                  vst1q_f32(y+c, v); vs = vaddq_f32(vs, v); }
          } else {
              for (; c + 4 <= E; c += 4) {
                  float32x4_t v = vaddq_f32(vld1q_f32(ww+c), vld1q_f32(pw+c));
                  vst1q_f32(y+c, v); vs = vaddq_f32(vs, v); } }
          sum = vaddvq_f32(vs); }
#elif defined(__AVX512F__)
        { __m512 vs = _mm512_setzero_ps();
          if (type_weight) {
              for (; c + 16 <= E; c += 16) {
                  __m512 v = _mm512_add_ps(_mm512_add_ps(_mm512_loadu_ps(ww+c),
                                                          _mm512_loadu_ps(pw+c)),
                                           _mm512_loadu_ps(type_weight+c));
                  _mm512_storeu_ps(y+c, v); vs = _mm512_add_ps(vs, v); }
          } else {
              for (; c + 16 <= E; c += 16) {
                  __m512 v = _mm512_add_ps(_mm512_loadu_ps(ww+c), _mm512_loadu_ps(pw+c));
                  _mm512_storeu_ps(y+c, v); vs = _mm512_add_ps(vs, v); } }
          sum = hsum512(vs); }
#elif defined(__AVX2__)
        { __m256 vs = _mm256_setzero_ps();
          if (type_weight) {
              for (; c + 8 <= E; c += 8) {
                  __m256 v = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(ww+c),
                                                          _mm256_loadu_ps(pw+c)),
                                           _mm256_loadu_ps(type_weight+c));
                  _mm256_storeu_ps(y+c, v); vs = _mm256_add_ps(vs, v); }
          } else {
              for (; c + 8 <= E; c += 8) {
                  __m256 v = _mm256_add_ps(_mm256_loadu_ps(ww+c), _mm256_loadu_ps(pw+c));
                  _mm256_storeu_ps(y+c, v); vs = _mm256_add_ps(vs, v); } }
          sum = hsum256(vs); }
#endif
        for (; c < E; ++c) {
            y[c] = ww[c] + pw[c] + (type_weight ? type_weight[c] : 0.f);
            sum += y[c];
        }
        float mean = sum * inv_E;

        // variance
        float var = 0.f;
        c = 0;
#ifdef __ARM_NEON
        { float32x4_t vm = vdupq_n_f32(mean), vv = vdupq_n_f32(0.f);
          for (; c + 4 <= E; c += 4) {
              float32x4_t d = vsubq_f32(vld1q_f32(y + c), vm);
              vv = vfmaq_f32(vv, d, d); }
          var = vaddvq_f32(vv); }
#elif defined(__AVX512F__)
        { __m512 vm = _mm512_set1_ps(mean), vv = _mm512_setzero_ps();
          for (; c + 16 <= E; c += 16) {
              __m512 d = _mm512_sub_ps(_mm512_loadu_ps(y + c), vm);
              vv = _mm512_fmadd_ps(d, d, vv); }
          var = hsum512(vv); }
#elif defined(__AVX2__)
        { __m256 vm = _mm256_set1_ps(mean), vv = _mm256_setzero_ps();
          for (; c + 8 <= E; c += 8) {
              __m256 d = _mm256_sub_ps(_mm256_loadu_ps(y + c), vm);
              vv = _mm256_fmadd_ps(d, d, vv); }
          var = hsum256(vv); }
#endif
        for (; c < E; ++c) { float d = y[c] - mean; var += d * d; }
        float inv_std = 1.f / sqrtf(var * inv_E + eps);

        // normalize + scale/bias
        c = 0;
#ifdef __ARM_NEON
        { float32x4_t vm = vdupq_n_f32(mean), vis = vdupq_n_f32(inv_std);
          if (ln_weight && ln_bias) {
              for (; c + 4 <= E; c += 4) {
                  float32x4_t xn = vmulq_f32(vsubq_f32(vld1q_f32(y+c), vm), vis);
                  vst1q_f32(y+c, vfmaq_f32(vld1q_f32(ln_bias+c), xn, vld1q_f32(ln_weight+c))); }
          } else if (ln_weight) {
              for (; c + 4 <= E; c += 4) {
                  float32x4_t xn = vmulq_f32(vsubq_f32(vld1q_f32(y+c), vm), vis);
                  vst1q_f32(y+c, vmulq_f32(xn, vld1q_f32(ln_weight+c))); }
          } else {
              for (; c + 4 <= E; c += 4)
                  vst1q_f32(y+c, vmulq_f32(vsubq_f32(vld1q_f32(y+c), vm), vis)); } }
#elif defined(__AVX512F__)
        { __m512 vm = _mm512_set1_ps(mean), vis = _mm512_set1_ps(inv_std);
          if (ln_weight && ln_bias) {
              for (; c + 16 <= E; c += 16) {
                  __m512 xn = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(y+c), vm), vis);
                  _mm512_storeu_ps(y+c, _mm512_fmadd_ps(xn, _mm512_loadu_ps(ln_weight+c),
                                                          _mm512_loadu_ps(ln_bias+c))); }
          } else if (ln_weight) {
              for (; c + 16 <= E; c += 16) {
                  __m512 xn = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(y+c), vm), vis);
                  _mm512_storeu_ps(y+c, _mm512_mul_ps(xn, _mm512_loadu_ps(ln_weight+c))); }
          } else {
              for (; c + 16 <= E; c += 16)
                  _mm512_storeu_ps(y+c, _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(y+c),vm),vis)); } }
#elif defined(__AVX2__)
        { __m256 vm = _mm256_set1_ps(mean), vis = _mm256_set1_ps(inv_std);
          if (ln_weight && ln_bias) {
              for (; c + 8 <= E; c += 8) {
                  __m256 xn = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(y+c), vm), vis);
                  _mm256_storeu_ps(y+c, _mm256_fmadd_ps(xn, _mm256_loadu_ps(ln_weight+c),
                                                          _mm256_loadu_ps(ln_bias+c))); }
          } else if (ln_weight) {
              for (; c + 8 <= E; c += 8) {
                  __m256 xn = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(y+c), vm), vis);
                  _mm256_storeu_ps(y+c, _mm256_mul_ps(xn, _mm256_loadu_ps(ln_weight+c))); }
          } else {
              for (; c + 8 <= E; c += 8)
                  _mm256_storeu_ps(y+c, _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(y+c),vm),vis)); } }
#endif
        for (; c < E; ++c) {
            float x_norm = (y[c] - mean) * inv_std;
            y[c] = ln_weight ? ln_weight[c] * x_norm + (ln_bias ? ln_bias[c] : 0.f)
                             : x_norm;
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Length regulate: repeat each feature row by its duration
// Returns actual output length
// ─────────────────────────────────────────────────────────────────

int length_regulate_fp32(const float* feat, const float* durs, float* out,
                          int T, int C, float duration_scale)
{
    int out_t = 0;
    for (int t = 0; t < T; ++t) {
        int rep = std::max(1, (int)roundf(durs[t] * duration_scale));
        const float* src = feat + (size_t)t * C;
        for (int r = 0; r < rep; ++r) {
            memcpy(out + (size_t)out_t * C, src, C * sizeof(float));
            ++out_t;
        }
    }
    return out_t;
}
