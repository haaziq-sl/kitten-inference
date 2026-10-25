// attention.cpp — Self-attention and patch preparation kernels (FP32 + INT8)

#include "../ops_neon.hpp"
#include "profile_internal.hpp"
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>
#include <vector>

// ─────────────────────────────────────────────────────────────────
// Dynamic symmetric INT8 quantization  max|x|/127 → int8
// ─────────────────────────────────────────────────────────────────

static float quantize_to_int8_neon(const float* in, int8_t* out, int N)
{
    float max_abs = 0.0f;
#ifdef __ARM_NEON
    float32x4_t vmax = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= N - 4; i += 4)
        vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(in + i)));
    max_abs = vmaxvq_f32(vmax);
    for (; i < N; ++i) { float a = std::abs(in[i]); if (a > max_abs) max_abs = a; }
#elif defined(__AVX512F__)
    { __m512 vm = _mm512_setzero_ps();
      const __m512 sign_mask = _mm512_set1_ps(-0.f);
      int i = 0;
      for (; i + 16 <= N; i += 16)
          vm = _mm512_max_ps(vm, _mm512_andnot_ps(sign_mask, _mm512_loadu_ps(in + i)));
      max_abs = _mm512_reduce_max_ps(vm);
      for (; i < N; ++i) { float a = std::abs(in[i]); if (a > max_abs) max_abs = a; } }
#elif defined(__AVX2__)
    { __m256 vm = _mm256_setzero_ps();
      const __m256 sign_mask = _mm256_set1_ps(-0.f);
      int i = 0;
      for (; i + 8 <= N; i += 8)
          vm = _mm256_max_ps(vm, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(in + i)));
      // horizontal max
      __m128 lo = _mm256_castps256_ps128(vm);
      __m128 hi = _mm256_extractf128_ps(vm, 1);
      __m128 m4 = _mm_max_ps(lo, hi);
      m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
      m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
      max_abs = _mm_cvtss_f32(m4);
      for (; i < N; ++i) { float a = std::abs(in[i]); if (a > max_abs) max_abs = a; } }
#else
    for (int i = 0; i < N; ++i) { float a = std::abs(in[i]); if (a > max_abs) max_abs = a; }
#endif
    float in_scale  = std::max(max_abs / 127.0f, 1e-8f);
    float inv_scale = 1.0f / in_scale;

#ifdef __ARM_NEON
    float32x4_t vinv = vdupq_n_f32(inv_scale);
    int k = 0;
    for (; k <= N - 16; k += 16) {
        int32x4_t q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(in + k     ), vinv));
        int32x4_t q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(in + k +  4), vinv));
        int32x4_t q2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(in + k +  8), vinv));
        int32x4_t q3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(in + k + 12), vinv));
        int8x16_t r = vcombine_s8(
            vqmovn_s16(vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1))),
            vqmovn_s16(vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3))));
        vst1q_s8(out + k, r);
    }
    for (; k < N; ++k) {
        int q = (int)std::roundf(in[k] * inv_scale);
        out[k] = (int8_t)std::clamp(q, -128, 127);
    }
#elif defined(__AVX512F__)
    {
        __m512 vinv = _mm512_set1_ps(inv_scale);
        __m512 v127 = _mm512_set1_ps(127.f);
        __m512 vm127 = _mm512_set1_ps(-127.f);
        int k = 0;
        for (; k + 16 <= N; k += 16) {
            __m512 vf = _mm512_mul_ps(_mm512_loadu_ps(in + k), vinv);
            __m512i vi = _mm512_cvtps_epi32(vf);   // round-to-nearest
            // clamp to [-127, 127]
            __m512i vlo = _mm512_cvtps_epi32(vm127);
            __m512i vhi = _mm512_cvtps_epi32(v127);
            vi = _mm512_max_epi32(vi, vlo);
            vi = _mm512_min_epi32(vi, vhi);
            // pack int32 → int8 via int16
            __m256i lo = _mm512_cvtepi32_epi16(vi);  // 16× int32 → 16× int16
            __m128i b  = _mm256_cvtepi16_epi8(lo);   // 16× int16 → 16× int8
            _mm_storeu_si128((__m128i*)(out + k), b);
        }
        for (; k < N; ++k) {
            int q = (int)std::roundf(in[k] * inv_scale);
            out[k] = (int8_t)std::clamp(q, -128, 127);
        }
    }
#elif defined(__AVX2__)
    {
        __m256 vinv = _mm256_set1_ps(inv_scale);
        int k = 0;
        for (; k + 8 <= N; k += 8) {
            __m256 vf = _mm256_mul_ps(_mm256_loadu_ps(in + k), vinv);
            __m256i vi = _mm256_cvtps_epi32(vf);
            // pack int32 → int8: int32→int16→int8 using 128-bit halves
            __m128i lo = _mm256_castsi256_si128(vi);
            __m128i hi = _mm256_extracti128_si256(vi, 1);
            __m128i p16 = _mm_packs_epi32(lo, hi);
            __m128i p8  = _mm_packs_epi16(p16, p16);
            _mm_storel_epi64((__m128i*)(out + k), p8);
        }
        for (; k < N; ++k) {
            int q = (int)std::roundf(in[k] * inv_scale);
            out[k] = (int8_t)std::clamp(q, -128, 127);
        }
    }
#else
    for (int k = 0; k < N; ++k) {
        int q = (int)std::roundf(in[k] * inv_scale);
        out[k] = (int8_t)std::clamp(q, -128, 127);
    }
#endif
    return in_scale;
}

// ─────────────────────────────────────────────────────────────────
// seqgemm_int8_raw: INT8 GEMM + FP32 bias (no timing, no GELU)
// Uses per-row activation quantization (matches QNNPACK dynamic quant).
// ─────────────────────────────────────────────────────────────────

static void seqgemm_int8_raw(
    const float*   input,
    const int8_t*  w_packed,
    const float*   w_scales,
    const int64_t* eff_zeros,
    const float*   bias,
    int8_t*        inp_scratch,
    float*         req_scratch,
    float*         output,
    int N, int C_in, int C_out,
    const int32_t* w_row_sums)
{
    // Per-row activation quantization: each row (token) gets its own scale.
    // This matches PyTorch QNNPACK dynamic quantization behavior.
    static thread_local std::vector<float> in_scales;
    in_scales.resize(N);
    for (int r = 0; r < N; ++r)
        in_scales[r] = quantize_to_int8_neon(input  + (size_t)r * C_in,
                                              inp_scratch + (size_t)r * C_in, C_in);

    // req_scratch = w_scales only; per-row in_scale factor applied after GEMM.
    {
        int n = 0;
#ifdef __AVX512F__
        for (; n + 16 <= C_out; n += 16)
            _mm512_storeu_ps(req_scratch + n, _mm512_loadu_ps(w_scales + n));
#elif defined(__AVX2__)
        for (; n + 8 <= C_out; n += 8)
            _mm256_storeu_ps(req_scratch + n, _mm256_loadu_ps(w_scales + n));
#endif
        for (; n < C_out; ++n) req_scratch[n] = w_scales[n];
    }

    gemm_int8(inp_scratch, w_packed, eff_zeros,
              nullptr, nullptr, req_scratch,
              0, output, true,
              N, C_in, C_out,
              false, false, w_row_sums, nullptr);

    // Post-scale: multiply each row by its per-row input scale.
    for (int r = 0; r < N; ++r) {
        float s = in_scales[r];
        float* row = output + (size_t)r * C_out;
        int c = 0;
#ifdef __ARM_NEON
        float32x4_t vs = vdupq_n_f32(s);
        for (; c <= C_out - 4; c += 4)
            vst1q_f32(row + c, vmulq_f32(vld1q_f32(row + c), vs));
#elif defined(__AVX512F__)
        __m512 vs512 = _mm512_set1_ps(s);
        for (; c + 16 <= C_out; c += 16)
            _mm512_storeu_ps(row+c, _mm512_mul_ps(_mm512_loadu_ps(row+c), vs512));
#elif defined(__AVX2__)
        __m256 vs256 = _mm256_set1_ps(s);
        for (; c + 8 <= C_out; c += 8)
            _mm256_storeu_ps(row+c, _mm256_mul_ps(_mm256_loadu_ps(row+c), vs256));
#endif
        for (; c < C_out; ++c) row[c] *= s;
    }

    if (bias) {
        for (int r = 0; r < N; ++r) {
            float* row = output + (size_t)r * C_out;
            int c = 0;
#ifdef __ARM_NEON
            for (; c <= C_out - 4; c += 4)
                vst1q_f32(row + c, vaddq_f32(vld1q_f32(row + c), vld1q_f32(bias + c)));
#elif defined(__AVX512F__)
            for (; c + 16 <= C_out; c += 16)
                _mm512_storeu_ps(row+c, _mm512_add_ps(_mm512_loadu_ps(row+c), _mm512_loadu_ps(bias+c)));
#elif defined(__AVX2__)
            for (; c + 8 <= C_out; c += 8)
                _mm256_storeu_ps(row+c, _mm256_add_ps(_mm256_loadu_ps(row+c), _mm256_loadu_ps(bias+c)));
#endif
            for (; c < C_out; ++c) row[c] += bias[c];
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Patch Prep: flatten spatial input → prepend CLS → add pos_embed
// ─────────────────────────────────────────────────────────────────

void patch_prep_fp32(const float* patches, const float* cls_token,
                     const float* pos_embed, float* output,
                     int N_patches, int D)
{
    const int total = N_patches + 1;

    {
        const float* pe0 = pos_embed;
        int d = 0;
#ifdef __ARM_NEON
        for (; d <= D - 4; d += 4)
            vst1q_f32(output + d, vaddq_f32(vld1q_f32(cls_token + d), vld1q_f32(pe0 + d)));
#elif defined(__AVX512F__)
        for (; d + 16 <= D; d += 16)
            _mm512_storeu_ps(output+d, _mm512_add_ps(_mm512_loadu_ps(cls_token+d), _mm512_loadu_ps(pe0+d)));
#elif defined(__AVX2__)
        for (; d + 8 <= D; d += 8)
            _mm256_storeu_ps(output+d, _mm256_add_ps(_mm256_loadu_ps(cls_token+d), _mm256_loadu_ps(pe0+d)));
#endif
        for (; d < D; ++d) output[d] = cls_token[d] + pe0[d];
    }

    for (int p = 0; p < N_patches; ++p) {
        const float* src = patches   + (size_t)p * D;
        const float* pe  = pos_embed + (size_t)(p + 1) * D;
        float*       dst = output    + (size_t)(p + 1) * D;
        int d = 0;
#ifdef __ARM_NEON
        for (; d <= D - 4; d += 4)
            vst1q_f32(dst + d, vaddq_f32(vld1q_f32(src + d), vld1q_f32(pe + d)));
#elif defined(__AVX512F__)
        for (; d + 16 <= D; d += 16)
            _mm512_storeu_ps(dst+d, _mm512_add_ps(_mm512_loadu_ps(src+d), _mm512_loadu_ps(pe+d)));
#elif defined(__AVX2__)
        for (; d + 8 <= D; d += 8)
            _mm256_storeu_ps(dst+d, _mm256_add_ps(_mm256_loadu_ps(src+d), _mm256_loadu_ps(pe+d)));
#endif
        for (; d < D; ++d) dst[d] = src[d] + pe[d];
    }
    (void)total;
}

// ─────────────────────────────────────────────────────────────────
// CLS token extraction
// ─────────────────────────────────────────────────────────────────

void cls_extract_fp32(const float* input, float* output, int D)
{
    memcpy(output, input, (size_t)D * sizeof(float));
}

// ─────────────────────────────────────────────────────────────────
// Sequence GEMM  [N, C_in] @ W[C_out, C_in]^T + bias → [N, C_out]
// ─────────────────────────────────────────────────────────────────

void seqgemm_fp32(const float* input, const float* weight, const float* bias,
                  float* output, int N, int C_in, int C_out,
                  bool gelu, const float* w_packed, StreamHandle /* stream */)
{
    double t0 = now_ms();

    if (w_packed) {
        sgemm_f32(input, w_packed, bias, output, false, N, C_in, C_out, false);
    } else {
        for (int n = 0; n < N; ++n) {
            for (int co = 0; co < C_out; ++co) {
                float s = bias ? bias[co] : 0.0f;
                const float* wi = weight + (size_t)co * C_in;
                const float* xi = input  + (size_t)n  * C_in;
                for (int kk = 0; kk < C_in; ++kk) s += xi[kk] * wi[kk];
                output[(size_t)n * C_out + co] = s;
            }
        }
    }

    g_vit_seqgemm_ms += now_ms() - t0;

    if (gelu) gelu_fp32(output, N * C_out);
}

// ─────────────────────────────────────────────────────────────────
// Flash Attention-2 Multi-Head Self-Attention (FP32)
// ─────────────────────────────────────────────────────────────────

void attention_flash_fp32(
    const float* input,
    const float* w_qkv,
    const float* b_qkv,
    const float* w_proj,
    const float* b_proj,
    float*       output,
    int N, int D, int num_heads,
    float* scratch_qkv,
    const float* w_qkv_packed,
    const float* w_proj_packed,
    StreamHandle /* stream */,
    bool dq_attn_out)
{
    double t0 = now_ms();

    const int   d_k   = D / num_heads;
    const float scale = 1.0f / sqrtf((float)d_k);

    sgemm_f32(input, w_qkv_packed, b_qkv, scratch_qkv, false, N, D, 3 * D, false);

    memset(output, 0, (size_t)N * D * sizeof(float));

    constexpr int Br = 64;
    constexpr int Bc = 64;

#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        // Thread-local scratch: persists between calls; reallocates only when needed.
        static thread_local std::vector<float> Q_h, K_h, V_h, S_buf, O_blk, m_blk, l_blk, sp;
        const size_t head_seq = (size_t)N * d_k;
        if (Q_h.size() < head_seq) Q_h.resize(head_seq);
        if (K_h.size() < head_seq) K_h.resize(head_seq);
        if (V_h.size() < head_seq) V_h.resize(head_seq);
        if (S_buf.size() < (size_t)Br * Bc) S_buf.resize((size_t)Br * Bc);
        const size_t oblk_need = (size_t)Br * d_k;
        if (O_blk.size() < oblk_need) O_blk.resize(oblk_need);
        if (m_blk.size() < (size_t)Br) m_blk.resize(Br);
        if (l_blk.size() < (size_t)Br) l_blk.resize(Br);
        if (sp.size() < (size_t)Bc) sp.resize(Bc);

#ifdef _OPENMP
#pragma omp for schedule(static, 1) nowait
#endif
        for (int h = 0; h < num_heads; ++h) {
            const float* Qh_src = scratch_qkv           + h * d_k;
            const float* Kh_src = scratch_qkv + D       + h * d_k;
            const float* Vh_src = scratch_qkv + 2 * D   + h * d_k;

            for (int n = 0; n < N; ++n) {
                const float* qs = Qh_src + (size_t)n * 3 * D;
                const float* ks = Kh_src + (size_t)n * 3 * D;
                const float* vs = Vh_src + (size_t)n * 3 * D;
                float* qd = Q_h.data() + (size_t)n * d_k;
                float* kd = K_h.data() + (size_t)n * d_k;
                float* vd = V_h.data() + (size_t)n * d_k;
                int d = 0;
#ifdef __ARM_NEON
                for (; d <= d_k - 4; d += 4) {
                    vst1q_f32(qd + d, vld1q_f32(qs + d));
                    vst1q_f32(kd + d, vld1q_f32(ks + d));
                    vst1q_f32(vd + d, vld1q_f32(vs + d));
                }
#elif defined(__AVX512F__)
                for (; d + 16 <= d_k; d += 16) {
                    _mm512_storeu_ps(qd+d, _mm512_loadu_ps(qs+d));
                    _mm512_storeu_ps(kd+d, _mm512_loadu_ps(ks+d));
                    _mm512_storeu_ps(vd+d, _mm512_loadu_ps(vs+d));
                }
#elif defined(__AVX2__)
                for (; d + 8 <= d_k; d += 8) {
                    _mm256_storeu_ps(qd+d, _mm256_loadu_ps(qs+d));
                    _mm256_storeu_ps(kd+d, _mm256_loadu_ps(ks+d));
                    _mm256_storeu_ps(vd+d, _mm256_loadu_ps(vs+d));
                }
#endif
                for (; d < d_k; ++d) { qd[d] = qs[d]; kd[d] = ks[d]; vd[d] = vs[d]; }
            }

            float* Oh = output + h * d_k;
            const int num_q = (N + Br - 1) / Br;
            const int num_k = (N + Bc - 1) / Bc;

            for (int qi = 0; qi < num_q; ++qi) {
                const int q0   = qi * Br;
                const int q1   = std::min(q0 + Br, N);
                const int qlen = q1 - q0;

                for (int r = 0; r < qlen; ++r) {
                    m_blk[r] = -std::numeric_limits<float>::infinity();
                    l_blk[r] = 0.0f;
                    float* ob = O_blk.data() + r * d_k;
                    int d = 0;
#ifdef __ARM_NEON
                    for (; d <= d_k - 4; d += 4) vst1q_f32(ob + d, vdupq_n_f32(0.0f));
#elif defined(__AVX512F__)
                    { __m512 z = _mm512_setzero_ps();
                      for (; d + 16 <= d_k; d += 16) _mm512_storeu_ps(ob+d, z); }
#elif defined(__AVX2__)
                    { __m256 z = _mm256_setzero_ps();
                      for (; d + 8 <= d_k; d += 8) _mm256_storeu_ps(ob+d, z); }
#endif
                    for (; d < d_k; ++d) ob[d] = 0.0f;
                }

                for (int kj = 0; kj < num_k; ++kj) {
                    const int k0   = kj * Bc;
                    const int k1   = std::min(k0 + Bc, N);
                    const int klen = k1 - k0;

                    for (int r = 0; r < qlen; ++r) {
                        const float* qr = Q_h.data() + (size_t)(q0 + r) * d_k;
                        for (int c = 0; c < klen; ++c) {
                            const float* kc = K_h.data() + (size_t)(k0 + c) * d_k;
                            float s = 0.0f;
#ifdef __ARM_NEON
                            float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
                            float32x4_t s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);
                            int d = 0;
                            for (; d <= d_k - 16; d += 16) {
                                s0 = vfmaq_f32(s0, vld1q_f32(qr+d   ), vld1q_f32(kc+d   ));
                                s1 = vfmaq_f32(s1, vld1q_f32(qr+d+4 ), vld1q_f32(kc+d+4 ));
                                s2 = vfmaq_f32(s2, vld1q_f32(qr+d+8 ), vld1q_f32(kc+d+8 ));
                                s3 = vfmaq_f32(s3, vld1q_f32(qr+d+12), vld1q_f32(kc+d+12));
                            }
                            float32x4_t ss = vaddq_f32(vaddq_f32(s0,s1), vaddq_f32(s2,s3));
                            for (; d <= d_k - 4; d += 4)
                                ss = vfmaq_f32(ss, vld1q_f32(qr+d), vld1q_f32(kc+d));
                            s = vaddvq_f32(ss);
                            for (; d < d_k; ++d) s += qr[d] * kc[d];
#elif defined(__AVX512F__)
                            { __m512 vs = _mm512_setzero_ps();
                              int d = 0;
                              for (; d + 16 <= d_k; d += 16)
                                  vs = _mm512_fmadd_ps(_mm512_loadu_ps(qr+d), _mm512_loadu_ps(kc+d), vs);
                              s = _mm512_reduce_add_ps(vs);
                              for (; d < d_k; ++d) s += qr[d] * kc[d]; }
#elif defined(__AVX2__)
                            { __m256 vs = _mm256_setzero_ps();
                              int d = 0;
                              for (; d + 8 <= d_k; d += 8)
                                  vs = _mm256_fmadd_ps(_mm256_loadu_ps(qr+d), _mm256_loadu_ps(kc+d), vs);
                              s = hsum256(vs);
                              for (; d < d_k; ++d) s += qr[d] * kc[d]; }
#else
                            for (int d = 0; d < d_k; ++d) s += qr[d] * kc[d];
#endif
                            S_buf[r * Bc + c] = s * scale;
                        }
                    }

                    for (int r = 0; r < qlen; ++r) {
                        float* Sr   = S_buf.data() + r * Bc;
                        float* or_  = O_blk.data() + r * d_k;

                        float m_j;
#ifdef __ARM_NEON
                        {
                            float32x4_t vmax = vdupq_n_f32(-std::numeric_limits<float>::infinity());
                            int c = 0;
                            for (; c <= klen - 4; c += 4)
                                vmax = vmaxq_f32(vmax, vld1q_f32(Sr + c));
                            m_j = vmaxvq_f32(vmax);
                            for (; c < klen; ++c) if (Sr[c] > m_j) m_j = Sr[c];
                        }
#elif defined(__AVX512F__)
                        { __m512 vm = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
                          int c = 0;
                          for (; c + 16 <= klen; c += 16)
                              vm = _mm512_max_ps(vm, _mm512_loadu_ps(Sr + c));
                          m_j = _mm512_reduce_max_ps(vm);
                          for (; c < klen; ++c) if (Sr[c] > m_j) m_j = Sr[c]; }
#elif defined(__AVX2__)
                        { __m256 vm = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
                          int c = 0;
                          for (; c + 8 <= klen; c += 8)
                              vm = _mm256_max_ps(vm, _mm256_loadu_ps(Sr + c));
                          __m128 lo = _mm256_castps256_ps128(vm);
                          __m128 hi = _mm256_extractf128_ps(vm, 1);
                          __m128 m4 = _mm_max_ps(lo, hi);
                          m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
                          m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
                          m_j = _mm_cvtss_f32(m4);
                          for (; c < klen; ++c) if (Sr[c] > m_j) m_j = Sr[c]; }
#else
                        m_j = Sr[0];
                        for (int c = 1; c < klen; ++c) if (Sr[c] > m_j) m_j = Sr[c];
#endif
                        float l_j = 0.0f;
#ifdef __ARM_NEON
                        {
                            float32x4_t vm_j = vdupq_n_f32(m_j);
                            float32x4_t vl   = vdupq_n_f32(0.0f);
                            int c = 0;
                            for (; c <= klen - 4; c += 4) {
                                float32x4_t ve = vexpq_f32_fast(
                                    vsubq_f32(vld1q_f32(Sr + c), vm_j));
                                vst1q_f32(Sr + c, ve);
                                vl = vaddq_f32(vl, ve);
                            }
                            l_j = vaddvq_f32(vl);
                            for (; c < klen; ++c) { Sr[c] = expf(Sr[c] - m_j); l_j += Sr[c]; }
                        }
#elif defined(__AVX512F__)
                        { __m512 vm_j = _mm512_set1_ps(m_j), vl = _mm512_setzero_ps();
                          int c = 0;
                          for (; c + 16 <= klen; c += 16) {
                              __m512 ve = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(Sr+c), vm_j));
                              _mm512_storeu_ps(Sr+c, ve);
                              vl = _mm512_add_ps(vl, ve);
                          }
                          l_j = _mm512_reduce_add_ps(vl);
                          for (; c < klen; ++c) { Sr[c] = expf(Sr[c] - m_j); l_j += Sr[c]; } }
#elif defined(__AVX2__)
                        { __m256 vm_j = _mm256_set1_ps(m_j), vl = _mm256_setzero_ps();
                          int c = 0;
                          for (; c + 8 <= klen; c += 8) {
                              __m256 ve = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(Sr+c), vm_j));
                              _mm256_storeu_ps(Sr+c, ve);
                              vl = _mm256_add_ps(vl, ve);
                          }
                          l_j = hsum256(vl);
                          for (; c < klen; ++c) { Sr[c] = expf(Sr[c] - m_j); l_j += Sr[c]; } }
#else
                        for (int c = 0; c < klen; ++c) { Sr[c] = expf(Sr[c] - m_j); l_j += Sr[c]; }
#endif
                        const float m_new  = (m_blk[r] > m_j) ? m_blk[r] : m_j;
                        const float alpha  = expf(m_blk[r] - m_new);
                        const float beta_j = expf(m_j       - m_new);

#ifdef __ARM_NEON
                        {
                            float32x4_t va = vdupq_n_f32(alpha);
                            int d = 0;
                            for (; d <= d_k - 4; d += 4)
                                vst1q_f32(or_+d, vmulq_f32(vld1q_f32(or_+d), va));
                            for (; d < d_k; ++d) or_[d] *= alpha;
                        }
#elif defined(__AVX512F__)
                        { __m512 va = _mm512_set1_ps(alpha);
                          int d = 0;
                          for (; d + 16 <= d_k; d += 16)
                              _mm512_storeu_ps(or_+d, _mm512_mul_ps(_mm512_loadu_ps(or_+d), va));
                          for (; d < d_k; ++d) or_[d] *= alpha; }
#elif defined(__AVX2__)
                        { __m256 va = _mm256_set1_ps(alpha);
                          int d = 0;
                          for (; d + 8 <= d_k; d += 8)
                              _mm256_storeu_ps(or_+d, _mm256_mul_ps(_mm256_loadu_ps(or_+d), va));
                          for (; d < d_k; ++d) or_[d] *= alpha; }
#else
                        for (int d = 0; d < d_k; ++d) or_[d] *= alpha;
#endif
                        for (int c = 0; c < klen; ++c) sp[c] = beta_j * Sr[c];

#ifdef __ARM_NEON
                        for (int c = 0; c < klen; ++c) {
                            float32x4_t vsc  = vdupq_n_f32(sp[c]);
                            const float* vrow = V_h.data() + (size_t)(k0 + c) * d_k;
                            int d = 0;
                            for (; d <= d_k - 16; d += 16) {
                                vst1q_f32(or_+d,    vfmaq_f32(vld1q_f32(or_+d),    vsc, vld1q_f32(vrow+d)));
                                vst1q_f32(or_+d+4,  vfmaq_f32(vld1q_f32(or_+d+4),  vsc, vld1q_f32(vrow+d+4)));
                                vst1q_f32(or_+d+8,  vfmaq_f32(vld1q_f32(or_+d+8),  vsc, vld1q_f32(vrow+d+8)));
                                vst1q_f32(or_+d+12, vfmaq_f32(vld1q_f32(or_+d+12), vsc, vld1q_f32(vrow+d+12)));
                            }
                            for (; d <= d_k - 4; d += 4)
                                vst1q_f32(or_+d, vfmaq_f32(vld1q_f32(or_+d), vsc, vld1q_f32(vrow+d)));
                            for (; d < d_k; ++d) or_[d] += sp[c] * vrow[d];
                        }
#elif defined(__AVX512F__)
                        for (int c = 0; c < klen; ++c) {
                            __m512 vsc = _mm512_set1_ps(sp[c]);
                            const float* vrow = V_h.data() + (size_t)(k0 + c) * d_k;
                            int d = 0;
                            for (; d + 16 <= d_k; d += 16)
                                _mm512_storeu_ps(or_+d, _mm512_fmadd_ps(vsc, _mm512_loadu_ps(vrow+d), _mm512_loadu_ps(or_+d)));
                            for (; d < d_k; ++d) or_[d] += sp[c] * vrow[d];
                        }
#elif defined(__AVX2__)
                        for (int c = 0; c < klen; ++c) {
                            __m256 vsc = _mm256_set1_ps(sp[c]);
                            const float* vrow = V_h.data() + (size_t)(k0 + c) * d_k;
                            int d = 0;
                            for (; d + 8 <= d_k; d += 8)
                                _mm256_storeu_ps(or_+d, _mm256_fmadd_ps(vsc, _mm256_loadu_ps(vrow+d), _mm256_loadu_ps(or_+d)));
                            for (; d < d_k; ++d) or_[d] += sp[c] * vrow[d];
                        }
#else
                        for (int c = 0; c < klen; ++c) {
                            const float* vrow = V_h.data() + (size_t)(k0 + c) * d_k;
                            for (int d = 0; d < d_k; ++d) or_[d] += sp[c] * vrow[d];
                        }
#endif
                        l_blk[r] = alpha * l_blk[r] + beta_j * l_j;
                        m_blk[r] = m_new;
                    }
                }

                for (int r = 0; r < qlen; ++r) {
                    float inv_l = 1.0f / l_blk[r];
                    float* or_  = O_blk.data() + r * d_k;
                    float* dst  = Oh + (size_t)(q0 + r) * D;
#ifdef __ARM_NEON
                    float32x4_t vinvl = vdupq_n_f32(inv_l);
                    int d = 0;
                    for (; d <= d_k - 4; d += 4)
                        vst1q_f32(dst + d, vmulq_f32(vld1q_f32(or_ + d), vinvl));
                    for (; d < d_k; ++d) dst[d] = or_[d] * inv_l;
#elif defined(__AVX512F__)
                    { __m512 vinvl = _mm512_set1_ps(inv_l);
                      int d = 0;
                      for (; d + 16 <= d_k; d += 16)
                          _mm512_storeu_ps(dst+d, _mm512_mul_ps(_mm512_loadu_ps(or_+d), vinvl));
                      for (; d < d_k; ++d) dst[d] = or_[d] * inv_l; }
#elif defined(__AVX2__)
                    { __m256 vinvl = _mm256_set1_ps(inv_l);
                      int d = 0;
                      for (; d + 8 <= d_k; d += 8)
                          _mm256_storeu_ps(dst+d, _mm256_mul_ps(_mm256_loadu_ps(or_+d), vinvl));
                      for (; d < d_k; ++d) dst[d] = or_[d] * inv_l; }
#else
                    for (int d = 0; d < d_k; ++d) dst[d] = or_[d] * inv_l;
#endif
                }
            }
        }
#ifdef _OPENMP
    }
#endif

    float* tmp = scratch_qkv;
    memcpy(tmp, output, (size_t)N * D * sizeof(float));

    // Simulate ONNX DynamicQuantizeLinear on attention output before output projection
    if (dq_attn_out)
        dynamic_quantize_dequant_f32(tmp, N * D);

    sgemm_f32(tmp, w_proj_packed, b_proj, output, false, N, D, D, false);

    g_vit_attn_ms += now_ms() - t0;
}

// ─────────────────────────────────────────────────────────────────
// INT8 Sequence GEMM
// ─────────────────────────────────────────────────────────────────

void seqgemm_int8(
    const float*   input,
    const int8_t*  w_packed,
    const float*   w_scales,
    const int64_t* eff_zeros,
    const float*   bias,
    int8_t*        inp_scratch,
    float*         req_scratch,
    float*         output,
    int N, int C_in, int C_out,
    bool gelu,
    StreamHandle /* stream */)
{
    double t0 = now_ms();
    seqgemm_int8_raw(input, w_packed, w_scales, eff_zeros, bias,
                     inp_scratch, req_scratch, output, N, C_in, C_out, nullptr);
    g_vit_seqgemm_ms += now_ms() - t0;
    if (gelu) gelu_fp32(output, N * C_out);
}

void seqgemm_int8_with_rows(
    const float*   input,
    const int8_t*  w_packed,
    const float*   w_scales,
    const int64_t* eff_zeros,
    const float*   bias,
    int8_t*        inp_scratch,
    float*         req_scratch,
    float*         output,
    int N, int C_in, int C_out,
    bool gelu,
    const int32_t* w_row_sums,
    StreamHandle /* stream */)
{
    double t0 = now_ms();
    seqgemm_int8_raw(input, w_packed, w_scales, eff_zeros, bias,
                     inp_scratch, req_scratch, output, N, C_in, C_out,
                     w_row_sums);
    g_vit_seqgemm_ms += now_ms() - t0;
    if (gelu) gelu_fp32(output, N * C_out);
}

// ─────────────────────────────────────────────────────────────────
// seqgemm_int8_matvec — single-token decode GEMV (N=1)
//
// W8A16: weight_i8[C_out, C_in] INT8 × input[C_in] FP32 → output[C_out] FP32
//
// Strategy:
//   1. Quantize the FP32 input to INT8 once (all output rows share it).
//   2. OMP-parallel loop over output channels in blocks of 8.
//   3. Inner loop uses SDOT (vdotq_s32) to accumulate 16 elements/step.
//      The input vector (16 bytes) is loaded once per K-step and reused
//      across all 8 output channels → maximises arithmetic intensity.
//   4. No packing overhead; streams weight rows sequentially (cache-friendly).
// ─────────────────────────────────────────────────────────────────

void seqgemm_int8_matvec(
    const float*   input,
    const int8_t*  weight_i8,
    const float*   w_scales,
    const float*   bias,
    float*         output,
    int            C_in,
    int            C_out,
    StreamHandle   /* stream */)
{
    const double t0_mv = now_ms();
    // Quantize input once; all OMP threads share it read-only.
    static thread_local std::vector<int8_t> tls_ai8;
    tls_ai8.resize(C_in);
    const float a_scale = quantize_to_int8_neon(input, tls_ai8.data(), C_in);
    const int8_t* ai8   = tls_ai8.data();

#ifdef __ARM_FEATURE_DOTPROD
    // ── SDOT path: sequential 1-OC loop — single streaming read per row ──
    // Hardware prefetcher sees one sequential stream → near-peak DRAM utilisation.
    // K×4 unroll: 64 bytes (1 cache line) per step, 4 accumulators for pipelining.
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
#endif
    for (int co = 0; co < C_out; ++co) {
        const int8_t* W = weight_i8 + (size_t)co * C_in;
        int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
        int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
        int k = 0;
        for (; k + 64 <= C_in; k += 64) {
            a0 = vdotq_s32(a0, vld1q_s8(W + k),      vld1q_s8(ai8 + k));
            a1 = vdotq_s32(a1, vld1q_s8(W + k + 16), vld1q_s8(ai8 + k + 16));
            a2 = vdotq_s32(a2, vld1q_s8(W + k + 32), vld1q_s8(ai8 + k + 32));
            a3 = vdotq_s32(a3, vld1q_s8(W + k + 48), vld1q_s8(ai8 + k + 48));
        }
        for (; k + 16 <= C_in; k += 16)
            a0 = vdotq_s32(a0, vld1q_s8(W + k), vld1q_s8(ai8 + k));
        int32_t acc = vaddvq_s32(vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3)));
        for (; k < C_in; ++k) acc += (int32_t)W[k] * (int32_t)ai8[k];
        output[co] = (float)acc * a_scale * w_scales[co] + (bias ? bias[co] : 0.f);
    }

#elif defined(__AVX512BW__)
    // ── AVX512BW path: 64 int8 elements per step, 4-accumulator unroll ──
    // _mm512_cvtepi8_epi16: 32 int8 (256-bit) → 32 int16 (512-bit)
    // _mm512_madd_epi16: 32 int16 pairs → 16 int32 (adjacent pair sums)
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
#endif
    for (int co = 0; co < C_out; ++co) {
        const int8_t* W = weight_i8 + (size_t)co * C_in;
        __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
        __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
        int k = 0;
        for (; k + 128 <= C_in; k += 128) {
            a0 = _mm512_add_epi32(a0, _mm512_madd_epi16(
                _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(W+k))),
                _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(ai8+k)))));
            a1 = _mm512_add_epi32(a1, _mm512_madd_epi16(
                _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(W+k+32))),
                _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(ai8+k+32)))));
            a2 = _mm512_add_epi32(a2, _mm512_madd_epi16(
                _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(W+k+64))),
                _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(ai8+k+64)))));
            a3 = _mm512_add_epi32(a3, _mm512_madd_epi16(
                _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(W+k+96))),
                _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(ai8+k+96)))));
        }
        for (; k + 32 <= C_in; k += 32) {
            a0 = _mm512_add_epi32(a0, _mm512_madd_epi16(
                _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(W+k))),
                _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(ai8+k)))));
        }
        __m512i acc = _mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3));
        int32_t sum = _mm512_reduce_add_epi32(acc);
        for (; k < C_in; ++k) sum += (int32_t)W[k] * (int32_t)ai8[k];
        output[co] = (float)sum * a_scale * w_scales[co] + (bias ? bias[co] : 0.f);
    }

#elif defined(__AVX2__)
    // ── AVX2 path: 64 int8 elements per step, 4-accumulator unroll ──
    // _mm256_cvtepi8_epi16: 16 int8 (128-bit) → 16 int16 (256-bit)
    // _mm256_madd_epi16: 16 int16 pairs → 8 int32
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
#endif
    for (int co = 0; co < C_out; ++co) {
        const int8_t* W = weight_i8 + (size_t)co * C_in;
        __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
        __m256i a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
        int k = 0;
        for (; k + 64 <= C_in; k += 64) {
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(W+k))),
                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(ai8+k)))));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(W+k+16))),
                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(ai8+k+16)))));
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(W+k+32))),
                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(ai8+k+32)))));
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(W+k+48))),
                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(ai8+k+48)))));
        }
        for (; k + 16 <= C_in; k += 16) {
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(W+k))),
                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(ai8+k)))));
        }
        __m256i acc = _mm256_add_epi32(_mm256_add_epi32(a0, a1), _mm256_add_epi32(a2, a3));
        // Horizontal sum of 8 int32s
        __m128i lo  = _mm256_castsi256_si128(acc);
        __m128i hi  = _mm256_extracti128_si256(acc, 1);
        __m128i s4  = _mm_add_epi32(lo, hi);
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, _MM_SHUFFLE(1,0,3,2)));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, _MM_SHUFFLE(0,1,0,1)));
        int32_t sum = _mm_cvtsi128_si32(s4);
        for (; k < C_in; ++k) sum += (int32_t)W[k] * (int32_t)ai8[k];
        output[co] = (float)sum * a_scale * w_scales[co] + (bias ? bias[co] : 0.f);
    }

#else
    // ── Scalar fallback ──────────────────────────────────────────────
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
#endif
    for (int co = 0; co < C_out; ++co) {
        const int8_t* w = weight_i8 + (size_t)co * C_in;
        int32_t acc = 0;
        for (int k = 0; k < C_in; ++k) acc += (int32_t)w[k] * (int32_t)ai8[k];
        output[co] = (float)acc * a_scale * w_scales[co] + (bias ? bias[co] : 0.f);
    }
#endif
    g_lm_matvec_ms += now_ms() - t0_mv;
}

// ─────────────────────────────────────────────────────────────────
// INT8 Flash Attention-2 Multi-Head Self-Attention
// ─────────────────────────────────────────────────────────────────

static void attention_flash_int8_impl(
    const float*   input,
    const int8_t*  w_qkv_packed,
    const float*   w_qkv_scales,
    const int64_t* eff_zeros_qkv,
    const float*   b_qkv,
    const int8_t*  w_proj_packed,
    const float*   w_proj_scales,
    const int64_t* eff_zeros_proj,
    const float*   b_proj,
    float*         output,
    int N, int D, int num_heads,
    float*         scratch_qkv,
    int8_t*        scratch_int8,
    float*         scratch_req,
    const int32_t* qkv_row_sums,
    const int32_t* proj_row_sums,
    StreamHandle /* stream */)
{
    double t0 = now_ms();

    const int   d_k   = D / num_heads;
    const float scale = 1.0f / sqrtf((float)d_k);

    seqgemm_int8_raw(input, w_qkv_packed, w_qkv_scales, eff_zeros_qkv, b_qkv,
                     scratch_int8, scratch_req, scratch_qkv, N, D, 3 * D,
                     qkv_row_sums);

    memset(output, 0, (size_t)N * D * sizeof(float));

    constexpr int Br = 64;
    constexpr int Bc = 64;

#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        // Thread-local scratch: persists between calls; reallocates only when needed.
        static thread_local std::vector<float> Q_h, K_h, V_h, S_buf, O_blk, m_blk, l_blk, sp;
        const size_t head_seq = (size_t)N * d_k;
        if (Q_h.size() < head_seq) Q_h.resize(head_seq);
        if (K_h.size() < head_seq) K_h.resize(head_seq);
        if (V_h.size() < head_seq) V_h.resize(head_seq);
        if (S_buf.size() < (size_t)Br * Bc) S_buf.resize((size_t)Br * Bc);
        const size_t oblk_need = (size_t)Br * d_k;
        if (O_blk.size() < oblk_need) O_blk.resize(oblk_need);
        if (m_blk.size() < (size_t)Br) m_blk.resize(Br);
        if (l_blk.size() < (size_t)Br) l_blk.resize(Br);
        if (sp.size() < (size_t)Bc) sp.resize(Bc);

#ifdef _OPENMP
#pragma omp for schedule(static, 1) nowait
#endif
        for (int h = 0; h < num_heads; ++h) {
            const float* Qh_src = scratch_qkv           + h * d_k;
            const float* Kh_src = scratch_qkv + D       + h * d_k;
            const float* Vh_src = scratch_qkv + 2 * D   + h * d_k;

            for (int n = 0; n < N; ++n) {
                const float* qs = Qh_src + (size_t)n * 3 * D;
                const float* ks = Kh_src + (size_t)n * 3 * D;
                const float* vs = Vh_src + (size_t)n * 3 * D;
                float* qd = Q_h.data() + (size_t)n * d_k;
                float* kd = K_h.data() + (size_t)n * d_k;
                float* vd = V_h.data() + (size_t)n * d_k;
                int d = 0;
#ifdef __ARM_NEON
                for (; d <= d_k - 4; d += 4) {
                    vst1q_f32(qd + d, vld1q_f32(qs + d));
                    vst1q_f32(kd + d, vld1q_f32(ks + d));
                    vst1q_f32(vd + d, vld1q_f32(vs + d));
                }
#elif defined(__AVX512F__)
                for (; d + 16 <= d_k; d += 16) {
                    _mm512_storeu_ps(qd+d, _mm512_loadu_ps(qs+d));
                    _mm512_storeu_ps(kd+d, _mm512_loadu_ps(ks+d));
                    _mm512_storeu_ps(vd+d, _mm512_loadu_ps(vs+d));
                }
#elif defined(__AVX2__)
                for (; d + 8 <= d_k; d += 8) {
                    _mm256_storeu_ps(qd+d, _mm256_loadu_ps(qs+d));
                    _mm256_storeu_ps(kd+d, _mm256_loadu_ps(ks+d));
                    _mm256_storeu_ps(vd+d, _mm256_loadu_ps(vs+d));
                }
#endif
                for (; d < d_k; ++d) { qd[d] = qs[d]; kd[d] = ks[d]; vd[d] = vs[d]; }
            }

            float* Oh = output + h * d_k;
            const int num_q = (N + Br - 1) / Br;
            const int num_k = (N + Bc - 1) / Bc;

            for (int qi = 0; qi < num_q; ++qi) {
                const int q0   = qi * Br;
                const int q1   = std::min(q0 + Br, N);
                const int qlen = q1 - q0;

                for (int r = 0; r < qlen; ++r) {
                    m_blk[r] = -std::numeric_limits<float>::infinity();
                    l_blk[r] = 0.0f;
                    float* ob = O_blk.data() + r * d_k;
                    int d = 0;
#ifdef __ARM_NEON
                    for (; d <= d_k - 4; d += 4) vst1q_f32(ob + d, vdupq_n_f32(0.0f));
#elif defined(__AVX512F__)
                    { __m512 z = _mm512_setzero_ps();
                      for (; d + 16 <= d_k; d += 16) _mm512_storeu_ps(ob+d, z); }
#elif defined(__AVX2__)
                    { __m256 z = _mm256_setzero_ps();
                      for (; d + 8 <= d_k; d += 8) _mm256_storeu_ps(ob+d, z); }
#endif
                    for (; d < d_k; ++d) ob[d] = 0.0f;
                }

                for (int kj = 0; kj < num_k; ++kj) {
                    const int k0   = kj * Bc;
                    const int k1   = std::min(k0 + Bc, N);
                    const int klen = k1 - k0;

                    for (int r = 0; r < qlen; ++r) {
                        const float* qr = Q_h.data() + (size_t)(q0 + r) * d_k;
                        for (int c = 0; c < klen; ++c) {
                            const float* kc = K_h.data() + (size_t)(k0 + c) * d_k;
                            float s = 0.0f;
#ifdef __ARM_NEON
                            float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
                            float32x4_t s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);
                            int d = 0;
                            for (; d <= d_k - 16; d += 16) {
                                s0 = vfmaq_f32(s0, vld1q_f32(qr+d   ), vld1q_f32(kc+d   ));
                                s1 = vfmaq_f32(s1, vld1q_f32(qr+d+4 ), vld1q_f32(kc+d+4 ));
                                s2 = vfmaq_f32(s2, vld1q_f32(qr+d+8 ), vld1q_f32(kc+d+8 ));
                                s3 = vfmaq_f32(s3, vld1q_f32(qr+d+12), vld1q_f32(kc+d+12));
                            }
                            float32x4_t ss = vaddq_f32(vaddq_f32(s0,s1), vaddq_f32(s2,s3));
                            for (; d <= d_k - 4; d += 4)
                                ss = vfmaq_f32(ss, vld1q_f32(qr+d), vld1q_f32(kc+d));
                            s = vaddvq_f32(ss);
                            for (; d < d_k; ++d) s += qr[d] * kc[d];
#elif defined(__AVX512F__)
                            { __m512 vs = _mm512_setzero_ps();
                              int d = 0;
                              for (; d + 16 <= d_k; d += 16)
                                  vs = _mm512_fmadd_ps(_mm512_loadu_ps(qr+d), _mm512_loadu_ps(kc+d), vs);
                              s = _mm512_reduce_add_ps(vs);
                              for (; d < d_k; ++d) s += qr[d] * kc[d]; }
#elif defined(__AVX2__)
                            { __m256 vs = _mm256_setzero_ps();
                              int d = 0;
                              for (; d + 8 <= d_k; d += 8)
                                  vs = _mm256_fmadd_ps(_mm256_loadu_ps(qr+d), _mm256_loadu_ps(kc+d), vs);
                              s = hsum256(vs);
                              for (; d < d_k; ++d) s += qr[d] * kc[d]; }
#else
                            for (int d = 0; d < d_k; ++d) s += qr[d] * kc[d];
#endif
                            S_buf[r * Bc + c] = s * scale;
                        }
                    }

                    for (int r = 0; r < qlen; ++r) {
                        float* Sr  = S_buf.data() + r * Bc;
                        float* or_ = O_blk.data() + r * d_k;

                        float m_j;
#ifdef __ARM_NEON
                        {
                            float32x4_t vmax = vdupq_n_f32(-std::numeric_limits<float>::infinity());
                            int c = 0;
                            for (; c <= klen - 4; c += 4)
                                vmax = vmaxq_f32(vmax, vld1q_f32(Sr + c));
                            m_j = vmaxvq_f32(vmax);
                            for (; c < klen; ++c) if (Sr[c] > m_j) m_j = Sr[c];
                        }
#else
                        m_j = Sr[0]; for (int c = 1; c < klen; ++c) if (Sr[c] > m_j) m_j = Sr[c];
#endif
                        float l_j = 0.0f;
#ifdef __ARM_NEON
                        {
                            float32x4_t vm_j = vdupq_n_f32(m_j), vl = vdupq_n_f32(0.0f);
                            int c = 0;
                            for (; c <= klen - 4; c += 4) {
                                float32x4_t ve = vexpq_f32_fast(vsubq_f32(vld1q_f32(Sr + c), vm_j));
                                vst1q_f32(Sr + c, ve);
                                vl = vaddq_f32(vl, ve);
                            }
                            l_j = vaddvq_f32(vl);
                            for (; c < klen; ++c) { Sr[c] = expf(Sr[c] - m_j); l_j += Sr[c]; }
                        }
#else
                        for (int c = 0; c < klen; ++c) { Sr[c] = expf(Sr[c] - m_j); l_j += Sr[c]; }
#endif
                        const float m_new  = (m_blk[r] > m_j) ? m_blk[r] : m_j;
                        const float alpha  = expf(m_blk[r] - m_new);
                        const float beta_j = expf(m_j       - m_new);
#ifdef __ARM_NEON
                        {
                            float32x4_t va = vdupq_n_f32(alpha);
                            int d = 0;
                            for (; d <= d_k - 4; d += 4)
                                vst1q_f32(or_+d, vmulq_f32(vld1q_f32(or_+d), va));
                            for (; d < d_k; ++d) or_[d] *= alpha;
                        }
#else
                        for (int d = 0; d < d_k; ++d) or_[d] *= alpha;
#endif
                        for (int c = 0; c < klen; ++c) sp[c] = beta_j * Sr[c];
#ifdef __ARM_NEON
                        for (int c = 0; c < klen; ++c) {
                            float32x4_t vsc = vdupq_n_f32(sp[c]);
                            const float* vrow = V_h.data() + (size_t)(k0 + c) * d_k;
                            int d = 0;
                            for (; d <= d_k - 16; d += 16) {
                                vst1q_f32(or_+d,    vfmaq_f32(vld1q_f32(or_+d),    vsc, vld1q_f32(vrow+d)));
                                vst1q_f32(or_+d+4,  vfmaq_f32(vld1q_f32(or_+d+4),  vsc, vld1q_f32(vrow+d+4)));
                                vst1q_f32(or_+d+8,  vfmaq_f32(vld1q_f32(or_+d+8),  vsc, vld1q_f32(vrow+d+8)));
                                vst1q_f32(or_+d+12, vfmaq_f32(vld1q_f32(or_+d+12), vsc, vld1q_f32(vrow+d+12)));
                            }
                            for (; d <= d_k - 4; d += 4)
                                vst1q_f32(or_+d, vfmaq_f32(vld1q_f32(or_+d), vsc, vld1q_f32(vrow+d)));
                            for (; d < d_k; ++d) or_[d] += sp[c] * vrow[d];
                        }
#else
                        for (int c = 0; c < klen; ++c) {
                            const float* vrow = V_h.data() + (size_t)(k0 + c) * d_k;
                            for (int d = 0; d < d_k; ++d) or_[d] += sp[c] * vrow[d];
                        }
#endif
                        l_blk[r] = alpha * l_blk[r] + beta_j * l_j;
                        m_blk[r] = m_new;
                    }
                }

                for (int r = 0; r < qlen; ++r) {
                    float inv_l = 1.0f / l_blk[r];
                    float* or_  = O_blk.data() + r * d_k;
                    float* dst  = Oh + (size_t)(q0 + r) * D;
#ifdef __ARM_NEON
                    float32x4_t vinvl = vdupq_n_f32(inv_l);
                    int d = 0;
                    for (; d <= d_k - 4; d += 4)
                        vst1q_f32(dst + d, vmulq_f32(vld1q_f32(or_ + d), vinvl));
                    for (; d < d_k; ++d) dst[d] = or_[d] * inv_l;
#elif defined(__AVX512F__)
                    { __m512 vinvl = _mm512_set1_ps(inv_l);
                      int d = 0;
                      for (; d + 16 <= d_k; d += 16)
                          _mm512_storeu_ps(dst+d, _mm512_mul_ps(_mm512_loadu_ps(or_+d), vinvl));
                      for (; d < d_k; ++d) dst[d] = or_[d] * inv_l; }
#elif defined(__AVX2__)
                    { __m256 vinvl = _mm256_set1_ps(inv_l);
                      int d = 0;
                      for (; d + 8 <= d_k; d += 8)
                          _mm256_storeu_ps(dst+d, _mm256_mul_ps(_mm256_loadu_ps(or_+d), vinvl));
                      for (; d < d_k; ++d) dst[d] = or_[d] * inv_l; }
#else
                    for (int d = 0; d < d_k; ++d) dst[d] = or_[d] * inv_l;
#endif
                }
            }
        }
#ifdef _OPENMP
    }
#endif

    float* tmp = scratch_qkv;
    memcpy(tmp, output, (size_t)N * D * sizeof(float));
    seqgemm_int8_raw(tmp, w_proj_packed, w_proj_scales, eff_zeros_proj, b_proj,
                     scratch_int8, scratch_req, output, N, D, D,
                     proj_row_sums);

    g_vit_attn_ms += now_ms() - t0;
}

void attention_flash_int8(
    const float*   input,
    const int8_t*  w_qkv_packed,
    const float*   w_qkv_scales,
    const int64_t* eff_zeros_qkv,
    const float*   b_qkv,
    const int8_t*  w_proj_packed,
    const float*   w_proj_scales,
    const int64_t* eff_zeros_proj,
    const float*   b_proj,
    float*         output,
    int N, int D, int num_heads,
    float*         scratch_qkv,
    int8_t*        scratch_int8,
    float*         scratch_req,
    StreamHandle   stream)
{
    attention_flash_int8_impl(input,
                              w_qkv_packed, w_qkv_scales, eff_zeros_qkv, b_qkv,
                              w_proj_packed, w_proj_scales, eff_zeros_proj, b_proj,
                              output, N, D, num_heads,
                              scratch_qkv, scratch_int8, scratch_req,
                              nullptr, nullptr, stream);
}

void attention_flash_int8_with_rows(
    const float*   input,
    const int8_t*  w_qkv_packed,
    const float*   w_qkv_scales,
    const int64_t* eff_zeros_qkv,
    const float*   b_qkv,
    const int8_t*  w_proj_packed,
    const float*   w_proj_scales,
    const int64_t* eff_zeros_proj,
    const float*   b_proj,
    float*         output,
    int N, int D, int num_heads,
    float*         scratch_qkv,
    int8_t*        scratch_int8,
    float*         scratch_req,
    const int32_t* qkv_row_sums,
    const int32_t* proj_row_sums,
    StreamHandle   stream)
{
    attention_flash_int8_impl(input,
                              w_qkv_packed, w_qkv_scales, eff_zeros_qkv, b_qkv,
                              w_proj_packed, w_proj_scales, eff_zeros_proj, b_proj,
                              output, N, D, num_heads,
                              scratch_qkv, scratch_int8, scratch_req,
                              qkv_row_sums, proj_row_sums, stream);
}

// ─────────────────────────────────────────────────────────────────
// apply_rope_inplace
//   Apply Rotary Position Embeddings to q [N, num_heads, head_dim]
//   and k [N, num_kv_heads, head_dim] in-place.
//   Positions: [step_pos, step_pos+1, ..., step_pos+N-1]
//   theta_i = 1 / (rope_theta ^ (2i / head_dim))
// ─────────────────────────────────────────────────────────────────

static void apply_rope_inplace(float* qk, int N, int num_q_heads, int num_kv_heads,
                                int head_dim, int step_pos, float rope_theta)
{
    const int total_heads = num_q_heads + num_kv_heads;
    const int half_hd = head_dim / 2;

    for (int t = 0; t < N; ++t) {
        int pos = step_pos + t;
        float* qk_t = qk + (size_t)t * total_heads * head_dim;

        for (int h = 0; h < total_heads; ++h) {
            float* vec = qk_t + h * head_dim;
            for (int i = 0; i < half_hd; ++i) {
                float theta = (float)pos / powf(rope_theta, (float)(2 * i) / (float)head_dim);
                float cos_t = cosf(theta);
                float sin_t = sinf(theta);
                float v0    = vec[i];
                float v1    = vec[i + half_hd];
                vec[i]          = v0 * cos_t - v1 * sin_t;
                vec[i + half_hd] = v0 * sin_t + v1 * cos_t;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// causal_attn_core_fp32
//   RoPE + KV-cache update + causal softmax attention
//
//   Layout of q: [N, num_q_heads * head_dim]  (row-major)
//   Layout of k: [N, num_kv_heads * head_dim]
//   Layout of v: [N, num_kv_heads * head_dim]
//   kv_cache: [2, cache_len, num_kv_heads * head_dim]
//   output: [N, num_q_heads * head_dim]
// ─────────────────────────────────────────────────────────────────

void causal_attn_core_fp32(
    const float* q_in,
    const float* k_in,
    const float* v_in,
    float*       kv_cache,
    int          cache_pos,
    int          cache_len,
    float*       output,
    int N, int num_q_heads, int num_kv_heads, int head_dim,
    int          step_pos,
    float        rope_theta,
    bool         is_decode,
    float*       scratch,
    StreamHandle /* stream */)
{
    const double t0_attn = now_ms();
    (void)is_decode;  // used for clarity only — behaviour is identical between modes
    const int nq   = num_q_heads;
    const int nkv  = num_kv_heads;
    const int hd   = head_dim;
    const int nqkv = nq + nkv;  // total heads for RoPE scratch
    const int kv_stride = nkv * hd;  // floats per row in K/V cache
    const int q_stride  = nq  * hd;  // floats per row in Q

    float* kv_k = kv_cache;                               // [cache_len, nkv*hd]
    float* kv_v = kv_cache + (size_t)cache_len * kv_stride; // [cache_len, nkv*hd]

    // ── Step 1: copy Q and QK into RoPE scratch, apply RoPE in-place ──
    // scratch layout: [N, (nq + nkv) * hd]
    float* rope_buf = scratch;
    for (int t = 0; t < N; ++t) {
        float* dst = rope_buf + (size_t)t * nqkv * hd;
        memcpy(dst,          q_in + (size_t)t * q_stride,  (size_t)nq  * hd * sizeof(float));
        memcpy(dst + nq * hd, k_in + (size_t)t * kv_stride, (size_t)nkv * hd * sizeof(float));
    }
    apply_rope_inplace(rope_buf, N, nq, nkv, hd, step_pos, rope_theta);

    // ── Step 2: write K, V into KV cache ──
    for (int t = 0; t < N; ++t) {
        int pos = cache_pos + t;
        const float* k_t = rope_buf + (size_t)t * nqkv * hd + nq * hd;  // K after RoPE
        const float* v_t = v_in    + (size_t)t * kv_stride;
        memcpy(kv_k + (size_t)pos * kv_stride, k_t, (size_t)kv_stride * sizeof(float));
        memcpy(kv_v + (size_t)pos * kv_stride, v_t, (size_t)kv_stride * sizeof(float));
    }

    // ── Step 3: compute attention ──
    const int total_kv_len = cache_pos + N;  // number of valid K/V tokens
    const float scale = 1.0f / sqrtf((float)hd);
    const int group_size = nq / nkv;  // GQA: each KV head serves group_size Q heads

    // scratch2: attention score buffer [nq * total_kv_len] for one query position
    float* score_buf = scratch + (size_t)N * nqkv * hd;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(2) if(!omp_in_parallel())
#endif
    for (int t = 0; t < N; ++t) {
        for (int qh = 0; qh < nq; ++qh) {
            const float* q_t = rope_buf + (size_t)t * nqkv * hd;
            float* out_t     = output   + (size_t)t * q_stride;
            float* scores_t  = score_buf + (size_t)t * (size_t)nq * total_kv_len;
            const int causal_kv_end = cache_pos + t + 1;
            const int kvh = qh / group_size;
            const float* q_vec = q_t + qh * hd;
            float* s = scores_t + qh * total_kv_len;

            for (int kp = 0; kp < causal_kv_end; ++kp) {
                const float* k_vec = kv_k + (size_t)kp * kv_stride + kvh * hd;
                float dot = 0.0f;
                int d = 0;
#ifdef __ARM_NEON
                { float32x4_t vs = vdupq_n_f32(0.f);
                  for (; d + 4 <= hd; d += 4) vs = vfmaq_f32(vs, vld1q_f32(q_vec + d), vld1q_f32(k_vec + d));
                  dot = vaddvq_f32(vs); }
#elif defined(__AVX2__)
                { __m256 vs = _mm256_setzero_ps();
                  for (; d + 8 <= hd; d += 8) vs = _mm256_fmadd_ps(_mm256_loadu_ps(q_vec + d), _mm256_loadu_ps(k_vec + d), vs);
                  __m128 lo = _mm256_castps256_ps128(vs); __m128 hi = _mm256_extractf128_ps(vs, 1);
                  lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
                  dot = _mm_cvtss_f32(lo); }
#endif
                for (; d < hd; ++d) dot += q_vec[d] * k_vec[d];
                s[kp] = dot * scale;
            }
            // Causal masking: positions beyond causal_kv_end are -inf
            for (int kp = causal_kv_end; kp < total_kv_len; ++kp) s[kp] = -1e30f;

            // Softmax over [0, total_kv_len)
            float max_s = s[0];
            for (int kp = 1; kp < causal_kv_end; ++kp) if (s[kp] > max_s) max_s = s[kp];
            float sum = 0.0f;
            for (int kp = 0; kp < causal_kv_end; ++kp) { s[kp] = expf(s[kp] - max_s); sum += s[kp]; }
            const float inv_sum = (sum > 0.f) ? 1.0f / sum : 0.0f;
            for (int kp = 0; kp < causal_kv_end; ++kp) s[kp] *= inv_sum;

            // Weighted sum of V: out_t[qh, :] = sum_kp(s[kp] * V_cache[kp, kvh, :])
            float* out_h = out_t + qh * hd;
            memset(out_h, 0, hd * sizeof(float));
            for (int kp = 0; kp < causal_kv_end; ++kp) {
                const float* v_vec = kv_v + (size_t)kp * kv_stride + kvh * hd;
                float w = s[kp];
                int d2 = 0;
#ifdef __ARM_NEON
                { float32x4_t vw = vdupq_n_f32(w);
                  for (; d2 + 4 <= hd; d2 += 4)
                      vst1q_f32(out_h + d2, vfmaq_f32(vld1q_f32(out_h + d2), vw, vld1q_f32(v_vec + d2))); }
#elif defined(__AVX2__)
                { __m256 vw = _mm256_set1_ps(w);
                  for (; d2 + 8 <= hd; d2 += 8)
                      _mm256_storeu_ps(out_h + d2, _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_vec + d2),
                                                                     _mm256_loadu_ps(out_h + d2))); }
#endif
                for (; d2 < hd; ++d2) out_h[d2] += w * v_vec[d2];
            }
        }
    }
    g_lm_attn_ms += now_ms() - t0_attn;
}
