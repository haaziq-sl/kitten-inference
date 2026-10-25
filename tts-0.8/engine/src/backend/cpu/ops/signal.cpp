// signal.cpp — Signal processing kernels: upsample, STFT, ISTFT, sine gen

#include "../ops_neon.hpp"
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─────────────────────────────────────────────────────────────────
// Nearest-neighbour 1D upsampling
// ─────────────────────────────────────────────────────────────────

void upsample_nearest1d_fp32(const float* in, float* out, int T, int C, int sf, StreamHandle /* stream */)
{
    for (int t = 0; t < T; ++t) {
        const float* src = in + (size_t)t * C;
        for (int r = 0; r < sf; ++r)
            memcpy(out + ((size_t)t * sf + r) * C, src, C * sizeof(float));
    }
}

// ─────────────────────────────────────────────────────────────────
// Bilinear 2D upsampling (align_corners=False)
// ─────────────────────────────────────────────────────────────────

void upsample_bilinear2d_fp32(const float* in, float* out,
                               int C, int iH, int iW, int oH, int oW, StreamHandle /* stream */)
{
    for (int oh = 0; oh < oH; ++oh) {
        float fh = (oh + 0.5f) * iH / oH - 0.5f;
        int ih0 = (int)floorf(fh);
        int ih1 = ih0 + 1;
        float wh1 = fh - ih0;
        float wh0 = 1.f - wh1;
        ih0 = std::max(0, std::min(ih0, iH - 1));
        ih1 = std::max(0, std::min(ih1, iH - 1));

        for (int ow = 0; ow < oW; ++ow) {
            float fw = (ow + 0.5f) * iW / oW - 0.5f;
            int iw0 = (int)floorf(fw);
            int iw1 = iw0 + 1;
            float ww1 = fw - iw0;
            float ww0 = 1.f - ww1;
            iw0 = std::max(0, std::min(iw0, iW - 1));
            iw1 = std::max(0, std::min(iw1, iW - 1));

            const float* p00 = in + (ih0 * iW + iw0) * C;
            const float* p01 = in + (ih0 * iW + iw1) * C;
            const float* p10 = in + (ih1 * iW + iw0) * C;
            const float* p11 = in + (ih1 * iW + iw1) * C;
            float* dst = out + (oh * oW + ow) * C;

                int c = 0;
            float w00 = wh0 * ww0, w01 = wh0 * ww1;
            float w10 = wh1 * ww0, w11 = wh1 * ww1;
#ifdef __ARM_NEON
            float32x4_t vw00 = vdupq_n_f32(w00), vw01 = vdupq_n_f32(w01);
            float32x4_t vw10 = vdupq_n_f32(w10), vw11 = vdupq_n_f32(w11);
            for (; c + 4 <= C; c += 4) {
                float32x4_t v = vmulq_f32(vld1q_f32(p00+c), vw00);
                v = vfmaq_f32(v, vld1q_f32(p01+c), vw01);
                v = vfmaq_f32(v, vld1q_f32(p10+c), vw10);
                v = vfmaq_f32(v, vld1q_f32(p11+c), vw11);
                vst1q_f32(dst+c, v); }
#elif defined(__AVX512F__)
            __m512 vw00_v = _mm512_set1_ps(w00), vw01_v = _mm512_set1_ps(w01);
            __m512 vw10_v = _mm512_set1_ps(w10), vw11_v = _mm512_set1_ps(w11);
            for (; c + 16 <= C; c += 16) {
                __m512 v = _mm512_mul_ps(_mm512_loadu_ps(p00+c), vw00_v);
                v = _mm512_fmadd_ps(_mm512_loadu_ps(p01+c), vw01_v, v);
                v = _mm512_fmadd_ps(_mm512_loadu_ps(p10+c), vw10_v, v);
                v = _mm512_fmadd_ps(_mm512_loadu_ps(p11+c), vw11_v, v);
                _mm512_storeu_ps(dst+c, v); }
#elif defined(__AVX2__)
            __m256 vw00_v = _mm256_set1_ps(w00), vw01_v = _mm256_set1_ps(w01);
            __m256 vw10_v = _mm256_set1_ps(w10), vw11_v = _mm256_set1_ps(w11);
            for (; c + 8 <= C; c += 8) {
                __m256 v = _mm256_mul_ps(_mm256_loadu_ps(p00+c), vw00_v);
                v = _mm256_fmadd_ps(_mm256_loadu_ps(p01+c), vw01_v, v);
                v = _mm256_fmadd_ps(_mm256_loadu_ps(p10+c), vw10_v, v);
                v = _mm256_fmadd_ps(_mm256_loadu_ps(p11+c), vw11_v, v);
                _mm256_storeu_ps(dst+c, v); }
#endif
            for (; c < C; ++c)
                dst[c] = w00 * p00[c] + w01 * p01[c]
                       + w10 * p10[c] + w11 * p11[c];
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// STFT: short-time Fourier transform (edge-padded, Hann window)
//   audio: [T_audio], out: [T_stft, 2*(n_fft/2+1)]
//   channels: mag[0..K-1], phase[K..2K-1] per frame
//
// Matches ONNX implementation exactly:
//   - Edge padding (clamp) on both sides by n_fft/2
//   - Precomputed float32 weight arrays: w_real[k][n] = hann[n]*cos(2π*k*n/N)
//                                        w_imag[k][n] = -hann[n]*sin(2π*k*n/N)
//   - Inner dot product matches ONNX Conv float32 accumulation order
// ─────────────────────────────────────────────────────────────────

// Disable fast-math for this function to ensure sequential float32 accumulation
// matches the ONNX Conv operator (which also uses sequential float32 arithmetic).
// With -ffast-math the compiler vectorizes the dot product using SIMD, which
// reorders additions and produces different results for near-zero magnitude bins.
#pragma GCC push_options
#pragma GCC optimize("O3,no-fast-math")
void stft_fp32(const float* audio, float* out,
               int T_audio, int n_fft, int hop_size,
               const float* w_real_in, const float* w_imag_in)
{
    const int K      = n_fft / 2 + 1;
    const int pad    = n_fft / 2;
    const int T_stft = T_audio / hop_size + 1;

    // Use precomputed weights if provided (loaded from ONNX via bin file),
    // otherwise fall back to computing them on-the-fly.
    std::vector<float> w_real_buf, w_imag_buf;
    const float* w_real;
    const float* w_imag;

    if (w_real_in && w_imag_in) {
        w_real = w_real_in;
        w_imag = w_imag_in;
    } else {
        // Fallback: compute Hann-windowed cosine/sine weights
        const float two_pi = 2.f * (float)M_PI;
        w_real_buf.resize(K * n_fft);
        w_imag_buf.resize(K * n_fft);
        for (int n = 0; n < n_fft; ++n) {
            float hann = 0.5f * (1.f - cosf(two_pi * n / (float)n_fft));
            for (int k = 0; k < K; ++k) {
                float angle = two_pi * k * n / (float)n_fft;
                w_real_buf[k * n_fft + n] =  hann * cosf(angle);
                w_imag_buf[k * n_fft + n] = -hann * sinf(angle);
            }
        }
        w_real = w_real_buf.data();
        w_imag = w_imag_buf.data();
    }

    // Edge padding: clamp to boundary sample (matches ONNX Pad mode=edge)
    std::vector<float> padded(T_audio + 2 * pad);
    for (int i = 0; i < pad; ++i)
        padded[i] = audio[0];
    memcpy(padded.data() + pad, audio, T_audio * sizeof(float));
    for (int i = 0; i < pad; ++i)
        padded[pad + T_audio + i] = audio[T_audio - 1];

    for (int frame = 0; frame < T_stft; ++frame) {
        const float* x       = padded.data() + frame * hop_size;
        float*       mag_out = out + (size_t)frame * (2 * K);
        float*       ph_out  = out + (size_t)frame * (2 * K) + K;

        for (int k = 0; k < K; ++k) {
            float re = 0.f, im = 0.f;
            const float* wr = w_real + k * n_fft;
            const float* wi = w_imag + k * n_fft;
            for (int n = 0; n < n_fft; ++n) {
                re += wr[n] * x[n];
                im += wi[n] * x[n];
            }
            mag_out[k] = sqrtf(re * re + im * im);
            ph_out[k]  = atan2f(im, re);
        }
    }
}
#pragma GCC pop_options

// ─────────────────────────────────────────────────────────────────
// ISTFT: inverse short-time Fourier transform
//   mag/phase: [T_frames, n_fft/2+1], out: [(T_frames-1)*hop_size]
//
// Matches the learned STFT/ISTFT pair used in the ONNX model:
//   output[n] = sum_f sum_k  mag[k,f]*cos(phase[k,f]+2πk(n-f*hop)/N)
//                            * hann(n-f*hop) / N
// No factor-of-2 for interior bins; raw overlap-add without win_sum
// normalization; n_fft/2 samples trimmed from each end.
// ─────────────────────────────────────────────────────────────────

void istft_fp32(const float* mag, const float* phase, float* out,
                int T_frames, int n_fft, int hop_size, bool normalized)
{
    const int n_bins  = n_fft / 2 + 1;
    const int ola_len = (T_frames - 1) * hop_size + n_fft;
    const int trim    = n_fft / 2;
    const float inv_nfft = 1.f / (float)n_fft;

    std::vector<float> win(n_fft);
    for (int i = 0; i < n_fft; ++i)
        win[i] = 0.5f * (1.f - cosf(2.f * (float)M_PI * i / (float)n_fft));

    // normalized=true: compute win^2 envelope for win-sum normalization.
    // Matches torch.istft: output = OLA(irfft(spec)*win) / OLA(win^2)
    std::vector<float> win_env;
    if (normalized) {
        win_env.assign(ola_len, 0.f);
        for (int f = 0; f < T_frames; ++f) {
            const int start = f * hop_size;
            for (int n = 0; n < n_fft; ++n)
                win_env[start + n] += win[n] * win[n];
        }
    }

    std::vector<float> ola(ola_len, 0.f);

    const float two_pi_inv_nfft = 2.f * (float)M_PI / (float)n_fft;
    for (int f = 0; f < T_frames; ++f) {
        const float* m = mag   + (size_t)f * n_bins;
        const float* p = phase + (size_t)f * n_bins;
        const int start = f * hop_size;

        for (int n = 0; n < n_fft; ++n) {
            float s = m[0] * cosf(p[0]);
            if (normalized) {
                // Factor-of-2 for interior bins (k=1..K-2): matches real IFFT
                for (int k = 1; k < n_bins - 1; ++k)
                    s += 2.0f * m[k] * cosf(p[k] + two_pi_inv_nfft * k * n);
            } else {
                for (int k = 1; k < n_bins - 1; ++k)
                    s += m[k] * cosf(p[k] + two_pi_inv_nfft * k * n);
            }
            s += m[n_bins - 1] * cosf(p[n_bins - 1] + (float)M_PI * n);
            ola[start + n] += s * win[n] * inv_nfft;
        }
    }

    // Trim n_fft/2 from each end (matches ONNX Slice_3 [10:-10])
    const int out_len = ola_len - 2 * trim;
    if (normalized) {
        // Divide by OLA(win^2), NOT by OLA(win^2)/N.
        // ola already has the 1/N factor; dividing by win_env (not win_env/N)
        // gives: (1/N * OLA(2*sum*win)) / OLA(win^2) = torch.istft
        for (int i = 0; i < out_len; ++i) {
            float env = win_env[i + trim];
            out[i] = (env > 1e-8f) ? ola[i + trim] / env : 0.f;
        }
    } else {
        for (int i = 0; i < out_len; ++i)
            out[i] = ola[i + trim];
    }
}

// ─────────────────────────────────────────────────────────────────
// Sine generator: harmonic sinusoids from F0 with phase accumulation
// ─────────────────────────────────────────────────────────────────

void sine_gen_fp32(const float* f0, float* out, int T,
                   int harmonic_num, int sample_rate,
                   float sine_amp, float voiced_threshold,
                   float noise_std, int upsample_scale)
{
    const int    H      = harmonic_num + 1;
    const double two_pi = 2.0 * M_PI;
    const double inv_sr = 1.0 / (double)sample_rate;

    // ── ONNX-compatible algorithm: downsample → cumsum → upsample ──────────
    // Matches istftnet_mini_3.py SineGen._f02sine() with align_corners=False.
    // Activated when upsample_scale > 0.
    if (upsample_scale > 0) {
        const int   T_dn  = T / upsample_scale;
        const float s     = (float)upsample_scale;
        const float inv_s = 1.0f / s;

        auto fill_harmonic = [&](int h,
                                 std::vector<float>& rad,
                                 std::vector<float>& rad_dn,
                                 std::vector<float>& phase_dn) {
            // rad[t] = (f0[t]*(h+1) / sr) % 1  — Python-style positive modulo
            for (int t = 0; t < T; ++t) {
                float r = fmodf(f0[t] * (h + 1) * (float)inv_sr, 1.0f);
                if (r < 0.0f) r += 1.0f;
                rad[t] = r;
            }

            // Downsample rad from T to T_dn (align_corners=False)
            // src = (i + 0.5) * s - 0.5
            for (int i = 0; i < T_dn; ++i) {
                float src = (i + 0.5f) * s - 0.5f;
                int i0 = (int)floorf(src);
                int i1 = i0 + 1;
                float w1 = src - (float)i0;
                i0 = std::max(0, std::min(i0, T - 1));
                i1 = std::max(0, std::min(i1, T - 1));
                rad_dn[i] = (1.0f - w1) * rad[i0] + w1 * rad[i1];
            }

            // Cumsum
            float cum = 0.0f;
            for (int i = 0; i < T_dn; ++i) {
                cum += rad_dn[i];
                phase_dn[i] = cum;
            }

            // Upsample phase_dn*s from T_dn to T (align_corners=False), then sin
            // src = (j + 0.5) * inv_s - 0.5
            for (int j = 0; j < T; ++j) {
                float src = (j + 0.5f) * inv_s - 0.5f;
                int i0 = (int)floorf(src);
                int i1 = i0 + 1;
                float w1 = src - (float)i0;
                i0 = std::max(0, std::min(i0, T_dn - 1));
                i1 = std::max(0, std::min(i1, T_dn - 1));
                float ph = ((1.0f - w1) * phase_dn[i0] + w1 * phase_dn[i1]) * s;
                bool voiced = f0[j] > voiced_threshold;
                out[(size_t)j * H + h] = voiced ? sine_amp * sinf((float)(two_pi * ph)) : 0.0f;
            }
        };

#ifdef _OPENMP
        if (H > 1 && T >= 4096 && !omp_in_parallel() && omp_get_max_threads() > 1) {
#pragma omp parallel for schedule(static)
            for (int h = 0; h < H; ++h) {
                std::vector<float> rad(T);
                std::vector<float> rad_dn(T_dn);
                std::vector<float> phase_dn(T_dn);
                fill_harmonic(h, rad, rad_dn, phase_dn);
            }
            return;
        }
#endif

        std::vector<float> rad(T);
        std::vector<float> rad_dn(T_dn);
        std::vector<float> phase_dn(T_dn);
        for (int h = 0; h < H; ++h) {
            fill_harmonic(h, rad, rad_dn, phase_dn);
        }
        return;
    }

    // ── Legacy algorithm (phase accumulation) ──────────────────────────────
    const float uv_noise_std = sine_amp / 3.0f;

    std::vector<double> phase(H, 0.0);

    // Box-Muller Gaussian RNG (deterministic fixed seed per call)
    uint64_t rng = 0x9E3779B97F4A7C15ULL;
    auto randn = [&]() -> float {
        rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
        float u1 = ((rng * 0x2545F4914F6CDD1DULL) >> 32) / 4294967296.0f;
        rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
        float u2 = ((rng * 0x2545F4914F6CDD1DULL) >> 32) / 4294967296.0f;
        u1 = u1 < 1e-7f ? 1e-7f : u1;
        return sqrtf(-2.0f * logf(u1)) * cosf((float)two_pi * u2);
    };

    for (int t = 0; t < T; ++t) {
        float  f  = f0[t];
        float* y  = out + (size_t)t * H;
        bool voiced = (f > voiced_threshold);
        for (int h = 0; h < H; ++h) {
            double freq_norm = (double)f * (h + 1) * inv_sr;
            phase[h] += freq_norm;
            phase[h] -= (double)(long long)phase[h];
            float sine_val = voiced ? sine_amp * sinf((float)(two_pi * phase[h])) : 0.f;
            if (noise_std > 0.f) {
                float r = randn();
                y[h] = sine_val + (voiced ? noise_std : uv_noise_std) * r;
            } else {
                y[h] = sine_val;
            }
        }
    }
}
