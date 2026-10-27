// fp32_conv1d.cpp — Winograd conv1d transforms (F(4,3), F(2,7), F(2,11))
//
// Input and output transforms are vectorised with AVX-512 / AVX2 / scalar
// fallback.  The caller provides pre-packed weight tiles (one per Winograd
// point) and a scratch buffer that is sub-divided internally.
//
// Scratch layout for all three variants:
//   [padded]  pad_rows_ext × C_in  floats
//   [V]       n_points × n_tiles × C_in  floats   (input transform)
//   [M]       n_points × n_tiles × C_out floats   (GEMM output)

#include "../ops_neon.hpp"
#include <cstring>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
// F(4,3)  — tile size 4, kernel size 3, 6 Winograd points
// ─────────────────────────────────────────────────────────────────────────────
void conv1d_wino_f43_fp32(
    const float* in_buf,
    const float* const* w_packed,   // [6] packed weight pointers
    const float* bias, bool relu,
    int T, int T_out, int C_in, int C_out, int padding,
    float* out_buf,
    float* scratch)
{
    const int n_tiles      = (T_out + 3) / 4;
    const int pad_rows     = T + 2 * padding;
    const int pad_rows_ext = std::max(pad_rows, 4 * n_tiles + 2);

    float* padded = scratch;
    float* V_data = padded + (size_t)pad_rows_ext * C_in;
    float* M_data = V_data + (size_t)6 * n_tiles * C_in;

    const size_t row_bytes = (size_t)C_in * sizeof(float);
    if (padding > 0) memset(padded, 0, padding * row_bytes);
    memset(padded + (size_t)(padding + T) * C_in, 0,
           (size_t)(pad_rows_ext - padding - T) * row_bytes);
    memcpy(padded + (size_t)padding * C_in, in_buf, (size_t)T * row_bytes);

    // Input transform BT6 × d → V
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
#endif
    for (int t = 0; t < n_tiles; ++t) {
        const float* db = padded + (size_t)(4 * t) * C_in;
        const float* d0=db, *d1=db+C_in, *d2=db+2*C_in,
                    *d3=db+3*C_in, *d4=db+4*C_in, *d5=db+5*C_in;
        float* V0=V_data+(size_t)(0*n_tiles+t)*C_in;
        float* V1=V_data+(size_t)(1*n_tiles+t)*C_in;
        float* V2=V_data+(size_t)(2*n_tiles+t)*C_in;
        float* V3=V_data+(size_t)(3*n_tiles+t)*C_in;
        float* V4=V_data+(size_t)(4*n_tiles+t)*C_in;
        float* V5=V_data+(size_t)(5*n_tiles+t)*C_in;
        int ci = 0;
#if defined(__AVX512F__)
        for (; ci + 15 < C_in; ci += 16) {
            __m512 D0=_mm512_loadu_ps(d0+ci),D1=_mm512_loadu_ps(d1+ci);
            __m512 D2=_mm512_loadu_ps(d2+ci),D3=_mm512_loadu_ps(d3+ci);
            __m512 D4=_mm512_loadu_ps(d4+ci),D5=_mm512_loadu_ps(d5+ci);
            const __m512 c4=_mm512_set1_ps(4.f),cm5=_mm512_set1_ps(-5.f);
            const __m512 cm4=_mm512_set1_ps(-4.f),c2=_mm512_set1_ps(2.f);
            const __m512 cm2=_mm512_set1_ps(-2.f),cm1=_mm512_set1_ps(-1.f);
            _mm512_storeu_ps(V0+ci,_mm512_fmadd_ps(c4,D0,_mm512_fmadd_ps(cm5,D2,D4)));
            _mm512_storeu_ps(V1+ci,_mm512_fmadd_ps(cm4,_mm512_add_ps(D1,D2),_mm512_add_ps(D3,D4)));
            _mm512_storeu_ps(V2+ci,_mm512_fmadd_ps(c4,_mm512_sub_ps(D1,D2),_mm512_sub_ps(D4,D3)));
            _mm512_storeu_ps(V3+ci,_mm512_fmadd_ps(cm2,D1,_mm512_fmadd_ps(c2,D3,_mm512_fmadd_ps(cm1,D2,D4))));
            _mm512_storeu_ps(V4+ci,_mm512_fmadd_ps(c2,_mm512_sub_ps(D1,D3),_mm512_sub_ps(D4,D2)));
            _mm512_storeu_ps(V5+ci,_mm512_fmadd_ps(c4,D1,_mm512_fmadd_ps(cm5,D3,D5)));
        }
#elif defined(__AVX2__)
        for (; ci + 7 < C_in; ci += 8) {
            __m256 D0=_mm256_loadu_ps(d0+ci),D1=_mm256_loadu_ps(d1+ci);
            __m256 D2=_mm256_loadu_ps(d2+ci),D3=_mm256_loadu_ps(d3+ci);
            __m256 D4=_mm256_loadu_ps(d4+ci),D5=_mm256_loadu_ps(d5+ci);
            const __m256 c4=_mm256_set1_ps(4.f),cm5=_mm256_set1_ps(-5.f);
            const __m256 cm4=_mm256_set1_ps(-4.f),c2=_mm256_set1_ps(2.f);
            const __m256 cm2=_mm256_set1_ps(-2.f),cm1=_mm256_set1_ps(-1.f);
            _mm256_storeu_ps(V0+ci,_mm256_fmadd_ps(c4,D0,_mm256_fmadd_ps(cm5,D2,D4)));
            _mm256_storeu_ps(V1+ci,_mm256_fmadd_ps(cm4,_mm256_add_ps(D1,D2),_mm256_add_ps(D3,D4)));
            _mm256_storeu_ps(V2+ci,_mm256_fmadd_ps(c4,_mm256_sub_ps(D1,D2),_mm256_sub_ps(D4,D3)));
            _mm256_storeu_ps(V3+ci,_mm256_fmadd_ps(cm2,D1,_mm256_fmadd_ps(c2,D3,_mm256_fmadd_ps(cm1,D2,D4))));
            _mm256_storeu_ps(V4+ci,_mm256_fmadd_ps(c2,_mm256_sub_ps(D1,D3),_mm256_sub_ps(D4,D2)));
            _mm256_storeu_ps(V5+ci,_mm256_fmadd_ps(c4,D1,_mm256_fmadd_ps(cm5,D3,D5)));
        }
#endif
        for (; ci < C_in; ++ci) {
            float d0v=d0[ci],d1v=d1[ci],d2v=d2[ci],d3v=d3[ci],d4v=d4[ci],d5v=d5[ci];
            V0[ci]=4*d0v-5*d2v+d4v; V1[ci]=-4*d1v-4*d2v+d3v+d4v;
            V2[ci]=4*d1v-4*d2v-d3v+d4v; V3[ci]=-2*d1v-d2v+2*d3v+d4v;
            V4[ci]=2*d1v-d2v-2*d3v+d4v; V5[ci]=4*d1v-5*d3v+d5v;
        }
    }

    // 6 point GEMMs: M[p] = V[p] × Uw[p]
    for (int p = 0; p < 6; ++p) {
        sgemm_f32(V_data+(size_t)p*n_tiles*C_in, w_packed[p],
                  nullptr, M_data+(size_t)p*n_tiles*C_out, false,
                  n_tiles, C_in, C_out, /*in_parallel=*/false);
    }

    // Output transform AT6 × M + bias + relu
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
#endif
    for (int t = 0; t < n_tiles; ++t) {
        const int out_rows = std::min(4, T_out - 4 * t);
        int co = 0;
#if defined(__AVX512F__)
        for (; co + 15 < C_out; co += 16) {
            __m512 m0=_mm512_loadu_ps(M_data+(size_t)(0*n_tiles+t)*C_out+co);
            __m512 m1=_mm512_loadu_ps(M_data+(size_t)(1*n_tiles+t)*C_out+co);
            __m512 m2=_mm512_loadu_ps(M_data+(size_t)(2*n_tiles+t)*C_out+co);
            __m512 m3=_mm512_loadu_ps(M_data+(size_t)(3*n_tiles+t)*C_out+co);
            __m512 m4=_mm512_loadu_ps(M_data+(size_t)(4*n_tiles+t)*C_out+co);
            __m512 m5=_mm512_loadu_ps(M_data+(size_t)(5*n_tiles+t)*C_out+co);
            const __m512 c2=_mm512_set1_ps(2.f),c4=_mm512_set1_ps(4.f),c8=_mm512_set1_ps(8.f);
            __m512 y0=_mm512_add_ps(_mm512_add_ps(_mm512_add_ps(m0,m1),_mm512_add_ps(m2,m3)),m4);
            __m512 y1=_mm512_fmadd_ps(c2,_mm512_sub_ps(m3,m4),_mm512_sub_ps(m1,m2));
            __m512 y2=_mm512_fmadd_ps(c4,_mm512_add_ps(m3,m4),_mm512_add_ps(m1,m2));
            __m512 y3=_mm512_add_ps(_mm512_fmadd_ps(c8,_mm512_sub_ps(m3,m4),_mm512_sub_ps(m1,m2)),m5);
            if (bias) { __m512 bv=_mm512_loadu_ps(bias+co);
                y0=_mm512_add_ps(y0,bv); y1=_mm512_add_ps(y1,bv);
                y2=_mm512_add_ps(y2,bv); y3=_mm512_add_ps(y3,bv); }
            if (relu) { __m512 zv=_mm512_setzero_ps();
                y0=_mm512_max_ps(y0,zv); y1=_mm512_max_ps(y1,zv);
                y2=_mm512_max_ps(y2,zv); y3=_mm512_max_ps(y3,zv); }
            float* ot=out_buf+(size_t)(4*t)*C_out+co;
            if(out_rows>0)_mm512_storeu_ps(ot+0*C_out,y0);
            if(out_rows>1)_mm512_storeu_ps(ot+1*C_out,y1);
            if(out_rows>2)_mm512_storeu_ps(ot+2*C_out,y2);
            if(out_rows>3)_mm512_storeu_ps(ot+3*C_out,y3);
        }
#elif defined(__AVX2__)
        for (; co + 7 < C_out; co += 8) {
            __m256 m0=_mm256_loadu_ps(M_data+(size_t)(0*n_tiles+t)*C_out+co);
            __m256 m1=_mm256_loadu_ps(M_data+(size_t)(1*n_tiles+t)*C_out+co);
            __m256 m2=_mm256_loadu_ps(M_data+(size_t)(2*n_tiles+t)*C_out+co);
            __m256 m3=_mm256_loadu_ps(M_data+(size_t)(3*n_tiles+t)*C_out+co);
            __m256 m4=_mm256_loadu_ps(M_data+(size_t)(4*n_tiles+t)*C_out+co);
            __m256 m5=_mm256_loadu_ps(M_data+(size_t)(5*n_tiles+t)*C_out+co);
            const __m256 c2=_mm256_set1_ps(2.f),c4=_mm256_set1_ps(4.f),c8=_mm256_set1_ps(8.f);
            __m256 y0=_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(m0,m1),_mm256_add_ps(m2,m3)),m4);
            __m256 y1=_mm256_fmadd_ps(c2,_mm256_sub_ps(m3,m4),_mm256_sub_ps(m1,m2));
            __m256 y2=_mm256_fmadd_ps(c4,_mm256_add_ps(m3,m4),_mm256_add_ps(m1,m2));
            __m256 y3=_mm256_add_ps(_mm256_fmadd_ps(c8,_mm256_sub_ps(m3,m4),_mm256_sub_ps(m1,m2)),m5);
            if (bias) { __m256 bv=_mm256_loadu_ps(bias+co);
                y0=_mm256_add_ps(y0,bv); y1=_mm256_add_ps(y1,bv);
                y2=_mm256_add_ps(y2,bv); y3=_mm256_add_ps(y3,bv); }
            if (relu) { __m256 zv=_mm256_setzero_ps();
                y0=_mm256_max_ps(y0,zv); y1=_mm256_max_ps(y1,zv);
                y2=_mm256_max_ps(y2,zv); y3=_mm256_max_ps(y3,zv); }
            float* ot=out_buf+(size_t)(4*t)*C_out+co;
            if(out_rows>0)_mm256_storeu_ps(ot+0*C_out,y0);
            if(out_rows>1)_mm256_storeu_ps(ot+1*C_out,y1);
            if(out_rows>2)_mm256_storeu_ps(ot+2*C_out,y2);
            if(out_rows>3)_mm256_storeu_ps(ot+3*C_out,y3);
        }
#endif
        for (; co < C_out; ++co) {
            float m0v=M_data[(size_t)(0*n_tiles+t)*C_out+co],m1v=M_data[(size_t)(1*n_tiles+t)*C_out+co];
            float m2v=M_data[(size_t)(2*n_tiles+t)*C_out+co],m3v=M_data[(size_t)(3*n_tiles+t)*C_out+co];
            float m4v=M_data[(size_t)(4*n_tiles+t)*C_out+co],m5v=M_data[(size_t)(5*n_tiles+t)*C_out+co];
            float y[4];
            y[0]=m0v+m1v+m2v+m3v+m4v; y[1]=(m1v-m2v)+2*(m3v-m4v);
            y[2]=(m1v+m2v)+4*(m3v+m4v); y[3]=(m1v-m2v)+8*(m3v-m4v)+m5v;
            float* ot=out_buf+(size_t)(4*t)*C_out+co;
            for (int r = 0; r < out_rows; ++r) {
                float v=y[r]+(bias?bias[co]:0.f);
                ot[r*C_out]= relu ? std::max(0.f,v) : v;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// F(2,7)  — tile size 2, kernel size 7, 8 Winograd points
// ─────────────────────────────────────────────────────────────────────────────
void conv1d_wino_f27_fp32(
    const float* in_buf,
    const float* const* w_packed,   // [8] packed weight pointers
    const float* bias, bool relu,
    int T, int T_out, int C_in, int C_out, int padding,
    float* out_buf,
    float* scratch)
{
    const int n_tiles      = (T_out + 1) / 2;
    const int pad_rows     = T + 2 * padding;
    const int pad_rows_ext = std::max(pad_rows, 2 * n_tiles + 6);

    float* padded = scratch;
    float* V_data = padded + (size_t)pad_rows_ext * C_in;
    float* M_data = V_data + (size_t)8 * n_tiles * C_in;

    const size_t row_bytes = (size_t)C_in * sizeof(float);
    if (padding > 0) memset(padded, 0, padding * row_bytes);
    memset(padded + (size_t)(padding + T) * C_in, 0,
           (size_t)(pad_rows_ext - padding - T) * row_bytes);
    memcpy(padded + (size_t)padding * C_in, in_buf, (size_t)T * row_bytes);

    // Input transform BT8 × d → V
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
#endif
    for (int t = 0; t < n_tiles; ++t) {
        const float* db = padded + (size_t)(2 * t) * C_in;
        const float* d0=db,*d1=db+C_in,*d2=db+2*C_in,*d3=db+3*C_in,
                    *d4=db+4*C_in,*d5=db+5*C_in,*d6=db+6*C_in,*d7=db+7*C_in;
        float* V0=V_data+(size_t)(0*n_tiles+t)*C_in;
        float* V1=V_data+(size_t)(1*n_tiles+t)*C_in;
        float* V2=V_data+(size_t)(2*n_tiles+t)*C_in;
        float* V3=V_data+(size_t)(3*n_tiles+t)*C_in;
        float* V4=V_data+(size_t)(4*n_tiles+t)*C_in;
        float* V5=V_data+(size_t)(5*n_tiles+t)*C_in;
        float* V6=V_data+(size_t)(6*n_tiles+t)*C_in;
        float* V7=V_data+(size_t)(7*n_tiles+t)*C_in;
        int ci = 0;
#if defined(__AVX512F__)
        for (; ci + 15 < C_in; ci += 16) {
            __m512 D0=_mm512_loadu_ps(d0+ci),D1=_mm512_loadu_ps(d1+ci);
            __m512 D2=_mm512_loadu_ps(d2+ci),D3=_mm512_loadu_ps(d3+ci);
            __m512 D4=_mm512_loadu_ps(d4+ci),D5=_mm512_loadu_ps(d5+ci);
            __m512 D6=_mm512_loadu_ps(d6+ci),D7=_mm512_loadu_ps(d7+ci);
            const __m512 c36=_mm512_set1_ps(36.f),cm36=_mm512_set1_ps(-36.f);
            const __m512 c49=_mm512_set1_ps(49.f),cm14=_mm512_set1_ps(-14.f);
            const __m512 c13=_mm512_set1_ps(13.f);
            const __m512 c18=_mm512_set1_ps(18.f),c9=_mm512_set1_ps(9.f);
            const __m512 cm20=_mm512_set1_ps(-20.f),cm10=_mm512_set1_ps(-10.f);
            const __m512 c2=_mm512_set1_ps(2.f);
            const __m512 c12=_mm512_set1_ps(12.f),c4=_mm512_set1_ps(4.f);
            const __m512 cm15=_mm512_set1_ps(-15.f),cm5=_mm512_set1_ps(-5.f);
            const __m512 c3=_mm512_set1_ps(3.f);
            __m512 S12=_mm512_add_ps(D1,D2),D12=_mm512_sub_ps(D2,D1);
            __m512 S34=_mm512_add_ps(D3,D4),D34=_mm512_sub_ps(D3,D4);
            __m512 S56=_mm512_add_ps(D5,D6),D56=_mm512_sub_ps(D6,D5);
            __m512 P34=_mm512_fmadd_ps(c9,D2,_mm512_fmadd_ps(cm10,D4,D6));
            __m512 Q34=_mm512_fmadd_ps(c18,D1,_mm512_fmadd_ps(cm20,D3,_mm512_mul_ps(c2,D5)));
            __m512 P56e=_mm512_fmadd_ps(c4,D2,_mm512_fmadd_ps(cm5,D4,D6));
            __m512 Q56e=_mm512_fmadd_ps(c12,D1,_mm512_fmadd_ps(cm15,D3,_mm512_mul_ps(c3,D5)));
            _mm512_storeu_ps(V0+ci,_mm512_fmadd_ps(cm36,D0,_mm512_fmadd_ps(c49,D2,_mm512_fmadd_ps(cm14,D4,D6))));
            _mm512_storeu_ps(V1+ci,_mm512_fmadd_ps(c36,S12,_mm512_fmadd_ps(_mm512_set1_ps(-13.f),S34,S56)));
            _mm512_storeu_ps(V2+ci,_mm512_fmadd_ps(c36,D12,_mm512_fmadd_ps(c13,D34,D56)));
            _mm512_storeu_ps(V3+ci,_mm512_add_ps(P34,Q34));
            _mm512_storeu_ps(V4+ci,_mm512_sub_ps(P34,Q34));
            _mm512_storeu_ps(V5+ci,_mm512_add_ps(P56e,Q56e));
            _mm512_storeu_ps(V6+ci,_mm512_sub_ps(P56e,Q56e));
            _mm512_storeu_ps(V7+ci,_mm512_fmadd_ps(cm36,D1,_mm512_fmadd_ps(c49,D3,_mm512_fmadd_ps(cm14,D5,D7))));
        }
#elif defined(__AVX2__)
        for (; ci + 7 < C_in; ci += 8) {
            __m256 D0=_mm256_loadu_ps(d0+ci),D1=_mm256_loadu_ps(d1+ci);
            __m256 D2=_mm256_loadu_ps(d2+ci),D3=_mm256_loadu_ps(d3+ci);
            __m256 D4=_mm256_loadu_ps(d4+ci),D5=_mm256_loadu_ps(d5+ci);
            __m256 D6=_mm256_loadu_ps(d6+ci),D7=_mm256_loadu_ps(d7+ci);
            const __m256 c36=_mm256_set1_ps(36.f),cm36=_mm256_set1_ps(-36.f);
            const __m256 c49=_mm256_set1_ps(49.f),cm14=_mm256_set1_ps(-14.f);
            const __m256 c13=_mm256_set1_ps(13.f);
            const __m256 c18=_mm256_set1_ps(18.f),c9=_mm256_set1_ps(9.f);
            const __m256 cm20=_mm256_set1_ps(-20.f),cm10=_mm256_set1_ps(-10.f);
            const __m256 c2=_mm256_set1_ps(2.f);
            const __m256 c12=_mm256_set1_ps(12.f),c4=_mm256_set1_ps(4.f);
            const __m256 cm15=_mm256_set1_ps(-15.f),cm5=_mm256_set1_ps(-5.f);
            const __m256 c3=_mm256_set1_ps(3.f);
            __m256 S12=_mm256_add_ps(D1,D2),D12=_mm256_sub_ps(D2,D1);
            __m256 S34=_mm256_add_ps(D3,D4),D34=_mm256_sub_ps(D3,D4);
            __m256 S56=_mm256_add_ps(D5,D6),D56=_mm256_sub_ps(D6,D5);
            __m256 P34=_mm256_fmadd_ps(c9,D2,_mm256_fmadd_ps(cm10,D4,D6));
            __m256 Q34=_mm256_fmadd_ps(c18,D1,_mm256_fmadd_ps(cm20,D3,_mm256_mul_ps(c2,D5)));
            __m256 P56e=_mm256_fmadd_ps(c4,D2,_mm256_fmadd_ps(cm5,D4,D6));
            __m256 Q56e=_mm256_fmadd_ps(c12,D1,_mm256_fmadd_ps(cm15,D3,_mm256_mul_ps(c3,D5)));
            _mm256_storeu_ps(V0+ci,_mm256_fmadd_ps(cm36,D0,_mm256_fmadd_ps(c49,D2,_mm256_fmadd_ps(cm14,D4,D6))));
            _mm256_storeu_ps(V1+ci,_mm256_fmadd_ps(c36,S12,_mm256_fmadd_ps(_mm256_set1_ps(-13.f),S34,S56)));
            _mm256_storeu_ps(V2+ci,_mm256_fmadd_ps(c36,D12,_mm256_fmadd_ps(c13,D34,D56)));
            _mm256_storeu_ps(V3+ci,_mm256_add_ps(P34,Q34));
            _mm256_storeu_ps(V4+ci,_mm256_sub_ps(P34,Q34));
            _mm256_storeu_ps(V5+ci,_mm256_add_ps(P56e,Q56e));
            _mm256_storeu_ps(V6+ci,_mm256_sub_ps(P56e,Q56e));
            _mm256_storeu_ps(V7+ci,_mm256_fmadd_ps(cm36,D1,_mm256_fmadd_ps(c49,D3,_mm256_fmadd_ps(cm14,D5,D7))));
        }
#endif
        for (; ci < C_in; ++ci) {
            float d0v=d0[ci],d1v=d1[ci],d2v=d2[ci],d3v=d3[ci];
            float d4v=d4[ci],d5v=d5[ci],d6v=d6[ci],d7v=d7[ci];
            float s12=d1v+d2v,d12=d2v-d1v,s34=d3v+d4v,d34=d3v-d4v;
            float s56=d5v+d6v,d56=d6v-d5v;
            float p34=9.f*d2v-10.f*d4v+d6v, q34=18.f*d1v-20.f*d3v+2.f*d5v;
            float p56e=4.f*d2v-5.f*d4v+d6v, q56e=12.f*d1v-15.f*d3v+3.f*d5v;
            V0[ci]=-36.f*d0v+49.f*d2v-14.f*d4v+d6v;
            V1[ci]=36.f*s12-13.f*s34+s56; V2[ci]=36.f*d12+13.f*d34+d56;
            V3[ci]=p34+q34; V4[ci]=p34-q34; V5[ci]=p56e+q56e; V6[ci]=p56e-q56e;
            V7[ci]=-36.f*d1v+49.f*d3v-14.f*d5v+d7v;
        }
    }

    // 8 point GEMMs: M[p] = V[p] × Uw[p]
    for (int p = 0; p < 8; ++p) {
        sgemm_f32(V_data+(size_t)p*n_tiles*C_in, w_packed[p],
                  nullptr, M_data+(size_t)p*n_tiles*C_out, false,
                  n_tiles, C_in, C_out, /*in_parallel=*/false);
    }

    // Output transform AT8 × M + bias + relu
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
#endif
    for (int t = 0; t < n_tiles; ++t) {
        const int out_rows = std::min(2, T_out - 2 * t);
        int co = 0;
#if defined(__AVX512F__)
        for (; co + 15 < C_out; co += 16) {
            __m512 m0=_mm512_loadu_ps(M_data+(size_t)(0*n_tiles+t)*C_out+co);
            __m512 m1=_mm512_loadu_ps(M_data+(size_t)(1*n_tiles+t)*C_out+co);
            __m512 m2=_mm512_loadu_ps(M_data+(size_t)(2*n_tiles+t)*C_out+co);
            __m512 m3=_mm512_loadu_ps(M_data+(size_t)(3*n_tiles+t)*C_out+co);
            __m512 m4=_mm512_loadu_ps(M_data+(size_t)(4*n_tiles+t)*C_out+co);
            __m512 m5=_mm512_loadu_ps(M_data+(size_t)(5*n_tiles+t)*C_out+co);
            __m512 m6=_mm512_loadu_ps(M_data+(size_t)(6*n_tiles+t)*C_out+co);
            __m512 m7=_mm512_loadu_ps(M_data+(size_t)(7*n_tiles+t)*C_out+co);
            const __m512 c2=_mm512_set1_ps(2.f),c3=_mm512_set1_ps(3.f);
            __m512 S12=_mm512_add_ps(m1,m2),S34=_mm512_add_ps(m3,m4),S56=_mm512_add_ps(m5,m6);
            __m512 A=_mm512_sub_ps(m1,m2),B=_mm512_sub_ps(m3,m4),C_=_mm512_sub_ps(m5,m6);
            __m512 y0=_mm512_add_ps(_mm512_add_ps(_mm512_add_ps(m0,S12),S34),S56);
            __m512 y1=_mm512_add_ps(_mm512_fmadd_ps(c2,B,_mm512_fmadd_ps(c3,C_,A)),m7);
            if (bias) { __m512 bv=_mm512_loadu_ps(bias+co);
                y0=_mm512_add_ps(y0,bv); y1=_mm512_add_ps(y1,bv); }
            if (relu) { __m512 zv=_mm512_setzero_ps();
                y0=_mm512_max_ps(y0,zv); y1=_mm512_max_ps(y1,zv); }
            float* ot=out_buf+(size_t)(2*t)*C_out+co;
            if(out_rows>0)_mm512_storeu_ps(ot+0*C_out,y0);
            if(out_rows>1)_mm512_storeu_ps(ot+1*C_out,y1);
        }
#elif defined(__AVX2__)
        for (; co + 7 < C_out; co += 8) {
            __m256 m0=_mm256_loadu_ps(M_data+(size_t)(0*n_tiles+t)*C_out+co);
            __m256 m1=_mm256_loadu_ps(M_data+(size_t)(1*n_tiles+t)*C_out+co);
            __m256 m2=_mm256_loadu_ps(M_data+(size_t)(2*n_tiles+t)*C_out+co);
            __m256 m3=_mm256_loadu_ps(M_data+(size_t)(3*n_tiles+t)*C_out+co);
            __m256 m4=_mm256_loadu_ps(M_data+(size_t)(4*n_tiles+t)*C_out+co);
            __m256 m5=_mm256_loadu_ps(M_data+(size_t)(5*n_tiles+t)*C_out+co);
            __m256 m6=_mm256_loadu_ps(M_data+(size_t)(6*n_tiles+t)*C_out+co);
            __m256 m7=_mm256_loadu_ps(M_data+(size_t)(7*n_tiles+t)*C_out+co);
            const __m256 c2=_mm256_set1_ps(2.f),c3=_mm256_set1_ps(3.f);
            __m256 S12=_mm256_add_ps(m1,m2),S34=_mm256_add_ps(m3,m4),S56=_mm256_add_ps(m5,m6);
            __m256 A=_mm256_sub_ps(m1,m2),B=_mm256_sub_ps(m3,m4),C_=_mm256_sub_ps(m5,m6);
            __m256 y0=_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(m0,S12),S34),S56);
            __m256 y1=_mm256_add_ps(_mm256_fmadd_ps(c2,B,_mm256_fmadd_ps(c3,C_,A)),m7);
            if (bias) { __m256 bv=_mm256_loadu_ps(bias+co);
                y0=_mm256_add_ps(y0,bv); y1=_mm256_add_ps(y1,bv); }
            if (relu) { __m256 zv=_mm256_setzero_ps();
                y0=_mm256_max_ps(y0,zv); y1=_mm256_max_ps(y1,zv); }
            float* ot=out_buf+(size_t)(2*t)*C_out+co;
            if(out_rows>0)_mm256_storeu_ps(ot+0*C_out,y0);
            if(out_rows>1)_mm256_storeu_ps(ot+1*C_out,y1);
        }
#endif
        for (; co < C_out; ++co) {
            float m0v=M_data[(size_t)(0*n_tiles+t)*C_out+co],m1v=M_data[(size_t)(1*n_tiles+t)*C_out+co];
            float m2v=M_data[(size_t)(2*n_tiles+t)*C_out+co],m3v=M_data[(size_t)(3*n_tiles+t)*C_out+co];
            float m4v=M_data[(size_t)(4*n_tiles+t)*C_out+co],m5v=M_data[(size_t)(5*n_tiles+t)*C_out+co];
            float m6v=M_data[(size_t)(6*n_tiles+t)*C_out+co],m7v=M_data[(size_t)(7*n_tiles+t)*C_out+co];
            float Av=m1v-m2v,Bv=m3v-m4v,Cv=m5v-m6v;
            float y[2];
            y[0]=m0v+(m1v+m2v)+(m3v+m4v)+(m5v+m6v);
            y[1]=Av+2.f*Bv+3.f*Cv+m7v;
            float* ot=out_buf+(size_t)(2*t)*C_out+co;
            for (int r = 0; r < out_rows; ++r) {
                float v=y[r]+(bias?bias[co]:0.f);
                ot[r*C_out]= relu ? std::max(0.f,v) : v;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// F(2,11)  — tile size 2, kernel size 11, 12 Winograd points
// ─────────────────────────────────────────────────────────────────────────────
void conv1d_wino_f211_fp32(
    const float* in_buf,
    const float* const* w_packed,   // [12] packed weight pointers
    const float* bias, bool relu,
    int T, int T_out, int C_in, int C_out, int padding,
    float* out_buf,
    float* scratch)
{
    const int n_tiles      = (T_out + 1) / 2;
    const int pad_rows     = T + 2 * padding;
    const int pad_rows_ext = std::max(pad_rows, 2 * n_tiles + 10);

    float* padded = scratch;
    float* V_data = padded + (size_t)pad_rows_ext * C_in;
    float* M_data = V_data + (size_t)12 * n_tiles * C_in;

    const size_t row_bytes = (size_t)C_in * sizeof(float);
    if (padding > 0) memset(padded, 0, padding * row_bytes);
    memset(padded + (size_t)(padding + T) * C_in, 0,
           (size_t)(pad_rows_ext - padding - T) * row_bytes);
    memcpy(padded + (size_t)padding * C_in, in_buf, (size_t)T * row_bytes);

    // Input transform BT12 × d → V
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
#endif
    for (int t = 0; t < n_tiles; ++t) {
        const float* db = padded + (size_t)(2 * t) * C_in;
        const float* d0=db,*d1=db+C_in,*d2=db+2*C_in,*d3=db+3*C_in;
        const float* d4=db+4*C_in,*d5=db+5*C_in,*d6=db+6*C_in,*d7=db+7*C_in;
        const float* d8=db+8*C_in,*d9=db+9*C_in,*d10=db+10*C_in,*d11=db+11*C_in;
        float* V0 =V_data+(size_t)( 0*n_tiles+t)*C_in;
        float* V1 =V_data+(size_t)( 1*n_tiles+t)*C_in;
        float* V2 =V_data+(size_t)( 2*n_tiles+t)*C_in;
        float* V3 =V_data+(size_t)( 3*n_tiles+t)*C_in;
        float* V4 =V_data+(size_t)( 4*n_tiles+t)*C_in;
        float* V5 =V_data+(size_t)( 5*n_tiles+t)*C_in;
        float* V6 =V_data+(size_t)( 6*n_tiles+t)*C_in;
        float* V7 =V_data+(size_t)( 7*n_tiles+t)*C_in;
        float* V8 =V_data+(size_t)( 8*n_tiles+t)*C_in;
        float* V9 =V_data+(size_t)( 9*n_tiles+t)*C_in;
        float* V10=V_data+(size_t)(10*n_tiles+t)*C_in;
        float* V11=V_data+(size_t)(11*n_tiles+t)*C_in;
        int ci = 0;
#if defined(__AVX512F__)
        for (; ci + 15 < C_in; ci += 16) {
            __m512 D0=_mm512_loadu_ps(d0+ci),D1=_mm512_loadu_ps(d1+ci);
            __m512 D2=_mm512_loadu_ps(d2+ci),D3=_mm512_loadu_ps(d3+ci);
            __m512 D4=_mm512_loadu_ps(d4+ci),D5=_mm512_loadu_ps(d5+ci);
            __m512 D6=_mm512_loadu_ps(d6+ci),D7=_mm512_loadu_ps(d7+ci);
            __m512 D8=_mm512_loadu_ps(d8+ci),D9=_mm512_loadu_ps(d9+ci);
            __m512 D10=_mm512_loadu_ps(d10+ci),D11=_mm512_loadu_ps(d11+ci);
            const __m512 c14400=_mm512_set1_ps(14400.f),cm14400=_mm512_set1_ps(-14400.f);
            const __m512 c21076=_mm512_set1_ps(21076.f),cm7645=_mm512_set1_ps(-7645.f);
            const __m512 c1023=_mm512_set1_ps(1023.f),cm55=_mm512_set1_ps(-55.f);
            const __m512 cm6676=_mm512_set1_ps(-6676.f),c969=_mm512_set1_ps(969.f);
            const __m512 cm54=_mm512_set1_ps(-54.f),c6676=_mm512_set1_ps(6676.f),cm969=_mm512_set1_ps(-969.f),c54=_mm512_set1_ps(54.f);
            const __m512 c3600=_mm512_set1_ps(3600.f),cm4369=_mm512_set1_ps(-4369.f);
            const __m512 c819=_mm512_set1_ps(819.f),cm51=_mm512_set1_ps(-51.f);
            const __m512 c7200=_mm512_set1_ps(7200.f),cm8738=_mm512_set1_ps(-8738.f);
            const __m512 c1638=_mm512_set1_ps(1638.f),cm102=_mm512_set1_ps(-102.f),c2=_mm512_set1_ps(2.f);
            const __m512 c1600=_mm512_set1_ps(1600.f),cm2164=_mm512_set1_ps(-2164.f);
            const __m512 c609=_mm512_set1_ps(609.f),cm46=_mm512_set1_ps(-46.f);
            const __m512 c4800=_mm512_set1_ps(4800.f),cm6492=_mm512_set1_ps(-6492.f);
            const __m512 c1827=_mm512_set1_ps(1827.f),cm138=_mm512_set1_ps(-138.f),c3=_mm512_set1_ps(3.f);
            const __m512 c900=_mm512_set1_ps(900.f),cm1261=_mm512_set1_ps(-1261.f);
            const __m512 c399=_mm512_set1_ps(399.f),cm39=_mm512_set1_ps(-39.f);
            const __m512 c3600b=_mm512_set1_ps(3600.f),cm5044=_mm512_set1_ps(-5044.f);
            const __m512 c1596=_mm512_set1_ps(1596.f),cm156=_mm512_set1_ps(-156.f),c4=_mm512_set1_ps(4.f);
            const __m512 c576=_mm512_set1_ps(576.f),cm820=_mm512_set1_ps(-820.f);
            const __m512 c273=_mm512_set1_ps(273.f),cm30=_mm512_set1_ps(-30.f);
            const __m512 c2880=_mm512_set1_ps(2880.f),cm4100=_mm512_set1_ps(-4100.f);
            const __m512 c1365=_mm512_set1_ps(1365.f),cm150=_mm512_set1_ps(-150.f),c5=_mm512_set1_ps(5.f);
            __m512 S12=_mm512_add_ps(D1,D2),D12=_mm512_sub_ps(D2,D1);
            __m512 S34=_mm512_add_ps(D3,D4),D34=_mm512_sub_ps(D3,D4);
            __m512 S56=_mm512_add_ps(D5,D6),D56=_mm512_sub_ps(D5,D6);
            __m512 S78=_mm512_add_ps(D7,D8),D78=_mm512_sub_ps(D7,D8);
            __m512 S910=_mm512_add_ps(D9,D10),D910=_mm512_sub_ps(D9,D10);
            __m512 P1=_mm512_fmadd_ps(c3600,D2,_mm512_fmadd_ps(cm4369,D4,_mm512_fmadd_ps(c819,D6,_mm512_fmadd_ps(cm51,D8,D10))));
            __m512 Q1=_mm512_fmadd_ps(c7200,D1,_mm512_fmadd_ps(cm8738,D3,_mm512_fmadd_ps(c1638,D5,_mm512_fmadd_ps(cm102,D7,_mm512_mul_ps(c2,D9)))));
            __m512 P2=_mm512_fmadd_ps(c1600,D2,_mm512_fmadd_ps(cm2164,D4,_mm512_fmadd_ps(c609,D6,_mm512_fmadd_ps(cm46,D8,D10))));
            __m512 Q2=_mm512_fmadd_ps(c4800,D1,_mm512_fmadd_ps(cm6492,D3,_mm512_fmadd_ps(c1827,D5,_mm512_fmadd_ps(cm138,D7,_mm512_mul_ps(c3,D9)))));
            __m512 P3=_mm512_fmadd_ps(c900,D2,_mm512_fmadd_ps(cm1261,D4,_mm512_fmadd_ps(c399,D6,_mm512_fmadd_ps(cm39,D8,D10))));
            __m512 Q3=_mm512_fmadd_ps(c3600b,D1,_mm512_fmadd_ps(cm5044,D3,_mm512_fmadd_ps(c1596,D5,_mm512_fmadd_ps(cm156,D7,_mm512_mul_ps(c4,D9)))));
            __m512 P4=_mm512_fmadd_ps(c576,D2,_mm512_fmadd_ps(cm820,D4,_mm512_fmadd_ps(c273,D6,_mm512_fmadd_ps(cm30,D8,D10))));
            __m512 Q4=_mm512_fmadd_ps(c2880,D1,_mm512_fmadd_ps(cm4100,D3,_mm512_fmadd_ps(c1365,D5,_mm512_fmadd_ps(cm150,D7,_mm512_mul_ps(c5,D9)))));
            _mm512_storeu_ps(V0+ci, _mm512_fmadd_ps(cm14400,D0,_mm512_fmadd_ps(c21076,D2,_mm512_fmadd_ps(cm7645,D4,_mm512_fmadd_ps(c1023,D6,_mm512_fmadd_ps(cm55,D8,D10))))));
            _mm512_storeu_ps(V1+ci, _mm512_fmadd_ps(c14400,S12,_mm512_fmadd_ps(cm6676,S34,_mm512_fmadd_ps(c969,S56,_mm512_fmadd_ps(cm54,S78,S910)))));
            _mm512_storeu_ps(V2+ci, _mm512_fmadd_ps(c14400,D12,_mm512_fmadd_ps(c6676,D34,_mm512_fmadd_ps(cm969,D56,_mm512_fmadd_ps(c54,D78,_mm512_sub_ps(_mm512_setzero_ps(),D910))))));
            _mm512_storeu_ps(V3+ci,  _mm512_add_ps(P1,Q1)); _mm512_storeu_ps(V4+ci,  _mm512_sub_ps(P1,Q1));
            _mm512_storeu_ps(V5+ci,  _mm512_add_ps(P2,Q2)); _mm512_storeu_ps(V6+ci,  _mm512_sub_ps(P2,Q2));
            _mm512_storeu_ps(V7+ci,  _mm512_add_ps(P3,Q3)); _mm512_storeu_ps(V8+ci,  _mm512_sub_ps(P3,Q3));
            _mm512_storeu_ps(V9+ci,  _mm512_add_ps(P4,Q4)); _mm512_storeu_ps(V10+ci, _mm512_sub_ps(P4,Q4));
            _mm512_storeu_ps(V11+ci, _mm512_fmadd_ps(cm14400,D1,_mm512_fmadd_ps(c21076,D3,_mm512_fmadd_ps(cm7645,D5,_mm512_fmadd_ps(c1023,D7,_mm512_fmadd_ps(cm55,D9,D11))))));
        }
#elif defined(__AVX2__)
        for (; ci + 7 < C_in; ci += 8) {
            __m256 D0=_mm256_loadu_ps(d0+ci),D1=_mm256_loadu_ps(d1+ci);
            __m256 D2=_mm256_loadu_ps(d2+ci),D3=_mm256_loadu_ps(d3+ci);
            __m256 D4=_mm256_loadu_ps(d4+ci),D5=_mm256_loadu_ps(d5+ci);
            __m256 D6=_mm256_loadu_ps(d6+ci),D7=_mm256_loadu_ps(d7+ci);
            __m256 D8=_mm256_loadu_ps(d8+ci),D9=_mm256_loadu_ps(d9+ci);
            __m256 D10=_mm256_loadu_ps(d10+ci),D11=_mm256_loadu_ps(d11+ci);
            const __m256 c14400=_mm256_set1_ps(14400.f),cm14400=_mm256_set1_ps(-14400.f);
            const __m256 c21076=_mm256_set1_ps(21076.f),cm7645=_mm256_set1_ps(-7645.f);
            const __m256 c1023=_mm256_set1_ps(1023.f),cm55=_mm256_set1_ps(-55.f);
            const __m256 cm6676=_mm256_set1_ps(-6676.f),c969=_mm256_set1_ps(969.f);
            const __m256 cm54=_mm256_set1_ps(-54.f),c6676=_mm256_set1_ps(6676.f),cm969=_mm256_set1_ps(-969.f),c54=_mm256_set1_ps(54.f);
            const __m256 c3600=_mm256_set1_ps(3600.f),cm4369=_mm256_set1_ps(-4369.f);
            const __m256 c819=_mm256_set1_ps(819.f),cm51=_mm256_set1_ps(-51.f);
            const __m256 c7200=_mm256_set1_ps(7200.f),cm8738=_mm256_set1_ps(-8738.f);
            const __m256 c1638=_mm256_set1_ps(1638.f),cm102=_mm256_set1_ps(-102.f),c2=_mm256_set1_ps(2.f);
            const __m256 c1600=_mm256_set1_ps(1600.f),cm2164=_mm256_set1_ps(-2164.f);
            const __m256 c609=_mm256_set1_ps(609.f),cm46=_mm256_set1_ps(-46.f);
            const __m256 c4800=_mm256_set1_ps(4800.f),cm6492=_mm256_set1_ps(-6492.f);
            const __m256 c1827=_mm256_set1_ps(1827.f),cm138=_mm256_set1_ps(-138.f),c3=_mm256_set1_ps(3.f);
            const __m256 c900=_mm256_set1_ps(900.f),cm1261=_mm256_set1_ps(-1261.f);
            const __m256 c399=_mm256_set1_ps(399.f),cm39=_mm256_set1_ps(-39.f);
            const __m256 c3600b=_mm256_set1_ps(3600.f),cm5044=_mm256_set1_ps(-5044.f);
            const __m256 c1596=_mm256_set1_ps(1596.f),cm156=_mm256_set1_ps(-156.f),c4=_mm256_set1_ps(4.f);
            const __m256 c576=_mm256_set1_ps(576.f),cm820=_mm256_set1_ps(-820.f);
            const __m256 c273=_mm256_set1_ps(273.f),cm30=_mm256_set1_ps(-30.f);
            const __m256 c2880=_mm256_set1_ps(2880.f),cm4100=_mm256_set1_ps(-4100.f);
            const __m256 c1365=_mm256_set1_ps(1365.f),cm150=_mm256_set1_ps(-150.f),c5=_mm256_set1_ps(5.f);
            __m256 S12=_mm256_add_ps(D1,D2),D12=_mm256_sub_ps(D2,D1);
            __m256 S34=_mm256_add_ps(D3,D4),D34=_mm256_sub_ps(D3,D4);
            __m256 S56=_mm256_add_ps(D5,D6),D56=_mm256_sub_ps(D5,D6);
            __m256 S78=_mm256_add_ps(D7,D8),D78=_mm256_sub_ps(D7,D8);
            __m256 S910=_mm256_add_ps(D9,D10),D910=_mm256_sub_ps(D9,D10);
            __m256 P1=_mm256_fmadd_ps(c3600,D2,_mm256_fmadd_ps(cm4369,D4,_mm256_fmadd_ps(c819,D6,_mm256_fmadd_ps(cm51,D8,D10))));
            __m256 Q1=_mm256_fmadd_ps(c7200,D1,_mm256_fmadd_ps(cm8738,D3,_mm256_fmadd_ps(c1638,D5,_mm256_fmadd_ps(cm102,D7,_mm256_mul_ps(c2,D9)))));
            __m256 P2=_mm256_fmadd_ps(c1600,D2,_mm256_fmadd_ps(cm2164,D4,_mm256_fmadd_ps(c609,D6,_mm256_fmadd_ps(cm46,D8,D10))));
            __m256 Q2=_mm256_fmadd_ps(c4800,D1,_mm256_fmadd_ps(cm6492,D3,_mm256_fmadd_ps(c1827,D5,_mm256_fmadd_ps(cm138,D7,_mm256_mul_ps(c3,D9)))));
            __m256 P3=_mm256_fmadd_ps(c900,D2,_mm256_fmadd_ps(cm1261,D4,_mm256_fmadd_ps(c399,D6,_mm256_fmadd_ps(cm39,D8,D10))));
            __m256 Q3=_mm256_fmadd_ps(c3600b,D1,_mm256_fmadd_ps(cm5044,D3,_mm256_fmadd_ps(c1596,D5,_mm256_fmadd_ps(cm156,D7,_mm256_mul_ps(c4,D9)))));
            __m256 P4=_mm256_fmadd_ps(c576,D2,_mm256_fmadd_ps(cm820,D4,_mm256_fmadd_ps(c273,D6,_mm256_fmadd_ps(cm30,D8,D10))));
            __m256 Q4=_mm256_fmadd_ps(c2880,D1,_mm256_fmadd_ps(cm4100,D3,_mm256_fmadd_ps(c1365,D5,_mm256_fmadd_ps(cm150,D7,_mm256_mul_ps(c5,D9)))));
            _mm256_storeu_ps(V0+ci, _mm256_fmadd_ps(cm14400,D0,_mm256_fmadd_ps(c21076,D2,_mm256_fmadd_ps(cm7645,D4,_mm256_fmadd_ps(c1023,D6,_mm256_fmadd_ps(cm55,D8,D10))))));
            _mm256_storeu_ps(V1+ci, _mm256_fmadd_ps(c14400,S12,_mm256_fmadd_ps(cm6676,S34,_mm256_fmadd_ps(c969,S56,_mm256_fmadd_ps(cm54,S78,S910)))));
            _mm256_storeu_ps(V2+ci, _mm256_fmadd_ps(c14400,D12,_mm256_fmadd_ps(c6676,D34,_mm256_fmadd_ps(cm969,D56,_mm256_fmadd_ps(c54,D78,_mm256_sub_ps(_mm256_setzero_ps(),D910))))));
            _mm256_storeu_ps(V3+ci,  _mm256_add_ps(P1,Q1)); _mm256_storeu_ps(V4+ci,  _mm256_sub_ps(P1,Q1));
            _mm256_storeu_ps(V5+ci,  _mm256_add_ps(P2,Q2)); _mm256_storeu_ps(V6+ci,  _mm256_sub_ps(P2,Q2));
            _mm256_storeu_ps(V7+ci,  _mm256_add_ps(P3,Q3)); _mm256_storeu_ps(V8+ci,  _mm256_sub_ps(P3,Q3));
            _mm256_storeu_ps(V9+ci,  _mm256_add_ps(P4,Q4)); _mm256_storeu_ps(V10+ci, _mm256_sub_ps(P4,Q4));
            _mm256_storeu_ps(V11+ci, _mm256_fmadd_ps(cm14400,D1,_mm256_fmadd_ps(c21076,D3,_mm256_fmadd_ps(cm7645,D5,_mm256_fmadd_ps(c1023,D7,_mm256_fmadd_ps(cm55,D9,D11))))));
        }
#endif
        for (; ci < C_in; ++ci) {
            float d0v=d0[ci],d1v=d1[ci],d2v=d2[ci],d3v=d3[ci],d4v=d4[ci],d5v=d5[ci];
            float d6v=d6[ci],d7v=d7[ci],d8v=d8[ci],d9v=d9[ci],d10v=d10[ci],d11v=d11[ci];
            float s12=d1v+d2v,d12=d2v-d1v,s34=d3v+d4v,d34=d3v-d4v;
            float s56=d5v+d6v,d56=d5v-d6v,s78=d7v+d8v,d78=d7v-d8v;
            float s910=d9v+d10v,d910=d9v-d10v;
            float P1=3600.f*d2v-4369.f*d4v+819.f*d6v-51.f*d8v+d10v;
            float Q1=7200.f*d1v-8738.f*d3v+1638.f*d5v-102.f*d7v+2.f*d9v;
            float P2=1600.f*d2v-2164.f*d4v+609.f*d6v-46.f*d8v+d10v;
            float Q2=4800.f*d1v-6492.f*d3v+1827.f*d5v-138.f*d7v+3.f*d9v;
            float P3=900.f*d2v-1261.f*d4v+399.f*d6v-39.f*d8v+d10v;
            float Q3=3600.f*d1v-5044.f*d3v+1596.f*d5v-156.f*d7v+4.f*d9v;
            float P4=576.f*d2v-820.f*d4v+273.f*d6v-30.f*d8v+d10v;
            float Q4=2880.f*d1v-4100.f*d3v+1365.f*d5v-150.f*d7v+5.f*d9v;
            V0[ci] =-14400.f*d0v+21076.f*d2v-7645.f*d4v+1023.f*d6v-55.f*d8v+d10v;
            V1[ci] = 14400.f*s12-6676.f*s34+969.f*s56-54.f*s78+s910;
            V2[ci] = 14400.f*d12+6676.f*d34-969.f*d56+54.f*d78-d910;
            V3[ci]=P1+Q1; V4[ci]=P1-Q1; V5[ci]=P2+Q2; V6[ci]=P2-Q2;
            V7[ci]=P3+Q3; V8[ci]=P3-Q3; V9[ci]=P4+Q4; V10[ci]=P4-Q4;
            V11[ci]=-14400.f*d1v+21076.f*d3v-7645.f*d5v+1023.f*d7v-55.f*d9v+d11v;
        }
    }

    // 12 point GEMMs: M[p] = V[p] × Uw[p]
    for (int p = 0; p < 12; ++p) {
        sgemm_f32(V_data+(size_t)p*n_tiles*C_in, w_packed[p],
                  nullptr, M_data+(size_t)p*n_tiles*C_out, false,
                  n_tiles, C_in, C_out, /*in_parallel=*/false);
    }

    // Output transform AT12 × M + bias + relu
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
#endif
    for (int t = 0; t < n_tiles; ++t) {
        const int out_rows = std::min(2, T_out - 2 * t);
        int co = 0;
#if defined(__AVX512F__)
        for (; co + 15 < C_out; co += 16) {
            __m512 m0 =_mm512_loadu_ps(M_data+(size_t)( 0*n_tiles+t)*C_out+co);
            __m512 m1 =_mm512_loadu_ps(M_data+(size_t)( 1*n_tiles+t)*C_out+co);
            __m512 m2 =_mm512_loadu_ps(M_data+(size_t)( 2*n_tiles+t)*C_out+co);
            __m512 m3 =_mm512_loadu_ps(M_data+(size_t)( 3*n_tiles+t)*C_out+co);
            __m512 m4 =_mm512_loadu_ps(M_data+(size_t)( 4*n_tiles+t)*C_out+co);
            __m512 m5 =_mm512_loadu_ps(M_data+(size_t)( 5*n_tiles+t)*C_out+co);
            __m512 m6 =_mm512_loadu_ps(M_data+(size_t)( 6*n_tiles+t)*C_out+co);
            __m512 m7 =_mm512_loadu_ps(M_data+(size_t)( 7*n_tiles+t)*C_out+co);
            __m512 m8 =_mm512_loadu_ps(M_data+(size_t)( 8*n_tiles+t)*C_out+co);
            __m512 m9 =_mm512_loadu_ps(M_data+(size_t)( 9*n_tiles+t)*C_out+co);
            __m512 m10=_mm512_loadu_ps(M_data+(size_t)(10*n_tiles+t)*C_out+co);
            __m512 m11=_mm512_loadu_ps(M_data+(size_t)(11*n_tiles+t)*C_out+co);
            const __m512 c2=_mm512_set1_ps(2.f),c3=_mm512_set1_ps(3.f),c4=_mm512_set1_ps(4.f),c5=_mm512_set1_ps(5.f);
            __m512 S12=_mm512_add_ps(m1,m2),A12=_mm512_sub_ps(m1,m2);
            __m512 S34=_mm512_add_ps(m3,m4),B34=_mm512_sub_ps(m3,m4);
            __m512 S56=_mm512_add_ps(m5,m6),C56=_mm512_sub_ps(m5,m6);
            __m512 S78=_mm512_add_ps(m7,m8),D78=_mm512_sub_ps(m7,m8);
            __m512 S910=_mm512_add_ps(m9,m10),E910=_mm512_sub_ps(m9,m10);
            __m512 y0=_mm512_add_ps(_mm512_add_ps(_mm512_add_ps(m0,S12),_mm512_add_ps(S34,S56)),_mm512_add_ps(S78,S910));
            __m512 y1=_mm512_add_ps(_mm512_fmadd_ps(c2,B34,_mm512_fmadd_ps(c3,C56,A12)),
                                    _mm512_add_ps(_mm512_fmadd_ps(c4,D78,_mm512_mul_ps(c5,E910)),m11));
            if (bias) { __m512 bv=_mm512_loadu_ps(bias+co);
                y0=_mm512_add_ps(y0,bv); y1=_mm512_add_ps(y1,bv); }
            if (relu) { __m512 zv=_mm512_setzero_ps();
                y0=_mm512_max_ps(y0,zv); y1=_mm512_max_ps(y1,zv); }
            float* ot=out_buf+(size_t)(2*t)*C_out+co;
            if(out_rows>0)_mm512_storeu_ps(ot+0*C_out,y0);
            if(out_rows>1)_mm512_storeu_ps(ot+1*C_out,y1);
        }
#elif defined(__AVX2__)
        for (; co + 7 < C_out; co += 8) {
            __m256 m0 =_mm256_loadu_ps(M_data+(size_t)( 0*n_tiles+t)*C_out+co);
            __m256 m1 =_mm256_loadu_ps(M_data+(size_t)( 1*n_tiles+t)*C_out+co);
            __m256 m2 =_mm256_loadu_ps(M_data+(size_t)( 2*n_tiles+t)*C_out+co);
            __m256 m3 =_mm256_loadu_ps(M_data+(size_t)( 3*n_tiles+t)*C_out+co);
            __m256 m4 =_mm256_loadu_ps(M_data+(size_t)( 4*n_tiles+t)*C_out+co);
            __m256 m5 =_mm256_loadu_ps(M_data+(size_t)( 5*n_tiles+t)*C_out+co);
            __m256 m6 =_mm256_loadu_ps(M_data+(size_t)( 6*n_tiles+t)*C_out+co);
            __m256 m7 =_mm256_loadu_ps(M_data+(size_t)( 7*n_tiles+t)*C_out+co);
            __m256 m8 =_mm256_loadu_ps(M_data+(size_t)( 8*n_tiles+t)*C_out+co);
            __m256 m9 =_mm256_loadu_ps(M_data+(size_t)( 9*n_tiles+t)*C_out+co);
            __m256 m10=_mm256_loadu_ps(M_data+(size_t)(10*n_tiles+t)*C_out+co);
            __m256 m11=_mm256_loadu_ps(M_data+(size_t)(11*n_tiles+t)*C_out+co);
            const __m256 c2=_mm256_set1_ps(2.f),c3=_mm256_set1_ps(3.f),c4=_mm256_set1_ps(4.f),c5=_mm256_set1_ps(5.f);
            __m256 S12=_mm256_add_ps(m1,m2),A12=_mm256_sub_ps(m1,m2);
            __m256 S34=_mm256_add_ps(m3,m4),B34=_mm256_sub_ps(m3,m4);
            __m256 S56=_mm256_add_ps(m5,m6),C56=_mm256_sub_ps(m5,m6);
            __m256 S78=_mm256_add_ps(m7,m8),D78=_mm256_sub_ps(m7,m8);
            __m256 S910=_mm256_add_ps(m9,m10),E910=_mm256_sub_ps(m9,m10);
            __m256 y0=_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(m0,S12),_mm256_add_ps(S34,S56)),_mm256_add_ps(S78,S910));
            __m256 y1=_mm256_add_ps(_mm256_fmadd_ps(c2,B34,_mm256_fmadd_ps(c3,C56,A12)),
                                    _mm256_add_ps(_mm256_fmadd_ps(c4,D78,_mm256_mul_ps(c5,E910)),m11));
            if (bias) { __m256 bv=_mm256_loadu_ps(bias+co);
                y0=_mm256_add_ps(y0,bv); y1=_mm256_add_ps(y1,bv); }
            if (relu) { __m256 zv=_mm256_setzero_ps();
                y0=_mm256_max_ps(y0,zv); y1=_mm256_max_ps(y1,zv); }
            float* ot=out_buf+(size_t)(2*t)*C_out+co;
            if(out_rows>0)_mm256_storeu_ps(ot+0*C_out,y0);
            if(out_rows>1)_mm256_storeu_ps(ot+1*C_out,y1);
        }
#endif
        for (; co < C_out; ++co) {
            float m0v =M_data[(size_t)( 0*n_tiles+t)*C_out+co];
            float m1v =M_data[(size_t)( 1*n_tiles+t)*C_out+co];
            float m2v =M_data[(size_t)( 2*n_tiles+t)*C_out+co];
            float m3v =M_data[(size_t)( 3*n_tiles+t)*C_out+co];
            float m4v =M_data[(size_t)( 4*n_tiles+t)*C_out+co];
            float m5v =M_data[(size_t)( 5*n_tiles+t)*C_out+co];
            float m6v =M_data[(size_t)( 6*n_tiles+t)*C_out+co];
            float m7v =M_data[(size_t)( 7*n_tiles+t)*C_out+co];
            float m8v =M_data[(size_t)( 8*n_tiles+t)*C_out+co];
            float m9v =M_data[(size_t)( 9*n_tiles+t)*C_out+co];
            float m10v=M_data[(size_t)(10*n_tiles+t)*C_out+co];
            float m11v=M_data[(size_t)(11*n_tiles+t)*C_out+co];
            float A=m1v-m2v,B=m3v-m4v,C_v=m5v-m6v,D=m7v-m8v,E=m9v-m10v;
            float y[2];
            y[0]=m0v+(m1v+m2v)+(m3v+m4v)+(m5v+m6v)+(m7v+m8v)+(m9v+m10v);
            y[1]=A+2.f*B+3.f*C_v+4.f*D+5.f*E+m11v;
            float* ot=out_buf+(size_t)(2*t)*C_out+co;
            for (int r = 0; r < out_rows; ++r) {
                float v=y[r]+(bias?bias[co]:0.f);
                ot[r*C_out]= relu ? std::max(0.f,v) : v;
            }
        }
    }
}
