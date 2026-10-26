#include "work_split.hpp"
#include "../ops_neon.hpp"
#include "profile_internal.hpp"
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>

// ── Signed×signed int8 dot-product accumulate (AVX2 emulation of vpdpbssd) ──
// Computes: for each of 8 int32 lanes j:
//   acc[j] += a[j*4+0]*b[j*4+0] + a[j*4+1]*b[j*4+1] + a[j*4+2]*b[j*4+2] + a[j*4+3]*b[j*4+3]
// Uses abs/sign/maddubs trick for signed×signed.  Max error vs true product:
//   when a=-128: b sign-flip overflows for unsigned abs→signed, giving error = 256 per element.
//   This is identical to the TFLite/XNNPACK AVX2 approach.
// Signed×signed int8 dot-product accumulate (AVX2, 256-bit).
//
// Avoids the sign_epi8 wrapping bug: sign_epi8(b, a) wraps -(-128) to -128 when
// a < 0 and b == -128 (error = 256*|a| per element, not just ±1).
// Avoids the maddubs saturation bug: maddubs(a_u8, b) can overflow int16 when
// a_u8 is near 255 and |b| is near 128.
//
// Solution: sign-extend both operands to int16 and use madd_epi16 (exact int32).
// The 256-bit b vector holds 8 channels × 4 K-elements; we process it in two
// 128-bit halves (channels 0..3 and 4..7), then hadd + permute to align the
// per-channel sums into the 8 int32 acc lanes.
static inline __m256i dpbssd_avx2(__m256i acc, __m256i a, __m256i b) {
    // a has 4 unique s8 values broadcast to all 8 int32 lanes (set1_epi32 pattern).
    // Lower 128 bits of a = those 4 bytes × 4.
    __m256i a16    = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a));   // 16×s16
    __m256i b_lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b));   // ch0..3 × 4 K
    __m256i b_hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 1)); // ch4..7
    // p_lo[2j..2j+1] = pair sums for channel j (j=0..3); p_hi for j=4..7
    __m256i p_lo   = _mm256_madd_epi16(a16, b_lo16);
    __m256i p_hi   = _mm256_madd_epi16(a16, b_hi16);
    // hadd sums adjacent pairs: [ch0,ch1,ch4,ch5 | ch2,ch3,ch6,ch7]
    // permute4x64 0xD8 ([0,2,1,3]) reorders to [ch0,ch1,ch2,ch3,ch4,ch5,ch6,ch7]
    __m256i h      = _mm256_hadd_epi32(p_lo, p_hi);
    __m256i sums   = _mm256_permute4x64_epi64(h, 0xD8);
    return _mm256_add_epi32(acc, sums);
}


#if defined(__AVX2__) && !defined(__AVX512BW__)
// ── AVX2 INT8 K-step macro: 8 rows × 8 channels, hadd-free K-pair madd ──────
// Requires local vars: b_co (const int8_t*), a0..a7 (row pointers), acc0..acc7.
// B layout: K-pair interleaved — 32-byte block per K-step:
//   bytes  0..15: (ch0_k0,ch0_k1),(ch1_k0,ch1_k1),...,(ch7_k0,ch7_k1)  [K-pair 0]
//   bytes 16..31: (ch0_k2,ch0_k3),(ch1_k2,ch1_k3),...,(ch7_k2,ch7_k3)  [K-pair 1]
// pair_code = (s16(k_even)|(s16(k_odd)<<16)) as int32 → set1_epi32 + madd_epi16
//   gives 8 channel dot-product sums with NO hadd/permute.
// Eliminates ~17 port-5 uops per K-step vs dpbssd_avx2 + enables full inlining
// (no function call / kloop_out store-load roundtrip).
#define AVXK8(kb) do { \
    const __m256i _bvlo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b_co + (kb)*32))); \
    const __m256i _bvhi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b_co + (kb)*32 + 16))); \
    int32_t _i0; memcpy(&_i0, a0+(kb)*4, 4); \
    int32_t _i1; memcpy(&_i1, a1+(kb)*4, 4); \
    int32_t _i2; memcpy(&_i2, a2+(kb)*4, 4); \
    int32_t _i3; memcpy(&_i3, a3+(kb)*4, 4); \
    int32_t _i4; memcpy(&_i4, a4+(kb)*4, 4); \
    int32_t _i5; memcpy(&_i5, a5+(kb)*4, 4); \
    int32_t _i6; memcpy(&_i6, a6+(kb)*4, 4); \
    int32_t _i7; memcpy(&_i7, a7+(kb)*4, 4); \
    acc0=_mm256_add_epi32(acc0,_mm256_madd_epi16(_mm256_set1_epi32((uint16_t)(int8_t)_i0|((uint32_t)(uint16_t)(int8_t)(_i0>>8)<<16)),_bvlo)); \
    acc1=_mm256_add_epi32(acc1,_mm256_madd_epi16(_mm256_set1_epi32((uint16_t)(int8_t)_i1|((uint32_t)(uint16_t)(int8_t)(_i1>>8)<<16)),_bvlo)); \
    acc2=_mm256_add_epi32(acc2,_mm256_madd_epi16(_mm256_set1_epi32((uint16_t)(int8_t)_i2|((uint32_t)(uint16_t)(int8_t)(_i2>>8)<<16)),_bvlo)); \
    acc3=_mm256_add_epi32(acc3,_mm256_madd_epi16(_mm256_set1_epi32((uint16_t)(int8_t)_i3|((uint32_t)(uint16_t)(int8_t)(_i3>>8)<<16)),_bvlo)); \
    acc4=_mm256_add_epi32(acc4,_mm256_madd_epi16(_mm256_set1_epi32((uint16_t)(int8_t)_i4|((uint32_t)(uint16_t)(int8_t)(_i4>>8)<<16)),_bvlo)); \
    acc5=_mm256_add_epi32(acc5,_mm256_madd_epi16(_mm256_set1_epi32((uint16_t)(int8_t)_i5|((uint32_t)(uint16_t)(int8_t)(_i5>>8)<<16)),_bvlo)); \
    acc6=_mm256_add_epi32(acc6,_mm256_madd_epi16(_mm256_set1_epi32((uint16_t)(int8_t)_i6|((uint32_t)(uint16_t)(int8_t)(_i6>>8)<<16)),_bvlo)); \
    acc7=_mm256_add_epi32(acc7,_mm256_madd_epi16(_mm256_set1_epi32((uint16_t)(int8_t)_i7|((uint32_t)(uint16_t)(int8_t)(_i7>>8)<<16)),_bvlo)); \
    acc0=_mm256_add_epi32(acc0,_mm256_madd_epi16(_mm256_set1_epi32((uint16_t)(int8_t)(_i0>>16)|((uint32_t)(uint16_t)(int8_t)(_i0>>24)<<16)),_bvhi)); \
    acc1=_mm256_add_epi32(acc1,_mm256_madd_epi16(_mm256_set1_epi32((uint16_t)(int8_t)(_i1>>16)|((uint32_t)(uint16_t)(int8_t)(_i1>>24)<<16)),_bvhi)); \
    acc2=_mm256_add_epi32(acc2,_mm256_madd_epi16(_mm256_set1_epi32((uint16_t)(int8_t)(_i2>>16)|((uint32_t)(uint16_t)(int8_t)(_i2>>24)<<16)),_bvhi)); \
    acc3=_mm256_add_epi32(acc3,_mm256_madd_epi16(_mm256_set1_epi32((uint16_t)(int8_t)(_i3>>16)|((uint32_t)(uint16_t)(int8_t)(_i3>>24)<<16)),_bvhi)); \
    acc4=_mm256_add_epi32(acc4,_mm256_madd_epi16(_mm256_set1_epi32((uint16_t)(int8_t)(_i4>>16)|((uint32_t)(uint16_t)(int8_t)(_i4>>24)<<16)),_bvhi)); \
    acc5=_mm256_add_epi32(acc5,_mm256_madd_epi16(_mm256_set1_epi32((uint16_t)(int8_t)(_i5>>16)|((uint32_t)(uint16_t)(int8_t)(_i5>>24)<<16)),_bvhi)); \
    acc6=_mm256_add_epi32(acc6,_mm256_madd_epi16(_mm256_set1_epi32((uint16_t)(int8_t)(_i6>>16)|((uint32_t)(uint16_t)(int8_t)(_i6>>24)<<16)),_bvhi)); \
    acc7=_mm256_add_epi32(acc7,_mm256_madd_epi16(_mm256_set1_epi32((uint16_t)(int8_t)(_i7>>16)|((uint32_t)(uint16_t)(int8_t)(_i7>>24)<<16)),_bvhi)); \
} while(0)
// K-tail macro: handles 1-3 remaining K elements per row.
// Requires: a0..a7, acc0..acc7, k_full (int), bvlo/bvhi loaded by caller, K (int).
#define AVXK8_TAIL(ar, acc, bvlo, bvhi) do { \
    uint8_t _t[4]={}; \
    for (int _j=0; _j<(K&3); ++_j) _t[_j]=(uint8_t)(ar)[k_full*4+_j]; \
    int32_t _iv; memcpy(&_iv, _t, 4); \
    const int32_t _p01=(uint16_t)(int8_t)_iv|((uint32_t)(uint16_t)(int8_t)(_iv>>8)<<16); \
    const int32_t _p23=(uint16_t)(int8_t)(_iv>>16)|((uint32_t)(uint16_t)(int8_t)(_iv>>24)<<16); \
    acc=_mm256_add_epi32(acc,_mm256_madd_epi16(_mm256_set1_epi32(_p01),(bvlo))); \
    acc=_mm256_add_epi32(acc,_mm256_madd_epi16(_mm256_set1_epi32(_p23),(bvhi))); \
} while(0)

#if defined(__AVXVNNI__)
// AVX-VNNI computes u8*s8 dot products.  XORing signed activations with 0x80
// maps them to unsigned, then the caller subtracts 128 * sum(weights) per lane.
#define AVXVNNI8(kb) do { \
    const __m128i _bp01 = _mm_loadu_si128((const __m128i*)(b_co + (kb)*32)); \
    const __m128i _bp23 = _mm_loadu_si128((const __m128i*)(b_co + (kb)*32 + 16)); \
    const __m128i _b_lo = _mm_unpacklo_epi16(_bp01, _bp23); \
    const __m128i _b_hi = _mm_unpackhi_epi16(_bp01, _bp23); \
    const __m256i _bv = _mm256_inserti128_si256(_mm256_castsi128_si256(_b_lo), _b_hi, 1); \
    constexpr int32_t _xmask = (int32_t)0x80808080u; \
    int32_t _i0; memcpy(&_i0, a0+(kb)*4, 4); \
    int32_t _i1; memcpy(&_i1, a1+(kb)*4, 4); \
    int32_t _i2; memcpy(&_i2, a2+(kb)*4, 4); \
    int32_t _i3; memcpy(&_i3, a3+(kb)*4, 4); \
    int32_t _i4; memcpy(&_i4, a4+(kb)*4, 4); \
    int32_t _i5; memcpy(&_i5, a5+(kb)*4, 4); \
    int32_t _i6; memcpy(&_i6, a6+(kb)*4, 4); \
    int32_t _i7; memcpy(&_i7, a7+(kb)*4, 4); \
    acc0=_mm256_dpbusd_epi32(acc0,_mm256_set1_epi32(_i0 ^ _xmask),_bv); \
    acc1=_mm256_dpbusd_epi32(acc1,_mm256_set1_epi32(_i1 ^ _xmask),_bv); \
    acc2=_mm256_dpbusd_epi32(acc2,_mm256_set1_epi32(_i2 ^ _xmask),_bv); \
    acc3=_mm256_dpbusd_epi32(acc3,_mm256_set1_epi32(_i3 ^ _xmask),_bv); \
    acc4=_mm256_dpbusd_epi32(acc4,_mm256_set1_epi32(_i4 ^ _xmask),_bv); \
    acc5=_mm256_dpbusd_epi32(acc5,_mm256_set1_epi32(_i5 ^ _xmask),_bv); \
    acc6=_mm256_dpbusd_epi32(acc6,_mm256_set1_epi32(_i6 ^ _xmask),_bv); \
    acc7=_mm256_dpbusd_epi32(acc7,_mm256_set1_epi32(_i7 ^ _xmask),_bv); \
} while(0)
#define AVXVNNI8_TAIL_ROW(ar, acc, bv) do { \
    uint8_t _t[4]={}; \
    for (int _j=0; _j<(K&3); ++_j) _t[_j]=(uint8_t)(ar)[k_full*4+_j]; \
    int32_t _iv; memcpy(&_iv, _t, 4); \
    constexpr int32_t _xmask = (int32_t)0x80808080u; \
    acc=_mm256_dpbusd_epi32(acc,_mm256_set1_epi32(_iv ^ _xmask),(bv)); \
} while(0)
#endif
#endif // __AVX2__ && !__AVX512BW__

#ifdef __AVX512BW__
// ── Signed×signed int8 dot-product accumulate (512-bit, AVX-512BW) ──────────
// Emulates signed×signed vpdpbssd using abs/sign trick + maddubs.
// a is treated as signed int8 broadcast; b is signed int8 weights.
// Correctness: abs(a_i)=128 when a_i=-128, which is safe as uint8 in maddubs.
static inline __m512i dpbssd_avx512bw(__m512i acc, __m512i a, __m512i b) {
    const __m512i zero = _mm512_setzero_si512();
    __m512i a_abs = _mm512_abs_epi8(a);
    // Negate b where a < 0 (zeros in a are handled by abs giving 0, so a_zero not needed)
    __mmask64 neg = _mm512_cmplt_epi8_mask(a, zero);
    __m512i b_sgn = _mm512_mask_blend_epi8(neg, b, _mm512_sub_epi8(zero, b));
    __m512i p16   = _mm512_maddubs_epi16(a_abs, b_sgn);   // uint8 × int8 → int16
    return _mm512_add_epi32(acc, _mm512_madd_epi16(p16, _mm512_set1_epi16(1)));
}
#endif  // __AVX512BW__

// ── noinline K-loop helper: 4 input rows × 3 N-blocks → 12 INT32 accumulators ─
// By isolating the K-loop in its own function, GCC allocates registers fresh:
//   12 acc + 3 B loads + 1 A broadcast = 16 zmm → no spilling.
// The caller handles VNNI correction, bias, and requantization after the call.
// Two specializations: VNNI (dpbusd, s8→u8 via XOR 0x80808080) and BW (dpbssd).
#ifdef __AVX512BW__

// Non-VNNI specialization: pure signed×signed (dpbssd emulation via abs/sign).
__attribute__((noinline))
static void gemm_4x3_kloop_bw(
    const int8_t* const* ar,   // ar[0..3]: 4 input row pointers
    const int8_t* b0, const int8_t* b1, const int8_t* b2,
    int k_full, int k_tail,    // K/4, K%4
    int32_t* __restrict__ out  // aligned(64), [12*16] int32 output
) {
    __m512i acc0a=_mm512_setzero_si512(), acc0b=_mm512_setzero_si512(), acc0c=_mm512_setzero_si512();
    __m512i acc1a=_mm512_setzero_si512(), acc1b=_mm512_setzero_si512(), acc1c=_mm512_setzero_si512();
    __m512i acc2a=_mm512_setzero_si512(), acc2b=_mm512_setzero_si512(), acc2c=_mm512_setzero_si512();
    __m512i acc3a=_mm512_setzero_si512(), acc3b=_mm512_setzero_si512(), acc3c=_mm512_setzero_si512();
#define K3BW(kb) do { \
    __m512i _b0=_mm512_loadu_si512(b0+(kb)*64); \
    __m512i _b1=_mm512_loadu_si512(b1+(kb)*64); \
    __m512i _b2=_mm512_loadu_si512(b2+(kb)*64); \
    int32_t _i0,_i1,_i2,_i3; \
    memcpy(&_i0,ar[0]+(kb)*4,4); memcpy(&_i1,ar[1]+(kb)*4,4); \
    memcpy(&_i2,ar[2]+(kb)*4,4); memcpy(&_i3,ar[3]+(kb)*4,4); \
    { __m512i _a=_mm512_set1_epi32(_i0); acc0a=dpbssd_avx512bw(acc0a,_a,_b0); acc0b=dpbssd_avx512bw(acc0b,_a,_b1); acc0c=dpbssd_avx512bw(acc0c,_a,_b2); } \
    { __m512i _a=_mm512_set1_epi32(_i1); acc1a=dpbssd_avx512bw(acc1a,_a,_b0); acc1b=dpbssd_avx512bw(acc1b,_a,_b1); acc1c=dpbssd_avx512bw(acc1c,_a,_b2); } \
    { __m512i _a=_mm512_set1_epi32(_i2); acc2a=dpbssd_avx512bw(acc2a,_a,_b0); acc2b=dpbssd_avx512bw(acc2b,_a,_b1); acc2c=dpbssd_avx512bw(acc2c,_a,_b2); } \
    { __m512i _a=_mm512_set1_epi32(_i3); acc3a=dpbssd_avx512bw(acc3a,_a,_b0); acc3b=dpbssd_avx512bw(acc3b,_a,_b1); acc3c=dpbssd_avx512bw(acc3c,_a,_b2); } \
} while(0)
    int kb = 0;
    for (; kb+3 < k_full; kb+=4) { K3BW(kb); K3BW(kb+1); K3BW(kb+2); K3BW(kb+3); }
    for (; kb < k_full; ++kb) { K3BW(kb); }
#undef K3BW
    if (k_tail > 0) {
        const int kb2 = k_full;
        __m512i bv0=_mm512_loadu_si512(b0+kb2*64), bv1=_mm512_loadu_si512(b1+kb2*64), bv2=_mm512_loadu_si512(b2+kb2*64);
        { uint8_t t[4]={}; for(int j=0;j<k_tail;++j)t[j]=(uint8_t)ar[0][kb2*4+j]; int32_t iv; memcpy(&iv,t,4); __m512i _a=_mm512_set1_epi32(iv); acc0a=dpbssd_avx512bw(acc0a,_a,bv0); acc0b=dpbssd_avx512bw(acc0b,_a,bv1); acc0c=dpbssd_avx512bw(acc0c,_a,bv2); }
        { uint8_t t[4]={}; for(int j=0;j<k_tail;++j)t[j]=(uint8_t)ar[1][kb2*4+j]; int32_t iv; memcpy(&iv,t,4); __m512i _a=_mm512_set1_epi32(iv); acc1a=dpbssd_avx512bw(acc1a,_a,bv0); acc1b=dpbssd_avx512bw(acc1b,_a,bv1); acc1c=dpbssd_avx512bw(acc1c,_a,bv2); }
        { uint8_t t[4]={}; for(int j=0;j<k_tail;++j)t[j]=(uint8_t)ar[2][kb2*4+j]; int32_t iv; memcpy(&iv,t,4); __m512i _a=_mm512_set1_epi32(iv); acc2a=dpbssd_avx512bw(acc2a,_a,bv0); acc2b=dpbssd_avx512bw(acc2b,_a,bv1); acc2c=dpbssd_avx512bw(acc2c,_a,bv2); }
        { uint8_t t[4]={}; for(int j=0;j<k_tail;++j)t[j]=(uint8_t)ar[3][kb2*4+j]; int32_t iv; memcpy(&iv,t,4); __m512i _a=_mm512_set1_epi32(iv); acc3a=dpbssd_avx512bw(acc3a,_a,bv0); acc3b=dpbssd_avx512bw(acc3b,_a,bv1); acc3c=dpbssd_avx512bw(acc3c,_a,bv2); }
    }
    _mm512_store_si512(out+ 0*16,acc0a); _mm512_store_si512(out+ 1*16,acc0b); _mm512_store_si512(out+ 2*16,acc0c);
    _mm512_store_si512(out+ 3*16,acc1a); _mm512_store_si512(out+ 4*16,acc1b); _mm512_store_si512(out+ 5*16,acc1c);
    _mm512_store_si512(out+ 6*16,acc2a); _mm512_store_si512(out+ 7*16,acc2b); _mm512_store_si512(out+ 8*16,acc2c);
    _mm512_store_si512(out+ 9*16,acc3a); _mm512_store_si512(out+10*16,acc3b); _mm512_store_si512(out+11*16,acc3c);
}

#ifdef __AVX512VNNI__
// VNNI specialization: s8→u8 via XOR 0x80808080, use dpbusd.
// Caller must subtract 128*b_row_sums from each output channel's accumulators.
__attribute__((noinline))
static void gemm_4x3_kloop_vnni(
    const int8_t* const* ar,
    const int8_t* b0, const int8_t* b1, const int8_t* b2,
    int k_full, int k_tail,
    int32_t* __restrict__ out
) {
    static constexpr int32_t XM = (int32_t)0x80808080u;
    __m512i acc0a=_mm512_setzero_si512(), acc0b=_mm512_setzero_si512(), acc0c=_mm512_setzero_si512();
    __m512i acc1a=_mm512_setzero_si512(), acc1b=_mm512_setzero_si512(), acc1c=_mm512_setzero_si512();
    __m512i acc2a=_mm512_setzero_si512(), acc2b=_mm512_setzero_si512(), acc2c=_mm512_setzero_si512();
    __m512i acc3a=_mm512_setzero_si512(), acc3b=_mm512_setzero_si512(), acc3c=_mm512_setzero_si512();
#define K3VN(kb) do { \
    __m512i _b0=_mm512_loadu_si512(b0+(kb)*64); \
    __m512i _b1=_mm512_loadu_si512(b1+(kb)*64); \
    __m512i _b2=_mm512_loadu_si512(b2+(kb)*64); \
    int32_t _i0,_i1,_i2,_i3; \
    memcpy(&_i0,ar[0]+(kb)*4,4); memcpy(&_i1,ar[1]+(kb)*4,4); \
    memcpy(&_i2,ar[2]+(kb)*4,4); memcpy(&_i3,ar[3]+(kb)*4,4); \
    _i0^=XM; _i1^=XM; _i2^=XM; _i3^=XM; \
    { __m512i _a=_mm512_set1_epi32(_i0); acc0a=_mm512_dpbusd_epi32(acc0a,_a,_b0); acc0b=_mm512_dpbusd_epi32(acc0b,_a,_b1); acc0c=_mm512_dpbusd_epi32(acc0c,_a,_b2); } \
    { __m512i _a=_mm512_set1_epi32(_i1); acc1a=_mm512_dpbusd_epi32(acc1a,_a,_b0); acc1b=_mm512_dpbusd_epi32(acc1b,_a,_b1); acc1c=_mm512_dpbusd_epi32(acc1c,_a,_b2); } \
    { __m512i _a=_mm512_set1_epi32(_i2); acc2a=_mm512_dpbusd_epi32(acc2a,_a,_b0); acc2b=_mm512_dpbusd_epi32(acc2b,_a,_b1); acc2c=_mm512_dpbusd_epi32(acc2c,_a,_b2); } \
    { __m512i _a=_mm512_set1_epi32(_i3); acc3a=_mm512_dpbusd_epi32(acc3a,_a,_b0); acc3b=_mm512_dpbusd_epi32(acc3b,_a,_b1); acc3c=_mm512_dpbusd_epi32(acc3c,_a,_b2); } \
} while(0)
    int kb = 0;
    for (; kb+3 < k_full; kb+=4) { K3VN(kb); K3VN(kb+1); K3VN(kb+2); K3VN(kb+3); }
    for (; kb < k_full; ++kb) { K3VN(kb); }
#undef K3VN
    if (k_tail > 0) {
        const int kb2 = k_full;
        __m512i bv0=_mm512_loadu_si512(b0+kb2*64), bv1=_mm512_loadu_si512(b1+kb2*64), bv2=_mm512_loadu_si512(b2+kb2*64);
        { uint8_t t[4]={}; for(int j=0;j<k_tail;++j)t[j]=(uint8_t)ar[0][kb2*4+j]; int32_t iv; memcpy(&iv,t,4); iv^=XM; __m512i _a=_mm512_set1_epi32(iv); acc0a=_mm512_dpbusd_epi32(acc0a,_a,bv0); acc0b=_mm512_dpbusd_epi32(acc0b,_a,bv1); acc0c=_mm512_dpbusd_epi32(acc0c,_a,bv2); }
        { uint8_t t[4]={}; for(int j=0;j<k_tail;++j)t[j]=(uint8_t)ar[1][kb2*4+j]; int32_t iv; memcpy(&iv,t,4); iv^=XM; __m512i _a=_mm512_set1_epi32(iv); acc1a=_mm512_dpbusd_epi32(acc1a,_a,bv0); acc1b=_mm512_dpbusd_epi32(acc1b,_a,bv1); acc1c=_mm512_dpbusd_epi32(acc1c,_a,bv2); }
        { uint8_t t[4]={}; for(int j=0;j<k_tail;++j)t[j]=(uint8_t)ar[2][kb2*4+j]; int32_t iv; memcpy(&iv,t,4); iv^=XM; __m512i _a=_mm512_set1_epi32(iv); acc2a=_mm512_dpbusd_epi32(acc2a,_a,bv0); acc2b=_mm512_dpbusd_epi32(acc2b,_a,bv1); acc2c=_mm512_dpbusd_epi32(acc2c,_a,bv2); }
        { uint8_t t[4]={}; for(int j=0;j<k_tail;++j)t[j]=(uint8_t)ar[3][kb2*4+j]; int32_t iv; memcpy(&iv,t,4); iv^=XM; __m512i _a=_mm512_set1_epi32(iv); acc3a=_mm512_dpbusd_epi32(acc3a,_a,bv0); acc3b=_mm512_dpbusd_epi32(acc3b,_a,bv1); acc3c=_mm512_dpbusd_epi32(acc3c,_a,bv2); }
    }
    _mm512_store_si512(out+ 0*16,acc0a); _mm512_store_si512(out+ 1*16,acc0b); _mm512_store_si512(out+ 2*16,acc0c);
    _mm512_store_si512(out+ 3*16,acc1a); _mm512_store_si512(out+ 4*16,acc1b); _mm512_store_si512(out+ 5*16,acc1c);
    _mm512_store_si512(out+ 6*16,acc2a); _mm512_store_si512(out+ 7*16,acc2b); _mm512_store_si512(out+ 8*16,acc2c);
    _mm512_store_si512(out+ 9*16,acc3a); _mm512_store_si512(out+10*16,acc3b); _mm512_store_si512(out+11*16,acc3c);
}
#endif  // __AVX512VNNI__
#endif  // __AVX512BW__ (noinline helpers)

#endif  // __AVX2__ || __AVX512F__

// ──────────────────────────────────────────────────────────────
// Weight packing: [C_out, K] → packed format
//
// SMMLA (ARM I8MM, __ARM_FEATURE_MATMUL_INT8):
//   tile_n=8, tile_k=8 → [N/8, K/8, 64 bytes]
//   Each 64-byte block = 4 col-pairs × [col0[k0..7], col1[k0..7]]
//   Enables vmmlaq_s32: 2 rows × 8 k × 2 cols → 2×2 int32 result
//
// SDOT (ARM NEON, fallback):
//   [N/8, K/4, 32 bytes] — b_co0/b_co1 contiguous (16 bytes apart, same cache line)
//   in NR=8 GEMM loop. Reduces B cache misses vs old [N/4, K/4, 16] format.
//
// AVX-512: tile=16, AVX2: tile=8 (unchanged)
// ──────────────────────────────────────────────────────────────
int8_t* pack_weights_sdot(const int8_t* w, int C_out, int K)
{
#ifdef __ARM_FEATURE_MATMUL_INT8
    // SMMLA packing: 8 output channels × 8 k-elements per block (64 bytes)
    // Layout: [Co_t8, K8, 4_col_pairs × 16_bytes]
    // col_pair j: bytes [0..7] = w[co_blk*8+j*2, k_base..k_base+7]
    //             bytes [8..15] = w[co_blk*8+j*2+1, k_base..k_base+7]
    const int Co_t8 = (C_out + 7) / 8;
    const int K8    = (K + 7) / 8;
    int8_t* packed = new int8_t[Co_t8 * K8 * 64]();
    for (int co_blk = 0; co_blk < Co_t8; ++co_blk) {
        for (int k_blk = 0; k_blk < K8; ++k_blk) {
            int8_t* dst = packed + (co_blk * K8 + k_blk) * 64;
            for (int cp = 0; cp < 4; ++cp) {
                int c0 = co_blk * 8 + cp * 2;
                int c1 = c0 + 1;
                for (int ki = 0; ki < 8; ++ki) {
                    int k = k_blk * 8 + ki;
                    dst[cp * 16 + ki]     = (c0 < C_out && k < K) ? w[c0 * K + k] : 0;
                    dst[cp * 16 + 8 + ki] = (c1 < C_out && k < K) ? w[c1 * K + k] : 0;
                }
            }
        }
    }
    return packed;
#elif defined(__ARM_NEON)
    // SDOT path: [Co8, K4, 32] layout.
    // Block (co8_blk, k_blk): bytes [0..15]  = chans co8_blk*8+0..+3, k-vals k_blk*4..+3
    //                          bytes [16..31] = chans co8_blk*8+4..+7, k-vals k_blk*4..+3
    // In the NR=8 GEMM loop: b_co0 = b_tile+k_blk*32, b_co1 = b_tile+k_blk*32+16 (contiguous).
    {
        const int Co_t8 = (C_out + 7) / 8;
        const int K4    = (K + 3) / 4;
        int8_t* packed  = new int8_t[(size_t)Co_t8 * K4 * 32]();
        for (int co8_blk = 0; co8_blk < Co_t8; ++co8_blk) {
            for (int k_blk = 0; k_blk < K4; ++k_blk) {
                int8_t* dst = packed + ((size_t)co8_blk * K4 + k_blk) * 32;
                for (int oc = 0; oc < 8; ++oc) {
                    const int co = co8_blk * 8 + oc;
                    for (int ki = 0; ki < 4; ++ki) {
                        const int k = k_blk * 4 + ki;
                        dst[oc * 4 + ki] = (co < C_out && k < K) ? w[co * K + k] : 0;
                    }
                }
            }
        }
        return packed;
    }
#elif defined(__AVX512BW__)
    constexpr int tile = 16;
#elif defined(__AVX2__)
    constexpr int tile = 8;
#else
    constexpr int tile = 4;
#endif
#if !defined(__ARM_FEATURE_MATMUL_INT8) && !defined(__ARM_NEON)
    const int Co_t = (C_out + tile - 1) / tile;
    const int K4   = (K     + 3)        / 4;
    const int blk  = tile * 4;           // bytes per (co_blk, k_blk) block
    int8_t* packed = new int8_t[Co_t * K4 * blk]();   // zero-init

    for (int co_blk = 0; co_blk < Co_t; ++co_blk) {
        for (int k_blk = 0; k_blk < K4; ++k_blk) {
            int8_t* dst = packed + (co_blk * K4 + k_blk) * blk;
            for (int oc = 0; oc < tile; ++oc) {
                const int co = co_blk * tile + oc;
                for (int ki = 0; ki < 4; ++ki) {
                    const int k = k_blk * 4 + ki;
#if defined(__AVX2__) && !defined(__AVX512BW__)
                    // AVX2 K-pair interleaved layout: enables hadd-free madd_epi16 kernel.
                    // block: [kp*16 + oc*2 + ki2] where kp=ki/2, ki2=ki%2
                    dst[(ki >> 1) * 16 + oc * 2 + (ki & 1)] =
                        (co < C_out && k < K) ? w[co * K + k] : 0;
#else
                    dst[oc * 4 + ki] = (co < C_out && k < K) ? w[co * K + k] : 0;
#endif
                }
            }
        }
    }
    return packed;
#endif
}

void free_packed(int8_t* p) { delete[] p; }


// ──────────────────────────────────────────────────────────────
// INT8 GEMM  A[M,K] × B_packed → C
//
// Loop order: co_blk (N/4) outer → B slice stays in L1 across all M rows.
// Inner m-loop unrolled by 4 → 4× compute-to-load ratio improvement.
// nchw_out=true: writes C[n*M + m] (NCHW layout) instead of C[m*N + n].
//   Eliminates the caller's post-GEMM [oHW,N]→[N,oHW] transpose.
//
// Recursive dispatch: when in_parallel=false, forks an OMP team and calls
// itself with in_parallel=true; each thread runs the body directly using
// omp_get_thread_num()/omp_get_num_threads() from the enclosing team.
// ──────────────────────────────────────────────────────────────
void gemm_int8(
    const int8_t*  A,
    const int8_t*  B_packed,
    const int64_t* eff_bias,
    const int32_t* req_mult,
    const int32_t* req_exp,
    const float*   req_scale_f,
    int8_t         out_zp,
    void*          C,
    bool           is_float,
    int M, int K, int N,
    bool           nchw_out,
    bool           in_parallel,
    const int32_t* b_row_sums,
    StreamHandle   /* stream */)
{
    if (!in_parallel) {
#ifdef _OPENMP
        const int nthreads = omp_get_max_threads();
        // Don't spawn a new parallel region if already inside one (e.g. group-parallel).
        // In that case let the calling thread do all the work (single-thread SMMLA below).
        if (nthreads > 1 && !omp_in_parallel()) {
#pragma omp parallel num_threads(nthreads)
            gemm_int8(A, B_packed, eff_bias, req_mult, req_exp, req_scale_f,
                      out_zp, C, is_float,
                      M, K, N, nchw_out, /*in_parallel=*/true, b_row_sums);
            return;
        }
#endif
    }
    int K4  = (K + 3) / 4;
    int Co4 = (N + 3) / 4;

#ifdef __ARM_NEON
#ifdef __ARM_FEATURE_MATMUL_INT8
{
    // === SMMLA (I8MM) kernel: MR=8 rows × NR=8 cols, k-step=8, k-unroll=2 ===
    // vmmlaq_s32: 2-row×8k A × 8k×2-col B → 2×2 int32 result (32 MACs/insn)
    // k-unroll=2: 8A + 8B + 16acc = 32 NEON regs, zero spills.
    // B_packed: [N/8, K/8, 4_col_pairs×16_bytes] (64 bytes/block, packed by pack_weights_sdot)
    //   col_pair j: [w[n0+2j, k_base..k_base+7], w[n0+2j+1, k_base..k_base+7]]
    const int K8      = (K + 7) / 8;
    const int Co8     = (N + 7) / 8;
    const int m8_count = M / 8;
    // When in_parallel=true: use OMP thread info to split work across threads.
    // When in_parallel=false (including when called from a group-parallel context):
    //   use tid=0, nT=1 so this thread computes all m8-tiles (single-thread mode).
#ifdef _OPENMP
    const int tid = in_parallel ? omp_get_thread_num() : 0;
    const int nT  = in_parallel ? omp_get_num_threads() : 1;
#else
    const int tid = 0;
    const int nT  = 1;
#endif
    const bool n_part = (nT > 1) && (M < N);
    const int mi_s = n_part ? 0           : (tid * m8_count) / nT;
    const int mi_e = n_part ? m8_count    : ((tid + 1) * m8_count) / nT;
    const int co_s = n_part ? (tid * Co8) / nT       : 0;
    const int co_e = n_part ? ((tid + 1) * Co8) / nT : Co8;

    // A packing buffer: per m8-tile, layout [K8, 4_row_pairs × 16_bytes] = K8 × 64 bytes
    // a_pk[k_t8*64 + rp*16 + 0..7]  = A[m+rp*2,   k_t8*8..k_t8*8+7]
    // a_pk[k_t8*64 + rp*16 + 8..15] = A[m+rp*2+1, k_t8*8..k_t8*8+7]
    static thread_local std::vector<int8_t> tls_a_smmla;
    if ((int)tls_a_smmla.size() < K8 * 64) tls_a_smmla.resize(K8 * 64);

    // SMMLA inner-loop macro: 16 vmmlaq_s32 for 4 A-pairs × 4 B-pairs
    #define SMMLA16(va0,va1,va2,va3,vb0,vb1,vb2,vb3) \
        acc00=vmmlaq_s32(acc00,va0,vb0); acc01=vmmlaq_s32(acc01,va0,vb1); \
        acc02=vmmlaq_s32(acc02,va0,vb2); acc03=vmmlaq_s32(acc03,va0,vb3); \
        acc10=vmmlaq_s32(acc10,va1,vb0); acc11=vmmlaq_s32(acc11,va1,vb1); \
        acc12=vmmlaq_s32(acc12,va1,vb2); acc13=vmmlaq_s32(acc13,va1,vb3); \
        acc20=vmmlaq_s32(acc20,va2,vb0); acc21=vmmlaq_s32(acc21,va2,vb1); \
        acc22=vmmlaq_s32(acc22,va2,vb2); acc23=vmmlaq_s32(acc23,va2,vb3); \
        acc30=vmmlaq_s32(acc30,va3,vb0); acc31=vmmlaq_s32(acc31,va3,vb1); \
        acc32=vmmlaq_s32(acc32,va3,vb2); acc33=vmmlaq_s32(acc33,va3,vb3);

    for (int mi = mi_s; mi < mi_e; ++mi) {
        const int m = mi * 8;
        // Pack 8 rows of A into row-pair × K8 layout.
        // No full memset: full K8-blocks filled completely by 2×memcpy(8 bytes).
        // Only the partial tail block (if K%8 != 0) needs local zero-fill.
        int8_t* a_pk = tls_a_smmla.data();
        const int k8_full = K / 8;
        for (int rp = 0; rp < 4; ++rp) {
            const int8_t* ar0 = A + (size_t)(m + rp*2)   * K;
            const int8_t* ar1 = A + (size_t)(m + rp*2+1) * K;
            for (int k_t8 = 0; k_t8 < k8_full; ++k_t8) {
                int8_t* dst = a_pk + k_t8 * 64 + rp * 16;
                memcpy(dst,     ar0 + k_t8 * 8, 8);
                memcpy(dst + 8, ar1 + k_t8 * 8, 8);
            }
            if (K & 7) {
                int kb = k8_full * 8, kt = K & 7;
                int8_t* dst = a_pk + k8_full * 64 + rp * 16;
                memset(dst, 0, 16);              // zero the 16-byte slot
                memcpy(dst,     ar0 + kb, kt);   // fill valid prefix
                memcpy(dst + 8, ar1 + kb, kt);
            }
        }

        for (int co_blk = co_s; co_blk < co_e; ++co_blk) {
            const int8_t* b_co = B_packed + (size_t)co_blk * K8 * 64;
            const int n0      = co_blk * 8;
            const int n_valid = std::min(8, N - n0);
            __builtin_prefetch(b_co, 0, 3);
            __builtin_prefetch(b_co + 64, 0, 3);

            // 16 accumulators for 4 row-pairs × 4 col-pairs
            // acc{rp}{cp} = {C[2rp,2cp], C[2rp,2cp+1], C[2rp+1,2cp], C[2rp+1,2cp+1]}
            int32x4_t acc00=vdupq_n_s32(0), acc01=vdupq_n_s32(0);
            int32x4_t acc02=vdupq_n_s32(0), acc03=vdupq_n_s32(0);
            int32x4_t acc10=vdupq_n_s32(0), acc11=vdupq_n_s32(0);
            int32x4_t acc12=vdupq_n_s32(0), acc13=vdupq_n_s32(0);
            int32x4_t acc20=vdupq_n_s32(0), acc21=vdupq_n_s32(0);
            int32x4_t acc22=vdupq_n_s32(0), acc23=vdupq_n_s32(0);
            int32x4_t acc30=vdupq_n_s32(0), acc31=vdupq_n_s32(0);
            int32x4_t acc32=vdupq_n_s32(0), acc33=vdupq_n_s32(0);

            // k-unroll=2: 16 loads + 32 SMMlas per 2 k8-blocks → 32 registers exact
            int k_t8 = 0;
            for (; k_t8 + 1 < K8; k_t8 += 2) {
                __builtin_prefetch(b_co + (k_t8 + 4) * 64, 0, 3);
                const int8_t* ak0 = a_pk + k_t8 * 64;
                const int8_t* bk0 = b_co + k_t8 * 64;
                const int8_t* ak1 = ak0 + 64;
                const int8_t* bk1 = bk0 + 64;
                int8x16_t va00=vld1q_s8(ak0),    va10=vld1q_s8(ak0+16);
                int8x16_t va20=vld1q_s8(ak0+32), va30=vld1q_s8(ak0+48);
                int8x16_t vb00=vld1q_s8(bk0),    vb10=vld1q_s8(bk0+16);
                int8x16_t vb20=vld1q_s8(bk0+32), vb30=vld1q_s8(bk0+48);
                int8x16_t va01=vld1q_s8(ak1),    va11=vld1q_s8(ak1+16);
                int8x16_t va21=vld1q_s8(ak1+32), va31=vld1q_s8(ak1+48);
                int8x16_t vb01=vld1q_s8(bk1),    vb11=vld1q_s8(bk1+16);
                int8x16_t vb21=vld1q_s8(bk1+32), vb31=vld1q_s8(bk1+48);
                SMMLA16(va00,va10,va20,va30,vb00,vb10,vb20,vb30)
                SMMLA16(va01,va11,va21,va31,vb01,vb11,vb21,vb31)
            }
            for (; k_t8 < K8; ++k_t8) {
                const int8_t* ak = a_pk + k_t8 * 64;
                const int8_t* bk = b_co + k_t8 * 64;
                int8x16_t va0=vld1q_s8(ak),    va1=vld1q_s8(ak+16);
                int8x16_t va2=vld1q_s8(ak+32), va3=vld1q_s8(ak+48);
                int8x16_t vb0=vld1q_s8(bk),    vb1=vld1q_s8(bk+16);
                int8x16_t vb2=vld1q_s8(bk+32), vb3=vld1q_s8(bk+48);
                SMMLA16(va0,va1,va2,va3,vb0,vb1,vb2,vb3)
            }
            #undef SMMLA16

            // ── Output writing ────────────────────────────────────────────────
            // acc{rp}{cp}: {C[2rp,2cp], C[2rp,2cp+1], C[2rp+1,2cp], C[2rp+1,2cp+1]}
            // Even row rp*2: low halves  (elements [0,1]) of acc[rp][*]
            // Odd  row rp*2+1: high halves (elements [2,3]) of acc[rp][*]
            if (!nchw_out && !is_float) {
                // NHWC INT8: 8 channels per row
                int8_t* out_i = static_cast<int8_t*>(C);
                int32x4_t vblo={}, vbhi={};
                for (int i=0;i<4;i++) vblo[i]=n0+i<N?(int32_t)eff_bias[n0+i]:0;
                for (int i=0;i<4;i++) vbhi[i]=n0+4+i<N?(int32_t)eff_bias[n0+4+i]:0;
                int32x4_t vmlo={req_mult[n0],n0+1<N?req_mult[n0+1]:1,n0+2<N?req_mult[n0+2]:1,n0+3<N?req_mult[n0+3]:1};
                int32x4_t vmhi={n0+4<N?req_mult[n0+4]:1,n0+5<N?req_mult[n0+5]:1,n0+6<N?req_mult[n0+6]:1,n0+7<N?req_mult[n0+7]:1};
                int32x4_t velo={req_exp[n0],n0+1<N?req_exp[n0+1]:0,n0+2<N?req_exp[n0+2]:0,n0+3<N?req_exp[n0+3]:0};
                int32x4_t vehi={n0+4<N?req_exp[n0+4]:0,n0+5<N?req_exp[n0+5]:0,n0+6<N?req_exp[n0+6]:0,n0+7<N?req_exp[n0+7]:0};
                int32x4_t vzp=vdupq_n_s32((int32_t)out_zp);
                auto rq4v=[&](int32x4_t v,int32x4_t b,int32x4_t m_,int32x4_t e) __attribute__((always_inline)) {
                    return vaddq_s32(vrshlq_s32(vqrdmulhq_s32(vaddq_s32(v,b),m_),e),vzp); };
                auto wr8=[&](int row,int32x4_t lo,int32x4_t hi) __attribute__((always_inline)) {
                    int8x8_t q8=vqmovn_s16(vcombine_s16(vqmovn_s32(rq4v(lo,vblo,vmlo,velo)),
                                                         vqmovn_s32(rq4v(hi,vbhi,vmhi,vehi))));
                    int8_t* p=out_i+(size_t)(m+row)*N+n0;
                    if (n_valid>=8) vst1_s8(p,q8);
                    else { int64_t t; vst1_lane_s64((int64_t*)&t,vreinterpret_s64_s8(q8),0); memcpy(p,&t,n_valid); }
                };
                wr8(0,vcombine_s32(vget_low_s32(acc00),vget_low_s32(acc01)),vcombine_s32(vget_low_s32(acc02),vget_low_s32(acc03)));
                wr8(1,vcombine_s32(vget_high_s32(acc00),vget_high_s32(acc01)),vcombine_s32(vget_high_s32(acc02),vget_high_s32(acc03)));
                wr8(2,vcombine_s32(vget_low_s32(acc10),vget_low_s32(acc11)),vcombine_s32(vget_low_s32(acc12),vget_low_s32(acc13)));
                wr8(3,vcombine_s32(vget_high_s32(acc10),vget_high_s32(acc11)),vcombine_s32(vget_high_s32(acc12),vget_high_s32(acc13)));
                wr8(4,vcombine_s32(vget_low_s32(acc20),vget_low_s32(acc21)),vcombine_s32(vget_low_s32(acc22),vget_low_s32(acc23)));
                wr8(5,vcombine_s32(vget_high_s32(acc20),vget_high_s32(acc21)),vcombine_s32(vget_high_s32(acc22),vget_high_s32(acc23)));
                wr8(6,vcombine_s32(vget_low_s32(acc30),vget_low_s32(acc31)),vcombine_s32(vget_low_s32(acc32),vget_low_s32(acc33)));
                wr8(7,vcombine_s32(vget_high_s32(acc30),vget_high_s32(acc31)),vcombine_s32(vget_high_s32(acc32),vget_high_s32(acc33)));
            } else if (nchw_out) {
                // NCHW INT8: 8 rows per channel
                // vuzp1q_s32(a,b)={a[0],a[2],b[0],b[2]}: even-row elements; vuzp2q=odd
                // For col cp*2+0: vuzp1q(acc[rp0][cp], acc[rp1][cp]) = rows 2rp0,2rp0+1,2rp1,2rp1+1
                auto wr8nchw=[&](int n,int32x4_t r03,int32x4_t r47) __attribute__((always_inline)) {
                    int32x4_t vb=vdupq_n_s32((int32_t)eff_bias[n]),vm=vdupq_n_s32(req_mult[n]);
                    int32x4_t ve=vdupq_n_s32(req_exp[n]),vz=vdupq_n_s32((int32_t)out_zp);
                    auto rq=[&](int32x4_t v){return vaddq_s32(vrshlq_s32(vqrdmulhq_s32(vaddq_s32(v,vb),vm),ve),vz);};
                    vst1_s8(static_cast<int8_t*>(C)+n*M+m,
                            vqmovn_s16(vcombine_s16(vqmovn_s32(rq(r03)),vqmovn_s32(rq(r47)))));
                };
                if(n_valid>0)wr8nchw(n0+0,vuzp1q_s32(acc00,acc10),vuzp1q_s32(acc20,acc30));
                if(n_valid>1)wr8nchw(n0+1,vuzp2q_s32(acc00,acc10),vuzp2q_s32(acc20,acc30));
                if(n_valid>2)wr8nchw(n0+2,vuzp1q_s32(acc01,acc11),vuzp1q_s32(acc21,acc31));
                if(n_valid>3)wr8nchw(n0+3,vuzp2q_s32(acc01,acc11),vuzp2q_s32(acc21,acc31));
                if(n_valid>4)wr8nchw(n0+4,vuzp1q_s32(acc02,acc12),vuzp1q_s32(acc22,acc32));
                if(n_valid>5)wr8nchw(n0+5,vuzp2q_s32(acc02,acc12),vuzp2q_s32(acc22,acc32));
                if(n_valid>6)wr8nchw(n0+6,vuzp1q_s32(acc03,acc13),vuzp1q_s32(acc23,acc33));
                if(n_valid>7)wr8nchw(n0+7,vuzp2q_s32(acc03,acc13),vuzp2q_s32(acc23,acc33));
            } else {
                // float output (scalar fallback — rare)
                int32_t v[16][4];
                vst1q_s32(v[0],acc00);vst1q_s32(v[1],acc01);vst1q_s32(v[2],acc02);vst1q_s32(v[3],acc03);
                vst1q_s32(v[4],acc10);vst1q_s32(v[5],acc11);vst1q_s32(v[6],acc12);vst1q_s32(v[7],acc13);
                vst1q_s32(v[8],acc20);vst1q_s32(v[9],acc21);vst1q_s32(v[10],acc22);vst1q_s32(v[11],acc23);
                vst1q_s32(v[12],acc30);vst1q_s32(v[13],acc31);vst1q_s32(v[14],acc32);vst1q_s32(v[15],acc33);
                float* out_f=static_cast<float*>(C);
                for(int r=0;r<8;++r){int rp=r/2,e=r%2;
                    for(int c=0;c<n_valid;++c){int cp=c/2,ec=c%2;
                        out_f[(m+r)*N+(n0+c)]=(float)((int64_t)v[rp*4+cp][e*2+ec]+eff_bias[n0+c])*req_scale_f[n0+c];}}
            }
        }   // end co_blk loop
    }   // end mi loop

    // ── Scalar M tail (M%8 rows) using SMMLA B layout ─────────────────────────
    if (n_part || tid == 0) {
        const int co_s2 = n_part ? co_s : 0;
        const int co_e2 = n_part ? co_e : Co8;
        for (int mt = m8_count * 8; mt < M; ++mt) {
            const int8_t* ar = A + (size_t)mt * K;
            for (int co_blk = co_s2; co_blk < co_e2; ++co_blk) {
                const int8_t* b_co = B_packed + (size_t)co_blk * K8 * 64;
                int n0 = co_blk * 8, n_valid = std::min(8, N - n0);
                int32_t acc[8] = {};
                for (int k_t8 = 0; k_t8 < K8; ++k_t8) {
                    const int8_t* bk = b_co + k_t8 * 64;
                    int k_base = k_t8 * 8, ktail = std::min(8, K - k_base);
                    for (int ki = 0; ki < ktail; ++ki) {
                        int av = (int)ar[k_base + ki];
                        for (int cp = 0; cp < 4; ++cp) {
                            if (cp*2   < n_valid) acc[cp*2]   += av * (int)bk[cp*16 + ki];
                            if (cp*2+1 < n_valid) acc[cp*2+1] += av * (int)bk[cp*16 + 8 + ki];
                        }
                    }
                }
                for (int c = 0; c < n_valid; ++c) {
                    int n = n0 + c;
                    int32_t total = (int32_t)((int64_t)acc[c] + eff_bias[n]);
                    if (nchw_out)
                        static_cast<int8_t*>(C)[n*M+mt] = requant_fixedpoint(total,req_mult[n],req_exp[n],out_zp);
                    else if (is_float)
                        static_cast<float*>(C)[mt*N+n] = (float)((int64_t)acc[c]+eff_bias[n])*req_scale_f[n];
                    else
                        static_cast<int8_t*>(C)[mt*N+n] = requant_fixedpoint(total,req_mult[n],req_exp[n],out_zp);
                }
            }
        }
    }
}
#else  // !__ARM_FEATURE_MATMUL_INT8 → SDOT (LD1R) path
    // Hybrid M/N partition — thread-local range computation.
    //  M >= N  →  M-partition: thread tid owns m-tiles [mi_s, mi_e).
    //  M < N   →  N-partition: all threads own co_blks [co_s, co_e).
    const int k_full = K / 4;
    const int m8_count = M / 8;

    // ── Thread-local range computation ────────────────────────────────────────
    // When in_parallel=true: running inside a spawned OMP team → use actual tid/nT.
    // When in_parallel=false but omp_in_parallel(): called from a groups parallel-for
    //   (each thread does a full single-group GEMM) → use tid=0/nT=1 so this thread
    //   processes all M-rows for its assigned group.
#ifdef _OPENMP
    const int tid = in_parallel ? omp_get_thread_num() : 0;
    const int nT  = in_parallel ? omp_get_num_threads() : 1;
#else
    const int tid = 0;
    const int nT  = 1;
#endif
    const bool n_part = (nT > 1) && (M < N);
    // M-partition ranges (used when M >= N)
    const int mi_s = n_part ? 0          : (tid * m8_count) / nT;
    const int mi_e = n_part ? m8_count   : ((tid + 1) * m8_count) / nT;
    // N-partition ranges: split in Co8 units to keep co_s always even,
    // as required by [Co8,K4,32] B-packing (NR=8 loop needs even co_blk).
    const int Co8  = (Co4 + 1) / 2;
    const int co_s = n_part ? ((tid * Co8) / nT) * 2 : 0;
    const int co_e = n_part ? (tid == nT - 1 ? Co4 : (((tid + 1) * Co8) / nT) * 2) : Co4;

    // MR=8 NR=8: process 8 rows × 2 B-tiles (8 cols) per iteration.
    // Sharing av0..av7 across both tiles halves A-loads vs MR=8 NR=4.
    // Exactly 32 SIMD regs: 8 av + 8 bv + 16 acc = 32 (zero spills).
    for (int mi = mi_s; mi < mi_e; ++mi) {
        const int m = mi * 8;
        const int8_t* a0 = A + (size_t)(m+0) * K;
        const int8_t* a1 = A + (size_t)(m+1) * K;
        const int8_t* a2 = A + (size_t)(m+2) * K;
        const int8_t* a3 = A + (size_t)(m+3) * K;
        const int8_t* a4 = A + (size_t)(m+4) * K;
        const int8_t* a5 = A + (size_t)(m+5) * K;
        const int8_t* a6 = A + (size_t)(m+6) * K;
        const int8_t* a7 = A + (size_t)(m+7) * K;

        // NR=8: process 2 consecutive B-tiles at once, sharing av0..av7
        int co_blk = co_s;
        for (; co_blk + 1 < co_e; co_blk += 2) {
            const int n0a = co_blk * 4;
            const int n0b = n0a + 4;
            if (n0b >= N) break;  // second tile fully out of range
            // [Co8,K4,32]: b_co0 at b_tile+k*32, b_co1 at b_tile+k*32+16 (same cache line)
            const int8_t* b_tile = B_packed + (size_t)(co_blk >> 1) * K4 * 32;
            const int n_valid_b = std::min(4, N - n0b);

            int32x4_t acc0a=vdupq_n_s32(0), acc1a=vdupq_n_s32(0);
            int32x4_t acc2a=vdupq_n_s32(0), acc3a=vdupq_n_s32(0);
            int32x4_t acc4a=vdupq_n_s32(0), acc5a=vdupq_n_s32(0);
            int32x4_t acc6a=vdupq_n_s32(0), acc7a=vdupq_n_s32(0);
            int32x4_t acc0b=vdupq_n_s32(0), acc1b=vdupq_n_s32(0);
            int32x4_t acc2b=vdupq_n_s32(0), acc3b=vdupq_n_s32(0);
            int32x4_t acc4b=vdupq_n_s32(0), acc5b=vdupq_n_s32(0);
            int32x4_t acc6b=vdupq_n_s32(0), acc7b=vdupq_n_s32(0);

            // 8av+8bv+16acc=32 regs: zero spills. All loads before SDOT for OOO overlap.
            #define DOT8ALL(b0,b1,b2,b3,b4,b5,b6,b7) \
                acc0a=vdotq_laneq_s32(acc0a,b0,av0,0);acc0b=vdotq_laneq_s32(acc0b,b1,av0,0); \
                acc1a=vdotq_laneq_s32(acc1a,b0,av1,0);acc1b=vdotq_laneq_s32(acc1b,b1,av1,0); \
                acc2a=vdotq_laneq_s32(acc2a,b0,av2,0);acc2b=vdotq_laneq_s32(acc2b,b1,av2,0); \
                acc3a=vdotq_laneq_s32(acc3a,b0,av3,0);acc3b=vdotq_laneq_s32(acc3b,b1,av3,0); \
                acc4a=vdotq_laneq_s32(acc4a,b0,av4,0);acc4b=vdotq_laneq_s32(acc4b,b1,av4,0); \
                acc5a=vdotq_laneq_s32(acc5a,b0,av5,0);acc5b=vdotq_laneq_s32(acc5b,b1,av5,0); \
                acc6a=vdotq_laneq_s32(acc6a,b0,av6,0);acc6b=vdotq_laneq_s32(acc6b,b1,av6,0); \
                acc7a=vdotq_laneq_s32(acc7a,b0,av7,0);acc7b=vdotq_laneq_s32(acc7b,b1,av7,0); \
                acc0a=vdotq_laneq_s32(acc0a,b2,av0,1);acc0b=vdotq_laneq_s32(acc0b,b3,av0,1); \
                acc1a=vdotq_laneq_s32(acc1a,b2,av1,1);acc1b=vdotq_laneq_s32(acc1b,b3,av1,1); \
                acc2a=vdotq_laneq_s32(acc2a,b2,av2,1);acc2b=vdotq_laneq_s32(acc2b,b3,av2,1); \
                acc3a=vdotq_laneq_s32(acc3a,b2,av3,1);acc3b=vdotq_laneq_s32(acc3b,b3,av3,1); \
                acc4a=vdotq_laneq_s32(acc4a,b2,av4,1);acc4b=vdotq_laneq_s32(acc4b,b3,av4,1); \
                acc5a=vdotq_laneq_s32(acc5a,b2,av5,1);acc5b=vdotq_laneq_s32(acc5b,b3,av5,1); \
                acc6a=vdotq_laneq_s32(acc6a,b2,av6,1);acc6b=vdotq_laneq_s32(acc6b,b3,av6,1); \
                acc7a=vdotq_laneq_s32(acc7a,b2,av7,1);acc7b=vdotq_laneq_s32(acc7b,b3,av7,1); \
                acc0a=vdotq_laneq_s32(acc0a,b4,av0,2);acc0b=vdotq_laneq_s32(acc0b,b5,av0,2); \
                acc1a=vdotq_laneq_s32(acc1a,b4,av1,2);acc1b=vdotq_laneq_s32(acc1b,b5,av1,2); \
                acc2a=vdotq_laneq_s32(acc2a,b4,av2,2);acc2b=vdotq_laneq_s32(acc2b,b5,av2,2); \
                acc3a=vdotq_laneq_s32(acc3a,b4,av3,2);acc3b=vdotq_laneq_s32(acc3b,b5,av3,2); \
                acc4a=vdotq_laneq_s32(acc4a,b4,av4,2);acc4b=vdotq_laneq_s32(acc4b,b5,av4,2); \
                acc5a=vdotq_laneq_s32(acc5a,b4,av5,2);acc5b=vdotq_laneq_s32(acc5b,b5,av5,2); \
                acc6a=vdotq_laneq_s32(acc6a,b4,av6,2);acc6b=vdotq_laneq_s32(acc6b,b5,av6,2); \
                acc7a=vdotq_laneq_s32(acc7a,b4,av7,2);acc7b=vdotq_laneq_s32(acc7b,b5,av7,2); \
                acc0a=vdotq_laneq_s32(acc0a,b6,av0,3);acc0b=vdotq_laneq_s32(acc0b,b7,av0,3); \
                acc1a=vdotq_laneq_s32(acc1a,b6,av1,3);acc1b=vdotq_laneq_s32(acc1b,b7,av1,3); \
                acc2a=vdotq_laneq_s32(acc2a,b6,av2,3);acc2b=vdotq_laneq_s32(acc2b,b7,av2,3); \
                acc3a=vdotq_laneq_s32(acc3a,b6,av3,3);acc3b=vdotq_laneq_s32(acc3b,b7,av3,3); \
                acc4a=vdotq_laneq_s32(acc4a,b6,av4,3);acc4b=vdotq_laneq_s32(acc4b,b7,av4,3); \
                acc5a=vdotq_laneq_s32(acc5a,b6,av5,3);acc5b=vdotq_laneq_s32(acc5b,b7,av5,3); \
                acc6a=vdotq_laneq_s32(acc6a,b6,av6,3);acc6b=vdotq_laneq_s32(acc6b,b7,av6,3); \
                acc7a=vdotq_laneq_s32(acc7a,b6,av7,3);acc7b=vdotq_laneq_s32(acc7b,b7,av7,3)
            int k_blk = 0;
            for (; k_blk + 3 < k_full; k_blk += 4) {
                const int off = k_blk * 4;
                const int8_t* bp = b_tile + k_blk * 32;
                int8x16_t av0=vld1q_s8(a0+off), av1=vld1q_s8(a1+off);
                int8x16_t av2=vld1q_s8(a2+off), av3=vld1q_s8(a3+off);
                int8x16_t av4=vld1q_s8(a4+off), av5=vld1q_s8(a5+off);
                int8x16_t av6=vld1q_s8(a6+off), av7=vld1q_s8(a7+off);
                int8x16_t bv0=vld1q_s8(bp),    bv1=vld1q_s8(bp+16);
                int8x16_t bv2=vld1q_s8(bp+32), bv3=vld1q_s8(bp+48);
                int8x16_t bv4=vld1q_s8(bp+64), bv5=vld1q_s8(bp+80);
                int8x16_t bv6=vld1q_s8(bp+96), bv7=vld1q_s8(bp+112);
                __builtin_prefetch(bp + 8*32, 0, 3);
                DOT8ALL(bv0,bv1,bv2,bv3,bv4,bv5,bv6,bv7);
            }
            #undef DOT8ALL
            for (; k_blk < k_full; ++k_blk) {
                int8x16_t wva = vld1q_s8(b_tile + k_blk * 32);
                int8x16_t wvb = vld1q_s8(b_tile + k_blk * 32 + 16);
                int32_t i0,i1,i2,i3,i4,i5,i6,i7;
                memcpy(&i0,a0+k_blk*4,4); memcpy(&i1,a1+k_blk*4,4);
                memcpy(&i2,a2+k_blk*4,4); memcpy(&i3,a3+k_blk*4,4);
                memcpy(&i4,a4+k_blk*4,4); memcpy(&i5,a5+k_blk*4,4);
                memcpy(&i6,a6+k_blk*4,4); memcpy(&i7,a7+k_blk*4,4);
                #define S2(acca,accb,iv) \
                    acca=vdotq_s32(acca,vreinterpretq_s8_s32(vdupq_n_s32(iv)),wva); \
                    accb=vdotq_s32(accb,vreinterpretq_s8_s32(vdupq_n_s32(iv)),wvb)
                S2(acc0a,acc0b,i0); S2(acc1a,acc1b,i1); S2(acc2a,acc2b,i2); S2(acc3a,acc3b,i3);
                S2(acc4a,acc4b,i4); S2(acc5a,acc5b,i5); S2(acc6a,acc6b,i6); S2(acc7a,acc7b,i7);
                #undef S2
            }
            if (K & 3) {
                int k_base = k_full * 4;
                uint8_t t0[4]={},t1[4]={},t2[4]={},t3[4]={},t4[4]={},t5[4]={},t6[4]={},t7[4]={};
                for (int i = 0; i < (K & 3); ++i) {
                    t0[i]=(uint8_t)a0[k_base+i]; t1[i]=(uint8_t)a1[k_base+i];
                    t2[i]=(uint8_t)a2[k_base+i]; t3[i]=(uint8_t)a3[k_base+i];
                    t4[i]=(uint8_t)a4[k_base+i]; t5[i]=(uint8_t)a5[k_base+i];
                    t6[i]=(uint8_t)a6[k_base+i]; t7[i]=(uint8_t)a7[k_base+i];
                }
                int32_t i0,i1,i2,i3,i4,i5,i6,i7;
                memcpy(&i0,t0,4); memcpy(&i1,t1,4); memcpy(&i2,t2,4); memcpy(&i3,t3,4);
                memcpy(&i4,t4,4); memcpy(&i5,t5,4); memcpy(&i6,t6,4); memcpy(&i7,t7,4);
                int8x16_t wva = vld1q_s8(b_tile + k_full * 32);
                int8x16_t wvb = vld1q_s8(b_tile + k_full * 32 + 16);
                #define S2(acca,accb,iv) \
                    acca=vdotq_s32(acca,vreinterpretq_s8_s32(vdupq_n_s32(iv)),wva); \
                    accb=vdotq_s32(accb,vreinterpretq_s8_s32(vdupq_n_s32(iv)),wvb)
                S2(acc0a,acc0b,i0); S2(acc1a,acc1b,i1); S2(acc2a,acc2b,i2); S2(acc3a,acc3b,i3);
                S2(acc4a,acc4b,i4); S2(acc5a,acc5b,i5); S2(acc6a,acc6b,i6); S2(acc7a,acc7b,i7);
                #undef S2
            }

            // Write NR=8 output (both tiles a and b)
            if (nchw_out) {
                // Transpose [8rows×4chans] → channel-major for each tile
                auto trnq8 = [](int32x4_t r0, int32x4_t r1, int32x4_t r2, int32x4_t r3,
                                int32x4_t r4, int32x4_t r5, int32x4_t r6, int32x4_t r7,
                                int32x4_t& c0, int32x4_t& c1, int32x4_t& c2, int32x4_t& c3,
                                int32x4_t& c4, int32x4_t& c5, int32x4_t& c6, int32x4_t& c7) {
                    int32x4x2_t tr0=vtrnq_s32(r0,r1), tr1=vtrnq_s32(r2,r3);
                    int32x4x2_t tr2=vtrnq_s32(r4,r5), tr3=vtrnq_s32(r6,r7);
                    c0=vcombine_s32(vget_low_s32(tr0.val[0]),vget_low_s32(tr1.val[0]));
                    c1=vcombine_s32(vget_low_s32(tr0.val[1]),vget_low_s32(tr1.val[1]));
                    c2=vcombine_s32(vget_high_s32(tr0.val[0]),vget_high_s32(tr1.val[0]));
                    c3=vcombine_s32(vget_high_s32(tr0.val[1]),vget_high_s32(tr1.val[1]));
                    c4=vcombine_s32(vget_low_s32(tr2.val[0]),vget_low_s32(tr3.val[0]));
                    c5=vcombine_s32(vget_low_s32(tr2.val[1]),vget_low_s32(tr3.val[1]));
                    c6=vcombine_s32(vget_high_s32(tr2.val[0]),vget_high_s32(tr3.val[0]));
                    c7=vcombine_s32(vget_high_s32(tr2.val[1]),vget_high_s32(tr3.val[1]));
                };
                auto write8nc = [&](int n_, int32x4_t ga, int32x4_t gb) {
                    int32x4_t vm=vdupq_n_s32(req_mult[n_]),  ve=vdupq_n_s32(req_exp[n_]);
                    int32x4_t vb=vdupq_n_s32((int32_t)eff_bias[n_]), vz=vdupq_n_s32((int32_t)out_zp);
                    auto rq4=[&](int32x4_t v){return vaddq_s32(vrshlq_s32(vqrdmulhq_s32(vaddq_s32(v,vb),vm),ve),vz);};
                    vst1_s8(static_cast<int8_t*>(C)+n_*M+m,
                            vqmovn_s16(vcombine_s16(vqmovn_s32(rq4(ga)),vqmovn_s32(rq4(gb)))));
                };
                int32x4_t g0a,g1a,g2a,g3a,g4a,g5a,g6a,g7a;
                trnq8(acc0a,acc1a,acc2a,acc3a,acc4a,acc5a,acc6a,acc7a,
                      g0a,g1a,g2a,g3a,g4a,g5a,g6a,g7a);
                write8nc(n0a+0,g0a,g4a); write8nc(n0a+1,g1a,g5a);
                write8nc(n0a+2,g2a,g6a); write8nc(n0a+3,g3a,g7a);
                int32x4_t g0b,g1b,g2b,g3b,g4b,g5b,g6b,g7b;
                trnq8(acc0b,acc1b,acc2b,acc3b,acc4b,acc5b,acc6b,acc7b,
                      g0b,g1b,g2b,g3b,g4b,g5b,g6b,g7b);
                if (n_valid_b > 0) write8nc(n0b+0,g0b,g4b);
                if (n_valid_b > 1) write8nc(n0b+1,g1b,g5b);
                if (n_valid_b > 2) write8nc(n0b+2,g2b,g6b);
                if (n_valid_b > 3) write8nc(n0b+3,g3b,g7b);
            } else if (!is_float) {
                int8_t* out_i = static_cast<int8_t*>(C);
                // n_valid_a == 4 always (first tile never at N-boundary in NR=8 loop)
                int32x4_t vba = {(int32_t)eff_bias[n0a],(int32_t)eff_bias[n0a+1],(int32_t)eff_bias[n0a+2],(int32_t)eff_bias[n0a+3]};
                int32x4_t vma = {req_mult[n0a],req_mult[n0a+1],req_mult[n0a+2],req_mult[n0a+3]};
                int32x4_t vea = {req_exp[n0a],req_exp[n0a+1],req_exp[n0a+2],req_exp[n0a+3]};
                int32x4_t vbb = {(int32_t)eff_bias[n0b], n0b+1<N?(int32_t)eff_bias[n0b+1]:0, n0b+2<N?(int32_t)eff_bias[n0b+2]:0, n0b+3<N?(int32_t)eff_bias[n0b+3]:0};
                int32x4_t vmb = {req_mult[n0b], n0b+1<N?req_mult[n0b+1]:0, n0b+2<N?req_mult[n0b+2]:0, n0b+3<N?req_mult[n0b+3]:0};
                int32x4_t veb = {req_exp[n0b], n0b+1<N?req_exp[n0b+1]:0, n0b+2<N?req_exp[n0b+2]:0, n0b+3<N?req_exp[n0b+3]:0};
                int32x4_t vzp8 = vdupq_n_s32((int32_t)out_zp);
                auto rq4 = [&](int32x4_t v, int32x4_t vb_, int32x4_t vm, int32x4_t ve) __attribute__((always_inline)) {
                    return vaddq_s32(vrshlq_s32(vqrdmulhq_s32(vaddq_s32(v,vb_),vm),ve),vzp8);
                };
                if (n_valid_b == 4) {
                    // Fast path: combined 8-byte vst1_s8 per row (tile a + tile b)
                    auto wr8x2 = [&](int row, int32x4_t va, int32x4_t vb_) __attribute__((always_inline)) {
                        vst1_s8(out_i+(size_t)(m+row)*N+n0a,
                                vqmovn_s16(vcombine_s16(vqmovn_s32(rq4(va,vba,vma,vea)),
                                                        vqmovn_s32(rq4(vb_,vbb,vmb,veb)))));
                    };
                    wr8x2(0,acc0a,acc0b); wr8x2(1,acc1a,acc1b);
                    wr8x2(2,acc2a,acc2b); wr8x2(3,acc3a,acc3b);
                    wr8x2(4,acc4a,acc4b); wr8x2(5,acc5a,acc5b);
                    wr8x2(6,acc6a,acc6b); wr8x2(7,acc7a,acc7b);
                } else {
                    // Tile a: full 4-byte store
                    auto wr4a = [&](int row, int32x4_t va) __attribute__((always_inline)) {
                        int32x4_t q = rq4(va,vba,vma,vea);
                        int8x8_t b8v = vqmovn_s16(vcombine_s16(vqmovn_s32(q),vqmovn_s32(q)));
                        vst1_lane_s32((int32_t*)(out_i+(size_t)(m+row)*N+n0a),vreinterpret_s32_s8(b8v),0);
                    };
                    wr4a(0,acc0a); wr4a(1,acc1a); wr4a(2,acc2a); wr4a(3,acc3a);
                    wr4a(4,acc4a); wr4a(5,acc5a); wr4a(6,acc6a); wr4a(7,acc7a);
                    // Tile b: partial store
                    auto wr4b_p = [&](int row, int32x4_t vb_) {
                        int32x4_t q = rq4(vb_,vbb,vmb,veb);
                        int8x8_t b8v = vqmovn_s16(vcombine_s16(vqmovn_s32(q),vqmovn_s32(q)));
                        int32_t tmp; vst1_lane_s32(&tmp,vreinterpret_s32_s8(b8v),0);
                        memcpy(out_i+(size_t)(m+row)*N+n0b,&tmp,n_valid_b);
                    };
                    wr4b_p(0,acc0b); wr4b_p(1,acc1b); wr4b_p(2,acc2b); wr4b_p(3,acc3b);
                    wr4b_p(4,acc4b); wr4b_p(5,acc5b); wr4b_p(6,acc6b); wr4b_p(7,acc7b);
                }
            } else {
                float* out_f = static_cast<float*>(C);
                int32_t ra[8][4], rb[8][4];
                vst1q_s32(ra[0],acc0a); vst1q_s32(ra[1],acc1a);
                vst1q_s32(ra[2],acc2a); vst1q_s32(ra[3],acc3a);
                vst1q_s32(ra[4],acc4a); vst1q_s32(ra[5],acc5a);
                vst1q_s32(ra[6],acc6a); vst1q_s32(ra[7],acc7a);
                vst1q_s32(rb[0],acc0b); vst1q_s32(rb[1],acc1b);
                vst1q_s32(rb[2],acc2b); vst1q_s32(rb[3],acc3b);
                vst1q_s32(rb[4],acc4b); vst1q_s32(rb[5],acc5b);
                vst1q_s32(rb[6],acc6b); vst1q_s32(rb[7],acc7b);
                for (int oc = 0; oc < 4; ++oc) {
                    const int na = n0a+oc; const int64_t ba=eff_bias[na]; const float sa=req_scale_f[na];
                    for (int row=0;row<8;++row) out_f[(m+row)*N+na]=(float)((int64_t)ra[row][oc]+ba)*sa;
                }
                for (int oc = 0; oc < n_valid_b; ++oc) {
                    const int nb = n0b+oc; const int64_t bb=eff_bias[nb]; const float sb=req_scale_f[nb];
                    for (int row=0;row<8;++row) out_f[(m+row)*N+nb]=(float)((int64_t)rb[row][oc]+bb)*sb;
                }
            }
        }   // end NR=8 co_blk loop

        // NR=4 fallback for last tile (handles odd Co4 or single-tile N-partition boundary)
        for (; co_blk < co_e; ++co_blk) {
            // [Co8,K4,32]: even co_blk uses bytes 0-15, odd uses bytes 16-31 per k-group
            const int8_t* b_co = B_packed + (size_t)(co_blk >> 1) * K4 * 32 + (co_blk & 1) * 16;
            const int n0      = co_blk * 4;
            const int n_valid = std::min(4, N - n0);

            int32x4_t acc0=vdupq_n_s32(0), acc1=vdupq_n_s32(0);
            int32x4_t acc2=vdupq_n_s32(0), acc3=vdupq_n_s32(0);
            int32x4_t acc4=vdupq_n_s32(0), acc5=vdupq_n_s32(0);
            int32x4_t acc6=vdupq_n_s32(0), acc7=vdupq_n_s32(0);

            int k_blk = 0;
            for (; k_blk + 3 < k_full; k_blk += 4) {
                const int off = k_blk * 4;
                const int8_t* bp = b_co + k_blk * 32;
                int8x16_t av0=vld1q_s8(a0+off), av1=vld1q_s8(a1+off);
                int8x16_t av2=vld1q_s8(a2+off), av3=vld1q_s8(a3+off);
                int8x16_t av4=vld1q_s8(a4+off), av5=vld1q_s8(a5+off);
                int8x16_t av6=vld1q_s8(a6+off), av7=vld1q_s8(a7+off);
                int8x16_t bv0=vld1q_s8(bp), bv1=vld1q_s8(bp+32);
                int8x16_t bv2=vld1q_s8(bp+64), bv3=vld1q_s8(bp+96);
                acc0=vdotq_laneq_s32(acc0,bv0,av0,0); acc1=vdotq_laneq_s32(acc1,bv0,av1,0);
                acc2=vdotq_laneq_s32(acc2,bv0,av2,0); acc3=vdotq_laneq_s32(acc3,bv0,av3,0);
                acc4=vdotq_laneq_s32(acc4,bv0,av4,0); acc5=vdotq_laneq_s32(acc5,bv0,av5,0);
                acc6=vdotq_laneq_s32(acc6,bv0,av6,0); acc7=vdotq_laneq_s32(acc7,bv0,av7,0);
                acc0=vdotq_laneq_s32(acc0,bv1,av0,1); acc1=vdotq_laneq_s32(acc1,bv1,av1,1);
                acc2=vdotq_laneq_s32(acc2,bv1,av2,1); acc3=vdotq_laneq_s32(acc3,bv1,av3,1);
                acc4=vdotq_laneq_s32(acc4,bv1,av4,1); acc5=vdotq_laneq_s32(acc5,bv1,av5,1);
                acc6=vdotq_laneq_s32(acc6,bv1,av6,1); acc7=vdotq_laneq_s32(acc7,bv1,av7,1);
                acc0=vdotq_laneq_s32(acc0,bv2,av0,2); acc1=vdotq_laneq_s32(acc1,bv2,av1,2);
                acc2=vdotq_laneq_s32(acc2,bv2,av2,2); acc3=vdotq_laneq_s32(acc3,bv2,av3,2);
                acc4=vdotq_laneq_s32(acc4,bv2,av4,2); acc5=vdotq_laneq_s32(acc5,bv2,av5,2);
                acc6=vdotq_laneq_s32(acc6,bv2,av6,2); acc7=vdotq_laneq_s32(acc7,bv2,av7,2);
                acc0=vdotq_laneq_s32(acc0,bv3,av0,3); acc1=vdotq_laneq_s32(acc1,bv3,av1,3);
                acc2=vdotq_laneq_s32(acc2,bv3,av2,3); acc3=vdotq_laneq_s32(acc3,bv3,av3,3);
                acc4=vdotq_laneq_s32(acc4,bv3,av4,3); acc5=vdotq_laneq_s32(acc5,bv3,av5,3);
                acc6=vdotq_laneq_s32(acc6,bv3,av6,3); acc7=vdotq_laneq_s32(acc7,bv3,av7,3);
            }
            for (; k_blk < k_full; ++k_blk) {
                int8x16_t wv = vld1q_s8(b_co + k_blk * 32);
                int32_t i0, i1, i2, i3, i4, i5, i6, i7;
                memcpy(&i0, a0+k_blk*4, 4); memcpy(&i1, a1+k_blk*4, 4);
                memcpy(&i2, a2+k_blk*4, 4); memcpy(&i3, a3+k_blk*4, 4);
                memcpy(&i4, a4+k_blk*4, 4); memcpy(&i5, a5+k_blk*4, 4);
                memcpy(&i6, a6+k_blk*4, 4); memcpy(&i7, a7+k_blk*4, 4);
                acc0 = vdotq_s32(acc0, vreinterpretq_s8_s32(vdupq_n_s32(i0)), wv);
                acc1 = vdotq_s32(acc1, vreinterpretq_s8_s32(vdupq_n_s32(i1)), wv);
                acc2 = vdotq_s32(acc2, vreinterpretq_s8_s32(vdupq_n_s32(i2)), wv);
                acc3 = vdotq_s32(acc3, vreinterpretq_s8_s32(vdupq_n_s32(i3)), wv);
                acc4 = vdotq_s32(acc4, vreinterpretq_s8_s32(vdupq_n_s32(i4)), wv);
                acc5 = vdotq_s32(acc5, vreinterpretq_s8_s32(vdupq_n_s32(i5)), wv);
                acc6 = vdotq_s32(acc6, vreinterpretq_s8_s32(vdupq_n_s32(i6)), wv);
                acc7 = vdotq_s32(acc7, vreinterpretq_s8_s32(vdupq_n_s32(i7)), wv);
            }
            if (K & 3) {
                int k_base = k_full * 4;
                uint8_t t0[4]={},t1[4]={},t2[4]={},t3[4]={},t4[4]={},t5[4]={},t6[4]={},t7[4]={};
                for (int i = 0; i < (K & 3); ++i) {
                    t0[i]=(uint8_t)a0[k_base+i]; t1[i]=(uint8_t)a1[k_base+i];
                    t2[i]=(uint8_t)a2[k_base+i]; t3[i]=(uint8_t)a3[k_base+i];
                    t4[i]=(uint8_t)a4[k_base+i]; t5[i]=(uint8_t)a5[k_base+i];
                    t6[i]=(uint8_t)a6[k_base+i]; t7[i]=(uint8_t)a7[k_base+i];
                }
                int32_t i0,i1,i2,i3,i4,i5,i6,i7;
                memcpy(&i0,t0,4); memcpy(&i1,t1,4); memcpy(&i2,t2,4); memcpy(&i3,t3,4);
                memcpy(&i4,t4,4); memcpy(&i5,t5,4); memcpy(&i6,t6,4); memcpy(&i7,t7,4);
                int8x16_t wv = vld1q_s8(b_co + k_full * 32);
                acc0=vdotq_s32(acc0,vreinterpretq_s8_s32(vdupq_n_s32(i0)),wv);
                acc1=vdotq_s32(acc1,vreinterpretq_s8_s32(vdupq_n_s32(i1)),wv);
                acc2=vdotq_s32(acc2,vreinterpretq_s8_s32(vdupq_n_s32(i2)),wv);
                acc3=vdotq_s32(acc3,vreinterpretq_s8_s32(vdupq_n_s32(i3)),wv);
                acc4=vdotq_s32(acc4,vreinterpretq_s8_s32(vdupq_n_s32(i4)),wv);
                acc5=vdotq_s32(acc5,vreinterpretq_s8_s32(vdupq_n_s32(i5)),wv);
                acc6=vdotq_s32(acc6,vreinterpretq_s8_s32(vdupq_n_s32(i6)),wv);
                acc7=vdotq_s32(acc7,vreinterpretq_s8_s32(vdupq_n_s32(i7)),wv);
            }

            if (nchw_out) {
                int32x4x2_t tr0=vtrnq_s32(acc0,acc1), tr1=vtrnq_s32(acc2,acc3);
                int32x4x2_t tr2=vtrnq_s32(acc4,acc5), tr3=vtrnq_s32(acc6,acc7);
                int32x4_t g0c0=vcombine_s32(vget_low_s32(tr0.val[0]),vget_low_s32(tr1.val[0]));
                int32x4_t g0c1=vcombine_s32(vget_low_s32(tr0.val[1]),vget_low_s32(tr1.val[1]));
                int32x4_t g0c2=vcombine_s32(vget_high_s32(tr0.val[0]),vget_high_s32(tr1.val[0]));
                int32x4_t g0c3=vcombine_s32(vget_high_s32(tr0.val[1]),vget_high_s32(tr1.val[1]));
                int32x4_t g1c0=vcombine_s32(vget_low_s32(tr2.val[0]),vget_low_s32(tr3.val[0]));
                int32x4_t g1c1=vcombine_s32(vget_low_s32(tr2.val[1]),vget_low_s32(tr3.val[1]));
                int32x4_t g1c2=vcombine_s32(vget_high_s32(tr2.val[0]),vget_high_s32(tr3.val[0]));
                int32x4_t g1c3=vcombine_s32(vget_high_s32(tr2.val[1]),vget_high_s32(tr3.val[1]));
                auto write8 = [&](int oc, int32x4_t a, int32x4_t b) {
                    const int n = n0 + oc;
                    const int32x4_t vmult = vdupq_n_s32(req_mult[n]);
                    const int32x4_t vexp  = vdupq_n_s32(req_exp[n]);
                    const int32x4_t vbias = vdupq_n_s32((int32_t)eff_bias[n]);
                    const int32x4_t vzp   = vdupq_n_s32((int32_t)out_zp);
                    auto rq4 = [&](int32x4_t v) {
                        int32x4_t s = vaddq_s32(v, vbias);
                        int32x4_t mm = vqrdmulhq_s32(s, vmult);
                        return vaddq_s32(vrshlq_s32(mm, vexp), vzp);
                    };
                    vst1_s8(static_cast<int8_t*>(C)+n*M+m,
                            vqmovn_s16(vcombine_s16(vqmovn_s32(rq4(a)),vqmovn_s32(rq4(b)))));
                };
                if (n_valid > 0) write8(0, g0c0, g1c0);
                if (n_valid > 1) write8(1, g0c1, g1c1);
                if (n_valid > 2) write8(2, g0c2, g1c2);
                if (n_valid > 3) write8(3, g0c3, g1c3);
            } else if (!is_float) {
                int8_t* out_i = static_cast<int8_t*>(C);
                int32x4_t vbias8 = {(int32_t)eff_bias[n0], n0+1<N?(int32_t)eff_bias[n0+1]:0, n0+2<N?(int32_t)eff_bias[n0+2]:0, n0+3<N?(int32_t)eff_bias[n0+3]:0};
                int32x4_t vmult8 = {req_mult[n0], n0+1<N?req_mult[n0+1]:0, n0+2<N?req_mult[n0+2]:0, n0+3<N?req_mult[n0+3]:0};
                int32x4_t vexp8 = {req_exp[n0], n0+1<N?req_exp[n0+1]:0, n0+2<N?req_exp[n0+2]:0, n0+3<N?req_exp[n0+3]:0};
                int32x4_t vzp8 = vdupq_n_s32((int32_t)out_zp);
                if (n_valid == 4) {
                    auto wr8f = [&](int row, int32x4_t v) __attribute__((always_inline)) {
                        int32x4_t s = vaddq_s32(v, vbias8);
                        int32x4_t mq = vqrdmulhq_s32(s, vmult8);
                        int32x4_t q = vaddq_s32(vrshlq_s32(mq, vexp8), vzp8);
                        int16x4_t h = vqmovn_s32(q);
                        int8x8_t b8v = vqmovn_s16(vcombine_s16(h, h));
                        vst1_lane_s32((int32_t*)(out_i + (size_t)(m+row)*N + n0),
                                      vreinterpret_s32_s8(b8v), 0);
                    };
                    wr8f(0,acc0); wr8f(1,acc1); wr8f(2,acc2); wr8f(3,acc3);
                    wr8f(4,acc4); wr8f(5,acc5); wr8f(6,acc6); wr8f(7,acc7);
                } else {
                    auto wr8 = [&](int row, int32x4_t v) {
                        int32x4_t s = vaddq_s32(v, vbias8);
                        int32x4_t mq = vqrdmulhq_s32(s, vmult8);
                        int32x4_t q = vaddq_s32(vrshlq_s32(mq, vexp8), vzp8);
                        int16x4_t h = vqmovn_s32(q);
                        int8x8_t b8v = vqmovn_s16(vcombine_s16(h, h));
                        int32_t tmp; vst1_lane_s32(&tmp, vreinterpret_s32_s8(b8v), 0);
                        memcpy(out_i + (size_t)(m+row)*N + n0, &tmp, n_valid);
                    };
                    wr8(0,acc0); wr8(1,acc1); wr8(2,acc2); wr8(3,acc3);
                    wr8(4,acc4); wr8(5,acc5); wr8(6,acc6); wr8(7,acc7);
                }
            } else {
                int32_t r0[4],r1[4],r2[4],r3[4],r4[4],r5[4],r6[4],r7[4];
                vst1q_s32(r0,acc0); vst1q_s32(r1,acc1); vst1q_s32(r2,acc2); vst1q_s32(r3,acc3);
                vst1q_s32(r4,acc4); vst1q_s32(r5,acc5); vst1q_s32(r6,acc6); vst1q_s32(r7,acc7);
                for (int oc = 0; oc < n_valid; ++oc) {
                    const int n   = n0 + oc;
                    const int64_t b64 = eff_bias[n];
                    float* out_f = static_cast<float*>(C);
                    const float s = req_scale_f[n];
                    out_f[(m+0)*N+n]=(float)((int64_t)r0[oc]+b64)*s; out_f[(m+1)*N+n]=(float)((int64_t)r1[oc]+b64)*s;
                    out_f[(m+2)*N+n]=(float)((int64_t)r2[oc]+b64)*s; out_f[(m+3)*N+n]=(float)((int64_t)r3[oc]+b64)*s;
                    out_f[(m+4)*N+n]=(float)((int64_t)r4[oc]+b64)*s; out_f[(m+5)*N+n]=(float)((int64_t)r5[oc]+b64)*s;
                    out_f[(m+6)*N+n]=(float)((int64_t)r6[oc]+b64)*s; out_f[(m+7)*N+n]=(float)((int64_t)r7[oc]+b64)*s;
                }
            }
        }   // end NR=4 fallback loop
    }   // end 8-row mi loop

    // ── 8/4/scalar row tail loops (remainder < 8 rows) ─────────────────────────
    // M-partition: only thread 0 runs the tail (≤7 rows, cost is negligible).
    // N-partition: all threads run the tail with their co_blk range.
    if (n_part || tid == 0) {
    const int co_s2 = n_part ? co_s : 0;
    const int co_e2 = n_part ? co_e : Co4;
    int m = m8_count * 8;

    for (; m + 3 < M; m += 4) {
        const int8_t* a0 = A + (size_t)(m+0) * K;
        const int8_t* a1 = A + (size_t)(m+1) * K;
        const int8_t* a2 = A + (size_t)(m+2) * K;
        const int8_t* a3 = A + (size_t)(m+3) * K;

        for (int co_blk = co_s2; co_blk < co_e2; ++co_blk) {
            const int8_t* b_co = B_packed + (size_t)(co_blk >> 1) * K4 * 32 + (co_blk & 1) * 16;
            const int n0      = co_blk * 4;
            const int n_valid = std::min(4, N - n0);

            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);

            int k_blk = 0;
            for (; k_blk + 3 < k_full; k_blk += 4) {
                int8x16_t av0 = vld1q_s8(a0 + k_blk * 4);
                int8x16_t av1 = vld1q_s8(a1 + k_blk * 4);
                int8x16_t av2 = vld1q_s8(a2 + k_blk * 4);
                int8x16_t av3 = vld1q_s8(a3 + k_blk * 4);
                int8x16_t bv0 = vld1q_s8(b_co + (k_blk+0) * 32);
                int8x16_t bv1 = vld1q_s8(b_co + (k_blk+1) * 32);
                int8x16_t bv2 = vld1q_s8(b_co + (k_blk+2) * 32);
                int8x16_t bv3 = vld1q_s8(b_co + (k_blk+3) * 32);
                acc0 = vdotq_laneq_s32(acc0, bv0, av0, 0); acc0 = vdotq_laneq_s32(acc0, bv1, av0, 1);
                acc0 = vdotq_laneq_s32(acc0, bv2, av0, 2); acc0 = vdotq_laneq_s32(acc0, bv3, av0, 3);
                acc1 = vdotq_laneq_s32(acc1, bv0, av1, 0); acc1 = vdotq_laneq_s32(acc1, bv1, av1, 1);
                acc1 = vdotq_laneq_s32(acc1, bv2, av1, 2); acc1 = vdotq_laneq_s32(acc1, bv3, av1, 3);
                acc2 = vdotq_laneq_s32(acc2, bv0, av2, 0); acc2 = vdotq_laneq_s32(acc2, bv1, av2, 1);
                acc2 = vdotq_laneq_s32(acc2, bv2, av2, 2); acc2 = vdotq_laneq_s32(acc2, bv3, av2, 3);
                acc3 = vdotq_laneq_s32(acc3, bv0, av3, 0); acc3 = vdotq_laneq_s32(acc3, bv1, av3, 1);
                acc3 = vdotq_laneq_s32(acc3, bv2, av3, 2); acc3 = vdotq_laneq_s32(acc3, bv3, av3, 3);
            }
            for (; k_blk < k_full; ++k_blk) {
                int8x16_t wv = vld1q_s8(b_co + k_blk * 32);
                int32_t in0, in1, in2, in3;
                memcpy(&in0, a0+k_blk*4, 4); memcpy(&in1, a1+k_blk*4, 4);
                memcpy(&in2, a2+k_blk*4, 4); memcpy(&in3, a3+k_blk*4, 4);
                acc0 = vdotq_s32(acc0, vreinterpretq_s8_s32(vdupq_n_s32(in0)), wv);
                acc1 = vdotq_s32(acc1, vreinterpretq_s8_s32(vdupq_n_s32(in1)), wv);
                acc2 = vdotq_s32(acc2, vreinterpretq_s8_s32(vdupq_n_s32(in2)), wv);
                acc3 = vdotq_s32(acc3, vreinterpretq_s8_s32(vdupq_n_s32(in3)), wv);
            }
            if (K & 3) {
                int k_base = k_full * 4;
                uint8_t t0[4]={}, t1[4]={}, t2[4]={}, t3[4]={};
                for (int i = 0; i < (K & 3); ++i) {
                    t0[i] = (uint8_t)a0[k_base+i];
                    t1[i] = (uint8_t)a1[k_base+i];
                    t2[i] = (uint8_t)a2[k_base+i];
                    t3[i] = (uint8_t)a3[k_base+i];
                }
                int32_t in0, in1, in2, in3;
                memcpy(&in0, t0, 4); memcpy(&in1, t1, 4);
                memcpy(&in2, t2, 4); memcpy(&in3, t3, 4);
                int8x16_t wv = vld1q_s8(b_co + k_full * 32);
                acc0 = vdotq_s32(acc0, vreinterpretq_s8_s32(vdupq_n_s32(in0)), wv);
                acc1 = vdotq_s32(acc1, vreinterpretq_s8_s32(vdupq_n_s32(in1)), wv);
                acc2 = vdotq_s32(acc2, vreinterpretq_s8_s32(vdupq_n_s32(in2)), wv);
                acc3 = vdotq_s32(acc3, vreinterpretq_s8_s32(vdupq_n_s32(in3)), wv);
            }

            if (!nchw_out && !is_float) {
                // NHWC vectorized requant for 4-row tile
                int8_t* out_i = static_cast<int8_t*>(C);
                int32x4_t vbias4 = {(int32_t)eff_bias[n0], n0+1<N?(int32_t)eff_bias[n0+1]:0, n0+2<N?(int32_t)eff_bias[n0+2]:0, n0+3<N?(int32_t)eff_bias[n0+3]:0};
                int32x4_t vmult4 = {req_mult[n0], n0+1<N?req_mult[n0+1]:0, n0+2<N?req_mult[n0+2]:0, n0+3<N?req_mult[n0+3]:0};
                int32x4_t vexp4 = {req_exp[n0], n0+1<N?req_exp[n0+1]:0, n0+2<N?req_exp[n0+2]:0, n0+3<N?req_exp[n0+3]:0};
                int32x4_t vzp4 = vdupq_n_s32((int32_t)out_zp);
                if (n_valid == 4) {
                    auto wr4f = [&](int row, int32x4_t v) __attribute__((always_inline)) {
                        int32x4_t s = vaddq_s32(v, vbias4);
                        int32x4_t mq = vqrdmulhq_s32(s, vmult4);
                        int32x4_t q = vaddq_s32(vrshlq_s32(mq, vexp4), vzp4);
                        int16x4_t h = vqmovn_s32(q);
                        int8x8_t b8 = vqmovn_s16(vcombine_s16(h, h));
                        vst1_lane_s32((int32_t*)(out_i + (size_t)(m+row)*N + n0),
                                      vreinterpret_s32_s8(b8), 0);
                    };
                    wr4f(0,acc0); wr4f(1,acc1); wr4f(2,acc2); wr4f(3,acc3);
                } else {
                    auto wr4 = [&](int row, int32x4_t v) {
                        int32x4_t s = vaddq_s32(v, vbias4);
                        int32x4_t mq = vqrdmulhq_s32(s, vmult4);
                        int32x4_t q = vaddq_s32(vrshlq_s32(mq, vexp4), vzp4);
                        int16x4_t h = vqmovn_s32(q);
                        int8x8_t b8 = vqmovn_s16(vcombine_s16(h, h));
                        int32_t tmp; vst1_lane_s32(&tmp, vreinterpret_s32_s8(b8), 0);
                        memcpy(out_i + (size_t)(m+row)*N + n0, &tmp, n_valid);
                    };
                    wr4(0,acc0); wr4(1,acc1); wr4(2,acc2); wr4(3,acc3);
                }
            } else if (nchw_out) {
                // vtrnq_s32 transpose: [4rows×4chans] → [4chans×4rows]
                int32x4x2_t tr0 = vtrnq_s32(acc0, acc1);
                int32x4x2_t tr1 = vtrnq_s32(acc2, acc3);
                int32x4_t cha0 = vcombine_s32(vget_low_s32(tr0.val[0]), vget_low_s32(tr1.val[0]));
                int32x4_t cha1 = vcombine_s32(vget_low_s32(tr0.val[1]), vget_low_s32(tr1.val[1]));
                int32x4_t cha2 = vcombine_s32(vget_high_s32(tr0.val[0]), vget_high_s32(tr1.val[0]));
                int32x4_t cha3 = vcombine_s32(vget_high_s32(tr0.val[1]), vget_high_s32(tr1.val[1]));
                auto write4nchw = [&](int oc, int32x4_t cv) {
                    const int n = n0 + oc;
                    int32x4_t s  = vaddq_s32(cv, vdupq_n_s32((int32_t)eff_bias[n]));
                    int32x4_t mq = vqrdmulhq_s32(s, vdupq_n_s32(req_mult[n]));
                    int32x4_t qv = vaddq_s32(vrshlq_s32(mq, vdupq_n_s32(req_exp[n])),
                                             vdupq_n_s32((int32_t)out_zp));
                    int8_t* dst = static_cast<int8_t*>(C) + n * M + m;
                    vst1_lane_s32((int32_t*)dst,
                        vreinterpret_s32_s8(vqmovn_s16(vcombine_s16(vqmovn_s32(qv), vqmovn_s32(qv)))), 0);
                };
                if (n_valid > 0) write4nchw(0, cha0);
                if (n_valid > 1) write4nchw(1, cha1);
                if (n_valid > 2) write4nchw(2, cha2);
                if (n_valid > 3) write4nchw(3, cha3);
            } else {
                int32_t r0[4], r1[4], r2[4], r3[4];
                vst1q_s32(r0, acc0); vst1q_s32(r1, acc1);
                vst1q_s32(r2, acc2); vst1q_s32(r3, acc3);
                for (int oc = 0; oc < n_valid; ++oc) {
                    const int n   = n0 + oc;
                    const int64_t b64 = eff_bias[n];
                    float* out_f = static_cast<float*>(C);
                    const float s = req_scale_f[n];
                    out_f[(m+0)*N+n] = (float)((int64_t)r0[oc]+b64) * s;
                    out_f[(m+1)*N+n] = (float)((int64_t)r1[oc]+b64) * s;
                    out_f[(m+2)*N+n] = (float)((int64_t)r2[oc]+b64) * s;
                    out_f[(m+3)*N+n] = (float)((int64_t)r3[oc]+b64) * s;
                }
            }
        }   // end co_blk loop for 4-row tile
    }

    // ── Scalar tail: remaining 0–3 rows ─────────────────────
    for (; m < M; ++m) {
        const int8_t* a_row = A + (size_t)m * K;
        for (int co_blk = co_s2; co_blk < co_e2; ++co_blk) {
            const int8_t* b_co = B_packed + (size_t)(co_blk >> 1) * K4 * 32 + (co_blk & 1) * 16;
            const int n0      = co_blk * 4;
            const int n_valid = std::min(4, N - n0);

            int32x4_t acc = vdupq_n_s32(0);
            for (int k_blk = 0; k_blk < K4; ++k_blk) {
                int32_t in4 = 0;
                int k_base = k_blk * 4;
                if (k_base + 3 < K) {
                    memcpy(&in4, a_row + k_base, 4);
                } else {
                    uint8_t tmp[4] = {};
                    for (int i = 0; i < (K & 3); ++i)
                        tmp[i] = (uint8_t)a_row[k_base + i];
                    memcpy(&in4, tmp, 4);
                }
                int8x16_t wv = vld1q_s8(b_co + k_blk * 32);
                acc = vdotq_s32(acc, vreinterpretq_s8_s32(vdupq_n_s32(in4)), wv);
            }
            if (!nchw_out && !is_float) {
                // NHWC vectorized requant for scalar tail (1 row)
                int8_t* out_i = static_cast<int8_t*>(C);
                int32x4_t vbias_s = {(int32_t)eff_bias[n0], n0+1<N?(int32_t)eff_bias[n0+1]:0, n0+2<N?(int32_t)eff_bias[n0+2]:0, n0+3<N?(int32_t)eff_bias[n0+3]:0};
                int32x4_t vmult_s = {req_mult[n0], n0+1<N?req_mult[n0+1]:0, n0+2<N?req_mult[n0+2]:0, n0+3<N?req_mult[n0+3]:0};
                int32x4_t vexp_s = {req_exp[n0], n0+1<N?req_exp[n0+1]:0, n0+2<N?req_exp[n0+2]:0, n0+3<N?req_exp[n0+3]:0};
                int32x4_t vzp_s = vdupq_n_s32((int32_t)out_zp);
                int32x4_t s = vaddq_s32(acc, vbias_s);
                int32x4_t mq = vqrdmulhq_s32(s, vmult_s);
                int32x4_t q = vaddq_s32(vrshlq_s32(mq, vexp_s), vzp_s);
                int16x4_t h = vqmovn_s32(q);
                int8x8_t b8 = vqmovn_s16(vcombine_s16(h, h));
                if (n_valid == 4) {
                    vst1_lane_s32((int32_t*)(out_i + (size_t)m*N + n0),
                                  vreinterpret_s32_s8(b8), 0);
                } else {
                    int32_t tmp; vst1_lane_s32(&tmp, vreinterpret_s32_s8(b8), 0);
                    memcpy(out_i + (size_t)m*N + n0, &tmp, n_valid);
                }
            } else {
                int32_t acc_arr[4];
                vst1q_s32(acc_arr, acc);
                for (int oc = 0; oc < n_valid; ++oc) {
                    const int n = n0 + oc;
                    int32_t total = (int32_t)((int64_t)acc_arr[oc] + eff_bias[n]);
                    if (nchw_out) {
                        static_cast<int8_t*>(C)[n * M + m] = requant_fixedpoint(total, req_mult[n], req_exp[n], out_zp);
                    } else if (is_float) {
                        static_cast<float*>(C)[m * N + n] = (float)((int64_t)acc_arr[oc] + eff_bias[n]) * req_scale_f[n];
                    } else {
                        static_cast<int8_t*>(C)[m * N + n] = requant_fixedpoint(total, req_mult[n], req_exp[n], out_zp);
                    }
                }
            }
        }   // end co_blk loop for scalar tail
    }   // end scalar tail m loop
    }   // end if (n_part || tid == 0)
#endif  // !__ARM_FEATURE_MATMUL_INT8
#elif defined(__AVX512BW__)
{
    // ── AVX-512 VNNI path: tile=16 channels, 8-row×2-co_blk micro-kernel ─────
    // B_packed: [N/16, K/4, 64]  (64 bytes = 1 zmm = 16 ch × 4 K-elems)
    // Each broadcast of A is reused for two B blocks simultaneously,
    // halving port-5 (vpbroadcastd) pressure vs. the 16-row×1-co_blk tile.
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
    const int nT  = omp_get_num_threads();
#else
    const int tid = 0, nT = 1;
#endif
    const int Co16 = (N + 15) / 16;
    const bool n_part = (nT > 1) && (M < N);
    const int m8_c = M / 8;
    const int mi_s = n_part ? 0      : (tid * m8_c) / nT;
    const int mi_e = n_part ? m8_c   : ((tid + 1) * m8_c) / nT;
    const int co_s = n_part ? (tid * Co16) / nT       : 0;
    const int co_e = n_part ? ((tid + 1) * Co16) / nT : Co16;
    const int k_full = K / 4;

#ifdef __AVX512VNNI__
    const bool do_vnni = (b_row_sums != nullptr);
    static const int32_t XMASK_S32 = (int32_t)0x80808080u;
#else
    constexpr bool do_vnni = false;
#endif

    // Per-row requantize-and-write for one m_row, sharing bias/mult/exp vectors.
    // Processes nv output channels starting at n0.
    auto rq_store_512 = [&](__m512i acc_v, int m_row, int n0, int nv,
                             __m512i vbias, __m512i vmult, __m512i vnexp) {
        acc_v = _mm512_add_epi32(acc_v, vbias);
        if (is_float) {
            alignas(64) int32_t av[16];
            _mm512_store_si512(av, acc_v);
            float* out_f = static_cast<float*>(C);
            for (int i = 0; i < nv; ++i)
                out_f[(size_t)m_row * N + n0 + i] = (float)((int64_t)av[i] + eff_bias[n0+i] - (int32_t)eff_bias[n0+i])
                                                   * req_scale_f[n0 + i];
            // Note: av[i] already includes (int32_t)eff_bias[n0+i], but eff_bias may differ
            // after truncation.  Use the proper formula:
            // out = (int64_t)raw_acc + eff_bias[n], but raw_acc is acc before adding bias.
            // Since we already added vbias (= truncated eff_bias), we need to undo for float.
            // Simpler: this path (is_float=true) is only used for M=1 FC layer, so fall
            // through to scalar below.  (Caller uses is_float only for the last linear layer.)
            // Re-do correctly:
            _mm512_store_si512(av, _mm512_sub_epi32(acc_v, vbias));  // restore raw acc
            for (int i = 0; i < nv; ++i)
                out_f[(size_t)m_row * N + n0 + i] = (float)((int64_t)av[i] + eff_bias[n0+i]) * req_scale_f[n0+i];
            return;
        }
        __m512i qm   = qrdmulh_epi32_512(acc_v, vmult);
        __m512i qout = _mm512_add_epi32(_mm512_srav_epi32(qm, vnexp),
                                         _mm512_set1_epi32((int32_t)out_zp));
        __m128i out8 = _mm512_cvtsepi32_epi8(qout);
        int8_t* out_i = static_cast<int8_t*>(C);
        if (!nchw_out) {
            if (nv == 16) {
                _mm_storeu_si128((__m128i*)(out_i + (size_t)m_row * N + n0), out8);
            } else {
                alignas(16) int8_t tmp[16];
                _mm_store_si128((__m128i*)tmp, out8);
                memcpy(out_i + (size_t)m_row * N + n0, tmp, nv);
            }
        } else {
            alignas(16) int8_t tmp[16];
            _mm_store_si128((__m128i*)tmp, out8);
            for (int i = 0; i < nv; ++i)
                out_i[(size_t)(n0 + i) * M + m_row] = tmp[i];
        }
    };

    // ── 8-row tile loop ───────────────────────────────────────────────────────
    for (int mi = mi_s; mi < mi_e; ++mi) {
        const int m = mi * 8;
        const int8_t* ar[8];
        for (int i = 0; i < 8; ++i) ar[i] = A + (size_t)(m + i) * K;

        int co_blk = co_s;

        // ── Main: 3 co_blk at once (noinline K-loop helpers, 4-row halves) ─
        // Each helper processes 4 rows × 3 N-blocks with 12 acc zmm — fits
        // cleanly in 32 registers, eliminating the register spills that killed
        // the inline 3-co_blk attempts.
#ifdef __AVX512BW__
        for (; co_blk + 2 < co_e; co_blk += 3) {
            const int8_t* b0 = B_packed + (size_t)co_blk     * K4 * 64;
            const int8_t* b1 = B_packed + (size_t)(co_blk+1) * K4 * 64;
            const int8_t* b2 = B_packed + (size_t)(co_blk+2) * K4 * 64;
            const int n0 = co_blk * 16, n1 = n0+16, n2 = n0+32;
            const int nv0 = std::min(16, N-n0), nv1 = std::min(16, N-n1), nv2 = std::min(16, N-n2);
            const int k_tail = K & 3;
            // Shared bias/mult/nexp across both 4-row halves
            alignas(64) int32_t bias0[16]={}, mult0[16]={}, nexp0[16]={};
            alignas(64) int32_t bias1[16]={}, mult1[16]={}, nexp1[16]={};
            alignas(64) int32_t bias2[16]={}, mult2[16]={}, nexp2[16]={};
            for (int i=0;i<nv0;++i) { bias0[i]=(int32_t)eff_bias[n0+i]; if(!is_float){mult0[i]=req_mult[n0+i];nexp0[i]=-req_exp[n0+i];} }
            for (int i=0;i<nv1;++i) { bias1[i]=(int32_t)eff_bias[n1+i]; if(!is_float){mult1[i]=req_mult[n1+i];nexp1[i]=-req_exp[n1+i];} }
            for (int i=0;i<nv2;++i) { bias2[i]=(int32_t)eff_bias[n2+i]; if(!is_float){mult2[i]=req_mult[n2+i];nexp2[i]=-req_exp[n2+i];} }
            __m512i vbias0=_mm512_load_si512(bias0), vmult0=_mm512_load_si512(mult0), vnexp0=_mm512_load_si512(nexp0);
            __m512i vbias1=_mm512_load_si512(bias1), vmult1=_mm512_load_si512(mult1), vnexp1=_mm512_load_si512(nexp1);
            __m512i vbias2=_mm512_load_si512(bias2), vmult2=_mm512_load_si512(mult2), vnexp2=_mm512_load_si512(nexp2);
#ifdef __AVX512VNNI__
            __m512i vc0=_mm512_setzero_si512(), vc1=_mm512_setzero_si512(), vc2=_mm512_setzero_si512();
            if (do_vnni) {
                alignas(64) int32_t corr0[16]={}, corr1[16]={}, corr2[16]={};
                for (int i=0;i<nv0;++i) corr0[i]=128*b_row_sums[n0+i];
                for (int i=0;i<nv1;++i) corr1[i]=128*b_row_sums[n1+i];
                for (int i=0;i<nv2;++i) corr2[i]=128*b_row_sums[n2+i];
                vc0=_mm512_load_si512(corr0); vc1=_mm512_load_si512(corr1); vc2=_mm512_load_si512(corr2);
            }
#endif
            // Two 4-row halves; reuse the same output buffer sequentially
            alignas(64) int32_t acc_buf[12*16];
            for (int half = 0; half < 2; ++half) {
                const int8_t* const* ar4 = ar + half*4;
                const int row_base = m + half*4;
#ifdef __AVX512VNNI__
                if (do_vnni) gemm_4x3_kloop_vnni(ar4, b0, b1, b2, k_full, k_tail, acc_buf);
                else
#endif
                gemm_4x3_kloop_bw(ar4, b0, b1, b2, k_full, k_tail, acc_buf);
                // Apply VNNI correction + requantize for 4 rows × 3 N-blocks
                for (int r = 0; r < 4; ++r) {
                    __m512i va = _mm512_load_si512(acc_buf + (r*3+0)*16);
                    __m512i vb = _mm512_load_si512(acc_buf + (r*3+1)*16);
                    __m512i vc = _mm512_load_si512(acc_buf + (r*3+2)*16);
#ifdef __AVX512VNNI__
                    if (do_vnni) { va=_mm512_sub_epi32(va,vc0); vb=_mm512_sub_epi32(vb,vc1); vc=_mm512_sub_epi32(vc,vc2); }
#endif
                    rq_store_512(va, row_base+r, n0, nv0, vbias0, vmult0, vnexp0);
                    rq_store_512(vb, row_base+r, n1, nv1, vbias1, vmult1, vnexp1);
                    rq_store_512(vc, row_base+r, n2, nv2, vbias2, vmult2, vnexp2);
                }
            }
        }
#endif  // __AVX512BW__

        // ── Main: 2 co_blk at once ─────────────────────────────────────────
        for (; co_blk + 1 < co_e; co_blk += 2) {
            const int8_t* b0 = B_packed + (size_t)co_blk * K4 * 64;
            const int8_t* b1 = B_packed + (size_t)(co_blk + 1) * K4 * 64;
            const int n0 = co_blk * 16, n1 = n0 + 16;
            const int nv0 = std::min(16, N - n0), nv1 = std::min(16, N - n1);

            __m512i acc0a=_mm512_setzero_si512(), acc0b=_mm512_setzero_si512();
            __m512i acc1a=_mm512_setzero_si512(), acc1b=_mm512_setzero_si512();
            __m512i acc2a=_mm512_setzero_si512(), acc2b=_mm512_setzero_si512();
            __m512i acc3a=_mm512_setzero_si512(), acc3b=_mm512_setzero_si512();
            __m512i acc4a=_mm512_setzero_si512(), acc4b=_mm512_setzero_si512();
            __m512i acc5a=_mm512_setzero_si512(), acc5b=_mm512_setzero_si512();
            __m512i acc6a=_mm512_setzero_si512(), acc6b=_mm512_setzero_si512();
            __m512i acc7a=_mm512_setzero_si512(), acc7b=_mm512_setzero_si512();

#define VNNI8x2(kb) do { \
    __m512i _bv0 = _mm512_loadu_si512(b0 + (kb)*64); \
    __m512i _bv1 = _mm512_loadu_si512(b1 + (kb)*64); \
    int32_t _i0,_i1,_i2,_i3,_i4,_i5,_i6,_i7; \
    memcpy(&_i0,ar[0]+(kb)*4,4); memcpy(&_i1,ar[1]+(kb)*4,4); \
    memcpy(&_i2,ar[2]+(kb)*4,4); memcpy(&_i3,ar[3]+(kb)*4,4); \
    memcpy(&_i4,ar[4]+(kb)*4,4); memcpy(&_i5,ar[5]+(kb)*4,4); \
    memcpy(&_i6,ar[6]+(kb)*4,4); memcpy(&_i7,ar[7]+(kb)*4,4); \
    if (do_vnni) { \
        _i0^=XMASK_S32;_i1^=XMASK_S32;_i2^=XMASK_S32;_i3^=XMASK_S32; \
        _i4^=XMASK_S32;_i5^=XMASK_S32;_i6^=XMASK_S32;_i7^=XMASK_S32; \
        { __m512i _av=_mm512_set1_epi32(_i0); \
          acc0a=_mm512_dpbusd_epi32(acc0a,_av,_bv0); \
          acc0b=_mm512_dpbusd_epi32(acc0b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i1); \
          acc1a=_mm512_dpbusd_epi32(acc1a,_av,_bv0); \
          acc1b=_mm512_dpbusd_epi32(acc1b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i2); \
          acc2a=_mm512_dpbusd_epi32(acc2a,_av,_bv0); \
          acc2b=_mm512_dpbusd_epi32(acc2b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i3); \
          acc3a=_mm512_dpbusd_epi32(acc3a,_av,_bv0); \
          acc3b=_mm512_dpbusd_epi32(acc3b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i4); \
          acc4a=_mm512_dpbusd_epi32(acc4a,_av,_bv0); \
          acc4b=_mm512_dpbusd_epi32(acc4b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i5); \
          acc5a=_mm512_dpbusd_epi32(acc5a,_av,_bv0); \
          acc5b=_mm512_dpbusd_epi32(acc5b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i6); \
          acc6a=_mm512_dpbusd_epi32(acc6a,_av,_bv0); \
          acc6b=_mm512_dpbusd_epi32(acc6b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i7); \
          acc7a=_mm512_dpbusd_epi32(acc7a,_av,_bv0); \
          acc7b=_mm512_dpbusd_epi32(acc7b,_av,_bv1); } \
    } else { \
        { __m512i _av=_mm512_set1_epi32(_i0); \
          acc0a=dpbssd_avx512bw(acc0a,_av,_bv0); \
          acc0b=dpbssd_avx512bw(acc0b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i1); \
          acc1a=dpbssd_avx512bw(acc1a,_av,_bv0); \
          acc1b=dpbssd_avx512bw(acc1b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i2); \
          acc2a=dpbssd_avx512bw(acc2a,_av,_bv0); \
          acc2b=dpbssd_avx512bw(acc2b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i3); \
          acc3a=dpbssd_avx512bw(acc3a,_av,_bv0); \
          acc3b=dpbssd_avx512bw(acc3b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i4); \
          acc4a=dpbssd_avx512bw(acc4a,_av,_bv0); \
          acc4b=dpbssd_avx512bw(acc4b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i5); \
          acc5a=dpbssd_avx512bw(acc5a,_av,_bv0); \
          acc5b=dpbssd_avx512bw(acc5b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i6); \
          acc6a=dpbssd_avx512bw(acc6a,_av,_bv0); \
          acc6b=dpbssd_avx512bw(acc6b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i7); \
          acc7a=dpbssd_avx512bw(acc7a,_av,_bv0); \
          acc7b=dpbssd_avx512bw(acc7b,_av,_bv1); } \
    } \
} while(0)

            int kb = 0;
            for (; kb + 3 < k_full; kb += 4) {
                VNNI8x2(kb+0); VNNI8x2(kb+1); VNNI8x2(kb+2); VNNI8x2(kb+3);
            }
            for (; kb < k_full; ++kb) { VNNI8x2(kb); }
#undef VNNI8x2
            // K tail
            if (K & 3) {
                const int kbase = k_full * 4;
                __m512i bv0 = _mm512_loadu_si512(b0 + k_full * 64);
                __m512i bv1 = _mm512_loadu_si512(b1 + k_full * 64);
                int32_t iv[8] = {};
                for (int r = 0; r < 8; ++r) {
                    uint8_t t[4] = {};
                    for (int j = 0; j < (K & 3); ++j) t[j] = (uint8_t)ar[r][kbase + j];
                    memcpy(&iv[r], t, 4);
                }
#ifdef __AVX512VNNI__
                if (do_vnni) {
                    for (int r = 0; r < 8; ++r) iv[r] ^= XMASK_S32;
                    { __m512i av=_mm512_set1_epi32(iv[0]); acc0a=_mm512_dpbusd_epi32(acc0a,av,bv0); acc0b=_mm512_dpbusd_epi32(acc0b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[1]); acc1a=_mm512_dpbusd_epi32(acc1a,av,bv0); acc1b=_mm512_dpbusd_epi32(acc1b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[2]); acc2a=_mm512_dpbusd_epi32(acc2a,av,bv0); acc2b=_mm512_dpbusd_epi32(acc2b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[3]); acc3a=_mm512_dpbusd_epi32(acc3a,av,bv0); acc3b=_mm512_dpbusd_epi32(acc3b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[4]); acc4a=_mm512_dpbusd_epi32(acc4a,av,bv0); acc4b=_mm512_dpbusd_epi32(acc4b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[5]); acc5a=_mm512_dpbusd_epi32(acc5a,av,bv0); acc5b=_mm512_dpbusd_epi32(acc5b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[6]); acc6a=_mm512_dpbusd_epi32(acc6a,av,bv0); acc6b=_mm512_dpbusd_epi32(acc6b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[7]); acc7a=_mm512_dpbusd_epi32(acc7a,av,bv0); acc7b=_mm512_dpbusd_epi32(acc7b,av,bv1); }
                } else
#endif
                {
                    { __m512i av=_mm512_set1_epi32(iv[0]); acc0a=dpbssd_avx512bw(acc0a,av,bv0); acc0b=dpbssd_avx512bw(acc0b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[1]); acc1a=dpbssd_avx512bw(acc1a,av,bv0); acc1b=dpbssd_avx512bw(acc1b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[2]); acc2a=dpbssd_avx512bw(acc2a,av,bv0); acc2b=dpbssd_avx512bw(acc2b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[3]); acc3a=dpbssd_avx512bw(acc3a,av,bv0); acc3b=dpbssd_avx512bw(acc3b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[4]); acc4a=dpbssd_avx512bw(acc4a,av,bv0); acc4b=dpbssd_avx512bw(acc4b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[5]); acc5a=dpbssd_avx512bw(acc5a,av,bv0); acc5b=dpbssd_avx512bw(acc5b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[6]); acc6a=dpbssd_avx512bw(acc6a,av,bv0); acc6b=dpbssd_avx512bw(acc6b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[7]); acc7a=dpbssd_avx512bw(acc7a,av,bv0); acc7b=dpbssd_avx512bw(acc7b,av,bv1); }
                }
            }
            // VNNI correction (subtract 128 * b_row_sums for s8->u8 conversion)
#ifdef __AVX512VNNI__
            if (do_vnni) {
                alignas(64) int32_t corr0[16]={}, corr1[16]={};
                for (int i = 0; i < nv0; ++i) corr0[i] = 128 * b_row_sums[n0 + i];
                for (int i = 0; i < nv1; ++i) corr1[i] = 128 * b_row_sums[n1 + i];
                __m512i vc0 = _mm512_load_si512(corr0);
                __m512i vc1 = _mm512_load_si512(corr1);
                acc0a=_mm512_sub_epi32(acc0a,vc0); acc0b=_mm512_sub_epi32(acc0b,vc1);
                acc1a=_mm512_sub_epi32(acc1a,vc0); acc1b=_mm512_sub_epi32(acc1b,vc1);
                acc2a=_mm512_sub_epi32(acc2a,vc0); acc2b=_mm512_sub_epi32(acc2b,vc1);
                acc3a=_mm512_sub_epi32(acc3a,vc0); acc3b=_mm512_sub_epi32(acc3b,vc1);
                acc4a=_mm512_sub_epi32(acc4a,vc0); acc4b=_mm512_sub_epi32(acc4b,vc1);
                acc5a=_mm512_sub_epi32(acc5a,vc0); acc5b=_mm512_sub_epi32(acc5b,vc1);
                acc6a=_mm512_sub_epi32(acc6a,vc0); acc6b=_mm512_sub_epi32(acc6b,vc1);
                acc7a=_mm512_sub_epi32(acc7a,vc0); acc7b=_mm512_sub_epi32(acc7b,vc1);
            }
#endif
            // Load bias/mult/exp for both co_blk blocks
            alignas(64) int32_t bias0[16]={}, mult0[16]={}, nexp0[16]={};
            alignas(64) int32_t bias1[16]={}, mult1[16]={}, nexp1[16]={};
            for (int i = 0; i < nv0; ++i) {
                bias0[i] = (int32_t)eff_bias[n0 + i];
                if (!is_float) { mult0[i] = req_mult[n0 + i]; nexp0[i] = -req_exp[n0 + i]; }
            }
            for (int i = 0; i < nv1; ++i) {
                bias1[i] = (int32_t)eff_bias[n1 + i];
                if (!is_float) { mult1[i] = req_mult[n1 + i]; nexp1[i] = -req_exp[n1 + i]; }
            }
            __m512i vbias0=_mm512_load_si512(bias0), vmult0=_mm512_load_si512(mult0), vnexp0=_mm512_load_si512(nexp0);
            __m512i vbias1=_mm512_load_si512(bias1), vmult1=_mm512_load_si512(mult1), vnexp1=_mm512_load_si512(nexp1);

            rq_store_512(acc0a, m+0, n0, nv0, vbias0, vmult0, vnexp0);
            rq_store_512(acc0b, m+0, n1, nv1, vbias1, vmult1, vnexp1);
            rq_store_512(acc1a, m+1, n0, nv0, vbias0, vmult0, vnexp0);
            rq_store_512(acc1b, m+1, n1, nv1, vbias1, vmult1, vnexp1);
            rq_store_512(acc2a, m+2, n0, nv0, vbias0, vmult0, vnexp0);
            rq_store_512(acc2b, m+2, n1, nv1, vbias1, vmult1, vnexp1);
            rq_store_512(acc3a, m+3, n0, nv0, vbias0, vmult0, vnexp0);
            rq_store_512(acc3b, m+3, n1, nv1, vbias1, vmult1, vnexp1);
            rq_store_512(acc4a, m+4, n0, nv0, vbias0, vmult0, vnexp0);
            rq_store_512(acc4b, m+4, n1, nv1, vbias1, vmult1, vnexp1);
            rq_store_512(acc5a, m+5, n0, nv0, vbias0, vmult0, vnexp0);
            rq_store_512(acc5b, m+5, n1, nv1, vbias1, vmult1, vnexp1);
            rq_store_512(acc6a, m+6, n0, nv0, vbias0, vmult0, vnexp0);
            rq_store_512(acc6b, m+6, n1, nv1, vbias1, vmult1, vnexp1);
            rq_store_512(acc7a, m+7, n0, nv0, vbias0, vmult0, vnexp0);
            rq_store_512(acc7b, m+7, n1, nv1, vbias1, vmult1, vnexp1);
        }

        // ── Tail: odd co_blk ────────────────────────────────────────────────
        if (co_blk < co_e) {
            const int8_t* b_co = B_packed + (size_t)co_blk * K4 * 64;
            const int n0 = co_blk * 16;
            const int nv = std::min(16, N - n0);

            __m512i acc0=_mm512_setzero_si512(), acc1=_mm512_setzero_si512();
            __m512i acc2=_mm512_setzero_si512(), acc3=_mm512_setzero_si512();
            __m512i acc4=_mm512_setzero_si512(), acc5=_mm512_setzero_si512();
            __m512i acc6=_mm512_setzero_si512(), acc7=_mm512_setzero_si512();

#define VNNI8x1(kb) do { \
    __m512i _bv = _mm512_loadu_si512(b_co + (kb)*64); \
    int32_t _i0,_i1,_i2,_i3,_i4,_i5,_i6,_i7; \
    memcpy(&_i0,ar[0]+(kb)*4,4); memcpy(&_i1,ar[1]+(kb)*4,4); \
    memcpy(&_i2,ar[2]+(kb)*4,4); memcpy(&_i3,ar[3]+(kb)*4,4); \
    memcpy(&_i4,ar[4]+(kb)*4,4); memcpy(&_i5,ar[5]+(kb)*4,4); \
    memcpy(&_i6,ar[6]+(kb)*4,4); memcpy(&_i7,ar[7]+(kb)*4,4); \
    if (do_vnni) { \
        _i0^=XMASK_S32;_i1^=XMASK_S32;_i2^=XMASK_S32;_i3^=XMASK_S32; \
        _i4^=XMASK_S32;_i5^=XMASK_S32;_i6^=XMASK_S32;_i7^=XMASK_S32; \
        acc0=_mm512_dpbusd_epi32(acc0,_mm512_set1_epi32(_i0),_bv); \
        acc1=_mm512_dpbusd_epi32(acc1,_mm512_set1_epi32(_i1),_bv); \
        acc2=_mm512_dpbusd_epi32(acc2,_mm512_set1_epi32(_i2),_bv); \
        acc3=_mm512_dpbusd_epi32(acc3,_mm512_set1_epi32(_i3),_bv); \
        acc4=_mm512_dpbusd_epi32(acc4,_mm512_set1_epi32(_i4),_bv); \
        acc5=_mm512_dpbusd_epi32(acc5,_mm512_set1_epi32(_i5),_bv); \
        acc6=_mm512_dpbusd_epi32(acc6,_mm512_set1_epi32(_i6),_bv); \
        acc7=_mm512_dpbusd_epi32(acc7,_mm512_set1_epi32(_i7),_bv); \
    } else { \
        acc0=dpbssd_avx512bw(acc0,_mm512_set1_epi32(_i0),_bv); \
        acc1=dpbssd_avx512bw(acc1,_mm512_set1_epi32(_i1),_bv); \
        acc2=dpbssd_avx512bw(acc2,_mm512_set1_epi32(_i2),_bv); \
        acc3=dpbssd_avx512bw(acc3,_mm512_set1_epi32(_i3),_bv); \
        acc4=dpbssd_avx512bw(acc4,_mm512_set1_epi32(_i4),_bv); \
        acc5=dpbssd_avx512bw(acc5,_mm512_set1_epi32(_i5),_bv); \
        acc6=dpbssd_avx512bw(acc6,_mm512_set1_epi32(_i6),_bv); \
        acc7=dpbssd_avx512bw(acc7,_mm512_set1_epi32(_i7),_bv); \
    } \
} while(0)

            int kb2 = 0;
            for (; kb2 + 3 < k_full; kb2 += 4) { VNNI8x1(kb2+0); VNNI8x1(kb2+1); VNNI8x1(kb2+2); VNNI8x1(kb2+3); }
            for (; kb2 < k_full; ++kb2) { VNNI8x1(kb2); }
#undef VNNI8x1
            // K tail
            if (K & 3) {
                const int kbase = k_full * 4;
                __m512i bv = _mm512_loadu_si512(b_co + k_full * 64);
                int32_t iv[8] = {};
                for (int r = 0; r < 8; ++r) {
                    uint8_t t[4] = {};
                    for (int j = 0; j < (K & 3); ++j) t[j] = (uint8_t)ar[r][kbase + j];
                    memcpy(&iv[r], t, 4);
                }
#ifdef __AVX512VNNI__
                if (do_vnni) {
                    for (int r = 0; r < 8; ++r) iv[r] ^= XMASK_S32;
                    acc0=_mm512_dpbusd_epi32(acc0,_mm512_set1_epi32(iv[0]),bv);
                    acc1=_mm512_dpbusd_epi32(acc1,_mm512_set1_epi32(iv[1]),bv);
                    acc2=_mm512_dpbusd_epi32(acc2,_mm512_set1_epi32(iv[2]),bv);
                    acc3=_mm512_dpbusd_epi32(acc3,_mm512_set1_epi32(iv[3]),bv);
                    acc4=_mm512_dpbusd_epi32(acc4,_mm512_set1_epi32(iv[4]),bv);
                    acc5=_mm512_dpbusd_epi32(acc5,_mm512_set1_epi32(iv[5]),bv);
                    acc6=_mm512_dpbusd_epi32(acc6,_mm512_set1_epi32(iv[6]),bv);
                    acc7=_mm512_dpbusd_epi32(acc7,_mm512_set1_epi32(iv[7]),bv);
                } else
#endif
                {
                    acc0=dpbssd_avx512bw(acc0,_mm512_set1_epi32(iv[0]),bv);
                    acc1=dpbssd_avx512bw(acc1,_mm512_set1_epi32(iv[1]),bv);
                    acc2=dpbssd_avx512bw(acc2,_mm512_set1_epi32(iv[2]),bv);
                    acc3=dpbssd_avx512bw(acc3,_mm512_set1_epi32(iv[3]),bv);
                    acc4=dpbssd_avx512bw(acc4,_mm512_set1_epi32(iv[4]),bv);
                    acc5=dpbssd_avx512bw(acc5,_mm512_set1_epi32(iv[5]),bv);
                    acc6=dpbssd_avx512bw(acc6,_mm512_set1_epi32(iv[6]),bv);
                    acc7=dpbssd_avx512bw(acc7,_mm512_set1_epi32(iv[7]),bv);
                }
            }
#ifdef __AVX512VNNI__
            if (do_vnni) {
                alignas(64) int32_t corr[16] = {};
                for (int i = 0; i < nv; ++i) corr[i] = 128 * b_row_sums[n0 + i];
                __m512i vcorr = _mm512_load_si512(corr);
                acc0=_mm512_sub_epi32(acc0,vcorr); acc1=_mm512_sub_epi32(acc1,vcorr);
                acc2=_mm512_sub_epi32(acc2,vcorr); acc3=_mm512_sub_epi32(acc3,vcorr);
                acc4=_mm512_sub_epi32(acc4,vcorr); acc5=_mm512_sub_epi32(acc5,vcorr);
                acc6=_mm512_sub_epi32(acc6,vcorr); acc7=_mm512_sub_epi32(acc7,vcorr);
            }
#endif
            alignas(64) int32_t bias_buf[16]={}, mult_buf[16]={}, nexp_buf[16]={};
            for (int i = 0; i < nv; ++i) {
                bias_buf[i] = (int32_t)eff_bias[n0 + i];
                if (!is_float) { mult_buf[i] = req_mult[n0 + i]; nexp_buf[i] = -req_exp[n0 + i]; }
            }
            __m512i vbias=_mm512_load_si512(bias_buf), vmult=_mm512_load_si512(mult_buf), vnexp=_mm512_load_si512(nexp_buf);
            rq_store_512(acc0, m+0, n0, nv, vbias, vmult, vnexp);
            rq_store_512(acc1, m+1, n0, nv, vbias, vmult, vnexp);
            rq_store_512(acc2, m+2, n0, nv, vbias, vmult, vnexp);
            rq_store_512(acc3, m+3, n0, nv, vbias, vmult, vnexp);
            rq_store_512(acc4, m+4, n0, nv, vbias, vmult, vnexp);
            rq_store_512(acc5, m+5, n0, nv, vbias, vmult, vnexp);
            rq_store_512(acc6, m+6, n0, nv, vbias, vmult, vnexp);
            rq_store_512(acc7, m+7, n0, nv, vbias, vmult, vnexp);
        }
    }   // end 8-row mi loop

    // ── Row tail: remaining < 8 rows ─────────────────────────────────────────
    if (n_part || tid == 0) {
        const int co_s2 = n_part ? co_s : 0;
        const int co_e2 = n_part ? co_e : Co16;
        for (int m_row = m8_c * 8; m_row < M; ++m_row) {
            const int8_t* a_row = A + (size_t)m_row * K;
            for (int co_blk = co_s2; co_blk < co_e2; ++co_blk) {
                const int8_t* b_co = B_packed + (size_t)co_blk * K4 * 64;
                const int n0 = co_blk * 16;
                const int nv = std::min(16, N - n0);
                __m512i acc = _mm512_setzero_si512();
                for (int kb = 0; kb < k_full; ++kb) {
                    int32_t iv; memcpy(&iv, a_row + kb * 4, 4);
#ifdef __AVX512VNNI__
                    if (do_vnni) {
                        iv ^= XMASK_S32;
                        acc = _mm512_dpbusd_epi32(acc, _mm512_set1_epi32(iv),
                                                      _mm512_loadu_si512(b_co + kb * 64));
                    } else
#endif
                    {
                        acc = dpbssd_avx512bw(acc, _mm512_set1_epi32(iv),
                                                  _mm512_loadu_si512(b_co + kb * 64));
                    }
                }
                if (K & 3) {
                    uint8_t t[4] = {};
                    for (int j = 0; j < (K & 3); ++j) t[j] = (uint8_t)a_row[k_full*4+j];
                    int32_t iv; memcpy(&iv, t, 4);
#ifdef __AVX512VNNI__
                    if (do_vnni) {
                        iv ^= XMASK_S32;
                        acc = _mm512_dpbusd_epi32(acc, _mm512_set1_epi32(iv),
                                                      _mm512_loadu_si512(b_co + k_full * 64));
                    } else
#endif
                    {
                        acc = dpbssd_avx512bw(acc, _mm512_set1_epi32(iv),
                                                  _mm512_loadu_si512(b_co + k_full * 64));
                    }
                }
#ifdef __AVX512VNNI__
                if (do_vnni) {
                    alignas(64) int32_t corr[16] = {};
                    for (int i = 0; i < nv; ++i) corr[i] = 128 * b_row_sums[n0 + i];
                    acc = _mm512_sub_epi32(acc, _mm512_load_si512(corr));
                }
#endif
                alignas(64) int32_t bias_buf[16]={}, mult_buf[16]={}, nexp_buf[16]={};
                for (int i = 0; i < nv; ++i) {
                    bias_buf[i] = (int32_t)eff_bias[n0+i];
                    if (!is_float) {
                        mult_buf[i] = req_mult[n0+i];
                        nexp_buf[i] = -req_exp[n0+i];
                    }
                }
                rq_store_512(acc, m_row, n0, nv,
                             _mm512_load_si512(bias_buf),
                             _mm512_load_si512(mult_buf),
                             _mm512_load_si512(nexp_buf));
            }
        }
    }
}
#elif defined(__AVX2__)
{
    // ── AVX2 path: tile=8 channels, 8-row micro-kernel ───────────────────────
    // B_packed: [N/8, K/4, 32]  (32 bytes = 1 ymm = 8 ch × 4 K-elems)
    // dpbssd_avx2 emulates signed×signed 4-byte dot product for 8 lanes.
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
    const int nT  = omp_get_num_threads();
#else
    const int tid = 0, nT = 1;
#endif
    const int Co8   = (N + 7) / 8;
    const bool n_part = (nT > 1) && (M < N);
    const int m8_c  = M / 8;
    const int k_full = K / 4;
#if defined(__AVXVNNI__)
    const bool use_avx_vnni = (b_row_sums != nullptr);
#else
    constexpr bool use_avx_vnni = false;
#endif

    // Store requantized result for one m_row × nv channels (AVX2 path).
    auto rq_store_256 = [&](__m256i acc_v, int m_row, int n0, int nv,
                             __m256i vbias, __m256i vmult, __m256i vnexp) {
        acc_v = _mm256_add_epi32(acc_v, vbias);
        if (is_float) {
            alignas(32) int32_t av[8];
            _mm256_store_si256((__m256i*)av, acc_v);
            float* out_f = static_cast<float*>(C);
            // Restore raw acc for proper int64 bias handling
            _mm256_store_si256((__m256i*)av,
                _mm256_sub_epi32(acc_v, vbias));
            for (int i = 0; i < nv; ++i)
                out_f[(size_t)m_row * N + n0 + i] =
                    (float)((int64_t)av[i] + eff_bias[n0+i]) * req_scale_f[n0+i];
            return;
        }
        __m256i qm   = qrdmulh_epi32_256(acc_v, vmult);
        __m256i qout = _mm256_add_epi32(_mm256_srav_epi32(qm, vnexp),
                                         _mm256_set1_epi32((int32_t)out_zp));
        // Saturate int32→int8: clamp then two-stage packing
        qout = _mm256_min_epi32(_mm256_max_epi32(qout, _mm256_set1_epi32(-128)),
                                 _mm256_set1_epi32(127));
        __m128i lo   = _mm256_castsi256_si128(qout);
        __m128i hi   = _mm256_extracti128_si256(qout, 1);
        __m128i q16  = _mm_packs_epi32(lo, hi);
        __m128i out8 = _mm_packs_epi16(q16, _mm_setzero_si128());
        int8_t* out_i = static_cast<int8_t*>(C);
        if (!nchw_out) {
            if (nv == 8) {
                _mm_storel_epi64((__m128i*)(out_i + (size_t)m_row * N + n0), out8);
            } else {
                alignas(16) int8_t tmp[16];
                _mm_store_si128((__m128i*)tmp, out8);
                memcpy(out_i + (size_t)m_row * N + n0, tmp, nv);
            }
        } else {
            alignas(16) int8_t tmp[16];
            _mm_store_si128((__m128i*)tmp, out8);
            for (int i = 0; i < nv; ++i)
                out_i[(size_t)(n0 + i) * M + m_row] = tmp[i];
        }
    };

    // Integer sums: every split gives the same values.
    kt_split_tiles(n_part, m8_c, Co8, tid, nT, in_parallel,
                   [&](const int mi_s, const int mi_e, const int co_s, const int co_e, const bool tail) {
    // ── 8-row tile loop (M-cache blocked, inline AVXK8 kernel) ──────────────
    // Inline kernel eliminates function-call overhead and kloop_out store/load.
    // 4× K-loop unrolling reduces loop overhead and improves OOO scheduling.
    // M-cache blocking (MC=32) reuses B data across multiple m-tiles.
    constexpr int MC = 32;
    for (int mi_blk = mi_s; mi_blk < mi_e; mi_blk += MC) {
        const int mi_blk_end = std::min(mi_blk + MC, mi_e);
        for (int co_blk = co_s; co_blk < co_e; ++co_blk) {
            const int8_t* b_co = B_packed + (size_t)co_blk * K4 * 32;
            const int n0 = co_blk * 8;
            const int nv = std::min(8, N - n0);
            // Hoist requant params outside mi loop
            alignas(32) int32_t bias_buf[8]={}, mult_buf[8]={}, nexp_buf[8]={};
            for (int i = 0; i < nv; ++i) {
                bias_buf[i] = (int32_t)eff_bias[n0+i];
                if (!is_float) {
                    mult_buf[i] = req_mult[n0+i];
                    nexp_buf[i] = -req_exp[n0+i];
                }
            }
            __m256i vbias = _mm256_load_si256((const __m256i*)bias_buf);
            __m256i vmult = _mm256_load_si256((const __m256i*)mult_buf);
            __m256i vnexp = _mm256_load_si256((const __m256i*)nexp_buf);
#if defined(__AVXVNNI__)
            alignas(32) int32_t corr_buf[8] = {};
            if (use_avx_vnni) {
                for (int i = 0; i < nv; ++i) corr_buf[i] = 128 * b_row_sums[n0 + i];
            }
            const __m256i vcorr = _mm256_load_si256((const __m256i*)corr_buf);
#endif

            for (int mi = mi_blk; mi < mi_blk_end; ++mi) {
                const int m = mi * 8;
                const int8_t* a0 = A + (size_t)(m+0)*K;
                const int8_t* a1 = A + (size_t)(m+1)*K;
                const int8_t* a2 = A + (size_t)(m+2)*K;
                const int8_t* a3 = A + (size_t)(m+3)*K;
                const int8_t* a4 = A + (size_t)(m+4)*K;
                const int8_t* a5 = A + (size_t)(m+5)*K;
                const int8_t* a6 = A + (size_t)(m+6)*K;
                const int8_t* a7 = A + (size_t)(m+7)*K;
                __m256i acc0=_mm256_setzero_si256(), acc1=_mm256_setzero_si256();
                __m256i acc2=_mm256_setzero_si256(), acc3=_mm256_setzero_si256();
                __m256i acc4=_mm256_setzero_si256(), acc5=_mm256_setzero_si256();
                __m256i acc6=_mm256_setzero_si256(), acc7=_mm256_setzero_si256();
                int kb = 0;
#if defined(__AVXVNNI__)
                if (use_avx_vnni) {
                    for (; kb + 3 < k_full; kb += 4) {
                        __builtin_prefetch(b_co + (kb + 16) * 32, 0, 0);
                        AVXVNNI8(kb); AVXVNNI8(kb+1); AVXVNNI8(kb+2); AVXVNNI8(kb+3);
                    }
                    for (; kb < k_full; ++kb) { AVXVNNI8(kb); }
                } else
#endif
                {
                for (; kb + 3 < k_full; kb += 4) {
                    // Prefetch B 16 K-steps ahead to warm L1 for large-K workloads.
                    __builtin_prefetch(b_co + (kb + 16) * 32, 0, 0);
                    AVXK8(kb); AVXK8(kb+1); AVXK8(kb+2); AVXK8(kb+3);
                }
                for (; kb < k_full; ++kb) { AVXK8(kb); }
                }
                if (K & 3) {
                    const int8_t* btail = b_co + k_full * 32;
#if defined(__AVXVNNI__)
                    if (use_avx_vnni) {
                        const __m128i _bp01 = _mm_loadu_si128((const __m128i*)btail);
                        const __m128i _bp23 = _mm_loadu_si128((const __m128i*)(btail + 16));
                        const __m128i _b_lo = _mm_unpacklo_epi16(_bp01, _bp23);
                        const __m128i _b_hi = _mm_unpackhi_epi16(_bp01, _bp23);
                        const __m256i bv = _mm256_inserti128_si256(_mm256_castsi128_si256(_b_lo), _b_hi, 1);
                        AVXVNNI8_TAIL_ROW(a0,acc0,bv); AVXVNNI8_TAIL_ROW(a1,acc1,bv);
                        AVXVNNI8_TAIL_ROW(a2,acc2,bv); AVXVNNI8_TAIL_ROW(a3,acc3,bv);
                        AVXVNNI8_TAIL_ROW(a4,acc4,bv); AVXVNNI8_TAIL_ROW(a5,acc5,bv);
                        AVXVNNI8_TAIL_ROW(a6,acc6,bv); AVXVNNI8_TAIL_ROW(a7,acc7,bv);
                    } else
#endif
                    {
                    const __m256i btlo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)btail));
                    const __m256i bthi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(btail+16)));
                    AVXK8_TAIL(a0,acc0,btlo,bthi); AVXK8_TAIL(a1,acc1,btlo,bthi);
                    AVXK8_TAIL(a2,acc2,btlo,bthi); AVXK8_TAIL(a3,acc3,btlo,bthi);
                    AVXK8_TAIL(a4,acc4,btlo,bthi); AVXK8_TAIL(a5,acc5,btlo,bthi);
                    AVXK8_TAIL(a6,acc6,btlo,bthi); AVXK8_TAIL(a7,acc7,btlo,bthi);
                    }
                }
#if defined(__AVXVNNI__)
                if (use_avx_vnni) {
                    acc0=_mm256_sub_epi32(acc0, vcorr); acc1=_mm256_sub_epi32(acc1, vcorr);
                    acc2=_mm256_sub_epi32(acc2, vcorr); acc3=_mm256_sub_epi32(acc3, vcorr);
                    acc4=_mm256_sub_epi32(acc4, vcorr); acc5=_mm256_sub_epi32(acc5, vcorr);
                    acc6=_mm256_sub_epi32(acc6, vcorr); acc7=_mm256_sub_epi32(acc7, vcorr);
                }
#endif
                rq_store_256(acc0, m+0, n0, nv, vbias, vmult, vnexp);
                rq_store_256(acc1, m+1, n0, nv, vbias, vmult, vnexp);
                rq_store_256(acc2, m+2, n0, nv, vbias, vmult, vnexp);
                rq_store_256(acc3, m+3, n0, nv, vbias, vmult, vnexp);
                rq_store_256(acc4, m+4, n0, nv, vbias, vmult, vnexp);
                rq_store_256(acc5, m+5, n0, nv, vbias, vmult, vnexp);
                rq_store_256(acc6, m+6, n0, nv, vbias, vmult, vnexp);
                rq_store_256(acc7, m+7, n0, nv, vbias, vmult, vnexp);
            }
        }
    }   // end 8-row mi loop

    // ── Row tail: remaining < 8 rows (K-pair layout) ─────────────────────────
    if (tail) {
        const int co_s2 = co_s;
        const int co_e2 = co_e;
        for (int m_row = m8_c * 8; m_row < M; ++m_row) {
            const int8_t* a_row = A + (size_t)m_row * K;
            for (int co_blk = co_s2; co_blk < co_e2; ++co_blk) {
                const int8_t* b_co = B_packed + (size_t)co_blk * K4 * 32;
                const int n0 = co_blk * 8;
                const int nv = std::min(8, N - n0);
                __m256i acc = _mm256_setzero_si256();
                // K-pair layout: bytes 0..15 = K-pair 0, bytes 16..31 = K-pair 1
                for (int kb = 0; kb < k_full; ++kb) {
                    int32_t iv; memcpy(&iv, a_row + kb * 4, 4);
                    const __m256i bvlo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b_co + kb * 32)));
                    const __m256i bvhi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b_co + kb * 32 + 16)));
                    const int32_t p01 = (uint16_t)(int8_t)iv | ((uint32_t)(uint16_t)(int8_t)(iv >> 8) << 16);
                    const int32_t p23 = (uint16_t)(int8_t)(iv >> 16) | ((uint32_t)(uint16_t)(int8_t)(iv >> 24) << 16);
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_set1_epi32(p01), bvlo));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_set1_epi32(p23), bvhi));
                }
                if (K & 3) {
                    uint8_t t[4] = {};
                    for (int j = 0; j < (K & 3); ++j) t[j] = (uint8_t)a_row[k_full*4+j];
                    int32_t iv; memcpy(&iv, t, 4);
                    const __m256i bvlo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b_co + k_full * 32)));
                    const __m256i bvhi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b_co + k_full * 32 + 16)));
                    const int32_t p01 = (uint16_t)(int8_t)iv | ((uint32_t)(uint16_t)(int8_t)(iv >> 8) << 16);
                    const int32_t p23 = (uint16_t)(int8_t)(iv >> 16) | ((uint32_t)(uint16_t)(int8_t)(iv >> 24) << 16);
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_set1_epi32(p01), bvlo));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_set1_epi32(p23), bvhi));
                }
                alignas(32) int32_t bias_buf[8]={}, mult_buf[8]={}, nexp_buf[8]={};
                for (int i = 0; i < nv; ++i) {
                    bias_buf[i] = (int32_t)eff_bias[n0+i];
                    if (!is_float) {
                        mult_buf[i] = req_mult[n0+i];
                        nexp_buf[i] = -req_exp[n0+i];
                    }
                }
                rq_store_256(acc, m_row, n0, nv,
                             _mm256_load_si256((const __m256i*)bias_buf),
                             _mm256_load_si256((const __m256i*)mult_buf),
                             _mm256_load_si256((const __m256i*)nexp_buf));
            }
        }
    }
    });
}
#else
    // Scalar fallback (no SIMD)
    for (int m = 0; m < M; ++m) {
        const int8_t* a_row = A + (size_t)m * K;
        for (int n = 0; n < N; ++n) {
            int64_t acc = eff_bias[n];
            int co_blk = n / 4, oc = n % 4;
            for (int k = 0; k < K; ++k) {
                int k_blk = k / 4, ki = k % 4;
                acc += (int32_t)a_row[k] *
                       (int32_t)B_packed[(co_blk * K4 + k_blk) * 16 + oc * 4 + ki];
            }
            if (nchw_out)
                static_cast<int8_t*>(C)[n * M + m] = requant_fixedpoint((int32_t)acc, req_mult[n], req_exp[n], out_zp);
            else if (is_float)
                static_cast<float*>(C)[m * N + n] = (float)acc * req_scale_f[n];
            else
                static_cast<int8_t*>(C)[m * N + n] = requant_fixedpoint((int32_t)acc, req_mult[n], req_exp[n], out_zp);
        }
    }
#endif
}   // end gemm_int8_neon



void gemm_int8_int32(
    const int8_t*  A,
    const int8_t*  B_packed,
    int32_t*       C,
    int M, int K, int N,
    const int32_t* b_row_sums,
    StreamHandle   /* stream */)
{
    int K4  = (K + 3) / 4;
    int Co4 = (N + 3) / 4;

#if defined(__ARM_FEATURE_MATMUL_INT8)
// ── SMMLA (I8MM) path: B packed as [Co_t8, K8, 64] ──────────────────────────
// Matches pack_weights_sdot when __ARM_FEATURE_MATMUL_INT8 is defined.
{
    const int K8    = (K + 7) / 8;
    const int Co_t8 = (N + 7) / 8;
    // A packing buffer: 2 rows × [K8 × 16 bytes]
    // a_pk[kb*16 + 0..7]  = A[m,   kb*8 .. kb*8+7]
    // a_pk[kb*16 + 8..15] = A[m+1, kb*8 .. kb*8+7]
    std::vector<int8_t> a_pk((size_t)K8 * 16, 0);

    int m = 0;
    for (; m + 1 < M; m += 2) {
        const int8_t* a0 = A + (size_t)m       * K;
        const int8_t* a1 = A + (size_t)(m + 1) * K;
        std::memset(a_pk.data(), 0, (size_t)K8 * 16);
        for (int kb = 0; kb < K8; ++kb) {
            int8_t* dst = a_pk.data() + kb * 16;
            for (int ki = 0; ki < 8; ++ki) {
                int k = kb * 8 + ki;
                if (k < K) { dst[ki] = a0[k]; dst[8 + ki] = a1[k]; }
            }
        }
        for (int co8 = 0; co8 < Co_t8; ++co8) {
            int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
            for (int kb = 0; kb < K8; ++kb) {
                int8x16_t av = vld1q_s8(a_pk.data() + kb * 16);
                const int8_t* bp = B_packed + ((size_t)co8 * K8 + kb) * 64;
                acc0 = vmmlaq_s32(acc0, av, vld1q_s8(bp));
                acc1 = vmmlaq_s32(acc1, av, vld1q_s8(bp + 16));
                acc2 = vmmlaq_s32(acc2, av, vld1q_s8(bp + 32));
                acc3 = vmmlaq_s32(acc3, av, vld1q_s8(bp + 48));
            }
            // vmmlaq_s32 result layout: [row0·col0, row0·col1, row1·col0, row1·col1]
            int32_t r0[4], r1[4], r2[4], r3[4];
            vst1q_s32(r0, acc0); vst1q_s32(r1, acc1);
            vst1q_s32(r2, acc2); vst1q_s32(r3, acc3);
            for (int cp = 0; cp < 4; ++cp) {
                int c0 = co8 * 8 + cp * 2, c1 = c0 + 1;
                const int32_t* r = (cp == 0) ? r0 : (cp == 1) ? r1 : (cp == 2) ? r2 : r3;
                if (c0 < N) { C[(size_t)m*N + c0] = r[0]; C[(size_t)(m+1)*N + c0] = r[2]; }
                if (c1 < N) { C[(size_t)m*N + c1] = r[1]; C[(size_t)(m+1)*N + c1] = r[3]; }
            }
        }
    }
    // 1-row tail
    if (m < M) {
        const int8_t* a0 = A + (size_t)m * K;
        std::memset(a_pk.data(), 0, (size_t)K8 * 16);
        for (int kb = 0; kb < K8; ++kb)
            for (int ki = 0; ki < 8; ++ki)
                if (kb * 8 + ki < K) a_pk[kb * 16 + ki] = a0[kb * 8 + ki];
        for (int co8 = 0; co8 < Co_t8; ++co8) {
            int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
            for (int kb = 0; kb < K8; ++kb) {
                int8x16_t av = vld1q_s8(a_pk.data() + kb * 16);
                const int8_t* bp = B_packed + ((size_t)co8 * K8 + kb) * 64;
                acc0 = vmmlaq_s32(acc0, av, vld1q_s8(bp));
                acc1 = vmmlaq_s32(acc1, av, vld1q_s8(bp + 16));
                acc2 = vmmlaq_s32(acc2, av, vld1q_s8(bp + 32));
                acc3 = vmmlaq_s32(acc3, av, vld1q_s8(bp + 48));
            }
            int32_t r0[4], r1[4], r2[4], r3[4];
            vst1q_s32(r0, acc0); vst1q_s32(r1, acc1);
            vst1q_s32(r2, acc2); vst1q_s32(r3, acc3);
            for (int cp = 0; cp < 4; ++cp) {
                int c0 = co8 * 8 + cp * 2, c1 = c0 + 1;
                const int32_t* r = (cp == 0) ? r0 : (cp == 1) ? r1 : (cp == 2) ? r2 : r3;
                if (c0 < N) C[(size_t)m*N + c0] = r[0];
                if (c1 < N) C[(size_t)m*N + c1] = r[1];
            }
        }
    }
    return;
}
#elif defined(__ARM_NEON)
    const int k_full = K / 4;
    const int m8_count = M / 8;

    // ── MR=8/NR=8: 8 rows × 2 B-tiles, 32 regs exact, zero spills ───────────
    for (int mi = 0; mi < m8_count; ++mi) {
        const int m = mi * 8;
        const int8_t* a0 = A + (size_t)(m+0)*K;
        const int8_t* a1 = A + (size_t)(m+1)*K;
        const int8_t* a2 = A + (size_t)(m+2)*K;
        const int8_t* a3 = A + (size_t)(m+3)*K;
        const int8_t* a4 = A + (size_t)(m+4)*K;
        const int8_t* a5 = A + (size_t)(m+5)*K;
        const int8_t* a6 = A + (size_t)(m+6)*K;
        const int8_t* a7 = A + (size_t)(m+7)*K;

        // NR=8: process 2 co_blks simultaneously, sharing av0..av7
        // co_blk always starts at 0 (always even) → [Co8,K4,32] safe.
        int co_blk = 0;
        for (; co_blk + 1 < Co4; co_blk += 2) {
            const int n0a = co_blk * 4;
            const int n0b = n0a + 4;
            if (n0b >= N) break;
            // [Co8,K4,32]: b_co0 at b_tile+k*32, b_co1 at b_tile+k*32+16 (same cache line)
            const int8_t* b_tile = B_packed + (size_t)(co_blk >> 1) * K4 * 32;
            const int n_valid_b = std::min(4, N - n0b);

            int32x4_t acc0a=vdupq_n_s32(0), acc1a=vdupq_n_s32(0);
            int32x4_t acc2a=vdupq_n_s32(0), acc3a=vdupq_n_s32(0);
            int32x4_t acc4a=vdupq_n_s32(0), acc5a=vdupq_n_s32(0);
            int32x4_t acc6a=vdupq_n_s32(0), acc7a=vdupq_n_s32(0);
            int32x4_t acc0b=vdupq_n_s32(0), acc1b=vdupq_n_s32(0);
            int32x4_t acc2b=vdupq_n_s32(0), acc3b=vdupq_n_s32(0);
            int32x4_t acc4b=vdupq_n_s32(0), acc5b=vdupq_n_s32(0);
            int32x4_t acc6b=vdupq_n_s32(0), acc7b=vdupq_n_s32(0);

            int k_blk = 0;
            for (; k_blk + 3 < k_full; k_blk += 4) {
                const int off = k_blk * 4;
                const int8_t* bp = b_tile + k_blk * 32;
                __builtin_prefetch(bp + 8*32, 0, 3);
                int8x16_t av0=vld1q_s8(a0+off), av1=vld1q_s8(a1+off);
                int8x16_t av2=vld1q_s8(a2+off), av3=vld1q_s8(a3+off);
                int8x16_t av4=vld1q_s8(a4+off), av5=vld1q_s8(a5+off);
                int8x16_t av6=vld1q_s8(a6+off), av7=vld1q_s8(a7+off);
                int8x16_t bv0a=vld1q_s8(bp),    bv1a=vld1q_s8(bp+32);
                int8x16_t bv2a=vld1q_s8(bp+64), bv3a=vld1q_s8(bp+96);
                int8x16_t bv0b=vld1q_s8(bp+16), bv1b=vld1q_s8(bp+48);
                int8x16_t bv2b=vld1q_s8(bp+80), bv3b=vld1q_s8(bp+112);
                acc0a=vdotq_laneq_s32(acc0a,bv0a,av0,0); acc0b=vdotq_laneq_s32(acc0b,bv0b,av0,0);
                acc1a=vdotq_laneq_s32(acc1a,bv0a,av1,0); acc1b=vdotq_laneq_s32(acc1b,bv0b,av1,0);
                acc2a=vdotq_laneq_s32(acc2a,bv0a,av2,0); acc2b=vdotq_laneq_s32(acc2b,bv0b,av2,0);
                acc3a=vdotq_laneq_s32(acc3a,bv0a,av3,0); acc3b=vdotq_laneq_s32(acc3b,bv0b,av3,0);
                acc4a=vdotq_laneq_s32(acc4a,bv0a,av4,0); acc4b=vdotq_laneq_s32(acc4b,bv0b,av4,0);
                acc5a=vdotq_laneq_s32(acc5a,bv0a,av5,0); acc5b=vdotq_laneq_s32(acc5b,bv0b,av5,0);
                acc6a=vdotq_laneq_s32(acc6a,bv0a,av6,0); acc6b=vdotq_laneq_s32(acc6b,bv0b,av6,0);
                acc7a=vdotq_laneq_s32(acc7a,bv0a,av7,0); acc7b=vdotq_laneq_s32(acc7b,bv0b,av7,0);
                acc0a=vdotq_laneq_s32(acc0a,bv1a,av0,1); acc0b=vdotq_laneq_s32(acc0b,bv1b,av0,1);
                acc1a=vdotq_laneq_s32(acc1a,bv1a,av1,1); acc1b=vdotq_laneq_s32(acc1b,bv1b,av1,1);
                acc2a=vdotq_laneq_s32(acc2a,bv1a,av2,1); acc2b=vdotq_laneq_s32(acc2b,bv1b,av2,1);
                acc3a=vdotq_laneq_s32(acc3a,bv1a,av3,1); acc3b=vdotq_laneq_s32(acc3b,bv1b,av3,1);
                acc4a=vdotq_laneq_s32(acc4a,bv1a,av4,1); acc4b=vdotq_laneq_s32(acc4b,bv1b,av4,1);
                acc5a=vdotq_laneq_s32(acc5a,bv1a,av5,1); acc5b=vdotq_laneq_s32(acc5b,bv1b,av5,1);
                acc6a=vdotq_laneq_s32(acc6a,bv1a,av6,1); acc6b=vdotq_laneq_s32(acc6b,bv1b,av6,1);
                acc7a=vdotq_laneq_s32(acc7a,bv1a,av7,1); acc7b=vdotq_laneq_s32(acc7b,bv1b,av7,1);
                acc0a=vdotq_laneq_s32(acc0a,bv2a,av0,2); acc0b=vdotq_laneq_s32(acc0b,bv2b,av0,2);
                acc1a=vdotq_laneq_s32(acc1a,bv2a,av1,2); acc1b=vdotq_laneq_s32(acc1b,bv2b,av1,2);
                acc2a=vdotq_laneq_s32(acc2a,bv2a,av2,2); acc2b=vdotq_laneq_s32(acc2b,bv2b,av2,2);
                acc3a=vdotq_laneq_s32(acc3a,bv2a,av3,2); acc3b=vdotq_laneq_s32(acc3b,bv2b,av3,2);
                acc4a=vdotq_laneq_s32(acc4a,bv2a,av4,2); acc4b=vdotq_laneq_s32(acc4b,bv2b,av4,2);
                acc5a=vdotq_laneq_s32(acc5a,bv2a,av5,2); acc5b=vdotq_laneq_s32(acc5b,bv2b,av5,2);
                acc6a=vdotq_laneq_s32(acc6a,bv2a,av6,2); acc6b=vdotq_laneq_s32(acc6b,bv2b,av6,2);
                acc7a=vdotq_laneq_s32(acc7a,bv2a,av7,2); acc7b=vdotq_laneq_s32(acc7b,bv2b,av7,2);
                acc0a=vdotq_laneq_s32(acc0a,bv3a,av0,3); acc0b=vdotq_laneq_s32(acc0b,bv3b,av0,3);
                acc1a=vdotq_laneq_s32(acc1a,bv3a,av1,3); acc1b=vdotq_laneq_s32(acc1b,bv3b,av1,3);
                acc2a=vdotq_laneq_s32(acc2a,bv3a,av2,3); acc2b=vdotq_laneq_s32(acc2b,bv3b,av2,3);
                acc3a=vdotq_laneq_s32(acc3a,bv3a,av3,3); acc3b=vdotq_laneq_s32(acc3b,bv3b,av3,3);
                acc4a=vdotq_laneq_s32(acc4a,bv3a,av4,3); acc4b=vdotq_laneq_s32(acc4b,bv3b,av4,3);
                acc5a=vdotq_laneq_s32(acc5a,bv3a,av5,3); acc5b=vdotq_laneq_s32(acc5b,bv3b,av5,3);
                acc6a=vdotq_laneq_s32(acc6a,bv3a,av6,3); acc6b=vdotq_laneq_s32(acc6b,bv3b,av6,3);
                acc7a=vdotq_laneq_s32(acc7a,bv3a,av7,3); acc7b=vdotq_laneq_s32(acc7b,bv3b,av7,3);
            }
            for (; k_blk < k_full; ++k_blk) {
                int8x16_t wva = vld1q_s8(b_tile + k_blk * 32);
                int8x16_t wvb = vld1q_s8(b_tile + k_blk * 32 + 16);
                int32_t i0,i1,i2,i3,i4,i5,i6,i7;
                memcpy(&i0,a0+k_blk*4,4); memcpy(&i1,a1+k_blk*4,4);
                memcpy(&i2,a2+k_blk*4,4); memcpy(&i3,a3+k_blk*4,4);
                memcpy(&i4,a4+k_blk*4,4); memcpy(&i5,a5+k_blk*4,4);
                memcpy(&i6,a6+k_blk*4,4); memcpy(&i7,a7+k_blk*4,4);
                #define S2I32(acca,accb,iv) \
                    acca=vdotq_s32(acca,vreinterpretq_s8_s32(vdupq_n_s32(iv)),wva); \
                    accb=vdotq_s32(accb,vreinterpretq_s8_s32(vdupq_n_s32(iv)),wvb)
                S2I32(acc0a,acc0b,i0); S2I32(acc1a,acc1b,i1); S2I32(acc2a,acc2b,i2); S2I32(acc3a,acc3b,i3);
                S2I32(acc4a,acc4b,i4); S2I32(acc5a,acc5b,i5); S2I32(acc6a,acc6b,i6); S2I32(acc7a,acc7b,i7);
                #undef S2I32
            }
            if (K & 3) {
                int k_base = k_full * 4;
                uint8_t t0[4]={},t1[4]={},t2[4]={},t3[4]={},t4[4]={},t5[4]={},t6[4]={},t7[4]={};
                for (int i = 0; i < (K & 3); ++i) {
                    t0[i]=(uint8_t)a0[k_base+i]; t1[i]=(uint8_t)a1[k_base+i];
                    t2[i]=(uint8_t)a2[k_base+i]; t3[i]=(uint8_t)a3[k_base+i];
                    t4[i]=(uint8_t)a4[k_base+i]; t5[i]=(uint8_t)a5[k_base+i];
                    t6[i]=(uint8_t)a6[k_base+i]; t7[i]=(uint8_t)a7[k_base+i];
                }
                int32_t i0,i1,i2,i3,i4,i5,i6,i7;
                memcpy(&i0,t0,4); memcpy(&i1,t1,4); memcpy(&i2,t2,4); memcpy(&i3,t3,4);
                memcpy(&i4,t4,4); memcpy(&i5,t5,4); memcpy(&i6,t6,4); memcpy(&i7,t7,4);
                int8x16_t wva = vld1q_s8(b_tile + k_full * 32);
                int8x16_t wvb = vld1q_s8(b_tile + k_full * 32 + 16);
                #define S2I32(acca,accb,iv) \
                    acca=vdotq_s32(acca,vreinterpretq_s8_s32(vdupq_n_s32(iv)),wva); \
                    accb=vdotq_s32(accb,vreinterpretq_s8_s32(vdupq_n_s32(iv)),wvb)
                S2I32(acc0a,acc0b,i0); S2I32(acc1a,acc1b,i1); S2I32(acc2a,acc2b,i2); S2I32(acc3a,acc3b,i3);
                S2I32(acc4a,acc4b,i4); S2I32(acc5a,acc5b,i5); S2I32(acc6a,acc6b,i6); S2I32(acc7a,acc7b,i7);
                #undef S2I32
            }
            // Store tile-a (always full) and tile-b
            vst1q_s32(C+(m+0)*N+n0a, acc0a); vst1q_s32(C+(m+1)*N+n0a, acc1a);
            vst1q_s32(C+(m+2)*N+n0a, acc2a); vst1q_s32(C+(m+3)*N+n0a, acc3a);
            vst1q_s32(C+(m+4)*N+n0a, acc4a); vst1q_s32(C+(m+5)*N+n0a, acc5a);
            vst1q_s32(C+(m+6)*N+n0a, acc6a); vst1q_s32(C+(m+7)*N+n0a, acc7a);
            if (n_valid_b == 4) {
                vst1q_s32(C+(m+0)*N+n0b, acc0b); vst1q_s32(C+(m+1)*N+n0b, acc1b);
                vst1q_s32(C+(m+2)*N+n0b, acc2b); vst1q_s32(C+(m+3)*N+n0b, acc3b);
                vst1q_s32(C+(m+4)*N+n0b, acc4b); vst1q_s32(C+(m+5)*N+n0b, acc5b);
                vst1q_s32(C+(m+6)*N+n0b, acc6b); vst1q_s32(C+(m+7)*N+n0b, acc7b);
            } else {
                int32_t rb[8][4];
                vst1q_s32(rb[0],acc0b); vst1q_s32(rb[1],acc1b); vst1q_s32(rb[2],acc2b); vst1q_s32(rb[3],acc3b);
                vst1q_s32(rb[4],acc4b); vst1q_s32(rb[5],acc5b); vst1q_s32(rb[6],acc6b); vst1q_s32(rb[7],acc7b);
                for (int oc = 0; oc < n_valid_b; ++oc)
                    for (int row = 0; row < 8; ++row)
                        C[(m+row)*N+n0b+oc] = rb[row][oc];
            }
        }  // end NR=8 loop

        // NR=4 fallback for last odd co_blk
        for (; co_blk < Co4; ++co_blk) {
            // [Co8,K4,32]: even co_blk uses bytes 0-15, odd uses bytes 16-31 per k-group
            const int8_t* b_co = B_packed + (size_t)(co_blk >> 1) * K4 * 32 + (co_blk & 1) * 16;
            const int n0      = co_blk * 4;
            const int n_valid = std::min(4, N - n0);

            int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
            int32x4_t acc4 = vdupq_n_s32(0), acc5 = vdupq_n_s32(0);
            int32x4_t acc6 = vdupq_n_s32(0), acc7 = vdupq_n_s32(0);

            int k_blk = 0;
            for (; k_blk + 3 < k_full; k_blk += 4) {
                const int off = k_blk * 4;
                const int8_t* bp = b_co + k_blk * 32;
                int8x16_t av0=vld1q_s8(a0+off), av1=vld1q_s8(a1+off);
                int8x16_t av2=vld1q_s8(a2+off), av3=vld1q_s8(a3+off);
                int8x16_t av4=vld1q_s8(a4+off), av5=vld1q_s8(a5+off);
                int8x16_t av6=vld1q_s8(a6+off), av7=vld1q_s8(a7+off);
                int8x16_t bv0=vld1q_s8(bp), bv1=vld1q_s8(bp+32);
                int8x16_t bv2=vld1q_s8(bp+64), bv3=vld1q_s8(bp+96);
                acc0=vdotq_laneq_s32(acc0,bv0,av0,0); acc1=vdotq_laneq_s32(acc1,bv0,av1,0);
                acc2=vdotq_laneq_s32(acc2,bv0,av2,0); acc3=vdotq_laneq_s32(acc3,bv0,av3,0);
                acc4=vdotq_laneq_s32(acc4,bv0,av4,0); acc5=vdotq_laneq_s32(acc5,bv0,av5,0);
                acc6=vdotq_laneq_s32(acc6,bv0,av6,0); acc7=vdotq_laneq_s32(acc7,bv0,av7,0);
                acc0=vdotq_laneq_s32(acc0,bv1,av0,1); acc1=vdotq_laneq_s32(acc1,bv1,av1,1);
                acc2=vdotq_laneq_s32(acc2,bv1,av2,1); acc3=vdotq_laneq_s32(acc3,bv1,av3,1);
                acc4=vdotq_laneq_s32(acc4,bv1,av4,1); acc5=vdotq_laneq_s32(acc5,bv1,av5,1);
                acc6=vdotq_laneq_s32(acc6,bv1,av6,1); acc7=vdotq_laneq_s32(acc7,bv1,av7,1);
                acc0=vdotq_laneq_s32(acc0,bv2,av0,2); acc1=vdotq_laneq_s32(acc1,bv2,av1,2);
                acc2=vdotq_laneq_s32(acc2,bv2,av2,2); acc3=vdotq_laneq_s32(acc3,bv2,av3,2);
                acc4=vdotq_laneq_s32(acc4,bv2,av4,2); acc5=vdotq_laneq_s32(acc5,bv2,av5,2);
                acc6=vdotq_laneq_s32(acc6,bv2,av6,2); acc7=vdotq_laneq_s32(acc7,bv2,av7,2);
                acc0=vdotq_laneq_s32(acc0,bv3,av0,3); acc1=vdotq_laneq_s32(acc1,bv3,av1,3);
                acc2=vdotq_laneq_s32(acc2,bv3,av2,3); acc3=vdotq_laneq_s32(acc3,bv3,av3,3);
                acc4=vdotq_laneq_s32(acc4,bv3,av4,3); acc5=vdotq_laneq_s32(acc5,bv3,av5,3);
                acc6=vdotq_laneq_s32(acc6,bv3,av6,3); acc7=vdotq_laneq_s32(acc7,bv3,av7,3);
            }
            for (; k_blk < k_full; ++k_blk) {
                int8x16_t wv = vld1q_s8(b_co + k_blk * 32);
                int32_t i0,i1,i2,i3,i4,i5,i6,i7;
                memcpy(&i0,a0+k_blk*4,4); memcpy(&i1,a1+k_blk*4,4);
                memcpy(&i2,a2+k_blk*4,4); memcpy(&i3,a3+k_blk*4,4);
                memcpy(&i4,a4+k_blk*4,4); memcpy(&i5,a5+k_blk*4,4);
                memcpy(&i6,a6+k_blk*4,4); memcpy(&i7,a7+k_blk*4,4);
                acc0=vdotq_s32(acc0,vreinterpretq_s8_s32(vdupq_n_s32(i0)),wv);
                acc1=vdotq_s32(acc1,vreinterpretq_s8_s32(vdupq_n_s32(i1)),wv);
                acc2=vdotq_s32(acc2,vreinterpretq_s8_s32(vdupq_n_s32(i2)),wv);
                acc3=vdotq_s32(acc3,vreinterpretq_s8_s32(vdupq_n_s32(i3)),wv);
                acc4=vdotq_s32(acc4,vreinterpretq_s8_s32(vdupq_n_s32(i4)),wv);
                acc5=vdotq_s32(acc5,vreinterpretq_s8_s32(vdupq_n_s32(i5)),wv);
                acc6=vdotq_s32(acc6,vreinterpretq_s8_s32(vdupq_n_s32(i6)),wv);
                acc7=vdotq_s32(acc7,vreinterpretq_s8_s32(vdupq_n_s32(i7)),wv);
            }
            if (K & 3) {
                int k_base = k_full * 4;
                uint8_t t0[4]={},t1[4]={},t2[4]={},t3[4]={},t4[4]={},t5[4]={},t6[4]={},t7[4]={};
                for (int i = 0; i < (K & 3); ++i) {
                    t0[i]=(uint8_t)a0[k_base+i]; t1[i]=(uint8_t)a1[k_base+i];
                    t2[i]=(uint8_t)a2[k_base+i]; t3[i]=(uint8_t)a3[k_base+i];
                    t4[i]=(uint8_t)a4[k_base+i]; t5[i]=(uint8_t)a5[k_base+i];
                    t6[i]=(uint8_t)a6[k_base+i]; t7[i]=(uint8_t)a7[k_base+i];
                }
                int32_t i0,i1,i2,i3,i4,i5,i6,i7;
                memcpy(&i0,t0,4); memcpy(&i1,t1,4); memcpy(&i2,t2,4); memcpy(&i3,t3,4);
                memcpy(&i4,t4,4); memcpy(&i5,t5,4); memcpy(&i6,t6,4); memcpy(&i7,t7,4);
                int8x16_t wv = vld1q_s8(b_co + k_full * 32);
                acc0=vdotq_s32(acc0,vreinterpretq_s8_s32(vdupq_n_s32(i0)),wv);
                acc1=vdotq_s32(acc1,vreinterpretq_s8_s32(vdupq_n_s32(i1)),wv);
                acc2=vdotq_s32(acc2,vreinterpretq_s8_s32(vdupq_n_s32(i2)),wv);
                acc3=vdotq_s32(acc3,vreinterpretq_s8_s32(vdupq_n_s32(i3)),wv);
                acc4=vdotq_s32(acc4,vreinterpretq_s8_s32(vdupq_n_s32(i4)),wv);
                acc5=vdotq_s32(acc5,vreinterpretq_s8_s32(vdupq_n_s32(i5)),wv);
                acc6=vdotq_s32(acc6,vreinterpretq_s8_s32(vdupq_n_s32(i6)),wv);
                acc7=vdotq_s32(acc7,vreinterpretq_s8_s32(vdupq_n_s32(i7)),wv);
            }
            if (n_valid == 4) {
                vst1q_s32(C+(m+0)*N+n0, acc0); vst1q_s32(C+(m+1)*N+n0, acc1);
                vst1q_s32(C+(m+2)*N+n0, acc2); vst1q_s32(C+(m+3)*N+n0, acc3);
                vst1q_s32(C+(m+4)*N+n0, acc4); vst1q_s32(C+(m+5)*N+n0, acc5);
                vst1q_s32(C+(m+6)*N+n0, acc6); vst1q_s32(C+(m+7)*N+n0, acc7);
            } else {
                int32_t r0[4],r1[4],r2[4],r3[4],r4[4],r5[4],r6[4],r7[4];
                vst1q_s32(r0,acc0); vst1q_s32(r1,acc1); vst1q_s32(r2,acc2); vst1q_s32(r3,acc3);
                vst1q_s32(r4,acc4); vst1q_s32(r5,acc5); vst1q_s32(r6,acc6); vst1q_s32(r7,acc7);
                for (int oc = 0; oc < n_valid; ++oc) {
                    C[(m+0)*N+n0+oc]=r0[oc]; C[(m+1)*N+n0+oc]=r1[oc];
                    C[(m+2)*N+n0+oc]=r2[oc]; C[(m+3)*N+n0+oc]=r3[oc];
                    C[(m+4)*N+n0+oc]=r4[oc]; C[(m+5)*N+n0+oc]=r5[oc];
                    C[(m+6)*N+n0+oc]=r6[oc]; C[(m+7)*N+n0+oc]=r7[oc];
                }
            }
        }  // end NR=4 fallback
    }  // end MR=8 main loop

    // ── 4-row tail ────────────────────────────────────────────────
    int m = m8_count * 8;

    // ── 4-row tail ────────────────────────────────────────────────
    for (; m + 3 < M; m += 4) {
        const int8_t* a0 = A + (size_t)(m+0) * K;
        const int8_t* a1 = A + (size_t)(m+1) * K;
        const int8_t* a2 = A + (size_t)(m+2) * K;
        const int8_t* a3 = A + (size_t)(m+3) * K;

        for (int co_blk = 0; co_blk < Co4; ++co_blk) {
            const int8_t* b_co = B_packed + (size_t)(co_blk >> 1) * K4 * 32 + (co_blk & 1) * 16;
            const int n0      = co_blk * 4;
            const int n_valid = std::min(4, N - n0);

            int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);

            int k_blk = 0;
            for (; k_blk + 3 < k_full; k_blk += 4) {
                int8x16_t av0 = vld1q_s8(a0 + k_blk * 4);
                int8x16_t av1 = vld1q_s8(a1 + k_blk * 4);
                int8x16_t av2 = vld1q_s8(a2 + k_blk * 4);
                int8x16_t av3 = vld1q_s8(a3 + k_blk * 4);
                int8x16_t bv0 = vld1q_s8(b_co + (k_blk+0) * 32);
                int8x16_t bv1 = vld1q_s8(b_co + (k_blk+1) * 32);
                int8x16_t bv2 = vld1q_s8(b_co + (k_blk+2) * 32);
                int8x16_t bv3 = vld1q_s8(b_co + (k_blk+3) * 32);
                acc0=vdotq_laneq_s32(acc0,bv0,av0,0); acc0=vdotq_laneq_s32(acc0,bv1,av0,1);
                acc0=vdotq_laneq_s32(acc0,bv2,av0,2); acc0=vdotq_laneq_s32(acc0,bv3,av0,3);
                acc1=vdotq_laneq_s32(acc1,bv0,av1,0); acc1=vdotq_laneq_s32(acc1,bv1,av1,1);
                acc1=vdotq_laneq_s32(acc1,bv2,av1,2); acc1=vdotq_laneq_s32(acc1,bv3,av1,3);
                acc2=vdotq_laneq_s32(acc2,bv0,av2,0); acc2=vdotq_laneq_s32(acc2,bv1,av2,1);
                acc2=vdotq_laneq_s32(acc2,bv2,av2,2); acc2=vdotq_laneq_s32(acc2,bv3,av2,3);
                acc3=vdotq_laneq_s32(acc3,bv0,av3,0); acc3=vdotq_laneq_s32(acc3,bv1,av3,1);
                acc3=vdotq_laneq_s32(acc3,bv2,av3,2); acc3=vdotq_laneq_s32(acc3,bv3,av3,3);
            }
            for (; k_blk < k_full; ++k_blk) {
                int8x16_t wv = vld1q_s8(b_co + k_blk * 32);
                int32_t in0,in1,in2,in3;
                memcpy(&in0,a0+k_blk*4,4); memcpy(&in1,a1+k_blk*4,4);
                memcpy(&in2,a2+k_blk*4,4); memcpy(&in3,a3+k_blk*4,4);
                acc0=vdotq_s32(acc0,vreinterpretq_s8_s32(vdupq_n_s32(in0)),wv);
                acc1=vdotq_s32(acc1,vreinterpretq_s8_s32(vdupq_n_s32(in1)),wv);
                acc2=vdotq_s32(acc2,vreinterpretq_s8_s32(vdupq_n_s32(in2)),wv);
                acc3=vdotq_s32(acc3,vreinterpretq_s8_s32(vdupq_n_s32(in3)),wv);
            }
            if (K & 3) {
                int k_base = k_full * 4;
                uint8_t t0[4]={},t1[4]={},t2[4]={},t3[4]={};
                for (int i = 0; i < (K & 3); ++i) {
                    t0[i]=(uint8_t)a0[k_base+i]; t1[i]=(uint8_t)a1[k_base+i];
                    t2[i]=(uint8_t)a2[k_base+i]; t3[i]=(uint8_t)a3[k_base+i];
                }
                int32_t in0,in1,in2,in3;
                memcpy(&in0,t0,4); memcpy(&in1,t1,4);
                memcpy(&in2,t2,4); memcpy(&in3,t3,4);
                int8x16_t wv = vld1q_s8(b_co + k_full * 32);
                acc0=vdotq_s32(acc0,vreinterpretq_s8_s32(vdupq_n_s32(in0)),wv);
                acc1=vdotq_s32(acc1,vreinterpretq_s8_s32(vdupq_n_s32(in1)),wv);
                acc2=vdotq_s32(acc2,vreinterpretq_s8_s32(vdupq_n_s32(in2)),wv);
                acc3=vdotq_s32(acc3,vreinterpretq_s8_s32(vdupq_n_s32(in3)),wv);
            }

            if (n_valid == 4) {
                vst1q_s32(C+(m+0)*N+n0, acc0); vst1q_s32(C+(m+1)*N+n0, acc1);
                vst1q_s32(C+(m+2)*N+n0, acc2); vst1q_s32(C+(m+3)*N+n0, acc3);
            } else {
                int32_t r0[4],r1[4],r2[4],r3[4];
                vst1q_s32(r0,acc0); vst1q_s32(r1,acc1); vst1q_s32(r2,acc2); vst1q_s32(r3,acc3);
                for (int oc = 0; oc < n_valid; ++oc) {
                    C[(m+0)*N+n0+oc]=r0[oc]; C[(m+1)*N+n0+oc]=r1[oc];
                    C[(m+2)*N+n0+oc]=r2[oc]; C[(m+3)*N+n0+oc]=r3[oc];
                }
            }
        }
    }

    // ── 2-row tail ────────────────────────────────────────────────
    for (; m + 1 < M; m += 2) {
        const int8_t* a0 = A + (size_t)(m+0) * K;
        const int8_t* a1 = A + (size_t)(m+1) * K;

        // NR=8: process 2 co_blks simultaneously (halves B loads)
        int co_blk = 0;
        for (; co_blk + 1 < Co4; co_blk += 2) {
            const int n0a = co_blk * 4;
            const int n0b = n0a + 4;
            if (n0b >= N) break;
            // co_blk is always even here (loop steps by 2)
            const int8_t* b_tile2 = B_packed + (size_t)(co_blk >> 1) * K4 * 32;
            const int8_t* b_co0 = b_tile2;        // bytes 0-15 per k-group
            const int8_t* b_co1 = b_tile2 + 16;   // bytes 16-31 per k-group
            const int n_valid_b = std::min(4, N - n0b);

            int32x4_t acc0a = vdupq_n_s32(0), acc0b = vdupq_n_s32(0);
            int32x4_t acc1a = vdupq_n_s32(0), acc1b = vdupq_n_s32(0);

            int k_blk = 0;
            for (; k_blk + 3 < k_full; k_blk += 4) {
                const int off = k_blk * 4;
                int8x16_t av0 = vld1q_s8(a0 + off), av1 = vld1q_s8(a1 + off);
                int8x16_t bv0a=vld1q_s8(b_co0+k_blk*32   ), bv1a=vld1q_s8(b_co0+k_blk*32+32);
                int8x16_t bv2a=vld1q_s8(b_co0+k_blk*32+64), bv3a=vld1q_s8(b_co0+k_blk*32+96);
                int8x16_t bv0b=vld1q_s8(b_co1+k_blk*32   ), bv1b=vld1q_s8(b_co1+k_blk*32+32);
                int8x16_t bv2b=vld1q_s8(b_co1+k_blk*32+64), bv3b=vld1q_s8(b_co1+k_blk*32+96);
                acc0a=vdotq_laneq_s32(acc0a,bv0a,av0,0); acc0a=vdotq_laneq_s32(acc0a,bv1a,av0,1);
                acc0a=vdotq_laneq_s32(acc0a,bv2a,av0,2); acc0a=vdotq_laneq_s32(acc0a,bv3a,av0,3);
                acc0b=vdotq_laneq_s32(acc0b,bv0b,av0,0); acc0b=vdotq_laneq_s32(acc0b,bv1b,av0,1);
                acc0b=vdotq_laneq_s32(acc0b,bv2b,av0,2); acc0b=vdotq_laneq_s32(acc0b,bv3b,av0,3);
                acc1a=vdotq_laneq_s32(acc1a,bv0a,av1,0); acc1a=vdotq_laneq_s32(acc1a,bv1a,av1,1);
                acc1a=vdotq_laneq_s32(acc1a,bv2a,av1,2); acc1a=vdotq_laneq_s32(acc1a,bv3a,av1,3);
                acc1b=vdotq_laneq_s32(acc1b,bv0b,av1,0); acc1b=vdotq_laneq_s32(acc1b,bv1b,av1,1);
                acc1b=vdotq_laneq_s32(acc1b,bv2b,av1,2); acc1b=vdotq_laneq_s32(acc1b,bv3b,av1,3);
            }
            for (; k_blk < k_full; ++k_blk) {
                int8x16_t wva = vld1q_s8(b_co0 + k_blk * 32);
                int8x16_t wvb = vld1q_s8(b_co1 + k_blk * 32);
                int32_t i0, i1;
                memcpy(&i0, a0 + k_blk*4, 4); memcpy(&i1, a1 + k_blk*4, 4);
                int8x16_t av0 = vreinterpretq_s8_s32(vdupq_n_s32(i0));
                int8x16_t av1 = vreinterpretq_s8_s32(vdupq_n_s32(i1));
                acc0a=vdotq_s32(acc0a,av0,wva); acc0b=vdotq_s32(acc0b,av0,wvb);
                acc1a=vdotq_s32(acc1a,av1,wva); acc1b=vdotq_s32(acc1b,av1,wvb);
            }
            if (K & 3) {
                int k_base = k_full * 4;
                uint8_t t0[4]={}, t1[4]={};
                for (int i = 0; i < (K & 3); ++i) {
                    t0[i]=(uint8_t)a0[k_base+i]; t1[i]=(uint8_t)a1[k_base+i];
                }
                int32_t i0, i1;
                memcpy(&i0, t0, 4); memcpy(&i1, t1, 4);
                int8x16_t wva = vld1q_s8(b_co0 + k_full * 32);
                int8x16_t wvb = vld1q_s8(b_co1 + k_full * 32);
                int8x16_t av0 = vreinterpretq_s8_s32(vdupq_n_s32(i0));
                int8x16_t av1 = vreinterpretq_s8_s32(vdupq_n_s32(i1));
                acc0a=vdotq_s32(acc0a,av0,wva); acc0b=vdotq_s32(acc0b,av0,wvb);
                acc1a=vdotq_s32(acc1a,av1,wva); acc1b=vdotq_s32(acc1b,av1,wvb);
            }
            vst1q_s32(C+(m+0)*N+n0a, acc0a); vst1q_s32(C+(m+1)*N+n0a, acc1a);
            if (n_valid_b == 4) {
                vst1q_s32(C+(m+0)*N+n0b, acc0b); vst1q_s32(C+(m+1)*N+n0b, acc1b);
            } else {
                int32_t rb0[4], rb1[4];
                vst1q_s32(rb0, acc0b); vst1q_s32(rb1, acc1b);
                for (int oc = 0; oc < n_valid_b; ++oc) {
                    C[(m+0)*N+n0b+oc]=rb0[oc]; C[(m+1)*N+n0b+oc]=rb1[oc];
                }
            }
        }
        // NR=4 fallback for last odd co_blk
        for (; co_blk < Co4; ++co_blk) {
            const int8_t* b_co = B_packed + (size_t)(co_blk >> 1) * K4 * 32 + (co_blk & 1) * 16;
            const int n0      = co_blk * 4;
            const int n_valid = std::min(4, N - n0);

            int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
            int k_blk = 0;
            for (; k_blk + 3 < k_full; k_blk += 4) {
                int8x16_t av0 = vld1q_s8(a0 + k_blk*4), av1 = vld1q_s8(a1 + k_blk*4);
                int8x16_t bv0=vld1q_s8(b_co+k_blk*32   ), bv1=vld1q_s8(b_co+k_blk*32+32);
                int8x16_t bv2=vld1q_s8(b_co+k_blk*32+64), bv3=vld1q_s8(b_co+k_blk*32+96);
                acc0=vdotq_laneq_s32(acc0,bv0,av0,0); acc0=vdotq_laneq_s32(acc0,bv1,av0,1);
                acc0=vdotq_laneq_s32(acc0,bv2,av0,2); acc0=vdotq_laneq_s32(acc0,bv3,av0,3);
                acc1=vdotq_laneq_s32(acc1,bv0,av1,0); acc1=vdotq_laneq_s32(acc1,bv1,av1,1);
                acc1=vdotq_laneq_s32(acc1,bv2,av1,2); acc1=vdotq_laneq_s32(acc1,bv3,av1,3);
            }
            for (; k_blk < k_full; ++k_blk) {
                int8x16_t wv = vld1q_s8(b_co + k_blk * 32);
                int32_t i0, i1;
                memcpy(&i0, a0+k_blk*4, 4); memcpy(&i1, a1+k_blk*4, 4);
                acc0=vdotq_s32(acc0,vreinterpretq_s8_s32(vdupq_n_s32(i0)),wv);
                acc1=vdotq_s32(acc1,vreinterpretq_s8_s32(vdupq_n_s32(i1)),wv);
            }
            if (K & 3) {
                int k_base = k_full * 4;
                uint8_t t0[4]={}, t1[4]={};
                for (int i = 0; i < (K & 3); ++i) {
                    t0[i]=(uint8_t)a0[k_base+i]; t1[i]=(uint8_t)a1[k_base+i];
                }
                int32_t i0, i1;
                memcpy(&i0,t0,4); memcpy(&i1,t1,4);
                int8x16_t wv = vld1q_s8(b_co + k_full * 32);
                acc0=vdotq_s32(acc0,vreinterpretq_s8_s32(vdupq_n_s32(i0)),wv);
                acc1=vdotq_s32(acc1,vreinterpretq_s8_s32(vdupq_n_s32(i1)),wv);
            }
            if (n_valid == 4) {
                vst1q_s32(C+(m+0)*N+n0, acc0); vst1q_s32(C+(m+1)*N+n0, acc1);
            } else {
                int32_t r0[4], r1[4];
                vst1q_s32(r0,acc0); vst1q_s32(r1,acc1);
                for (int oc = 0; oc < n_valid; ++oc) {
                    C[(m+0)*N+n0+oc]=r0[oc]; C[(m+1)*N+n0+oc]=r1[oc];
                }
            }
        }
    }

    // ── Scalar tail: remaining 0-3 rows ──────────────────────────
    for (; m < M; ++m) {
        const int8_t* a_row = A + (size_t)m * K;
        for (int co_blk = 0; co_blk < Co4; ++co_blk) {
            const int8_t* b_co = B_packed + (size_t)(co_blk >> 1) * K4 * 32 + (co_blk & 1) * 16;
            const int n0      = co_blk * 4;
            const int n_valid = std::min(4, N - n0);

            int32x4_t acc = vdupq_n_s32(0);
            for (int k_blk = 0; k_blk < K4; ++k_blk) {
                int32_t in4 = 0;
                int k_base = k_blk * 4;
                if (k_base + 3 < K) {
                    memcpy(&in4, a_row + k_base, 4);
                } else {
                    uint8_t tmp[4] = {};
                    for (int i = 0; i < (K & 3); ++i)
                        tmp[i] = (uint8_t)a_row[k_base + i];
                    memcpy(&in4, tmp, 4);
                }
                int8x16_t wv = vld1q_s8(b_co + k_blk * 32);
                acc = vdotq_s32(acc, vreinterpretq_s8_s32(vdupq_n_s32(in4)), wv);
            }
            int32_t acc_arr[4];
            vst1q_s32(acc_arr, acc);
            for (int oc = 0; oc < n_valid; ++oc)
                C[m*N + n0+oc] = acc_arr[oc];
        }
    }
#elif defined(__AVX512BW__)
{
    // ── AVX-512 VNNI: 8-row × 2-co_blk tile, raw int32 output ───────────────
    // B_packed: [N/16, K/4, 64]  (64 bytes = 1 zmm = 16 ch × 4 K-elems)
    // Each A broadcast is reused for two B blocks, halving port-5 pressure.
    const int Co16  = (N + 15) / 16;
    const int m8_c  = M / 8;
    const int k_full = K / 4;

#ifdef __AVX512VNNI__
    const bool do_vnni = (b_row_sums != nullptr);
    static const int32_t XMASK_S32 = (int32_t)0x80808080u;
#else
    constexpr bool do_vnni = false;
#endif

    for (int mi = 0; mi < m8_c; ++mi) {
        const int m = mi * 8;
        const int8_t* ar[8];
        for (int i = 0; i < 8; ++i) ar[i] = A + (size_t)(m + i) * K;

        int co_blk = 0;

        // ── Main: 2 co_blk at once ─────────────────────────────────────────
        for (; co_blk + 1 < Co16; co_blk += 2) {
            const int8_t* b0 = B_packed + (size_t)co_blk * K4 * 64;
            const int8_t* b1 = B_packed + (size_t)(co_blk + 1) * K4 * 64;
            const int n0 = co_blk * 16, n1 = n0 + 16;
            const int nv0 = std::min(16, N - n0), nv1 = std::min(16, N - n1);

            __m512i acc0a=_mm512_setzero_si512(), acc0b=_mm512_setzero_si512();
            __m512i acc1a=_mm512_setzero_si512(), acc1b=_mm512_setzero_si512();
            __m512i acc2a=_mm512_setzero_si512(), acc2b=_mm512_setzero_si512();
            __m512i acc3a=_mm512_setzero_si512(), acc3b=_mm512_setzero_si512();
            __m512i acc4a=_mm512_setzero_si512(), acc4b=_mm512_setzero_si512();
            __m512i acc5a=_mm512_setzero_si512(), acc5b=_mm512_setzero_si512();
            __m512i acc6a=_mm512_setzero_si512(), acc6b=_mm512_setzero_si512();
            __m512i acc7a=_mm512_setzero_si512(), acc7b=_mm512_setzero_si512();

#define VNNI8x2I32(kb) do { \
    __m512i _bv0 = _mm512_loadu_si512(b0 + (kb)*64); \
    __m512i _bv1 = _mm512_loadu_si512(b1 + (kb)*64); \
    int32_t _i0,_i1,_i2,_i3,_i4,_i5,_i6,_i7; \
    memcpy(&_i0,ar[0]+(kb)*4,4); memcpy(&_i1,ar[1]+(kb)*4,4); \
    memcpy(&_i2,ar[2]+(kb)*4,4); memcpy(&_i3,ar[3]+(kb)*4,4); \
    memcpy(&_i4,ar[4]+(kb)*4,4); memcpy(&_i5,ar[5]+(kb)*4,4); \
    memcpy(&_i6,ar[6]+(kb)*4,4); memcpy(&_i7,ar[7]+(kb)*4,4); \
    if (do_vnni) { \
        _i0^=XMASK_S32;_i1^=XMASK_S32;_i2^=XMASK_S32;_i3^=XMASK_S32; \
        _i4^=XMASK_S32;_i5^=XMASK_S32;_i6^=XMASK_S32;_i7^=XMASK_S32; \
        { __m512i _av=_mm512_set1_epi32(_i0); \
          acc0a=_mm512_dpbusd_epi32(acc0a,_av,_bv0); \
          acc0b=_mm512_dpbusd_epi32(acc0b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i1); \
          acc1a=_mm512_dpbusd_epi32(acc1a,_av,_bv0); \
          acc1b=_mm512_dpbusd_epi32(acc1b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i2); \
          acc2a=_mm512_dpbusd_epi32(acc2a,_av,_bv0); \
          acc2b=_mm512_dpbusd_epi32(acc2b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i3); \
          acc3a=_mm512_dpbusd_epi32(acc3a,_av,_bv0); \
          acc3b=_mm512_dpbusd_epi32(acc3b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i4); \
          acc4a=_mm512_dpbusd_epi32(acc4a,_av,_bv0); \
          acc4b=_mm512_dpbusd_epi32(acc4b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i5); \
          acc5a=_mm512_dpbusd_epi32(acc5a,_av,_bv0); \
          acc5b=_mm512_dpbusd_epi32(acc5b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i6); \
          acc6a=_mm512_dpbusd_epi32(acc6a,_av,_bv0); \
          acc6b=_mm512_dpbusd_epi32(acc6b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i7); \
          acc7a=_mm512_dpbusd_epi32(acc7a,_av,_bv0); \
          acc7b=_mm512_dpbusd_epi32(acc7b,_av,_bv1); } \
    } else { \
        { __m512i _av=_mm512_set1_epi32(_i0); \
          acc0a=dpbssd_avx512bw(acc0a,_av,_bv0); \
          acc0b=dpbssd_avx512bw(acc0b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i1); \
          acc1a=dpbssd_avx512bw(acc1a,_av,_bv0); \
          acc1b=dpbssd_avx512bw(acc1b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i2); \
          acc2a=dpbssd_avx512bw(acc2a,_av,_bv0); \
          acc2b=dpbssd_avx512bw(acc2b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i3); \
          acc3a=dpbssd_avx512bw(acc3a,_av,_bv0); \
          acc3b=dpbssd_avx512bw(acc3b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i4); \
          acc4a=dpbssd_avx512bw(acc4a,_av,_bv0); \
          acc4b=dpbssd_avx512bw(acc4b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i5); \
          acc5a=dpbssd_avx512bw(acc5a,_av,_bv0); \
          acc5b=dpbssd_avx512bw(acc5b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i6); \
          acc6a=dpbssd_avx512bw(acc6a,_av,_bv0); \
          acc6b=dpbssd_avx512bw(acc6b,_av,_bv1); } \
        { __m512i _av=_mm512_set1_epi32(_i7); \
          acc7a=dpbssd_avx512bw(acc7a,_av,_bv0); \
          acc7b=dpbssd_avx512bw(acc7b,_av,_bv1); } \
    } \
} while(0)

            int kb = 0;
            for (; kb + 3 < k_full; kb += 4) {
                VNNI8x2I32(kb+0); VNNI8x2I32(kb+1); VNNI8x2I32(kb+2); VNNI8x2I32(kb+3);
            }
            for (; kb < k_full; ++kb) { VNNI8x2I32(kb); }
#undef VNNI8x2I32
            // K tail
            if (K & 3) {
                const int kbase = k_full * 4;
                __m512i bv0 = _mm512_loadu_si512(b0 + k_full * 64);
                __m512i bv1 = _mm512_loadu_si512(b1 + k_full * 64);
                int32_t iv[8] = {};
                for (int r = 0; r < 8; ++r) {
                    uint8_t t[4] = {};
                    for (int j = 0; j < (K & 3); ++j) t[j] = (uint8_t)ar[r][kbase + j];
                    memcpy(&iv[r], t, 4);
                }
#ifdef __AVX512VNNI__
                if (do_vnni) {
                    for (int r = 0; r < 8; ++r) iv[r] ^= XMASK_S32;
                    { __m512i av=_mm512_set1_epi32(iv[0]); acc0a=_mm512_dpbusd_epi32(acc0a,av,bv0); acc0b=_mm512_dpbusd_epi32(acc0b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[1]); acc1a=_mm512_dpbusd_epi32(acc1a,av,bv0); acc1b=_mm512_dpbusd_epi32(acc1b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[2]); acc2a=_mm512_dpbusd_epi32(acc2a,av,bv0); acc2b=_mm512_dpbusd_epi32(acc2b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[3]); acc3a=_mm512_dpbusd_epi32(acc3a,av,bv0); acc3b=_mm512_dpbusd_epi32(acc3b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[4]); acc4a=_mm512_dpbusd_epi32(acc4a,av,bv0); acc4b=_mm512_dpbusd_epi32(acc4b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[5]); acc5a=_mm512_dpbusd_epi32(acc5a,av,bv0); acc5b=_mm512_dpbusd_epi32(acc5b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[6]); acc6a=_mm512_dpbusd_epi32(acc6a,av,bv0); acc6b=_mm512_dpbusd_epi32(acc6b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[7]); acc7a=_mm512_dpbusd_epi32(acc7a,av,bv0); acc7b=_mm512_dpbusd_epi32(acc7b,av,bv1); }
                } else
#endif
                {
                    { __m512i av=_mm512_set1_epi32(iv[0]); acc0a=dpbssd_avx512bw(acc0a,av,bv0); acc0b=dpbssd_avx512bw(acc0b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[1]); acc1a=dpbssd_avx512bw(acc1a,av,bv0); acc1b=dpbssd_avx512bw(acc1b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[2]); acc2a=dpbssd_avx512bw(acc2a,av,bv0); acc2b=dpbssd_avx512bw(acc2b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[3]); acc3a=dpbssd_avx512bw(acc3a,av,bv0); acc3b=dpbssd_avx512bw(acc3b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[4]); acc4a=dpbssd_avx512bw(acc4a,av,bv0); acc4b=dpbssd_avx512bw(acc4b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[5]); acc5a=dpbssd_avx512bw(acc5a,av,bv0); acc5b=dpbssd_avx512bw(acc5b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[6]); acc6a=dpbssd_avx512bw(acc6a,av,bv0); acc6b=dpbssd_avx512bw(acc6b,av,bv1); }
                    { __m512i av=_mm512_set1_epi32(iv[7]); acc7a=dpbssd_avx512bw(acc7a,av,bv0); acc7b=dpbssd_avx512bw(acc7b,av,bv1); }
                }
            }
            // VNNI correction
#ifdef __AVX512VNNI__
            if (do_vnni) {
                alignas(64) int32_t corr0[16]={}, corr1[16]={};
                for (int i = 0; i < nv0; ++i) corr0[i] = 128 * b_row_sums[n0 + i];
                for (int i = 0; i < nv1; ++i) corr1[i] = 128 * b_row_sums[n1 + i];
                __m512i vc0 = _mm512_load_si512(corr0);
                __m512i vc1 = _mm512_load_si512(corr1);
                acc0a=_mm512_sub_epi32(acc0a,vc0); acc0b=_mm512_sub_epi32(acc0b,vc1);
                acc1a=_mm512_sub_epi32(acc1a,vc0); acc1b=_mm512_sub_epi32(acc1b,vc1);
                acc2a=_mm512_sub_epi32(acc2a,vc0); acc2b=_mm512_sub_epi32(acc2b,vc1);
                acc3a=_mm512_sub_epi32(acc3a,vc0); acc3b=_mm512_sub_epi32(acc3b,vc1);
                acc4a=_mm512_sub_epi32(acc4a,vc0); acc4b=_mm512_sub_epi32(acc4b,vc1);
                acc5a=_mm512_sub_epi32(acc5a,vc0); acc5b=_mm512_sub_epi32(acc5b,vc1);
                acc6a=_mm512_sub_epi32(acc6a,vc0); acc6b=_mm512_sub_epi32(acc6b,vc1);
                acc7a=_mm512_sub_epi32(acc7a,vc0); acc7b=_mm512_sub_epi32(acc7b,vc1);
            }
#endif
            // Store raw int32 accumulators for both co_blk blocks
            auto store_acc = [&](__m512i acc, int row, int nc, int nv) {
                int32_t* out_row = C + (size_t)(m + row) * N + nc;
                if (nv == 16) {
                    _mm512_storeu_si512(out_row, acc);
                } else {
                    alignas(64) int32_t tmp[16];
                    _mm512_store_si512(tmp, acc);
                    memcpy(out_row, tmp, nv * sizeof(int32_t));
                }
            };
            store_acc(acc0a,0,n0,nv0); store_acc(acc0b,0,n1,nv1);
            store_acc(acc1a,1,n0,nv0); store_acc(acc1b,1,n1,nv1);
            store_acc(acc2a,2,n0,nv0); store_acc(acc2b,2,n1,nv1);
            store_acc(acc3a,3,n0,nv0); store_acc(acc3b,3,n1,nv1);
            store_acc(acc4a,4,n0,nv0); store_acc(acc4b,4,n1,nv1);
            store_acc(acc5a,5,n0,nv0); store_acc(acc5b,5,n1,nv1);
            store_acc(acc6a,6,n0,nv0); store_acc(acc6b,6,n1,nv1);
            store_acc(acc7a,7,n0,nv0); store_acc(acc7b,7,n1,nv1);
        }

        // ── Tail: odd co_blk ────────────────────────────────────────────────
        if (co_blk < Co16) {
            const int8_t* b_co = B_packed + (size_t)co_blk * K4 * 64;
            const int n0 = co_blk * 16;
            const int nv = std::min(16, N - n0);

            __m512i acc0=_mm512_setzero_si512(), acc1=_mm512_setzero_si512();
            __m512i acc2=_mm512_setzero_si512(), acc3=_mm512_setzero_si512();
            __m512i acc4=_mm512_setzero_si512(), acc5=_mm512_setzero_si512();
            __m512i acc6=_mm512_setzero_si512(), acc7=_mm512_setzero_si512();

#define VNNI8x1I32(kb) do { \
    __m512i _bv = _mm512_loadu_si512(b_co + (kb)*64); \
    int32_t _i0,_i1,_i2,_i3,_i4,_i5,_i6,_i7; \
    memcpy(&_i0,ar[0]+(kb)*4,4); memcpy(&_i1,ar[1]+(kb)*4,4); \
    memcpy(&_i2,ar[2]+(kb)*4,4); memcpy(&_i3,ar[3]+(kb)*4,4); \
    memcpy(&_i4,ar[4]+(kb)*4,4); memcpy(&_i5,ar[5]+(kb)*4,4); \
    memcpy(&_i6,ar[6]+(kb)*4,4); memcpy(&_i7,ar[7]+(kb)*4,4); \
    if (do_vnni) { \
        _i0^=XMASK_S32;_i1^=XMASK_S32;_i2^=XMASK_S32;_i3^=XMASK_S32; \
        _i4^=XMASK_S32;_i5^=XMASK_S32;_i6^=XMASK_S32;_i7^=XMASK_S32; \
        acc0=_mm512_dpbusd_epi32(acc0,_mm512_set1_epi32(_i0),_bv); \
        acc1=_mm512_dpbusd_epi32(acc1,_mm512_set1_epi32(_i1),_bv); \
        acc2=_mm512_dpbusd_epi32(acc2,_mm512_set1_epi32(_i2),_bv); \
        acc3=_mm512_dpbusd_epi32(acc3,_mm512_set1_epi32(_i3),_bv); \
        acc4=_mm512_dpbusd_epi32(acc4,_mm512_set1_epi32(_i4),_bv); \
        acc5=_mm512_dpbusd_epi32(acc5,_mm512_set1_epi32(_i5),_bv); \
        acc6=_mm512_dpbusd_epi32(acc6,_mm512_set1_epi32(_i6),_bv); \
        acc7=_mm512_dpbusd_epi32(acc7,_mm512_set1_epi32(_i7),_bv); \
    } else { \
        acc0=dpbssd_avx512bw(acc0,_mm512_set1_epi32(_i0),_bv); \
        acc1=dpbssd_avx512bw(acc1,_mm512_set1_epi32(_i1),_bv); \
        acc2=dpbssd_avx512bw(acc2,_mm512_set1_epi32(_i2),_bv); \
        acc3=dpbssd_avx512bw(acc3,_mm512_set1_epi32(_i3),_bv); \
        acc4=dpbssd_avx512bw(acc4,_mm512_set1_epi32(_i4),_bv); \
        acc5=dpbssd_avx512bw(acc5,_mm512_set1_epi32(_i5),_bv); \
        acc6=dpbssd_avx512bw(acc6,_mm512_set1_epi32(_i6),_bv); \
        acc7=dpbssd_avx512bw(acc7,_mm512_set1_epi32(_i7),_bv); \
    } \
} while(0)

            int kb2 = 0;
            for (; kb2 + 3 < k_full; kb2 += 4) { VNNI8x1I32(kb2+0); VNNI8x1I32(kb2+1); VNNI8x1I32(kb2+2); VNNI8x1I32(kb2+3); }
            for (; kb2 < k_full; ++kb2) { VNNI8x1I32(kb2); }
#undef VNNI8x1I32
            // K tail
            if (K & 3) {
                const int kbase = k_full * 4;
                __m512i bv = _mm512_loadu_si512(b_co + k_full * 64);
                int32_t iv[8] = {};
                for (int r = 0; r < 8; ++r) {
                    uint8_t t[4] = {};
                    for (int j = 0; j < (K & 3); ++j) t[j] = (uint8_t)ar[r][kbase + j];
                    memcpy(&iv[r], t, 4);
                }
#ifdef __AVX512VNNI__
                if (do_vnni) {
                    for (int r = 0; r < 8; ++r) iv[r] ^= XMASK_S32;
                    acc0=_mm512_dpbusd_epi32(acc0,_mm512_set1_epi32(iv[0]),bv);
                    acc1=_mm512_dpbusd_epi32(acc1,_mm512_set1_epi32(iv[1]),bv);
                    acc2=_mm512_dpbusd_epi32(acc2,_mm512_set1_epi32(iv[2]),bv);
                    acc3=_mm512_dpbusd_epi32(acc3,_mm512_set1_epi32(iv[3]),bv);
                    acc4=_mm512_dpbusd_epi32(acc4,_mm512_set1_epi32(iv[4]),bv);
                    acc5=_mm512_dpbusd_epi32(acc5,_mm512_set1_epi32(iv[5]),bv);
                    acc6=_mm512_dpbusd_epi32(acc6,_mm512_set1_epi32(iv[6]),bv);
                    acc7=_mm512_dpbusd_epi32(acc7,_mm512_set1_epi32(iv[7]),bv);
                } else
#endif
                {
                    acc0=dpbssd_avx512bw(acc0,_mm512_set1_epi32(iv[0]),bv);
                    acc1=dpbssd_avx512bw(acc1,_mm512_set1_epi32(iv[1]),bv);
                    acc2=dpbssd_avx512bw(acc2,_mm512_set1_epi32(iv[2]),bv);
                    acc3=dpbssd_avx512bw(acc3,_mm512_set1_epi32(iv[3]),bv);
                    acc4=dpbssd_avx512bw(acc4,_mm512_set1_epi32(iv[4]),bv);
                    acc5=dpbssd_avx512bw(acc5,_mm512_set1_epi32(iv[5]),bv);
                    acc6=dpbssd_avx512bw(acc6,_mm512_set1_epi32(iv[6]),bv);
                    acc7=dpbssd_avx512bw(acc7,_mm512_set1_epi32(iv[7]),bv);
                }
            }
#ifdef __AVX512VNNI__
            if (do_vnni) {
                alignas(64) int32_t corr[16] = {};
                for (int i = 0; i < nv; ++i) corr[i] = 128 * b_row_sums[n0 + i];
                __m512i vcorr = _mm512_load_si512(corr);
                acc0=_mm512_sub_epi32(acc0,vcorr); acc1=_mm512_sub_epi32(acc1,vcorr);
                acc2=_mm512_sub_epi32(acc2,vcorr); acc3=_mm512_sub_epi32(acc3,vcorr);
                acc4=_mm512_sub_epi32(acc4,vcorr); acc5=_mm512_sub_epi32(acc5,vcorr);
                acc6=_mm512_sub_epi32(acc6,vcorr); acc7=_mm512_sub_epi32(acc7,vcorr);
            }
#endif
            // Store raw int32 accumulators (single co_blk)
            __m512i* accs[8] = {&acc0,&acc1,&acc2,&acc3,&acc4,&acc5,&acc6,&acc7};
            for (int r = 0; r < 8; ++r) {
                int32_t* out_row = C + (size_t)(m + r) * N + n0;
                if (nv == 16) {
                    _mm512_storeu_si512(out_row, *accs[r]);
                } else {
                    alignas(64) int32_t tmp[16];
                    _mm512_store_si512(tmp, *accs[r]);
                    memcpy(out_row, tmp, nv * sizeof(int32_t));
                }
            }
        }
    }   // end 8-row mi loop

    // ── Row tail ──────────────────────────────────────────────────────────────
    for (int m_row = m8_c * 8; m_row < M; ++m_row) {
        const int8_t* a_row = A + (size_t)m_row * K;
        for (int co_blk = 0; co_blk < Co16; ++co_blk) {
            const int8_t* b_co = B_packed + (size_t)co_blk * K4 * 64;
            const int n0 = co_blk * 16;
            const int nv = std::min(16, N - n0);
            __m512i acc = _mm512_setzero_si512();
            for (int kb = 0; kb < k_full; ++kb) {
                int32_t iv; memcpy(&iv, a_row + kb * 4, 4);
#ifdef __AVX512VNNI__
                if (do_vnni) {
                    iv ^= XMASK_S32;
                    acc = _mm512_dpbusd_epi32(acc, _mm512_set1_epi32(iv),
                                                  _mm512_loadu_si512(b_co + kb * 64));
                } else
#endif
                {
                    acc = dpbssd_avx512bw(acc, _mm512_set1_epi32(iv),
                                              _mm512_loadu_si512(b_co + kb * 64));
                }
            }
            if (K & 3) {
                uint8_t t[4] = {};
                for (int j = 0; j < (K & 3); ++j) t[j] = (uint8_t)a_row[k_full*4+j];
                int32_t iv; memcpy(&iv, t, 4);
#ifdef __AVX512VNNI__
                if (do_vnni) {
                    iv ^= XMASK_S32;
                    acc = _mm512_dpbusd_epi32(acc, _mm512_set1_epi32(iv),
                                                  _mm512_loadu_si512(b_co + k_full * 64));
                } else
#endif
                {
                    acc = dpbssd_avx512bw(acc, _mm512_set1_epi32(iv),
                                              _mm512_loadu_si512(b_co + k_full * 64));
                }
            }
#ifdef __AVX512VNNI__
            if (do_vnni) {
                alignas(64) int32_t corr[16] = {};
                for (int i = 0; i < nv; ++i) corr[i] = 128 * b_row_sums[n0 + i];
                acc = _mm512_sub_epi32(acc, _mm512_load_si512(corr));
            }
#endif
            int32_t* out_row = C + (size_t)m_row * N + n0;
            if (nv == 16) {
                _mm512_storeu_si512(out_row, acc);
            } else {
                alignas(64) int32_t tmp[16];
                _mm512_store_si512(tmp, acc);
                memcpy(out_row, tmp, nv * sizeof(int32_t));
            }
        }
    }
}
#elif defined(__AVX2__)
{
    // ── AVX2: 8-row × 8-channel tile, raw int32 output (inline AVXK8) ─────────
    // Row pointers a0-a7 are computed once per mi (outside co_blk loop).
    // 4× K-loop unrolling reduces loop overhead and improves OOO scheduling.
    const int Co8   = (N + 7) / 8;
    const int m8_c  = M / 8;
    const int k_full = K / 4;
#if defined(__AVXVNNI__)
    const bool use_avx_vnni = (b_row_sums != nullptr);
#else
    constexpr bool use_avx_vnni = false;
#endif

    for (int mi = 0; mi < m8_c; ++mi) {
        const int m = mi * 8;
        const int8_t* a0 = A + (size_t)(m+0)*K;
        const int8_t* a1 = A + (size_t)(m+1)*K;
        const int8_t* a2 = A + (size_t)(m+2)*K;
        const int8_t* a3 = A + (size_t)(m+3)*K;
        const int8_t* a4 = A + (size_t)(m+4)*K;
        const int8_t* a5 = A + (size_t)(m+5)*K;
        const int8_t* a6 = A + (size_t)(m+6)*K;
        const int8_t* a7 = A + (size_t)(m+7)*K;
        for (int co_blk = 0; co_blk < Co8; ++co_blk) {
            const int8_t* b_co = B_packed + (size_t)co_blk * K4 * 32;
            const int n0 = co_blk * 8;
            const int nv = std::min(8, N - n0);
#if defined(__AVXVNNI__)
            alignas(32) int32_t corr_buf[8] = {};
            if (use_avx_vnni) {
                for (int i = 0; i < nv; ++i)
                    corr_buf[i] = 128 * b_row_sums[n0 + i];
            }
            const __m256i vcorr = _mm256_load_si256((const __m256i*)corr_buf);
#endif
            __m256i acc0=_mm256_setzero_si256(), acc1=_mm256_setzero_si256();
            __m256i acc2=_mm256_setzero_si256(), acc3=_mm256_setzero_si256();
            __m256i acc4=_mm256_setzero_si256(), acc5=_mm256_setzero_si256();
            __m256i acc6=_mm256_setzero_si256(), acc7=_mm256_setzero_si256();
            int kb = 0;
#if defined(__AVXVNNI__)
            if (use_avx_vnni) {
                for (; kb + 3 < k_full; kb += 4) {
                    AVXVNNI8(kb); AVXVNNI8(kb+1); AVXVNNI8(kb+2); AVXVNNI8(kb+3);
                }
                for (; kb < k_full; ++kb) { AVXVNNI8(kb); }
            } else
#endif
            {
            for (; kb + 3 < k_full; kb += 4) {
                AVXK8(kb); AVXK8(kb+1); AVXK8(kb+2); AVXK8(kb+3);
            }
            for (; kb < k_full; ++kb) { AVXK8(kb); }
            }
            if (K & 3) {
                const int8_t* btail = b_co + k_full * 32;
#if defined(__AVXVNNI__)
                if (use_avx_vnni) {
                    const __m128i _bp01 = _mm_loadu_si128((const __m128i*)btail);
                    const __m128i _bp23 = _mm_loadu_si128((const __m128i*)(btail + 16));
                    const __m128i _b_lo = _mm_unpacklo_epi16(_bp01, _bp23);
                    const __m128i _b_hi = _mm_unpackhi_epi16(_bp01, _bp23);
                    const __m256i bv = _mm256_inserti128_si256(_mm256_castsi128_si256(_b_lo), _b_hi, 1);
                    AVXVNNI8_TAIL_ROW(a0,acc0,bv); AVXVNNI8_TAIL_ROW(a1,acc1,bv);
                    AVXVNNI8_TAIL_ROW(a2,acc2,bv); AVXVNNI8_TAIL_ROW(a3,acc3,bv);
                    AVXVNNI8_TAIL_ROW(a4,acc4,bv); AVXVNNI8_TAIL_ROW(a5,acc5,bv);
                    AVXVNNI8_TAIL_ROW(a6,acc6,bv); AVXVNNI8_TAIL_ROW(a7,acc7,bv);
                } else
#endif
                {
                const __m256i btlo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)btail));
                const __m256i bthi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(btail+16)));
                AVXK8_TAIL(a0,acc0,btlo,bthi); AVXK8_TAIL(a1,acc1,btlo,bthi);
                AVXK8_TAIL(a2,acc2,btlo,bthi); AVXK8_TAIL(a3,acc3,btlo,bthi);
                AVXK8_TAIL(a4,acc4,btlo,bthi); AVXK8_TAIL(a5,acc5,btlo,bthi);
                AVXK8_TAIL(a6,acc6,btlo,bthi); AVXK8_TAIL(a7,acc7,btlo,bthi);
                }
            }
#if defined(__AVXVNNI__)
            if (use_avx_vnni) {
                acc0=_mm256_sub_epi32(acc0, vcorr); acc1=_mm256_sub_epi32(acc1, vcorr);
                acc2=_mm256_sub_epi32(acc2, vcorr); acc3=_mm256_sub_epi32(acc3, vcorr);
                acc4=_mm256_sub_epi32(acc4, vcorr); acc5=_mm256_sub_epi32(acc5, vcorr);
                acc6=_mm256_sub_epi32(acc6, vcorr); acc7=_mm256_sub_epi32(acc7, vcorr);
            }
#endif
            // Store directly to output (no kloop_out roundtrip)
            auto store8i = [&](__m256i a, int r) {
                int32_t* out_row = C + (size_t)(m+r)*N + n0;
                if (nv == 8) { _mm256_storeu_si256((__m256i*)out_row, a); }
                else { alignas(32) int32_t tmp[8]; _mm256_store_si256((__m256i*)tmp,a); memcpy(out_row,tmp,nv*4); }
            };
            store8i(acc0,0); store8i(acc1,1); store8i(acc2,2); store8i(acc3,3);
            store8i(acc4,4); store8i(acc5,5); store8i(acc6,6); store8i(acc7,7);
        }
    }   // end 8-row mi loop

    // ── Row tail (K-pair layout) ──────────────────────────────────────────────
    for (int m_row = m8_c * 8; m_row < M; ++m_row) {
        const int8_t* a_row = A + (size_t)m_row * K;
        for (int co_blk = 0; co_blk < Co8; ++co_blk) {
            const int8_t* b_co = B_packed + (size_t)co_blk * K4 * 32;
            const int n0 = co_blk * 8;
            const int nv = std::min(8, N - n0);
            __m256i acc = _mm256_setzero_si256();
#if defined(__AVXVNNI__)
            if (use_avx_vnni) {
                for (int kb = 0; kb < k_full; ++kb) {
                    int32_t iv; memcpy(&iv, a_row + kb * 4, 4);
                    const __m128i _bp01 = _mm_loadu_si128((const __m128i*)(b_co + kb * 32));
                    const __m128i _bp23 = _mm_loadu_si128((const __m128i*)(b_co + kb * 32 + 16));
                    const __m128i _b_lo = _mm_unpacklo_epi16(_bp01, _bp23);
                    const __m128i _b_hi = _mm_unpackhi_epi16(_bp01, _bp23);
                    const __m256i bv = _mm256_inserti128_si256(_mm256_castsi128_si256(_b_lo), _b_hi, 1);
                    acc = _mm256_dpbusd_epi32(acc, _mm256_set1_epi32(iv ^ 0x80808080), bv);
                }
                if (K & 3) {
                    uint8_t t[4] = {};
                    for (int j = 0; j < (K & 3); ++j) t[j] = (uint8_t)a_row[k_full*4+j];
                    int32_t iv; memcpy(&iv, t, 4);
                    const int8_t* btail = b_co + k_full * 32;
                    const __m128i _bp01 = _mm_loadu_si128((const __m128i*)btail);
                    const __m128i _bp23 = _mm_loadu_si128((const __m128i*)(btail + 16));
                    const __m128i _b_lo = _mm_unpacklo_epi16(_bp01, _bp23);
                    const __m128i _b_hi = _mm_unpackhi_epi16(_bp01, _bp23);
                    const __m256i bv = _mm256_inserti128_si256(_mm256_castsi128_si256(_b_lo), _b_hi, 1);
                    acc = _mm256_dpbusd_epi32(acc, _mm256_set1_epi32(iv ^ 0x80808080), bv);
                }
                alignas(32) int32_t corr_buf[8] = {};
                for (int i = 0; i < nv; ++i)
                    corr_buf[i] = 128 * b_row_sums[n0 + i];
                acc = _mm256_sub_epi32(acc, _mm256_load_si256((const __m256i*)corr_buf));
            } else
#endif
            {
            for (int kb = 0; kb < k_full; ++kb) {
                int32_t iv; memcpy(&iv, a_row + kb * 4, 4);
                const __m256i bvlo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b_co + kb * 32)));
                const __m256i bvhi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b_co + kb * 32 + 16)));
                const int32_t p01 = (uint16_t)(int8_t)iv | ((uint32_t)(uint16_t)(int8_t)(iv >> 8) << 16);
                const int32_t p23 = (uint16_t)(int8_t)(iv >> 16) | ((uint32_t)(uint16_t)(int8_t)(iv >> 24) << 16);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_set1_epi32(p01), bvlo));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_set1_epi32(p23), bvhi));
            }
            if (K & 3) {
                uint8_t t[4] = {};
                for (int j = 0; j < (K & 3); ++j) t[j] = (uint8_t)a_row[k_full*4+j];
                int32_t iv; memcpy(&iv, t, 4);
                const __m256i bvlo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b_co + k_full * 32)));
                const __m256i bvhi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b_co + k_full * 32 + 16)));
                const int32_t p01 = (uint16_t)(int8_t)iv | ((uint32_t)(uint16_t)(int8_t)(iv >> 8) << 16);
                const int32_t p23 = (uint16_t)(int8_t)(iv >> 16) | ((uint32_t)(uint16_t)(int8_t)(iv >> 24) << 16);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_set1_epi32(p01), bvlo));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_set1_epi32(p23), bvhi));
            }
            }
            int32_t* out_row = C + (size_t)m_row * N + n0;
            if (nv == 8) {
                _mm256_storeu_si256((__m256i*)out_row, acc);
            } else {
                alignas(32) int32_t tmp[8];
                _mm256_store_si256((__m256i*)tmp, acc);
                memcpy(out_row, tmp, nv * sizeof(int32_t));
            }
        }
    }
}
#else
    // Scalar fallback (no SIMD)
    for (int m = 0; m < M; ++m) {
        const int8_t* a_row = A + (size_t)m * K;
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            int co_blk = n / 4, oc = n % 4;
            for (int k = 0; k < K; ++k) {
                int k_blk = k / 4, ki = k % 4;
                acc += (int32_t)a_row[k] *
                       (int32_t)B_packed[(co_blk * K4 + k_blk) * 16 + oc * 4 + ki];
            }
            C[m*N + n] = acc;
        }
    }
#endif
}

void gemm_int8_dequant_accum(
    const int8_t*  A,
    const int8_t*  B_packed,
    float*         C,
    int M, int K, int N,
    const float*   w_scales,
    const float*   row_scales,
    const float*   bias,
    bool           accumulate,
    const int32_t* b_row_sums,
    StreamHandle   /* stream */)
{
    const int K4 = (K + 3) / 4;

#if defined(__AVX2__) && !defined(__AVX512BW__)
    const int Co8 = (N + 7) / 8;
    const int m8_c = M / 8;
    const int k_full = K / 4;
#if defined(__AVXVNNI__)
    const bool use_avx_vnni = (b_row_sums != nullptr);
#else
    constexpr bool use_avx_vnni = false;
#endif

    auto store_float = [&](const __m256i acc, int m_row, int n0, int nv,
                           const __m256 vscale, const __m256 vbias) {
        __m256 vf = _mm256_mul_ps(_mm256_cvtepi32_ps(acc),
                                  _mm256_mul_ps(vscale, _mm256_set1_ps(row_scales[m_row])));
        float* out = C + (size_t)m_row * N + n0;
        if (accumulate) {
            vf = _mm256_add_ps(vf, _mm256_loadu_ps(out));
        } else if (bias) {
            vf = _mm256_add_ps(vf, vbias);
        }
        if (nv == 8) {
            _mm256_storeu_ps(out, vf);
        } else {
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, vf);
            memcpy(out, tmp, (size_t)nv * sizeof(float));
        }
    };

    for (int mi = 0; mi < m8_c; ++mi) {
        const int m = mi * 8;
        const int8_t* a0 = A + (size_t)(m+0)*K;
        const int8_t* a1 = A + (size_t)(m+1)*K;
        const int8_t* a2 = A + (size_t)(m+2)*K;
        const int8_t* a3 = A + (size_t)(m+3)*K;
        const int8_t* a4 = A + (size_t)(m+4)*K;
        const int8_t* a5 = A + (size_t)(m+5)*K;
        const int8_t* a6 = A + (size_t)(m+6)*K;
        const int8_t* a7 = A + (size_t)(m+7)*K;

        for (int co_blk = 0; co_blk < Co8; ++co_blk) {
            const int8_t* b_co = B_packed + (size_t)co_blk * K4 * 32;
            const int n0 = co_blk * 8;
            const int nv = std::min(8, N - n0);
            const __m256 vscale = _mm256_loadu_ps(w_scales + n0);
            const __m256 vbias = bias ? _mm256_loadu_ps(bias + n0) : _mm256_setzero_ps();
#if defined(__AVXVNNI__)
            alignas(32) int32_t corr_buf[8] = {};
            if (use_avx_vnni) {
                for (int i = 0; i < nv; ++i)
                    corr_buf[i] = 128 * b_row_sums[n0 + i];
            }
            const __m256i vcorr = _mm256_load_si256((const __m256i*)corr_buf);
#endif
            __m256i acc0=_mm256_setzero_si256(), acc1=_mm256_setzero_si256();
            __m256i acc2=_mm256_setzero_si256(), acc3=_mm256_setzero_si256();
            __m256i acc4=_mm256_setzero_si256(), acc5=_mm256_setzero_si256();
            __m256i acc6=_mm256_setzero_si256(), acc7=_mm256_setzero_si256();
            int kb = 0;
#if defined(__AVXVNNI__)
            if (use_avx_vnni) {
                for (; kb + 3 < k_full; kb += 4) {
                    AVXVNNI8(kb); AVXVNNI8(kb+1); AVXVNNI8(kb+2); AVXVNNI8(kb+3);
                }
                for (; kb < k_full; ++kb) { AVXVNNI8(kb); }
            } else
#endif
            {
                for (; kb + 3 < k_full; kb += 4) {
                    AVXK8(kb); AVXK8(kb+1); AVXK8(kb+2); AVXK8(kb+3);
                }
                for (; kb < k_full; ++kb) { AVXK8(kb); }
            }
            if (K & 3) {
                const int8_t* btail = b_co + k_full * 32;
#if defined(__AVXVNNI__)
                if (use_avx_vnni) {
                    const __m128i _bp01 = _mm_loadu_si128((const __m128i*)btail);
                    const __m128i _bp23 = _mm_loadu_si128((const __m128i*)(btail + 16));
                    const __m128i _b_lo = _mm_unpacklo_epi16(_bp01, _bp23);
                    const __m128i _b_hi = _mm_unpackhi_epi16(_bp01, _bp23);
                    const __m256i bv = _mm256_inserti128_si256(_mm256_castsi128_si256(_b_lo), _b_hi, 1);
                    AVXVNNI8_TAIL_ROW(a0,acc0,bv); AVXVNNI8_TAIL_ROW(a1,acc1,bv);
                    AVXVNNI8_TAIL_ROW(a2,acc2,bv); AVXVNNI8_TAIL_ROW(a3,acc3,bv);
                    AVXVNNI8_TAIL_ROW(a4,acc4,bv); AVXVNNI8_TAIL_ROW(a5,acc5,bv);
                    AVXVNNI8_TAIL_ROW(a6,acc6,bv); AVXVNNI8_TAIL_ROW(a7,acc7,bv);
                } else
#endif
                {
                    const __m256i btlo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)btail));
                    const __m256i bthi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(btail+16)));
                    AVXK8_TAIL(a0,acc0,btlo,bthi); AVXK8_TAIL(a1,acc1,btlo,bthi);
                    AVXK8_TAIL(a2,acc2,btlo,bthi); AVXK8_TAIL(a3,acc3,btlo,bthi);
                    AVXK8_TAIL(a4,acc4,btlo,bthi); AVXK8_TAIL(a5,acc5,btlo,bthi);
                    AVXK8_TAIL(a6,acc6,btlo,bthi); AVXK8_TAIL(a7,acc7,btlo,bthi);
                }
            }
#if defined(__AVXVNNI__)
            if (use_avx_vnni) {
                acc0=_mm256_sub_epi32(acc0, vcorr); acc1=_mm256_sub_epi32(acc1, vcorr);
                acc2=_mm256_sub_epi32(acc2, vcorr); acc3=_mm256_sub_epi32(acc3, vcorr);
                acc4=_mm256_sub_epi32(acc4, vcorr); acc5=_mm256_sub_epi32(acc5, vcorr);
                acc6=_mm256_sub_epi32(acc6, vcorr); acc7=_mm256_sub_epi32(acc7, vcorr);
            }
#endif
            store_float(acc0, m+0, n0, nv, vscale, vbias);
            store_float(acc1, m+1, n0, nv, vscale, vbias);
            store_float(acc2, m+2, n0, nv, vscale, vbias);
            store_float(acc3, m+3, n0, nv, vscale, vbias);
            store_float(acc4, m+4, n0, nv, vscale, vbias);
            store_float(acc5, m+5, n0, nv, vscale, vbias);
            store_float(acc6, m+6, n0, nv, vscale, vbias);
            store_float(acc7, m+7, n0, nv, vscale, vbias);
        }
    }

    for (int m_row = m8_c * 8; m_row < M; ++m_row) {
        const int8_t* a_row = A + (size_t)m_row * K;
        for (int co_blk = 0; co_blk < Co8; ++co_blk) {
            const int8_t* b_co = B_packed + (size_t)co_blk * K4 * 32;
            const int n0 = co_blk * 8;
            const int nv = std::min(8, N - n0);
            const __m256 vscale = _mm256_loadu_ps(w_scales + n0);
            const __m256 vbias = bias ? _mm256_loadu_ps(bias + n0) : _mm256_setzero_ps();
            __m256i acc = _mm256_setzero_si256();
#if defined(__AVXVNNI__)
            if (use_avx_vnni) {
                for (int kb = 0; kb < k_full; ++kb) {
                    int32_t iv; memcpy(&iv, a_row + kb * 4, 4);
                    const __m128i _bp01 = _mm_loadu_si128((const __m128i*)(b_co + kb * 32));
                    const __m128i _bp23 = _mm_loadu_si128((const __m128i*)(b_co + kb * 32 + 16));
                    const __m128i _b_lo = _mm_unpacklo_epi16(_bp01, _bp23);
                    const __m128i _b_hi = _mm_unpackhi_epi16(_bp01, _bp23);
                    const __m256i bv = _mm256_inserti128_si256(_mm256_castsi128_si256(_b_lo), _b_hi, 1);
                    acc = _mm256_dpbusd_epi32(acc, _mm256_set1_epi32(iv ^ 0x80808080), bv);
                }
                if (K & 3) {
                    uint8_t t[4] = {};
                    for (int j = 0; j < (K & 3); ++j) t[j] = (uint8_t)a_row[k_full*4+j];
                    int32_t iv; memcpy(&iv, t, 4);
                    const int8_t* btail = b_co + k_full * 32;
                    const __m128i _bp01 = _mm_loadu_si128((const __m128i*)btail);
                    const __m128i _bp23 = _mm_loadu_si128((const __m128i*)(btail + 16));
                    const __m128i _b_lo = _mm_unpacklo_epi16(_bp01, _bp23);
                    const __m128i _b_hi = _mm_unpackhi_epi16(_bp01, _bp23);
                    const __m256i bv = _mm256_inserti128_si256(_mm256_castsi128_si256(_b_lo), _b_hi, 1);
                    acc = _mm256_dpbusd_epi32(acc, _mm256_set1_epi32(iv ^ 0x80808080), bv);
                }
                alignas(32) int32_t corr_buf[8] = {};
                for (int i = 0; i < nv; ++i)
                    corr_buf[i] = 128 * b_row_sums[n0 + i];
                acc = _mm256_sub_epi32(acc, _mm256_load_si256((const __m256i*)corr_buf));
            } else
#endif
            {
                for (int kb = 0; kb < k_full; ++kb) {
                    int32_t iv; memcpy(&iv, a_row + kb * 4, 4);
                    const __m256i bvlo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b_co + kb * 32)));
                    const __m256i bvhi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b_co + kb * 32 + 16)));
                    const int32_t p01 = (uint16_t)(int8_t)iv | ((uint32_t)(uint16_t)(int8_t)(iv >> 8) << 16);
                    const int32_t p23 = (uint16_t)(int8_t)(iv >> 16) | ((uint32_t)(uint16_t)(int8_t)(iv >> 24) << 16);
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_set1_epi32(p01), bvlo));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_set1_epi32(p23), bvhi));
                }
                if (K & 3) {
                    uint8_t t[4] = {};
                    for (int j = 0; j < (K & 3); ++j) t[j] = (uint8_t)a_row[k_full*4+j];
                    int32_t iv; memcpy(&iv, t, 4);
                    const __m256i bvlo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b_co + k_full * 32)));
                    const __m256i bvhi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b_co + k_full * 32 + 16)));
                    const int32_t p01 = (uint16_t)(int8_t)iv | ((uint32_t)(uint16_t)(int8_t)(iv >> 8) << 16);
                    const int32_t p23 = (uint16_t)(int8_t)(iv >> 16) | ((uint32_t)(uint16_t)(int8_t)(iv >> 24) << 16);
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_set1_epi32(p01), bvlo));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_set1_epi32(p23), bvhi));
                }
            }
            store_float(acc, m_row, n0, nv, vscale, vbias);
        }
    }
#else
    std::vector<int32_t> tmp((size_t)M * N);
    gemm_int8_int32(A, B_packed, tmp.data(), M, K, N, b_row_sums, nullptr);
    for (int m = 0; m < M; ++m) {
        float* row = C + (size_t)m * N;
        const int32_t* acc_row = tmp.data() + (size_t)m * N;
        const float rs = row_scales[m];
        for (int n = 0; n < N; ++n) {
            const float v = (float)acc_row[n] * w_scales[n] * rs;
            if (accumulate) row[n] += v;
            else row[n] = v + (bias ? bias[n] : 0.0f);
        }
    }
#endif
}

// ──────────────────────────────────────────────────────────────
// Winograd F(2,3) for 3x3 stride=1 pad=1 INT8 convolutions (NHWC).
//
// Input/output are in NHWC layout: input[h][w][c] at offset (h*W+w)*C+c
// The transform domain GEMMs use the same SDOT kernel.
// ──────────────────────────────────────────────────────────────
