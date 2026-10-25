#pragma once

#if defined(_MSC_VER) && !defined(__clang__)
#  ifndef __attribute__
#    define __attribute__(x)
#  endif
#  ifndef __builtin_expect
#    define __builtin_expect(expr, expected) (expr)
#  endif
#  ifndef __builtin_prefetch
#    if defined(_M_IX86) || defined(_M_X64)
#      include <xmmintrin.h>
#      define __builtin_prefetch(addr, rw, locality) \
          _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#    else
#      define __builtin_prefetch(addr, rw, locality) ((void)0)
#    endif
#  endif
#endif

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>

// StreamHandle — opaque async-stream token (always nullptr on CPU backend).
// Guard against redefinition when backend.hpp is also included.
#ifndef STREAM_HANDLE_DEFINED
#define STREAM_HANDLE_DEFINED
using StreamHandle = void*;
#endif

// OpenMP compatibility shim — when OpenMP is absent the runtime calls are
// replaced with trivial stubs so all translation units compile cleanly.
#ifdef _OPENMP
#  include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num()  { return 0; }
inline int omp_in_parallel()     { return 0; }
#endif

// ──────────────────────────────────────────────
// QuantizeMultiplier: decompose float scale into Q31 fixed-point.
//   scale = mult * 2^(exp - 31)
//   where mult ∈ [2^30, 2^31-1]  (Q31 normalised mantissa)
//   and   exp  = frexp exponent (≤ 0 for scale < 1)
//
// Apply with: vqrdmulhq_s32(acc, mult) then vrshlq_s32(result, exp)
//   (vrshlq with exp < 0 → rounding right-shift by |exp|)
// ──────────────────────────────────────────────
inline void QuantizeMultiplier(float scale, int32_t* mult, int32_t* exp) {
    int e;
    double m = std::frexp((double)scale, &e);
    int64_t q = static_cast<int64_t>(std::round(m * (double)(1LL << 31)));
    if (q == (1LL << 31)) { q >>= 1; ++e; }   // rounding overflow: renormalise
    *mult = static_cast<int32_t>(q);
    *exp  = e;   // ≤ 0 for scale < 1 → vrshlq right-shifts by |e|
}

// ──────────────────────────────────────────────
// apply_q31_scalar: integer equivalent of round(x * scale).
//   Uses single int64 multiply + rounding right-shift (no double rounding).
// ──────────────────────────────────────────────
inline int32_t apply_q31_scalar(int32_t x, int32_t mult, int32_t exp) {
    int64_t prod = static_cast<int64_t>(x) * mult;
    // total right-shift = 31 - exp  (positive since exp ≤ 31 for any practical scale)
    int rshift = 31 - exp;
    return static_cast<int32_t>((prod + (1LL << (rshift - 1))) >> rshift);
}

// ──────────────────────────────────────────────
// requant_fixedpoint: pure-integer requantisation.
//   Replaces the float-based requant(). No float ops.
// ──────────────────────────────────────────────
inline int8_t requant_fixedpoint(int32_t acc, int32_t mult, int32_t exp, int out_zp) {
    int32_t q = apply_q31_scalar(acc, mult, exp) + out_zp;
    return static_cast<int8_t>(std::clamp(q, -128, 127));
}

// ──────────────────────────────────────────────
// Repack weights for SDOT: [C_out, K] → [C_out/4, K/4, 4, 4]
// Handles padding when C_out or K are not multiples of 4.
// Caller must free returned pointer (allocated with new[]).
// ──────────────────────────────────────────────
int8_t* pack_weights_sdot(const int8_t* w, int C_out, int K);
void    free_packed(int8_t* p);

// ──────────────────────────────────────────────
// INT8 GEMM with NEON SDOT (for 1×1 conv / FC)
//   A: [M, K]  (int8, row-major)
//   B: packed  [N/4, K/4, 4, 4]   where N = C_out
//   C: [M, N]  (float32 output or int8 output depending on is_float)
//
// For int8 output (is_float=false):
//   out[m,n] = clamp(apply_q31(acc+eff_bias, req_mult[n], req_exp[n]) + out_zp, -128, 127)
//   req_mult/req_exp: Q31 fixed-point params from QuantizeMultiplier — no float in hot path.
//
// For float output (is_float=true, final FC layer only):
//   out[m,n] = (float)(acc + eff_bias[n]) * req_scale_f[n]
//   req_scale_f must be non-null; req_mult/req_exp are unused.
// ──────────────────────────────────────────────
void gemm_int8(
    const int8_t*  A,             // [M, K]
    const int8_t*  B_packed,      // packed [N/4, K/4, 4, 4]
    const int64_t* eff_bias,      // [N]
    const int32_t* req_mult,      // [N]  Q31 multipliers   (used when is_float=false)
    const int32_t* req_exp,       // [N]  frexp exponents   (used when is_float=false)
    const float*   req_scale_f,   // [N]  float scales      (used when is_float=true)
    int8_t         out_zp,
    void*          C,             // [M, N] row-major, or [N, M] if nchw_out
    bool           is_float,
    int M, int K, int N,
    bool           nchw_out    = false,
    bool           in_parallel = false,
    const int32_t* b_row_sums  = nullptr,   // [N] weight row sums for AVX-512 VNNI s8->u8 correction
    StreamHandle   stream      = nullptr
);

// ──────────────────────────────────────────────
// Direct int8 conv2d (NCHW layout).
// Handles any kernel size; uses im2col + GEMM internally.
// w_pre_packed: optional pre-packed weights (groups=1 only); pass nullptr to auto-pack.
// in_zp: input zero point used for padding (MUST match layer's in_zp).
// req_mult/req_exp: Q31 fixed-point requantisation params (from QuantizeMultiplier).
// ──────────────────────────────────────────────
void conv2d_int8(
    const int8_t*  input,         // [1, C_in, H, W] NCHW  or [H, W, C_in] NHWC
    const int8_t*  weight,        // [C_out, C_in, kH, kW]
    const int8_t*  w_pre_packed,  // pre-packed (may be nullptr)
    const int64_t* eff_bias,      // [C_out]
    const int32_t* req_mult,      // [C_out]  Q31 multipliers
    const int32_t* req_exp,       // [C_out]  frexp exponents
    int8_t         in_zp,         // input zero point (padding value)
    int8_t         out_zp,
    int8_t*        output,        // [1, C_out, oH, oW] NCHW  or [oH, oW, C_out] NHWC
    int C_in, int H, int W,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int groups = 1,
    int8_t* scratch_col = nullptr, // [oHW * K] pre-allocated (nullptr -> internal alloc)
    bool nhwc = false,             // true: input/output are NHWC layout
    const int32_t* b_row_sums = nullptr,  // [C_out] weight row sums for VNNI correction
    StreamHandle   stream = nullptr
);

// ──────────────────────────────────────────────
// INT8 GEMM -> INT32 accumulators (no requantization).
// Used for Winograd domain GEMMs where outputs must stay INT32
// for the inverse transform + fused requantization step.
//   A: [M, K]  (int8, row-major)
//   B: packed  [N/4, K/4, 4, 4]   where N = C_out
//   C: [M, N]  (int32 output, row-major)
// ──────────────────────────────────────────────
void gemm_int8_int32(
    const int8_t*  A,             // [M, K]
    const int8_t*  B_packed,      // packed [N/4, K/4, 4, 4]
    int32_t*       C,             // [M, N] row-major (INT32 accumulators)
    int M, int K, int N,
    const int32_t* b_row_sums  = nullptr,   // [N] weight row sums for AVX-512 VNNI s8->u8 correction
    StreamHandle   stream      = nullptr
);

// INT8 GEMM fused with FP32 dequantization and optional accumulation.
// Used by Conv1dInt8: C[m,n] = acc(A*B) * w_scales[n] * row_scales[m]
// plus bias on the first tap, or plus the existing C value on later taps.
void gemm_int8_dequant_accum(
    const int8_t*  A,             // [M, K]
    const int8_t*  B_packed,      // packed [N/4, K/4, 4, 4]
    float*         C,             // [M, N] row-major FP32
    int M, int K, int N,
    const float*   w_scales,      // [N]
    const float*   row_scales,    // [M]
    const float*   bias,          // [N], nullable, used when accumulate=false
    bool           accumulate,
    const int32_t* b_row_sums = nullptr,
    StreamHandle   stream = nullptr
);

// ──────────────────────────────────────────────
// Winograd F(2,3) convolution for 3x3 stride=1 pad=1 (NHWC).
// Uses pre-transformed weights (w_wino_packed[16]) and INT8 SDOT GEMMs.
// Input/output are NHWC layout.
// ──────────────────────────────────────────────
void conv2d_winograd_nhwc_int8(
    const int8_t*  input,              // [H, W, C_in] NHWC
    const int8_t*  const* w_wino_packed,  // [16] packed weight arrays
    const float*   wino_weight_scale,  // [16] per-position weight scales
    float          wino_input_scale,   // typically 0.25
    const int64_t* eff_bias,           // [C_out]
    const int32_t* req_mult,           // [C_out]  Q31 multipliers
    const int32_t* req_exp,            // [C_out]  frexp exponents
    const float*   req_scale,          // [C_out]  float scales
    int8_t         in_zp,
    int8_t         out_zp,
    int8_t*        output,             // [oH, oW, C_out] NHWC
    int C_in, int H, int W,
    int C_out,
    int8_t*  scratch_input_hat = nullptr,  // [16 * num_tiles * C_in]
    int32_t* scratch_output_hat = nullptr, // [16 * num_tiles * C_out]
    const int32_t* const* w_wino_row_sums = nullptr,  // [16] pointers to [C_out] row sums
    StreamHandle   stream = nullptr
);

// Final FC layer: output is float32
void linear_int8_to_float(
    const int8_t*  input,      // [C_in]
    const int8_t*  weight,     // [C_out, C_in]
    const int64_t* eff_bias,   // [C_out]
    const float*   req_scale,  // [C_out]
    float*         output,     // [C_out]
    int C_in, int C_out
);

// ──────────────────────────────────────────────
// MaxPool int8 (NCHW)
// ──────────────────────────────────────────────
void maxpool_int8(
    const int8_t* input,   // [1, C, H, W]
    int8_t*       output,  // [1, C, oH, oW]
    int C, int H, int W,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    StreamHandle stream = nullptr
);

// ──────────────────────────────────────────────
// MaxPool int8 (NHWC)
// ──────────────────────────────────────────────
void maxpool_int8_nhwc(
    const int8_t* input,   // [H, W, C]
    int8_t*       output,  // [oH, oW, C]
    int C, int H, int W,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    StreamHandle stream = nullptr
);

// ──────────────────────────────────────────────
// Global average pool int8 → int8 (NCHW)
// ──────────────────────────────────────────────
void avgpool_global_int8(
    const int8_t* input,       // [1, C, H, W]
    float         in_scale,
    int           in_zp,
    float         out_scale,
    int           out_zp,
    int8_t*       output,      // [C]
    int C, int H, int W,
    StreamHandle  stream = nullptr
);

// ──────────────────────────────────────────────
// Global average pool int8 → int8 (NHWC)
// ──────────────────────────────────────────────
void avgpool_global_int8_nhwc(
    const int8_t* input,       // [H, W, C]
    float         in_scale,
    int           in_zp,
    float         out_scale,
    int           out_zp,
    int8_t*       output,      // [C]
    int C, int H, int W,
    StreamHandle  stream = nullptr
);

// ──────────────────────────────────────────────
// Native NHWC 3×3 depthwise INT8 convolution.
// Avoids NHWC↔NCHW transpositions; operates end-to-end in NHWC layout.
// w_hwc:    weights in [9, C] tap-major order (pre-transposed at load time).
// eff_b32:  eff_bias truncated to int32 (pre-computed at load time).
// stride:   1 or 2 (only stride=1 and stride=2 are accelerated).
// oH, oW:  output spatial dimensions (needed for stride=2).
// ──────────────────────────────────────────────
void conv2d_depthwise_nhwc_int8(
    const int8_t*  input,     // [H, W, C] NHWC
    int8_t*        output,    // [oH, oW, C] NHWC
    int H, int W, int C,
    int oH, int oW,
    int stride,
    int8_t         in_zp,
    const int8_t*  w_hwc,    // [9, C] tap-major weights
    const int32_t* eff_b32,  // [C] truncated eff_bias
    const int32_t* req_mult, // [C]
    const int32_t* req_exp,  // [C]
    int8_t         out_zp,
    int            nthreads = 0,  // 0 = use omp_get_max_threads()
    StreamHandle   stream   = nullptr
);

// ──────────────────────────────────────────────
// Grouped NHWC depthwise fallback: NHWC→NCHW → conv2d_int8 → NCHW→NHWC.
// Handles grouped convolutions that the native NHWC kernel cannot (non-3×3,
// dilated, etc.).  TLS scratch in ops avoids per-call malloc.
// ──────────────────────────────────────────────
void conv2d_grouped_nhwc_fallback_int8(
    const int8_t*  input,
    const int8_t*  weight,
    const int64_t* eff_bias,
    const int32_t* req_mult,
    const int32_t* req_exp,
    int8_t         in_zp,
    int8_t         out_zp,
    int8_t*        output,
    int C, int H, int W,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int groups,
    int8_t*        scratch_col  = nullptr,
    const int32_t* w_row_sums   = nullptr,
    StreamHandle   stream       = nullptr
);

// ──────────────────────────────────────────────
// Op profiling (call reset before benchmark, print after)
// ──────────────────────────────────────────────
struct OpsProfileEntry {
    const char* name;
    double      total_ms;
};

void ops_profile_reset();
void ops_profile_set_enabled(bool enabled);
bool ops_profile_enabled();
void ops_profile_print(int runs);
const OpsProfileEntry* ops_profile_entries(int* count);

// ──────────────────────────────────────────────
// FP32 weight packing  (mirrors INT8 pack_weights_sdot)
//   w:       [C_out, K]  (K = C_in * kH * kW, NHWC-reordered for kH>1)
//   returns: [Co_t, K, TILE]  where TILE = 16 (AVX-512) / 8 (AVX2) / 4 (NEON/scalar)
// Caller must free with free_packed_f32().
// ──────────────────────────────────────────────
float* pack_weights_f32(const float* w, int C_out, int K);
void   free_packed_f32(float* p);
// Returns the number of floats in the packed buffer for (C_out, K).
size_t packed_f32_elems(int C_out, int K);

// ──────────────────────────────────────────────
// Merged conv1d weight pack for kfused GEMM.
//   w:   [C_out, kernel_size, C_in_g]  (NHWC, after reorder)
//   →    [Co_t, kernel_size, C_in_g, TILE]
//        offset = (co_blk * kernel_size + kp) * C_in_g * TILE
// Caller must free with free_packed_f32().
// ──────────────────────────────────────────────
float* pack_merged_weights_f32(const float* w, int C_out, int C_in_g, int kernel_size);

// ──────────────────────────────────────────────
// Fused conv1d K-split GEMM: accumulates all kernel positions in a single
// pass, keeping partial sums in SIMD registers.  C is written ONCE per
// output element.
//   A_base:    padded input [T_padded, C_in]
//   B_merged:  [Co_t, kernel_size, C_in, TILE]  (from pack_merged_weights_f32)
//   C:         output [M, C_out]  (M = T_out)
// ──────────────────────────────────────────────
void conv1d_kfused_sgemm_f32(
    const float* A_base,
    const float* B_merged,
    const float* bias,
    float*       C,
    bool         relu,
    int M, int C_in, int C_out,
    int kernel_size, int dilation,
    bool in_parallel = false
);

// ──────────────────────────────────────────────
// FP32 SGEMM
//   A:        [M, K]           row-major float
//   B_packed: [Co_t, K, TILE]  packed floats
//   bias:     [N]  or nullptr
//   C:        [M, N]           row-major float (output)
//   relu:     apply max(0, ·) after bias add
//
// Recursive OMP dispatch: call with in_parallel=false outside a
// parallel region; set in_parallel=true when already inside a team.
// ──────────────────────────────────────────────
void sgemm_f32(
    const float* A,
    const float* B_packed,
    const float* bias,
    float*       C,
    bool         relu,
    int M, int K, int N,
    bool in_parallel = false,
    StreamHandle stream = nullptr
);

// ──────────────────────────────────────────────
// FP32 kernels (NHWC layout, SIMD-accelerated)
// ──────────────────────────────────────────────
void conv2d_fp32_nhwc(
    const float* input,        // [H, W, C_in] NHWC
    const float* weight,       // [C_out, C_in, kH, kW] NCHW kernel ordering
    const float* bias,         // [C_out]  (may be nullptr)
    float*       output,       // [oH, oW, C_out] NHWC
    int C_in, int H, int W,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    bool relu = false,
    const float* w_packed    = nullptr,  // pre-packed [Co_t, K, TILE]
    float*       scratch_col = nullptr,  // [oHW * K] floats  (alloc internally if nullptr)
    int dilation_h = 1, int dilation_w = 1,
    StreamHandle stream = nullptr
);

// ──────────────────────────────────────────────
// FP32 Winograd conv1d: F(4,3), F(2,7), F(2,11)
//   in_buf:   [T, C_in]  (input signal, no padding)
//   w_packed: array of packed weight pointers (6 / 8 / 12 entries)
//   bias:     [C_out] or nullptr
//   out_buf:  [T_out, C_out]
//   scratch:  workspace  (see scratch_bytes in layer.hpp for sizing)
// ──────────────────────────────────────────────
void conv1d_wino_f43_fp32(
    const float* in_buf,
    const float* const* w_packed,
    const float* bias, bool relu,
    int T, int T_out, int C_in, int C_out, int padding,
    float* out_buf, float* scratch);

void conv1d_wino_f27_fp32(
    const float* in_buf,
    const float* const* w_packed,
    const float* bias, bool relu,
    int T, int T_out, int C_in, int C_out, int padding,
    float* out_buf, float* scratch);

void conv1d_wino_f211_fp32(
    const float* in_buf,
    const float* const* w_packed,
    const float* bias, bool relu,
    int T, int T_out, int C_in, int C_out, int padding,
    float* out_buf, float* scratch);

// ──────────────────────────────────────────────
// FP32 Winograd F(2,3) for 3×3 stride=1 pad=1 convolutions (NHWC).
// Reduces DRAM-bound direct GEMM to 16 cache-friendly GEMMs.
//   scratch: >= 16 * ceil(H/2)*ceil(W/2) * (C_in + C_out) floats
// ──────────────────────────────────────────────
void conv2d_winograd_nhwc_fp32(
    const float* input,                 // [H, W, C_in] NHWC
    const float* const* w_wino_packed,  // [16] packed [Co_t, C_in, TILE]
    const float* bias,                  // [C_out] or nullptr
    float*       output,                // [H, W, C_out] NHWC
    int C_in, int H, int W, int C_out, bool relu,
    float* scratch,
    StreamHandle stream = nullptr,
    const float* raw_weight = nullptr   // ignored on CPU
);

void gemm_fp32_vec(
    const float* input,        // [C_in]
    const float* weight,       // [C_out, C_in]
    const float* bias,         // [C_out]  (may be nullptr)
    float*       output,       // [C_out]
    int C_in, int C_out,
    const float* w_packed = nullptr,    // pre-packed [Co_t, K, TILE]
    StreamHandle stream   = nullptr
);

void add_fp32(
    const float* in1,      // [N]
    const float* in2,      // [N]
    float*       output,   // [N]
    int N,
    bool relu = false,
    StreamHandle stream = nullptr
);

void maxpool_fp32_nhwc(
    const float* input,    // [H, W, C]
    float*       output,   // [oH, oW, C]
    int C, int H, int W,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    StreamHandle stream = nullptr
);

void avgpool_global_fp32_nhwc(
    const float* input,    // [H, W, C]
    float*       output,   // [C]
    int C, int H, int W,
    StreamHandle stream = nullptr
);

// ──────────────────────────────────────────────
// Residual Add with requantization (NCHW)
// ──────────────────────────────────────────────
void add_requant_int8(
    const int8_t* in1,      // [C, H, W]
    const int8_t* in2,      // [C, H, W]
    float in1_scale, int in1_zp,
    float in2_scale, int in2_zp,
    float out_scale, int out_zp,
    int8_t*       output,   // [C, H, W]
    int N,
    StreamHandle  stream = nullptr
);

// ──────────────────────────────────────────────
// ViT FP32 ops  (sequences + attention)
// ──────────────────────────────────────────────

// Layer Normalisation  [N, C] → [N, C]
void layernorm_fp32(
    const float* input,   // [N, C]
    const float* gamma,   // [C]
    const float* beta,    // [C]
    float*       output,  // [N, C]
    int N, int C,
    float eps = 1e-6f,
    StreamHandle stream = nullptr
);

// Patch Prep: flatten spatial input → prepend CLS → add pos_embed
//   patches:   [N_patches, D]  (conv output, NHWC-flattened)
//   output:    [N_patches+1, D]
void patch_prep_fp32(
    const float* patches,    // [N_patches, D]
    const float* cls_token,  // [D]
    const float* pos_embed,  // [(N_patches+1), D]
    float*       output,     // [N_patches+1, D]
    int N_patches, int D
);

// Flash Attention-2 Multi-Head Self-Attention (optimised, OMP head-parallel).
//   Parallelised over attention heads via OpenMP.
void attention_flash_fp32(
    const float* input,           // [N, D]
    const float* w_qkv,           // [3D, D]
    const float* b_qkv,           // [3D]  (may be nullptr)
    const float* w_proj,          // [D, D]
    const float* b_proj,          // [D]   (may be nullptr)
    float*       output,          // [N, D]
    int N, int D, int num_heads,
    float*       scratch_qkv,     // [N * 3 * D] caller-provided temp buffer
    const float* w_qkv_packed,    // pre-packed [Co_t, D, TILE] for w_qkv
    const float* w_proj_packed,   // pre-packed [Co_t, D, TILE] for w_proj
    StreamHandle stream = nullptr,
    bool         dq_attn_out = false  // DQ attention output before proj (ONNX ALBERT sim)
);

// Sequence GEMM  [N, C_in] @ W[C_out, C_in]^T + bias → [N, C_out]
void seqgemm_fp32(
    const float* input,    // [N, C_in]
    const float* weight,   // [C_out, C_in]
    const float* bias,     // [C_out]  (may be nullptr)
    float*       output,   // [N, C_out]
    int N, int C_in, int C_out,
    bool         gelu     = false,
    const float* w_packed = nullptr,   // pre-packed [Co_t, C_in, TILE]
    StreamHandle stream   = nullptr
);

// GELU activation (in-place).
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
void gelu_fp32(float* inout, int N, StreamHandle stream = nullptr);

// ──────────────────────────────────────────────
// NEON vectorised activation kernels (inline, ARM-only).
// Declared here so all ops and layers share one copy.
// ──────────────────────────────────────────────
#ifdef __ARM_NEON
#include <arm_neon.h>

// Fast exp: Cephes-style degree-6 Horner polynomial (~1 ULP error).
static inline float32x4_t vexpq_f32_fast(float32x4_t x)
{
    x = vminq_f32(x, vdupq_n_f32( 88.0f));  // 88.376... causes m=±128 → ±inf; clamp at 88.0f is safe
    x = vmaxq_f32(x, vdupq_n_f32(-88.0f));
    float32x4_t m = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.44269504088896341f)));
    float32x4_t r = vfmsq_f32(x, m, vdupq_n_f32( 6.93359375e-1f));
    r             = vfmsq_f32(r, m, vdupq_n_f32(-2.12194440e-4f));
    float32x4_t p = vdupq_n_f32(1.9875691500e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f),             p, r);  // r^1 term
    p = vfmaq_f32(vdupq_n_f32(1.0f),             p, r);  // r^0 constant
    int32x4_t em = vcvtq_s32_f32(m);
    return vmulq_f32(p, vreinterpretq_f32_s32(
        vshlq_n_s32(vaddq_s32(em, vdupq_n_s32(127)), 23)));
}

// Fast sin: degree-9 minimax via range-reduction to [-π/2, π/2].
static inline float32x4_t vsinq_f32_fast(float32x4_t x)
{
    float32x4_t n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(0.31830988618f)));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(3.14159265359f));
    int32x4_t sign = vshlq_n_s32(vandq_s32(vcvtq_s32_f32(n), vdupq_n_s32(1)), 31);
    float32x4_t r2 = vmulq_f32(r, r);
    float32x4_t p  = vdupq_n_f32( 2.75573172e-6f);
    p = vfmaq_f32(vdupq_n_f32(-1.98412698e-4f), p, r2);
    p = vfmaq_f32(vdupq_n_f32( 8.33333333e-3f), p, r2);
    p = vfmaq_f32(vdupq_n_f32(-1.66666667e-1f), p, r2);
    p = vfmaq_f32(vdupq_n_f32( 1.00000000e+0f), p, r2);
    p = vmulq_f32(r, p);
    return vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(p), sign));
}

// sigmoid(x) = 1 / (1 + exp(-x))  via fast exp + 2-step Newton reciprocal.
static inline float32x4_t vsigmoidq_f32(float32x4_t x)
{
    float32x4_t e   = vexpq_f32_fast(vnegq_f32(x));
    float32x4_t ep1 = vaddq_f32(e, vdupq_n_f32(1.0f));
    float32x4_t rec = vrecpeq_f32(ep1);
    rec = vmulq_f32(rec, vrecpsq_f32(ep1, rec));
    rec = vmulq_f32(rec, vrecpsq_f32(ep1, rec));
    return rec;
}

// tanh(x) = 2*sigmoid(2x) - 1
static inline float32x4_t vtanhq_f32(float32x4_t x)
{
    return vsubq_f32(
        vmulq_f32(vdupq_n_f32(2.0f), vsigmoidq_f32(vmulq_f32(vdupq_n_f32(2.0f), x))),
        vdupq_n_f32(1.0f));
}
#endif // __ARM_NEON

// ──────────────────────────────────────────────────────────────────
// x86 AVX2 / AVX-512 vectorised activation helpers.
// Same polynomial coefficients as the ARM NEON helpers above,
// so NEON and AVX results are bit-identical on IEEE 754 hardware.
// ──────────────────────────────────────────────────────────────────
#ifdef __AVX2__
#include <immintrin.h>

// Q31 rounding multiply for __m256i (matches NEON vqrdmulhq_s32).
// Computes round(a * b / 2^31) per int32 lane using int64 intermediates.
static inline __m256i qrdmulh_epi32_256(__m256i a, __m256i b) {
    __m256i pe = _mm256_mul_epi32(a, b);
    __m256i ao = _mm256_srli_epi64(a, 32);
    __m256i bo = _mm256_srli_epi64(b, 32);
    __m256i po = _mm256_mul_epi32(ao, bo);
    __m256i r  = _mm256_set1_epi64x(1LL << 30);
    pe = _mm256_srli_epi64(_mm256_add_epi64(pe, r), 31);
    po = _mm256_srli_epi64(_mm256_add_epi64(po, r), 31);
    po = _mm256_slli_epi64(po, 32);
    return _mm256_blend_epi32(pe, po, 0xAA);
}

// Horizontal sum of 8 floats — shuffle-based to avoid vhaddps port-5 pressure.
// On Skylake, vhaddps uses port 5 (latency 5, throughput 1/cycle) whereas
// vmovshdup/vmovhlps use port 5 at 0.5/cycle throughput.  Halves port-5 usage
// vs 2× hadd while maintaining the same ~10-cycle dependency chain.
static inline float hsum256(__m256 v) {
    __m128 lo   = _mm256_castps256_ps128(v);
    __m128 hi   = _mm256_extractf128_ps(v, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);                // 8 → 4
    __m128 shuf = _mm_movehdup_ps(sum4);             // [1, 1, 3, 3]
    __m128 sum2 = _mm_add_ps(sum4, shuf);            // [0+1, -, 2+3, -]
    shuf        = _mm_movehl_ps(shuf, sum2);          // [2+3, 3+3, -, -]
    return _mm_cvtss_f32(_mm_add_ss(sum2, shuf));    // 0+1+2+3
}

// Fast exp — Cephes degree-6 Horner, same coefficients as vexpq_f32_fast.
static inline __m256 exp256_ps(__m256 x) {
    x = _mm256_min_ps(x, _mm256_set1_ps( 88.0f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    __m256 m = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(m, _mm256_set1_ps(6.93359375e-1f), x);
    r        = _mm256_fmadd_ps (m, _mm256_set1_ps(2.12194440e-4f), r);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    __m256i em = _mm256_add_epi32(_mm256_cvtps_epi32(m), _mm256_set1_epi32(127));
    return _mm256_mul_ps(p, _mm256_castsi256_ps(_mm256_slli_epi32(em, 23)));
}

// Fast sin — degree-9 minimax, same coefficients as vsinq_f32_fast.
static inline __m256 sin256_ps(__m256 x) {
    __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(0.31830988618f)),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(3.14159265359f), x);
    __m256i ni   = _mm256_cvtps_epi32(n);
    __m256i sign = _mm256_slli_epi32(_mm256_and_si256(ni, _mm256_set1_epi32(1)), 31);
    __m256 r2 = _mm256_mul_ps(r, r);
    __m256 p  = _mm256_set1_ps( 2.75573172e-6f);
    p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(-1.98412698e-4f));
    p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps( 8.33333333e-3f));
    p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(-1.66666667e-1f));
    p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps( 1.00000000e+0f));
    p = _mm256_mul_ps(r, p);
    return _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(p), sign));
}

// sigmoid(x) = 1/(1+exp(-x)) — fast exp + 2-step Newton reciprocal.
static inline __m256 sigmoid256_ps(__m256 x) {
    __m256 e   = exp256_ps(_mm256_xor_ps(x, _mm256_set1_ps(-0.0f)));
    __m256 ep1 = _mm256_add_ps(e, _mm256_set1_ps(1.0f));
    __m256 r   = _mm256_rcp_ps(ep1);
    r = _mm256_mul_ps(r, _mm256_fnmadd_ps(ep1, r, _mm256_set1_ps(2.0f)));
    r = _mm256_mul_ps(r, _mm256_fnmadd_ps(ep1, r, _mm256_set1_ps(2.0f)));
    return r;
}

// tanh(x) = 2*sigmoid(2x) - 1
static inline __m256 tanh256_ps(__m256 x) {
    return _mm256_fmadd_ps(_mm256_set1_ps(2.0f),
                           sigmoid256_ps(_mm256_add_ps(x, x)),
                           _mm256_set1_ps(-1.0f));
}

#ifdef __AVX512BW__
// ── Q31 rounding multiply (shared between gemm.cpp and conv.cpp) ──────────
// Matches NEON vqrdmulhq_s32. Defined here so all TUs see one copy.
static inline __m512i qrdmulh_epi32_512(__m512i a, __m512i b) {
    __m512i pe = _mm512_mul_epi32(a, b);
    __m512i ao = _mm512_srli_epi64(a, 32);
    __m512i bo = _mm512_srli_epi64(b, 32);
    __m512i po = _mm512_mul_epi32(ao, bo);
    __m512i r  = _mm512_set1_epi64(1LL << 30);
    pe = _mm512_srli_epi64(_mm512_add_epi64(pe, r), 31);
    po = _mm512_srli_epi64(_mm512_add_epi64(po, r), 31);
    po = _mm512_slli_epi64(po, 32);
    return _mm512_mask_blend_epi32(0xAAAAu, pe, po);
}
#endif // __AVX512BW__

#ifdef __AVX512F__
// ── AVX-512 (512-bit) helpers ─────────────────────────────────────

static inline float hsum512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}

static inline __m512 exp512_ps(__m512 x) {
    x = _mm512_min_ps(x, _mm512_set1_ps( 88.0f));
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.0f));
    __m512 m = _mm512_roundscale_ps(
        _mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)),
        _MM_FROUND_TO_NEAREST_INT);
    __m512 r = _mm512_fnmadd_ps(m, _mm512_set1_ps(6.93359375e-1f), x);
    r        = _mm512_fmadd_ps (m, _mm512_set1_ps(2.12194440e-4f), r);
    __m512 p = _mm512_set1_ps(1.9875691500e-4f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.3981999507e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.3334519073e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.1665795894e-2f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.6666665459e-1f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(5.0000001201e-1f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    __m512i em = _mm512_add_epi32(_mm512_cvtps_epi32(m), _mm512_set1_epi32(127));
    return _mm512_mul_ps(p, _mm512_castsi512_ps(_mm512_slli_epi32(em, 23)));
}

static inline __m512 sin512_ps(__m512 x) {
    __m512 n = _mm512_roundscale_ps(
        _mm512_mul_ps(x, _mm512_set1_ps(0.31830988618f)),
        _MM_FROUND_TO_NEAREST_INT);
    __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(3.14159265359f), x);
    __m512i ni   = _mm512_cvtps_epi32(n);
    __m512i sign = _mm512_slli_epi32(_mm512_and_si512(ni, _mm512_set1_epi32(1)), 31);
    __m512 r2 = _mm512_mul_ps(r, r);
    __m512 p  = _mm512_set1_ps( 2.75573172e-6f);
    p = _mm512_fmadd_ps(p, r2, _mm512_set1_ps(-1.98412698e-4f));
    p = _mm512_fmadd_ps(p, r2, _mm512_set1_ps( 8.33333333e-3f));
    p = _mm512_fmadd_ps(p, r2, _mm512_set1_ps(-1.66666667e-1f));
    p = _mm512_fmadd_ps(p, r2, _mm512_set1_ps( 1.00000000e+0f));
    p = _mm512_mul_ps(r, p);
    return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(p), sign));
}

static inline __m512 sigmoid512_ps(__m512 x) {
    __m512 e   = exp512_ps(_mm512_xor_ps(x, _mm512_set1_ps(-0.0f)));
    __m512 ep1 = _mm512_add_ps(e, _mm512_set1_ps(1.0f));
    __m512 r   = _mm512_rcp14_ps(ep1);
    r = _mm512_mul_ps(r, _mm512_fnmadd_ps(ep1, r, _mm512_set1_ps(2.0f)));
    r = _mm512_mul_ps(r, _mm512_fnmadd_ps(ep1, r, _mm512_set1_ps(2.0f)));
    return r;
}

static inline __m512 tanh512_ps(__m512 x) {
    return _mm512_fmadd_ps(_mm512_set1_ps(2.0f),
                           sigmoid512_ps(_mm512_add_ps(x, x)),
                           _mm512_set1_ps(-1.0f));
}
#endif // __AVX512F__
#endif // __AVX2__

// ──────────────────────────────────────────────
// LSTM recurrence scan (one direction, FP32).
//   proj   [T, 4*H] — pre-computed input projection (W_ih @ x + b_ih)
//   w_hh   [4*H, H] — hidden-to-hidden weights (raw, for gemm_fp32)
//   b_hh   [4*H]    — hidden-to-hidden bias (may be nullptr)
//   w_hh_packed     — pre-packed w_hh for gemm_fp32 (may be nullptr → fallback)
//   h_out  [T, H]   — output hidden states
//   reverse         — if true, scan from T-1 to 0
// ──────────────────────────────────────────────
void lstm_scan_fp32(
    const float* proj,
    const float* w_hh,
    const float* b_hh,
    const float* w_hh_packed,
    float*       h_out,
    int T, int H,
    bool reverse = false
);

// CLS token extraction: copy first row of sequence.
//   input:  [N, D]   output: [D]
void cls_extract_fp32(
    const float* input,   // [N, D]
    float*       output,  // [D]
    int D
);

// ──────────────────────────────────────────────
// ViT INT8 ops  (dynamic activation quant)
// ──────────────────────────────────────────────

// INT8 Sequence GEMM: [N, C_in] @ W^T + bias → [N, C_out]
//   Dynamic activation quantization (per-call in_scale = max|x|/127).
//   Per-channel weight quantization (w_scales[C_out]).
//   Optional GELU fused after bias add.
void seqgemm_int8(
    const float*   input,        // [N, C_in]
    const int8_t*  w_packed,     // packed [C_out/4, C_in/4, 4, 4]
    const float*   w_scales,     // [C_out] per-channel weight scales
    const int64_t* eff_zeros,    // [C_out] all zeros (required by gemm_int8)
    const float*   bias,         // [C_out]
    int8_t*        inp_scratch,  // [N * C_in] temp int8 buffer
    float*         req_scratch,  // [C_out] temp req_scale buffer
    float*         output,       // [N, C_out]
    int N, int C_in, int C_out,
    bool gelu = false,
    StreamHandle stream = nullptr
);

// CPU-only fast path: same as seqgemm_int8, with precomputed weight row sums
// for AVX-VNNI signed INT8 dot products.
void seqgemm_int8_with_rows(
    const float*   input,        // [N, C_in]
    const int8_t*  w_packed,     // packed [C_out/4, C_in/4, 4, 4]
    const float*   w_scales,     // [C_out] per-channel weight scales
    const int64_t* eff_zeros,    // [C_out] all zeros (required by gemm_int8)
    const float*   bias,         // [C_out]
    int8_t*        inp_scratch,  // [N * C_in] temp int8 buffer
    float*         req_scratch,  // [C_out] temp req_scale buffer
    float*         output,       // [N, C_out]
    int N, int C_in, int C_out,
    bool gelu,
    const int32_t* w_row_sums,
    StreamHandle stream = nullptr
);

// Single-token GEMV: weight_i8[C_out, C_in] × input[C_in] → output[C_out]
void seqgemm_int8_matvec(
    const float*   input,
    const int8_t*  weight_i8,
    const float*   w_scales,
    const float*   bias,
    float*         output,
    int C_in, int C_out,
    StreamHandle stream = nullptr
);

// INT8 Flash Attention: INT8 QKV/proj projections, FP32 FA-2 inner loop.
void attention_flash_int8(
    const float*   input,           // [N, D]
    const int8_t*  w_qkv_packed,    // packed [3D/4, D/4, 4, 4]
    const float*   w_qkv_scales,    // [3D] per-channel weight scales
    const int64_t* eff_zeros_qkv,   // [3D] all zeros
    const float*   b_qkv,           // [3D]
    const int8_t*  w_proj_packed,   // packed [D/4, D/4, 4, 4]
    const float*   w_proj_scales,   // [D] per-channel weight scales
    const int64_t* eff_zeros_proj,  // [D] all zeros (first D of eff_zeros_qkv)
    const float*   b_proj,          // [D]
    float*         output,          // [N, D]
    int N, int D, int num_heads,
    float*         scratch_qkv,     // [N * 3 * D] fp32 QKV buffer
    int8_t*        scratch_int8,    // [N * D] temp int8 for quantized activations
    float*         scratch_req,     // [3D] temp req_scale buffer
    StreamHandle   stream = nullptr
);

// CPU-only INT8 Flash Attention variant that passes row sums to the QKV and
// projection GEMMs, enabling the AVX-VNNI path inside gemm_int8.
void attention_flash_int8_with_rows(
    const float*   input,           // [N, D]
    const int8_t*  w_qkv_packed,    // packed [3D/4, D/4, 4, 4]
    const float*   w_qkv_scales,    // [3D] per-channel weight scales
    const int64_t* eff_zeros_qkv,   // [3D] all zeros
    const float*   b_qkv,           // [3D]
    const int8_t*  w_proj_packed,   // packed [D/4, D/4, 4, 4]
    const float*   w_proj_scales,   // [D] per-channel weight scales
    const int64_t* eff_zeros_proj,  // [D] all zeros (first D of eff_zeros_qkv)
    const float*   b_proj,          // [D]
    float*         output,          // [N, D]
    int N, int D, int num_heads,
    float*         scratch_qkv,     // [N * 3 * D] fp32 QKV buffer
    int8_t*        scratch_int8,    // [N * D] temp int8 for quantized activations
    float*         scratch_req,     // [3D] temp req_scale buffer
    const int32_t* qkv_row_sums,    // [3D]
    const int32_t* proj_row_sums,   // [D]
    StreamHandle   stream = nullptr
);

// ──────────────────────────────────────────────
// TTS FP32 ops
// ──────────────────────────────────────────────

// Elementwise activations  [N] → [N]
void leaky_relu_fp32(const float* in, float* out, int N, float alpha, StreamHandle stream = nullptr);
void exp_fp32       (const float* in, float* out, int N, StreamHandle stream = nullptr);
void sin_fp32       (const float* in, float* out, int N, StreamHandle stream = nullptr);
void sigmoid_fp32   (const float* in, float* out, int N, StreamHandle stream = nullptr);
void tanh_fp32      (const float* in, float* out, int N, StreamHandle stream = nullptr);

// Upsample ops
void upsample_nearest1d_fp32(const float* in, float* out, int T, int C, int sf, StreamHandle stream = nullptr);
void upsample_bilinear2d_fp32(const float* in, float* out,
                               int C, int iH, int iW, int oH, int oW, StreamHandle stream = nullptr);

// AdaIN1d: instance-norm per-channel + style modulation  [T, C] → [T, C]
//   gamma/beta: pre-projected style vectors [C] each
void ada_in1d_fp32(const float* feat, float* out,
                   const float* gamma, const float* beta,
                   int T, int C, float eps = 1e-5f, StreamHandle stream = nullptr,
                   const float* norm_weight = nullptr,  // InstanceNorm learnable scale [C]
                   const float* norm_bias   = nullptr); // InstanceNorm learnable shift [C]

// AdaLayerNorm1d: layer-norm per time-step + style modulation  [T, C] → [T, C]
void ada_layer_norm1d_fp32(const float* feat, float* out,
                            const float* gamma, const float* beta,
                            int T, int C, float eps = 1e-5f, StreamHandle stream = nullptr);

// Vector / scalar utilities
void add_vectors_fp32(float* dst, const float* src, int N, StreamHandle stream = nullptr);   // dst[i] += src[i]
void relu_fp32(float* inout, int N, StreamHandle stream = nullptr);                            // in-place ReLU
void bias_add_rows_fp32(float* out, const float* bias, int T, int C, StreamHandle stream = nullptr);
void scale_fp32(const float* in, float* out, size_t N, float scalar, StreamHandle stream = nullptr);
void sum_channels_fp32(const float* in, float* out, int T, int C, StreamHandle stream = nullptr);

// Embedding / tensor layout ops
void embedding_lookup_fp32(const float* ids, const float* weight, float* out,
                            int T, int num_embeddings, int embedding_dim, StreamHandle stream = nullptr);
void concat1d_fp32(const float* in1, const float* in2, float* out,
                   int T, int C1, int C2, bool broadcast, StreamHandle stream = nullptr);
void slice_channels_fp32(const float* in, float* out, int T, int C_in,
                          int ch_start, int C_out, StreamHandle stream = nullptr);
int  length_regulate_fp32(const float* feat, const float* durs, float* out,
                           int T, int C, float duration_scale = 1.0f);

// TTS compute ops
void snake1d_fp32(const float* in, float* out, int T, int C,
                  const float* alpha, const float* inv_alpha);
void sine_gen_fp32(const float* f0, float* out, int T,
                   int harmonic_num, int sample_rate,
                   float sine_amp, float voiced_threshold,
                   float noise_std = 0.0f, int upsample_scale = 0);
void istft_fp32(const float* mag, const float* phase, float* out,
                int T_frames, int n_fft, int hop_size, bool normalized = false);
// w_real/w_imag: precomputed Hann-windowed cosine/sine weights [K*n_fft].
// If null, weights are computed on-the-fly (less accurate for near-zero bins).
void stft_fp32(const float* audio, float* out,
               int T_audio, int n_fft, int hop_size,
               const float* w_real = nullptr, const float* w_imag = nullptr);
void bert_embeddings_fp32(const float* ids, float* out, int T, int E,
                           const float* word_weight, const float* pos_weight,
                           const float* type_weight,
                           const float* ln_weight, const float* ln_bias,
                           float eps);

// ConvTranspose1D grouped fallback (groups>1)
void conv_transpose1d_grouped_fp32(
    const float* in, float* out, const float* weight,
    int T, int T_out, int C_in, int C_out,
    int C_in_g, int C_out_g,
    int groups, int stride, int padding, int kernel_size);

// INT8 per-row quantization helpers
void quantize_row_fp32_to_int8(const float* row, int8_t* qrow,
                                float& out_scale, int C);
void dequant_bias_row_fp32(float* row, float row_scale, const float* bias, int C);
void dequant_accum_row_fp32(float* dst, const float* src, float row_scale, int C);

// ──────────────────────────────────────────────
// INT8 conv1d K-split: multi-thread quantize → GEMM → dequant loop.
// Caller pre-allocates inp_scratch [T_out*C_in_g], tmp_buf [T_out*C_out],
// row_scale_buf [T_out].
// ──────────────────────────────────────────────
void conv1d_int8_ksplit(
    const float*         A_base,
    const int8_t* const* w_per_k_packed,
    const int64_t*       eff_zeros,
    const float*         req_scale,
    const float*         bias,
    float*               out_buf,
    float*               tmp_buf,
    int8_t*              inp_scratch,
    float*               row_scale_buf,
    int T_out, int C_in_g, int C_out,
    int kernel_size, int dilation,
    int nthreads);

// ──────────────────────────────────────────────
// Dynamic quantize-dequantize (uint8 asymmetric, per-tensor).
// Simulates ONNX DynamicQuantizeLinear: rounds inputs to nearest
// representable uint8 value, then dequantizes back to fp32.
// In-place: x[i] = round(x[i] / scale + zp) * scale - zp * scale
// where scale = (max - min) / 255, zp = clip(round(-min / scale), 0, 255).
// ──────────────────────────────────────────────
void dynamic_quantize_dequant_f32(float* x, int n);

// ── LM ops ───────────────────────────────────────────────────────

// RMSNorm: y = x / rms(x) * gamma,  rms(x) = sqrt(mean(x^2) + eps)
void rmsnorm_fp32(
    const float* input,
    const float* gamma,
    float*       output,
    int N, int C,
    float eps = 1e-6f,
    StreamHandle stream = nullptr
);

// SiLU: out[i] = in[i] * sigmoid(in[i])
void silu_fp32(const float* in, float* out, int N, StreamHandle stream = nullptr);

// Elementwise multiply: out[i] = a[i] * b[i]
void elemwise_mul_fp32(const float* a, const float* b, float* out, int N,
                        StreamHandle stream = nullptr);

// Causal self-attention with KV cache and RoPE
void causal_attn_core_fp32(
    const float* q,
    const float* k,
    const float* v,
    float*       kv_cache,
    int          cache_pos,
    int          cache_len,
    float*       output,
    int N, int num_q_heads, int num_kv_heads, int head_dim,
    int          step_pos,
    float        rope_theta,
    bool         is_decode,
    float*       scratch,
    StreamHandle stream = nullptr
);
