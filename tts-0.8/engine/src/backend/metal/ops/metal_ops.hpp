// src/backend/metal/ops/metal_ops.hpp
// Pure C++ header declaring all Metal op wrapper functions.
// These are implemented in .mm files and called from metal_backend.mm.
#pragma once
#include "backend/backend.hpp"
#include <cstdint>
#include <cstddef>

// ── Weight packing (delegate to CPU) ─────────────────────────────
int8_t* metal_pack_weights_int8(const int8_t* w, int C_out, int K);
float*  metal_pack_weights_fp32(const float*  w, int C_out, int K);
void    metal_free_packed_int8(int8_t* p);
void    metal_free_packed_fp32(float*  p);

// ── GEMM ──────────────────────────────────────────────────────────
void metal_gemm_int8(
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
    StreamHandle   stream);

void metal_gemm_int8_int32(
    const int8_t*  A,
    const int8_t*  B_packed,
    int32_t*       C,
    int M, int K, int N,
    const int32_t* b_row_sums,
    StreamHandle   stream);

void metal_sgemm_f32(
    const float* A,
    const float* B_packed,
    const float* bias,
    float*       C,
    bool         relu,
    int M, int K, int N,
    bool         in_parallel,
    StreamHandle stream);

void metal_gemm_fp32_vec(
    const float* input,
    const float* weight,
    const float* bias,
    float*       output,
    int C_in, int C_out,
    const float* w_packed,
    StreamHandle stream);

// ── Conv ──────────────────────────────────────────────────────────
void metal_conv2d_int8(
    const int8_t*  input,
    const int8_t*  weight,
    const int8_t*  w_pre_packed,
    const int64_t* eff_bias,
    const int32_t* req_mult,
    const int32_t* req_exp,
    int8_t         in_zp,
    int8_t         out_zp,
    int8_t*        output,
    int C_in, int H, int W,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    int groups,
    int8_t*        scratch_col,
    bool           nhwc,
    const int32_t* b_row_sums,
    StreamHandle   stream);

void metal_conv2d_winograd_nhwc_int8(
    const int8_t*        input,
    const int8_t* const* w_wino_packed,
    const float*         wino_weight_scale,
    float                wino_input_scale,
    const int64_t*       eff_bias,
    const int32_t*       req_mult,
    const int32_t*       req_exp,
    const float*         req_scale,
    int8_t               in_zp,
    int8_t               out_zp,
    int8_t*              output,
    int C_in, int H, int W,
    int C_out,
    int8_t*              scratch_input_hat,
    int32_t*             scratch_output_hat,
    const int32_t* const* w_wino_row_sums,
    StreamHandle          stream);

void metal_conv2d_fp32_nhwc(
    const float* input,
    const float* weight,
    const float* bias,
    float*       output,
    int C_in, int H, int W,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    bool         relu,
    const float* w_packed,
    float*       scratch_col,
    int dilation_h, int dilation_w,
    StreamHandle stream);

void metal_conv1d_int8_nhwc(
    const float*   input,
    const int8_t*  w_packed,
    const float*   w_scales,
    const float*   bias,
    float*         output,
    int T, int C_in, int C_out,
    int kernel_size, int stride, int padding, int dilation,
    StreamHandle stream);

void metal_conv1d_fp32(
    const float* input,
    const float* weight,
    const float* bias,
    float*       output,
    int C_in, int T, int C_out,
    int kernel_size, int stride, int padding, int dilation,
    bool relu,
    StreamHandle stream);

void metal_conv2d_winograd_nhwc_fp32(
    const float*         input,
    const float* const*  w_wino_packed,
    const float*         bias,
    float*               output,
    int C_in, int H, int W, int C_out, bool relu,
    float*               scratch,
    StreamHandle         stream,
    const float*         raw_weight);

void metal_conv_transpose1d_fp32(
    const float* input,
    const float* weight,
    const float* bias,
    float*       output,
    int C_in, int T, int C_out,
    int kernel_size, int stride, int padding, int output_padding,
    int groups,
    StreamHandle stream);

// ── Sequence GEMM ─────────────────────────────────────────────────
void metal_seqgemm_fp32(
    const float* input,
    const float* weight,
    const float* bias,
    float*       output,
    int N, int C_in, int C_out,
    bool         gelu,
    const float* w_packed,
    StreamHandle stream);

void metal_seqgemm_int8(
    const float*   input,
    const int8_t*  w_packed,
    const float*   w_scales,
    const int64_t* eff_zeros,
    const float*   bias,
    int8_t*        inp_scratch,
    float*         req_scratch,
    float*         output,
    int N, int C_in, int C_out,
    bool           gelu,
    StreamHandle   stream);

// ── Attention ─────────────────────────────────────────────────────
void metal_attention_flash_fp32(
    const float* input,
    const float* w_qkv,
    const float* b_qkv,
    const float* w_proj,
    const float* b_proj,
    float*       output,
    int N, int D, int num_heads,
    float*       scratch_qkv,
    const float* w_qkv_packed,
    const float* w_proj_packed,
    StreamHandle stream,
    bool         dq_attn_out);

void metal_attention_flash_int8(
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
    StreamHandle   stream);

// ── Layer Norm ────────────────────────────────────────────────────
void metal_layernorm_fp32(
    const float* input,
    const float* gamma,
    const float* beta,
    float*       output,
    int N, int C,
    float        eps,
    StreamHandle stream);

void metal_ada_in1d_fp32(
    const float* feat, float* out,
    const float* gamma, const float* beta,
    int T, int C, float eps, StreamHandle stream,
    const float* norm_weight = nullptr,
    const float* norm_bias   = nullptr);

void metal_ada_layer_norm1d_fp32(
    const float* feat, float* out,
    const float* gamma, const float* beta,
    int T, int C, float eps, StreamHandle stream);

// ── Pool ──────────────────────────────────────────────────────────
void metal_maxpool_int8(
    const int8_t*, int8_t*,
    int C, int H, int W, int kH, int kW,
    int sh, int sw, int ph, int pw,
    StreamHandle);

void metal_maxpool_int8_nhwc(
    const int8_t*, int8_t*,
    int C, int H, int W, int kH, int kW,
    int sh, int sw, int ph, int pw,
    StreamHandle);

void metal_maxpool_fp32_nhwc(
    const float*, float*,
    int C, int H, int W, int kH, int kW,
    int sh, int sw, int ph, int pw,
    StreamHandle);

void metal_avgpool_global_int8(
    const int8_t*, float in_sc, int in_zp, float out_sc, int out_zp,
    int8_t*, int C, int H, int W,
    StreamHandle);

void metal_avgpool_global_int8_nhwc(
    const int8_t*, float in_sc, int in_zp, float out_sc, int out_zp,
    int8_t*, int C, int H, int W,
    StreamHandle);

void metal_avgpool_global_fp32_nhwc(
    const float*, float*, int C, int H, int W,
    StreamHandle);

// ── ModelIOTensor ops ────────────────────────────────────────────────────
void metal_add_fp32(const float*, const float*, float*, int N, bool relu, StreamHandle);

void metal_add_requant_int8(
    const int8_t*, const int8_t*,
    float in1_sc, int in1_zp, float in2_sc, int in2_zp,
    float out_sc, int out_zp,
    int8_t*, int N,
    StreamHandle);

void metal_add_vectors_fp32(float* dst, const float* src, int N, StreamHandle);
void metal_bias_add_rows_fp32(float* out, const float* bias, int T, int C, StreamHandle);
void metal_scale_fp32(const float* in, float* out, size_t N, float scalar, StreamHandle);
void metal_sum_channels_fp32(const float* in, float* out, int T, int C, StreamHandle);
void metal_relu_inplace_fp32(float* inout, int N, StreamHandle);
void metal_embedding_lookup_fp32(const float* ids, const float* w, float* out,
                                  int T, int num_emb, int D, StreamHandle);
void metal_concat1d_fp32(const float*, const float*, float*,
                          int T, int C1, int C2, bool bcast, StreamHandle);
void metal_slice_channels_fp32(const float*, float*, int T, int C_in,
                                int ch_start, int C_out, StreamHandle);
void metal_upsample_nearest1d_fp32(const float*, float*, int T, int C, int sf, StreamHandle);
void metal_upsample_bilinear2d_fp32(const float*, float*,
                                     int C, int iH, int iW, int oH, int oW, StreamHandle);
void metal_snake1d_fp32(const float* in, float* out,
                         const float* alpha, const float* inv_alpha,
                         int T, int C, StreamHandle);

// ── Sequence ops (LSTM, STFT, ISTFT, Pad, SineGen) ───────────────
void metal_lstm_fp32(const float*, const float*, const float*, const float*, const float*,
                     const float*, const float*, const float*, const float*,
                     float*, int, int, int, bool, StreamHandle);
void metal_stft_fp32(const float*, const float*, const float*, float*, int, int, int, StreamHandle);
void metal_istft_fp32(const float*, const float*, float*, int, int, int, bool, StreamHandle);
void metal_reflection_pad1d(const float*, float*, int, int, StreamHandle);
void metal_sine_gen_fp32(const float*, float*, int, int, int, int, float, float, StreamHandle);

// ── Activations ───────────────────────────────────────────────────
void metal_relu_fp32(float* inout, int N, StreamHandle stream);
void metal_gelu_fp32(float* inout, int N, StreamHandle stream);
void metal_sigmoid_fp32(const float* in, float* out, int N, StreamHandle stream);
void metal_tanh_fp32(const float* in, float* out, int N, StreamHandle stream);
void metal_leaky_relu_fp32(const float* in, float* out, int N, float alpha, StreamHandle stream);
void metal_exp_fp32(const float* in, float* out, int N, StreamHandle stream);
void metal_sin_fp32(const float* in, float* out, int N, StreamHandle stream);
void metal_silu_fp32(const float* in, float* out, int N, StreamHandle stream);
void metal_elemwise_mul_fp32(const float* a, const float* b, float* out, int N, StreamHandle stream);
