// cuda_ops.hpp — declarations of all CUDA op functions
#pragma once
#include "backend/backend.hpp"
#include <cstdint>

// GEMM
void cuda_gemm_int8(const int8_t* A, const int8_t* B_packed,
    const int64_t* eff_bias, const int32_t* req_mult, const int32_t* req_exp,
    const float* req_scale_f, int8_t out_zp, void* C, bool is_float,
    int M, int K, int N, bool nchw_out, bool in_parallel,
    const int32_t* b_row_sums, StreamHandle stream);

void cuda_gemm_int8_int32(const int8_t* A, const int8_t* B_packed,
    int32_t* C, int M, int K, int N,
    const int32_t* b_row_sums, StreamHandle stream);

void cuda_sgemm_f32(const float* A, const float* B_packed,
    const float* bias, float* C, bool relu,
    int M, int K, int N, bool in_parallel, StreamHandle stream);

void cuda_gemm_fp32_vec(const float* input, const float* weight,
    const float* bias, float* output, int C_in, int C_out,
    const float* w_packed, StreamHandle stream);

// Conv
void cuda_conv2d_int8(const int8_t* input, const int8_t* weight,
    const int8_t* w_pre_packed, const int64_t* eff_bias,
    const int32_t* req_mult, const int32_t* req_exp,
    int8_t in_zp, int8_t out_zp, int8_t* output,
    int C_in, int H, int W, int C_out, int kH, int kW,
    int stride_h, int stride_w, int pad_h, int pad_w, int groups,
    int8_t* scratch_col, bool nhwc, const int32_t* b_row_sums,
    StreamHandle stream);

void cuda_conv2d_winograd_nhwc_int8(const int8_t* input,
    const int8_t* const* w_wino_packed, const float* wino_weight_scale,
    float wino_input_scale, const int64_t* eff_bias,
    const int32_t* req_mult, const int32_t* req_exp, const float* req_scale,
    int8_t in_zp, int8_t out_zp, int8_t* output,
    int C_in, int H, int W, int C_out,
    int8_t* scratch_input_hat, int32_t* scratch_output_hat,
    const int32_t* const* w_wino_row_sums, StreamHandle stream);

void cuda_conv2d_fp32_nhwc(const float* input, const float* weight,
    const float* bias, float* output,
    int C_in, int H, int W, int C_out, int kH, int kW,
    int stride_h, int stride_w, int pad_h, int pad_w,
    bool relu, const float* w_packed, float* scratch_col,
    int dilation_h, int dilation_w, StreamHandle stream);

void cuda_conv2d_winograd_nhwc_fp32(const float* input,
    const float* const* w_wino_packed, const float* bias,
    float* output, int C_in, int H, int W, int C_out, bool relu,
    float* scratch, StreamHandle stream,
    const float* raw_weight = nullptr);  // CPU NHWC weights for cuDNN path

// SeqGEMM + Attention
void cuda_seqgemm_fp32(const float* input, const float* weight,
    const float* bias, float* output, int N, int C_in, int C_out,
    bool gelu, const float* w_packed, StreamHandle stream);

void cuda_seqgemm_int8(const float* input, const int8_t* w_packed,
    const float* w_scales, const int64_t* eff_zeros, const float* bias,
    int8_t* inp_scratch, float* req_scratch, float* output,
    int N, int C_in, int C_out, bool gelu, StreamHandle stream);

// LM activation ops
void cuda_silu_fp32(const float* in, float* out, int N, StreamHandle stream);
void cuda_elemwise_mul_fp32(const float* a, const float* b, float* out, int N, StreamHandle stream);
void cuda_rmsnorm_fp32(const float* input, const float* gamma, float* output,
    int N, int C, float eps, StreamHandle stream);

void cuda_seqgemm_int8_matvec(const float* input, const int8_t* weight_i8,
    const float* w_scales, const float* bias, float* output,
    int C_in, int C_out, StreamHandle stream);

void cuda_causal_attn_core_fp32(
    const float* q, const float* k, const float* v,
    float* kv_cache, int cache_pos, int cache_len,
    float* output,
    int N, int num_q_heads, int num_kv_heads, int head_dim,
    int step_pos, float rope_theta, bool is_decode,
    float* scratch, StreamHandle stream);

void cuda_conv1d_int8_nhwc(
    const float* input, const int8_t* w_packed,
    const float* w_scales, const float* bias,
    float* output,
    int T, int C_in, int C_out,
    int kernel_size, int stride, int padding, int dilation,
    StreamHandle stream);

void cuda_attention_flash_fp32(const float* input,
    const float* w_qkv, const float* b_qkv,
    const float* w_proj, const float* b_proj,
    float* output, int N, int D, int num_heads,
    float* scratch_qkv, const float* w_qkv_packed,
    const float* w_proj_packed, StreamHandle stream,
    bool dq_attn_out);

void cuda_attention_flash_int8(const float* input,
    const int8_t* w_qkv_packed, const float* w_qkv_scales,
    const int64_t* eff_zeros_qkv, const float* b_qkv,
    const int8_t* w_proj_packed, const float* w_proj_scales,
    const int64_t* eff_zeros_proj, const float* b_proj,
    float* output, int N, int D, int num_heads,
    float* scratch_qkv, int8_t* scratch_int8, float* scratch_req,
    StreamHandle stream);

// Norm + Pool
void cuda_layernorm_fp32(const float* input, const float* gamma,
    const float* beta, float* output, int N, int C, float eps,
    StreamHandle stream);

void cuda_maxpool_int8(const int8_t* in, int8_t* out,
    int C, int H, int W, int kH, int kW,
    int sh, int sw, int ph, int pw, StreamHandle stream);

void cuda_maxpool_int8_nhwc(const int8_t* in, int8_t* out,
    int C, int H, int W, int kH, int kW,
    int sh, int sw, int ph, int pw, StreamHandle stream);

void cuda_maxpool_fp32_nhwc(const float* in, float* out,
    int C, int H, int W, int kH, int kW,
    int sh, int sw, int ph, int pw, StreamHandle stream);

void cuda_avgpool_global_int8(const int8_t* in, float in_sc, int in_zp,
    float out_sc, int out_zp, int8_t* out, int C, int H, int W,
    StreamHandle stream);

void cuda_avgpool_global_int8_nhwc(const int8_t* in, float in_sc, int in_zp,
    float out_sc, int out_zp, int8_t* out, int C, int H, int W,
    StreamHandle stream);

void cuda_avgpool_global_fp32_nhwc(const float* in, float* out,
    int C, int H, int W, StreamHandle stream);

// Activations + Add
void cuda_add_fp32(const float* a, const float* b, float* out,
    int N, bool relu, StreamHandle stream);

void cuda_add_requant_int8(const int8_t* a, const int8_t* b,
    float in1_sc, int in1_zp, float in2_sc, int in2_zp,
    float out_sc, int out_zp, int8_t* out, int N, StreamHandle stream);

void cuda_relu_fp32(float* x, int N, StreamHandle stream);
void cuda_gelu_fp32(float* x, int N, StreamHandle stream);
void cuda_sigmoid_fp32(const float* in, float* out, int N, StreamHandle stream);
void cuda_tanh_fp32(const float* in, float* out, int N, StreamHandle stream);
void cuda_leaky_relu_fp32(const float* in, float* out, int N, float alpha, StreamHandle stream);
void cuda_exp_fp32(const float* in, float* out, int N, StreamHandle stream);
void cuda_sin_fp32(const float* in, float* out, int N, StreamHandle stream);

// Norm + Upsample
void cuda_upsample_nearest1d_fp32(const float* in, float* out,
    int T, int C, int sf, StreamHandle stream);

void cuda_upsample_bilinear2d_fp32(const float* in, float* out,
    int C, int iH, int iW, int oH, int oW, StreamHandle stream);

void cuda_ada_in1d_fp32(const float* feat, float* out,
    const float* gamma, const float* beta,
    int T, int C, float eps, StreamHandle stream,
    const float* norm_weight = nullptr,
    const float* norm_bias   = nullptr);

void cuda_ada_layer_norm1d_fp32(const float* feat, float* out,
    const float* gamma, const float* beta,
    int T, int C, float eps, StreamHandle stream);

void cuda_snake1d_fp32(const float* in, float* out,
    const float* alpha, const float* inv_alpha,
    int T, int C, StreamHandle stream);

// ModelIOTensor ops
void cuda_add_vectors_fp32(float* dst, const float* src, int N, StreamHandle stream);
void cuda_bias_add_rows_fp32(float* out, const float* bias, int T, int C, StreamHandle stream);
void cuda_scale_fp32(const float* in, float* out, size_t N, float scalar, StreamHandle stream);
void cuda_sum_channels_fp32(const float* in, float* out, int T, int C, StreamHandle stream);
void cuda_relu_inplace_fp32(float* x, int N, StreamHandle stream);
void cuda_embedding_lookup_fp32(const float* ids, const float* weight,
    float* out, int T, int vocab, int D, StreamHandle stream);
void cuda_concat1d_fp32(const float* a, const float* b, float* out,
    int T, int C1, int C2, bool broadcast, StreamHandle stream);
void cuda_slice_channels_fp32(const float* in, float* out,
    int T, int C_in, int ch_start, int C_out, StreamHandle stream);
