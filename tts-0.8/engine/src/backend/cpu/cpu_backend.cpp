// src/backend/cpu/cpu_backend.cpp — wraps ops_neon.hpp; registers at priority 0
#include "backend/cpu/cpu_backend.hpp"
#include "ops_neon.hpp"

static CpuAllocator s_cpu_alloc;
static Allocator* cpu_make_allocator() { return &s_cpu_alloc; }

// Wrapper to match backend signature (alpha/inv_alpha before T/C, plus StreamHandle)
static void snake1d_fp32_cpu(const float* in, float* out,
                               const float* alpha, const float* inv_alpha,
                               int T, int C, StreamHandle) {
    snake1d_fp32(in, out, T, C, alpha, inv_alpha);
}

// ─────────────────────────────────────────────────────────────────
// Registration
// ─────────────────────────────────────────────────────────────────
static BackendOps s_cpu_ops = {
    .name             = "cpu",
    .priority         = 0,
    .mem_space        = MemSpace::Host,
    .make_allocator   = cpu_make_allocator,
    .pack_weights_int8 = pack_weights_sdot,
    .pack_weights_fp32 = pack_weights_f32,
    .free_packed_int8 = free_packed,
    .free_packed_fp32 = free_packed_f32,
    .gemm_int8        = gemm_int8,
    .gemm_int8_int32  = gemm_int8_int32,
    .sgemm_f32        = sgemm_f32,
    .gemm_fp32_vec    = gemm_fp32_vec,
    .conv2d_int8      = conv2d_int8,
    .conv2d_winograd_nhwc_int8  = conv2d_winograd_nhwc_int8,
    .conv2d_fp32_nhwc           = conv2d_fp32_nhwc,
    .conv2d_winograd_nhwc_fp32  = conv2d_winograd_nhwc_fp32,
    .seqgemm_fp32     = seqgemm_fp32,
    .seqgemm_int8     = seqgemm_int8,
    .seqgemm_int8_matvec = seqgemm_int8_matvec,
    .attention_flash_fp32  = attention_flash_fp32,
    .attention_flash_int8  = attention_flash_int8,
    .layernorm_fp32   = layernorm_fp32,
    .maxpool_int8     = maxpool_int8,
    .maxpool_int8_nhwc = maxpool_int8_nhwc,
    .maxpool_fp32_nhwc = maxpool_fp32_nhwc,
    .avgpool_global_int8      = avgpool_global_int8,
    .avgpool_global_int8_nhwc = avgpool_global_int8_nhwc,
    .avgpool_global_fp32_nhwc = avgpool_global_fp32_nhwc,
    .add_fp32         = add_fp32,
    .add_requant_int8 = add_requant_int8,
    .relu_fp32        = relu_fp32,
    .gelu_fp32        = gelu_fp32,
    .sigmoid_fp32     = sigmoid_fp32,
    .tanh_fp32        = tanh_fp32,
    .leaky_relu_fp32  = leaky_relu_fp32,
    .exp_fp32         = exp_fp32,
    .sin_fp32         = sin_fp32,
    .upsample_nearest1d_fp32  = upsample_nearest1d_fp32,
    .upsample_bilinear2d_fp32 = upsample_bilinear2d_fp32,
    .ada_in1d_fp32        = ada_in1d_fp32,
    .ada_layer_norm1d_fp32 = ada_layer_norm1d_fp32,
    .add_vectors_fp32  = add_vectors_fp32,
    .bias_add_rows_fp32 = bias_add_rows_fp32,
    .scale_fp32        = scale_fp32,
    .sum_channels_fp32 = sum_channels_fp32,
    .relu_inplace_fp32 = relu_fp32,
    .embedding_lookup_fp32 = embedding_lookup_fp32,
    .concat1d_fp32     = concat1d_fp32,
    .slice_channels_fp32 = slice_channels_fp32,

    .snake1d_fp32 = snake1d_fp32_cpu,

    // LM ops
    .rmsnorm_fp32       = rmsnorm_fp32,
    .silu_fp32          = silu_fp32,
    .elemwise_mul_fp32  = elemwise_mul_fp32,
    .causal_attn_core_fp32 = causal_attn_core_fp32,
};

// Static initializer: register before main() runs.
static const bool s_registered = []() {
    register_backend(&s_cpu_ops);
    return true;
}();
