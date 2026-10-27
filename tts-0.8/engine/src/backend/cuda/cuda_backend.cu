// cuda_backend.cu — CUDA backend registration
#include "cuda_backend.hpp"
#include "backend/backend.hpp"
#include "ops/cuda_ops.hpp"
#include "ops/cuda_ops_common.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#ifdef HAVE_CUDNN
#  include <cudnn.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <new>

// ─────────────────────────────────────────────────────────────────
// GPU scratch pool — defined here, extern'd in cuda_ops_common.cuh
// ─────────────────────────────────────────────────────────────────
GpuScratch g_scratch;

// ─────────────────────────────────────────────────────────────────
// GPU constant cache — persistent bias/scale arrays, uploaded once
// ─────────────────────────────────────────────────────────────────
GpuConstCache g_const_cache;

// ─────────────────────────────────────────────────────────────────
// cuBLAS handle
// ─────────────────────────────────────────────────────────────────
static cublasHandle_t s_cublas = nullptr;
cublasHandle_t get_cublas() {
    if (!s_cublas) {
        CUBLAS_CHECK(cublasCreate(&s_cublas));
        // Enable TF32 ModelIOTensor Cores for all FP32 GEMMs (Ampere+)
        // Gives ~2-3x speedup on A100/A10g with negligible precision impact
        CUBLAS_CHECK(cublasSetMathMode(s_cublas, CUBLAS_TF32_TENSOR_OP_MATH));
    }
    return s_cublas;
}

#ifdef HAVE_CUDNN
// ─────────────────────────────────────────────────────────────────
// cuDNN handle
// ─────────────────────────────────────────────────────────────────
static cudnnHandle_t s_cudnn = nullptr;
cudnnHandle_t get_cudnn() {
    if (!s_cudnn) {
        cudnnStatus_t st = cudnnCreate(&s_cudnn);
        if (st != CUDNN_STATUS_SUCCESS) {
            fprintf(stderr, "cudnnCreate failed: %d\n", (int)st);
            exit(1);
        }
    }
    return s_cudnn;
}
#endif // HAVE_CUDNN

// ─────────────────────────────────────────────────────────────────
// CudaAllocator
// ─────────────────────────────────────────────────────────────────
void* CudaAllocator::alloc(size_t n) { void* p; CUDA_CHECK(cudaMalloc(&p, n)); return p; }
void CudaAllocator::free(void* p) { cudaFree(p); }
void CudaAllocator::copy_h2d(void* dst, const void* src, size_t n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice));
}
void CudaAllocator::copy_d2h(void* dst, const void* src, size_t n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost));
}
void CudaAllocator::copy_d2d(void* dst, const void* src, size_t n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToDevice));
}
void CudaAllocator::sync() { CUDA_CHECK(cudaDeviceSynchronize()); }
void CudaAllocator::zero_memory(void* ptr, size_t n) { CUDA_CHECK(cudaMemset(ptr, 0, n)); }
static CudaAllocator s_cuda_alloc;
static Allocator* cuda_make_allocator() { return &s_cuda_alloc; }

// ─────────────────────────────────────────────────────────────────
// Weight packing (runs on CPU, returns host pointers)
// ─────────────────────────────────────────────────────────────────
static int8_t* cuda_pack_weights_int8(const int8_t* w, int C_out, int K) {
    int N_pad = (C_out + 3) / 4 * 4;
    int K_pad = (K    + 3) / 4 * 4;
    int8_t* p = new int8_t[(size_t)N_pad * K_pad]();
    for (int n = 0; n < C_out; ++n)
        for (int k = 0; k < K; ++k)
            p[(size_t)n * K_pad + k] = w[(size_t)n * K + k];
    return p;
}

static float* cuda_pack_weights_fp32(const float* w, int C_out, int K) {
    int N_pad = (C_out + 3) / 4 * 4;
    int K_pad = (K    + 3) / 4 * 4;
    float* p = new float[(size_t)N_pad * K_pad]();
    for (int n = 0; n < C_out; ++n)
        for (int k = 0; k < K; ++k)
            p[(size_t)n * K_pad + k] = w[(size_t)n * K + k];
    return p;
}

static void cuda_free_packed_int8(int8_t* p) { delete[] p; }
static void cuda_free_packed_fp32(float*  p) { delete[] p; }

// ─────────────────────────────────────────────────────────────────
// BackendOps registration
// ─────────────────────────────────────────────────────────────────
// nvcc 12 + C++20 bug: designated initializers in .cu global structs are silently
// zeroed at runtime. Populate all fields explicitly inside the lambda instead.
static BackendOps s_cuda_ops;

static const bool s_cuda_registered = []() {
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) return true;

    s_cuda_ops.name             = "cuda";
    s_cuda_ops.priority         = -1;
    s_cuda_ops.mem_space        = MemSpace::CUDA;
    s_cuda_ops.make_allocator   = cuda_make_allocator;

    s_cuda_ops.pack_weights_int8 = cuda_pack_weights_int8;
    s_cuda_ops.pack_weights_fp32 = cuda_pack_weights_fp32;
    s_cuda_ops.free_packed_int8  = cuda_free_packed_int8;
    s_cuda_ops.free_packed_fp32  = cuda_free_packed_fp32;

    s_cuda_ops.gemm_int8        = cuda_gemm_int8;
    s_cuda_ops.gemm_int8_int32  = cuda_gemm_int8_int32;
    s_cuda_ops.sgemm_f32        = cuda_sgemm_f32;
    s_cuda_ops.gemm_fp32_vec    = cuda_gemm_fp32_vec;

    s_cuda_ops.conv2d_int8               = cuda_conv2d_int8;
    s_cuda_ops.conv2d_winograd_nhwc_int8 = cuda_conv2d_winograd_nhwc_int8;
    s_cuda_ops.conv2d_fp32_nhwc          = cuda_conv2d_fp32_nhwc;
    s_cuda_ops.conv2d_winograd_nhwc_fp32 = cuda_conv2d_winograd_nhwc_fp32;

    s_cuda_ops.seqgemm_fp32          = cuda_seqgemm_fp32;
    s_cuda_ops.seqgemm_int8          = cuda_seqgemm_int8;
    s_cuda_ops.seqgemm_int8_matvec   = cuda_seqgemm_int8_matvec;
    s_cuda_ops.conv1d_int8_nhwc      = cuda_conv1d_int8_nhwc;
    s_cuda_ops.attention_flash_fp32  = cuda_attention_flash_fp32;
    s_cuda_ops.attention_flash_int8  = cuda_attention_flash_int8;
    s_cuda_ops.layernorm_fp32        = cuda_layernorm_fp32;

    s_cuda_ops.maxpool_int8             = cuda_maxpool_int8;
    s_cuda_ops.maxpool_int8_nhwc        = cuda_maxpool_int8_nhwc;
    s_cuda_ops.maxpool_fp32_nhwc        = cuda_maxpool_fp32_nhwc;
    s_cuda_ops.avgpool_global_int8      = cuda_avgpool_global_int8;
    s_cuda_ops.avgpool_global_int8_nhwc = cuda_avgpool_global_int8_nhwc;
    s_cuda_ops.avgpool_global_fp32_nhwc = cuda_avgpool_global_fp32_nhwc;

    s_cuda_ops.add_fp32         = cuda_add_fp32;
    s_cuda_ops.add_requant_int8 = cuda_add_requant_int8;

    s_cuda_ops.relu_fp32        = cuda_relu_fp32;
    s_cuda_ops.gelu_fp32        = cuda_gelu_fp32;
    s_cuda_ops.sigmoid_fp32     = cuda_sigmoid_fp32;
    s_cuda_ops.tanh_fp32        = cuda_tanh_fp32;
    s_cuda_ops.leaky_relu_fp32  = cuda_leaky_relu_fp32;
    s_cuda_ops.exp_fp32         = cuda_exp_fp32;
    s_cuda_ops.sin_fp32         = cuda_sin_fp32;

    s_cuda_ops.upsample_nearest1d_fp32  = cuda_upsample_nearest1d_fp32;
    s_cuda_ops.upsample_bilinear2d_fp32 = cuda_upsample_bilinear2d_fp32;

    s_cuda_ops.ada_in1d_fp32         = cuda_ada_in1d_fp32;
    s_cuda_ops.ada_layer_norm1d_fp32 = cuda_ada_layer_norm1d_fp32;

    s_cuda_ops.add_vectors_fp32   = cuda_add_vectors_fp32;
    s_cuda_ops.bias_add_rows_fp32 = cuda_bias_add_rows_fp32;
    s_cuda_ops.scale_fp32         = cuda_scale_fp32;
    s_cuda_ops.sum_channels_fp32  = cuda_sum_channels_fp32;
    s_cuda_ops.relu_inplace_fp32  = cuda_relu_inplace_fp32;

    s_cuda_ops.embedding_lookup_fp32 = cuda_embedding_lookup_fp32;
    s_cuda_ops.concat1d_fp32         = cuda_concat1d_fp32;
    s_cuda_ops.slice_channels_fp32   = cuda_slice_channels_fp32;

    s_cuda_ops.snake1d_fp32 = cuda_snake1d_fp32;

    s_cuda_ops.rmsnorm_fp32          = cuda_rmsnorm_fp32;
    s_cuda_ops.silu_fp32             = cuda_silu_fp32;
    s_cuda_ops.elemwise_mul_fp32     = cuda_elemwise_mul_fp32;
    s_cuda_ops.causal_attn_core_fp32 = cuda_causal_attn_core_fp32;

    register_backend(&s_cuda_ops);
    return true;
}();
