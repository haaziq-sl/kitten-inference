// ops/gemm.cu — CUDA GEMM ops
// Optimizations:
//   - GPU scratch pool instead of cudaMalloc/cudaFree per call
//   - Native INT8 GEMM (CUDA_R_8I → CUDA_R_32I) when K%4==0 (ModelIOTensor Cores)
//   - TF32 ModelIOTensor Cores for FP32 GEMMs (set via cublasSetMathMode on handle)
#include "cuda_ops_common.cuh"
#include "cuda_ops.hpp"

void cuda_gemm_int8(
    const int8_t*  A,
    const int8_t*  B_packed,
    const int64_t* eff_bias,
    const int32_t* /*req_mult*/,
    const int32_t* /*req_exp*/,
    const float*   req_scale_f,
    int8_t         out_zp,
    void*          C,
    bool           is_float,
    int M, int K, int N,
    bool           nchw_out,
    bool           /*in_parallel*/,
    const int32_t* /*b_row_sums*/,
    StreamHandle   /*stream*/)
{
    int N_pad = (N + 3) / 4 * 4;
    int K_pad = (K + 3) / 4 * 4;

    auto mark = g_scratch.save();

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Use native INT8 GEMM → INT32 only for 4-aligned GEMM shapes.
    bool use_int8_tc = cublas_int8_tc_supported(N, M, K);

    int64_t* d_bias  = g_const_cache.get(eff_bias,    N);
    float*   d_scale = g_const_cache.get(req_scale_f, N);

    if (use_int8_tc) {
        int32_t* d_C = (int32_t*)g_scratch.alloc((size_t)M * N_pad * sizeof(int32_t));
        int32_t alpha_i = 1, beta_i = 0;
        CUBLAS_CHECK(cublasGemmEx(
            get_cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha_i,
            B_packed, CUDA_R_8I, K_pad,  // weight [N_pad, K_pad] row-major → col [K_pad, N_pad], lda=K_pad
            A,        CUDA_R_8I, K,       // input  [M, K] row-major → col [K, M], ldb=K
            &beta_i,
            d_C,      CUDA_R_32I, N_pad,  // output [M, N_pad], ldc=N_pad
            CUBLAS_COMPUTE_32I,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaGetLastError());

        if (is_float) {
            k_requant_int32_to_fp32<<<grid, block>>>(d_C, d_bias, d_scale, (float*)C, M, N, N_pad);
        } else if (nchw_out) {
            k_requant_int32_to_int8_nchw<<<grid, block>>>(d_C, d_bias, d_scale, out_zp, (int8_t*)C, M, N, N_pad);
        } else {
            k_requant_int32_to_int8<<<grid, block>>>(d_C, d_bias, d_scale, out_zp, (int8_t*)C, M, N, N_pad);
        }
    } else {
        // FP32 fallback (for K not divisible by 4)
        float* d_C    = (float*)g_scratch.alloc((size_t)M * N_pad * sizeof(float));
        float* d_W_fp = int8_to_fp32_gpu(B_packed, (size_t)N_pad * K_pad);
        float* d_A_fp = int8_to_fp32_gpu(A, (size_t)M * K);
        float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasGemmEx(
            get_cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_W_fp, CUDA_R_32F, K_pad,
            d_A_fp, CUDA_R_32F, K,
            &beta,
            d_C,    CUDA_R_32F, N_pad,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaGetLastError());

        if (is_float) {
            k_requant_fp32_to_fp32<<<grid, block>>>(d_C, d_bias, d_scale, (float*)C, M, N, N_pad);
        } else if (nchw_out) {
            k_requant_fp32_to_int8_nchw<<<grid, block>>>(d_C, d_bias, d_scale, out_zp, (int8_t*)C, M, N, N_pad);
        } else {
            k_requant_fp32_to_int8<<<grid, block>>>(d_C, d_bias, d_scale, out_zp, (int8_t*)C, M, N, N_pad);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    g_scratch.restore(mark);
}

void cuda_gemm_int8_int32(
    const int8_t* A,
    const int8_t* B_packed,
    int32_t*      C,
    int M, int K, int N,
    const int32_t* /*b_row_sums*/,
    StreamHandle   /*stream*/)
{
    int N_pad = (N + 3) / 4 * 4;
    int K_pad = (K + 3) / 4 * 4;

    auto mark = g_scratch.save();

    bool use_int8_tc = (K % 4 == 0);
    if (use_int8_tc) {
        int32_t alpha_i = 1, beta_i = 0;
        CUBLAS_CHECK(cublasGemmEx(
            get_cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha_i,
            B_packed, CUDA_R_8I, K_pad,
            A,        CUDA_R_8I, K,
            &beta_i,
            C,        CUDA_R_32I, N_pad,
            CUBLAS_COMPUTE_32I,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        float* d_W_fp = int8_to_fp32_gpu(B_packed, (size_t)N_pad * K_pad);
        float* d_A_fp = int8_to_fp32_gpu(A, (size_t)M * K);
        float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasGemmEx(
            get_cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_W_fp, CUDA_R_32F, K_pad,
            d_A_fp, CUDA_R_32F, K,
            &beta,
            (float*)C, CUDA_R_32F, N_pad,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CUDA_CHECK(cudaGetLastError());
    g_scratch.restore(mark);
}

void cuda_sgemm_f32(
    const float* A,
    const float* B_packed,
    const float* bias,
    float*       C,
    bool         relu,
    int M, int K, int N,
    bool         /*in_parallel*/,
    StreamHandle /*stream*/)
{
    int K_pad = (K + 3) / 4 * 4;
    float alpha = 1.f, beta = 0.f;
    // TF32 ModelIOTensor Cores via cublasSetMathMode on handle
    CUBLAS_CHECK(cublasGemmEx(
        get_cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B_packed, CUDA_R_32F, K_pad,
        A,        CUDA_R_32F, K,
        &beta,
        C,        CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_CHECK(cudaGetLastError());

    if (bias || relu) {
        auto mark = g_scratch.save();
        float* d_bias = g_const_cache.get(bias, N);
        if (d_bias) {
            dim3 block(32, 8);
            dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
            k_bias_add_rows<<<grid, block>>>(C, d_bias, M, N);
            CUDA_CHECK(cudaGetLastError());
        }
        if (relu) {
            k_relu_fp32<<<(M * N + 255) / 256, 256>>>(C, M * N);
            CUDA_CHECK(cudaGetLastError());
        }
        g_scratch.restore(mark);
    }
}

void cuda_gemm_fp32_vec(
    const float* input,
    const float* /*weight*/,
    const float* bias,
    float*       output,
    int C_in, int C_out,
    const float* w_packed,
    StreamHandle /*stream*/)
{
    // Use exact FP32 (not TF32) for style→gamma/beta: small K, correctness critical
    int K_pad = (C_in + 3) / 4 * 4;
    float alpha = 1.f, beta_v = 0.f;
    CUBLAS_CHECK(cublasGemmEx(
        get_cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        C_out, 1, C_in,
        &alpha,
        w_packed, CUDA_R_32F, K_pad,
        input,    CUDA_R_32F, C_in,
        &beta_v,
        output,   CUDA_R_32F, C_out,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
    CUDA_CHECK(cudaGetLastError());

    if (bias) {
        auto mark = g_scratch.save();
        float* d_bias = g_const_cache.get(bias, C_out);
        k_add_vecs_fp32<<<(C_out + 255) / 256, 256>>>(output, d_bias, C_out);
        CUDA_CHECK(cudaGetLastError());
        g_scratch.restore(mark);
    }
}
