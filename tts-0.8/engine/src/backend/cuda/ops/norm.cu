// ops/norm.cu — CUDA upsample and adaptive norm ops
#include "cuda_ops_common.cuh"
#include "cuda_ops.hpp"

void cuda_upsample_nearest1d_fp32(const float* in, float* out, int T, int C, int sf, StreamHandle /*stream*/) {
    dim3 block(256);
    dim3 grid((T*sf + block.x - 1) / block.x, 1, C);
    k_upsample_nearest1d<<<grid, block>>>(in, out, T, C, sf);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_upsample_bilinear2d_fp32(const float* in, float* out,
    int C, int iH, int iW, int oH, int oW, StreamHandle /*stream*/)
{
    dim3 block(16, 16);
    dim3 grid((oW + block.x - 1) / block.x, (oH + block.y - 1) / block.y, C);
    k_upsample_bilinear2d<<<grid, block>>>(in, out, C, iH, iW, oH, oW);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_ada_in1d_fp32(
    const float* feat, float* out,
    const float* gamma, const float* beta,
    int T, int C, float eps, StreamHandle /*stream*/,
    const float* norm_weight, const float* norm_bias)
{
    auto mark = g_scratch.save();
    float* d_nw = tmp_gpu(norm_weight, C);
    float* d_nb = tmp_gpu(norm_bias,   C);
    k_ada_in1d<<<C, 1>>>(feat, out, gamma, beta, T, C, eps, d_nw, d_nb);
    CUDA_CHECK(cudaGetLastError());
    g_scratch.restore(mark);
}

void cuda_ada_layer_norm1d_fp32(
    const float* feat, float* out,
    const float* gamma, const float* beta,
    int T, int C, float eps, StreamHandle /*stream*/)
{
    // Use parallel reduction: up to 256 threads per row
    int bsz = min(256, ((C + 31) / 32) * 32);
    bsz = max(bsz, 32);
    k_ada_layernorm1d<<<T, bsz, bsz * sizeof(float)>>>(feat, out, gamma, beta, T, C, eps);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_snake1d_fp32(const float* in, float* out,
    const float* alpha, const float* inv_alpha,
    int T, int C, StreamHandle /*stream*/)
{
    int n = T * C;
    k_snake1d<<<(n + 255) / 256, 256>>>(in, out, alpha, inv_alpha, T, C);
    CUDA_CHECK(cudaGetLastError());
}
