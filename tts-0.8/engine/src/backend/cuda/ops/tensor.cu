// ops/tensor.cu — CUDA tensor utility ops
#include "cuda_ops_common.cuh"
#include "cuda_ops.hpp"

void cuda_add_vectors_fp32(float* dst, const float* src, int N, StreamHandle /*stream*/) {
    k_add_vecs_fp32<<<(N + 255) / 256, 256>>>(dst, src, N);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_bias_add_rows_fp32(float* out, const float* bias, int T, int C, StreamHandle /*stream*/) {
    float* d_bias = tmp_gpu(bias, C);
    dim3 block(32, 8);
    dim3 grid((C + block.x - 1) / block.x, (T + block.y - 1) / block.y);
    k_bias_add_rows<<<grid, block>>>(out, d_bias, T, C);
    CUDA_CHECK(cudaGetLastError());
    cudaFree(d_bias);
}

void cuda_scale_fp32(const float* in, float* out, size_t N, float scalar, StreamHandle /*stream*/) {
    k_scale_fp32<<<((int)N + 255) / 256, 256>>>(in, out, (int)N, scalar);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_sum_channels_fp32(const float* in, float* out, int T, int C, StreamHandle /*stream*/) {
    k_sum_channels<<<(T + 255) / 256, 256>>>(in, out, T, C);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_relu_inplace_fp32(float* x, int N, StreamHandle /*stream*/) {
    k_relu_fp32<<<(N + 255) / 256, 256>>>(x, N);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_embedding_lookup_fp32(
    const float* ids, const float* weight, float* out,
    int T, int vocab, int D, StreamHandle /*stream*/)
{
    dim3 block(16, 16);
    dim3 grid((D + block.x - 1) / block.x, (T + block.y - 1) / block.y);
    k_embedding_lookup<<<grid, block>>>(ids, weight, out, T, vocab, D);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_concat1d_fp32(
    const float* a, const float* b, float* out,
    int T, int C1, int C2, bool broadcast, StreamHandle /*stream*/)
{
    int C = C1 + C2;
    dim3 block(16, 16);
    dim3 grid((C + block.x - 1) / block.x, (T + block.y - 1) / block.y);
    k_concat1d<<<grid, block>>>(a, b, out, T, C1, C2, broadcast);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_slice_channels_fp32(
    const float* in, float* out, int T, int C_in,
    int ch_start, int C_out, StreamHandle /*stream*/)
{
    dim3 block(16, 16);
    dim3 grid((C_out + block.x - 1) / block.x, (T + block.y - 1) / block.y);
    k_slice_channels<<<grid, block>>>(in, out, T, C_in, ch_start, C_out);
    CUDA_CHECK(cudaGetLastError());
}
