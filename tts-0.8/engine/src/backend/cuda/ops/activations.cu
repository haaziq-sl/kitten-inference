// ops/activations.cu — CUDA activation and add ops
#include "cuda_ops_common.cuh"
#include "cuda_ops.hpp"

void cuda_add_fp32(const float* a, const float* b, float* out, int N, bool relu, StreamHandle /*stream*/) {
    k_add_fp32<<<(N + 255) / 256, 256>>>(a, b, out, N, relu);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_add_requant_int8(
    const int8_t* a, const int8_t* b,
    float in1_sc, int in1_zp, float in2_sc, int in2_zp,
    float out_sc, int out_zp,
    int8_t* out, int N,
    StreamHandle /*stream*/)
{
    k_add_requant<<<(N + 255) / 256, 256>>>(a, b, in1_sc, in1_zp, in2_sc, in2_zp, out_sc, out_zp, out, N);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_relu_fp32(float* x, int N, StreamHandle /*stream*/) {
    k_relu_fp32<<<(N + 255) / 256, 256>>>(x, N);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_gelu_fp32(float* x, int N, StreamHandle /*stream*/) {
    k_gelu_fp32<<<(N + 255) / 256, 256>>>(x, N);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_sigmoid_fp32(const float* in, float* out, int N, StreamHandle /*stream*/) {
    k_sigmoid_fp32<<<(N + 255) / 256, 256>>>(in, out, N);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_tanh_fp32(const float* in, float* out, int N, StreamHandle /*stream*/) {
    k_tanh_fp32<<<(N + 255) / 256, 256>>>(in, out, N);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_leaky_relu_fp32(const float* in, float* out, int N, float alpha, StreamHandle /*stream*/) {
    k_leaky_relu_fp32<<<(N + 255) / 256, 256>>>(in, out, N, alpha);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_exp_fp32(const float* in, float* out, int N, StreamHandle /*stream*/) {
    k_exp_fp32<<<(N + 255) / 256, 256>>>(in, out, N);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_sin_fp32(const float* in, float* out, int N, StreamHandle /*stream*/) {
    k_sin_fp32<<<(N + 255) / 256, 256>>>(in, out, N);
    CUDA_CHECK(cudaGetLastError());
}

// SiLU: out[i] = in[i] * sigmoid(in[i])
static __global__ void k_silu_fp32(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] / (1.f + expf(-in[i]));
}

void cuda_silu_fp32(const float* in, float* out, int N, StreamHandle /*stream*/) {
    k_silu_fp32<<<(N + 255) / 256, 256>>>(in, out, N);
    CUDA_CHECK(cudaGetLastError());
}

// Elementwise multiply: out[i] = a[i] * b[i]
static __global__ void k_elemwise_mul_fp32(const float* a, const float* b, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = a[i] * b[i];
}

void cuda_elemwise_mul_fp32(const float* a, const float* b, float* out, int N, StreamHandle /*stream*/) {
    k_elemwise_mul_fp32<<<(N + 255) / 256, 256>>>(a, b, out, N);
    CUDA_CHECK(cudaGetLastError());
}

// RMSNorm: y = x / rms(x) * gamma,  rms(x) = sqrt(mean(x^2) + eps)
// Grid: (N,), Block: (min(C, 256),) — one block per token row
static __global__ void k_rmsnorm_fp32(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    float*       __restrict__ output,
    int N, int C, float eps)
{
    int row = blockIdx.x;
    if (row >= N) return;
    const float* x = input  + (size_t)row * C;
    float*       y = output + (size_t)row * C;

    extern __shared__ float smem[];
    int t = threadIdx.x;

    // Sum of squares
    float ss = 0.f;
    for (int c = t; c < C; c += blockDim.x) ss += x[c] * x[c];
    smem[t] = ss;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) smem[t] += smem[t + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(smem[0] / (float)C + eps);
    __syncthreads();

    // Apply: y = x * inv_rms * gamma
    for (int c = t; c < C; c += blockDim.x)
        y[c] = x[c] * inv_rms * gamma[c];
}

void cuda_rmsnorm_fp32(
    const float* input, const float* gamma, float* output,
    int N, int C, float eps, StreamHandle /*stream*/)
{
    int bsz = min(256, ((C + 31) / 32) * 32);
    bsz = max(bsz, 32);
    float* d_gamma = g_const_cache.get(gamma, C);
    k_rmsnorm_fp32<<<N, bsz, bsz * sizeof(float)>>>(input, d_gamma, output, N, C, eps);
    CUDA_CHECK(cudaGetLastError());
}
