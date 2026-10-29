// ops/pool.cu — CUDA pooling and layernorm ops
#include "cuda_ops_common.cuh"
#include "cuda_ops.hpp"

void cuda_layernorm_fp32(
    const float* input,
    const float* gamma,
    const float* beta,
    float*       output,
    int N, int C,
    float        eps,
    StreamHandle /*stream*/)
{
    // Use parallel reduction: 256 threads per row instead of 1
    int bsz = min(256, ((C + 31) / 32) * 32);  // round up to multiple of 32
    bsz = max(bsz, 32);

    auto mark = g_scratch.save();
    float* d_gamma = tmp_gpu(gamma, C);
    float* d_beta  = tmp_gpu(beta,  C);
    k_layernorm<<<N, bsz, bsz * sizeof(float)>>>(input, d_gamma, d_beta, output, N, C, eps);
    CUDA_CHECK(cudaGetLastError());
    g_scratch.restore(mark);
}

void cuda_maxpool_int8(
    const int8_t* in, int8_t* out,
    int C, int H, int W, int kH, int kW,
    int sh, int sw, int ph, int pw,
    StreamHandle /*stream*/)
{
    int oH = (H + 2*ph - kH) / sh + 1;
    int oW = (W + 2*pw - kW) / sw + 1;
    dim3 block(8, 8);
    dim3 grid((oW + block.x - 1) / block.x, (oH + block.y - 1) / block.y, C);
    k_maxpool_nchw_int8<<<grid, block>>>(in, out, C, H, W, kH, kW, sh, sw, ph, pw, oH, oW);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_maxpool_int8_nhwc(
    const int8_t* in, int8_t* out,
    int C, int H, int W, int kH, int kW,
    int sh, int sw, int ph, int pw,
    StreamHandle /*stream*/)
{
    int oH = (H + 2*ph - kH) / sh + 1;
    int oW = (W + 2*pw - kW) / sw + 1;
    dim3 block(8, 8);
    dim3 grid((oW + block.x - 1) / block.x, (oH + block.y - 1) / block.y, C);
    k_maxpool_nhwc_int8<<<grid, block>>>(in, out, C, H, W, kH, kW, sh, sw, ph, pw, oH, oW);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_maxpool_fp32_nhwc(
    const float* in, float* out,
    int C, int H, int W, int kH, int kW,
    int sh, int sw, int ph, int pw,
    StreamHandle /*stream*/)
{
    int oH = (H + 2*ph - kH) / sh + 1;
    int oW = (W + 2*pw - kW) / sw + 1;
    dim3 block(8, 8);
    dim3 grid((oW + block.x - 1) / block.x, (oH + block.y - 1) / block.y, C);
    k_maxpool_nhwc_fp32<<<grid, block>>>(in, out, C, H, W, kH, kW, sh, sw, ph, pw, oH, oW);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_avgpool_global_int8(
    const int8_t* in, float in_sc, int in_zp, float out_sc, int out_zp,
    int8_t* out, int C, int H, int W,
    StreamHandle /*stream*/)
{
    int HW = H * W;
    k_avgpool_global_nchw_int8<<<(C + 255) / 256, 256>>>(in, in_sc, in_zp, out_sc, out_zp, out, C, HW);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_avgpool_global_int8_nhwc(
    const int8_t* in, float in_sc, int in_zp, float out_sc, int out_zp,
    int8_t* out, int C, int H, int W,
    StreamHandle /*stream*/)
{
    int HW = H * W;
    k_avgpool_global_nhwc_int8<<<(C + 255) / 256, 256>>>(in, in_sc, in_zp, out_sc, out_zp, out, C, HW);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_avgpool_global_fp32_nhwc(
    const float* in, float* out, int C, int H, int W,
    StreamHandle /*stream*/)
{
    int HW = H * W;
    k_avgpool_global_nhwc_fp32<<<(C + 255) / 256, 256>>>(in, out, C, HW);
    CUDA_CHECK(cudaGetLastError());
}
