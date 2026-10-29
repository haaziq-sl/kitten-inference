// ops/conv.cu — CUDA Conv2D ops
// Optimizations:
//   - cuDNN for FP32 conv (Winograd/implicit-GEMM, TF32 ModelIOTensor Cores) when HAVE_CUDNN
//   - GPU scratch pool (no cudaMalloc/cudaFree in hot path)
//   - Native INT8 GEMM (CUDA_R_8I → CUDA_R_32I) when K%4==0
//   - FP32 im2col done on GPU (eliminates D2H + H2D transfers)
//   - TF32 ModelIOTensor Cores for all FP32 GEMMs via cuBLAS
#include "cuda_ops_common.cuh"
#include "cuda_ops.hpp"
#include <unordered_map>

// CPU weight pointer → GPU copy. Entry includes size + fingerprint to detect
// pointer reuse after model destruction (CPU allocator may reuse addresses).
struct WeightEntry { float* d_ptr; size_t n; float fp0, fpN; };
static std::unordered_map<const float*, WeightEntry> s_w_gpu_cache;

// Upload weight to GPU; re-upload if pointer is reused with different data.
static float* get_weight_gpu(const float* cpu_w, int size)
{
    float fp0 = size > 0 ? cpu_w[0] : 0.f;
    float fpN = size > 1 ? cpu_w[size-1] : fp0;
    auto it = s_w_gpu_cache.find(cpu_w);
    if (it != s_w_gpu_cache.end()) {
        const WeightEntry& e = it->second;
        if (e.n == (size_t)size && e.fp0 == fp0 && e.fpN == fpN)
            return e.d_ptr;
        if (e.n == (size_t)size) {
            CUDA_CHECK(cudaMemcpy(e.d_ptr, cpu_w, (size_t)size * sizeof(float), cudaMemcpyHostToDevice));
            it->second.fp0 = fp0; it->second.fpN = fpN;
            return e.d_ptr;
        }
        CUDA_CHECK(cudaFree(e.d_ptr));
    }
    float* d;
    CUDA_CHECK(cudaMalloc(&d, (size_t)size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, cpu_w, (size_t)size * sizeof(float), cudaMemcpyHostToDevice));
    s_w_gpu_cache[cpu_w] = WeightEntry{d, (size_t)size, fp0, fpN};
    return d;
}

#ifdef HAVE_CUDNN
// ─── cuDNN descriptor cache ────────────────────────────────────────
struct CuDNNConvKey {
    int C_in, H, W, C_out, kH, kW, sh, sw, ph, pw, dh, dw;
    bool operator==(const CuDNNConvKey& o) const {
        return C_in==o.C_in && H==o.H && W==o.W && C_out==o.C_out &&
               kH==o.kH && kW==o.kW && sh==o.sh && sw==o.sw &&
               ph==o.ph && pw==o.pw && dh==o.dh && dw==o.dw;
    }
};
struct CuDNNConvKeyHash {
    size_t operator()(const CuDNNConvKey& k) const {
        size_t h = 0;
        auto mix = [&](int v){
            h ^= std::hash<int>{}(v) + 0x9e3779b9u + (h<<6) + (h>>2);
        };
        mix(k.C_in); mix(k.H);   mix(k.W);   mix(k.C_out);
        mix(k.kH);   mix(k.kW); mix(k.sh); mix(k.sw);
        mix(k.ph);   mix(k.pw); mix(k.dh); mix(k.dw);
        return h;
    }
};
struct CuDNNConvDesc {
    cudnnTensorDescriptor_t      in_desc{}, out_desc{};
    cudnnFilterDescriptor_t      filt_desc{};
    cudnnConvolutionDescriptor_t conv_desc{};
    cudnnActivationDescriptor_t  act_relu{}, act_identity{};
    cudnnTensorDescriptor_t      bias_desc{};
    cudnnConvolutionFwdAlgo_t    algo{};
    size_t                       workspace_size{0};
    int                          oH{}, oW{};
};
static std::unordered_map<CuDNNConvKey, CuDNNConvDesc, CuDNNConvKeyHash> s_conv_cache;

static const CuDNNConvDesc& get_conv_desc(
    int C_in, int H, int W, int C_out,
    int kH, int kW, int sh, int sw, int ph, int pw, int dh, int dw)
{
    CuDNNConvKey key{C_in, H, W, C_out, kH, kW, sh, sw, ph, pw, dh, dw};
    auto it = s_conv_cache.find(key);
    if (it != s_conv_cache.end()) return it->second;

    CuDNNConvDesc d;
    d.oH = (H + 2*ph - dh*(kH-1) - 1) / sh + 1;
    d.oW = (W + 2*pw - dw*(kW-1) - 1) / sw + 1;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&d.in_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        d.in_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, C_in, H, W));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&d.out_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        d.out_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, C_out, d.oH, d.oW));

    CUDNN_CHECK(cudnnCreateFilterDescriptor(&d.filt_desc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        d.filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, C_out, C_in, kH, kW));

    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&d.conv_desc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        d.conv_desc, ph, pw, sh, sw, dh, dw,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    // Allow TF32 ModelIOTensor Cores (FP32 input, TF32 compute on Ampere)
    CUDNN_CHECK(cudnnSetConvolutionMathType(
        d.conv_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));

    // Heuristic algorithm selection (first call per unique shape)
    int n_algos = 0;
    cudnnConvolutionFwdAlgoPerf_t algo_perf[8];
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
        get_cudnn(), d.in_desc, d.filt_desc, d.conv_desc, d.out_desc,
        8, &n_algos, algo_perf));
    d.algo = algo_perf[0].algo;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        get_cudnn(), d.in_desc, d.filt_desc, d.conv_desc, d.out_desc,
        d.algo, &d.workspace_size));

    // Also build activation descriptor (IDENTITY by default; caller can use RELU variant)
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&d.act_relu));
    CUDNN_CHECK(cudnnSetActivationDescriptor(d.act_relu,
        CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&d.act_identity));
    CUDNN_CHECK(cudnnSetActivationDescriptor(d.act_identity,
        CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0));
    // Bias descriptor: [1, C_out, 1, 1] in NHWC (C_out in dim1)
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&d.bias_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        d.bias_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, C_out, 1, 1));

    s_conv_cache.emplace(key, d);
    return s_conv_cache.at(key);
}

#endif // HAVE_CUDNN

// Depthwise conv NHWC INT8: input/output in NHWC, w_packed in [C_out_pad, K_pad].
// Each output channel co uses weights w_packed[co * K_pad + k] for k in 0..kH*kW-1.
// Requant: acc_raw + eff_bias[co], then * req_scale[co] + out_zp.
static __global__ void k_depthwise_nhwc_int8(
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ w_packed,  // [C_out_pad, K_pad] (NHWC order: kh,kw inner)
    const int64_t* __restrict__ d_bias,
    const float*   __restrict__ d_scale,
    int8_t out_zp,
    int8_t* __restrict__ output,
    int H, int W, int C_in,
    int oH, int oW, int C_out,
    int kH, int kW, int K_pad,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int8_t in_zp)
{
    int co = blockIdx.x * blockDim.x + threadIdx.x;
    int ow = blockIdx.y * blockDim.y + threadIdx.y;
    int oh = blockIdx.z;
    if (co >= C_out || ow >= oW || oh >= oH) return;

    int64_t acc = d_bias ? d_bias[co] : 0LL;
    int k = 0;
    for (int kh = 0; kh < kH; ++kh) {
        for (int kw = 0; kw < kW; ++kw, ++k) {
            int ih = oh * stride_h - pad_h + kh;
            int iw = ow * stride_w - pad_w + kw;
            int8_t iv = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                ? input[((size_t)ih * W + iw) * C_in + co]
                : in_zp;
            int8_t wv = w_packed[(size_t)co * K_pad + k];
            acc += (int32_t)iv * (int32_t)wv;
        }
    }
    // eff_bias already accounts for -in_zp*row_sum(w); acc is raw dot product.
    float fval = (float)acc * d_scale[co];
    int q = __float2int_rn(fval) + (int)out_zp;
    output[((size_t)oh * oW + ow) * C_out + co] = (int8_t)max(-128, min(127, q));
}

void cuda_conv2d_int8(
    const int8_t*  input,
    const int8_t*  weight,
    const int8_t*  w_pre_packed,   // [C_out_pad, K_pad] on GPU
    const int64_t* eff_bias,       // [C_out] CPU
    const int32_t* req_mult,
    const int32_t* req_exp,
    int8_t         in_zp,
    int8_t         out_zp,
    int8_t*        output,
    int C_in, int H, int W,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int groups,
    int8_t*        scratch_col,    // ignored, we alloc from pool
    bool           nhwc,
    const int32_t* b_row_sums,
    StreamHandle   /*stream*/)
{
    // Grouped/depthwise conv
    if (groups > 1) {
        int oH = (H + 2 * pad_h - kH) / stride_h + 1;
        int oW = (W + 2 * pad_w - kW) / stride_w + 1;

        if (nhwc && w_pre_packed) {
            // GPU depthwise NHWC path: input/output are GPU NHWC tensors.
            int K     = kH * kW;  // C_in_g==1 for depthwise
            int K_pad = (K + 3) / 4 * 4;
            auto mark = g_scratch.save();
            int64_t* d_bias  = g_const_cache.get(eff_bias, C_out);
            float*   d_scale = get_req_scale_gpu(req_mult, req_exp, C_out);
            dim3 block(32, 8, 1);
            dim3 grid((C_out + block.x - 1) / block.x,
                      (oW   + block.y - 1) / block.y,
                      oH);
            k_depthwise_nhwc_int8<<<grid, block>>>(
                input, w_pre_packed, d_bias, d_scale, out_zp,
                output, H, W, C_in, oH, oW, C_out,
                kH, kW, K_pad, stride_h, stride_w, pad_h, pad_w, in_zp);
            CUDA_CHECK(cudaGetLastError());
            g_scratch.restore(mark);
            return;
        }

        // CPU scalar fallback (caller provides CPU buffers)
        int C_in_g = C_in / groups;
        memset(output, 0, (size_t)C_out * oH * oW);
        for (int g = 0; g < groups; ++g)
        for (int co = g * (C_out / groups); co < (g+1) * (C_out / groups); ++co)
        for (int oh = 0; oh < oH; ++oh) for (int ow = 0; ow < oW; ++ow) {
            int64_t acc = eff_bias ? eff_bias[co] : 0LL;
            for (int kh = 0; kh < kH; ++kh) for (int kw = 0; kw < kW; ++kw) {
                for (int ci = 0; ci < C_in_g; ++ci) {
                    int ih = oh * stride_h - pad_h + kh;
                    int iw = ow * stride_w - pad_w + kw;
                    int8_t iv = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        ? input[(size_t)(g * C_in_g + ci) * H * W + (size_t)ih * W + iw]
                        : in_zp;
                    int8_t wv = weight[(size_t)co * C_in_g * kH * kW + (size_t)ci * kH * kW + kh * kW + kw];
                    acc += (int64_t)(iv - in_zp) * wv;
                }
            }
            int32_t q;
            if (req_mult && req_exp) {
                int rshift = 31 - req_exp[co];
                int64_t prod = (int64_t)(int32_t)acc * req_mult[co];
                q = (int32_t)((prod + (1LL << (rshift - 1))) >> rshift) + out_zp;
            } else {
                q = (int32_t)acc + out_zp;
            }
            output[(size_t)co * oH * oW + (size_t)oh * oW + ow] = (int8_t)std::max(-128, std::min(127, q));
        }
        return;
    }

    int oH = (H + 2 * pad_h - kH) / stride_h + 1;
    int oW = (W + 2 * pad_w - kW) / stride_w + 1;
    int K     = kH * kW * C_in;
    int K_pad = (K + 3) / 4 * 4;
    int M     = oH * oW;
    int C_out_pad = (C_out + 3) / 4 * 4;

    auto mark = g_scratch.save();

    // im2col: [M, K_pad] INT8 on GPU
    int8_t* d_col = (int8_t*)g_scratch.alloc((size_t)M * K_pad);
    {
        dim3 block(16, 16);
        dim3 grid((M + block.x - 1) / block.x, (K_pad + block.y - 1) / block.y);
        k_im2col_nhwc<<<grid, block>>>(
            input, d_col,
            H, W, C_in, kH, kW, oH, oW,
            stride_h, stride_w, pad_h, pad_w,
            in_zp, K, K_pad);
        CUDA_CHECK(cudaGetLastError());
    }

    int64_t* d_bias  = g_const_cache.get(eff_bias, C_out);
    float*   d_scale = get_req_scale_gpu(req_mult, req_exp, C_out);

    dim3 block(16, 16);
    dim3 grid_rq((C_out + 15) / 16, (M + 15) / 16);

    // Native INT8 GEMM when K divisible by 4 (im2col col is padded to K_pad)
    // Use d_col [M, K_pad] and w_pre_packed [C_out_pad, K_pad]
    // ldb = K_pad (d_col stride), lda = K_pad (weight stride) — both mult of 4 ✓
    bool use_int8_tc = cublas_int8_tc_supported(C_out, M, K);

    if (use_int8_tc) {
        int32_t* d_C = (int32_t*)g_scratch.alloc((size_t)M * C_out_pad * sizeof(int32_t));
        int32_t alpha_i = 1, beta_i = 0;
        CUBLAS_CHECK(cublasGemmEx(
            get_cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            C_out, M, K,
            &alpha_i,
            w_pre_packed, CUDA_R_8I, K_pad,  // weight [C_out_pad, K_pad], lda=K_pad
            d_col,        CUDA_R_8I, K_pad,  // col [M, K_pad], ldb=K_pad
            &beta_i,
            d_C, CUDA_R_32I, C_out_pad,
            CUBLAS_COMPUTE_32I,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaGetLastError());

        // output is [M, C_out] NHWC
        k_requant_int32_to_int8<<<grid_rq, block>>>(
            d_C, d_bias, d_scale, out_zp, output, M, C_out, C_out_pad);
    } else {
        // FP32 fallback
        float* d_C    = (float*)g_scratch.alloc((size_t)M * C_out_pad * sizeof(float));
        float* d_W_fp = int8_to_fp32_gpu(w_pre_packed, (size_t)C_out_pad * K_pad);
        float* d_A_fp = int8_to_fp32_gpu(d_col, (size_t)M * K_pad);
        float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasGemmEx(
            get_cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            C_out, M, K,
            &alpha,
            d_W_fp, CUDA_R_32F, K_pad,
            d_A_fp, CUDA_R_32F, K_pad,
            &beta,
            d_C, CUDA_R_32F, C_out_pad,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaGetLastError());

        k_requant_fp32_to_int8<<<grid_rq, block>>>(
            d_C, d_bias, d_scale, out_zp, output, M, C_out, C_out_pad);
    }
    CUDA_CHECK(cudaGetLastError());

    g_scratch.restore(mark);
}

void cuda_conv2d_winograd_nhwc_int8(
    const int8_t*        input,
    const int8_t* const* w_wino_packed,     // [16] on GPU
    const float*         wino_weight_scale, // [16] CPU
    float                wino_input_scale,
    const int64_t*       eff_bias,          // [C_out] CPU
    const int32_t*       req_mult,
    const int32_t*       req_exp,
    const float*         req_scale,         // [C_out] CPU float req scales
    int8_t               in_zp,
    int8_t               out_zp,
    int8_t*              output,
    int C_in, int H, int W,
    int C_out,
    int8_t*              scratch_input_hat,  // [16 * num_tiles * C_in_pad] on GPU
    int32_t*             scratch_output_hat, // [16 * num_tiles * C_out] on GPU (used as float*)
    const int32_t* const* w_wino_row_sums,
    StreamHandle          /*stream*/)
{
    int tH = (H + 1) / 2, tW = (W + 1) / 2;
    int num_tiles = tH * tW;
    int C_in_pad  = (C_in + 3) / 4 * 4;
    int C_out_pad = (C_out + 3) / 4 * 4;

    // fwd_scale = wino_input_scale (same as CPU's >>2 = *0.25 convention).
    // CPU uses raw int8 values and divides by 4 (=wino_input_scale) in the transform,
    // so wino_eff_bias is computed for that convention. We match it here.
    float fwd_scale = (wino_input_scale > 0.f) ? wino_input_scale : 1.f;

    // Step 1: Input transform → scratch_input_hat [16, num_tiles, C_in_pad] INT8
    {
        dim3 block(16, 16);
        dim3 grid((num_tiles + block.x - 1) / block.x, (C_in + block.y - 1) / block.y);
        k_wino_input_transform<<<grid, block>>>(
            input, scratch_input_hat,
            H, W, C_in, C_in_pad, fwd_scale, in_zp, tH, tW);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 2: 16 GEMMs into scratch_output_hat (stored as FP32)
    float* acc = (float*)scratch_output_hat;
    bool use_int8_tc = cublas_int8_tc_supported(C_out, num_tiles, C_in);

    auto mark = g_scratch.save();

    for (int pos = 0; pos < 16; ++pos) {
        const int8_t* A_hat = scratch_input_hat + (size_t)pos * num_tiles * C_in_pad;
        float*        C_acc = acc + (size_t)pos * num_tiles * C_out_pad;
        // combined_scale = wino_weight_scale / wino_input_scale (matches CPU formula).
        // CPU: cs = wino_weight_scale[p] / wino_input_scale
        float combined_scale = wino_weight_scale[pos] / wino_input_scale;

        if (use_int8_tc) {
            // INT8 GEMM → INT32, then scale to FP32
            int32_t* d_C_i32 = (int32_t*)g_scratch.alloc((size_t)num_tiles * C_out_pad * sizeof(int32_t));
            int32_t alpha_i = 1, beta_i = 0;
            CUBLAS_CHECK(cublasGemmEx(
                get_cublas(),
                CUBLAS_OP_T, CUBLAS_OP_N,
                C_out, num_tiles, C_in,
                &alpha_i,
                w_wino_packed[pos], CUDA_R_8I, C_in_pad,
                A_hat,              CUDA_R_8I, C_in_pad,
                &beta_i,
                d_C_i32, CUDA_R_32I, C_out_pad,
                CUBLAS_COMPUTE_32I,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            CUDA_CHECK(cudaGetLastError());
            // Scale INT32 → FP32 in C_acc
            int total = num_tiles * C_out_pad;
            k_int32_scale_fp32<<<(total + 255) / 256, 256>>>(d_C_i32, C_acc, total, combined_scale);
            CUDA_CHECK(cudaGetLastError());
            // Reset scratch mark so we reuse memory for next pos
            g_scratch.restore(mark);
        } else {
            float* d_W_fp = int8_to_fp32_gpu(w_wino_packed[pos], (size_t)C_out_pad * C_in_pad);
            float* d_A_fp = int8_to_fp32_gpu(A_hat, (size_t)num_tiles * C_in_pad);
            float alpha = 1.f, beta = 0.f;
            CUBLAS_CHECK(cublasGemmEx(
                get_cublas(),
                CUBLAS_OP_T, CUBLAS_OP_N,
                C_out, num_tiles, C_in,
                &alpha,
                d_W_fp, CUDA_R_32F, C_in_pad,
                d_A_fp, CUDA_R_32F, C_in_pad,
                &beta,
                C_acc,  CUDA_R_32F, C_out_pad,
                CUBLAS_COMPUTE_32F_FAST_TF32,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            CUDA_CHECK(cudaGetLastError());
            if (combined_scale != 1.f) {
                int total = num_tiles * C_out_pad;
                k_scale_fp32<<<(total + 255) / 256, 256>>>(C_acc, C_acc, total, combined_scale);
                CUDA_CHECK(cudaGetLastError());
            }
            g_scratch.restore(mark);
        }
    }

    // Step 3: Output transform + requant
    int64_t* d_bias  = g_const_cache.get(eff_bias,  C_out);
    float*   d_scale = g_const_cache.get(req_scale, C_out);

    {
        dim3 block(16, 16);
        dim3 grid((num_tiles + block.x - 1) / block.x, (C_out + block.y - 1) / block.y);
        k_wino_output_transform<<<grid, block>>>(
            acc, d_bias, d_scale, out_zp,
            output, H, W, C_out, num_tiles, tW);
        CUDA_CHECK(cudaGetLastError());
    }

    g_scratch.restore(mark);
}

void cuda_conv2d_fp32_nhwc(
    const float* input,
    const float* weight,        // raw weights [C_out, kH, kW, C_in] CPU, NHWC filter order
    const float* bias,          // [C_out] CPU
    float*       output,
    int C_in, int H, int W,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    bool         relu,
    const float* w_packed,      // [C_out_pad, K_pad] on GPU (cuBLAS path only)
    float*       scratch_col,   // ignored, use GPU pool
    int dilation_h, int dilation_w,
    StreamHandle /*stream*/)
{
#ifdef HAVE_CUDNN
    // ── cuDNN path: Winograd / implicit-GEMM / TF32 ModelIOTensor Cores ──
    {
        float* d_weight = get_weight_gpu(weight, C_out * kH * kW * C_in);
        const auto& desc = get_conv_desc(C_in, H, W, C_out,
            kH, kW, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);

        auto mark = g_scratch.save();
        void* workspace = desc.workspace_size > 0
            ? g_scratch.alloc(desc.workspace_size) : nullptr;

        if (bias) {
            // Fused conv + bias + optional relu via cudnnConvolutionBiasActivationForward.
            // Eliminates 2 extra kernel launches vs separate bias/relu ops.
            float* d_bias = g_const_cache.get(bias, C_out);
            float alpha1 = 1.f, alpha2 = 0.f;
            CUDNN_CHECK(cudnnConvolutionBiasActivationForward(
                get_cudnn(), &alpha1,
                desc.in_desc,   input,
                desc.filt_desc, d_weight,
                desc.conv_desc, desc.algo,
                workspace, desc.workspace_size,
                &alpha2, desc.out_desc, output,   // alpha2=0: no residual add
                desc.bias_desc, d_bias,
                relu ? desc.act_relu : desc.act_identity,
                desc.out_desc, output));
        } else {
            float alpha = 1.f, beta = 0.f;
            CUDNN_CHECK(cudnnConvolutionForward(
                get_cudnn(), &alpha,
                desc.in_desc,   input,
                desc.filt_desc, d_weight,
                desc.conv_desc, desc.algo,
                workspace, desc.workspace_size,
                &beta, desc.out_desc, output));
            if (relu) {
                k_relu_fp32<<<(desc.oH * desc.oW * C_out + 255) / 256, 256>>>(
                    output, desc.oH * desc.oW * C_out);
                CUDA_CHECK(cudaGetLastError());
            }
        }
        g_scratch.restore(mark);
        return;
    }
#endif // HAVE_CUDNN

    // ── Fallback: GPU im2col + cuBLAS TF32 SGEMM ──
    int oH = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int oW = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1;
    int K     = kH * kW * C_in;
    int K_pad = (K + 3) / 4 * 4;
    int C_out_pad = (C_out + 3) / 4 * 4;
    int M = oH * oW;

    auto mark = g_scratch.save();

    float* d_col = (float*)g_scratch.alloc((size_t)M * K_pad * sizeof(float));
    {
        dim3 block(16, 16);
        dim3 grid((M + block.x - 1) / block.x, (K_pad + block.y - 1) / block.y);
        k_im2col_nhwc_fp32<<<grid, block>>>(
            input, d_col,
            H, W, C_in, kH, kW, oH, oW,
            stride_h, stride_w, pad_h, pad_w,
            dilation_h, dilation_w, K, K_pad);
        CUDA_CHECK(cudaGetLastError());
    }

    float alpha = 1.f, beta = 0.f;
    if (w_packed) {
        // Caller provided padded GPU weights [C_out_pad, K_pad] — use TF32 with aligned dims.
        // TF32 tensor ops require 4-element-aligned leading dimensions.
        // Use a padded scratch output [M, C_out_pad] then compact to output [M, C_out].
        float* d_out_pad = (C_out == C_out_pad)
            ? output
            : (float*)g_scratch.alloc((size_t)M * C_out_pad * sizeof(float));

        CUBLAS_CHECK(cublasGemmEx(
            get_cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            C_out_pad, M, K_pad,
            &alpha,
            w_packed,  CUDA_R_32F, K_pad,
            d_col,     CUDA_R_32F, K_pad,
            &beta,
            d_out_pad, CUDA_R_32F, C_out_pad,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaGetLastError());

        if (d_out_pad != output) {
            for (int row = 0; row < M; ++row)
                CUDA_CHECK(cudaMemcpy(output + row * C_out,
                                      d_out_pad + row * C_out_pad,
                                      C_out * sizeof(float),
                                      cudaMemcpyDeviceToDevice));
        }
    } else {
        // No pre-packed GPU weights (e.g. Conv1dFp32 fallback path).
        // Upload raw weight [C_out, K] and use standard FP32 SGEMM (no alignment requirement).
        float* d_w = get_weight_gpu(weight, C_out * K);
        CUBLAS_CHECK(cublasGemmEx(
            get_cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            C_out, M, K,
            &alpha,
            d_w,   CUDA_R_32F, K,
            d_col, CUDA_R_32F, K_pad,
            &beta,
            output, CUDA_R_32F, C_out,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT));
        CUDA_CHECK(cudaGetLastError());
    }

    if (bias) {
        float* d_bias = g_const_cache.get(bias, C_out);
        dim3 block(32, 8);
        dim3 grid((C_out + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        k_bias_add_rows<<<grid, block>>>(output, d_bias, M, C_out);
        CUDA_CHECK(cudaGetLastError());
    }
    if (relu) {
        k_relu_fp32<<<(M * C_out + 255) / 256, 256>>>(output, M * C_out);
        CUDA_CHECK(cudaGetLastError());
    }

    g_scratch.restore(mark);
}

// ── Conv1dInt8 GPU: FP32 im2col + INT8 seqgemm ──────────────────────────────
// im2col: expand [T, C_in] → [T_out, K] where K = kernel_size * C_in
static __global__ void k_im2col_fp32_1d(
    const float* __restrict__ input,  // [T, C_in]
    float* __restrict__ col,          // [T_out, K]
    int T, int C_in, int T_out, int K,
    int kernel_size, int stride, int padding, int dilation)
{
    int t_out = blockIdx.x * blockDim.x + threadIdx.x;
    int kidx  = blockIdx.y * blockDim.y + threadIdx.y;
    if (t_out >= T_out || kidx >= K) return;
    int k  = kidx / C_in;
    int ci = kidx % C_in;
    int t_in = t_out * stride + k * dilation - padding;
    col[(size_t)t_out * K + kidx] =
        (t_in >= 0 && t_in < T) ? input[(size_t)t_in * C_in + ci] : 0.f;
}

void cuda_conv1d_int8_nhwc(
    const float*   input,      // [T, C_in] FP32 on device
    const int8_t*  w_packed,   // [C_out, K_pad] INT8 on device
    const float*   w_scales,   // [C_out] FP32 CPU pointer
    const float*   bias,       // [C_out] FP32 CPU pointer, nullable
    float*         output,     // [T_out, C_out] FP32 on device
    int T, int C_in, int C_out,
    int kernel_size, int stride, int padding, int dilation,
    StreamHandle   /*stream*/)
{
    const int K     = kernel_size * C_in;
    const int T_out = (T + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1;
    const int K_pad = (K + 3) / 4 * 4;
    const int N_pad = (C_out + 3) / 4 * 4;

    auto mark = g_scratch.save();

    // 1. GPU im2col: [T, C_in] → [T_out, K]
    float* d_col = (float*)g_scratch.alloc((size_t)T_out * K * sizeof(float));
    {
        dim3 block(16, 16);
        dim3 grid((T_out + 15) / 16, (K + 15) / 16);
        k_im2col_fp32_1d<<<grid, block>>>(
            input, d_col, T, C_in, T_out, K,
            kernel_size, stride, padding, dilation);
        CUDA_CHECK(cudaGetLastError());
    }

    // 2. Per-row max-abs of [T_out, K] → [T_out]
    float* d_row_max = (float*)g_scratch.alloc((size_t)T_out * sizeof(float));
    {
        int bsz = 256;
        k_row_max_abs<<<T_out, bsz, bsz * sizeof(float)>>>(d_col, d_row_max, T_out, K);
        CUDA_CHECK(cudaGetLastError());
    }

    // 3. Per-row INT8 quantization: [T_out, K] → [T_out, K] INT8
    int8_t* d_int8 = (int8_t*)g_scratch.alloc((size_t)T_out * K);
    {
        int total = T_out * K;
        k_quantize_per_row<<<(total + 255) / 256, 256>>>(d_col, d_int8, d_row_max, T_out, K);
        CUDA_CHECK(cudaGetLastError());
    }

    // 4. INT8 GEMM: [T_out, K] × [C_out, K_pad]^T → INT32 [T_out, C_out]
    float* d_w_scales = g_const_cache.get(w_scales, C_out);
    if (cublas_int8_tc_supported(C_out, T_out, K)) {
        int32_t* d_C_i32 = (int32_t*)g_scratch.alloc((size_t)T_out * N_pad * sizeof(int32_t));
        int32_t alpha_i = 1, beta_i = 0;
        CUBLAS_CHECK(cublasGemmEx(
            get_cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            C_out, T_out, K,
            &alpha_i,
            w_packed, CUDA_R_8I, K_pad,
            d_int8,   CUDA_R_8I, K,
            &beta_i,
            d_C_i32,  CUDA_R_32I, N_pad,
            CUBLAS_COMPUTE_32I,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaGetLastError());

        // Dequantize with per-row input scale + per-channel weight scale
        dim3 block(16, 16);
        dim3 grid((C_out + 15) / 16, (T_out + 15) / 16);
        k_dequant_int32_perrow<<<grid, block>>>(d_C_i32, d_row_max, d_w_scales, output, T_out, C_out, N_pad);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // FP32 fallback for cuBLAS INT8 Tensor Core unsupported shapes.
        float* d_W_fp = int8_to_fp32_gpu(w_packed, (size_t)N_pad * K_pad);
        float* d_A_fp = int8_to_fp32_gpu(d_int8,   (size_t)T_out * K);
        float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasGemmEx(
            get_cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            C_out, T_out, K,
            &alpha,
            d_W_fp, CUDA_R_32F, K_pad,
            d_A_fp, CUDA_R_32F, K,
            &beta,
            output, CUDA_R_32F, C_out,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaGetLastError());

        // Apply per-row input scale + per-channel weight scale in-place
        dim3 block(16, 16);
        dim3 grid((C_out + 15) / 16, (T_out + 15) / 16);
        k_dequant_fp32_perrow<<<grid, block>>>(output, d_row_max, d_w_scales, T_out, C_out);
        CUDA_CHECK(cudaGetLastError());
    }

    // 5. Add bias
    if (bias) {
        float* d_bias = g_const_cache.get(bias, C_out);
        dim3 block(32, 8);
        dim3 grid((C_out + block.x - 1) / block.x, (T_out + block.y - 1) / block.y);
        k_bias_add_rows<<<grid, block>>>(output, d_bias, T_out, C_out);
        CUDA_CHECK(cudaGetLastError());
    }

    g_scratch.restore(mark);
}

void cuda_conv2d_winograd_nhwc_fp32(
    const float*         input,
    const float* const*  w_wino_packed,   // [16] on GPU, each [C_out_pad, C_in_pad] FP32
    const float*         bias,
    float*               output,
    int C_in, int H, int W, int C_out, bool relu,
    float*               scratch,         // [16 * num_tiles * max(C_in, C_out)] on GPU
    StreamHandle         /*stream*/,
    const float*         /*raw_weight*/)   // unused: custom Winograd is faster than cuDNN for 3×3
{
    int tH = (H + 1) / 2, tW = (W + 1) / 2;
    int num_tiles = tH * tW;
    int C_in_pad  = (C_in  + 3) / 4 * 4;
    int C_out_pad = (C_out + 3) / 4 * 4;

    auto mark = g_scratch.save();

    float* d_in_hat  = (float*)g_scratch.alloc((size_t)16 * num_tiles * C_in  * sizeof(float));
    float* d_out_hat = (float*)g_scratch.alloc((size_t)16 * num_tiles * C_out * sizeof(float));

    // Input transform
    {
        dim3 block(16, 16);
        dim3 grid((num_tiles + block.x - 1) / block.x, (C_in + block.y - 1) / block.y);
        k_wino_input_transform_fp32<<<grid, block>>>(
            input, d_in_hat, H, W, C_in, tH, tW);
        CUDA_CHECK(cudaGetLastError());
    }

    // 16 FP32 GEMMs with TF32 ModelIOTensor Cores
    for (int pos = 0; pos < 16; ++pos) {
        float* A_hat = d_in_hat  + (size_t)pos * num_tiles * C_in;
        float* C_acc = d_out_hat + (size_t)pos * num_tiles * C_out;
        float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasGemmEx(
            get_cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            C_out, num_tiles, C_in,
            &alpha,
            w_wino_packed[pos], CUDA_R_32F, C_in_pad,
            A_hat,              CUDA_R_32F, C_in,
            &beta,
            C_acc,              CUDA_R_32F, C_out,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaGetLastError());
    }

    // Output transform
    float* d_bias = g_const_cache.get(bias, C_out);
    {
        dim3 block(16, 16);
        dim3 grid((num_tiles + block.x - 1) / block.x, (C_out + block.y - 1) / block.y);
        k_wino_output_transform_fp32<<<grid, block>>>(
            d_out_hat, d_bias, relu, output, H, W, C_out, num_tiles, tW);
        CUDA_CHECK(cudaGetLastError());
    }

    g_scratch.restore(mark);
}
