// ops/attention.cu — CUDA SeqGEMM and Flash Attention ops
// Critical optimization: attention is now computed entirely on GPU.
// Previous implementation copied QKV back to CPU for the attention computation
// (O(N²*D) scalar loops), causing massive D2H + H2D transfer overhead.
// Now uses:
//   - cublasGemmStridedBatchedEx for batched Q×K^T and Attn×V
//   - k_softmax_2d GPU kernel with parallel reduction
//   - GPU scratch pool for all temporaries
#include "cuda_ops_common.cuh"
#include "cuda_ops.hpp"

void cuda_seqgemm_fp32(
    const float* input,
    const float* weight,
    const float* bias,
    float*       output,
    int N, int C_in, int C_out,
    bool         gelu,
    const float* w_packed,
    StreamHandle stream)
{
    cuda_sgemm_f32(input, w_packed, nullptr, output, false, N, C_in, C_out, false, stream);

    if (bias || gelu) {
        auto mark = g_scratch.save();
        float* d_bias = g_const_cache.get(bias, C_out);
        dim3 block(16, 16);
        dim3 grid((C_out + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        k_seqgemm_bias_act<<<grid, block>>>(output, d_bias, N, C_out, gelu);
        CUDA_CHECK(cudaGetLastError());
        g_scratch.restore(mark);
    }
}

void cuda_seqgemm_int8(
    const float*   input,
    const int8_t*  w_packed,
    const float*   w_scales,
    const int64_t* /*eff_zeros*/,
    const float*   bias,
    int8_t*        inp_scratch,
    float*         /*req_scratch*/,
    float*         output,
    int N, int C_in, int C_out,
    bool           gelu,
    StreamHandle   /*stream*/)
{
    int K_pad = (C_in  + 3) / 4 * 4;
    int N_pad = (C_out + 3) / 4 * 4;
    bool use_int8_tc = cublas_int8_tc_supported(C_out, N, C_in);

    auto mark = g_scratch.save();

    // 1. Find max-abs of input on GPU (per-tensor dynamic quantization)
    float* d_max_abs = (float*)g_scratch.alloc(sizeof(float));
    k_max_abs_1block<<<1, 1024>>>(input, d_max_abs, N * C_in);
    CUDA_CHECK(cudaGetLastError());

    // 2. Quantize input FP32 → INT8
    int total_in = N * C_in;
    k_quantize_with_dscale<<<(total_in + 255) / 256, 256>>>(
        input, inp_scratch, d_max_abs, total_in);
    CUDA_CHECK(cudaGetLastError());

    // 3. GEMM: INT8 input × INT8 weight → FP32 output (via INT32 accumulator or FP32)
    if (use_int8_tc) {
        // Native INT8 GEMM → INT32, then dequantize to FP32 output
        int32_t* d_C_i32 = (int32_t*)g_scratch.alloc((size_t)N * N_pad * sizeof(int32_t));
        int32_t alpha_i = 1, beta_i = 0;
        CUBLAS_CHECK(cublasGemmEx(
            get_cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            C_out, N, C_in,
            &alpha_i,
            w_packed,    CUDA_R_8I, K_pad,
            inp_scratch, CUDA_R_8I, C_in,
            &beta_i,
            d_C_i32, CUDA_R_32I, N_pad,
            CUBLAS_COMPUTE_32I,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaGetLastError());

        // Dequantize: output[n,c] = (float)d_C_i32[n*N_pad+c] * (max_abs/127) * w_scales[c]
        // Use strided kernel so N_pad != C_out is handled correctly
        float* d_w_scales = g_const_cache.get(w_scales, C_out);
        {
            dim3 block(16, 16);
            dim3 grid((C_out + block.x - 1) / block.x, (N + block.y - 1) / block.y);
            k_dequant_seqgemm_int32<<<grid, block>>>(
                d_C_i32, d_max_abs, d_w_scales, output, N, C_out, N_pad);
            CUDA_CHECK(cudaGetLastError());
        }
    } else {
        // FP32 fallback
        float* d_W_fp = int8_to_fp32_gpu(w_packed, (size_t)N_pad * K_pad);
        float* d_A_fp = int8_to_fp32_gpu(inp_scratch, (size_t)N * C_in);
        float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasGemmEx(
            get_cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            C_out, N, C_in,
            &alpha,
            d_W_fp, CUDA_R_32F, K_pad,
            d_A_fp, CUDA_R_32F, C_in,
            &beta,
            output, CUDA_R_32F, C_out,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaGetLastError());

        float* d_w_scales = g_const_cache.get(w_scales, C_out);
        {
            dim3 block(16, 16);
            dim3 grid((C_out + block.x - 1) / block.x, (N + block.y - 1) / block.y);
            k_dequant_seqgemm<<<grid, block>>>(output, d_max_abs, d_w_scales, N, C_out);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // 5. Add bias + optional GELU
    if (bias) {
        float* d_bias = g_const_cache.get(bias, C_out);
        dim3 block(16, 16);
        dim3 grid((C_out + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        k_seqgemm_bias_act<<<grid, block>>>(output, d_bias, N, C_out, gelu);
        CUDA_CHECK(cudaGetLastError());
    } else if (gelu) {
        k_gelu_fp32<<<(N * C_out + 255) / 256, 256>>>(output, N * C_out);
        CUDA_CHECK(cudaGetLastError());
    }

    g_scratch.restore(mark);
}

// Full GPU attention implementation using batched strided GEMM.
// Layout conventions (all row-major):
//   QKV input:  [N, 3*D]
//   Q, K, V:    [num_heads, N, head_dim]  (scratch intermediates)
//   S:          [num_heads, N, N]          (attention scores, softmax in-place)
//   O_heads:    [num_heads, N, head_dim]
//   out attn:   [N, D]  (written back to scratch_qkv for proj)
//
// cuBLAS column-major derivation:
//   S = Q × K^T  →  S_colmaj = K_colmaj^T × Q_colmaj
//     (CUBLAS_OP_T on K, CUBLAS_OP_N on Q, m=N, n=N, k=head_dim)
//   O = Attn × V →  O_colmaj = V_colmaj × Attn_colmaj
//     (CUBLAS_OP_N on V, CUBLAS_OP_N on Attn, m=head_dim, n=N, k=N)
void cuda_attention_flash_fp32(
    const float* input,
    const float* w_qkv,
    const float* b_qkv,
    const float* w_proj,
    const float* b_proj,
    float*       output,
    int N, int D, int num_heads,
    float*       scratch_qkv,       // [N, 3*D] on GPU — filled by QKV proj, reused as output
    const float* w_qkv_packed,
    const float* w_proj_packed,
    StreamHandle stream,
    bool         /* dq_attn_out */)
{
    // QKV projection: input [N, D] × [3D, D]^T → scratch_qkv [N, 3D]
    cuda_seqgemm_fp32(input, w_qkv, b_qkv, scratch_qkv, N, D, 3*D, false, w_qkv_packed, stream);

    int head_dim = D / num_heads;
    float scale  = 1.f / sqrtf((float)head_dim);

    auto mark = g_scratch.save();

    // Allocate Q, K, V: each [num_heads, N, head_dim]
    size_t qkv_bytes = (size_t)num_heads * N * head_dim * sizeof(float);
    float* d_Q = (float*)g_scratch.alloc(qkv_bytes);
    float* d_K = (float*)g_scratch.alloc(qkv_bytes);
    float* d_V = (float*)g_scratch.alloc(qkv_bytes);

    // Allocate S (attention scores): [num_heads, N, N]
    float* d_S = (float*)g_scratch.alloc((size_t)num_heads * N * N * sizeof(float));

    // Allocate O (per-head output): [num_heads, N, head_dim]
    float* d_O = (float*)g_scratch.alloc(qkv_bytes);

    // Step 1: Gather Q, K, V from scratch_qkv [N, 3D]
    {
        dim3 block(32, 4, 1);
        dim3 grid((head_dim + 31) / 32, (N + 3) / 4, num_heads);
        k_gather_qkv<<<grid, block>>>(scratch_qkv, d_Q, d_K, d_V, N, num_heads, head_dim);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 2: S = scale * Q × K^T  [num_heads, N, N]
    // In cuBLAS col-major: S_colmaj = scale * K_colmaj^T × Q_colmaj
    //   transa=T on K (lda=head_dim), transb=N on Q (ldb=head_dim)
    //   m=N, n=N, k=head_dim
    {
        float alpha = scale, beta = 0.f;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            get_cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, N, head_dim,
            &alpha,
            d_K, CUDA_R_32F, head_dim, (long long)N * head_dim,  // K batch stride
            d_Q, CUDA_R_32F, head_dim, (long long)N * head_dim,  // Q batch stride
            &beta,
            d_S, CUDA_R_32F, N, (long long)N * N,                // S batch stride
            num_heads,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 3: Softmax(S) in-place [num_heads * N, N]
    {
        int total_rows = num_heads * N;
        // Choose block size: nearest power-of-2 >= N, clamped to 256
        int bsz = 32;
        while (bsz < N && bsz < 256) bsz <<= 1;
        k_softmax_2d<<<total_rows, bsz, bsz * sizeof(float)>>>(d_S, total_rows, N);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 4: O = Attn × V  [num_heads, N, head_dim]
    // In cuBLAS col-major: O_colmaj = V_colmaj × Attn_colmaj
    //   transa=N on V (lda=head_dim), transb=N on Attn (ldb=N)
    //   m=head_dim, n=N, k=N
    {
        float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            get_cublas(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            head_dim, N, N,
            &alpha,
            d_V, CUDA_R_32F, head_dim, (long long)N * head_dim,  // V batch stride
            d_S, CUDA_R_32F, N,        (long long)N * N,          // Attn batch stride
            &beta,
            d_O, CUDA_R_32F, head_dim, (long long)N * head_dim,  // O batch stride
            num_heads,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 5: Scatter O heads → scratch_qkv [N, D] for projection
    {
        dim3 block(32, 4, 1);
        dim3 grid((head_dim + 31) / 32, (N + 3) / 4, num_heads);
        k_scatter_heads<<<grid, block>>>(d_O, scratch_qkv, N, num_heads, head_dim);
        CUDA_CHECK(cudaGetLastError());
    }

    g_scratch.restore(mark);

    // Step 6: Output projection: scratch_qkv [N, D] → output [N, D]
    cuda_seqgemm_fp32(scratch_qkv, w_proj, b_proj, output, N, D, D, false, w_proj_packed, stream);
}

void cuda_attention_flash_int8(
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
    StreamHandle   stream)
{
    // QKV via seqgemm_int8 → scratch_qkv [N, 3D] FP32
    cuda_seqgemm_int8(input, w_qkv_packed, w_qkv_scales, eff_zeros_qkv, b_qkv,
                      scratch_int8, scratch_req, scratch_qkv,
                      N, D, 3*D, false, stream);

    int head_dim = D / num_heads;
    float scale  = 1.f / sqrtf((float)head_dim);

    auto mark = g_scratch.save();

    size_t qkv_bytes = (size_t)num_heads * N * head_dim * sizeof(float);
    float* d_Q = (float*)g_scratch.alloc(qkv_bytes);
    float* d_K = (float*)g_scratch.alloc(qkv_bytes);
    float* d_V = (float*)g_scratch.alloc(qkv_bytes);
    float* d_S = (float*)g_scratch.alloc((size_t)num_heads * N * N * sizeof(float));
    float* d_O = (float*)g_scratch.alloc(qkv_bytes);

    // Gather QKV
    {
        dim3 block(32, 4, 1);
        dim3 grid((head_dim + 31) / 32, (N + 3) / 4, num_heads);
        k_gather_qkv<<<grid, block>>>(scratch_qkv, d_Q, d_K, d_V, N, num_heads, head_dim);
        CUDA_CHECK(cudaGetLastError());
    }

    // S = scale * Q × K^T
    {
        float alpha = scale, beta = 0.f;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            get_cublas(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, N, head_dim,
            &alpha,
            d_K, CUDA_R_32F, head_dim, (long long)N * head_dim,
            d_Q, CUDA_R_32F, head_dim, (long long)N * head_dim,
            &beta,
            d_S, CUDA_R_32F, N, (long long)N * N,
            num_heads,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaGetLastError());
    }

    // Softmax
    {
        int total_rows = num_heads * N;
        int bsz = 32;
        while (bsz < N && bsz < 256) bsz <<= 1;
        k_softmax_2d<<<total_rows, bsz, bsz * sizeof(float)>>>(d_S, total_rows, N);
        CUDA_CHECK(cudaGetLastError());
    }

    // O = Attn × V
    {
        float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            get_cublas(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            head_dim, N, N,
            &alpha,
            d_V, CUDA_R_32F, head_dim, (long long)N * head_dim,
            d_S, CUDA_R_32F, N,        (long long)N * N,
            &beta,
            d_O, CUDA_R_32F, head_dim, (long long)N * head_dim,
            num_heads,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaGetLastError());
    }

    // Scatter heads → scratch_qkv [N, D]
    {
        dim3 block(32, 4, 1);
        dim3 grid((head_dim + 31) / 32, (N + 3) / 4, num_heads);
        k_scatter_heads<<<grid, block>>>(d_O, scratch_qkv, N, num_heads, head_dim);
        CUDA_CHECK(cudaGetLastError());
    }

    g_scratch.restore(mark);

    // Output projection via seqgemm_int8
    cuda_seqgemm_int8(scratch_qkv, w_proj_packed, w_proj_scales, eff_zeros_proj, b_proj,
                      scratch_int8, scratch_req, output,
                      N, D, D, false, stream);
}

// ─────────────────────────────────────────────────────────────────
// LM-specific ops: INT8 GEMV (single-token decode) and causal attention
// ─────────────────────────────────────────────────────────────────

// INT8 matrix-vector product: output[co] = sum_k(float(W[co,k]) * input[k]) * w_scales[co] + bias[co]
// Input is FP32 on GPU; weights are INT8 on GPU (already uploaded via upload_weights).
// One block per output channel, threads reduce over C_in via warp shuffles.
static __global__ void k_lm_int8_matvec(
    const float*   __restrict__ input,
    const int8_t*  __restrict__ weight,
    const float*   __restrict__ w_scales,
    const float*   __restrict__ bias,
    float*         __restrict__ output,
    int C_in, int C_out)
{
    const int co = blockIdx.x;
    if (co >= C_out) return;
    const int8_t* w = weight + (size_t)co * C_in;

    float sum = 0.f;
    for (int k = threadIdx.x; k < C_in; k += blockDim.x)
        sum += (float)w[k] * input[k];

    // Warp-level reduction
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    // Cross-warp reduction via shared memory
    __shared__ float sdata[32];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) sdata[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        sum = (lane < nwarps) ? sdata[lane] : 0.f;
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
        if (lane == 0)
            output[co] = sum * w_scales[co] + (bias ? bias[co] : 0.f);
    }
}

void cuda_seqgemm_int8_matvec(
    const float*   input,     // [C_in] GPU
    const int8_t*  weight_i8, // [C_out, C_in] GPU (uploaded in upload_weights)
    const float*   w_scales,  // [C_out] CPU host pointer — uploaded via const cache
    const float*   bias,      // [C_out] CPU (may be nullptr) — uploaded via const cache
    float*         output,    // [C_out] GPU
    int C_in, int C_out,
    StreamHandle   /*stream*/)
{
    // w_scales is already a GPU pointer (uploaded by caller via upload_weights)
    float* d_bias = bias ? g_const_cache.get(bias, C_out) : nullptr;
    k_lm_int8_matvec<<<C_out, 128>>>(input, weight_i8, w_scales, d_bias, output, C_in, C_out);
    CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────
// Causal attention with RoPE and KV cache
// ─────────────────────────────────────────────────────────────────

// Apply RoPE to Q (→ q_roped scratch) and to K (→ KV-cache K slot).
// Each thread handles one (token, head, pair_i) for either Q or K.
static __global__ void k_lm_rope_kvcache_q(
    const float* __restrict__ q_in,     // [N, nq*hd]
    const float* __restrict__ k_in,     // [N, nkv*hd]
    float*       __restrict__ q_roped,  // [N, nq*hd]  scratch output
    float*       __restrict__ kv_cache, // [2, cache_len, nkv*hd]
    int N, int nq, int nkv, int hd2,    // hd2 = head_dim/2
    int cache_pos, int cache_len, int step_pos, float rope_theta)
{
    int hd = hd2 * 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_q = N * nq * hd2;
    int total   = total_q + N * nkv * hd2;
    if (idx >= total) return;

    bool is_q = (idx < total_q);
    int local  = is_q ? idx : (idx - total_q);
    int nheads = is_q ? nq : nkv;

    int t = local / (nheads * hd2);
    int h = (local % (nheads * hd2)) / hd2;
    int i = local % hd2;

    int pos = step_pos + t;
    // theta_i = pos / (rope_theta ^ (2i/hd))
    float theta = (float)pos * __expf(-(float)(2 * i) / (float)hd * __logf(rope_theta));
    float cos_t = __cosf(theta), sin_t = __sinf(theta);

    if (is_q) {
        float v0 = q_in[(size_t)t * nq * hd + h * hd + i];
        float v1 = q_in[(size_t)t * nq * hd + h * hd + i + hd2];
        q_roped[(size_t)t * nq * hd + h * hd + i]      = v0 * cos_t - v1 * sin_t;
        q_roped[(size_t)t * nq * hd + h * hd + i + hd2] = v0 * sin_t + v1 * cos_t;
    } else {
        // Write RoPE(K) to KV-cache K slot
        int pos_cache = cache_pos + t;
        float v0 = k_in[(size_t)t * nkv * hd + h * hd + i];
        float v1 = k_in[(size_t)t * nkv * hd + h * hd + i + hd2];
        kv_cache[(size_t)pos_cache * nkv * hd + h * hd + i]      = v0 * cos_t - v1 * sin_t;
        kv_cache[(size_t)pos_cache * nkv * hd + h * hd + i + hd2] = v0 * sin_t + v1 * cos_t;
    }
}

// Write V to KV-cache V slot (no transformation needed).
static __global__ void k_lm_v_write(
    const float* __restrict__ v_in,  // [N, nkv*hd]
    float*       __restrict__ kv_v,  // KV-cache V slot: [cache_len, nkv*hd]
    int N, int nkv, int hd, int cache_pos)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * nkv * hd) return;
    int t = idx / (nkv * hd);
    int j = idx % (nkv * hd);
    kv_v[(size_t)(cache_pos + t) * nkv * hd + j] = v_in[(size_t)t * nkv * hd + j];
}

// Compute attention scores Q·K^T for all (token, query_head) pairs.
// Handles GQA grouping: kvh = qh * nkv / nq.
// For prefill, causal mask is applied: positions >= causal_end get -1e30.
static __global__ void k_lm_attn_scores(
    const float* __restrict__ q_roped,  // [N, nq*hd]
    const float* __restrict__ kv_k,     // [cache_len, nkv*hd]
    float*       __restrict__ scores,   // [N*nq, total_kv]
    int N, int nq, int nkv, int hd, int total_kv, int cache_pos, float scale)
{
    // One block per (token, query_head); threads stride over total_kv keys.
    int tok_qh = blockIdx.x;
    int t  = tok_qh / nq;
    int qh = tok_qh % nq;
    if (t >= N) return;

    int kvh = (int)((long long)qh * nkv / nq);
    int causal_end = cache_pos + t + 1;  // causal: attend to [0, causal_end)
    const float* q = q_roped + (size_t)t * nq * hd + qh * hd;
    float* s = scores + (size_t)tok_qh * total_kv;

    for (int kp = threadIdx.x; kp < total_kv; kp += blockDim.x) {
        if (kp < causal_end) {
            const float* k = kv_k + (size_t)kp * nkv * hd + kvh * hd;
            float dot = 0.f;
            for (int d = 0; d < hd; ++d) dot += q[d] * k[d];
            s[kp] = dot * scale;
        } else {
            s[kp] = -1e30f;  // causal mask
        }
    }
}

// Accumulate attention-weighted V values into output.
// scores[N*nq, total_kv] contains softmax weights; output[N, nq*hd].
static __global__ void k_lm_attn_output(
    const float* __restrict__ scores,  // [N*nq, total_kv]
    const float* __restrict__ kv_v,    // [cache_len, nkv*hd]
    float*       __restrict__ output,  // [N, nq*hd]
    int N, int nq, int nkv, int hd, int total_kv, int /*cache_len*/)
{
    int tok_qh = blockIdx.x;
    int t  = tok_qh / nq;
    int qh = tok_qh % nq;
    if (t >= N) return;

    int kvh = (int)((long long)qh * nkv / nq);
    const float* s = scores + (size_t)tok_qh * total_kv;
    float* out_h = output + (size_t)t * nq * hd + qh * hd;

    // Each thread computes one output dimension
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.f;
        for (int kp = 0; kp < total_kv; ++kp)
            acc += s[kp] * kv_v[(size_t)kp * nkv * hd + kvh * hd + d];
        out_h[d] = acc;
    }
}

void cuda_causal_attn_core_fp32(
    const float* q_in, const float* k_in, const float* v_in,
    float* kv_cache,
    int cache_pos, int cache_len,
    float* output,
    int N, int nq, int nkv, int hd,
    int step_pos, float rope_theta, bool /*is_decode*/,
    float* /*cpu_scratch*/,   // ignored — GPU uses internal scratch
    StreamHandle /*stream*/)
{
    const int total_kv = cache_pos + N;
    const float scale  = rsqrtf((float)hd);

    auto mark = g_scratch.save();

    // Scratch: RoPE'd Q [N, nq*hd] and attention scores [N*nq, total_kv]
    float* d_q_roped = (float*)g_scratch.alloc((size_t)N * nq * hd * sizeof(float));
    float* d_scores  = (float*)g_scratch.alloc((size_t)N * nq * total_kv * sizeof(float));

    // Step 1: RoPE on Q → d_q_roped;  RoPE on K → KV-cache K slot
    {
        int hd2 = hd / 2;
        int total_threads = N * (nq + nkv) * hd2;
        k_lm_rope_kvcache_q<<<(total_threads + 255) / 256, 256>>>(
            q_in, k_in, d_q_roped, kv_cache,
            N, nq, nkv, hd2, cache_pos, cache_len, step_pos, rope_theta);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 2: V → KV-cache V slot
    {
        float* kv_v = kv_cache + (size_t)cache_len * nkv * hd;
        int total_threads = N * nkv * hd;
        k_lm_v_write<<<(total_threads + 255) / 256, 256>>>(
            v_in, kv_v, N, nkv, hd, cache_pos);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 3: Compute attention scores [N*nq, total_kv] with causal mask
    {
        k_lm_attn_scores<<<N * nq, 128>>>(
            d_q_roped, kv_cache, d_scores,
            N, nq, nkv, hd, total_kv, cache_pos, scale);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 4: Softmax over each row (scores are [N*nq, total_kv])
    {
        int bsz = 256;  // handles any total_kv via the loop in k_softmax_2d
        k_softmax_2d<<<N * nq, bsz, bsz * sizeof(float)>>>(d_scores, N * nq, total_kv);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 5: Weighted sum of V → output [N, nq*hd]
    {
        const float* kv_v = kv_cache + (size_t)cache_len * nkv * hd;
        k_lm_attn_output<<<N * nq, 64>>>(
            d_scores, kv_v, output,
            N, nq, nkv, hd, total_kv, cache_len);
        CUDA_CHECK(cudaGetLastError());
    }

    g_scratch.restore(mark);
}
