// src/backend/backend.hpp
#pragma once
#include "engine/buffer.hpp"
#include <cstdint>
#include <cstddef>
#include <cstring>

// Opaque stream handle.
//   CUDA:  cudaStream_t  cast to void*
//   HIP:   hipStream_t   cast to void*
//   Metal: id<MTLCommandBuffer> cast to void* (in .mm files)
//   CPU:   nullptr (ignored)
#ifndef STREAM_HANDLE_DEFINED
#define STREAM_HANDLE_DEFINED
using StreamHandle = void*;
#endif

// ─────────────────────────────────────────────────────────────────
// Allocator: backend-specific memory management.
// ─────────────────────────────────────────────────────────────────
struct Allocator {
    virtual MemSpace mem_space()  const = 0;
    virtual void*    alloc(size_t bytes) = 0;
    virtual void     free (void* ptr)    = 0;
    // DMA transfers
    virtual void copy_h2d(void* dst, const void* src, size_t n) = 0;
    virtual void copy_d2h(void* dst, const void* src, size_t n) = 0;
    virtual void copy_d2d(void* dst, const void* src, size_t n) = 0;
    // Block until all queued work for this allocator's device is complete.
    virtual void sync() = 0;
    // Zero a block of memory in this backend's memory space.
    virtual void zero_memory(void* ptr, size_t n) { memset(ptr, 0, n); }
    virtual ~Allocator() = default;

    // Convenience: allocate an owning Buffer in this backend's memory space.
    Buffer make_buffer(size_t bytes) {
        if (bytes == 0) bytes = 4;   // guard: some backends reject zero-size allocs
        void* p = alloc(bytes);
        return Buffer(p, bytes, mem_space(),
                            [this](void* q){ this->free(q); });
    }
};

// ─────────────────────────────────────────────────────────────────
// BackendOps: one flat vtable per backend.
// Function signatures match ops_neon.hpp exactly, extended with
// StreamHandle for async dispatch.
// ─────────────────────────────────────────────────────────────────
struct BackendOps {
    const char* name;       // "cpu" | "cuda" | "hip" | "metal"
    int         priority;   // higher = preferred when multiple are registered
    MemSpace    mem_space;  // where this backend allocates activation buffers

    // ── Allocator factory ─────────────────────────────────────────
    Allocator* (*make_allocator)();

    // ── Weight packing ────────────────────────────────────────────
    int8_t* (*pack_weights_int8)(const int8_t* w, int C_out, int K);
    float*  (*pack_weights_fp32)(const float*  w, int C_out, int K);
    void    (*free_packed_int8) (int8_t* p);
    void    (*free_packed_fp32) (float*  p);

    // ── INT8 GEMM ─────────────────────────────────────────────────
    void (*gemm_int8)(
        const int8_t*  A,
        const int8_t*  B_packed,
        const int64_t* eff_bias,
        const int32_t* req_mult,
        const int32_t* req_exp,
        const float*   req_scale_f,
        int8_t         out_zp,
        void*          C,
        bool           is_float,
        int M, int K, int N,
        bool           nchw_out,
        bool           in_parallel,
        const int32_t* b_row_sums,
        StreamHandle   stream
    );

    // ── INT8 GEMM → INT32 accumulators (Winograd domain) ─────────
    void (*gemm_int8_int32)(
        const int8_t*  A,
        const int8_t*  B_packed,
        int32_t*       C,
        int M, int K, int N,
        const int32_t* b_row_sums,
        StreamHandle   stream
    );

    // ── FP32 SGEMM ────────────────────────────────────────────────
    void (*sgemm_f32)(
        const float* A,
        const float* B_packed,
        const float* bias,
        float*       C,
        bool         relu,
        int M, int K, int N,
        bool         in_parallel,
        StreamHandle stream
    );

    // Single-vector GEMM (GemmFp32 / GemmInt8 final layer)
    void (*gemm_fp32_vec)(
        const float* input,
        const float* weight,
        const float* bias,
        float*       output,
        int C_in, int C_out,
        const float* w_packed,
        StreamHandle stream
    );

    // ── INT8 Conv2D ───────────────────────────────────────────────
    void (*conv2d_int8)(
        const int8_t*  input,
        const int8_t*  weight,
        const int8_t*  w_pre_packed,
        const int64_t* eff_bias,
        const int32_t* req_mult,
        const int32_t* req_exp,
        int8_t         in_zp,
        int8_t         out_zp,
        int8_t*        output,
        int C_in, int H, int W,
        int C_out, int kH, int kW,
        int stride_h, int stride_w,
        int pad_h,    int pad_w,
        int groups,
        int8_t*        scratch_col,
        bool           nhwc,
        const int32_t* b_row_sums,
        StreamHandle   stream
    );

    // ── Winograd F(2,3) INT8 Conv ─────────────────────────────────
    void (*conv2d_winograd_nhwc_int8)(
        const int8_t*        input,
        const int8_t* const* w_wino_packed,
        const float*         wino_weight_scale,
        float                wino_input_scale,
        const int64_t*       eff_bias,
        const int32_t*       req_mult,
        const int32_t*       req_exp,
        const float*         req_scale,
        int8_t               in_zp,
        int8_t               out_zp,
        int8_t*              output,
        int C_in, int H, int W,
        int C_out,
        int8_t*              scratch_input_hat,
        int32_t*             scratch_output_hat,
        const int32_t* const* w_wino_row_sums,
        StreamHandle          stream
    );

    // ── FP32 Conv2D (NHWC) ────────────────────────────────────────
    void (*conv2d_fp32_nhwc)(
        const float* input,
        const float* weight,
        const float* bias,
        float*       output,
        int C_in, int H, int W,
        int C_out, int kH, int kW,
        int stride_h, int stride_w,
        int pad_h,    int pad_w,
        bool         relu,
        const float* w_packed,
        float*       scratch_col,
        int dilation_h, int dilation_w,
        StreamHandle stream
    );

    // ── FP32 Winograd Conv ────────────────────────────────────────
    // raw_weight: [C_out, kH, kW, C_in] on CPU (NHWC filter order) — used by CUDA+cuDNN path
    void (*conv2d_winograd_nhwc_fp32)(
        const float*         input,
        const float* const*  w_wino_packed,
        const float*         bias,
        float*               output,
        int C_in, int H, int W, int C_out, bool relu,
        float*               scratch,
        StreamHandle         stream,
        const float*         raw_weight    // nullable; CPU weights for cuDNN path
    );

    // ── Sequence GEMM [N, C_in] → [N, C_out] ─────────────────────
    void (*seqgemm_fp32)(
        const float* input,
        const float* weight,
        const float* bias,
        float*       output,
        int N, int C_in, int C_out,
        bool         gelu,
        const float* w_packed,
        StreamHandle stream
    );

    // ── INT8 Sequence GEMM ────────────────────────────────────────
    void (*seqgemm_int8)(
        const float*   input,
        const int8_t*  w_packed,
        const float*   w_scales,
        const int64_t* eff_zeros,
        const float*   bias,
        int8_t*        inp_scratch,
        float*         req_scratch,
        float*         output,
        int N, int C_in, int C_out,
        bool           gelu,
        StreamHandle   stream
    );

    // ── INT8 GEMV for single-token decode (N=1) ───────────────────
    // Takes raw (unpacked) INT8 weights [C_out, C_in] row-major.
    // Nullable: GPU backends use seqgemm_int8 for all N.
    void (*seqgemm_int8_matvec)(
        const float*   input,      // [C_in] FP32
        const int8_t*  weight_i8,  // [C_out, C_in] INT8 row-major
        const float*   w_scales,   // [C_out] per-channel FP32
        const float*   bias,       // [C_out] FP32 or nullptr
        float*         output,     // [C_out] FP32
        int C_in, int C_out,
        StreamHandle   stream
    );

    // ── INT8 1D Conv (GPU: FP32 I/O, GPU im2col + INT8 GEMM) ──────
    // Nullable: only set by GPU backends. CPU Conv1dInt8 uses its own path.
    void (*conv1d_int8_nhwc)(
        const float*   input,      // [T, C_in] FP32 on device
        const int8_t*  w_packed,   // [C_out, K_pad] INT8 on device
        const float*   w_scales,   // [C_out] FP32 CPU pointer (cached on device)
        const float*   bias,       // [C_out] FP32 CPU pointer, nullable
        float*         output,     // [T_out, C_out] FP32 on device
        int T, int C_in, int C_out,
        int kernel_size, int stride, int padding, int dilation,
        StreamHandle   stream
    ) = nullptr;

    // ── FP32 1D Conv (GPU: im2col + MPS MatMul, flush-after) ──────
    // Nullable: only set by backends that support GPU FP32 1D conv.
    void (*conv1d_fp32_nhwc)(
        const float* input,    // [T, C_in]
        const float* weight,   // [C_out, kernel_size, C_in] NHWC
        const float* bias,     // [C_out] nullable
        float*       output,   // [T_out, C_out]
        int C_in, int T, int C_out,
        int kernel_size, int stride, int padding, int dilation,
        bool relu,
        StreamHandle stream
    ) = nullptr;

    // ── FP32 Transposed 1D Conv (GPU: GEMM-per-k + scatter-add) ─
    // groups==1: weight layout [kernel_size, C_out, C_in] row-major.
    // depthwise (groups==C_in==C_out): weight layout [C, kernel_size] row-major.
    void (*conv_transpose1d_fp32_nhwc)(
        const float* input,    // [T, C_in]
        const float* weight,   // GPU-packed weight (layout depends on groups)
        const float* bias,     // [C_out] nullable
        float*       output,   // [T_out, C_out]
        int C_in, int T, int C_out,
        int kernel_size, int stride, int padding, int output_padding,
        int groups,
        StreamHandle stream
    ) = nullptr;

    // ── Flash Attention FP32 ──────────────────────────────────────
    void (*attention_flash_fp32)(
        const float* input,
        const float* w_qkv,
        const float* b_qkv,
        const float* w_proj,
        const float* b_proj,
        float*       output,
        int N, int D, int num_heads,
        float*       scratch_qkv,
        const float* w_qkv_packed,
        const float* w_proj_packed,
        StreamHandle stream,
        bool         dq_attn_out
    );

    // ── INT8 Flash Attention ──────────────────────────────────────
    void (*attention_flash_int8)(
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
        StreamHandle   stream
    );

    // ── Layer Norm ────────────────────────────────────────────────
    void (*layernorm_fp32)(
        const float* input,
        const float* gamma,
        const float* beta,
        float*       output,
        int N, int C,
        float        eps,
        StreamHandle stream
    );

    // ── Pool ops ──────────────────────────────────────────────────
    void (*maxpool_int8)(
        const int8_t*, int8_t*,
        int C, int H, int W, int kH, int kW,
        int sh, int sw, int ph, int pw,
        StreamHandle);

    void (*maxpool_int8_nhwc)(
        const int8_t*, int8_t*,
        int C, int H, int W, int kH, int kW,
        int sh, int sw, int ph, int pw,
        StreamHandle);

    void (*maxpool_fp32_nhwc)(
        const float*, float*,
        int C, int H, int W, int kH, int kW,
        int sh, int sw, int ph, int pw,
        StreamHandle);

    void (*avgpool_global_int8)(
        const int8_t*, float in_sc, int in_zp, float out_sc, int out_zp,
        int8_t*, int C, int H, int W,
        StreamHandle);

    void (*avgpool_global_int8_nhwc)(
        const int8_t*, float in_sc, int in_zp, float out_sc, int out_zp,
        int8_t*, int C, int H, int W,
        StreamHandle);

    void (*avgpool_global_fp32_nhwc)(
        const float*, float*, int C, int H, int W,
        StreamHandle);

    // ── Residual Add ──────────────────────────────────────────────
    void (*add_fp32)(const float*, const float*, float*, int, bool relu, StreamHandle);
    void (*add_requant_int8)(
        const int8_t*, const int8_t*,
        float in1_sc, int in1_zp, float in2_sc, int in2_zp,
        float out_sc, int out_zp,
        int8_t*, int N,
        StreamHandle);

    // ── Elementwise activations ───────────────────────────────────
    void (*relu_fp32)      (float*, int, StreamHandle);
    void (*gelu_fp32)      (float*, int, StreamHandle);
    void (*sigmoid_fp32)   (const float*, float*, int, StreamHandle);
    void (*tanh_fp32)      (const float*, float*, int, StreamHandle);
    void (*leaky_relu_fp32)(const float*, float*, int, float alpha, StreamHandle);
    void (*exp_fp32)       (const float*, float*, int, StreamHandle);
    void (*sin_fp32)       (const float*, float*, int, StreamHandle);

    // ── Upsample ──────────────────────────────────────────────────
    void (*upsample_nearest1d_fp32)(const float*, float*, int T, int C, int sf, StreamHandle);
    void (*upsample_bilinear2d_fp32)(const float*, float*,
                                     int C, int iH, int iW, int oH, int oW, StreamHandle);

    // ── Norm ops ──────────────────────────────────────────────────
    void (*ada_in1d_fp32)(const float* feat, float* out,
                           const float* gamma, const float* beta,
                           int T, int C, float eps, StreamHandle,
                           const float* norm_weight,   // may be nullptr -> default 1
                           const float* norm_bias);    // may be nullptr -> default 0
    void (*ada_layer_norm1d_fp32)(const float* feat, float* out,
                                   const float* gamma, const float* beta,
                                   int T, int C, float eps, StreamHandle);

    // ── Vector utilities ──────────────────────────────────────────
    void (*add_vectors_fp32)   (float* dst, const float* src, int N, StreamHandle);
    void (*bias_add_rows_fp32) (float* out, const float* bias, int T, int C, StreamHandle);
    void (*scale_fp32)         (const float*, float*, size_t N, float scalar, StreamHandle);
    void (*sum_channels_fp32)  (const float*, float*, int T, int C, StreamHandle);
    void (*relu_inplace_fp32)  (float*, int N, StreamHandle);

    // ── Embedding / tensor layout ─────────────────────────────────
    void (*embedding_lookup_fp32)(const float* ids, const float* weight, float* out,
                                   int T, int num_embeddings, int emb_dim, StreamHandle);
    void (*concat1d_fp32)(const float*, const float*, float*, int T, int C1, int C2,
                           bool broadcast, StreamHandle);
    void (*slice_channels_fp32)(const float*, float*, int T, int C_in,
                                 int ch_start, int C_out, StreamHandle);

    // ── Snake1D activation (x + sin²(α·x)/α, per-channel α) ─────
    void (*snake1d_fp32)(const float* in, float* out,
                          const float* alpha, const float* inv_alpha,
                          int T, int C, StreamHandle);

    // ── LSTM full forward (input projection + recurrence scan) ─────────
    // Nullable: only Metal sets this.
    // w_ih[4H, C_in], w_hh[4H, H] — raw row-major layout (not CPU-packed).
    void (*lstm_fp32_metal)(
        const float* input,    // [T, C_in]
        const float* w_ih,     // [4H, C_in] GPU
        const float* b_ih,     // [4H] GPU, nullable
        const float* w_hh,     // [4H, H] GPU
        const float* b_hh,     // [4H] GPU, nullable
        const float* w_ih_r,   // [4H, C_in] GPU, nullptr if unidirectional
        const float* b_ih_r,   // [4H] GPU, nullable
        const float* w_hh_r,   // [4H, H] GPU, nullable
        const float* b_hh_r,   // [4H] GPU, nullable
        float*       output,   // [T, H*(1+bidir)]
        int T, int C_in, int H,
        bool bidirectional,
        StreamHandle stream
    ) = nullptr;

    // ── STFT ───────────────────────────────────────────────────────────
    void (*stft_fp32_metal)(
        const float* audio,    // [T_audio] (raw, no padding)
        const float* w_real,   // [K, n_fft] GPU
        const float* w_imag,   // [K, n_fft] GPU
        float*       out,      // [T_stft, 2*K]  interleaved [mag|phase] per frame
        int T_audio, int n_fft, int hop_size,
        StreamHandle stream
    ) = nullptr;

    // ── ISTFT ──────────────────────────────────────────────────────────
    void (*istft_fp32_metal)(
        const float* mag,      // [T_frames, K]
        const float* phase,    // [T_frames, K]
        float*       out,      // [out_len]  out_len = (T_frames-1)*hop_size
        int T_frames, int n_fft, int hop_size, bool normalized,
        StreamHandle stream
    ) = nullptr;

    // ── Reflection pad 1D (left-pad one reflected sample) ──────────────
    void (*reflection_pad1d_fp32_metal)(
        const float* in,       // [T, C]
        float*       out,      // [T+1, C]
        int T, int C,
        StreamHandle stream
    ) = nullptr;

    // ── SineGen (ONNX-compatible: upsample_scale > 0, no noise) ────────
    void (*sine_gen_fp32_metal)(
        const float* f0,       // [T]
        float*       out,      // [T, H]  H = harmonic_num + 1
        int T, int H, int sr, int upsample_scale,
        float sine_amp, float voiced_threshold,
        StreamHandle stream
    ) = nullptr;

    // ── LM ops (nullable — CPU-only layers use nullptr for GPU backends) ──

    // RMSNorm: y = x / rms(x) * gamma,  where rms(x) = sqrt(mean(x^2) + eps)
    // input/output: [N, C] — N tokens, C channels
    void (*rmsnorm_fp32)(
        const float* input,
        const float* gamma,    // [C] learnable scale
        float*       output,
        int N, int C,
        float        eps,
        StreamHandle stream
    ) = nullptr;

    // SiLU activation: out[i] = in[i] * sigmoid(in[i])
    void (*silu_fp32)(
        const float* input,
        float*       output,
        int          n_elems,
        StreamHandle stream
    ) = nullptr;

    // Elementwise multiply: out[i] = a[i] * b[i]
    void (*elemwise_mul_fp32)(
        const float* a,
        const float* b,
        float*       output,
        int          n_elems,
        StreamHandle stream
    ) = nullptr;

    // Causal self-attention core: RoPE + KV cache + softmax attention
    // Q:   [N, num_q_heads  * head_dim]  (FP32, row-major)
    // K,V: [N, num_kv_heads * head_dim]  (FP32, row-major)
    // kv_cache: [2, cache_len, num_kv_heads * head_dim]  read+write
    //   kv_cache[0] = K cache,  kv_cache[1] = V cache
    // step_pos: absolute position of the first token (for RoPE angle)
    // output: [N, num_q_heads * head_dim]
    void (*causal_attn_core_fp32)(
        const float* q,          // [N, nq * head_dim]
        const float* k,          // [N, nkv * head_dim]
        const float* v,          // [N, nkv * head_dim]
        float*       kv_cache,   // [2, cache_len, nkv * head_dim]
        int          cache_pos,  // write position (= step_pos for decode, 0 for prefill start)
        int          cache_len,  // max sequence length (kv_cache capacity)
        float*       output,     // [N, nq * head_dim]
        int N, int num_q_heads, int num_kv_heads, int head_dim,
        int          step_pos,   // first token's absolute position (for RoPE)
        float        rope_theta,
        bool         is_decode,  // false=prefill, true=decode
        float*       scratch,    // [N * (nq+nkv) * head_dim + (nq) * (cache_pos+N)] temp
        StreamHandle stream
    ) = nullptr;
};

// ─────────────────────────────────────────────────────────────────
// Registry API
// ─────────────────────────────────────────────────────────────────
void        register_backend(BackendOps* ops);
BackendOps* active_backend();
BackendOps* backend_by_name(const char* name);
bool        set_active_backend(const char* name);
void        require_active_backend(const char* name); // exits on failure
