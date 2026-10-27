// cuda_ops_common.cuh — shared kernels, helpers, and macros for all CUDA ops files
#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_map>

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(x) do { \
    cublasStatus_t _s = (x); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// cuBLAS handle — defined once in cuda_backend.cu
cublasHandle_t get_cublas();

#ifdef HAVE_CUDNN
#  include <cudnn.h>
#  define CUDNN_CHECK(x) do { \
    cudnnStatus_t _s = (x); \
    if (_s != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)
// cuDNN handle — defined once in cuda_backend.cu
cudnnHandle_t get_cudnn();
#endif // HAVE_CUDNN

// ─────────────────────────────────────────────────────────────────
// GPU scratch pool — eliminates cudaMalloc in hot path
// All temporary allocations within a forward pass use this pool.
// Ops call save()/restore() to manage their temporary allocations.
// Safe because all CUDA ops execute on the default stream (serialized).
// ─────────────────────────────────────────────────────────────────
struct GpuScratch {
    uint8_t* base = nullptr;
    size_t   cap  = 0;
    size_t   used = 0;

    void ensure_init() {
        if (base) return;
        // Try 512 MB first, fall back to 128 MB
        const size_t LARGE = 512ULL * 1024 * 1024;
        const size_t SMALL = 128ULL * 1024 * 1024;
        if (cudaMalloc(&base, LARGE) == cudaSuccess) {
            cap = LARGE;
        } else {
            CUDA_CHECK(cudaMalloc(&base, SMALL));
            cap = SMALL;
        }
        used = 0;
    }

    // Allocate from the pool (256-byte aligned for Tensor Core compatibility)
    void* alloc(size_t bytes) {
        ensure_init();
        bytes = (bytes + 255) & ~255ULL;
        if (used + bytes > cap) {
            fprintf(stderr, "GpuScratch pool exhausted! used=%zu cap=%zu need=%zu\n",
                    used, cap, bytes);
            void* p;
            CUDA_CHECK(cudaMalloc(&p, bytes));
            return p; // leaks but indicates pool too small
        }
        void* p = base + used;
        used += bytes;
        return p;
    }

    size_t save()            { ensure_init(); return used; }
    void   restore(size_t m) { used = m; }
};

// Defined in cuda_backend.cu; shared across all .cu translation units
extern GpuScratch g_scratch;

// ─────────────────────────────────────────────────────────────────
// GPU constant cache — persistent GPU copies of constant CPU arrays.
// Unlike tmp_gpu (scratch → freed each call), get() uploads once via
// cudaMalloc and returns the same GPU pointer on every subsequent call.
// Use for bias, eff_bias, req_scale, w_scales (constant after model load).
// ─────────────────────────────────────────────────────────────────
// Fingerprint: read first+last 8 bytes of the buffer for change detection.
// Detects pointer reuse after model destruction (allocator may reuse addresses).
static inline void const_cache_fingerprint(const void* ptr, size_t bytes,
                                            uint64_t& fp0, uint64_t& fpN) {
    fp0 = fpN = 0;
    if (bytes == 0) return;
    size_t take = bytes < sizeof(uint64_t) ? bytes : sizeof(uint64_t);
    memcpy(&fp0, ptr, take);
    if (bytes > sizeof(uint64_t))
        memcpy(&fpN, (const uint8_t*)ptr + bytes - sizeof(uint64_t), sizeof(uint64_t));
}

struct GpuConstEntry { void* d_ptr; size_t bytes; uint64_t fp0, fpN; };

struct GpuConstCache {
    std::unordered_map<const void*, GpuConstEntry> entries;

    // Returns persistent GPU pointer for cpu_ptr[0..n-1].
    // Returns nullptr if cpu_ptr is null or n==0.
    // Re-uploads automatically if the CPU data changed (pointer reuse detection).
    template<typename T>
    T* get(const T* cpu_ptr, int n) {
        if (!cpu_ptr || n == 0) return nullptr;
        size_t bytes = (size_t)n * sizeof(T);
        uint64_t fp0, fpN;
        const_cache_fingerprint(cpu_ptr, bytes, fp0, fpN);

        auto it = entries.find((const void*)cpu_ptr);
        if (it != entries.end()) {
            auto& e = it->second;
            if (e.bytes == bytes && e.fp0 == fp0 && e.fpN == fpN)
                return (T*)e.d_ptr;                    // cache hit — same data
            // Pointer reused with different data — re-upload
            if (e.bytes == bytes) {
                CUDA_CHECK(cudaMemcpy(e.d_ptr, cpu_ptr, bytes, cudaMemcpyHostToDevice));
                e.fp0 = fp0; e.fpN = fpN;
                return (T*)e.d_ptr;
            }
            CUDA_CHECK(cudaFree(e.d_ptr));
            entries.erase(it);
        }
        void* d;
        CUDA_CHECK(cudaMalloc(&d, bytes));
        CUDA_CHECK(cudaMemcpy(d, cpu_ptr, bytes, cudaMemcpyHostToDevice));
        entries[(const void*)cpu_ptr] = GpuConstEntry{d, bytes, fp0, fpN};
        return (T*)d;
    }
};

extern GpuConstCache g_const_cache;

// Build req_scale_f from req_mult/req_exp, cache persistently keyed by req_mult pointer.
// Handles cuda_conv2d_int8 where req_scale_f is a local vector (new address each call).
// When req_mult is null, uses &g_const_cache as sentinel key for an all-ones array.
// Fingerprint on req_mult content detects pointer reuse between models/tests.
static inline float* get_req_scale_gpu(const int32_t* req_mult, const int32_t* req_exp, int n) {
    if (n == 0) return nullptr;
    const void* key = req_mult ? (const void*)req_mult : (const void*)&g_const_cache;

    // Fingerprint based on req_mult content (nullptr → zero fingerprint → all-ones fallback)
    uint64_t fp0 = 0, fpN = 0;
    if (req_mult)
        const_cache_fingerprint(req_mult, (size_t)n * sizeof(int32_t), fp0, fpN);

    auto it = g_const_cache.entries.find(key);
    if (it != g_const_cache.entries.end()) {
        auto& e = it->second;
        if (e.bytes >= (size_t)n * sizeof(float) && e.fp0 == fp0 && e.fpN == fpN)
            return (float*)e.d_ptr;                    // cache hit
        CUDA_CHECK(cudaFree(e.d_ptr));
        g_const_cache.entries.erase(it);
    }

    std::vector<float> s(n);
    if (req_mult && req_exp)
        for (int i = 0; i < n; ++i)
            s[i] = (float)req_mult[i] * std::exp2f((float)(req_exp[i] - 31));
    else
        std::fill(s.begin(), s.end(), 1.f);

    float* d;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, s.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
    g_const_cache.entries[key] = GpuConstEntry{d, (size_t)n * sizeof(float), fp0, fpN};
    return d;
}

// ─────────────────────────────────────────────────────────────────
// Pool-backed helpers (replace cudaMalloc/cudaFree versions)
// ─────────────────────────────────────────────────────────────────

// Upload a CPU array to GPU scratch pool (use cudaMemcpy, not alloc+free)
template<typename T>
static inline T* tmp_gpu(const T* cpu_ptr, int n) {
    if (!cpu_ptr || n == 0) return nullptr;
    T* d = (T*)g_scratch.alloc((size_t)n * sizeof(T));
    CUDA_CHECK(cudaMemcpy(d, cpu_ptr, (size_t)n * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// Convert INT8 GPU buffer to FP32 GPU buffer using scratch pool
static __global__ void k_int8_to_fp32(const int8_t* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)in[i];
}

static inline float* int8_to_fp32_gpu(const int8_t* d_in, size_t n) {
    float* d_out = (float*)g_scratch.alloc(n * sizeof(float));
    k_int8_to_fp32<<<((int)n + 255) / 256, 256>>>(d_in, d_out, (int)n);
    CUDA_CHECK(cudaGetLastError());
    return d_out;
}

static inline bool cublas_int8_tc_supported(int m, int n, int k) {
    // cuBLAS int8 Tensor Core GEMM rejects shapes whose GEMM dimensions are
    // not 4-aligned. Fall back to FP32 accumulation for those dynamic shapes.
    return (m % 4 == 0) && (n % 4 == 0) && (k % 4 == 0);
}

// ─────────────────────────────────────────────────────────────────
// CUDA kernels — static so each .cu TU gets its own private copy
// ─────────────────────────────────────────────────────────────────

static __global__ void k_requant_fp32_to_int8(
    const float* __restrict__ C,
    const int64_t* __restrict__ eff_bias,
    const float*   __restrict__ req_scale,
    int8_t out_zp,
    int8_t* __restrict__ out,
    int M, int N, int N_pad)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;
    float v = C[(size_t)m * N_pad + n] + (float)eff_bias[n];
    int q = __float2int_rn(v * req_scale[n]) + (int)out_zp;
    out[(size_t)m * N + n] = (int8_t)max(-128, min(127, q));
}

static __global__ void k_requant_fp32_to_int8_nchw(
    const float* __restrict__ C,
    const int64_t* __restrict__ eff_bias,
    const float*   __restrict__ req_scale,
    int8_t out_zp,
    int8_t* __restrict__ out,
    int M, int N, int N_pad)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;
    float v = C[(size_t)m * N_pad + n] + (float)eff_bias[n];
    int q = __float2int_rn(v * req_scale[n]) + (int)out_zp;
    out[(size_t)n * M + m] = (int8_t)max(-128, min(127, q));
}

static __global__ void k_requant_fp32_to_fp32(
    const float* __restrict__ C,
    const int64_t* __restrict__ eff_bias,
    const float*   __restrict__ req_scale,
    float* __restrict__ out,
    int M, int N, int N_pad)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;
    out[(size_t)m * N + n] = (C[(size_t)m * N_pad + n] + (float)eff_bias[n]) * req_scale[n];
}

// Requantize INT32 accumulator → INT8 output (for native INT8 GEMM path)
static __global__ void k_requant_int32_to_int8(
    const int32_t* __restrict__ C,
    const int64_t* __restrict__ eff_bias,
    const float*   __restrict__ req_scale,
    int8_t out_zp,
    int8_t* __restrict__ out,
    int M, int N, int N_pad)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;
    float v = (float)C[(size_t)m * N_pad + n] + (float)eff_bias[n];
    int q = __float2int_rn(v * req_scale[n]) + (int)out_zp;
    out[(size_t)m * N + n] = (int8_t)max(-128, min(127, q));
}

static __global__ void k_requant_int32_to_int8_nchw(
    const int32_t* __restrict__ C,
    const int64_t* __restrict__ eff_bias,
    const float*   __restrict__ req_scale,
    int8_t out_zp,
    int8_t* __restrict__ out,
    int M, int N, int N_pad)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;
    float v = (float)C[(size_t)m * N_pad + n] + (float)eff_bias[n];
    int q = __float2int_rn(v * req_scale[n]) + (int)out_zp;
    out[(size_t)n * M + m] = (int8_t)max(-128, min(127, q));
}

static __global__ void k_requant_int32_to_fp32(
    const int32_t* __restrict__ C,
    const int64_t* __restrict__ eff_bias,
    const float*   __restrict__ req_scale,
    float* __restrict__ out,
    int M, int N, int N_pad)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;
    out[(size_t)m * N + n] = ((float)C[(size_t)m * N_pad + n] + (float)eff_bias[n]) * req_scale[n];
}

// INT8 im2col — NHWC input, fills [oH*oW, K_pad] output
static __global__ void k_im2col_nhwc(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ col,
    int H, int W, int C_in,
    int kH, int kW, int oH, int oW,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int8_t zp, int K, int K_pad)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= oH * oW || k >= K_pad) {
        if (m < oH * oW && k < K_pad) col[(size_t)m * K_pad + k] = 0;
        return;
    }
    int oh = m / oW, ow = m % oW;
    if (k >= K) { col[(size_t)m * K_pad + k] = 0; return; }
    int kh = k / (kW * C_in);
    int kw = (k / C_in) % kW;
    int c  = k % C_in;
    int ih = oh * stride_h - pad_h + kh;
    int iw = ow * stride_w - pad_w + kw;
    int8_t val = (ih >= 0 && ih < H && iw >= 0 && iw < W)
        ? input[((size_t)ih * W + iw) * C_in + c] : zp;
    col[(size_t)m * K_pad + k] = val;
}

// FP32 im2col — NHWC input, fills [oH*oW, K_pad] output (supports dilation)
static __global__ void k_im2col_nhwc_fp32(
    const float* __restrict__ input,
    float* __restrict__ col,
    int H, int W, int C_in,
    int kH, int kW, int oH, int oW,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int K, int K_pad)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= oH * oW || k >= K_pad) {
        if (m < oH * oW && k < K_pad) col[(size_t)m * K_pad + k] = 0.f;
        return;
    }
    int oh = m / oW, ow = m % oW;
    if (k >= K) { col[(size_t)m * K_pad + k] = 0.f; return; }
    int kh = k / (kW * C_in);
    int kw = (k / C_in) % kW;
    int c  = k % C_in;
    int ih = oh * stride_h - pad_h + kh * dilation_h;
    int iw = ow * stride_w - pad_w + kw * dilation_w;
    float val = (ih >= 0 && ih < H && iw >= 0 && iw < W)
        ? input[((size_t)ih * W + iw) * C_in + c] : 0.f;
    col[(size_t)m * K_pad + k] = val;
}

static __global__ void k_maxpool_nhwc_int8(
    const int8_t* __restrict__ in, int8_t* __restrict__ out,
    int C, int H, int W, int kH, int kW,
    int sh, int sw, int ph, int pw, int oH, int oW)
{
    int c  = blockIdx.z;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C || oh >= oH || ow >= oW) return;
    int8_t best = -128;
    for (int kh = 0; kh < kH; ++kh) for (int kw = 0; kw < kW; ++kw) {
        int ih = oh * sh - ph + kh, iw = ow * sw - pw + kw;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            int8_t v = in[((size_t)ih * W + iw) * C + c];
            if (v > best) best = v;
        }
    }
    out[((size_t)oh * oW + ow) * C + c] = best;
}

static __global__ void k_maxpool_nchw_int8(
    const int8_t* __restrict__ in, int8_t* __restrict__ out,
    int C, int H, int W, int kH, int kW,
    int sh, int sw, int ph, int pw, int oH, int oW)
{
    int c  = blockIdx.z;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C || oh >= oH || ow >= oW) return;
    int8_t best = -128;
    for (int kh = 0; kh < kH; ++kh) for (int kw = 0; kw < kW; ++kw) {
        int ih = oh * sh - ph + kh, iw = ow * sw - pw + kw;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            int8_t v = in[(size_t)c * H * W + (size_t)ih * W + iw];
            if (v > best) best = v;
        }
    }
    out[(size_t)c * oH * oW + (size_t)oh * oW + ow] = best;
}

static __global__ void k_maxpool_nhwc_fp32(
    const float* __restrict__ in, float* __restrict__ out,
    int C, int H, int W, int kH, int kW,
    int sh, int sw, int ph, int pw, int oH, int oW)
{
    int c  = blockIdx.z;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C || oh >= oH || ow >= oW) return;
    float best = -1e38f;
    for (int kh = 0; kh < kH; ++kh) for (int kw = 0; kw < kW; ++kw) {
        int ih = oh * sh - ph + kh, iw = ow * sw - pw + kw;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            float v = in[((size_t)ih * W + iw) * C + c];
            if (v > best) best = v;
        }
    }
    out[((size_t)oh * oW + ow) * C + c] = best;
}

static __global__ void k_avgpool_global_nhwc_int8(
    const int8_t* __restrict__ in,
    float in_sc, int in_zp, float out_sc, int out_zp,
    int8_t* __restrict__ out, int C, int HW)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float sum = 0.f;
    for (int i = 0; i < HW; ++i) sum += (float)(in[(size_t)i * C + c] - in_zp);
    float v = sum * in_sc / (float)HW;
    int q = __float2int_rn(v / out_sc) + out_zp;
    out[c] = (int8_t)max(-128, min(127, q));
}

static __global__ void k_avgpool_global_nchw_int8(
    const int8_t* __restrict__ in,
    float in_sc, int in_zp, float out_sc, int out_zp,
    int8_t* __restrict__ out, int C, int HW)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float sum = 0.f;
    const int8_t* ch = in + (size_t)c * HW;
    for (int i = 0; i < HW; ++i) sum += (float)(ch[i] - in_zp);
    float v = sum * in_sc / (float)HW;
    int q = __float2int_rn(v / out_sc) + out_zp;
    out[c] = (int8_t)max(-128, min(127, q));
}

static __global__ void k_avgpool_global_nhwc_fp32(
    const float* __restrict__ in, float* __restrict__ out, int C, int HW)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float sum = 0.f;
    for (int i = 0; i < HW; ++i) sum += in[(size_t)i * C + c];
    out[c] = sum / (float)HW;
}

static __global__ void k_add_requant(
    const int8_t* __restrict__ a, const int8_t* __restrict__ b,
    float in1_sc, int in1_zp, float in2_sc, int in2_zp,
    float out_sc, int out_zp,
    int8_t* __restrict__ out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = (float)(a[i] - in1_zp) * in1_sc + (float)(b[i] - in2_zp) * in2_sc;
    int q = __float2int_rn(v / out_sc) + out_zp;
    out[i] = (int8_t)max(-128, min(127, q));
}

static __global__ void k_add_fp32(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out, int N, bool relu)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = a[i] + b[i];
    out[i] = relu ? fmaxf(v, 0.f) : v;
}

static __global__ void k_relu_fp32(float* x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = fmaxf(x[i], 0.f);
}

static __global__ void k_gelu_fp32(float* x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float v = x[i];
        x[i] = 0.5f * v * (1.f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

static __global__ void k_sigmoid_fp32(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = 1.f / (1.f + expf(-in[i]));
}

static __global__ void k_tanh_fp32(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = tanhf(in[i]);
}

static __global__ void k_leaky_relu_fp32(const float* in, float* out, int N, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] >= 0.f ? in[i] : alpha * in[i];
}

static __global__ void k_exp_fp32(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = expf(in[i]);
}

static __global__ void k_sin_fp32(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = sinf(in[i]);
}

static __global__ void k_scale_fp32(const float* in, float* out, int N, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] * s;
}

static __global__ void k_add_vecs_fp32(float* dst, const float* src, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] += src[i];
}

static __global__ void k_bias_add_rows(float* out, const float* bias, int T, int C) {
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T && c < C) out[(size_t)t * C + c] += bias[c];
}

// Layernorm — parallel reduction (one block per row, 256 threads)
static __global__ void k_layernorm(
    const float* __restrict__ in,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    float* __restrict__ out, int N, int C, float eps)
{
    // One block per row
    int n = blockIdx.x;
    if (n >= N) return;
    const float* row = in + (size_t)n * C;
    float* orow = out + (size_t)n * C;

    extern __shared__ float smem[];
    int t = threadIdx.x;

    // Mean
    float thread_sum = 0.f;
    for (int c = t; c < C; c += blockDim.x) thread_sum += row[c];
    smem[t] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) smem[t] += smem[t + s];
        __syncthreads();
    }
    float mean = smem[0] / (float)C;
    __syncthreads();

    // Variance
    float thread_var = 0.f;
    for (int c = t; c < C; c += blockDim.x) {
        float d = row[c] - mean;
        thread_var += d * d;
    }
    smem[t] = thread_var;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) smem[t] += smem[t + s];
        __syncthreads();
    }
    float inv = rsqrtf(smem[0] / (float)C + eps);
    __syncthreads();

    for (int c = t; c < C; c += blockDim.x)
        orow[c] = (row[c] - mean) * inv * gamma[c] + beta[c];
}

static __global__ void k_upsample_nearest1d(
    const float* in, float* out, int T, int C, int sf)
{
    int c  = blockIdx.z;
    int ot = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C || ot >= T * sf) return;
    out[(size_t)(ot) * C + c] = in[(size_t)(ot / sf) * C + c];
}

static __global__ void k_upsample_bilinear2d(
    const float* in, float* out,
    int C, int iH, int iW, int oH, int oW)
{
    int c  = blockIdx.z;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C || oh >= oH || ow >= oW) return;
    float y = (oh + 0.5f) * (float)iH / (float)oH - 0.5f;
    float x = (ow + 0.5f) * (float)iW / (float)oW - 0.5f;
    int yf = (int)floorf(y), xf = (int)floorf(x);
    float dy = y - (float)yf, dx = x - (float)xf;
    int y0 = max(0, min(yf,     iH - 1)), y1 = max(0, min(yf + 1, iH - 1));
    int x0 = max(0, min(xf,     iW - 1)), x1 = max(0, min(xf + 1, iW - 1));
    float v = in[((size_t)y0 * iW + x0) * C + c] * (1.f - dy) * (1.f - dx)
            + in[((size_t)y0 * iW + x1) * C + c] * (1.f - dy) * dx
            + in[((size_t)y1 * iW + x0) * C + c] * dy * (1.f - dx)
            + in[((size_t)y1 * iW + x1) * C + c] * dy * dx;
    out[((size_t)oh * oW + ow) * C + c] = v;
}

static __global__ void k_sum_channels(const float* in, float* out, int T, int C) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float s = 0.f;
    for (int c = 0; c < C; ++c) s += in[(size_t)t * C + c];
    out[t] = s;
}

static __global__ void k_embedding_lookup(
    const float* ids, const float* weight, float* out,
    int T, int vocab, int D)
{
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T || d >= D) return;
    int id = (int)ids[t];
    if (id < 0 || id >= vocab) { out[(size_t)t * D + d] = 0.f; return; }
    out[(size_t)t * D + d] = weight[(size_t)id * D + d];
}

static __global__ void k_concat1d(
    const float* a, const float* b, float* out,
    int T, int C1, int C2, bool broadcast)
{
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T || c >= C1 + C2) return;
    if (c < C1) out[(size_t)t * (C1 + C2) + c] = a[(size_t)t * C1 + c];
    else        out[(size_t)t * (C1 + C2) + c] = b[(size_t)(broadcast ? 0 : t) * C2 + (c - C1)];
}

static __global__ void k_slice_channels(
    const float* in, float* out, int T, int C_in, int ch_start, int C_out)
{
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T || c >= C_out) return;
    out[(size_t)t * C_out + c] = in[(size_t)t * C_in + ch_start + c];
}

// AdaIN — parallel reduction per channel (one block per channel)
static __global__ void k_ada_in1d(
    const float* feat, float* out,
    const float* gamma, const float* beta,
    int T, int C, float eps,
    const float* norm_weight,   // may be nullptr -> scale=1
    const float* norm_bias)     // may be nullptr -> shift=0
{
    int c = blockIdx.x;
    if (c >= C) return;
    float mean = 0.f;
    for (int t = 0; t < T; ++t) mean += feat[(size_t)t * C + c];
    mean /= (float)T;
    float var = 0.f;
    for (int t = 0; t < T; ++t) { float d = feat[(size_t)t * C + c] - mean; var += d * d; }
    var /= (float)T;
    float inv = rsqrtf(var + eps);
    float nw = norm_weight ? norm_weight[c] : 1.f;
    float nb = norm_bias   ? norm_bias[c]   : 0.f;
    for (int t = 0; t < T; ++t) {
        float xraw = (feat[(size_t)t * C + c] - mean) * inv;
        float xn = xraw * nw + nb;  // apply InstanceNorm learnable scale/shift
        out[(size_t)t * C + c] = (1.f + gamma[c]) * xn + beta[c];
    }
}

// AdaLayerNorm — parallel reduction (one block per row)
static __global__ void k_ada_layernorm1d(
    const float* feat, float* out,
    const float* gamma, const float* beta,
    int T, int C, float eps)
{
    int t = blockIdx.x;
    if (t >= T) return;
    const float* row = feat + (size_t)t * C;
    float* orow = out + (size_t)t * C;

    extern __shared__ float smem[];
    int tid = threadIdx.x;

    float thread_sum = 0.f;
    for (int c = tid; c < C; c += blockDim.x) thread_sum += row[c];
    smem[tid] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float mean = smem[0] / (float)C;
    __syncthreads();

    float thread_var = 0.f;
    for (int c = tid; c < C; c += blockDim.x) {
        float d = row[c] - mean;
        thread_var += d * d;
    }
    smem[tid] = thread_var;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(smem[0] / (float)C + eps);
    __syncthreads();

    for (int c = tid; c < C; c += blockDim.x) {
        float xn = (row[c] - mean) * inv;
        orow[c] = (1.f + gamma[c]) * xn + beta[c];
    }
}

// Snake1D: out[t*C+c] = x + sin²(alpha[c]*x) * inv_alpha[c]
// Grid: (T*C + 255) / 256; input layout: [T, C] (NHWC with W=1)
static __global__ void k_snake1d(
    const float* in, float* out,
    const float* alpha, const float* inv_alpha,
    int T, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * C) return;
    int c = idx % C;
    float s = sinf(alpha[c] * in[idx]);
    out[idx] = in[idx] + s * s * inv_alpha[c];
}

static __global__ void k_seqgemm_bias_act(
    float* out, const float* bias, int N, int C_out, bool gelu)
{
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || c >= C_out) return;
    float v = out[(size_t)n * C_out + c] + bias[c];
    if (gelu) {
        v = 0.5f * v * (1.f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
    out[(size_t)n * C_out + c] = v;
}

static __global__ void k_wino_input_transform(
    const int8_t* __restrict__ in,
    int8_t* __restrict__ out_hat,
    int H, int W, int C_in,
    int C_in_pad, float fwd_scale, int8_t in_zp,
    int tH, int tW)
{
    int tile = blockIdx.x * blockDim.x + threadIdx.x;
    int c    = blockIdx.y * blockDim.y + threadIdx.y;
    if (tile >= tH * tW || c >= C_in) return;
    int th = tile / tW, tw = tile % tW;
    int h0 = th * 2 - 1, w0 = tw * 2 - 1;
    float d[4][4];
    // Use raw int8 values (same as CPU path); out-of-bounds filled with in_zp.
    // This matches the CPU Winograd convention so wino_eff_bias is correct.
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
        int ih = h0 + i, iw = w0 + j;
        d[i][j] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
            ? (float)in[((size_t)ih * W + iw) * C_in + c]
            : (float)(int)in_zp;
    }
    float Bd[4][4];
    for (int i = 0; i < 4; ++i) {
        Bd[0][i] = d[0][i] - d[2][i];
        Bd[1][i] = d[1][i] + d[2][i];
        Bd[2][i] = d[2][i] - d[1][i];
        Bd[3][i] = d[1][i] - d[3][i];
    }
    float BdB[4][4];
    for (int i = 0; i < 4; ++i) {
        BdB[i][0] = Bd[i][0] - Bd[i][2];
        BdB[i][1] = Bd[i][1] + Bd[i][2];
        BdB[i][2] = Bd[i][2] - Bd[i][1];
        BdB[i][3] = Bd[i][1] - Bd[i][3];
    }
    for (int pos = 0; pos < 16; ++pos) {
        int pi = pos / 4, pj = pos % 4;
        float q = BdB[pi][pj] * fwd_scale;
        int qi = __float2int_rn(q);
        qi = max(-127, min(127, qi));
        out_hat[(size_t)pos * tH * tW * C_in_pad + (size_t)tile * C_in_pad + c] = (int8_t)qi;
    }
}

static __global__ void k_wino_output_transform(
    const float* __restrict__ acc,
    const int64_t* __restrict__ eff_bias,
    const float*   __restrict__ req_scale,
    int8_t out_zp,
    int8_t* __restrict__ out,
    int oH, int oW, int C_out, int num_tiles, int tW)
{
    int tile = blockIdx.x * blockDim.x + threadIdx.x;
    int c    = blockIdx.y * blockDim.y + threadIdx.y;
    if (tile >= num_tiles || c >= C_out) return;
    float m[4][4];
    for (int pos = 0; pos < 16; ++pos)
        m[pos / 4][pos % 4] = acc[(size_t)pos * num_tiles * C_out + (size_t)tile * C_out + c];
    float Am[2][4];
    for (int j = 0; j < 4; ++j) {
        Am[0][j] = m[0][j] + m[1][j] + m[2][j];
        Am[1][j] = m[1][j] - m[2][j] - m[3][j];
    }
    float AmA[2][2];
    AmA[0][0] = Am[0][0] + Am[0][1] + Am[0][2];
    AmA[0][1] = Am[0][1] - Am[0][2] - Am[0][3];
    AmA[1][0] = Am[1][0] + Am[1][1] + Am[1][2];
    AmA[1][1] = Am[1][1] - Am[1][2] - Am[1][3];
    int th = tile / tW, tw = tile % tW;
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
        int oh = th * 2 + i, ow = tw * 2 + j;
        if (oh >= oH || ow >= oW) continue;
        float v = AmA[i][j] + (float)eff_bias[c];
        int q = __float2int_rn(v * req_scale[c]) + (int)out_zp;
        out[((size_t)oh * oW + ow) * C_out + c] = (int8_t)max(-128, min(127, q));
    }
}

static __global__ void k_wino_input_transform_fp32(
    const float* __restrict__ in,
    float* __restrict__ out_hat,
    int H, int W, int C_in,
    int tH, int tW)
{
    int tile = blockIdx.x * blockDim.x + threadIdx.x;
    int c    = blockIdx.y * blockDim.y + threadIdx.y;
    if (tile >= tH * tW || c >= C_in) return;
    int th = tile / tW, tw = tile % tW;
    int h0 = th * 2 - 1, w0 = tw * 2 - 1;
    float d[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
        int ih = h0 + i, iw = w0 + j;
        d[i][j] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
            ? in[((size_t)ih * W + iw) * C_in + c]
            : 0.f;
    }
    float Bd[4][4];
    for (int i = 0; i < 4; ++i) {
        Bd[0][i] = d[0][i] - d[2][i];
        Bd[1][i] = d[1][i] + d[2][i];
        Bd[2][i] = d[2][i] - d[1][i];
        Bd[3][i] = d[1][i] - d[3][i];
    }
    float BdB[4][4];
    for (int i = 0; i < 4; ++i) {
        BdB[i][0] = Bd[i][0] - Bd[i][2];
        BdB[i][1] = Bd[i][1] + Bd[i][2];
        BdB[i][2] = Bd[i][2] - Bd[i][1];
        BdB[i][3] = Bd[i][1] - Bd[i][3];
    }
    int num_tiles = tH * tW;
    for (int pos = 0; pos < 16; ++pos) {
        int pi = pos / 4, pj = pos % 4;
        out_hat[(size_t)pos * num_tiles * C_in + (size_t)tile * C_in + c] = BdB[pi][pj];
    }
}

static __global__ void k_wino_output_transform_fp32(
    const float* __restrict__ acc,
    const float* __restrict__ bias,
    bool relu,
    float* __restrict__ out,
    int oH, int oW, int C_out, int num_tiles, int tW)
{
    int tile = blockIdx.x * blockDim.x + threadIdx.x;
    int c    = blockIdx.y * blockDim.y + threadIdx.y;
    if (tile >= num_tiles || c >= C_out) return;
    float m[4][4];
    for (int pos = 0; pos < 16; ++pos)
        m[pos / 4][pos % 4] = acc[(size_t)pos * num_tiles * C_out + (size_t)tile * C_out + c];
    float Am[2][4];
    for (int j = 0; j < 4; ++j) {
        Am[0][j] = m[0][j] + m[1][j] + m[2][j];
        Am[1][j] = m[1][j] - m[2][j] - m[3][j];
    }
    float AmA[2][2];
    AmA[0][0] = Am[0][0] + Am[0][1] + Am[0][2];
    AmA[0][1] = Am[0][1] - Am[0][2] - Am[0][3];
    AmA[1][0] = Am[1][0] + Am[1][1] + Am[1][2];
    AmA[1][1] = Am[1][1] - Am[1][2] - Am[1][3];
    float bias_val = bias ? bias[c] : 0.f;
    int th = tile / tW, tw = tile % tW;
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
        int oh = th * 2 + i, ow = tw * 2 + j;
        if (oh >= oH || ow >= oW) continue;
        float v = AmA[i][j] + bias_val;
        if (relu) v = fmaxf(v, 0.f);
        out[((size_t)oh * oW + ow) * C_out + c] = v;
    }
}

static __global__ void k_int32_to_fp32(const int32_t* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)in[i];
}

static __global__ void k_int32_scale_fp32(const int32_t* in, float* out, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)in[i] * scale;
}

static __global__ void k_max_abs_1block(const float* in, float* out, int N) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    float val = 1e-10f;
    for (int i = tid; i < N; i += blockDim.x)
        val = fmaxf(val, fabsf(in[i]));
    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) out[0] = sdata[0];
}

static __global__ void k_quantize_with_dscale(const float* in, int8_t* out, const float* d_max_abs, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float inv = 127.f / fmaxf(*d_max_abs, 1e-10f);
    int q = __float2int_rn(in[i] * inv);
    out[i] = (int8_t)max(-127, min(127, q));
}

// Per-row max-abs reduction: [N_rows, K] → [N_rows] floats.
// One block per row; blockDim.x threads reduce K elements via shared memory.
static __global__ void k_row_max_abs(const float* __restrict__ inp,
                                      float* __restrict__ row_max,
                                      int N_rows, int K)
{
    extern __shared__ float srow[];
    int n   = blockIdx.x;
    int tid = threadIdx.x;
    if (n >= N_rows) return;
    float val = 1e-10f;
    for (int k = tid; k < K; k += blockDim.x)
        val = fmaxf(val, fabsf(inp[(size_t)n * K + k]));
    srow[tid] = val;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) srow[tid] = fmaxf(srow[tid], srow[tid + s]);
        __syncthreads();
    }
    if (tid == 0) row_max[n] = srow[0];
}

// Per-row INT8 quantization: uses row_max[n] for row n of [N_rows, K] matrix.
static __global__ void k_quantize_per_row(const float* __restrict__ inp,
                                           int8_t* __restrict__ out,
                                           const float* __restrict__ row_max,
                                           int N_rows, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_rows * K) return;
    int n = idx / K;
    float inv = 127.f / fmaxf(row_max[n], 1e-10f);
    int q = __float2int_rn(inp[idx] * inv);
    out[idx] = (int8_t)max(-127, min(127, q));
}

// Dequantize INT32 GEMM output with per-row input scale + per-column weight scale.
// c_i32: [N, N_pad], row_max: [N], w_scales: [C_out], out: [N, C_out]
static __global__ void k_dequant_int32_perrow(
    const int32_t* __restrict__ c_i32,
    const float*   __restrict__ row_max,
    const float*   __restrict__ w_scales,
    float*         __restrict__ out,
    int N, int C_out, int N_pad)
{
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || c >= C_out) return;
    float scale = (row_max[n] / 127.f) * w_scales[c];
    out[(size_t)n * C_out + c] = (float)c_i32[(size_t)n * N_pad + c] * scale;
}

// In-place scale FP32 GEMM output with per-row input scale + per-column weight scale.
// out: [N, C_out] in-place, row_max: [N], w_scales: [C_out]
static __global__ void k_dequant_fp32_perrow(
    float*       __restrict__ out,
    const float* __restrict__ row_max,
    const float* __restrict__ w_scales,
    int N, int C_out)
{
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || c >= C_out) return;
    float scale = (row_max[n] / 127.f) * w_scales[c];
    out[(size_t)n * C_out + c] *= scale;
}

static __global__ void k_dequant_seqgemm(float* out, const float* d_max_abs,
                                          const float* w_scales, int N, int C_out) {
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || c >= C_out) return;
    float scale = (*d_max_abs / 127.f) * w_scales[c];
    out[(size_t)n * C_out + c] *= scale;
}

// Dequantize INT32 GEMM output (stride N_pad) → FP32 output (stride C_out)
// Combines INT32→FP32 cast and per-column scale in one kernel
static __global__ void k_dequant_seqgemm_int32(
    const int32_t* __restrict__ c_i32,  // [N, N_pad] row-major
    const float* __restrict__ d_max_abs,
    const float* __restrict__ w_scales,
    float* __restrict__ out,             // [N, C_out] compact
    int N, int C_out, int N_pad)
{
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || c >= C_out) return;
    float scale = (*d_max_abs / 127.f) * w_scales[c];
    out[(size_t)n * C_out + c] = (float)c_i32[(size_t)n * N_pad + c] * scale;
}

// ─────────────────────────────────────────────────────────────────
// Attention GPU kernels
// ─────────────────────────────────────────────────────────────────

// Gather Q, K, V from interleaved [N, 3*D] → separate [num_heads, N, head_dim] buffers
static __global__ void k_gather_qkv(
    const float* __restrict__ qkv,   // [N, 3*D], D = num_heads * head_dim
    float* __restrict__ Q,            // [num_heads, N, head_dim]
    float* __restrict__ K,
    float* __restrict__ V,
    int N, int num_heads, int head_dim)
{
    int h = blockIdx.z;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_heads || n >= N || d >= head_dim) return;
    int D = num_heads * head_dim;
    int src = n * 3 * D + h * head_dim + d;
    int dst = h * N * head_dim + n * head_dim + d;
    Q[dst] = qkv[src];
    K[dst] = qkv[src + D];
    V[dst] = qkv[src + 2 * D];
}

// Scatter [num_heads, N, head_dim] → [N, D] (interleave heads)
static __global__ void k_scatter_heads(
    const float* __restrict__ heads,  // [num_heads, N, head_dim]
    float* __restrict__ out,           // [N, D], D = num_heads * head_dim
    int N, int num_heads, int head_dim)
{
    int h = blockIdx.z;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_heads || n >= N || d >= head_dim) return;
    int D = num_heads * head_dim;
    out[n * D + h * head_dim + d] = heads[h * N * head_dim + n * head_dim + d];
}

// Per-row softmax — one block per row, shared memory reduction
// Handles up to 1024 columns with blockDim.x threads (use 256)
static __global__ void k_softmax_2d(float* mat, int total_rows, int cols)
{
    int row = blockIdx.x;
    if (row >= total_rows) return;
    float* row_data = mat + (long long)row * cols;

    extern __shared__ float smem[];
    int t = threadIdx.x;

    // Find row max
    float mx = -1e38f;
    for (int j = t; j < cols; j += blockDim.x)
        mx = fmaxf(mx, row_data[j]);
    smem[t] = mx;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) smem[t] = fmaxf(smem[t], smem[t + s]);
        __syncthreads();
    }
    mx = smem[0];
    __syncthreads();

    // exp(x - max) and sum
    float sum = 0.f;
    for (int j = t; j < cols; j += blockDim.x) {
        float v = expf(row_data[j] - mx);
        row_data[j] = v;
        sum += v;
    }
    smem[t] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) smem[t] += smem[t + s];
        __syncthreads();
    }
    float inv = 1.f / smem[0];
    __syncthreads();

    for (int j = t; j < cols; j += blockDim.x)
        row_data[j] *= inv;
}

static __global__ void k_ada_in1d_v2(
    const float* feat, float* out,
    const float* gamma, const float* beta,
    int T, int C, float eps)
{
    // Re-use k_ada_in1d logic (single channel per block, single thread serial)
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float mean = 0.f;
    for (int t = 0; t < T; ++t) mean += feat[(size_t)t * C + c];
    mean /= (float)T;
    float var = 0.f;
    for (int t = 0; t < T; ++t) { float d = feat[(size_t)t * C + c] - mean; var += d * d; }
    var /= (float)T;
    float inv = rsqrtf(var + eps);
    for (int t = 0; t < T; ++t) {
        float xn = (feat[(size_t)t * C + c] - mean) * inv;
        out[(size_t)t * C + c] = (1.f + gamma[c]) * xn + beta[c];
    }
}
