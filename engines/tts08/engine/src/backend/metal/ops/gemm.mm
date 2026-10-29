// src/backend/metal/ops/gemm.mm
// GEMM ops for the Metal backend.
//
// Weight packing strategy:
//   metal_pack_weights_fp32/int8 perform the standard CPU packing (NEON tile
//   format) AND register a Metal buffer containing the ORIGINAL row-major
//   weights in the weight buffer registry (g_wt_map).  upload_weights() then
//   copies the packed data to a Metal buffer via copy_h2d, which migrates the
//   registry key from the CPU packed pointer to the new Metal contents pointer.
//   After that, metal_get_weight_buf(B_packed) returns the original-format
//   Metal buffer, which is suitable for MPS MatMul and custom Metal kernels.
//
// GPU paths:
//   sgemm_f32 / gemm_fp32_vec : MPS MatrixMultiplication
//   gemm_int8 (float output)  : gemm_int8_float_k custom kernel
//   gemm_int8 (int8 output)   : gemm_int8_requant_k custom kernel
//   seqgemm_fp32              : MPS MatrixMultiplication (unchanged)
//   seqgemm_int8              : CPU fallback (dynamic quantisation)
//   gemm_int8_int32           : CPU fallback (used only by Winograd INT8)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "metal_ops.hpp"
#include "backend/metal/metal_helpers.h"
#include "backend/cpu/ops_neon.hpp"

#include <cstdio>
#include <cstring>

// ── Weight packing ────────────────────────────────────────────────
// Call CPU pack AND register a Metal buffer with the original weights.

int8_t* metal_pack_weights_int8(const int8_t* w, int C_out, int K) {
    int8_t* packed = pack_weights_sdot(w, C_out, K);
    if (packed && g_metal_device && w && C_out > 0 && K > 0) {
        size_t sz = (size_t)C_out * K;
        id<MTLBuffer> buf = [g_metal_device newBufferWithBytes:w
                                                        length:sz
                                                       options:MTLResourceStorageModeShared];
        if (buf) metal_register_weight_buf(packed, buf);
    }
    return packed;
}

float* metal_pack_weights_fp32(const float* w, int C_out, int K) {
    float* packed = pack_weights_f32(w, C_out, K);
    if (packed && g_metal_device && w && C_out > 0 && K > 0) {
        size_t sz = (size_t)C_out * K * sizeof(float);
        id<MTLBuffer> buf = [g_metal_device newBufferWithBytes:w
                                                        length:sz
                                                       options:MTLResourceStorageModeShared];
        if (buf) metal_register_weight_buf(packed, buf);
    }
    return packed;
}

void metal_free_packed_int8(int8_t* p) {
    metal_unregister_weight_buf(p);
    free_packed(p);
}

void metal_free_packed_fp32(float* p) {
    metal_unregister_weight_buf(p);
    free_packed_f32(p);
}

// ── Shared MPS / kernel dispatch helpers ─────────────────────────

// Run the tiled INT8→INT8 GEMM kernel.
// A: Metal buf [M, K], B: Metal buf [N, K] (original row-major int8)
// ebias_buf, mult_buf, exp_buf: Metal bufs for per-column arrays
// C_buf: Metal buf [M, N] int8, dims.w = out_zp
static void dispatch_gemm_int8_requant(
    id<MTLBuffer> A_buf,     NSUInteger A_off,
    id<MTLBuffer> B_buf,     // [N, K] int8 row-major
    id<MTLBuffer> ebias_buf,
    id<MTLBuffer> mult_buf,
    id<MTLBuffer> exp_buf,
    id<MTLBuffer> C_buf,     NSUInteger C_off,
    int M, int K, int N, int out_zp)
{
    const int T = 16;
    int dims[4] = { M, K, N, out_zp };
    NSUInteger tg_n = (NSUInteger)((N + T - 1) / T);
    NSUInteger tg_m = (NSUInteger)((M + T - 1) / T);

    MetalDispatch d("gemm_int8_requant_k");
    d.buf(A_buf, A_off);
    d.buf(B_buf);
    d.buf(ebias_buf);
    d.buf(mult_buf);
    d.buf(exp_buf);
    d.buf(C_buf, C_off);
    d.bytes(&dims, sizeof(dims));
    d.tg_mem(2 * T * T);
    d.run_threadgroups(MTLSizeMake(tg_n, tg_m, 1), MTLSizeMake((NSUInteger)T, (NSUInteger)T, 1));
}

// Run the tiled INT8→float GEMM kernel.
static void dispatch_gemm_int8_float(
    id<MTLBuffer> A_buf,      NSUInteger A_off,
    id<MTLBuffer> B_buf,
    id<MTLBuffer> ebias_buf,
    id<MTLBuffer> rscale_buf,
    id<MTLBuffer> C_buf,      NSUInteger C_off,
    int M, int K, int N)
{
    const int T = 16;
    int dims[4] = { M, K, N, 0 };
    NSUInteger tg_n = (NSUInteger)((N + T - 1) / T);
    NSUInteger tg_m = (NSUInteger)((M + T - 1) / T);

    MetalDispatch d("gemm_int8_float_k");
    d.buf(A_buf, A_off);
    d.buf(B_buf);
    d.buf(ebias_buf);
    d.buf(rscale_buf);
    d.buf(C_buf, C_off);
    d.bytes(&dims, sizeof(dims));
    d.tg_mem(2 * T * T);
    d.run_threadgroups(MTLSizeMake(tg_n, tg_m, 1), MTLSizeMake((NSUInteger)T, (NSUInteger)T, 1));
}

// ── INT8 GEMM ─────────────────────────────────────────────────────
void metal_gemm_int8(
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
    StreamHandle   stream)
{
    // Retrieve Metal buffers.
    id<MTLBuffer> A_buf     = metal_get_buf(A);
    id<MTLBuffer> B_orig    = metal_get_weight_buf(B_packed);
    id<MTLBuffer> C_buf     = metal_get_buf(C);

    // nchw_out layout or missing buffers → abort (Metal always uses nhwc).
    if (nchw_out || !A_buf || !B_orig || !C_buf) {
        fprintf(stderr, "[metal] gemm_int8: nchw_out=%d A=%p B=%p C=%p — aborting\n",
                (int)nchw_out, (__bridge void*)A_buf, (__bridge void*)B_orig, (__bridge void*)C_buf);
        abort();
    }

    NSUInteger A_off = (NSUInteger)((const uint8_t*)A - (const uint8_t*)[A_buf contents]);
    NSUInteger C_off = (NSUInteger)((      uint8_t*)C - (      uint8_t*)[C_buf contents]);

    // Upload fixed arrays to Metal buffers (cached by CPU pointer).
    id<MTLBuffer> ebias_buf = metal_get_or_create_cpu_buf(eff_bias, (size_t)N * sizeof(int64_t));
    if (!ebias_buf) {
        fprintf(stderr, "[metal] gemm_int8: ebias_buf alloc failed — aborting\n");
        abort();
    }

    if (is_float) {
        // float output: use req_scale_f per-channel.
        id<MTLBuffer> rscale_buf = metal_get_or_create_cpu_buf(req_scale_f, (size_t)N * sizeof(float));
        if (!rscale_buf) {
            fprintf(stderr, "[metal] gemm_int8: rscale_buf alloc failed — aborting\n");
            abort();
        }
        dispatch_gemm_int8_float(A_buf, A_off, B_orig, ebias_buf, rscale_buf, C_buf, C_off, M, K, N);
    } else {
        // INT8 output: use Q31 requantisation.
        id<MTLBuffer> mult_buf = metal_get_or_create_cpu_buf(req_mult, (size_t)N * sizeof(int32_t));
        id<MTLBuffer> exp_buf  = metal_get_or_create_cpu_buf(req_exp,  (size_t)N * sizeof(int32_t));
        if (!mult_buf || !exp_buf) {
            fprintf(stderr, "[metal] gemm_int8: mult/exp buf alloc failed — aborting\n");
            abort();
        }
        dispatch_gemm_int8_requant(A_buf, A_off, B_orig, ebias_buf, mult_buf, exp_buf,
                                   C_buf, C_off, M, K, N, (int)out_zp);
    }
}

// ── INT8 GEMM → INT32 — CPU fallback ─────────────────────────────
// Used only by Winograd INT8 (frequency-domain accumulation).
void metal_gemm_int8_int32(
    const int8_t*  A,
    const int8_t*  B_packed,
    int32_t*       C,
    int M, int K, int N,
    const int32_t* b_row_sums,
    StreamHandle   stream)
{
    gemm_int8_int32(A, B_packed, C, M, K, N, b_row_sums, stream);
}

// ── FP32 SGEMM (general M×K × K×N) ─────────────────────────────
void metal_sgemm_f32(
    const float* A,
    const float* B_packed,
    const float* bias,
    float*       C,
    bool         relu,
    int M, int K, int N,
    bool         in_parallel,
    StreamHandle stream)
{
    id<MTLBuffer> A_buf  = metal_get_buf(A);
    id<MTLBuffer> B_orig = metal_get_weight_buf(B_packed);
    id<MTLBuffer> C_buf  = metal_get_buf(C);

    if (!A_buf || !B_orig || !C_buf) {
        fprintf(stderr, "[metal] sgemm_f32: buffer lookup failed — aborting\n");
        abort();
    }

    NSUInteger A_off = (NSUInteger)((const uint8_t*)A - (const uint8_t*)[A_buf contents]);
    NSUInteger C_off = (NSUInteger)((      uint8_t*)C - (      uint8_t*)[C_buf contents]);

    metal_mps_matmul(A_buf, A_off, B_orig, 0, C_buf, C_off, M, K, N);

    // Bias add.
    if (bias) {
        id<MTLBuffer> bias_buf = metal_get_or_create_cpu_buf(bias, (size_t)N * sizeof(float));
        if (bias_buf) {
            int tc[2] = { M, N };
            MetalDispatch d("seqgemm_bias_k");
            d.buf(C_buf, C_off);
            d.buf(bias_buf);
            d.bytes(tc, sizeof(tc));
            d.run((NSUInteger)(M * N));
        } else {
            fprintf(stderr, "[metal] sgemm_f32: bias_buf alloc failed — aborting\n");
            abort();
        }
    }

    // ReLU.
    if (relu) {
        MetalDispatch d("relu_inplace_k");
        d.buf(C_buf, C_off);
        d.run((NSUInteger)(M * N));
    }
}

// ── FP32 single-vector GEMM ──────────────────────────────────────
// gemm_fp32_vec: input [1, C_in] × weight [C_out, C_in]^T → output [1, C_out]
// weight is the original row-major CPU pointer.
// w_packed is the Metal buffer (after upload_weights) containing NEON-packed data;
// we look up the registry to get the original-format buffer.
void metal_gemm_fp32_vec(
    const float* input,
    const float* weight,   // CPU pointer (original row-major [C_out, C_in])
    const float* bias,
    float*       output,
    int C_in, int C_out,
    const float* w_packed,
    StreamHandle stream)
{
    id<MTLBuffer> A_buf  = metal_get_buf(input);
    // Look up original weights: first try registry (from upload_weights path),
    // then fall back to the CPU weight pointer directly.
    id<MTLBuffer> B_orig = metal_get_weight_buf(w_packed);
    if (!B_orig)
        B_orig = metal_get_or_create_cpu_buf(weight, (size_t)C_out * C_in * sizeof(float));
    id<MTLBuffer> C_buf  = metal_get_buf(output);

    if (!A_buf || !B_orig || !C_buf) {
        fprintf(stderr, "[metal] gemm_fp32_vec: buffer lookup failed — aborting\n");
        abort();
    }

    NSUInteger A_off = (NSUInteger)((const uint8_t*)input  - (const uint8_t*)[A_buf  contents]);
    NSUInteger C_off = (NSUInteger)((      uint8_t*)output - (      uint8_t*)[C_buf  contents]);

    // C[1, C_out] = input[1, C_in] × weight^T[C_in, C_out]
    metal_mps_matmul(A_buf, A_off, B_orig, 0, C_buf, C_off, 1, C_in, C_out);

    // Bias add.
    if (bias) {
        id<MTLBuffer> bias_buf = metal_get_or_create_cpu_buf(bias, (size_t)C_out * sizeof(float));
        if (bias_buf) {
            MetalDispatch d("add_vectors_k");
            d.buf(C_buf, C_off);
            d.buf(bias_buf);
            d.run((NSUInteger)C_out);
        } else {
            fprintf(stderr, "[metal] gemm_fp32_vec: bias_buf alloc failed — aborting\n");
            abort();
        }
    }
}

// ── seqgemm_fp32 — MPS MatrixMultiplication ──────────────────────
void metal_seqgemm_fp32(
    const float* input,
    const float* weight,
    const float* bias,
    float*       output,
    int N, int C_in, int C_out,
    bool         gelu,
    const float* w_packed,
    StreamHandle /*stream*/)
{
    id<MTLBuffer> buf_in  = metal_get_buf(input);
    id<MTLBuffer> buf_wt  = metal_get_weight_buf(w_packed);
    id<MTLBuffer> buf_out = metal_get_buf(output);

    if (!buf_in || !buf_wt || !buf_out) {
        fprintf(stderr, "[metal] seqgemm_fp32: buffer lookup failed — aborting\n");
        abort();
    }

    NSUInteger off_in  = (NSUInteger)((const uint8_t*)input  - (const uint8_t*)[buf_in  contents]);
    NSUInteger off_wt  = 0;   // original weight buffer starts at offset 0
    NSUInteger off_out = (NSUInteger)((const uint8_t*)output - (const uint8_t*)[buf_out contents]);

    // SEQGEMM_DEBUG: inspect output buffer before and after MPS matmul.
#ifdef SEQGEMM_DEBUG
    static int sg_call = 0;
    bool sg_match = (N == 13 && C_in == 768 && C_out == 2048);
    if (sg_match) {
        ++sg_call;
        metal_flush();  // ensure prior GPU writes are visible
        const float* dbg_pre = (const float*)((const uint8_t*)[buf_out contents] + off_out);
        printf("[SEQGEMM_DEBUG] call=%d pre-matmul (T=0,C=284)=%.8f  (T=12,C=284)=%.8f\n",
               sg_call, dbg_pre[0*C_out+284], dbg_pre[12*C_out+284]);
    }
#endif

    // C = input[N, C_in] × weight^T[C_in, C_out]
    metal_mps_matmul(buf_in, off_in, buf_wt, off_wt, buf_out, off_out, N, C_in, C_out);

#ifdef SEQGEMM_DEBUG
    if (sg_match) {
        metal_flush();  // commit the matmul
        const float* dbg_post = (const float*)((const uint8_t*)[buf_out contents] + off_out);
        // Also read bias at C=284 if available
        float bias_284 = bias ? bias[284] : 0.f;
        printf("[SEQGEMM_DEBUG] call=%d post-matmul (T=0,C=284)=%.8f  (T=12,C=284)=%.8f  bias[284]=%.8f  off_out=%lu\n",
               sg_call, dbg_post[0*C_out+284], dbg_post[12*C_out+284], bias_284, (unsigned long)off_out);
    }
#endif

    // Bias add.
    if (bias) {
        id<MTLBuffer> buf_bias = metal_get_buf(bias);
        NSUInteger bias_off = 0;
        id<MTLBuffer> tmp_bias = nil;
        if (buf_bias) {
            bias_off = (NSUInteger)((const uint8_t*)bias - (const uint8_t*)[buf_bias contents]);
        } else {
            tmp_bias = metal_get_or_create_cpu_buf(bias, (size_t)C_out * sizeof(float));
            buf_bias = tmp_bias;
        }
        if (buf_bias) {
            int tc[2] = { N, C_out };
            MetalDispatch d("seqgemm_bias_k");
            d.buf(buf_out, off_out);
            d.buf(buf_bias, bias_off);
            d.bytes(tc, sizeof(tc));
            d.run((NSUInteger)(N * C_out));
        }
    }

    // GELU.
    if (gelu) {
        MetalDispatch d("gelu_inplace_k");
        d.buf(buf_out, off_out);
        d.run((NSUInteger)(N * C_out));
    }

#ifdef SEQGEMM_DEBUG
    if (sg_match) {
        metal_flush();
        const float* dbg_final = (const float*)((const uint8_t*)[buf_out contents] + off_out);
        printf("[SEQGEMM_DEBUG] call=%d post-gelu  (T=0,C=284)=%.8f  (T=12,C=284)=%.8f\n",
               sg_call, dbg_final[0*C_out+284], dbg_final[12*C_out+284]);
        // Also check if bias pointer is in activation map (should be nil for weight bias).
        id<MTLBuffer> chk_bias = bias ? metal_get_buf(bias) : nil;
        printf("[SEQGEMM_DEBUG] call=%d bias_in_actmap=%s\n",
               sg_call, chk_bias ? "YES (WRONG!)" : "no");
    }
#endif

    // All sub-ops batched in g_active_cb; copy_d2h or next CPU boundary will flush.
}

// ── seqgemm_int8 — GPU ───────────────────────────────────────────
// Steps:
//  1. CPU: find max|x| over input (unified memory, direct read)
//  2. CPU: compute req_scratch[c] = in_scale * w_scales[c] (write to shared buf)
//  3. GPU: quantize float→int8 via quantize_float_to_int8_k
//  4. GPU: INT8 GEMM → float via gemm_int8_float_k
//  5. GPU: bias add + optional GELU
void metal_seqgemm_int8(
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
    StreamHandle   stream)
{
    id<MTLBuffer> buf_in  = metal_get_buf(input);
    id<MTLBuffer> B_orig  = metal_get_weight_buf(w_packed);
    id<MTLBuffer> buf_out = metal_get_buf(output);
    id<MTLBuffer> buf_inp = metal_get_buf(inp_scratch);
    id<MTLBuffer> buf_req = metal_get_buf(req_scratch);

    if (!buf_in || !B_orig || !buf_out || !buf_inp || !buf_req) {
        fprintf(stderr, "[metal] seqgemm_int8: buffer lookup failed — aborting\n");
        abort();
    }

    NSUInteger in_off  = (NSUInteger)((const uint8_t*)input       - (const uint8_t*)[buf_in  contents]);
    NSUInteger out_off = (NSUInteger)((const uint8_t*)output      - (const uint8_t*)[buf_out contents]);
    NSUInteger inp_off = (NSUInteger)((const uint8_t*)inp_scratch  - (const uint8_t*)[buf_inp contents]);
    NSUInteger req_off = (NSUInteger)((const uint8_t*)req_scratch  - (const uint8_t*)[buf_req contents]);

    // Step 1: find max|x| on CPU (unified memory).
    // The previous op's metal_flush() ensures GPU writes to input are visible.
    metal_flush(); // flush any pending GPU work from the previous op
    int n_elems = N * C_in;
    float max_abs = 0.f;
    for (int i = 0; i < n_elems; ++i) {
        float a = input[i] < 0.f ? -input[i] : input[i];
        if (a > max_abs) max_abs = a;
    }
    float in_scale  = max_abs > 0.f ? max_abs / 127.f : 1e-8f;
    float inv_scale = max_abs > 0.f ? 127.f / max_abs : 0.f;

    // Step 2: compute req_scratch (per-column float scale) on CPU.
    // req_scratch is in shared Metal buffer — direct CPU write is fine.
    for (int c = 0; c < C_out; ++c)
        req_scratch[c] = in_scale * w_scales[c];

    // Step 3: GPU quantize float → int8.
    {
        MetalDispatch d("quantize_float_to_int8_k");
        d.buf(buf_in,  in_off);
        d.buf(buf_inp, inp_off);
        d.bytes(&inv_scale, sizeof(float));
        d.run((NSUInteger)n_elems);
    }

    // Step 4: GPU INT8 GEMM → float.
    id<MTLBuffer> ebias_buf = metal_get_or_create_cpu_buf(eff_zeros, (size_t)C_out * sizeof(int64_t));
    if (!ebias_buf) {
        fprintf(stderr, "[metal] seqgemm_int8: ebias_buf alloc failed — aborting\n");
        abort();
    }
    {
        const int T = 16;
        int dims[4] = { N, C_in, C_out, 0 };
        NSUInteger tg_n = (NSUInteger)((C_out + T-1) / T);
        NSUInteger tg_m = (NSUInteger)((N     + T-1) / T);
        MetalDispatch d("gemm_int8_float_k");
        d.buf(buf_inp, inp_off);
        d.buf(B_orig);
        d.buf(ebias_buf);
        d.buf(buf_req, req_off);
        d.buf(buf_out, out_off);
        d.bytes(dims, sizeof(dims));
        d.tg_mem(2 * T * T);
        d.run_threadgroups(MTLSizeMake(tg_n, tg_m, 1), MTLSizeMake((NSUInteger)T, (NSUInteger)T, 1));
    }

    // Step 5: bias add.
    if (bias) {
        id<MTLBuffer> buf_bias = metal_get_buf(bias);
        NSUInteger    bias_off = 0;
        id<MTLBuffer> tmp_bias = nil;
        if (buf_bias) {
            bias_off = (NSUInteger)((const uint8_t*)bias - (const uint8_t*)[buf_bias contents]);
        } else {
            tmp_bias = metal_get_or_create_cpu_buf(bias, (size_t)C_out * sizeof(float));
            buf_bias = tmp_bias;
        }
        if (buf_bias) {
            int tc[2] = { N, C_out };
            MetalDispatch d("seqgemm_bias_k");
            d.buf(buf_out,  out_off);
            d.buf(buf_bias, bias_off);
            d.bytes(tc, sizeof(tc));
            d.run((NSUInteger)(N * C_out));
        }
    }

    // Step 6: GELU.
    if (gelu) {
        MetalDispatch d("gelu_inplace_k");
        d.buf(buf_out, out_off);
        d.run((NSUInteger)(N * C_out));
    }
    // GPU sub-ops (quantize + INT8 GEMM + bias + GELU) are batched in g_active_cb.
    // copy_d2h will commit the entire forward pass at once.
}
