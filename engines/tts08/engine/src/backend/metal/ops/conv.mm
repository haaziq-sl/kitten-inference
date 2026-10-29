// src/backend/metal/ops/conv.mm
// Convolution ops for the Metal backend.
//
// GPU paths implemented:
//   conv2d_fp32_nhwc          : im2col (im2col_fp32_nhwc_k) + MPS MatMul
//   conv2d_winograd_nhwc_fp32 : same, using raw_weight (kH=kW=3)
//   conv2d_int8 (nhwc=true, groups==1, kH>1||kW>1) : im2col + gemm_int8_requant_k
//   conv2d_winograd_nhwc_int8 : delegates to CPU (complex Winograd transform)
//   conv1d_int8_nhwc          : GPU im2col + CPU quantize + GPU gemm_int8_float_k
//
// All conditional error paths abort() — no silent CPU fallbacks.
// All other cases use the growing g_col_buf scratch buffer.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "metal_ops.hpp"
#include "backend/metal/metal_helpers.h"
#include "backend/cpu/ops_neon.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ── conv1d_int8 private scratch buffers ───────────────────────────
// g_c1i_int8:  quantized im2col [T_out, K_pad] INT8
// g_c1i_req:   per-channel req_scratch [C_out] float (in_scale * w_scales[c])
// g_c1i_ebias: zero-filled [C_out] int64 (conv1d_int8 has no weight zero-point)
static id<MTLBuffer> g_c1i_int8    = nil; static size_t g_c1i_int8_sz  = 0;
static id<MTLBuffer> g_c1i_req     = nil; static size_t g_c1i_req_sz   = 0;
static id<MTLBuffer> g_c1i_ebias   = nil; static size_t g_c1i_ebias_sz = 0;

static id<MTLBuffer> c1i_int8_buf(size_t need) {
    if (need > g_c1i_int8_sz) {
        g_c1i_int8    = [g_metal_device newBufferWithLength:need options:MTLResourceStorageModeShared];
        g_c1i_int8_sz = g_c1i_int8 ? need : 0;
    }
    return g_c1i_int8;
}
static id<MTLBuffer> c1i_req_buf(size_t need) {
    if (need > g_c1i_req_sz) {
        g_c1i_req    = [g_metal_device newBufferWithLength:need options:MTLResourceStorageModeShared];
        g_c1i_req_sz = g_c1i_req ? need : 0;
    }
    return g_c1i_req;
}
static id<MTLBuffer> c1i_ebias_buf(size_t need) {
    if (need > g_c1i_ebias_sz) {
        g_c1i_ebias    = [g_metal_device newBufferWithLength:need options:MTLResourceStorageModeShared];
        if (g_c1i_ebias) memset([g_c1i_ebias contents], 0, need);
        g_c1i_ebias_sz = g_c1i_ebias ? need : 0;
    }
    return g_c1i_ebias;
}

// ── Internal helpers ──────────────────────────────────────────────

// Dispatch im2col_fp32_nhwc_k: fills col_buf with [oH*oW, K] float
static void dispatch_im2col_fp32(
    id<MTLBuffer> in_buf,  NSUInteger in_off,
    id<MTLBuffer> col_buf,
    int H, int W, int C_in,
    int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH, int dilW,
    int oH, int oW)
{
    int in_dims[4] = { H, W, C_in, kH };
    int kp[4]      = { kW, padH, padW, strideH };
    int op[4]      = { strideW, oH, oW, dilH };

    MetalDispatch d("im2col_fp32_nhwc_k");
    d.buf(in_buf, in_off);
    d.buf(col_buf);
    d.bytes(&in_dims, sizeof(in_dims));
    d.bytes(&kp,      sizeof(kp));
    d.bytes(&op,      sizeof(op));
    d.bytes(&dilW,    sizeof(dilW));
    d.run((NSUInteger)(oH * oW));
}

// Dispatch im2col_int8_nhwc_k: fills col_buf with [oH*oW, K] int8
static void dispatch_im2col_int8(
    id<MTLBuffer> in_buf,  NSUInteger in_off,
    id<MTLBuffer> col_buf,
    int H, int W, int C_in,
    int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilH, int dilW,
    int oH, int oW,
    int in_zp)
{
    int in_dims[4] = { H, W, C_in, kH };
    int kp[4]      = { kW, padH, padW, strideH };
    int op[4]      = { strideW, oH, oW, dilH };

    MetalDispatch d("im2col_int8_nhwc_k");
    d.buf(in_buf, in_off);
    d.buf(col_buf);
    d.bytes(&in_dims, sizeof(in_dims));
    d.bytes(&kp,      sizeof(kp));
    d.bytes(&op,      sizeof(op));
    d.bytes(&dilW,    sizeof(dilW));
    d.bytes(&in_zp,   sizeof(in_zp));
    d.run((NSUInteger)(oH * oW));
}

// ── FP32 conv2d (NHWC) ───────────────────────────────────────────
void metal_conv2d_fp32_nhwc(
    const float* input,
    const float* weight,    // [C_out, kH, kW, C_in] NHWC row-major (CPU pointer)
    const float* bias,
    float*       output,
    int C_in, int H, int W,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    bool         relu,
    const float* w_packed,  // NEON-packed (unused for Metal GPU path)
    float*       scratch_col,
    int dilation_h, int dilation_w,
    StreamHandle stream)
{
    id<MTLBuffer> in_buf  = metal_get_buf(input);
    id<MTLBuffer> out_buf = metal_get_buf(output);
    if (!in_buf || !out_buf) {
        fprintf(stderr, "[metal] conv2d_fp32_nhwc: buffer lookup failed — aborting\n");
        abort();
    }

    const int oH = (H + 2*pad_h - dilation_h*(kH-1) - 1) / stride_h + 1;
    const int oW = (W + 2*pad_w - dilation_w*(kW-1) - 1) / stride_w + 1;
    const int K  = kH * kW * C_in;

    NSUInteger in_off  = (NSUInteger)((const uint8_t*)input  - (const uint8_t*)[in_buf  contents]);
    NSUInteger out_off = (NSUInteger)((      uint8_t*)output - (      uint8_t*)[out_buf contents]);

    // Get weight buffer: try the weight registry first (keyed by the packed ptr
    // after upload_weights), then fall back to the CPU-pointer cache.
    id<MTLBuffer> wt_buf = w_packed ? metal_get_weight_buf(w_packed) : nil;
    if (!wt_buf)
        wt_buf = metal_get_or_create_cpu_buf(weight, (size_t)C_out * K * sizeof(float));
    if (!wt_buf) {
        fprintf(stderr, "[metal] conv2d_fp32_nhwc: weight buffer alloc failed — aborting\n");
        abort();
    }

    const bool is_1x1 = (kH == 1 && kW == 1 &&
                         pad_h == 0 && pad_w == 0 &&
                         stride_h == 1 && stride_w == 1 &&
                         dilation_h == 1 && dilation_w == 1);

    if (is_1x1) {
        // Input [H*W, C_in] is already the column matrix.
        metal_mps_matmul(in_buf, in_off, wt_buf, 0, out_buf, out_off,
                         H * W, C_in, C_out);
    } else {
        // General case: im2col then MPS MatMul.
        size_t col_bytes = (size_t)oH * oW * K * sizeof(float);
        id<MTLBuffer> col_buf = metal_get_col_buf(col_bytes);
        if (!col_buf) {
            fprintf(stderr, "[metal] conv2d_fp32_nhwc: col_buf alloc failed — aborting\n");
            abort();
        }
        dispatch_im2col_fp32(in_buf, in_off, col_buf,
                             H, W, C_in, kH, kW,
                             pad_h, pad_w, stride_h, stride_w,
                             dilation_h, dilation_w, oH, oW);
        // col [oH*oW, K] × weight^T [K, C_out] → output [oH*oW, C_out]
        metal_mps_matmul(col_buf, 0, wt_buf, 0, out_buf, out_off,
                         oH * oW, K, C_out);
    }

    // Bias add + optional ReLU.
    if (bias) {
        id<MTLBuffer> bias_buf = metal_get_or_create_cpu_buf(bias, (size_t)C_out * sizeof(float));
        if (bias_buf) {
            int dims_br[2] = { oH * oW, C_out };
            int do_relu    = relu ? 1 : 0;
            MetalDispatch d("conv_bias_relu_k");
            d.buf(out_buf, out_off);
            d.buf(bias_buf);
            d.bytes(&dims_br, sizeof(dims_br));
            d.bytes(&do_relu, sizeof(do_relu));
            d.run((NSUInteger)(oH * oW * C_out));
        } else {
            fprintf(stderr, "[metal] conv2d_fp32_nhwc: bias_buf alloc failed — aborting\n");
            abort();
        }
    } else if (relu) {
        MetalDispatch d("relu_inplace_k");
        d.buf(out_buf, out_off);
        d.run((NSUInteger)(oH * oW * C_out));
    }
    // GPU sub-ops (im2col + matmul + bias/relu) are batched in g_active_cb.
}

// ── FP32 Winograd conv2d (NHWC) ───────────────────────────────────
// On Metal we skip the Winograd transform and do direct 3×3 conv via
// im2col + MPS, using the raw_weight parameter (original NHWC weights).
void metal_conv2d_winograd_nhwc_fp32(
    const float*         input,
    const float* const*  w_wino_packed,  // Winograd-packed (ignored on Metal)
    const float*         bias,
    float*               output,
    int C_in, int H, int W, int C_out, bool relu,
    float*               scratch,
    StreamHandle         stream,
    const float*         raw_weight)    // [C_out, 3, 3, C_in] NHWC
{
    if (!raw_weight) {
        fprintf(stderr, "[metal] conv2d_winograd_nhwc_fp32: raw_weight is null — aborting\n");
        abort();
    }
    // Delegate to the direct conv2d path (3×3 stride=1 pad=1 dil=1).
    metal_conv2d_fp32_nhwc(input, raw_weight, bias, output,
                           C_in, H, W, C_out,
                           /*kH=*/3, /*kW=*/3,
                           /*stride_h=*/1, /*stride_w=*/1,
                           /*pad_h=*/1,    /*pad_w=*/1,
                           relu,
                           /*w_packed=*/nullptr,
                           /*scratch_col=*/nullptr,
                           /*dilation_h=*/1, /*dilation_w=*/1,
                           stream);
}

// ── INT8 conv2d (NHWC) ────────────────────────────────────────────
void metal_conv2d_int8(
    const int8_t*  input,
    const int8_t*  weight,      // [C_out, K] NCHW-order original (CPU)
    const int8_t*  w_pre_packed,// NHWC-order SDOT-packed (Metal after upload)
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
    bool           nhwc,        // always true for Metal path
    const int32_t* b_row_sums,
    StreamHandle   stream)
{
    // Metal GPU path: nhwc=true, groups==1, any kernel size.
    if (kH == 1 && kW == 1) {
        fprintf(stderr, "[metal] conv2d_int8: 1×1 conv not yet supported on GPU — aborting\n");
        abort();
    }
    if (!nhwc) {
        fprintf(stderr, "[metal] conv2d_int8: NCHW layout not supported — aborting\n");
        abort();
    }
    if (groups > 1) {
        fprintf(stderr, "[metal] conv2d_int8: grouped/depthwise conv not yet implemented on GPU — aborting\n");
        abort();
    }

    id<MTLBuffer> in_buf  = metal_get_buf(input);
    id<MTLBuffer> out_buf = metal_get_buf(output);
    if (!in_buf || !out_buf) {
        fprintf(stderr, "[metal] conv2d_int8: buffer lookup failed — aborting\n");
        abort();
    }

    const int oH = (H + 2*pad_h - kH) / stride_h + 1;
    const int oW = (W + 2*pad_w - kW) / stride_w + 1;
    const int K  = kH * kW * C_in;

    NSUInteger in_off  = (NSUInteger)((const int8_t*)input  - (const int8_t*)[in_buf  contents]);
    NSUInteger out_off = (NSUInteger)((      int8_t*)output - (      int8_t*)[out_buf contents]);

    // Get original NHWC-ordered weight buffer.
    // w_pre_packed is the NHWC-order SDOT-packed buffer; the registry holds
    // the original NHWC-order int8 data keyed by its Metal contents pointer.
    id<MTLBuffer> wt_buf = metal_get_weight_buf(w_pre_packed);
    if (!wt_buf) {
        // Registry miss (e.g., 1×1 where w_packed_nhwc shares w_packed memory).
        // Fall back to creating a buffer from the CPU weight (NCHW order).
        // For 1×1 conv NCHW == NHWC, so this is correct.
        wt_buf = metal_get_or_create_cpu_buf(weight, (size_t)C_out * K);
    }
    if (!wt_buf) {
        fprintf(stderr, "[metal] conv2d_int8: weight buffer alloc failed — aborting\n");
        abort();
    }

    // Upload fixed per-channel arrays.
    id<MTLBuffer> ebias_buf = metal_get_or_create_cpu_buf(eff_bias, (size_t)C_out * sizeof(int64_t));
    id<MTLBuffer> mult_buf  = metal_get_or_create_cpu_buf(req_mult,  (size_t)C_out * sizeof(int32_t));
    id<MTLBuffer> exp_buf   = metal_get_or_create_cpu_buf(req_exp,   (size_t)C_out * sizeof(int32_t));
    if (!ebias_buf || !mult_buf || !exp_buf) {
        fprintf(stderr, "[metal] conv2d_int8: per-channel buffer alloc failed — aborting\n");
        abort();
    }

    const bool is_1x1 = (kH == 1 && kW == 1 &&
                         pad_h == 0 && pad_w == 0 &&
                         stride_h == 1 && stride_w == 1);

    id<MTLBuffer> col_buf = nil;
    if (!is_1x1) {
        size_t col_bytes = (size_t)oH * oW * K;
        col_buf = metal_get_col_buf(col_bytes);
        if (!col_buf) {
            fprintf(stderr, "[metal] conv2d_int8: col_buf alloc failed — aborting\n");
            abort();
        }
        dispatch_im2col_int8(in_buf, in_off, col_buf,
                             H, W, C_in, kH, kW,
                             pad_h, pad_w, stride_h, stride_w,
                             /*dilH=*/1, /*dilW=*/1,
                             oH, oW, (int)in_zp);
    }

    // GEMM: col [oH*oW, K] × wt [C_out, K]^T → output [oH*oW, C_out]
    id<MTLBuffer> A_buf   = is_1x1 ? in_buf  : col_buf;
    NSUInteger    A_off   = is_1x1 ? in_off  : 0;
    int           M       = oH * oW;

    const int T = 16;
    int dims[4] = { M, K, C_out, (int)out_zp };
    NSUInteger tg_n = (NSUInteger)((C_out + T - 1) / T);
    NSUInteger tg_m = (NSUInteger)((M     + T - 1) / T);

    MetalDispatch d("gemm_int8_requant_k");
    d.buf(A_buf,     A_off);
    d.buf(wt_buf);
    d.buf(ebias_buf);
    d.buf(mult_buf);
    d.buf(exp_buf);
    d.buf(out_buf,   out_off);
    d.bytes(&dims, sizeof(dims));
    d.tg_mem(2 * T * T);
    d.run_threadgroups(MTLSizeMake(tg_n, tg_m, 1),
                       MTLSizeMake((NSUInteger)T, (NSUInteger)T, 1));
    // GPU sub-ops (im2col + INT8 GEMM) are batched in g_active_cb.
}

// ── FP32 1D Conv ─────────────────────────────────────────────────
// Direct convolution kernel — avoids im2col intermediate buffer.
// Each GPU thread computes one output element: output[t_out, c_out].
// weight: [C_out, kernel_size, C_in] row-major (NHWC with W=1).
// All ops batch into g_active_cb; no metal_flush() at end.
void metal_conv1d_fp32(
    const float* input,
    const float* weight,   // [C_out, kernel_size, C_in] NHWC
    const float* bias,
    float*       output,
    int C_in, int T, int C_out,
    int kernel_size, int stride, int padding, int dilation,
    bool relu,
    StreamHandle stream)
{
    id<MTLBuffer> in_buf  = metal_get_buf(input);
    id<MTLBuffer> out_buf = metal_get_buf(output);
    if (!in_buf || !out_buf) {
        fprintf(stderr, "[metal] conv1d_fp32: buffer lookup failed — aborting\n");
        abort();
    }

    const int T_out = (T + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    NSUInteger in_off  = (NSUInteger)((const uint8_t*)input  - (const uint8_t*)[in_buf  contents]);
    NSUInteger out_off = (NSUInteger)((      uint8_t*)output - (      uint8_t*)[out_buf contents]);

    const size_t wt_bytes = (size_t)C_out * kernel_size * C_in * sizeof(float);
    id<MTLBuffer> wt_buf = metal_get_or_create_cpu_buf(weight, wt_bytes);
    if (!wt_buf) {
        fprintf(stderr, "[metal] conv1d_fp32: weight buffer alloc failed — aborting\n");
        abort();
    }

    id<MTLBuffer> bias_buf = nil;
    if (bias) {
        bias_buf = metal_get_or_create_cpu_buf(bias, (size_t)C_out * sizeof(float));
        if (!bias_buf) {
            fprintf(stderr, "[metal] conv1d_fp32: bias_buf alloc failed — aborting\n");
            abort();
        }
    }

    // Choose kernel based on filter volume per output channel.
    // Direct kernel avoids im2col intermediate (good for small C_in*ksz where weight
    // fits in GPU L2 cache).  For large filters, im2col+MPS wins because MPS tiles
    // the GEMM and reuses the weight matrix across all output positions.
    // Threshold: kernel_size * C_in <= 128 (e.g. C_in=32/ksz=3 → 96 ✓; C_in=256/ksz=7 → 1792 ✗).
    const bool use_direct = ((size_t)kernel_size * C_in <= 128);

    if (use_direct) {
        int dims4[4]  = { T, C_in, T_out, C_out };
        int kp4[4]    = { kernel_size, stride, padding, dilation };
        int flags2[2] = { bias_buf ? 1 : 0, relu ? 1 : 0 };

        const NSUInteger TG_X = 32, TG_Y = 8;
        NSUInteger tg_x = ((NSUInteger)C_out + TG_X - 1) / TG_X;
        NSUInteger tg_y = ((NSUInteger)T_out + TG_Y - 1) / TG_Y;

        MetalDispatch d("conv1d_direct_k");
        d.buf(in_buf,  in_off);
        d.buf(wt_buf);
        d.buf(bias_buf ? bias_buf : wt_buf);  // dummy when no bias; kernel checks flags.x
        d.buf(out_buf, out_off);
        d.bytes(dims4,  sizeof(dims4));
        d.bytes(kp4,    sizeof(kp4));
        d.bytes(flags2, sizeof(flags2));
        d.run_threadgroups(MTLSizeMake(tg_x, tg_y, 1), MTLSizeMake(TG_X, TG_Y, 1));
    } else {
        // im2col + MPS: better for large filters where MPS tiling reuses the weight.
        const int oH = T_out, oW = 1;
        const int K  = kernel_size * C_in;
        size_t col_bytes = (size_t)T_out * K * sizeof(float);
        id<MTLBuffer> col_buf = metal_get_col_buf(col_bytes);
        if (!col_buf) {
            fprintf(stderr, "[metal] conv1d_fp32: col_buf alloc failed — aborting\n");
            abort();
        }
        dispatch_im2col_fp32(in_buf, in_off, col_buf,
                             T, 1, C_in, kernel_size, 1,
                             padding, 0, stride, 1,
                             dilation, 1, oH, oW);
        metal_mps_matmul(col_buf, 0, wt_buf, 0, out_buf, out_off,
                         T_out, K, C_out);
        if (bias_buf) {
            int dims_br[2] = { T_out, C_out };
            int do_relu    = relu ? 1 : 0;
            MetalDispatch db("conv_bias_relu_k");
            db.buf(out_buf, out_off);
            db.buf(bias_buf);
            db.bytes(&dims_br, sizeof(dims_br));
            db.bytes(&do_relu, sizeof(do_relu));
            db.run((NSUInteger)(T_out * C_out));
        } else if (relu) {
            MetalDispatch dr("relu_inplace_k");
            dr.buf(out_buf, out_off);
            dr.run((NSUInteger)(T_out * C_out));
        }
    }
    // Batched in g_active_cb; no metal_flush() needed here.
}

// ── FP32 Transposed 1D Conv ───────────────────────────────────────
// groups == 1  (standard):
//   For each kernel position k:
//     scratch[T, C_out] = input[T, C_in] × W_k[C_out, C_in]^T   (MPS)
//     output[t*stride + k - padding, :] += scratch[t, :]         (col2im_scatter_k)
//   weight layout: [kernel_size, C_out, C_in] row-major
//
// depthwise (groups == C_in == C_out):
//   conv_transpose1d_dw_k — one thread per (t_out, c)
//   weight layout: [C, kernel_size] row-major
//
// NOTE: No metal_flush() at end — all ops batched in g_active_cb.
void metal_conv_transpose1d_fp32(
    const float* input,
    const float* weight,
    const float* bias,     // [C_out] nullable
    float*       output,   // [T_out, C_out]
    int C_in, int T, int C_out,
    int kernel_size, int stride, int padding, int output_padding,
    int groups,
    StreamHandle stream)
{
    id<MTLBuffer> in_buf  = metal_get_buf(input);
    id<MTLBuffer> out_buf = metal_get_buf(output);
    if (!in_buf || !out_buf) {
        fprintf(stderr, "[metal] conv_transpose1d_fp32: buffer lookup failed — aborting\n");
        abort();
    }

    NSUInteger in_off  = (NSUInteger)((const uint8_t*)input  - (const uint8_t*)[in_buf  contents]);
    NSUInteger out_off = (NSUInteger)((      uint8_t*)output - (      uint8_t*)[out_buf contents]);

    const int T_out = (T - 1) * stride + kernel_size - 2 * padding + output_padding;

    // ── Depthwise path (groups == C_in == C_out) ─────────────────
    if (groups > 1 && C_in == groups && C_out == groups) {
        // weight: [C, kernel_size] row-major
        const size_t wt_bytes = (size_t)C_in * kernel_size * sizeof(float);
        id<MTLBuffer> wt_buf = metal_get_or_create_cpu_buf(weight, wt_bytes);
        if (!wt_buf) {
            fprintf(stderr, "[metal] conv_transpose1d_fp32: dw weight buffer alloc failed — aborting\n");
            abort();
        }
        int dims4[4] = { T, C_in, T_out, kernel_size };
        int params2[2] = { stride, padding };
        const NSUInteger TG_X = 32, TG_Y = 8;
        NSUInteger tg_x = ((NSUInteger)C_in  + TG_X - 1) / TG_X;
        NSUInteger tg_y = ((NSUInteger)T_out + TG_Y - 1) / TG_Y;
        MetalDispatch d("conv_transpose1d_dw_k");
        d.buf(in_buf, in_off);
        d.buf(wt_buf);
        d.buf(out_buf, out_off);
        d.bytes(dims4,   sizeof(dims4));
        d.bytes(params2, sizeof(params2));
        d.run_threadgroups(MTLSizeMake(tg_x, tg_y, 1), MTLSizeMake(TG_X, TG_Y, 1));

        if (bias) {
            id<MTLBuffer> bias_buf = metal_get_or_create_cpu_buf(bias, (size_t)C_out * sizeof(float));
            if (!bias_buf) { fprintf(stderr, "[metal] conv_transpose1d_fp32: dw bias alloc failed\n"); abort(); }
            int TC[2] = { T_out, C_out };
            MetalDispatch db("seqgemm_bias_k");
            db.buf(out_buf, out_off);
            db.buf(bias_buf);
            db.bytes(TC, sizeof(TC));
            db.run((NSUInteger)(T_out * C_out));
        }
        return;
    }

    // ── groups == 1 path ─────────────────────────────────────────
    // Get weight buffer: [kernel_size, C_out, C_in] row-major.
    const size_t wt_bytes = (size_t)kernel_size * C_out * C_in * sizeof(float);
    id<MTLBuffer> wt_buf = metal_get_or_create_cpu_buf(weight, wt_bytes);
    if (!wt_buf) {
        fprintf(stderr, "[metal] conv_transpose1d_fp32: weight buffer alloc failed — aborting\n");
        abort();
    }

    // Scratch buffer: [T, C_out] for per-k GEMM output.
    const size_t scratch_bytes = (size_t)T * C_out * sizeof(float);
    id<MTLBuffer> scratch_buf = metal_get_col_buf(scratch_bytes);
    if (!scratch_buf) {
        fprintf(stderr, "[metal] conv_transpose1d_fp32: scratch_buf alloc failed — aborting\n");
        abort();
    }

    // Step 1: Zero output[T_out, C_out] using blit.
    {
        metal_end_lm_encoder();
        id<MTLCommandBuffer>      cb   = metal_cmd_buf();
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit fillBuffer:out_buf
                   range:NSMakeRange(out_off, (size_t)T_out * C_out * sizeof(float))
                   value:0];
        [blit endEncoding];
    }

    // Step 2: For each kernel position k:
    //   a) MPS matmul: scratch[T, C_out] = input[T, C_in] × W_k[C_out, C_in]^T
    //   b) col2im_scatter_k: output[t*stride+k-pad, c] += scratch[t, c]
    const NSUInteger wk_stride = (NSUInteger)(C_out * C_in) * sizeof(float);
    for (int k = 0; k < kernel_size; ++k) {
        NSUInteger wk_off = (NSUInteger)k * wk_stride;

        // (a) GEMM: scratch = input × W_k^T   (M=T, K=C_in, N=C_out)
        metal_mps_matmul(in_buf, in_off, wt_buf, wk_off, scratch_buf, 0, T, C_in, C_out);

        // (b) Scatter-add to output.
        int dims4[4] = { T, C_out, T_out, stride };
        int kp2[2]   = { k, padding };
        MetalDispatch d("col2im_scatter_k");
        d.buf(scratch_buf);
        d.buf(out_buf, out_off);
        d.bytes(dims4, sizeof(dims4));
        d.bytes(kp2,   sizeof(kp2));
        d.run((NSUInteger)(T * C_out));
    }

    // Step 3: Bias add.
    if (bias) {
        id<MTLBuffer> bias_buf = metal_get_or_create_cpu_buf(bias, (size_t)C_out * sizeof(float));
        if (bias_buf) {
            int TC[2] = { T_out, C_out };
            MetalDispatch d("seqgemm_bias_k");
            d.buf(out_buf, out_off);
            d.buf(bias_buf);
            d.bytes(TC, sizeof(TC));
            d.run((NSUInteger)(T_out * C_out));
        } else {
            fprintf(stderr, "[metal] conv_transpose1d_fp32: bias_buf alloc failed — aborting\n");
            abort();
        }
    }
    // All ops batched in g_active_cb; caller flushes at CPU boundary.
}

// ── conv1d_int8_nhwc: GPU im2col + CPU quantize + GPU gemm_int8_float_k ──────
//
// Pipeline:
//   1. GPU im2col: input [T, C_in] → col_buf [T_out, K] float
//   2. metal_flush() so CPU can read col_buf (unified memory)
//   3. CPU: global max-abs over col [T_out, K]; compute inv_scale / in_scale
//   4. CPU: quantize col → int8_buf [T_out, K_pad] (zero-pad K→K_pad per row)
//   5. CPU: req_buf[c] = in_scale * w_scales[c]
//   6. GPU: gemm_int8_float_k  int8_buf [T_out, K_pad] × w_packed [C_out, K_pad]^T
//            → out [T_out, C_out] float,  C[m,n] = (acc + ebias[n]) * req[n]
//            ebias = 0 (conv1d_int8 has no weight zero-point)
//   7. GPU: bias add via seqgemm_bias_k
//
// Notes:
//   - w_packed is [C_out, K_pad] row-major INT8, K_pad = ceil4(K = kernel_size*C_in).
//     The extra K_pad-K columns are zero-padded in load_weights, contributing 0 to GEMM.
//   - All GPU ops remain batched in g_active_cb; caller flushes at CPU boundary.
void metal_conv1d_int8_nhwc(
    const float*   input,
    const int8_t*  w_packed,
    const float*   w_scales,
    const float*   bias,
    float*         output,
    int T, int C_in, int C_out,
    int kernel_size, int stride, int padding, int dilation,
    StreamHandle)
{
    id<MTLBuffer> buf_in  = metal_get_buf(input);
    id<MTLBuffer> buf_w   = metal_get_buf(w_packed);
    id<MTLBuffer> buf_out = metal_get_buf(output);
    if (!buf_in || !buf_w || !buf_out) {
        fprintf(stderr, "[metal] conv1d_int8_nhwc: buffer lookup failed "
                "(in=%p w=%p out=%p) — aborting\n",
                (void*)input, (void*)w_packed, (void*)output);
        abort();
    }

    const int K     = kernel_size * C_in;
    const int K_pad = (K + 3) / 4 * 4;
    const int T_out = (T + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1;

    NSUInteger in_off  = (NSUInteger)((const uint8_t*)input  - (const uint8_t*)[buf_in  contents]);
    NSUInteger out_off = (NSUInteger)((      uint8_t*)output - (      uint8_t*)[buf_out contents]);

    // ── Step 1: GPU im2col → float col_buf [T_out, K] ─────────────
    size_t col_bytes = (size_t)T_out * K * sizeof(float);
    id<MTLBuffer> col_buf = metal_get_col_buf(col_bytes);
    if (!col_buf) {
        fprintf(stderr, "[metal] conv1d_int8_nhwc: col_buf alloc failed — aborting\n");
        abort();
    }
    dispatch_im2col_fp32(buf_in, in_off, col_buf,
                         T, 1, C_in, kernel_size, 1,
                         padding, 0, stride, 1,
                         dilation, 1, T_out, 1);

    // ── Step 2: flush im2col, then CPU max-abs + quantize ─────────
    metal_flush();

    const float* col_ptr = (const float*)[col_buf contents];
    const int    n_col   = T_out * K;

    float max_abs = 0.f;
    for (int i = 0; i < n_col; ++i) {
        float a = col_ptr[i] < 0.f ? -col_ptr[i] : col_ptr[i];
        if (a > max_abs) max_abs = a;
    }
    const float in_scale  = max_abs > 0.f ? max_abs / 127.f : 1e-8f;
    const float inv_scale = max_abs > 0.f ? 127.f  / max_abs : 0.f;

    // ── Step 3: fill int8_buf [T_out, K_pad] with quantized col ───
    // Each row: K quantized values + (K_pad - K) zeros.
    size_t int8_bytes = (size_t)T_out * K_pad;
    id<MTLBuffer> int8_buf = c1i_int8_buf(int8_bytes);
    if (!int8_buf) {
        fprintf(stderr, "[metal] conv1d_int8_nhwc: int8_buf alloc failed — aborting\n");
        abort();
    }
    int8_t* q_ptr = (int8_t*)[int8_buf contents];
    for (int t = 0; t < T_out; ++t) {
        const float* src = col_ptr + (size_t)t * K;
        int8_t*      dst = q_ptr   + (size_t)t * K_pad;
        for (int k = 0; k < K; ++k) {
            float v = src[k] * inv_scale;
            int   q = (int)(v >= 0.f ? v + 0.5f : v - 0.5f);
            dst[k]  = (int8_t)(q < -127 ? -127 : q > 127 ? 127 : q);
        }
        for (int k = K; k < K_pad; ++k)
            dst[k] = 0;
    }

    // ── Step 4: fill req_buf [C_out] with in_scale * w_scales[c] ──
    size_t req_bytes = (size_t)C_out * sizeof(float);
    id<MTLBuffer> req_buf = c1i_req_buf(req_bytes);
    if (!req_buf) {
        fprintf(stderr, "[metal] conv1d_int8_nhwc: req_buf alloc failed — aborting\n");
        abort();
    }
    float* rp = (float*)[req_buf contents];
    for (int c = 0; c < C_out; ++c)
        rp[c] = in_scale * w_scales[c];

    // ── Step 5: get zero ebias [C_out] int64 ──────────────────────
    size_t ebias_bytes = (size_t)C_out * sizeof(int64_t);
    id<MTLBuffer> ebias_buf = c1i_ebias_buf(ebias_bytes);
    if (!ebias_buf) {
        fprintf(stderr, "[metal] conv1d_int8_nhwc: ebias_buf alloc failed — aborting\n");
        abort();
    }

    // ── Step 6: GPU INT8 GEMM → float output ──────────────────────
    // int8_buf [T_out, K_pad] × w_packed [C_out, K_pad]^T → out [T_out, C_out]
    // C[m,n] = (acc + ebias[n]) * req[n]   (ebias = 0)
    {
        const int   TILE   = 16;
        const int   dims4[4] = { T_out, K_pad, C_out, 0 };
        NSUInteger  tg_n   = (NSUInteger)((C_out + TILE-1) / TILE);
        NSUInteger  tg_m   = (NSUInteger)((T_out + TILE-1) / TILE);

        MetalDispatch d("gemm_int8_float_k");
        d.buf(int8_buf);
        d.buf(buf_w);
        d.buf(ebias_buf);
        d.buf(req_buf);
        d.buf(buf_out, out_off);
        d.bytes(dims4, sizeof(dims4));
        d.tg_mem(2 * TILE * TILE);
        d.run_threadgroups(MTLSizeMake(tg_n, tg_m, 1), MTLSizeMake((NSUInteger)TILE, (NSUInteger)TILE, 1));
    }

    // ── Step 7: bias add ──────────────────────────────────────────
    if (bias) {
        id<MTLBuffer> bias_buf = metal_get_or_create_cpu_buf(bias, (size_t)C_out * sizeof(float));
        if (!bias_buf) {
            fprintf(stderr, "[metal] conv1d_int8_nhwc: bias_buf alloc failed — aborting\n");
            abort();
        }
        const int TC[2] = { T_out, C_out };
        MetalDispatch db("seqgemm_bias_k");
        db.buf(buf_out, out_off);
        db.buf(bias_buf);
        db.bytes(TC, sizeof(TC));
        db.run((NSUInteger)(T_out * C_out));
    }
    // GPU ops (GEMM + bias) stay batched in g_active_cb; caller flushes at CPU boundary.
}

// ── INT8 Winograd conv2d — CPU fallback ───────────────────────────
// The Winograd weight transform is complex and the 16 packed matrices
// are stored in SDOT-tiled Winograd frequency-domain format without a
// direct raw_weight equivalent.  Delegate to CPU.
void metal_conv2d_winograd_nhwc_int8(
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
    StreamHandle          stream)
{
    metal_flush();  // flush pending GPU writes to input before CPU reads it
    conv2d_winograd_nhwc_int8(input, w_wino_packed, wino_weight_scale,
                               wino_input_scale, eff_bias, req_mult, req_exp,
                               req_scale, in_zp, out_zp, output,
                               C_in, H, W, C_out,
                               scratch_input_hat, scratch_output_hat,
                               w_wino_row_sums, stream);
}
