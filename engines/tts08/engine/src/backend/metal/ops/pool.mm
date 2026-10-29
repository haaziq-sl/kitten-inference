// src/backend/metal/ops/pool.mm
// Pool ops for the Metal backend.
// Uses custom Metal compute kernels for all pool operations.
// Aborts if buffer lookup fails (all tensors are Metal-allocated).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_ops.hpp"
#include "backend/metal/metal_helpers.h"
#include "backend/cpu/ops_neon.hpp"

// ── Max pool FP32 NHWC ───────────────────────────────────────────
void metal_maxpool_fp32_nhwc(
    const float* input, float* output,
    int C, int H, int W, int kH, int kW,
    int sh, int sw, int ph, int pw,
    StreamHandle)
{
    id<MTLBuffer> buf_in  = metal_get_buf(input);
    id<MTLBuffer> buf_out = metal_get_buf(output);
    if (!buf_in || !buf_out) {
        fprintf(stderr, "[metal] maxpool_fp32_nhwc: buffer lookup failed — aborting\n");
        abort();
    }

    int oH = (H + 2 * ph - kH) / sh + 1;
    int oW = (W + 2 * pw - kW) / sw + 1;

    int dims[4]    = { H, W, C, kH };
    int params[4]  = { kW, ph, pw, sh };
    int oparams[4] = { sw, oH, oW, 0 };

    NSUInteger off_in  = (NSUInteger)((const uint8_t*)input  - (const uint8_t*)[buf_in  contents]);
    NSUInteger off_out = (NSUInteger)((const uint8_t*)output - (const uint8_t*)[buf_out contents]);

    MetalDispatch d("maxpool_fp32_nhwc_k");
    d.buf(buf_in,  off_in);
    d.buf(buf_out, off_out);
    d.bytes(dims,    sizeof(dims));
    d.bytes(params,  sizeof(params));
    d.bytes(oparams, sizeof(oparams));
    d.run((NSUInteger)(oH * oW * C));
}

// ── Max pool INT8 NHWC ───────────────────────────────────────────
void metal_maxpool_int8_nhwc(
    const int8_t* input, int8_t* output,
    int C, int H, int W, int kH, int kW,
    int sh, int sw, int ph, int pw,
    StreamHandle)
{
    id<MTLBuffer> buf_in  = metal_get_buf(input);
    id<MTLBuffer> buf_out = metal_get_buf(output);
    if (!buf_in || !buf_out) {
        fprintf(stderr, "[metal] maxpool_int8_nhwc: buffer lookup failed — aborting\n");
        abort();
    }

    int oH = (H + 2 * ph - kH) / sh + 1;
    int oW = (W + 2 * pw - kW) / sw + 1;

    int dims[4]    = { H, W, C, kH };
    int params[4]  = { kW, ph, pw, sh };
    int oparams[4] = { sw, oH, oW, 0 };

    NSUInteger off_in  = (NSUInteger)((const uint8_t*)input  - (const uint8_t*)[buf_in  contents]);
    NSUInteger off_out = (NSUInteger)((const uint8_t*)output - (const uint8_t*)[buf_out contents]);

    MetalDispatch d("maxpool_int8_nhwc_k");
    d.buf(buf_in,  off_in);
    d.buf(buf_out, off_out);
    d.bytes(dims,    sizeof(dims));
    d.bytes(params,  sizeof(params));
    d.bytes(oparams, sizeof(oparams));
    d.run((NSUInteger)(oH * oW * C));
}

// ── Max pool INT8 NCHW ────────────────────────────────────────────
void metal_maxpool_int8(
    const int8_t* input, int8_t* output,
    int C, int H, int W, int kH, int kW,
    int sh, int sw, int ph, int pw,
    StreamHandle)
{
    id<MTLBuffer> buf_in  = metal_get_buf(input);
    id<MTLBuffer> buf_out = metal_get_buf(output);
    if (!buf_in || !buf_out) {
        fprintf(stderr, "[metal] maxpool_int8_nchw: buffer lookup failed — aborting\n");
        abort();
    }

    int oH = (H + 2 * ph - kH) / sh + 1;
    int oW = (W + 2 * pw - kW) / sw + 1;

    int dims[4]    = { H, W, C, kH };
    int params[4]  = { kW, ph, pw, sh };
    int oparams[4] = { sw, oH, oW, 0 };

    NSUInteger off_in  = (NSUInteger)((const uint8_t*)input  - (const uint8_t*)[buf_in  contents]);
    NSUInteger off_out = (NSUInteger)((const uint8_t*)output - (const uint8_t*)[buf_out contents]);

    MetalDispatch d("maxpool_int8_nchw_k");
    d.buf(buf_in,  off_in);
    d.buf(buf_out, off_out);
    d.bytes(dims,    sizeof(dims));
    d.bytes(params,  sizeof(params));
    d.bytes(oparams, sizeof(oparams));
    d.run((NSUInteger)(C * oH * oW));
}

// ── Avg pool global FP32 NHWC ─────────────────────────────────────
void metal_avgpool_global_fp32_nhwc(
    const float* input, float* output,
    int C, int H, int W,
    StreamHandle)
{
    id<MTLBuffer> buf_in  = metal_get_buf(input);
    id<MTLBuffer> buf_out = metal_get_buf(output);
    if (!buf_in || !buf_out) {
        fprintf(stderr, "[metal] avgpool_global_fp32_nhwc: buffer lookup failed — aborting\n");
        abort();
    }

    int dims[3] = { H, W, C };

    NSUInteger off_in  = (NSUInteger)((const uint8_t*)input  - (const uint8_t*)[buf_in  contents]);
    NSUInteger off_out = (NSUInteger)((const uint8_t*)output - (const uint8_t*)[buf_out contents]);

    MetalDispatch d("avgpool_fp32_nhwc_k");
    d.buf(buf_in,  off_in);
    d.buf(buf_out, off_out);
    d.bytes(dims, sizeof(dims));
    d.run((NSUInteger)C);
}

// ── Avg pool global INT8 NHWC ─────────────────────────────────────
void metal_avgpool_global_int8_nhwc(
    const int8_t* input,
    float in_sc, int in_zp, float out_sc, int out_zp,
    int8_t* output,
    int C, int H, int W,
    StreamHandle)
{
    id<MTLBuffer> buf_in  = metal_get_buf(input);
    id<MTLBuffer> buf_out = metal_get_buf(output);
    if (!buf_in || !buf_out) {
        fprintf(stderr, "[metal] avgpool_global_int8_nhwc: buffer lookup failed — aborting\n");
        abort();
    }

    int   dims[3] = { H, W, C };
    float q[4]    = { in_sc, (float)in_zp, out_sc, (float)out_zp };

    NSUInteger off_in  = (NSUInteger)((const uint8_t*)input  - (const uint8_t*)[buf_in  contents]);
    NSUInteger off_out = (NSUInteger)((const uint8_t*)output - (const uint8_t*)[buf_out contents]);

    MetalDispatch d("avgpool_int8_nhwc_k");
    d.buf(buf_in,  off_in);
    d.buf(buf_out, off_out);
    d.bytes(dims, sizeof(dims));
    d.bytes(q,    sizeof(q));
    d.run((NSUInteger)C);
}

// ── Avg pool global INT8 NCHW ─────────────────────────────────────
void metal_avgpool_global_int8(
    const int8_t* input,
    float in_sc, int in_zp, float out_sc, int out_zp,
    int8_t* output,
    int C, int H, int W,
    StreamHandle)
{
    id<MTLBuffer> buf_in  = metal_get_buf(input);
    id<MTLBuffer> buf_out = metal_get_buf(output);
    if (!buf_in || !buf_out) {
        fprintf(stderr, "[metal] avgpool_global_int8_nchw: buffer lookup failed — aborting\n");
        abort();
    }

    int   dims[3] = { H, W, C };
    float q[4]    = { in_sc, (float)in_zp, out_sc, (float)out_zp };

    NSUInteger off_in  = (NSUInteger)((const uint8_t*)input  - (const uint8_t*)[buf_in  contents]);
    NSUInteger off_out = (NSUInteger)((const uint8_t*)output - (const uint8_t*)[buf_out contents]);

    MetalDispatch d("avgpool_int8_nchw_k");
    d.buf(buf_in,  off_in);
    d.buf(buf_out, off_out);
    d.bytes(dims, sizeof(dims));
    d.bytes(q,    sizeof(q));
    d.run((NSUInteger)C);
}
