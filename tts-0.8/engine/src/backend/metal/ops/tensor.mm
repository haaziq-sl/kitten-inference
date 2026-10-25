// src/backend/metal/ops/tensor.mm
// Tensor utility ops for the Metal backend.
// All ops use Metal compute kernels where possible.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_ops.hpp"
#include "backend/metal/metal_helpers.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

// ─────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────

// Create a temporary shared MTLBuffer populated with CPU data.
static id<MTLBuffer> tmp_buf(const void* data, size_t bytes) {
    return [g_metal_device newBufferWithBytes:data
                                      length:bytes
                                     options:MTLResourceStorageModeShared];
}

static void metal_abort(const char* op, const void* p1, const void* p2 = nullptr) {
    if (p2)
        fprintf(stderr, "[metal] %s: buffer lookup failed (a=%p b=%p) — aborting\n", op, p1, p2);
    else
        fprintf(stderr, "[metal] %s: buffer lookup failed (%p) — aborting\n", op, p1);
    abort();
}

// ─────────────────────────────────────────────────────────────────
// Add FP32 (element-wise, optional ReLU)
// ─────────────────────────────────────────────────────────────────
void metal_add_fp32(const float* a, const float* b, float* c, int N, bool relu, StreamHandle) {
    id<MTLBuffer> buf_a = metal_get_buf(a);
    id<MTLBuffer> buf_b = metal_get_buf(b);
    id<MTLBuffer> buf_c = metal_get_buf(c);
    if (!buf_a || !buf_b || !buf_c) metal_abort("add_fp32", a, b);

    int do_relu = relu ? 1 : 0;
    NSUInteger off_a = (NSUInteger)((const uint8_t*)a - (const uint8_t*)[buf_a contents]);
    NSUInteger off_b = (NSUInteger)((const uint8_t*)b - (const uint8_t*)[buf_b contents]);
    NSUInteger off_c = (NSUInteger)((const uint8_t*)c - (const uint8_t*)[buf_c contents]);

    MetalDispatch d("add_fp32_k");
    d.buf(buf_a, off_a);
    d.buf(buf_b, off_b);
    d.buf(buf_c, off_c);
    d.bytes(&do_relu, sizeof(int));
    d.run((NSUInteger)N);
}

// ─────────────────────────────────────────────────────────────────
// Add + requantize INT8
// ─────────────────────────────────────────────────────────────────
void metal_add_requant_int8(
    const int8_t* a, const int8_t* b,
    float in1_sc, int in1_zp, float in2_sc, int in2_zp,
    float out_sc, int out_zp,
    int8_t* c, int N,
    StreamHandle)
{
    id<MTLBuffer> buf_a = metal_get_buf(a);
    id<MTLBuffer> buf_b = metal_get_buf(b);
    id<MTLBuffer> buf_c = metal_get_buf(c);
    if (!buf_a || !buf_b || !buf_c) metal_abort("add_requant_int8", a, b);

    float m1 = in1_sc / out_sc;
    float m2 = in2_sc / out_sc;
    if (std::abs(m1) > 7.99f || std::abs(m2) > 7.99f) {
        fprintf(stderr, "[metal] add_requant_int8: multiplier ratio out of range (m1=%f m2=%f) — aborting\n", m1, m2);
        abort();
    }
    int32_t M1 = (int32_t)std::roundf(m1 * 4096.0f);
    int32_t M2 = (int32_t)std::roundf(m2 * 4096.0f);
    int32_t pre_bias = (int32_t)out_zp * 4096
                     - (int32_t)in1_zp * M1
                     - (int32_t)in2_zp * M2;
    int q[4] = { M1, M2, pre_bias, 0 };

    NSUInteger off_a = (NSUInteger)((const uint8_t*)a - (const uint8_t*)[buf_a contents]);
    NSUInteger off_b = (NSUInteger)((const uint8_t*)b - (const uint8_t*)[buf_b contents]);
    NSUInteger off_c = (NSUInteger)((const uint8_t*)c - (const uint8_t*)[buf_c contents]);

    MetalDispatch d("add_requant_int8_k");
    d.buf(buf_a, off_a);
    d.buf(buf_b, off_b);
    d.buf(buf_c, off_c);
    d.bytes(q, sizeof(q));
    d.run((NSUInteger)N);
}

// ─────────────────────────────────────────────────────────────────
// Add vectors (dst[i] += src[i])
// ─────────────────────────────────────────────────────────────────
void metal_add_vectors_fp32(float* dst, const float* src, int N, StreamHandle) {
    id<MTLBuffer> buf_dst = metal_get_buf(dst);
    id<MTLBuffer> buf_src = metal_get_buf(src);
    if (!buf_dst || !buf_src) metal_abort("add_vectors_fp32", dst, src);

    NSUInteger off_dst = (NSUInteger)((const uint8_t*)dst - (const uint8_t*)[buf_dst contents]);
    NSUInteger off_src = (NSUInteger)((const uint8_t*)src - (const uint8_t*)[buf_src contents]);

    MetalDispatch d("add_vectors_k");
    d.buf(buf_dst, off_dst);
    d.buf(buf_src, off_src);
    d.run((NSUInteger)N);
}

// ─────────────────────────────────────────────────────────────────
// Bias add: out[t*C + c] += bias[c]   for all t in [0,T), c in [0,C)
// bias may be a plain CPU allocation (weight tensors); upload as tmp MTLBuffer.
// ─────────────────────────────────────────────────────────────────
void metal_bias_add_rows_fp32(float* out, const float* bias, int T, int C, StreamHandle) {
    id<MTLBuffer> buf_out  = metal_get_buf(out);
    if (!buf_out) metal_abort("bias_add_rows_fp32", out);

    id<MTLBuffer> buf_bias = metal_get_buf(bias);
    id<MTLBuffer> tmp_bias_buf = nil;
    if (!buf_bias) {
        // bias is a plain CPU weight allocation — upload as temporary Metal buffer.
        tmp_bias_buf = tmp_buf(bias, (size_t)(C * sizeof(float)));
        buf_bias = tmp_bias_buf;
    }

    NSUInteger off_out  = (NSUInteger)((const uint8_t*)out - (const uint8_t*)[buf_out contents]);
    NSUInteger off_bias = (buf_bias == tmp_bias_buf) ? 0
                        : (NSUInteger)((const uint8_t*)bias - (const uint8_t*)[buf_bias contents]);

    MetalDispatch d("bias_add_flat_k");
    d.buf(buf_out,  off_out);
    d.buf(buf_bias, off_bias);
    d.bytes(&C, sizeof(int));
    d.run((NSUInteger)(T * C));
}

// ─────────────────────────────────────────────────────────────────
// Scale: out[i] = in[i] * scalar
// ─────────────────────────────────────────────────────────────────
void metal_scale_fp32(const float* in, float* out, size_t N, float scalar, StreamHandle) {
    id<MTLBuffer> buf_in  = metal_get_buf(in);
    id<MTLBuffer> buf_out = metal_get_buf(out);
    if (!buf_in || !buf_out) metal_abort("scale_fp32", in, out);

    NSUInteger off_in  = (NSUInteger)((const uint8_t*)in  - (const uint8_t*)[buf_in  contents]);
    NSUInteger off_out = (NSUInteger)((const uint8_t*)out - (const uint8_t*)[buf_out contents]);

    MetalDispatch d("scale_k");
    d.buf(buf_in,  off_in);
    d.buf(buf_out, off_out);
    d.bytes(&scalar, sizeof(float));
    d.run((NSUInteger)N);
}

// ─────────────────────────────────────────────────────────────────
// Sum channels: out[t] = sum_c in[t*C + c]
// ─────────────────────────────────────────────────────────────────
void metal_sum_channels_fp32(const float* in, float* out, int T, int C, StreamHandle) {
    id<MTLBuffer> buf_in  = metal_get_buf(in);
    id<MTLBuffer> buf_out = metal_get_buf(out);
    if (!buf_in || !buf_out) metal_abort("sum_channels_fp32", in, out);

    NSUInteger off_in  = (NSUInteger)((const uint8_t*)in  - (const uint8_t*)[buf_in  contents]);
    NSUInteger off_out = (NSUInteger)((const uint8_t*)out - (const uint8_t*)[buf_out contents]);

    MetalDispatch d("sum_channels_k");
    d.buf(buf_in,  off_in);
    d.buf(buf_out, off_out);
    d.bytes(&C, sizeof(int));
    d.run((NSUInteger)T);
}

// ─────────────────────────────────────────────────────────────────
// ReLU inplace (alias for activation)
// ─────────────────────────────────────────────────────────────────
void metal_relu_inplace_fp32(float* inout, int N, StreamHandle stream) {
    metal_relu_fp32(inout, N, stream);
}

// ─────────────────────────────────────────────────────────────────
// Embedding lookup: out[t*D + d] = weight[int(ids[t]) * D + d]
// ─────────────────────────────────────────────────────────────────
void metal_embedding_lookup_fp32(
    const float* ids, const float* weight, float* out,
    int T, int num_emb, int D, StreamHandle)
{
    id<MTLBuffer> buf_ids = metal_get_buf(ids);
    id<MTLBuffer> buf_wt  = metal_get_buf(weight);
    id<MTLBuffer> buf_out = metal_get_buf(out);
    if (!buf_ids || !buf_wt || !buf_out) metal_abort("embedding_lookup_fp32", ids, out);

    int p[2] = { T, D };
    NSUInteger off_ids = (NSUInteger)((const uint8_t*)ids    - (const uint8_t*)[buf_ids contents]);
    NSUInteger off_wt  = (NSUInteger)((const uint8_t*)weight - (const uint8_t*)[buf_wt  contents]);
    NSUInteger off_out = (NSUInteger)((const uint8_t*)out    - (const uint8_t*)[buf_out contents]);

    MetalDispatch d("embedding_lookup_k");
    d.buf(buf_ids, off_ids);
    d.buf(buf_wt,  off_wt);
    d.buf(buf_out, off_out);
    d.bytes(p, sizeof(p));
    d.run((NSUInteger)(T * D));
}

// ─────────────────────────────────────────────────────────────────
// Concat 1D: out[t*(C1+C2) + c] = a or b
// ─────────────────────────────────────────────────────────────────
void metal_concat1d_fp32(
    const float* a, const float* b, float* out,
    int T, int C1, int C2, bool bcast, StreamHandle)
{
    id<MTLBuffer> buf_a   = metal_get_buf(a);
    id<MTLBuffer> buf_b   = metal_get_buf(b);
    id<MTLBuffer> buf_out = metal_get_buf(out);
    if (!buf_a || !buf_b || !buf_out) metal_abort("concat1d_fp32", a, b);

    int p[4] = { T, C1, C2, bcast ? 1 : 0 };
    NSUInteger off_a   = (NSUInteger)((const uint8_t*)a   - (const uint8_t*)[buf_a   contents]);
    NSUInteger off_b   = (NSUInteger)((const uint8_t*)b   - (const uint8_t*)[buf_b   contents]);
    NSUInteger off_out = (NSUInteger)((const uint8_t*)out - (const uint8_t*)[buf_out contents]);

    MetalDispatch d("concat1d_k");
    d.buf(buf_a,   off_a);
    d.buf(buf_b,   off_b);
    d.buf(buf_out, off_out);
    d.bytes(p, sizeof(p));
    d.run((NSUInteger)(T * (C1 + C2)));
}

// ─────────────────────────────────────────────────────────────────
// Slice channels: out[t*C_out + c] = in[t*C_in + ch_start + c]
// ─────────────────────────────────────────────────────────────────
void metal_slice_channels_fp32(
    const float* in, float* out,
    int T, int C_in, int ch_start, int C_out, StreamHandle)
{
    id<MTLBuffer> buf_in  = metal_get_buf(in);
    id<MTLBuffer> buf_out = metal_get_buf(out);
    if (!buf_in || !buf_out) metal_abort("slice_channels_fp32", in, out);

    int p[4] = { T, C_in, ch_start, C_out };
    NSUInteger off_in  = (NSUInteger)((const uint8_t*)in  - (const uint8_t*)[buf_in  contents]);
    NSUInteger off_out = (NSUInteger)((const uint8_t*)out - (const uint8_t*)[buf_out contents]);

    MetalDispatch d("slice_channels_k");
    d.buf(buf_in,  off_in);
    d.buf(buf_out, off_out);
    d.bytes(p, sizeof(p));
    d.run((NSUInteger)(T * C_out));
}

// ─────────────────────────────────────────────────────────────────
// Upsample nearest 1D
// ─────────────────────────────────────────────────────────────────
void metal_upsample_nearest1d_fp32(
    const float* in, float* out, int T, int C, int sf, StreamHandle)
{
    id<MTLBuffer> buf_in  = metal_get_buf(in);
    id<MTLBuffer> buf_out = metal_get_buf(out);
    if (!buf_in || !buf_out) metal_abort("upsample_nearest1d_fp32", in, out);

    int T_out = T * sf;
    int p[3]  = { T_out, C, sf };
    NSUInteger off_in  = (NSUInteger)((const uint8_t*)in  - (const uint8_t*)[buf_in  contents]);
    NSUInteger off_out = (NSUInteger)((const uint8_t*)out - (const uint8_t*)[buf_out contents]);

    MetalDispatch d("upsample_nearest1d_k");
    d.buf(buf_in,  off_in);
    d.buf(buf_out, off_out);
    d.bytes(p, sizeof(p));
    d.run((NSUInteger)(T_out * C));
}

// ─────────────────────────────────────────────────────────────────
// Upsample bilinear 2D  (NHWC, N=1 implicit: input is [H,W,C])
// ─────────────────────────────────────────────────────────────────
void metal_upsample_bilinear2d_fp32(
    const float* in, float* out,
    int C, int iH, int iW, int oH, int oW, StreamHandle)
{
    id<MTLBuffer> buf_in  = metal_get_buf(in);
    id<MTLBuffer> buf_out = metal_get_buf(out);
    if (!buf_in || !buf_out) metal_abort("upsample_bilinear2d_fp32", in, out);

    int in_dims[4] = { C, iH, iW, oH };
    NSUInteger off_in  = (NSUInteger)((const uint8_t*)in  - (const uint8_t*)[buf_in  contents]);
    NSUInteger off_out = (NSUInteger)((const uint8_t*)out - (const uint8_t*)[buf_out contents]);

    MetalDispatch d("upsample_bilinear2d_k");
    d.buf(buf_in,  off_in);
    d.buf(buf_out, off_out);
    d.bytes(in_dims, sizeof(in_dims));
    d.bytes(&oW, sizeof(int));
    d.run((NSUInteger)(oH * oW * C));
}

// ─────────────────────────────────────────────────────────────────
// Snake1D: out[i] = in[i] + sin(alpha[c]*in[i])^2 * inv_alpha[c]
// alpha/inv_alpha may be plain CPU weight allocations; upload as tmp MTLBuffers.
// ─────────────────────────────────────────────────────────────────
void metal_snake1d_fp32(
    const float* in, float* out,
    const float* alpha, const float* inv_alpha,
    int T, int C, StreamHandle)
{
    id<MTLBuffer> buf_in  = metal_get_buf(in);
    id<MTLBuffer> buf_out = metal_get_buf(out);
    if (!buf_in || !buf_out) metal_abort("snake1d_fp32", in, out);

    id<MTLBuffer> buf_al  = metal_get_buf(alpha);
    id<MTLBuffer> buf_ial = metal_get_buf(inv_alpha);
    id<MTLBuffer> tmp_al = nil, tmp_ial = nil;
    if (!buf_al) {
        tmp_al = tmp_buf(alpha, (size_t)(C * sizeof(float)));
        buf_al = tmp_al;
    }
    if (!buf_ial) {
        tmp_ial = tmp_buf(inv_alpha, (size_t)(C * sizeof(float)));
        buf_ial = tmp_ial;
    }

    NSUInteger off_in  = (NSUInteger)((const uint8_t*)in  - (const uint8_t*)[buf_in  contents]);
    NSUInteger off_out = (NSUInteger)((const uint8_t*)out - (const uint8_t*)[buf_out contents]);
    NSUInteger off_al  = (buf_al  == tmp_al ) ? 0 : (NSUInteger)((const uint8_t*)alpha     - (const uint8_t*)[buf_al  contents]);
    NSUInteger off_ial = (buf_ial == tmp_ial) ? 0 : (NSUInteger)((const uint8_t*)inv_alpha - (const uint8_t*)[buf_ial contents]);

    MetalDispatch d("snake1d_k");
    d.buf(buf_in,  off_in);
    d.buf(buf_out, off_out);
    d.buf(buf_al,  off_al);
    d.buf(buf_ial, off_ial);
    d.bytes(&C, sizeof(int));
    d.run((NSUInteger)(T * C));
}
