// src/backend/metal/ops/activations.mm
// Elementwise activation ops using Metal compute kernels.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_ops.hpp"
#include "backend/metal/metal_helpers.h"
#include <cstdio>
#include <cstdlib>

// Helper: dispatch an in-place kernel on a single buffer.
static inline void dispatch_inplace(const char* kernel, float* inout, int N) {
    id<MTLBuffer> buf = metal_get_buf(inout);
    if (!buf) {
        fprintf(stderr, "[metal] %s: buffer lookup failed for %p — aborting\n", kernel, (void*)inout);
        abort();
    }
    NSUInteger off = (NSUInteger)((const uint8_t*)inout - (const uint8_t*)[buf contents]);
    MetalDispatch d(kernel);
    d.buf(buf, off);
    d.run((NSUInteger)N);
    // No flush — ops accumulate in g_active_cb; copy_d2h flushes the whole pass.
}

static inline void dispatch_binary(const char* kernel,
                                   const float* in, float* out, int N) {
    id<MTLBuffer> buf_in  = metal_get_buf(in);
    id<MTLBuffer> buf_out = metal_get_buf(out);
    if (!buf_in || !buf_out) {
        fprintf(stderr, "[metal] %s: buffer lookup failed (in=%p out=%p) — aborting\n",
                kernel, (void*)in, (void*)out);
        abort();
    }
    NSUInteger off_in  = (NSUInteger)((const uint8_t*)in  - (const uint8_t*)[buf_in  contents]);
    NSUInteger off_out = (NSUInteger)((const uint8_t*)out - (const uint8_t*)[buf_out contents]);
    MetalDispatch d(kernel);
    d.buf(buf_in,  off_in);
    d.buf(buf_out, off_out);
    d.run((NSUInteger)N);
    // No flush — ops accumulate in g_active_cb; copy_d2h flushes the whole pass.
}

// ── ReLU (in-place) ───────────────────────────────────────────────
void metal_relu_fp32(float* inout, int N, StreamHandle) {
    dispatch_inplace("relu_inplace_k", inout, N);
}

// ── GELU (in-place) ───────────────────────────────────────────────
void metal_gelu_fp32(float* inout, int N, StreamHandle) {
    dispatch_inplace("gelu_inplace_k", inout, N);
}

// ── Sigmoid ───────────────────────────────────────────────────────
void metal_sigmoid_fp32(const float* in, float* out, int N, StreamHandle) {
    dispatch_binary("sigmoid_k", in, out, N);
}

// ── Tanh ─────────────────────────────────────────────────────────
// Uses scalar tanhf() on CPU via unified memory rather than the GPU tanh_k
// kernel.  Metal's GPU tanh() is a polynomial approximation that differs from
// CPU tanhf() by ~4 ULP; tanh in TTS models (gen_l_tanh) feeds directly
// into StftFp32 where atan2() amplifies tiny input differences into large phase
// errors for near-zero-magnitude bins.  metal_flush() commits pending GPU work
// so the shared-memory buffer is readable; no copy is needed (unified memory).
void metal_tanh_fp32(const float* in, float* out, int N, StreamHandle) {
    metal_flush();   // commit pending GPU work → buffer visible to CPU
    for (int i = 0; i < N; ++i) out[i] = tanhf(in[i]);
}

// ── Leaky ReLU ────────────────────────────────────────────────────
void metal_leaky_relu_fp32(const float* in, float* out, int N, float alpha, StreamHandle) {
    id<MTLBuffer> buf_in  = metal_get_buf(in);
    id<MTLBuffer> buf_out = metal_get_buf(out);
    if (!buf_in || !buf_out) {
        fprintf(stderr, "[metal] leaky_relu: buffer lookup failed (in=%p out=%p) — aborting\n",
                (void*)in, (void*)out);
        abort();
    }
    NSUInteger off_in  = (NSUInteger)((const uint8_t*)in  - (const uint8_t*)[buf_in  contents]);
    NSUInteger off_out = (NSUInteger)((const uint8_t*)out - (const uint8_t*)[buf_out contents]);

    MetalDispatch d("leaky_relu_k");
    d.buf(buf_in,  off_in);
    d.buf(buf_out, off_out);
    d.bytes(&alpha, sizeof(float));
    d.run((NSUInteger)N);
}

// ── Exp ───────────────────────────────────────────────────────────
void metal_exp_fp32(const float* in, float* out, int N, StreamHandle) {
    dispatch_binary("exp_k", in, out, N);
}

// ── Sin ───────────────────────────────────────────────────────────
void metal_sin_fp32(const float* in, float* out, int N, StreamHandle) {
    dispatch_binary("sin_k", in, out, N);
}

// ── SiLU: out[i] = in[i] * sigmoid(in[i]) ────────────────────────
void metal_silu_fp32(const float* in, float* out, int N, StreamHandle) {
    dispatch_binary("silu_k", in, out, N);
}

// ── Elementwise multiply: out[i] = a[i] * b[i] ───────────────────
void metal_elemwise_mul_fp32(const float* a, const float* b, float* out, int N, StreamHandle) {
    id<MTLBuffer> buf_a   = metal_get_buf(a);
    id<MTLBuffer> buf_b   = metal_get_buf(b);
    id<MTLBuffer> buf_out = metal_get_buf(out);
    if (!buf_a || !buf_b || !buf_out) {
        fprintf(stderr, "[metal] elemwise_mul: buffer lookup failed (a=%p b=%p out=%p) — aborting\n",
                (void*)a, (void*)b, (void*)out);
        abort();
    }
    NSUInteger off_a   = (NSUInteger)((const uint8_t*)a   - (const uint8_t*)[buf_a   contents]);
    NSUInteger off_b   = (NSUInteger)((const uint8_t*)b   - (const uint8_t*)[buf_b   contents]);
    NSUInteger off_out = (NSUInteger)((const uint8_t*)out - (const uint8_t*)[buf_out contents]);
    MetalDispatch d("elemwise_mul_k");
    d.buf(buf_a,   off_a);
    d.buf(buf_b,   off_b);
    d.buf(buf_out, off_out);
    d.run((NSUInteger)N);
}
