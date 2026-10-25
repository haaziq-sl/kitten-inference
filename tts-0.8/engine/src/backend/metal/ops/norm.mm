// src/backend/metal/ops/norm.mm
// Normalization ops for the Metal backend.
// layernorm_fp32, ada_in1d_fp32, ada_layer_norm1d_fp32: all GPU.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_ops.hpp"
#include "backend/metal/metal_helpers.h"
#include "backend/cpu/ops_neon.hpp"
#include <vector>
#include <cstdio>
#include <cstdlib>

// ── Layer Norm ────────────────────────────────────────────────────
// Dispatches N threadgroups of 256 threads.
// Each threadgroup handles one row of [N, C].
void metal_layernorm_fp32(
    const float* input,
    const float* gamma,
    const float* beta,
    float*       output,
    int N, int C,
    float        eps,
    StreamHandle)
{
    id<MTLBuffer> buf_in    = metal_get_buf(input);
    id<MTLBuffer> buf_out   = metal_get_buf(output);
    id<MTLBuffer> buf_gamma = metal_get_buf(gamma);
    id<MTLBuffer> buf_beta  = metal_get_buf(beta);

    // Gamma and beta may be plain CPU pointers (e.g., loaded from weights
    // before Metal allocator took over). Create temporary shared buffers.
    id<MTLBuffer> tmp_gamma = nil;
    id<MTLBuffer> tmp_beta  = nil;

    if (!buf_gamma) {
        NSUInteger bytes = (NSUInteger)(C * sizeof(float));
        tmp_gamma = [g_metal_device newBufferWithBytes:gamma
                                               length:bytes
                                              options:MTLResourceStorageModeShared];
        buf_gamma = tmp_gamma;
    }
    if (!buf_beta) {
        NSUInteger bytes = (NSUInteger)(C * sizeof(float));
        tmp_beta = [g_metal_device newBufferWithBytes:beta
                                              length:bytes
                                             options:MTLResourceStorageModeShared];
        buf_beta = tmp_beta;
    }

    if (!buf_in || !buf_out || !buf_gamma || !buf_beta) {
        fprintf(stderr, "[metal] layernorm_fp32: buffer lookup failed — aborting\n");
        abort();
    }

    const NSUInteger TGS = 256;

    NSUInteger off_in    = (NSUInteger)((const uint8_t*)input  - (const uint8_t*)[buf_in    contents]);
    NSUInteger off_out   = (NSUInteger)((const uint8_t*)output - (const uint8_t*)[buf_out   contents]);
    NSUInteger off_gamma = buf_gamma == tmp_gamma ? 0
                         : (NSUInteger)((const uint8_t*)gamma - (const uint8_t*)[buf_gamma contents]);
    NSUInteger off_beta  = buf_beta  == tmp_beta  ? 0
                         : (NSUInteger)((const uint8_t*)beta  - (const uint8_t*)[buf_beta  contents]);

    id<MTLComputePipelineState> pso = metal_get_pso("layernorm_k");
    if (!pso) {
        fprintf(stderr, "[metal] layernorm_fp32: PSO not found — aborting\n");
        abort();
    }

    id<MTLCommandBuffer>        cb  = metal_cmd_buf();
    metal_end_lm_encoder();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];

    [enc setBuffer:buf_in    offset:off_in    atIndex:0];
    [enc setBuffer:buf_out   offset:off_out   atIndex:1];
    [enc setBuffer:buf_gamma offset:off_gamma atIndex:2];
    [enc setBuffer:buf_beta  offset:off_beta  atIndex:3];
    [enc setBytes:&C   length:sizeof(int)   atIndex:4];
    [enc setBytes:&eps length:sizeof(float) atIndex:5];

    // Threadgroup scratch: TGS floats
    [enc setThreadgroupMemoryLength:TGS * sizeof(float) atIndex:0];

    MTLSize tg_size  = MTLSizeMake(TGS, 1, 1);
    MTLSize tg_count = MTLSizeMake((NSUInteger)N, 1, 1);
    [enc dispatchThreadgroups:tg_count threadsPerThreadgroup:tg_size];
    [enc endEncoding];
}

// ── Helper: get Metal buffer + offset for a float pointer ────────
// Handles:
//  1. Exact base-pointer Metal buffers (e.g., gamma = gamma_beta_d_.ptr())
//  2. Interior Metal pointers (e.g., beta = gamma_beta_d_.ptr() + C)
//  3. Plain CPU pointers (creates a Metal buffer with a CPU data copy)
static id<MTLBuffer> norm_get_buf_and_off(const float* ptr, int count,
                                          NSUInteger* out_offset) {
    *out_offset = 0;
    if (!ptr || count == 0) return nil;
    // Case 1: exact base-pointer lookup
    id<MTLBuffer> buf = metal_get_buf(ptr);
    if (buf) {
        *out_offset = 0;  // ptr IS the base address
        return buf;
    }
    // Case 2: interior pointer (e.g., beta = gamma_beta + C)
    buf = metal_find_buf_containing(ptr, out_offset);
    if (buf) return buf;
    // Case 3: plain CPU pointer — create a Metal buffer from the CPU data
    *out_offset = 0;
    return metal_get_or_create_cpu_buf(ptr, (size_t)count * sizeof(float));
}

// ── Adaptive Instance Norm 1D — GPU ──────────────────────────────
// Input feat[T, C]; output out[T, C].
// One threadgroup per channel; threadgroup memory = 2*256 floats.
void metal_ada_in1d_fp32(
    const float* feat, float* out,
    const float* gamma, const float* beta,
    int T, int C, float eps, StreamHandle,
    const float* norm_weight, const float* norm_bias)
{
    id<MTLBuffer> buf_feat  = metal_get_buf(feat);
    id<MTLBuffer> buf_out   = metal_get_buf(out);
    NSUInteger off_feat  = (NSUInteger)((const uint8_t*)feat - (const uint8_t*)[buf_feat contents]);
    NSUInteger off_out   = (NSUInteger)((const uint8_t*)out  - (const uint8_t*)[buf_out  contents]);

    int dims[2] = { T, C };
    const NSUInteger TGS = 256;

    if (norm_weight || norm_bias) {
        // Fold norm_weight/norm_bias entirely on the GPU to avoid reading the
        // GPU-computed gamma_beta_d_ on the CPU before the command buffer commits.
        // Use ada_in1d_fold_k which computes:
        //   eff_gamma = gamma_raw[c] * nw[c]
        //   eff_beta  = beta_raw[c]  * nw[c] + nb[c]
        NSUInteger off_gamma = 0, off_beta = 0;
        id<MTLBuffer> buf_gamma = norm_get_buf_and_off(gamma, C, &off_gamma);
        id<MTLBuffer> buf_beta  = norm_get_buf_and_off(beta,  C, &off_beta);

        // norm_weight and norm_bias are static CPU weights — get or create Metal buffers.
        // Fallback arrays for null norm_weight (ones) or null norm_bias (zeros).
        // Use [ptr + size_offset] trick so different C values get distinct cache keys
        // and avoid constant cache invalidation in metal_get_or_create_cpu_buf.
        static float  s_fallback_nw[2048];  // ones; index from end for unique-per-C keys
        static float  s_fallback_nb[2048];  // zeros (default-initialized)
        static bool   s_fallback_init = false;
        if (!s_fallback_init) {
            for (int i = 0; i < 2048; ++i) s_fallback_nw[i] = 1.f;
            // s_fallback_nb is zero-initialized by C++ static storage rules
            s_fallback_init = true;
        }
        // Use tail of s_fallback_nw/nb so pointer is unique per C (max C = 2048).
        const float* nw_ptr = norm_weight ? norm_weight
                                          : (s_fallback_nw + (2048 - C));
        const float* nb_ptr = norm_bias   ? norm_bias
                                          : (s_fallback_nb + (2048 - C));
        id<MTLBuffer> buf_nw = metal_get_or_create_cpu_buf(nw_ptr, (size_t)C * sizeof(float));
        id<MTLBuffer> buf_nb = metal_get_or_create_cpu_buf(nb_ptr, (size_t)C * sizeof(float));

        if (!buf_feat || !buf_out || !buf_gamma || !buf_beta || !buf_nw || !buf_nb) {
            fprintf(stderr, "[metal] ada_in1d_fp32 (fold): buffer lookup failed — aborting\n");
            abort();
        }

        id<MTLComputePipelineState> pso = metal_get_pso("ada_in1d_fold_k");
        if (!pso) {
            fprintf(stderr, "[metal] ada_in1d_fp32 (fold): PSO not found — aborting\n");
            abort();
        }

        id<MTLCommandBuffer>         cb  = metal_cmd_buf();
        metal_end_lm_encoder();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buf_feat  offset:off_feat  atIndex:0];
        [enc setBuffer:buf_out   offset:off_out   atIndex:1];
        [enc setBuffer:buf_gamma offset:off_gamma atIndex:2];
        [enc setBuffer:buf_beta  offset:off_beta  atIndex:3];
        [enc setBuffer:buf_nw    offset:0         atIndex:4];
        [enc setBuffer:buf_nb    offset:0         atIndex:5];
        [enc setBytes:dims    length:sizeof(dims)  atIndex:6];
        [enc setBytes:&eps    length:sizeof(float) atIndex:7];
        [enc setThreadgroupMemoryLength:TGS * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)C, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(TGS, 1, 1)];
        [enc endEncoding];
        return;
    }

    // No norm_weight/norm_bias: use the simpler ada_in1d_k path.
    NSUInteger off_gamma = 0, off_beta = 0;
    id<MTLBuffer> buf_gamma = norm_get_buf_and_off(gamma, C, &off_gamma);
    id<MTLBuffer> buf_beta  = norm_get_buf_and_off(beta,  C, &off_beta);

    if (!buf_feat || !buf_out || !buf_gamma || !buf_beta) {
        fprintf(stderr, "[metal] ada_in1d_fp32: buffer lookup failed — aborting\n");
        abort();
    }

    id<MTLComputePipelineState> pso = metal_get_pso("ada_in1d_k");
    if (!pso) {
        fprintf(stderr, "[metal] ada_in1d_fp32: PSO not found — aborting\n");
        abort();
    }

    id<MTLCommandBuffer>         cb  = metal_cmd_buf();
    metal_end_lm_encoder();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf_feat  offset:off_feat  atIndex:0];
    [enc setBuffer:buf_out   offset:off_out   atIndex:1];
    [enc setBuffer:buf_gamma offset:off_gamma atIndex:2];
    [enc setBuffer:buf_beta  offset:off_beta  atIndex:3];
    [enc setBytes:dims    length:sizeof(dims)  atIndex:4];
    [enc setBytes:&eps    length:sizeof(float) atIndex:5];
    [enc setThreadgroupMemoryLength:2 * TGS * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)C, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(TGS, 1, 1)];
    [enc endEncoding];
}

// ── Adaptive Layer Norm 1D — GPU ─────────────────────────────────
// Like layernorm but output = (1 + gamma[i]) * xnorm + beta[i].
void metal_ada_layer_norm1d_fp32(
    const float* feat, float* out,
    const float* gamma, const float* beta,
    int T, int C, float eps, StreamHandle)
{
    id<MTLBuffer> buf_feat  = metal_get_buf(feat);
    id<MTLBuffer> buf_out   = metal_get_buf(out);
    NSUInteger off_gamma = 0, off_beta = 0;
    id<MTLBuffer> buf_gamma = norm_get_buf_and_off(gamma, C, &off_gamma);
    id<MTLBuffer> buf_beta  = norm_get_buf_and_off(beta,  C, &off_beta);

    if (!buf_feat || !buf_out || !buf_gamma || !buf_beta) {
        fprintf(stderr, "[metal] ada_layer_norm1d_fp32: buffer lookup failed — aborting\n");
        abort();
    }

    id<MTLComputePipelineState> pso = metal_get_pso("ada_layernorm_k");
    if (!pso) {
        fprintf(stderr, "[metal] ada_layer_norm1d_fp32: PSO not found — aborting\n");
        abort();
    }

    NSUInteger off_feat  = (NSUInteger)((const uint8_t*)feat  - (const uint8_t*)[buf_feat  contents]);
    NSUInteger off_out   = (NSUInteger)((const uint8_t*)out   - (const uint8_t*)[buf_out   contents]);

    const NSUInteger TGS = 256;

    id<MTLCommandBuffer>         cb  = metal_cmd_buf();
    metal_end_lm_encoder();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf_feat  offset:off_feat  atIndex:0];
    [enc setBuffer:buf_out   offset:off_out   atIndex:1];
    [enc setBuffer:buf_gamma offset:off_gamma atIndex:2];
    [enc setBuffer:buf_beta  offset:off_beta  atIndex:3];
    [enc setBytes:&C   length:sizeof(int)   atIndex:4];
    [enc setBytes:&eps length:sizeof(float) atIndex:5];
    [enc setThreadgroupMemoryLength:TGS * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)T, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(TGS, 1, 1)];
    [enc endEncoding];
}
