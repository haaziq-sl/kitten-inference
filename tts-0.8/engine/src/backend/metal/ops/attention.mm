// src/backend/metal/ops/attention.mm
// Attention ops for the Metal backend.
//
// attention_flash_fp32: full GPU implementation
//   Step 1: QKV projection via MPS (input[N,D] × w_qkv^T[3D,D] + b_qkv)
//   Step 2: per-head Q@K^T (MPS strided), softmax (custom kernel), @V (MPS strided)
//   Step 3: output projection via MPS (output[N,D] × w_proj^T[D,D] + b_proj)
//
// attention_flash_int8: CPU fallback (dynamic INT8 quant makes GPU less beneficial).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "metal_ops.hpp"
#include "backend/metal/metal_helpers.h"
#include "backend/cpu/ops_neon.hpp"
#include <cstdio>
#include <cstdlib>

#include <cmath>
#include <cstring>

// ── Growing scratch buffer for attention scores ───────────────────
// Separate from g_col_buf to avoid conflicts with conv ops.
static id<MTLBuffer> g_attn_s_buf       = nil;
static size_t        g_attn_s_buf_bytes = 0;

static id<MTLBuffer> attn_get_s_buf(size_t needed) {
    if (needed > g_attn_s_buf_bytes) {
        g_attn_s_buf = [g_metal_device newBufferWithLength:needed
                                                   options:MTLResourceStorageModeShared];
        g_attn_s_buf_bytes = g_attn_s_buf ? needed : 0;
    }
    return g_attn_s_buf;
}

// ── Helper: MPS matmul with arbitrary row strides ─────────────────
// C[M, N] = A[M, K] (rowbytes=A_rb) × op(B[...]) + 0
// transposeRight=YES: B stored as [N, K] row-major.
// transposeRight=NO : B stored as [K, N] row-major.
static void mps_matmul_strided(
    id<MTLBuffer> A_buf, NSUInteger A_off, NSUInteger A_rb,
    id<MTLBuffer> B_buf, NSUInteger B_off, NSUInteger B_rb, int B_rows, int B_cols,
    id<MTLBuffer> C_buf, NSUInteger C_off, NSUInteger C_rb,
    int M, int K, int N,
    BOOL transposeRight,
    double alpha)
{
    if (M == 0 || K == 0 || N == 0) return;

    MPSMatrixDescriptor* dA = [MPSMatrixDescriptor
        matrixDescriptorWithRows:(NSUInteger)M columns:(NSUInteger)K rowBytes:A_rb dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* dB = [MPSMatrixDescriptor
        matrixDescriptorWithRows:(NSUInteger)B_rows columns:(NSUInteger)B_cols rowBytes:B_rb dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* dC = [MPSMatrixDescriptor
        matrixDescriptorWithRows:(NSUInteger)M columns:(NSUInteger)N rowBytes:C_rb dataType:MPSDataTypeFloat32];

    MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:A_buf offset:A_off descriptor:dA];
    MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:B_buf offset:B_off descriptor:dB];
    MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:C_buf offset:C_off descriptor:dC];

    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:g_metal_device
         transposeLeft:NO
        transposeRight:transposeRight
           resultRows:(NSUInteger)M
        resultColumns:(NSUInteger)N
      interiorColumns:(NSUInteger)K
                 alpha:alpha
                  beta:0.0];

    id<MTLCommandBuffer> cb = metal_cmd_buf();
    [mm encodeToCommandBuffer:cb leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    // No commit/wait — batched with surrounding ops; caller calls metal_flush().
}

// ── Flash Attention FP32 — GPU ────────────────────────────────────
void metal_attention_flash_fp32(
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
    StreamHandle,
    bool         dq_attn_out)
{
    if (dq_attn_out) {
        fprintf(stderr, "[metal] attention_flash_fp32: dq_attn_out not supported on Metal — aborting\n");
        abort();
    }
    id<MTLBuffer> buf_in   = metal_get_buf(input);
    id<MTLBuffer> buf_out  = metal_get_buf(output);
    id<MTLBuffer> buf_qkv  = metal_get_buf(scratch_qkv);
    id<MTLBuffer> B_qkv    = metal_get_weight_buf(w_qkv_packed);
    id<MTLBuffer> B_proj   = metal_get_weight_buf(w_proj_packed);

    if (!buf_in || !buf_out || !buf_qkv || !B_qkv || !B_proj) {
        fprintf(stderr, "[metal] attention_flash_fp32: buffer lookup failed — aborting\n");
        abort();
    }

    NSUInteger in_off  = (NSUInteger)((const uint8_t*)input       - (const uint8_t*)[buf_in  contents]);
    NSUInteger out_off = (NSUInteger)((const uint8_t*)output      - (const uint8_t*)[buf_out contents]);
    NSUInteger qkv_off = (NSUInteger)((const uint8_t*)scratch_qkv - (const uint8_t*)[buf_qkv contents]);

    // ── Step 1: QKV projection ────────────────────────────────────
    // scratch_qkv[N, 3D] = input[N, D] × w_qkv^T[3D, D]
    metal_mps_matmul(buf_in, in_off, B_qkv, 0, buf_qkv, qkv_off, N, D, 3 * D);

    // Add b_qkv bias (broadcast over rows).
    if (b_qkv) {
        id<MTLBuffer> bias_buf = metal_get_or_create_cpu_buf(b_qkv, (size_t)(3 * D) * sizeof(float));
        if (bias_buf) {
            int tc[2] = { N, 3 * D };
            MetalDispatch d("seqgemm_bias_k");
            d.buf(buf_qkv, qkv_off);
            d.buf(bias_buf);
            d.bytes(tc, sizeof(tc));
            d.run((NSUInteger)(N * 3 * D));
        }
    }

    // ── Step 2: per-head Q@K^T + softmax + @V ────────────────────
    // No memset needed: mps_matmul_strided uses beta=0 (pure overwrite)
    // and each head writes to a non-overlapping column slice of output.

    const int   d_k   = D / num_heads;
    const float scale = 1.f / sqrtf((float)d_k);

    // MPS requires rowBytes to be a multiple of 16 bytes.
    // Round N*sizeof(float) up to the nearest 16-byte boundary for S_rb.
    NSUInteger S_rb_raw = (NSUInteger)(N * sizeof(float));
    NSUInteger S_rb     = (S_rb_raw + 15) & ~(NSUInteger)15;  // round up to 16 bytes

    // Scratch for attention scores S[N, N] using the padded row stride.
    id<MTLBuffer> S_buf = attn_get_s_buf((size_t)N * S_rb);
    if (!S_buf) {
        fprintf(stderr, "[metal] attention_flash_fp32: S_buf alloc failed — aborting\n");
        abort();
    }

    NSUInteger qkv_rb = (NSUInteger)(3 * D * sizeof(float));  // stride between QKV rows
    NSUInteger out_rb = (NSUInteger)(D * sizeof(float));       // stride between output rows

    for (int h = 0; h < num_heads; ++h) {
        // Offsets into scratch_qkv[N, 3D]:
        //   Q_h starts at column h*d_k            → offset = qkv_off + h*d_k*sizeof(float)
        //   K_h starts at column D + h*d_k        → offset = qkv_off + (D + h*d_k)*sizeof(float)
        //   V_h starts at column 2*D + h*d_k      → offset = qkv_off + (2*D + h*d_k)*sizeof(float)
        NSUInteger Q_off = qkv_off + (NSUInteger)(h * d_k) * sizeof(float);
        NSUInteger K_off = qkv_off + (NSUInteger)(D + h * d_k) * sizeof(float);
        NSUInteger V_off = qkv_off + (NSUInteger)(2 * D + h * d_k) * sizeof(float);
        NSUInteger O_off = out_off + (NSUInteger)(h * d_k) * sizeof(float);

        // S[N, N] = scale * Q_h[N, d_k] × K_h^T[d_k, N]
        // Q_h and K_h are [N, d_k] with rowBytes = 3D*sizeof(float).
        // transposeRight=YES: MPS reads K_h as [N, d_k] and transposes → [d_k, N].
        mps_matmul_strided(
            buf_qkv, Q_off, qkv_rb,
            buf_qkv, K_off, qkv_rb, N, d_k,
            S_buf, 0, S_rb,
            N, d_k, N, YES, (double)scale);

        // Softmax over each row of S[N, N] (in-place).
        // dims: [N_cols, N_rows, row_stride_in_floats, 0]
        // S_rb may be padded to 16-byte MPS alignment, so stride != N.
        {
            int S_stride = (int)(S_rb / sizeof(float));
            int dims[4] = { N, N, S_stride, 0 };
            MetalDispatch d("softmax_rows_k");
            d.buf(S_buf);
            d.bytes(dims, sizeof(dims));
            d.tg_mem(256 * sizeof(float));
            d.run_threadgroups(MTLSizeMake((NSUInteger)N, 1, 1), MTLSizeMake(256, 1, 1));
        }

        // O_h[N, d_k] = S[N, N] × V_h[N, d_k]
        // V_h is [N, d_k] with rowBytes = 3D*sizeof(float); transposeRight=NO.
        // O_h is [N, d_k] with rowBytes = D*sizeof(float) (strided into output).
        mps_matmul_strided(
            S_buf, 0, S_rb,
            buf_qkv, V_off, qkv_rb, N, d_k,
            buf_out, O_off, out_rb,
            N, N, d_k, NO, 1.0);
    }

    // ── Step 3: output projection ─────────────────────────────────
    // tmp[N, D] (stored in scratch_qkv[0..N*D-1]) = output[N, D] × w_proj^T[D, D]
    NSUInteger proj_rb = (NSUInteger)(D * sizeof(float));

    mps_matmul_strided(
        buf_out, out_off, out_rb,
        B_proj, 0, proj_rb, D, D,
        buf_qkv, qkv_off, proj_rb,
        N, D, D, YES, 1.0);

    // GPU blit: copy tmp (buf_qkv) → output (buf_out).
    // A MTLBlitCommandEncoder serializes with the preceding MPS op within the
    // same command buffer — no flush needed.
    {
        metal_end_lm_encoder();
        id<MTLCommandBuffer>      cb_blit = metal_cmd_buf();
        id<MTLBlitCommandEncoder> blit    = [cb_blit blitCommandEncoder];
        [blit copyFromBuffer:buf_qkv sourceOffset:qkv_off
                   toBuffer:buf_out  destinationOffset:out_off
                      size:(NSUInteger)(N * D * sizeof(float))];
        [blit endEncoding];
    }

    // Add b_proj bias to output (batched, no flush).
    if (b_proj) {
        id<MTLBuffer> bproj_buf = metal_get_or_create_cpu_buf(b_proj, (size_t)D * sizeof(float));
        if (bproj_buf) {
            int tc[2] = { N, D };
            MetalDispatch d("seqgemm_bias_k");
            d.buf(buf_out, out_off);
            d.buf(bproj_buf);
            d.bytes(tc, sizeof(tc));
            d.run((NSUInteger)(N * D));
        }
    }
    // All ops stay batched in g_active_cb; caller flushes at CPU boundary.
}

// ── Flash Attention INT8 — CPU fallback ──────────────────────────
void metal_attention_flash_int8(
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
    StreamHandle)
{
    metal_flush();  // flush pending GPU writes to input before CPU reads it
    attention_flash_int8(input,
                         w_qkv_packed, w_qkv_scales, eff_zeros_qkv, b_qkv,
                         w_proj_packed, w_proj_scales, eff_zeros_proj, b_proj,
                         output, N, D, num_heads,
                         scratch_qkv, scratch_int8, scratch_req, nullptr);
}
