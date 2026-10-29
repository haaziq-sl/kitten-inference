// src/backend/metal/ops/sequence.mm
// Metal GPU implementations of LSTM, STFT, ISTFT, ReflectionPad1d, SineGen.
// All ops batch into g_active_cb — no per-op flush.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_ops.hpp"
#include "backend/metal/metal_helpers.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ─────────────────────────────────────────────────────────────────
// Internal helper: dispatch a 2D grid kernel
// ─────────────────────────────────────────────────────────────────
static void dispatch2d(const char* kernel_name,
                       NSUInteger grid_x, NSUInteger grid_y,
                       NSUInteger tg_x,   NSUInteger tg_y,
                       void (^setup)(id<MTLComputeCommandEncoder>))
{
    id<MTLComputePipelineState> pso = metal_get_pso(kernel_name);
    if (!pso) {
        fprintf(stderr, "[metal] dispatch2d: PSO not found for '%s'\n", kernel_name);
        abort();
    }
    id<MTLCommandBuffer>         cb  = metal_cmd_buf();
    metal_end_lm_encoder();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    setup(enc);
    [enc dispatchThreads:MTLSizeMake(grid_x, grid_y, 1)
        threadsPerThreadgroup:MTLSizeMake(tg_x, tg_y, 1)];
    [enc endEncoding];
}

static void dispatch1d(const char* kernel_name,
                       NSUInteger total,
                       NSUInteger tg_size,
                       void (^setup)(id<MTLComputeCommandEncoder>))
{
    if (total == 0) return;
    id<MTLComputePipelineState> pso = metal_get_pso(kernel_name);
    if (!pso) {
        fprintf(stderr, "[metal] dispatch1d: PSO not found for '%s'\n", kernel_name);
        abort();
    }
    id<MTLCommandBuffer>         cb  = metal_cmd_buf();
    metal_end_lm_encoder();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    setup(enc);
    NSUInteger tg = (tg_size < total) ? tg_size : total;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
}

// ─────────────────────────────────────────────────────────────────
// Run one direction of LSTM scan.
//   proj_buf : [T, 4H] — pre-computed input projections
//   w_hh_buf : [4H, H]
//   b_hh_buf : [4H] or nil
//   h_all_buf: [T, H] — output hidden states for all time steps
//   reverse  : if true, scan from T-1 down to 0
// ─────────────────────────────────────────────────────────────────
static void run_lstm_scan(
    id<MTLBuffer> proj_buf,
    id<MTLBuffer> w_hh_buf,
    id<MTLBuffer> b_hh_buf,   // may be nil
    id<MTLBuffer> h_all_buf,  // [T, H] output
    int T, int H, bool reverse)
{
    @autoreleasepool {
        int fourH = 4 * H;
        size_t hc_bytes   = (size_t)H * sizeof(float);
        size_t gate_bytes = (size_t)fourH * sizeof(float);

        // Allocate h and c ping-pong buffers (zeroed initial state)
        id<MTLBuffer> h0 = [g_metal_device newBufferWithLength:hc_bytes
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> c0 = [g_metal_device newBufferWithLength:hc_bytes
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> h1 = [g_metal_device newBufferWithLength:hc_bytes
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> c1 = [g_metal_device newBufferWithLength:hc_bytes
                                                       options:MTLResourceStorageModeShared];
        memset([h0 contents], 0, hc_bytes);
        memset([c0 contents], 0, hc_bytes);

        // Shared gates buffer (written by lstm_gemv_k, read by lstm_update_k)
        id<MTLBuffer> gates_buf = [g_metal_device newBufferWithLength:gate_bytes
                                                              options:MTLResourceStorageModeShared];

        // Dummy zero bias for b_hh when null
        id<MTLBuffer> b_hh_dummy = nil;
        if (!b_hh_buf) {
            b_hh_dummy = [g_metal_device newBufferWithLength:gate_bytes
                                                     options:MTLResourceStorageModeShared];
            memset([b_hh_dummy contents], 0, gate_bytes);
            b_hh_buf = b_hh_dummy;
        }

        id<MTLComputePipelineState> pso_gemv   = metal_get_pso("lstm_gemv_k");
        id<MTLComputePipelineState> pso_update = metal_get_pso("lstm_update_k");
        if (!pso_gemv || !pso_update) {
            fprintf(stderr, "[metal] lstm_gemv_k / lstm_update_k PSO not found\n");
            abort();
        }

        // Threadgroup sizes: stay well below maxTotalThreadsPerThreadgroup
        // Use at most 256 threads regardless of H, so 4H=1024 is handled safely.
        NSUInteger tg_gemv   = MIN((NSUInteger)fourH, (NSUInteger)256);
        NSUInteger tg_update = MIN((NSUInteger)H,     (NSUInteger)256);

        // Flush the command buffer every 64 steps to avoid exceeding
        // Metal's command buffer encoder limit for long sequences.
        // This is necessary because each step uses 2 encoders (gemv + update),
        // and for T > 200 with 6 LSTM layers, the total encoder count can exceed
        // the Metal CB limit, causing undefined behavior (NaN / near-zero output).
        const int FLUSH_INTERVAL = 64;

        for (int step = 0; step < T; ++step) {
            if (step > 0 && step % FLUSH_INTERVAL == 0)
                metal_flush();
            int t = reverse ? (T - 1 - step) : step;

            NSUInteger proj_off = (NSUInteger)((size_t)t * fourH * sizeof(float));

            // ping-pong: step even → read h0/c0, write h1/c1
            id<MTLBuffer> h_in  = (step % 2 == 0) ? h0 : h1;
            id<MTLBuffer> c_in  = (step % 2 == 0) ? c0 : c1;
            id<MTLBuffer> h_out = (step % 2 == 0) ? h1 : h0;
            id<MTLBuffer> c_out = (step % 2 == 0) ? c1 : c0;

            // Pass 1: gates[4H] = proj_row + b_hh + w_hh * h_in
            {
                int dims2[2] = { H, 1 /* has_b_hh */ };
                id<MTLCommandBuffer>         cb  = metal_cmd_buf();
                metal_end_lm_encoder();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_gemv];
                [enc setBuffer:proj_buf  offset:proj_off atIndex:0];
                [enc setBuffer:w_hh_buf  offset:0        atIndex:1];
                [enc setBuffer:b_hh_buf  offset:0        atIndex:2];
                [enc setBuffer:h_in      offset:0        atIndex:3];
                [enc setBuffer:gates_buf offset:0        atIndex:4];
                [enc setBytes:dims2 length:sizeof(dims2) atIndex:5];
                [enc dispatchThreads:MTLSizeMake((NSUInteger)fourH, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(tg_gemv, 1, 1)];
                [enc endEncoding];
            }

            // Pass 2: apply gates, update h/c
            // Separate encoder ensures pass 1 is complete before pass 2 reads gates.
            {
                int dims2[2] = { H, t };
                id<MTLCommandBuffer>         cb  = metal_cmd_buf();
                metal_end_lm_encoder();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_update];
                [enc setBuffer:gates_buf offset:0        atIndex:0];
                [enc setBuffer:c_in      offset:0        atIndex:1];
                [enc setBuffer:h_out     offset:0        atIndex:2];
                [enc setBuffer:c_out     offset:0        atIndex:3];
                [enc setBuffer:h_all_buf offset:0        atIndex:4];
                [enc setBytes:dims2 length:sizeof(dims2) atIndex:5];
                [enc dispatchThreads:MTLSizeMake((NSUInteger)H, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(tg_update, 1, 1)];
                [enc endEncoding];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// metal_lstm_fp32
// ─────────────────────────────────────────────────────────────────
void metal_lstm_fp32(
    const float* input,
    const float* w_ih,   const float* b_ih,
    const float* w_hh,   const float* b_hh,
    const float* w_ih_r, const float* b_ih_r,
    const float* w_hh_r, const float* b_hh_r,
    float*       output,
    int T, int C_in, int H,
    bool bidirectional,
    StreamHandle /*stream*/)
{
    @autoreleasepool {
        id<MTLBuffer> buf_in  = metal_get_buf(input);
        id<MTLBuffer> buf_out = metal_get_buf(output);
        if (!buf_in || !buf_out) {
            fprintf(stderr, "[metal] lstm_fp32: input/output buffer lookup failed\n");
            abort();
        }

        // Get GPU buffers for weights (they were uploaded via upload_weights)
        id<MTLBuffer> buf_w_ih = metal_get_buf(w_ih);
        id<MTLBuffer> buf_w_hh = metal_get_buf(w_hh);
        id<MTLBuffer> buf_b_ih = b_ih ? metal_get_buf(b_ih) : nil;
        id<MTLBuffer> buf_b_hh = b_hh ? metal_get_buf(b_hh) : nil;

        if (!buf_w_ih || !buf_w_hh) {
            fprintf(stderr, "[metal] lstm_fp32: weight buffer lookup failed\n");
            abort();
        }

        int fourH = 4 * H;
        size_t proj_bytes = (size_t)T * fourH * sizeof(float);
        size_t h_bytes    = (size_t)T * H * sizeof(float);

        // Allocate projection and h_all buffers for forward direction
        id<MTLBuffer> proj_fwd = [g_metal_device newBufferWithLength:proj_bytes
                                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> h_fwd    = [g_metal_device newBufferWithLength:h_bytes
                                                             options:MTLResourceStorageModeShared];

        // Dummy zero bias for w_ih if b_ih is null
        id<MTLBuffer> b_ih_dummy = nil;
        id<MTLBuffer> buf_b_ih_used = buf_b_ih;
        if (!buf_b_ih_used) {
            b_ih_dummy = [g_metal_device newBufferWithLength:(size_t)fourH*sizeof(float)
                                                     options:MTLResourceStorageModeShared];
            memset([b_ih_dummy contents], 0, (size_t)fourH*sizeof(float));
        }

        // ── Forward input projection: proj_fwd[T, 4H] = input[T, C_in] × w_ih[4H, C_in]^T + b_ih ──
        {
            NSUInteger off_in  = (NSUInteger)((const uint8_t*)input  - (const uint8_t*)[buf_in  contents]);
            NSUInteger off_wih = (NSUInteger)((const uint8_t*)w_ih   - (const uint8_t*)[buf_w_ih contents]);
            int has_bias = buf_b_ih_used ? 1 : 0;
            int dims[4]  = { T, C_in, fourH, has_bias };

            id<MTLBuffer> b_ih_eff = buf_b_ih_used ? buf_b_ih_used : b_ih_dummy;
            NSUInteger off_bih = 0;
            if (buf_b_ih_used) {
                off_bih = (NSUInteger)((const uint8_t*)b_ih - (const uint8_t*)[buf_b_ih_used contents]);
            }

            id<MTLComputePipelineState> pso = metal_get_pso("lstm_proj_k");
            if (!pso) { fprintf(stderr, "[metal] lstm_proj_k PSO not found\n"); abort(); }

            id<MTLCommandBuffer>         cb  = metal_cmd_buf();
            metal_end_lm_encoder();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:buf_in   offset:off_in  atIndex:0];
            [enc setBuffer:buf_w_ih offset:off_wih atIndex:1];
            [enc setBuffer:b_ih_eff offset:off_bih atIndex:2];
            [enc setBuffer:proj_fwd offset:0        atIndex:3];
            [enc setBytes:dims length:sizeof(dims)  atIndex:4];
            NSUInteger tgx = MIN((NSUInteger)fourH, (NSUInteger)32);
            NSUInteger tgy = MIN((NSUInteger)T,     (NSUInteger)8);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)T, (NSUInteger)fourH, 1)
                threadsPerThreadgroup:MTLSizeMake(tgy, tgx, 1)];
            [enc endEncoding];
        }

        // ── Forward LSTM scan ──────────────────────────────────────────
        {
            NSUInteger off_whh = (NSUInteger)((const uint8_t*)w_hh - (const uint8_t*)[buf_w_hh contents]);
            id<MTLBuffer> w_hh_sub = [g_metal_device newBufferWithLength:[buf_w_hh length]
                                                                  options:MTLResourceStorageModeShared];
            memcpy([w_hh_sub contents], (const uint8_t*)[buf_w_hh contents] + off_whh,
                   (size_t)fourH * H * sizeof(float));
            id<MTLBuffer> b_hh_sub = nil;
            if (buf_b_hh) {
                NSUInteger off_bhh = (NSUInteger)((const uint8_t*)b_hh - (const uint8_t*)[buf_b_hh contents]);
                b_hh_sub = [g_metal_device newBufferWithLength:(size_t)fourH*sizeof(float)
                                                       options:MTLResourceStorageModeShared];
                memcpy([b_hh_sub contents], (const uint8_t*)[buf_b_hh contents] + off_bhh,
                       (size_t)fourH * sizeof(float));
            }
            run_lstm_scan(proj_fwd, w_hh_sub, b_hh_sub, h_fwd, T, H, /*reverse=*/false);
        }

        if (!bidirectional) {
            // Copy h_fwd → output
            NSUInteger off_out = (NSUInteger)((const uint8_t*)output - (const uint8_t*)[buf_out contents]);
            metal_end_lm_encoder();
            id<MTLBlitCommandEncoder> blit = [metal_cmd_buf() blitCommandEncoder];
            [blit copyFromBuffer:h_fwd  sourceOffset:0
                        toBuffer:buf_out destinationOffset:off_out
                            size:h_bytes];
            [blit endEncoding];
        } else {
            // ── Backward pass ──────────────────────────────────────────
            id<MTLBuffer> buf_w_ih_r = metal_get_buf(w_ih_r);
            id<MTLBuffer> buf_w_hh_r = metal_get_buf(w_hh_r);
            id<MTLBuffer> buf_b_ih_r = b_ih_r ? metal_get_buf(b_ih_r) : nil;
            id<MTLBuffer> buf_b_hh_r = b_hh_r ? metal_get_buf(b_hh_r) : nil;
            if (!buf_w_ih_r || !buf_w_hh_r) {
                fprintf(stderr, "[metal] lstm_fp32 bidir: reverse weight buffer lookup failed\n");
                abort();
            }

            id<MTLBuffer> proj_bwd = [g_metal_device newBufferWithLength:proj_bytes
                                                                 options:MTLResourceStorageModeShared];
            id<MTLBuffer> h_bwd    = [g_metal_device newBufferWithLength:h_bytes
                                                                 options:MTLResourceStorageModeShared];

            // Reverse input projection
            {
                NSUInteger off_in   = (NSUInteger)((const uint8_t*)input   - (const uint8_t*)[buf_in   contents]);
                NSUInteger off_wihr = (NSUInteger)((const uint8_t*)w_ih_r  - (const uint8_t*)[buf_w_ih_r contents]);
                int has_bias_r = buf_b_ih_r ? 1 : 0;
                int dims[4]    = { T, C_in, fourH, has_bias_r };

                id<MTLBuffer> b_ihr_dummy = nil;
                id<MTLBuffer> b_ihr_eff   = buf_b_ih_r;
                NSUInteger    off_bihr    = 0;
                if (buf_b_ih_r) {
                    off_bihr = (NSUInteger)((const uint8_t*)b_ih_r - (const uint8_t*)[buf_b_ih_r contents]);
                } else {
                    b_ihr_dummy = [g_metal_device newBufferWithLength:(size_t)fourH*sizeof(float)
                                                              options:MTLResourceStorageModeShared];
                    memset([b_ihr_dummy contents], 0, (size_t)fourH*sizeof(float));
                    b_ihr_eff = b_ihr_dummy;
                }

                id<MTLComputePipelineState> pso = metal_get_pso("lstm_proj_k");
                id<MTLCommandBuffer>         cb  = metal_cmd_buf();
                metal_end_lm_encoder();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:buf_in    offset:off_in   atIndex:0];
                [enc setBuffer:buf_w_ih_r offset:off_wihr atIndex:1];
                [enc setBuffer:b_ihr_eff  offset:off_bihr atIndex:2];
                [enc setBuffer:proj_bwd   offset:0        atIndex:3];
                [enc setBytes:dims length:sizeof(dims)    atIndex:4];
                NSUInteger tgx = MIN((NSUInteger)fourH, (NSUInteger)32);
                NSUInteger tgy = MIN((NSUInteger)T,     (NSUInteger)8);
                [enc dispatchThreads:MTLSizeMake((NSUInteger)T, (NSUInteger)fourH, 1)
                    threadsPerThreadgroup:MTLSizeMake(tgy, tgx, 1)];
                [enc endEncoding];
            }

            {
                NSUInteger off_whhr = (NSUInteger)((const uint8_t*)w_hh_r - (const uint8_t*)[buf_w_hh_r contents]);
                id<MTLBuffer> w_hhr_sub = [g_metal_device newBufferWithLength:(size_t)fourH*H*sizeof(float)
                                                                       options:MTLResourceStorageModeShared];
                memcpy([w_hhr_sub contents], (const uint8_t*)[buf_w_hh_r contents] + off_whhr,
                       (size_t)fourH * H * sizeof(float));
                id<MTLBuffer> b_hhr_sub = nil;
                if (buf_b_hh_r) {
                    NSUInteger off_bhhr = (NSUInteger)((const uint8_t*)b_hh_r - (const uint8_t*)[buf_b_hh_r contents]);
                    b_hhr_sub = [g_metal_device newBufferWithLength:(size_t)fourH*sizeof(float)
                                                            options:MTLResourceStorageModeShared];
                    memcpy([b_hhr_sub contents], (const uint8_t*)[buf_b_hh_r contents] + off_bhhr,
                           (size_t)fourH * sizeof(float));
                }
                run_lstm_scan(proj_bwd, w_hhr_sub, b_hhr_sub, h_bwd, T, H, /*reverse=*/true);
            }

            // Interleave fwd + bwd hidden states → output [T, 2H]
            {
                NSUInteger off_out = (NSUInteger)((const uint8_t*)output - (const uint8_t*)[buf_out contents]);
                int dims[2] = { T, H };
                id<MTLComputePipelineState> pso = metal_get_pso("lstm_interleave_k");
                if (!pso) { fprintf(stderr, "[metal] lstm_interleave_k PSO not found\n"); abort(); }
                id<MTLCommandBuffer>         cb  = metal_cmd_buf();
                metal_end_lm_encoder();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:h_fwd   offset:0       atIndex:0];
                [enc setBuffer:h_bwd   offset:0       atIndex:1];
                [enc setBuffer:buf_out offset:off_out  atIndex:2];
                [enc setBytes:dims length:sizeof(dims) atIndex:3];
                NSUInteger total = (NSUInteger)(T * 2 * H);
                NSUInteger tg    = MIN(total, (NSUInteger)256);
                [enc dispatchThreads:MTLSizeMake(total, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                [enc endEncoding];
            }
        }
    }
    // No metal_flush() — batched CB design
}

// ─────────────────────────────────────────────────────────────────
// metal_stft_fp32
// ─────────────────────────────────────────────────────────────────
void metal_stft_fp32(
    const float* audio,
    const float* w_real,
    const float* w_imag,
    float*       out,
    int T_audio, int n_fft, int hop_size,
    StreamHandle /*stream*/)
{
    if (!w_real || !w_imag) {
        fprintf(stderr, "[metal] stft_fp32: w_real/w_imag are null — cannot run GPU path\n");
        abort();
    }

    int K      = n_fft / 2 + 1;
    int T_stft = T_audio / hop_size + 1;

    id<MTLBuffer> buf_audio = metal_get_buf(audio);
    id<MTLBuffer> buf_out   = metal_get_buf(out);
    if (!buf_audio || !buf_out) {
        fprintf(stderr, "[metal] stft_fp32: buffer lookup failed\n");
        abort();
    }

    // w_real / w_imag are CPU weight vectors — get or create Metal shared buffers
    size_t w_bytes = (size_t)K * n_fft * sizeof(float);
    id<MTLBuffer> buf_wr = metal_get_or_create_cpu_buf(w_real, w_bytes);
    id<MTLBuffer> buf_wi = metal_get_or_create_cpu_buf(w_imag, w_bytes);
    if (!buf_wr || !buf_wi) {
        fprintf(stderr, "[metal] stft_fp32: weight buffer creation failed\n");
        abort();
    }

    NSUInteger off_audio = (NSUInteger)((const uint8_t*)audio - (const uint8_t*)[buf_audio contents]);
    NSUInteger off_out   = (NSUInteger)((const uint8_t*)out   - (const uint8_t*)[buf_out   contents]);

    int params[4] = { n_fft, hop_size, K, T_audio };

    id<MTLComputePipelineState> pso = metal_get_pso("stft_fp32_k");
    if (!pso) { fprintf(stderr, "[metal] stft_fp32_k PSO not found\n"); abort(); }

    id<MTLCommandBuffer>         cb  = metal_cmd_buf();
    metal_end_lm_encoder();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf_audio offset:off_audio atIndex:0];
    [enc setBuffer:buf_wr    offset:0         atIndex:1];
    [enc setBuffer:buf_wi    offset:0         atIndex:2];
    [enc setBuffer:buf_out   offset:off_out   atIndex:3];
    [enc setBytes:params length:sizeof(params) atIndex:4];
    NSUInteger tgx = MIN((NSUInteger)K,      (NSUInteger)32);
    NSUInteger tgy = MIN((NSUInteger)T_stft, (NSUInteger)8);
    [enc dispatchThreads:MTLSizeMake((NSUInteger)T_stft, (NSUInteger)K, 1)
        threadsPerThreadgroup:MTLSizeMake(tgy, tgx, 1)];
    [enc endEncoding];
    // No metal_flush() — batched CB design
}

// ─────────────────────────────────────────────────────────────────
// metal_istft_fp32
// ─────────────────────────────────────────────────────────────────
void metal_istft_fp32(
    const float* mag,
    const float* phase,
    float*       out,
    int T_frames, int n_fft, int hop_size, bool normalized,
    StreamHandle /*stream*/)
{
    int out_len = (T_frames - 1) * hop_size;
    if (out_len <= 0) return;

    id<MTLBuffer> buf_mag   = metal_get_buf(mag);
    id<MTLBuffer> buf_phase = metal_get_buf(phase);
    id<MTLBuffer> buf_out   = metal_get_buf(out);
    if (!buf_mag || !buf_phase || !buf_out) {
        fprintf(stderr, "[metal] istft_fp32: buffer lookup failed\n");
        abort();
    }

    NSUInteger off_mag   = (NSUInteger)((const uint8_t*)mag   - (const uint8_t*)[buf_mag   contents]);
    NSUInteger off_phase = (NSUInteger)((const uint8_t*)phase - (const uint8_t*)[buf_phase contents]);
    NSUInteger off_out   = (NSUInteger)((const uint8_t*)out   - (const uint8_t*)[buf_out   contents]);

    // Zero the output buffer first via blit encoder
    {
        metal_end_lm_encoder();
        id<MTLBlitCommandEncoder> blit = [metal_cmd_buf() blitCommandEncoder];
        [blit fillBuffer:buf_out range:NSMakeRange(off_out, (NSUInteger)out_len * sizeof(float)) value:0];
        [blit endEncoding];
    }

    int params[4] = { T_frames, n_fft, hop_size, normalized ? 1 : 0 };

    id<MTLComputePipelineState> pso = metal_get_pso("istft_sample_k");
    if (!pso) { fprintf(stderr, "[metal] istft_sample_k PSO not found\n"); abort(); }

    id<MTLCommandBuffer>         cb  = metal_cmd_buf();
    metal_end_lm_encoder();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf_mag   offset:off_mag   atIndex:0];
    [enc setBuffer:buf_phase offset:off_phase atIndex:1];
    [enc setBuffer:buf_out   offset:off_out   atIndex:2];
    [enc setBytes:params length:sizeof(params) atIndex:3];
    NSUInteger total = (NSUInteger)out_len;
    NSUInteger tg    = MIN(total, (NSUInteger)256);
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    // No metal_flush() — batched CB design
}

// ─────────────────────────────────────────────────────────────────
// metal_reflection_pad1d
// ─────────────────────────────────────────────────────────────────
void metal_reflection_pad1d(
    const float* in,
    float*       out,
    int T, int C,
    StreamHandle /*stream*/)
{
    id<MTLBuffer> buf_in  = metal_get_buf(in);
    id<MTLBuffer> buf_out = metal_get_buf(out);
    if (!buf_in || !buf_out) {
        fprintf(stderr, "[metal] reflection_pad1d: buffer lookup failed\n");
        abort();
    }

    NSUInteger off_in  = (NSUInteger)((const uint8_t*)in  - (const uint8_t*)[buf_in  contents]);
    NSUInteger off_out = (NSUInteger)((const uint8_t*)out - (const uint8_t*)[buf_out contents]);

    int dims[2] = { T, C };

    id<MTLComputePipelineState> pso = metal_get_pso("reflection_pad1d_k");
    if (!pso) { fprintf(stderr, "[metal] reflection_pad1d_k PSO not found\n"); abort(); }

    id<MTLCommandBuffer>         cb  = metal_cmd_buf();
    metal_end_lm_encoder();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf_in  offset:off_in  atIndex:0];
    [enc setBuffer:buf_out offset:off_out atIndex:1];
    [enc setBytes:dims length:sizeof(dims) atIndex:2];
    NSUInteger total = (NSUInteger)((T + 1) * C);
    NSUInteger tg    = MIN(total, (NSUInteger)256);
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    // No metal_flush() — batched CB design
}

// ─────────────────────────────────────────────────────────────────
// metal_sine_gen_fp32
// ONNX-compatible path (upsample_scale > 0):
//   stage 1: sine_gen_rad_dn_k   (H, T_dn)
//   stage 2: sine_gen_cumsum_k   (H)
//   stage 3: sine_gen_output_k   (T, H)
// ─────────────────────────────────────────────────────────────────
void metal_sine_gen_fp32(
    const float* f0,
    float*       out,
    int T, int H, int sr, int upsample_scale,
    float sine_amp, float voiced_threshold,
    StreamHandle /*stream*/)
{
    if (upsample_scale <= 0) {
        fprintf(stderr, "[metal] sine_gen_fp32: upsample_scale must be > 0 for GPU path\n");
        abort();
    }

    id<MTLBuffer> buf_f0  = metal_get_buf(f0);
    id<MTLBuffer> buf_out = metal_get_buf(out);
    if (!buf_f0 || !buf_out) {
        fprintf(stderr, "[metal] sine_gen_fp32: buffer lookup failed\n");
        abort();
    }

    NSUInteger off_f0  = (NSUInteger)((const uint8_t*)f0  - (const uint8_t*)[buf_f0  contents]);
    NSUInteger off_out = (NSUInteger)((const uint8_t*)out - (const uint8_t*)[buf_out contents]);

    // T_dn = T / upsample_scale  (integer, since T is audio-rate length produced by upsample)
    int T_dn = T / upsample_scale;
    if (T_dn <= 0) T_dn = 1;

    @autoreleasepool {
        size_t tmp_bytes = (size_t)H * T_dn * sizeof(float);
        id<MTLBuffer> rad_dn   = [g_metal_device newBufferWithLength:tmp_bytes
                                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> phase_dn = [g_metal_device newBufferWithLength:tmp_bytes
                                                             options:MTLResourceStorageModeShared];

        // Stage 1: sine_gen_rad_dn_k — grid (H, T_dn)
        {
            int params[4]  = { T, H, T_dn, sr };
            float scale_f  = (float)upsample_scale;

            id<MTLComputePipelineState> pso = metal_get_pso("sine_gen_rad_dn_k");
            if (!pso) { fprintf(stderr, "[metal] sine_gen_rad_dn_k PSO not found\n"); abort(); }
            id<MTLCommandBuffer>         cb  = metal_cmd_buf();
            metal_end_lm_encoder();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:buf_f0  offset:off_f0 atIndex:0];
            [enc setBuffer:rad_dn  offset:0      atIndex:1];
            [enc setBytes:params   length:sizeof(params)  atIndex:2];
            [enc setBytes:&scale_f length:sizeof(float)   atIndex:3];
            NSUInteger tgx = MIN((NSUInteger)H,    (NSUInteger)32);
            NSUInteger tgy = MIN((NSUInteger)T_dn, (NSUInteger)8);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)H, (NSUInteger)T_dn, 1)
                threadsPerThreadgroup:MTLSizeMake(tgx, tgy, 1)];
            [enc endEncoding];
        }

        // Stage 2: sine_gen_cumsum_k — 1 thread per harmonic
        {
            int dims[2] = { H, T_dn };
            id<MTLComputePipelineState> pso = metal_get_pso("sine_gen_cumsum_k");
            if (!pso) { fprintf(stderr, "[metal] sine_gen_cumsum_k PSO not found\n"); abort(); }
            id<MTLCommandBuffer>         cb  = metal_cmd_buf();
            metal_end_lm_encoder();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:rad_dn   offset:0      atIndex:0];
            [enc setBuffer:phase_dn offset:0      atIndex:1];
            [enc setBytes:dims length:sizeof(dims) atIndex:2];
            NSUInteger total = (NSUInteger)H;
            NSUInteger tg    = MIN(total, (NSUInteger)256);
            [enc dispatchThreads:MTLSizeMake(total, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
        }

        // Stage 3: sine_gen_output_k — grid (T, H)
        {
            int params[4]       = { T, H, T_dn, 0 };
            float ap[2]         = { sine_amp, voiced_threshold };
            float scale_f       = (float)upsample_scale;
            float inv_scale_f   = 1.0f / scale_f;

            id<MTLComputePipelineState> pso = metal_get_pso("sine_gen_output_k");
            if (!pso) { fprintf(stderr, "[metal] sine_gen_output_k PSO not found\n"); abort(); }
            id<MTLCommandBuffer>         cb  = metal_cmd_buf();
            metal_end_lm_encoder();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:phase_dn offset:0        atIndex:0];
            [enc setBuffer:buf_f0   offset:off_f0   atIndex:1];
            [enc setBuffer:buf_out  offset:off_out  atIndex:2];
            [enc setBytes:params    length:sizeof(params)    atIndex:3];
            [enc setBytes:ap        length:sizeof(ap)        atIndex:4];
            [enc setBytes:&inv_scale_f length:sizeof(float)  atIndex:5];
            [enc setBytes:&scale_f     length:sizeof(float)  atIndex:6];
            NSUInteger tgx = MIN((NSUInteger)T, (NSUInteger)32);
            NSUInteger tgy = MIN((NSUInteger)H, (NSUInteger)8);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)T, (NSUInteger)H, 1)
                threadsPerThreadgroup:MTLSizeMake(tgx, tgy, 1)];
            [enc endEncoding];
        }
    }
    // No metal_flush() — batched CB design
}
