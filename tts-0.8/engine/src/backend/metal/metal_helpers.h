// src/backend/metal/metal_helpers.h
// ObjC/ObjC++ header — only include from .mm files.
#pragma once

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>

#ifdef __cplusplus
#include <cstddef>

// ─────────────────────────────────────────────────────────────────
// Global Metal state (defined in metal_backend.mm)
// ─────────────────────────────────────────────────────────────────
extern id<MTLDevice>       g_metal_device;
extern id<MTLCommandQueue> g_metal_queue;
extern id<MTLLibrary>      g_metal_library;

// ─────────────────────────────────────────────────────────────────
// Buffer lookup by contents pointer
// Returns the MTLBuffer whose [buffer contents] == ptr.
// Returns nil if not registered (e.g., plain CPU malloc).
// ─────────────────────────────────────────────────────────────────
id<MTLBuffer> metal_get_buf(const void* ptr);

// Scan all registered activation buffers for one whose memory range contains ptr.
// Sets *out_offset to ptr's byte offset within that buffer.
// Returns nil if ptr is not inside any registered buffer.
// Use this for interior pointers (e.g., beta = gamma_beta_base + C).
id<MTLBuffer> metal_find_buf_containing(const void* ptr, NSUInteger* out_offset);

// ─────────────────────────────────────────────────────────────────
// Weight buffer registry
//   metal_register_weight_buf(packed_ptr, buf):
//     Store buf (original row-major weight data) keyed by packed_ptr.
//   metal_get_weight_buf(ptr):
//     Look up original row-major weight buffer by ptr.
//     ptr may be the CPU packed pointer OR the Metal buffer contents
//     pointer (after upload_weights migrates the key via copy_h2d).
//   metal_unregister_weight_buf(ptr):
//     Release the registered buffer (called from free_packed_*).
// ─────────────────────────────────────────────────────────────────
void          metal_register_weight_buf(const void* packed_ptr, id<MTLBuffer> buf);
id<MTLBuffer> metal_get_weight_buf(const void* ptr);
void          metal_unregister_weight_buf(const void* ptr);

// ─────────────────────────────────────────────────────────────────
// CPU → Metal buffer cache
//   Creates (or returns cached) a Metal buffer containing a copy of
//   the data at ptr.  Caches by CPU pointer — safe for fixed weights.
// ─────────────────────────────────────────────────────────────────
id<MTLBuffer> metal_get_or_create_cpu_buf(const void* ptr, size_t bytes);

// ─────────────────────────────────────────────────────────────────
// Growing scratch col buffer (for im2col intermediate storage).
//   Returns a buffer of at least needed_bytes.  The same buffer is
//   returned on subsequent calls (grown when necessary).
// ─────────────────────────────────────────────────────────────────
id<MTLBuffer> metal_get_col_buf(size_t needed_bytes);

// ─────────────────────────────────────────────────────────────────
// PSO cache and command buffer helpers
// ─────────────────────────────────────────────────────────────────
id<MTLComputePipelineState> metal_get_pso(const char* name);

// Returns the active (batching) command buffer, creating one if needed.
// Use metal_flush() to commit and wait when all sub-ops of a layer are done.
id<MTLCommandBuffer> metal_cmd_buf();

// Commit the active command buffer and wait for GPU completion.
// Call this at the end of each op function to ensure results are visible to CPU.
void metal_flush();

// End any open LM batch encoder (g_lm_enc) so a new regular encoder can be opened.
// Must be called before [cb computeCommandEncoder] in any op that bypasses MetalDispatch.
void metal_end_lm_encoder();

// ─────────────────────────────────────────────────────────────────
// MPS matrix multiply helper
//   Computes C = A[M,K] × B^T[N,K]   (B stored row-major as [N, K])
//   Encodes into the active command buffer (call metal_flush() to commit).
// ─────────────────────────────────────────────────────────────────
void metal_mps_matmul(
    id<MTLBuffer> A_buf, NSUInteger A_off,
    id<MTLBuffer> B_buf, NSUInteger B_off,
    id<MTLBuffer> C_buf, NSUInteger C_off,
    int M, int K, int N);

// ─────────────────────────────────────────────────────────────────
// MetalDispatch — RAII wrapper for encode + dispatch a single kernel.
// Encodes into the active command buffer (call metal_flush() to commit).
// ─────────────────────────────────────────────────────────────────
struct MetalDispatch {
    id<MTLCommandBuffer>         cb;
    id<MTLComputeCommandEncoder> enc;
    int                          idx;

    MetalDispatch(const char* fn_name);

    void buf(id<MTLBuffer> b, NSUInteger offset = 0);
    void bytes(const void* data, size_t n);
    void tg_mem(size_t n);

    // Dispatch N threads (linear) — encodes only, no commit.
    void run(NSUInteger N, NSUInteger tg_size = 256);

    // Dispatch explicit threadgroup grid — encodes only, no commit.
    void run_threadgroups(MTLSize tg_count, MTLSize tg_size);
};

#endif // __cplusplus
