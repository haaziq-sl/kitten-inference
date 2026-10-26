// src/backend/metal/metal_backend.mm
// Main Metal backend: global state, MetalAllocator, helper implementations,
// and BackendOps registration.
//
// Shaders are compiled from src/backend/metal/shaders/kernels.metal at build
// time (→ metal_kernels.metallib in the build dir). At runtime the .metallib
// is loaded first; on failure we fall back to compiling the source string that
// is kept in sync as a build-time fallback.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "metal_backend.hpp"
#include "metal_helpers.h"
#include "backend/backend.hpp"
#include "ops/metal_ops.hpp"
#include "backend/cpu/ops_neon.hpp"

#include <exception>
#include <unordered_map>
#include <mutex>
#include <cstring>
#include <cstdio>
#include <dlfcn.h>
#include <mach-o/dyld.h>

// ─────────────────────────────────────────────────────────────────
// Global Metal state
// ─────────────────────────────────────────────────────────────────
id<MTLDevice>       g_metal_device  = nil;
id<MTLCommandQueue> g_metal_queue   = nil;
id<MTLLibrary>      g_metal_library = nil;

// ── Activation buffer map ─────────────────────────────────────────
// Maps MTLBuffer.contents pointer → bridge-retained MTLBuffer.
// Used for activation tensors allocated via MetalAllocator.
static std::unordered_map<const void*, void*> g_buf_map;
static std::mutex                              g_buf_map_mtx;

// ── Weight buffer registry ────────────────────────────────────────
// Maps CPU-packed-weight pointer → bridge-retained MTLBuffer containing
// the ORIGINAL row-major weights (for use with MPS / custom kernels).
//
// Lifecycle:
//   1. metal_pack_weights_*() calls metal_register_weight_buf(packed_ptr, buf).
//   2. upload_weights() calls copy_h2d(metal_ptr, packed_ptr, n), which
//      migrates the entry: g_wt_map[metal_ptr] = g_wt_map[packed_ptr]; erase(packed_ptr).
//   3. Layer forward() calls metal_get_weight_buf(metal_ptr) to retrieve buf.
//   4. free_packed_*() calls metal_unregister_weight_buf(ptr) for cleanup.
static std::unordered_map<const void*, void*> g_wt_map;
static std::mutex                              g_wt_mtx;

// ── CPU weight buffer cache ───────────────────────────────────────
// Maps CPU pointer → {bridge-retained MTLBuffer, size}.
// Invalidated when the requested size differs (detects pointer reuse after free).
struct CpuBufEntry { void* buf_retained; size_t bytes; };
static std::unordered_map<const void*, CpuBufEntry> g_cpu_wt_map;
static std::mutex                                    g_cpu_wt_mtx;

// ── Growing scratch col buffer ────────────────────────────────────
// Single Metal buffer re-used across im2col calls; grows as needed.
static id<MTLBuffer> g_col_buf       = nil;
static size_t        g_col_buf_bytes = 0;
static std::mutex    g_col_mtx;

// ── PSO cache ─────────────────────────────────────────────────────
static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* g_pso_cache = nil;
static std::mutex g_pso_cache_mtx;

// ── Active (batching) command buffer ──────────────────────────────
// All GPU ops encode into this CB; metal_flush() commits and waits.
static id<MTLCommandBuffer> g_active_cb = nil;

// ── LM batch encoder ──────────────────────────────────────────────
// LM ops share a single compute encoder to avoid per-dispatch encoder
// creation overhead (each MetalDispatch encoder = ~0.25ms on M1).
// Ended lazily: when MetalDispatch is used (non-LM op) or at flush.
static id<MTLComputeCommandEncoder> g_lm_enc = nil;

void metal_end_lm_encoder() {
    if (g_lm_enc) {
        [g_lm_enc endEncoding];
        g_lm_enc = nil;
    }
}

// ── MPS MatMul kernel cache ───────────────────────────────────────
// Key: packed (M, K, N, transposeRight).  Reuse across identical shapes.
struct MpsMMKey {
    int M, K, N;
    BOOL transposeRight;
    double alpha;
    bool operator==(const MpsMMKey& o) const {
        return M == o.M && K == o.K && N == o.N &&
               transposeRight == o.transposeRight && alpha == o.alpha;
    }
};
struct MpsMMKeyHash {
    size_t operator()(const MpsMMKey& k) const {
        size_t h = (size_t)k.M * 100003 ^ (size_t)k.K * 10007 ^ (size_t)k.N;
        h ^= k.transposeRight ? 0xDEAD : 0xBEEF;
        return h;
    }
};
static std::unordered_map<MpsMMKey, void*, MpsMMKeyHash> g_mps_mm_cache;
static std::mutex g_mps_mm_mtx;

// ─────────────────────────────────────────────────────────────────
// Buffer lookup helpers
// ─────────────────────────────────────────────────────────────────
id<MTLBuffer> metal_get_buf(const void* ptr) {
    std::lock_guard<std::mutex> lk(g_buf_map_mtx);
    auto it = g_buf_map.find(ptr);
    if (it == g_buf_map.end()) return nil;
    return (__bridge id<MTLBuffer>)it->second;
}

id<MTLBuffer> metal_find_buf_containing(const void* ptr, NSUInteger* out_offset) {
    if (!ptr) return nil;
    const uint8_t* p = (const uint8_t*)ptr;
    std::lock_guard<std::mutex> lk(g_buf_map_mtx);
    for (auto& kv : g_buf_map) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)kv.second;
        const uint8_t* base = (const uint8_t*)kv.first;
        size_t len = [buf length];
        if (p >= base && p < base + len) {
            if (out_offset) *out_offset = (NSUInteger)(p - base);
            return buf;
        }
    }
    return nil;
}

// ─────────────────────────────────────────────────────────────────
// Weight buffer registry implementation
// ─────────────────────────────────────────────────────────────────
void metal_register_weight_buf(const void* packed_ptr, id<MTLBuffer> buf) {
    if (!packed_ptr || !buf) return;
    std::lock_guard<std::mutex> lk(g_wt_mtx);
    // Release any existing entry for this key first.
    auto it = g_wt_map.find(packed_ptr);
    if (it != g_wt_map.end()) {
        id<MTLBuffer> old NS_VALID_UNTIL_END_OF_SCOPE = (__bridge_transfer id<MTLBuffer>)it->second;
        (void)old;
        g_wt_map.erase(it);
    }
    g_wt_map[packed_ptr] = (__bridge_retained void*)buf;
}

id<MTLBuffer> metal_get_weight_buf(const void* ptr) {
    if (!ptr) return nil;
    std::lock_guard<std::mutex> lk(g_wt_mtx);
    auto it = g_wt_map.find(ptr);
    if (it == g_wt_map.end()) return nil;
    return (__bridge id<MTLBuffer>)it->second;
}

void metal_unregister_weight_buf(const void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lk(g_wt_mtx);
    auto it = g_wt_map.find(ptr);
    if (it != g_wt_map.end()) {
        id<MTLBuffer> buf NS_VALID_UNTIL_END_OF_SCOPE = (__bridge_transfer id<MTLBuffer>)it->second;
        (void)buf;
        g_wt_map.erase(it);
    }
}

// ─────────────────────────────────────────────────────────────────
// CPU → Metal buffer cache
// Returns a cached MTLBuffer for ptr if one exists with matching size
// AND matching content.  Invalidates on size mismatch (pointer reuse
// after free with different size) or content mismatch (pointer reuse
// after free with same size but different data — e.g., different model
// loaded at the same heap address).
// ─────────────────────────────────────────────────────────────────
id<MTLBuffer> metal_get_or_create_cpu_buf(const void* ptr, size_t bytes) {
    if (!ptr || !g_metal_device || bytes == 0) return nil;
    {
        std::lock_guard<std::mutex> lk(g_cpu_wt_mtx);
        auto it = g_cpu_wt_map.find(ptr);
        if (it != g_cpu_wt_map.end()) {
            if (it->second.bytes == bytes) {
                id<MTLBuffer> buf = (__bridge id<MTLBuffer>)it->second.buf_retained;
                // Quick content check: compare only first 64 bytes (fixed weights are
                // immutable during inference; full memcmp of large buffers is too costly).
                size_t check = bytes < 64 ? bytes : 64;
                if (memcmp([buf contents], ptr, check) == 0)
                    return buf;
            }
            // Stale entry (size or content mismatch).  Release it.
            id<MTLBuffer> old NS_VALID_UNTIL_END_OF_SCOPE =
                (__bridge_transfer id<MTLBuffer>)it->second.buf_retained;
            (void)old;
            g_cpu_wt_map.erase(it);
        }
    }
    id<MTLBuffer> buf = [g_metal_device newBufferWithBytes:ptr
                                                    length:bytes
                                                   options:MTLResourceStorageModeShared];
    if (!buf) return nil;
    std::lock_guard<std::mutex> lk(g_cpu_wt_mtx);
    g_cpu_wt_map[ptr] = { (__bridge_retained void*)buf, bytes };
    return buf;
}

// ─────────────────────────────────────────────────────────────────
// Growing scratch col buffer
// ─────────────────────────────────────────────────────────────────
id<MTLBuffer> metal_get_col_buf(size_t needed_bytes) {
    std::lock_guard<std::mutex> lk(g_col_mtx);
    if (needed_bytes > g_col_buf_bytes) {
        g_col_buf = [g_metal_device newBufferWithLength:needed_bytes
                                                options:MTLResourceStorageModeShared];
        g_col_buf_bytes = g_col_buf ? needed_bytes : 0;
    }
    return g_col_buf;
}

// ─────────────────────────────────────────────────────────────────
// MetalAllocator
// ─────────────────────────────────────────────────────────────────
MemSpace MetalAllocator::mem_space() const { return MemSpace::Metal; }

void* MetalAllocator::alloc(size_t n) {
    if (!g_metal_device) return nullptr;
    if (n == 0) n = 4;

    id<MTLBuffer> buf = [g_metal_device newBufferWithLength:n
                                                    options:MTLResourceStorageModeShared];
    if (!buf) {
        fprintf(stderr, "[metal] MetalAllocator::alloc(%zu) failed\n", n);
        return nullptr;
    }

    void* contents = [buf contents];
    {
        std::lock_guard<std::mutex> lk(g_buf_map_mtx);
        g_buf_map[contents] = (__bridge_retained void*)buf;
    }
    return contents;
}

void MetalAllocator::free(void* p) {
    if (!p) return;
    std::lock_guard<std::mutex> lk(g_buf_map_mtx);
    auto it = g_buf_map.find(p);
    if (it != g_buf_map.end()) {
        id<MTLBuffer> buf NS_VALID_UNTIL_END_OF_SCOPE =
            (__bridge_transfer id<MTLBuffer>)it->second;
        (void)buf;
        g_buf_map.erase(it);
    }
}

void MetalAllocator::copy_h2d(void* dst, const void* src, size_t n) {
    ::memcpy(dst, src, n);
    // Migrate weight registry: if src is a registered packed-weight pointer,
    // re-key its entry under dst (the new Metal buffer contents pointer).
    std::lock_guard<std::mutex> lk(g_wt_mtx);
    auto it = g_wt_map.find(src);
    if (it != g_wt_map.end()) {
        // Move value to new key — no retain/release needed (just pointer copy).
        g_wt_map[dst] = it->second;
        g_wt_map.erase(it);
    }
}

void MetalAllocator::copy_d2h(void* dst, const void* src, size_t n) {
    // Flush any pending GPU work before CPU reads from device memory.
    // This is the primary sync point for the entire forward pass — all GPU ops
    // accumulate in g_active_cb without flushing; this call commits them all at once.
    metal_flush();
    ::memcpy(dst, src, n);
}
void MetalAllocator::copy_d2d(void* dst, const void* src, size_t n) {
    ::memcpy(dst, src, n);
}
void MetalAllocator::sync() { /* unified memory: no-op */ }

// ─────────────────────────────────────────────────────────────────
// PSO cache
// ─────────────────────────────────────────────────────────────────
id<MTLComputePipelineState> metal_get_pso(const char* name) {
    NSString* key = [NSString stringWithUTF8String:name];
    {
        std::lock_guard<std::mutex> lk(g_pso_cache_mtx);
        id<MTLComputePipelineState> cached = g_pso_cache[key];
        if (cached) return cached;
    }

    id<MTLFunction> fn = [g_metal_library newFunctionWithName:key];
    if (!fn) {
        fprintf(stderr, "[metal] metal_get_pso: kernel '%s' not found\n", name);
        return nil;
    }

    NSError* err = nil;
    id<MTLComputePipelineState> pso =
        [g_metal_device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "[metal] metal_get_pso: PSO creation failed for '%s': %s\n",
                name, [[err localizedDescription] UTF8String]);
        return nil;
    }

    {
        std::lock_guard<std::mutex> lk(g_pso_cache_mtx);
        g_pso_cache[key] = pso;
    }
    return pso;
}

void metal_autorelease_scope(const std::function<void()>& fn) {
    // No exception crosses the pool's end: it would skip emptying the pool.
    std::exception_ptr error;
    @autoreleasepool {
        try {
            fn();
        } catch (...) {
            error = std::current_exception();
        }
    }
    if (error) std::rethrow_exception(error);
}

id<MTLCommandBuffer> metal_cmd_buf() {
    if (!g_active_cb)
        g_active_cb = [g_metal_queue commandBuffer];
    return g_active_cb;
}

void metal_flush() {
    metal_end_lm_encoder();  // end any open LM batch encoder before commit
    if (g_active_cb) {
        [g_active_cb commit];
        [g_active_cb waitUntilCompleted];
        g_active_cb = nil;
    }
}

// ─────────────────────────────────────────────────────────────────
// Cached MPS MatMul kernel lookup/creation.
// ─────────────────────────────────────────────────────────────────
static MPSMatrixMultiplication* metal_get_mps_mm(int M, int K, int N, BOOL transposeRight, double alpha) {
    MpsMMKey key{ M, K, N, transposeRight, alpha };
    {
        std::lock_guard<std::mutex> lk(g_mps_mm_mtx);
        auto it = g_mps_mm_cache.find(key);
        if (it != g_mps_mm_cache.end())
            return (__bridge MPSMatrixMultiplication*)it->second;
    }
    int resultCols = transposeRight ? N : N;  // N is always result columns
    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:g_metal_device
         transposeLeft:NO
        transposeRight:transposeRight
           resultRows:(NSUInteger)M
        resultColumns:(NSUInteger)N
      interiorColumns:(NSUInteger)K
                 alpha:alpha
                  beta:0.0];
    {
        std::lock_guard<std::mutex> lk(g_mps_mm_mtx);
        g_mps_mm_cache[key] = (__bridge_retained void*)mm;
    }
    return mm;
}

// ─────────────────────────────────────────────────────────────────
// MPS matrix multiply: C = A[M,K] × B^T[N,K]
// Encodes into the active command buffer — call metal_flush() to commit.
// ─────────────────────────────────────────────────────────────────
void metal_mps_matmul(
    id<MTLBuffer> A_buf, NSUInteger A_off,
    id<MTLBuffer> B_buf, NSUInteger B_off,
    id<MTLBuffer> C_buf, NSUInteger C_off,
    int M, int K, int N)
{
    if (M == 0 || K == 0 || N == 0) return;
    // MPS encodes into the command buffer using its own encoder.
    // End any open LM batch encoder first to avoid "encoder already active" error.
    metal_end_lm_encoder();

    MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
        matrixDescriptorWithRows:(NSUInteger)M
                         columns:(NSUInteger)K
                        rowBytes:(NSUInteger)(K * sizeof(float))
                        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descB = [MPSMatrixDescriptor
        matrixDescriptorWithRows:(NSUInteger)N
                         columns:(NSUInteger)K
                        rowBytes:(NSUInteger)(K * sizeof(float))
                        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descC = [MPSMatrixDescriptor
        matrixDescriptorWithRows:(NSUInteger)M
                         columns:(NSUInteger)N
                        rowBytes:(NSUInteger)(N * sizeof(float))
                        dataType:MPSDataTypeFloat32];

    MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:A_buf offset:A_off descriptor:descA];
    MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:B_buf offset:B_off descriptor:descB];
    MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:C_buf offset:C_off descriptor:descC];

    MPSMatrixMultiplication* mm = metal_get_mps_mm(M, K, N, YES, 1.0);
    id<MTLCommandBuffer> cb = metal_cmd_buf();
    [mm encodeToCommandBuffer:cb leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    // No commit/wait here — caller calls metal_flush() after all sub-ops.
}

// ─────────────────────────────────────────────────────────────────
// MetalDispatch implementation
// ─────────────────────────────────────────────────────────────────
MetalDispatch::MetalDispatch(const char* fn_name) : idx(0) {
    // End any open LM batch encoder first (only one encoder may be open at a time).
    metal_end_lm_encoder();
    id<MTLComputePipelineState> pso = metal_get_pso(fn_name);
    cb  = metal_cmd_buf();
    enc = [cb computeCommandEncoder];
    if (pso) [enc setComputePipelineState:pso];
}

void MetalDispatch::buf(id<MTLBuffer> b, NSUInteger offset) {
    [enc setBuffer:b offset:offset atIndex:idx++];
}

void MetalDispatch::bytes(const void* data, size_t n) {
    [enc setBytes:data length:n atIndex:idx++];
}

void MetalDispatch::tg_mem(size_t n) {
    [enc setThreadgroupMemoryLength:n atIndex:0];
}

void MetalDispatch::run(NSUInteger N, NSUInteger tg_size) {
    if (N == 0) { [enc endEncoding]; return; }
    NSUInteger tg = (tg_size < N) ? tg_size : N;
    [enc dispatchThreads:MTLSizeMake(N, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    // Caller is responsible for calling metal_flush() when all sub-ops are done.
}

void MetalDispatch::run_threadgroups(MTLSize tg_count, MTLSize tg_size) {
    [enc dispatchThreadgroups:tg_count threadsPerThreadgroup:tg_size];
    [enc endEncoding];
    // Caller is responsible for calling metal_flush() when all sub-ops are done.
}

// ─────────────────────────────────────────────────────────────────
// LMDispatch — lightweight dispatch that batches into g_lm_enc.
// Never calls endEncoding; the encoder stays open until
// metal_end_lm_encoder() is called (at flush or non-LM dispatch).
// ─────────────────────────────────────────────────────────────────
struct LMDispatch {
    int idx = 0;
    LMDispatch(const char* fn_name) {
        if (!g_lm_enc) g_lm_enc = [metal_cmd_buf() computeCommandEncoder];
        [g_lm_enc setComputePipelineState:metal_get_pso(fn_name)];
        idx = 0;
    }
    void buf(id<MTLBuffer> b, NSUInteger off = 0) {
        [g_lm_enc setBuffer:b offset:off atIndex:idx++];
    }
    void bytes(const void* d, size_t n) {
        [g_lm_enc setBytes:d length:n atIndex:idx++];
    }
    void tg_mem(size_t n) {
        [g_lm_enc setThreadgroupMemoryLength:n atIndex:0];
    }
    void run(NSUInteger N, NSUInteger tg = 256) {
        if (!N) return;
        NSUInteger t = (tg < N) ? tg : N;
        [g_lm_enc dispatchThreads:MTLSizeMake(N, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(t, 1, 1)];
    }
    void run_tg(MTLSize tg_count, MTLSize tg_size) {
        [g_lm_enc dispatchThreadgroups:tg_count threadsPerThreadgroup:tg_size];
    }
};

// ─────────────────────────────────────────────────────────────────
// LM op scratch buffers (growing, single-use — LM inference is serial)
// ─────────────────────────────────────────────────────────────────
static id<MTLBuffer> g_lm_rope_scratch  = nil; static size_t g_lm_rope_bytes  = 0;
static id<MTLBuffer> g_lm_score_scratch = nil; static size_t g_lm_score_bytes = 0;

static id<MTLBuffer> lm_rope_scratch(size_t n) {
    if (n > g_lm_rope_bytes) {
        g_lm_rope_scratch = [g_metal_device newBufferWithLength:n
                            options:MTLResourceStorageModeShared];
        g_lm_rope_bytes = g_lm_rope_scratch ? n : 0;
    }
    return g_lm_rope_scratch;
}
static id<MTLBuffer> lm_score_scratch(size_t n) {
    if (n > g_lm_score_bytes) {
        g_lm_score_scratch = [g_metal_device newBufferWithLength:n
                             options:MTLResourceStorageModeShared];
        g_lm_score_bytes = g_lm_score_scratch ? n : 0;
    }
    return g_lm_score_scratch;
}

// ─────────────────────────────────────────────────────────────────
// LM op implementations (GPU)
// ─────────────────────────────────────────────────────────────────

// Residual add (add_fp32) — LM variant using LMDispatch.
// Also handles add_vectors_fp32 which is the in-place residual add variant.
static void metal_lm_add_fp32(const float* a, const float* b, float* c, int N, bool relu, StreamHandle) {
    id<MTLBuffer> buf_a = metal_get_buf(a);
    id<MTLBuffer> buf_b = metal_get_buf(b);
    id<MTLBuffer> buf_c = metal_get_buf(c);
    if (!buf_a || !buf_b || !buf_c) {
        fprintf(stderr, "[metal] lm_add_fp32: buffer lookup failed\n"); abort();
    }
    NSUInteger off_a = (NSUInteger)((const uint8_t*)a - (const uint8_t*)[buf_a contents]);
    NSUInteger off_b = (NSUInteger)((const uint8_t*)b - (const uint8_t*)[buf_b contents]);
    NSUInteger off_c = (NSUInteger)((const uint8_t*)c - (const uint8_t*)[buf_c contents]);
    int do_relu = relu ? 1 : 0;
    LMDispatch d("add_fp32_k");
    d.buf(buf_a, off_a);
    d.buf(buf_b, off_b);
    d.buf(buf_c, off_c);
    d.bytes(&do_relu, sizeof(int));
    d.run((NSUInteger)N);
}

// Residual add (add_vectors_fp32) — LM variant using LMDispatch.
static void metal_lm_add_vectors_fp32(float* dst, const float* src, int N, StreamHandle) {
    id<MTLBuffer> buf_dst = metal_get_buf(dst);
    id<MTLBuffer> buf_src = metal_get_buf(src);
    if (!buf_dst || !buf_src) {
        fprintf(stderr, "[metal] lm_add_vectors: buffer lookup failed\n"); abort();
    }
    NSUInteger off_dst = (NSUInteger)((const uint8_t*)dst - (const uint8_t*)[buf_dst contents]);
    NSUInteger off_src = (NSUInteger)((const uint8_t*)src - (const uint8_t*)[buf_src contents]);
    LMDispatch d("add_vectors_k");
    d.buf(buf_dst, off_dst);
    d.buf(buf_src, off_src);
    d.run((NSUInteger)N);
}

// SiLU and elementwise-mul: LM-specific variants that use LMDispatch
// so they don't interrupt the persistent encoder.
static void metal_lm_silu_fp32(const float* in, float* out, int N, StreamHandle) {
    id<MTLBuffer> buf_in  = metal_get_buf(in);
    id<MTLBuffer> buf_out = metal_get_buf(out);
    if (!buf_in || !buf_out) {
        fprintf(stderr, "[metal] lm_silu: buffer lookup failed\n"); abort();
    }
    NSUInteger off_in  = (NSUInteger)((const uint8_t*)in  - (const uint8_t*)[buf_in  contents]);
    NSUInteger off_out = (NSUInteger)((const uint8_t*)out - (const uint8_t*)[buf_out contents]);
    LMDispatch d("silu_k");
    d.buf(buf_in,  off_in);
    d.buf(buf_out, off_out);
    d.run((NSUInteger)N);
}

static void metal_lm_elemwise_mul_fp32(const float* a, const float* b, float* out, int N, StreamHandle) {
    id<MTLBuffer> buf_a   = metal_get_buf(a);
    id<MTLBuffer> buf_b   = metal_get_buf(b);
    id<MTLBuffer> buf_out = metal_get_buf(out);
    if (!buf_a || !buf_b || !buf_out) {
        fprintf(stderr, "[metal] lm_elemwise_mul: buffer lookup failed\n"); abort();
    }
    NSUInteger off_a   = (NSUInteger)((const uint8_t*)a   - (const uint8_t*)[buf_a   contents]);
    NSUInteger off_b   = (NSUInteger)((const uint8_t*)b   - (const uint8_t*)[buf_b   contents]);
    NSUInteger off_out = (NSUInteger)((const uint8_t*)out - (const uint8_t*)[buf_out contents]);
    LMDispatch d("elemwise_mul_k");
    d.buf(buf_a,   off_a);
    d.buf(buf_b,   off_b);
    d.buf(buf_out, off_out);
    d.run((NSUInteger)N);
}

// RMSNorm on GPU.  gamma is a CPU pointer — cached into a Metal buffer once.
static void metal_lm_rmsnorm_fp32(const float* input, const float* gamma, float* output,
                                   int N, int C, float eps, StreamHandle)
{
    NSUInteger off_in = 0, off_out = 0;
    id<MTLBuffer> buf_in  = metal_find_buf_containing(input,  &off_in);
    id<MTLBuffer> buf_out = metal_find_buf_containing(output, &off_out);
    id<MTLBuffer> buf_gam = metal_get_or_create_cpu_buf(gamma, (size_t)C * sizeof(float));
    if (!buf_in || !buf_out || !buf_gam) {
        fprintf(stderr, "[metal] rmsnorm: buffer lookup failed\n"); abort();
    }
    struct { int x, y; } dims{N, C};
    NSUInteger tg = (C >= 256) ? 256 : (NSUInteger)C;
    LMDispatch d("rmsnorm_k");
    d.buf(buf_in,  off_in);
    d.buf(buf_gam, 0);
    d.buf(buf_out, off_out);
    d.bytes(&dims, sizeof(dims));
    d.bytes(&eps,  sizeof(float));
    d.tg_mem(tg * sizeof(float));
    d.run_tg(MTLSizeMake((NSUInteger)N, 1, 1), MTLSizeMake(tg, 1, 1));
}

// INT8 GEMV (N=1 decode).  weight_i8 may be a Metal pointer (uploaded) or CPU.
static void metal_lm_seqgemm_int8_matvec(
    const float* input, const int8_t* weight_i8,
    const float* w_scales, const float* bias,
    float* output, int C_in, int C_out, StreamHandle)
{
    NSUInteger in_off = 0, out_off = 0, w_off = 0;
    id<MTLBuffer> buf_in  = metal_find_buf_containing(input,  &in_off);
    id<MTLBuffer> buf_out = metal_find_buf_containing(output, &out_off);
    // weight may be in g_buf_map (Metal-allocated upload) or need cpu_buf
    id<MTLBuffer> buf_w = metal_get_buf(weight_i8);
    if (!buf_w) buf_w = metal_find_buf_containing(weight_i8, &w_off);
    if (!buf_w) buf_w = metal_get_or_create_cpu_buf(weight_i8, (size_t)C_out * C_in);
    id<MTLBuffer> buf_sc = metal_get_or_create_cpu_buf(w_scales, (size_t)C_out * sizeof(float));
    id<MTLBuffer> buf_bi = bias
        ? metal_get_or_create_cpu_buf(bias, (size_t)C_out * sizeof(float)) : nil;
    if (!buf_in || !buf_out || !buf_w || !buf_sc) {
        fprintf(stderr, "[metal] seqgemm_i8_matvec: buffer lookup failed\n"); abort();
    }
    struct { int x, y; } dims{C_in, C_out};
    int has_bias = (bias != nullptr) ? 1 : 0;
    LMDispatch d("seqgemm_i8_matvec_k");
    d.buf(buf_in,  in_off);
    d.buf(buf_w,   w_off);
    d.buf(buf_sc,  0);
    d.buf(buf_bi ? buf_bi : buf_sc, 0);   // dummy if no bias (kernel checks has_bias)
    d.buf(buf_out, out_off);
    d.bytes(&dims,     sizeof(dims));
    d.bytes(&has_bias, sizeof(int));
    // 128 threads per output channel (4 SIMD groups): full 128-byte cache-line utilization.
    d.tg_mem(4 * sizeof(float));
    d.run_tg(MTLSizeMake((NSUInteger)C_out, 1, 1), MTLSizeMake(128, 1, 1));
}

// INT8 batch GEMM for prefill (N>1).
// w_packed is NEON-packed Metal ptr; use metal_get_weight_buf() for row-major original.
static void metal_lm_seqgemm_int8(
    const float* input, const int8_t* w_packed,
    const float* w_scales, const int64_t* /*eff_zeros*/,
    const float* bias, int8_t* /*inp_scratch*/, float* /*req_scratch*/,
    float* output, int N, int C_in, int C_out, bool gelu, StreamHandle)
{
    NSUInteger in_off = 0, out_off = 0;
    id<MTLBuffer> buf_in  = metal_find_buf_containing(input,  &in_off);
    id<MTLBuffer> buf_out = metal_find_buf_containing(output, &out_off);
    // Get original row-major int8 weights from weight registry.
    id<MTLBuffer> buf_w = metal_get_weight_buf(w_packed);
    if (!buf_w) {
        // Fallback: if weight_i8_buf_ was uploaded directly, find via buf map.
        buf_w = metal_find_buf_containing(w_packed, nullptr);
    }
    id<MTLBuffer> buf_sc = metal_get_or_create_cpu_buf(w_scales, (size_t)C_out * sizeof(float));
    id<MTLBuffer> buf_bi = bias
        ? metal_get_or_create_cpu_buf(bias, (size_t)C_out * sizeof(float)) : nil;
    if (!buf_in || !buf_out || !buf_w || !buf_sc) {
        fprintf(stderr, "[metal] seqgemm_i8_batch: buffer lookup failed (w=%p wbuf=%p)\n",
                (void*)w_packed, (__bridge void*)buf_w); abort();
    }
    struct { int x, y, z, w; } dims{N, C_in, C_out, 0};
    int has_bias = (bias != nullptr) ? 1 : 0;
    {
        LMDispatch d("seqgemm_i8_batch_k");
        d.buf(buf_in,  in_off);
        d.buf(buf_w,   0);
        d.buf(buf_sc,  0);
        d.buf(buf_bi ? buf_bi : buf_sc, 0);
        d.buf(buf_out, out_off);
        d.bytes(&dims,     sizeof(dims));
        d.bytes(&has_bias, sizeof(int));
        d.run_tg(
            MTLSizeMake((NSUInteger)((C_out + 15) / 16), (NSUInteger)N, 1),
            MTLSizeMake(16, 1, 1));
    }
    if (gelu) {
        LMDispatch dg("gelu_inplace_k");
        dg.buf(buf_out, out_off);
        dg.run((NSUInteger)(N * C_out));
    }
}

// Full GPU causal attention: RoPE, KV cache write, attention compute.
static void metal_lm_causal_attn_core_fp32(
    const float* q_in, const float* k_in, const float* v_in,
    float* kv_cache, int cache_pos, int cache_len,
    float* output,
    int N, int num_q_heads, int num_kv_heads, int head_dim,
    int step_pos, float rope_theta, bool /*is_decode*/,
    float* /*cpu_scratch*/, StreamHandle)
{
    NSUInteger q_off=0, k_off=0, v_off=0, out_off=0;
    id<MTLBuffer> q_buf   = metal_find_buf_containing(q_in,  &q_off);
    id<MTLBuffer> k_buf   = metal_find_buf_containing(k_in,  &k_off);
    id<MTLBuffer> v_buf   = metal_find_buf_containing(v_in,  &v_off);
    id<MTLBuffer> out_buf = metal_find_buf_containing(output, &out_off);
    id<MTLBuffer> kvc_buf = metal_get_buf(kv_cache);  // allocated via MetalAllocator
    if (!q_buf || !k_buf || !v_buf || !out_buf || !kvc_buf) {
        fprintf(stderr, "[metal] causal_attn_core: buffer lookup failed "
                "(q=%p k=%p v=%p out=%p kvc=%p)\n",
                (void*)q_in,(void*)k_in,(void*)v_in,(void*)output,(void*)kv_cache);
        abort();
    }
    int nq = num_q_heads, nkv = num_kv_heads, hd = head_dim;
    int hd2 = hd / 2;
    NSUInteger total_kv = (NSUInteger)(cache_pos + N);

    // Allocate/grow scratch buffers.
    id<MTLBuffer> qrope_buf = lm_rope_scratch((size_t)N * nq * hd * sizeof(float));
    id<MTLBuffer> score_buf = lm_score_scratch((size_t)N * nq * total_kv * sizeof(float));
    if (!qrope_buf || !score_buf) {
        fprintf(stderr, "[metal] causal_attn_core: scratch alloc failed\n"); abort();
    }

    // ── 1. RoPE on Q+K; K→KV-cache K slot ───────────────────────
    {
        struct { int x,y,z,w; } dims{N, nq, nkv, hd};
        struct { int x,y; } pos{step_pos, cache_len};
        LMDispatch d("rope_qk_and_kvcache_k");
        d.buf(q_buf, q_off);
        d.buf(k_buf, k_off);
        d.buf(qrope_buf, 0);
        d.buf(kvc_buf, 0);
        d.bytes(&dims, sizeof(dims));
        d.bytes(&pos,  sizeof(pos));
        d.bytes(&rope_theta, sizeof(float));
        d.run((NSUInteger)N * (nq + nkv) * hd2);
    }
    // ── 2. V → KV-cache V slot ───────────────────────────────────
    {
        struct { int x,y,z,w; } dims{N, nkv, hd, 0};
        struct { int x,y; } pos{cache_pos, cache_len};
        LMDispatch d("kv_cache_v_write_k");
        d.buf(v_buf, v_off);
        d.buf(kvc_buf, 0);
        d.bytes(&dims, sizeof(dims));
        d.bytes(&pos,  sizeof(pos));
        d.run((NSUInteger)N * nkv * hd);
    }
    // ── 3. Attention ─────────────────────────────────────────────
    if (N == 1) {
        struct { int x,y,z,w; } dims{nq, nkv, hd, (int)total_kv};
        LMDispatch d("causal_attn_decode_k");
        d.buf(qrope_buf, 0);
        d.buf(kvc_buf,   0);
        d.buf(out_buf,   out_off);
        d.buf(score_buf, 0);
        d.bytes(&dims,      sizeof(dims));
        d.bytes(&cache_len, sizeof(int));
        d.run((NSUInteger)nq);
    } else {
        struct { int x,y,z,w; } dims{N, nq, nkv, hd};
        struct { int x,y; } pos{cache_pos, cache_len};
        LMDispatch d("causal_attn_prefill_k");
        d.buf(qrope_buf, 0);
        d.buf(kvc_buf,   0);
        d.buf(out_buf,   out_off);
        d.buf(score_buf, 0);
        d.bytes(&dims, sizeof(dims));
        d.bytes(&pos,  sizeof(pos));
        d.run((NSUInteger)N * nq);
    }
}

// ─────────────────────────────────────────────────────────────────
// BackendOps registration
// ─────────────────────────────────────────────────────────────────
static MetalAllocator s_metal_alloc;
static Allocator* metal_make_allocator() { return &s_metal_alloc; }

static BackendOps s_metal_ops = {
    .name           = "metal",
    .priority       = 10,
    .mem_space      = MemSpace::Metal,
    .make_allocator = metal_make_allocator,

    .pack_weights_int8 = metal_pack_weights_int8,
    .pack_weights_fp32 = metal_pack_weights_fp32,
    .free_packed_int8  = metal_free_packed_int8,
    .free_packed_fp32  = metal_free_packed_fp32,

    .gemm_int8         = metal_gemm_int8,
    .gemm_int8_int32   = metal_gemm_int8_int32,
    .sgemm_f32         = metal_sgemm_f32,
    .gemm_fp32_vec     = metal_gemm_fp32_vec,

    .conv2d_int8                = metal_conv2d_int8,
    .conv2d_winograd_nhwc_int8  = metal_conv2d_winograd_nhwc_int8,
    .conv2d_fp32_nhwc           = metal_conv2d_fp32_nhwc,
    .conv2d_winograd_nhwc_fp32  = metal_conv2d_winograd_nhwc_fp32,

    .seqgemm_fp32  = metal_seqgemm_fp32,
    .seqgemm_int8  = metal_lm_seqgemm_int8,

    .conv1d_int8_nhwc  = metal_conv1d_int8_nhwc,
    .conv1d_fp32_nhwc  = metal_conv1d_fp32,

    .conv_transpose1d_fp32_nhwc = metal_conv_transpose1d_fp32,

    .attention_flash_fp32 = metal_attention_flash_fp32,
    .attention_flash_int8 = metal_attention_flash_int8,

    .layernorm_fp32        = metal_layernorm_fp32,
    .ada_in1d_fp32         = metal_ada_in1d_fp32,
    .ada_layer_norm1d_fp32 = metal_ada_layer_norm1d_fp32,

    .maxpool_int8             = metal_maxpool_int8,
    .maxpool_int8_nhwc        = metal_maxpool_int8_nhwc,
    .maxpool_fp32_nhwc        = metal_maxpool_fp32_nhwc,
    .avgpool_global_int8      = metal_avgpool_global_int8,
    .avgpool_global_int8_nhwc = metal_avgpool_global_int8_nhwc,
    .avgpool_global_fp32_nhwc = metal_avgpool_global_fp32_nhwc,

    .add_fp32         = metal_add_fp32,
    .add_requant_int8 = metal_add_requant_int8,

    .relu_fp32       = metal_relu_fp32,
    .gelu_fp32       = metal_gelu_fp32,
    .sigmoid_fp32    = metal_sigmoid_fp32,
    .tanh_fp32       = metal_tanh_fp32,
    .leaky_relu_fp32 = metal_leaky_relu_fp32,
    .exp_fp32        = metal_exp_fp32,
    .sin_fp32        = metal_sin_fp32,

    .upsample_nearest1d_fp32  = metal_upsample_nearest1d_fp32,
    .upsample_bilinear2d_fp32 = metal_upsample_bilinear2d_fp32,

    .add_vectors_fp32   = metal_add_vectors_fp32,
    .bias_add_rows_fp32 = metal_bias_add_rows_fp32,
    .scale_fp32         = metal_scale_fp32,
    .sum_channels_fp32  = metal_sum_channels_fp32,
    .relu_inplace_fp32  = metal_relu_fp32,

    .embedding_lookup_fp32 = metal_embedding_lookup_fp32,
    .concat1d_fp32         = metal_concat1d_fp32,
    .slice_channels_fp32   = metal_slice_channels_fp32,

    .snake1d_fp32 = metal_snake1d_fp32,

    .lstm_fp32_metal             = metal_lstm_fp32,
    .stft_fp32_metal             = metal_stft_fp32,
    .istft_fp32_metal            = metal_istft_fp32,
    .reflection_pad1d_fp32_metal = metal_reflection_pad1d,
    .sine_gen_fp32_metal         = metal_sine_gen_fp32,

    // LM ops — all GPU, all batched into g_lm_enc (persistent encoder):
    .rmsnorm_fp32          = metal_lm_rmsnorm_fp32,
    .silu_fp32             = metal_lm_silu_fp32,
    .elemwise_mul_fp32     = metal_lm_elemwise_mul_fp32,
    .add_fp32              = metal_lm_add_fp32,
    .add_vectors_fp32      = metal_lm_add_vectors_fp32,
    .causal_attn_core_fp32 = metal_lm_causal_attn_core_fp32,
    .seqgemm_int8_matvec   = metal_lm_seqgemm_int8_matvec,
};

// ─────────────────────────────────────────────────────────────────
// Static initializer: runs before main().
// Loads the pre-compiled metallib from the installed extension directory, then
// falls back to source layouts used during local development.
// ─────────────────────────────────────────────────────────────────

// Fallback shader source (kept in sync with kernels.metal).
// Used only when metal_kernels.metallib is not found next to the binary.
static const char* kMetalShaderFallback = R"MSL(
#include <metal_stdlib>
using namespace metal;
// Minimal stub — build with CMake to get the full library.
kernel void relu_inplace_k(device float* d [[buffer(0)]], uint id [[thread_position_in_grid]])
{ d[id] = max(d[id], 0.f); }
)MSL";

static const bool s_registered = []() -> bool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
        NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
        if ([all count] > 0) dev = all[0];
    }
    if (!dev) {
        fprintf(stderr, "[metal] No Metal device found — backend not registered\n");
        return false;
    }

    id<MTLCommandQueue> q = [dev newCommandQueue];
    if (!q) {
        fprintf(stderr, "[metal] Failed to create MTLCommandQueue\n");
        return false;
    }

    // ── Try to load pre-compiled metallib ────────────────────────
    NSString* module_dir = nil;
    Dl_info dylib_info = {};
    if (dladdr(static_cast<const void*>(&s_metal_ops), &dylib_info)
        && dylib_info.dli_fname) {
        module_dir = [[[NSString stringWithUTF8String:dylib_info.dli_fname]
            stringByDeletingLastPathComponent] stringByResolvingSymlinksInPath];
    }

    NSString* exe_dir = nil;
    char exe_path[4096] = {};
    uint32_t exe_sz = sizeof(exe_path);
    if (_NSGetExecutablePath(exe_path, &exe_sz) == 0) {
        exe_dir = [[[NSString stringWithUTF8String:exe_path]
            stringByDeletingLastPathComponent] stringByResolvingSymlinksInPath];
    }

    NSMutableArray<NSString*>* lib_candidates = [NSMutableArray array];
    if (module_dir) {
        [lib_candidates addObject:[module_dir stringByAppendingPathComponent:@"metal_kernels.metallib"]];
    }
    if (exe_dir) {
        [lib_candidates addObject:[exe_dir stringByAppendingPathComponent:@"metal_kernels.metallib"]];
    }

    NSError* err = nil;
    id<MTLLibrary> lib = nil;

    for (NSString* lib_path in lib_candidates) {
        if (![[NSFileManager defaultManager] fileExistsAtPath:lib_path]) {
            fprintf(stderr, "[metal] metallib not found at %s\n", [lib_path UTF8String]);
            continue;
        }

        lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:lib_path] error:&err];
        if (lib) {
            fprintf(stderr, "[metal] Loaded metallib: %s\n", [lib_path UTF8String]);
            break;
        } else {
            fprintf(stderr, "[metal] Failed to load %s: %s\n",
                    [lib_path UTF8String], [[err localizedDescription] UTF8String]);
            err = nil;
        }
    }

    if (!lib) {
        fprintf(stderr, "[metal] Building kernels from source (run cmake to generate metallib)\n");
        NSMutableArray<NSString*>* src_candidates = [NSMutableArray array];
        if (module_dir) {
            [src_candidates addObject:[module_dir stringByAppendingPathComponent:@"../src/backend/metal/shaders/kernels.metal"]];
            [src_candidates addObject:[module_dir stringByAppendingPathComponent:@"../../src/backend/metal/shaders/kernels.metal"]];
        }
        if (exe_dir) {
            [src_candidates addObject:[exe_dir stringByAppendingPathComponent:@"../src/backend/metal/shaders/kernels.metal"]];
            [src_candidates addObject:[exe_dir stringByAppendingPathComponent:@"../../src/backend/metal/shaders/kernels.metal"]];
        }
        NSString* src_str = nil;
        for (NSString* cand in src_candidates) {
            NSString* resolved = [cand stringByResolvingSymlinksInPath];
            if ([[NSFileManager defaultManager] fileExistsAtPath:resolved]) {
                src_str = [NSString stringWithContentsOfFile:resolved
                                                    encoding:NSUTF8StringEncoding
                                                       error:nil];
                if (src_str) {
                    fprintf(stderr, "[metal] Compiling shaders from %s\n",
                            [resolved UTF8String]);
                    break;
                }
            }
        }
        if (!src_str) {
            src_str = [NSString stringWithUTF8String:kMetalShaderFallback];
            fprintf(stderr, "[metal] WARNING: using minimal fallback shader source\n");
        }
        MTLCompileOptions* opts = [MTLCompileOptions new];
        lib = [dev newLibraryWithSource:src_str options:opts error:&err];
        if (!lib) {
            fprintf(stderr, "[metal] Shader compilation failed:\n%s\n",
                    [[err localizedDescription] UTF8String]);
            return false;
        }
    }

    g_metal_device  = dev;
    g_metal_queue   = q;
    g_metal_library = lib;
    g_pso_cache     = [NSMutableDictionary new];

    register_backend(&s_metal_ops);
    fprintf(stderr, "[metal] Backend registered (device: %s)\n",
            [[dev name] UTF8String]);
    return true;
}();
