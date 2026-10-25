// Portable software versions of the SM 8.0 / 7.5 / 7.0 PTX instructions the engine's kernels use,
// for Turing (SM 7.5) and Pascal (SM 6.1).
//
// Every function here compiles for every arch, so a test on an SM 8.0+ GPU can compare it with the
// hardware instruction. Kernel wrappers call it only on the archs that lack the instruction:
//
//   #if KT2N_SM_ASM(800)          (kt2n/gpu_arch.h: SM 8.0+ device passes and the CUDA host pass)
//     <the original asm, unchanged>
//   #else
//     kt2n::compat::...
//   #endif
//
// (750 for ldmatrix, 700 for ld_nc_v4). SM 8.0+ code therefore never sees this header's code. KT2N_PORTABLE
// builds (always for HIP / AMD) take these software forms on every architecture.
//
// Precision: the mma emulations form each product exactly (bf16 x bf16 and tf32 x tf32 products fit in
// fp32's 24-bit significand) and accumulate in fp32 with fmaf, k ascending, starting from c. That is the
// precision class of the hardware instruction (exact products, fp32 accumulation); only the order and
// rounding of the fp32 additions may differ from the tensor core. No fp16 and no lower precision anywhere.
//
// Warp-collective functions (mma_*, ldmatrix_*) must be called by all 32 lanes of a converged warp, as
// the .sync.aligned hardware instructions require.
//
// No __device__ / __constant__ variables and no kernels: including this header adds no code to a TU
// that does not call it.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "kt2n/gpu_arch.h"

namespace kt2n::compat {

__device__ __forceinline__ int lane_id() {
#if defined(KT2N_HIP)
  return static_cast<int>(__lane_id() & 31u);  // lane within CUDA's 32-lane warp (RDNA: the whole wave32)
#else
  int lane;
  asm("mov.u32 %0, %%laneid;" : "=r"(lane));
  return lane;
#endif
}

__device__ __forceinline__ uint32_t shfl(uint32_t v, int src_lane) { return __shfl_sync(0xffffffffu, v, src_lane); }

// KT2N_QUAD_SHFL(v, arow, j) = shfl(v, arow + j) with arow = lane & ~3: every lane takes the value of lane j of its own
// group of 4 (the A-operand pattern of the mma emulations). On AMD it is one DPP quad permutation (an operand modifier
// of a VALU move) instead of an LDS-unit permute (ds_bpermute); the same value moves, so results are unchanged. On
// CUDA the macro is the shfl call itself, so the CUDA code is unchanged.
#if defined(KT2N_HIP)
__device__ __forceinline__ uint32_t quad_shfl(uint32_t v, int j) {
  switch (j & 3) {
    case 0: return static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(v), 0x00, 0xf, 0xf, false));
    case 1: return static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(v), 0x55, 0xf, 0xf, false));
    case 2: return static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(v), 0xaa, 0xf, 0xf, false));
    default: return static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(v), 0xff, 0xf, 0xf, false));
  }
}
#define KT2N_QUAD_SHFL(v, arow, j) kt2n::compat::quad_shfl((v), ((void)(arow), (j)))
#else
#define KT2N_QUAD_SHFL(v, arow, j) shfl(v, arow + j)
#endif

// bf16 halves of a packed .b32 (low 16 bits = lower index) as exact fp32 values.
__device__ __forceinline__ float bf16_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

// tf32 operand as the tensor core reads it: the low 13 significand bits are not used.
__device__ __forceinline__ float tf32_val(uint32_t v) { return __uint_as_float(v & 0xffffe000u); }

// ---------------------------------------------------------------------------------------------------------
// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32  d = a * b + c (d and c share registers).
// Fragments (g = lane / 4, t = lane % 4; packed pairs hold columns/rows 2t and 2t + 1):
//   a0: A[g][2t..]      a1: A[g+8][2t..]      a2: A[g][8+2t..]      a3: A[g+8][8+2t..]
//   b0: B[2t..][g]      b1: B[8+2t..][g]
//   d0: D[g][2t]  d1: D[g][2t+1]  d2: D[g+8][2t]  d3: D[g+8][2t+1]
__device__ __forceinline__ void mma_m16n8k16_bf16(float& d0, float& d1, float& d2, float& d3, uint32_t a0,
                                                  uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                                  uint32_t b1) {
  const int lane = lane_id();
  const int t = lane & 3;
  const int arow = lane & ~3;      // lanes arow + j hold rows g and g + 8, k pair j (j = 0..3)
  const int bcol0 = (2 * t) * 4;   // lanes bcol0 + j hold column 2t, k pair j
  const int bcol1 = bcol0 + 4;     // column 2t + 1
  float s0 = d0, s1 = d1, s2 = d2, s3 = d3;
#pragma unroll
  for (int h = 0; h < 2; ++h) {  // k 0..7 (a0/a1, b0), then k 8..15 (a2/a3, b1)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t x = KT2N_QUAD_SHFL(h ? a2 : a0, arow, j);   // A[g][8h + 2j], A[g][8h + 2j + 1]
      const uint32_t y = KT2N_QUAD_SHFL(h ? a3 : a1, arow, j);   // A[g+8][...]
      const uint32_t p = shfl(h ? b1 : b0, bcol0 + j);  // B[8h + 2j..][2t]
      const uint32_t q = shfl(h ? b1 : b0, bcol1 + j);  // B[8h + 2j..][2t+1]
      s0 = fmaf(bf16_lo(x), bf16_lo(p), s0);
      s1 = fmaf(bf16_lo(x), bf16_lo(q), s1);
      s2 = fmaf(bf16_lo(y), bf16_lo(p), s2);
      s3 = fmaf(bf16_lo(y), bf16_lo(q), s3);
      s0 = fmaf(bf16_hi(x), bf16_hi(p), s0);
      s1 = fmaf(bf16_hi(x), bf16_hi(q), s1);
      s2 = fmaf(bf16_hi(y), bf16_hi(p), s2);
      s3 = fmaf(bf16_hi(y), bf16_hi(q), s3);
    }
  }
  d0 = s0;
  d1 = s1;
  d2 = s2;
  d3 = s3;
}
__device__ __forceinline__ void mma_m16n8k16_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                                  uint32_t a3, uint32_t b0, uint32_t b1) {
  mma_m16n8k16_bf16(d[0], d[1], d[2], d[3], a0, a1, a2, a3, b0, b1);
}

// Out-of-line form (same arithmetic): the inline emulation is ~100 instructions, and kernels that unroll
// hundreds of mma calls can make ptxas take tens of minutes for SM 6.1 / 7.5. One copy per translation unit;
// float4 in and out keeps the accumulators in registers across the call.
inline __device__ __noinline__ float4 mma_m16n8k16_bf16_ool(float4 c, uint32_t a0, uint32_t a1, uint32_t a2,
                                                            uint32_t a3, uint32_t b0, uint32_t b1) {
  mma_m16n8k16_bf16(c.x, c.y, c.z, c.w, a0, a1, a2, a3, b0, b1);
  return c;
}

// ---------------------------------------------------------------------------------------------------------
// mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32  d = a * b + c. Operands are tf32 in .b32 registers
// (normally from cvt_rna_tf32); the low 13 bits are ignored, as the tensor core ignores them.
//   a0: A[g][t]   a1: A[g+8][t]   a2: A[g][t+4]   a3: A[g+8][t+4]
//   b0: B[t][g]   b1: B[t+4][g]
//   d0..d3 as for m16n8k16.
__device__ __forceinline__ void mma_m16n8k8_tf32(float& d0, float& d1, float& d2, float& d3, uint32_t a0,
                                                 uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                                 uint32_t b1) {
  const int lane = lane_id();
  const int t = lane & 3;
  const int arow = lane & ~3;
  const int bcol0 = (2 * t) * 4;
  const int bcol1 = bcol0 + 4;
  float s0 = d0, s1 = d1, s2 = d2, s3 = d3;
#pragma unroll
  for (int h = 0; h < 2; ++h) {  // k 0..3 (a0/a1, b0), then k 4..7 (a2/a3, b1)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float x = tf32_val(KT2N_QUAD_SHFL(h ? a2 : a0, arow, j));   // A[g][4h + j]
      const float y = tf32_val(KT2N_QUAD_SHFL(h ? a3 : a1, arow, j));   // A[g+8][4h + j]
      const float p = tf32_val(shfl(h ? b1 : b0, bcol0 + j));  // B[4h + j][2t]
      const float q = tf32_val(shfl(h ? b1 : b0, bcol1 + j));  // B[4h + j][2t+1]
      s0 = fmaf(x, p, s0);
      s1 = fmaf(x, q, s1);
      s2 = fmaf(y, p, s2);
      s3 = fmaf(y, q, s3);
    }
  }
  d0 = s0;
  d1 = s1;
  d2 = s2;
  d3 = s3;
}
__device__ __forceinline__ void mma_m16n8k8_tf32(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                                 uint32_t a3, uint32_t b0, uint32_t b1) {
  mma_m16n8k8_tf32(d[0], d[1], d[2], d[3], a0, a1, a2, a3, b0, b1);
}

inline __device__ __noinline__ float4 mma_m16n8k8_tf32_ool(float4 c, uint32_t a0, uint32_t a1, uint32_t a2,
                                                           uint32_t a3, uint32_t b0, uint32_t b1) {
  mma_m16n8k8_tf32(c.x, c.y, c.z, c.w, a0, a1, a2, a3, b0, b1);
  return c;
}

// ---------------------------------------------------------------------------------------------------------
// cvt.rna.tf32.f32: round to 10 significand bits, ties away from zero (the result keeps fp32 layout with
// the low 13 bits zero). Adding half an ulp to the magnitude bits and truncating is round-half-away for
// sign-magnitude; a carry into the exponent gives the next binade or inf, as rounding should.
// Inf and NaN are truncated instead (the hardware does this too, so a NaN whose payload sits only in the
// low 13 bits becomes inf); bit-exact with the SM 12.0 instruction over all 2^32 inputs.
__device__ __forceinline__ uint32_t cvt_rna_tf32(float x) {
  const uint32_t u = __float_as_uint(x);
  if ((u & 0x7f800000u) == 0x7f800000u) return u & 0xffffe000u;
  return (u + 0x1000u) & 0xffffe000u;
}

// ---------------------------------------------------------------------------------------------------------
// cp.async.{cg,ca}.shared.global [smem], [gmem], 16, src_bytes: copy src_bytes (0..16) bytes and zero the
// rest of the 16-byte destination. The source is not read when src_bytes == 0. Done synchronously: kernels
// that are race-free under cp.async's rules (destination untouched until wait_group + a barrier) see the
// same data, since completing the copy at issue time is one of the executions cp.async allows.
// Both addresses are 16-byte aligned, as cp.async requires.
template <bool kCacheAll>
__device__ __forceinline__ void cp_async16_sync(void* smem, const void* gmem, int src_bytes) {
  uint4 v = make_uint4(0u, 0u, 0u, 0u);
  if (src_bytes >= 16) {
    const uint4* g = static_cast<const uint4*>(gmem);
    v = kCacheAll ? __ldca(g) : __ldcg(g);
  } else if (src_bytes > 0) {
    const unsigned char* g = static_cast<const unsigned char*>(gmem);
    uint32_t w[4] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      if (i < src_bytes) w[i >> 2] |= uint32_t(kCacheAll ? __ldca(g + i) : __ldcg(g + i)) << (8 * (i & 3));
    }
    v = make_uint4(w[0], w[1], w[2], w[3]);
  }
  *static_cast<uint4*>(smem) = v;
}
__device__ __forceinline__ void cp_async_cg16(void* smem, const void* gmem, int src_bytes) {
  cp_async16_sync<false>(smem, gmem, src_bytes);
}
__device__ __forceinline__ void cp_async_ca16(void* smem, const void* gmem, int src_bytes) {
  cp_async16_sync<true>(smem, gmem, src_bytes);
}
// Copies complete at issue, so commit and wait only keep the compiler from moving memory accesses across
// them (the hardware versions carry the same "memory" clobber).
__device__ __forceinline__ void cp_async_commit() { asm volatile("" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("" ::: "memory");
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("" ::: "memory"); }

// ---------------------------------------------------------------------------------------------------------
// ldmatrix.sync.aligned.m8n8.{x1,x2,x4}[.trans].shared.b16. Lanes 8i..8i+7 give the row addresses of
// matrix i (16-byte rows in shared memory, 16-byte aligned); only lanes below 8 x count are read.
// Lane l receives in r[i]: row l / 4, columns 2(l % 4) and 2(l % 4) + 1 of matrix i (low half = lower
// column); with .trans, column l / 4, rows 2(l % 4) and 2(l % 4) + 1.
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint32_t lds_u32(uint32_t addr) {
  return *static_cast<const uint32_t*>(__cvta_shared_to_generic(addr));
}
__device__ __forceinline__ uint32_t lds_u16(uint32_t addr) {
  return *static_cast<const unsigned short*>(__cvta_shared_to_generic(addr));
}
template <int kCount>
__device__ __forceinline__ void ldmatrix_rows(uint32_t* r, const void* smem) {
  const uint32_t a = smem_addr(smem);
  const int lane = lane_id();
#pragma unroll
  for (int i = 0; i < kCount; ++i) {
    const uint32_t row = shfl(a, 8 * i + (lane >> 2));
    r[i] = lds_u32(row + 4u * uint32_t(lane & 3));
  }
}
template <int kCount>
__device__ __forceinline__ void ldmatrix_rows_trans(uint32_t* r, const void* smem) {
  const uint32_t a = smem_addr(smem);
  const int lane = lane_id();
  const uint32_t col = 2u * uint32_t(lane >> 2);
#pragma unroll
  for (int i = 0; i < kCount; ++i) {
    const uint32_t row0 = shfl(a, 8 * i + 2 * (lane & 3));
    const uint32_t row1 = shfl(a, 8 * i + 2 * (lane & 3) + 1);
    r[i] = lds_u16(row0 + col) | (lds_u16(row1 + col) << 16);
  }
}

__device__ __forceinline__ void ldmatrix_x1(uint32_t (&r)[1], const void* smem) { ldmatrix_rows<1>(r, smem); }
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void* smem) { ldmatrix_rows<2>(r, smem); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem) { ldmatrix_rows<4>(r, smem); }
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* smem) {
  ldmatrix_rows_trans<4>(r, smem);
}
// Scalar-reference forms for callers that keep fragments in separate variables or behind a pointer.
__device__ __forceinline__ void ldmatrix_x1(uint32_t& r0, const void* smem) {
  uint32_t r[1];
  ldmatrix_rows<1>(r, smem);
  r0 = r[0];
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t& r0, uint32_t& r1, const void* smem) {
  uint32_t r[2];
  ldmatrix_rows<2>(r, smem);
  r0 = r[0];
  r1 = r[1];
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                            const void* smem) {
  uint32_t r[4];
  ldmatrix_rows<4>(r, smem);
  r0 = r[0];
  r1 = r[1];
  r2 = r[2];
  r3 = r[3];
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                                  const void* smem) {
  uint32_t r[4];
  ldmatrix_rows_trans<4>(r, smem);
  r0 = r[0];
  r1 = r[1];
  r2 = r[2];
  r3 = r[3];
}

// ---------------------------------------------------------------------------------------------------------
// ld.global.nc.L1::no_allocate.v4.u32 without the L1 eviction hint (the hint needs SM 7.0): same
// non-coherent read-only path, so the same rule applies (only for data constant during the kernel).
__device__ __forceinline__ uint4 ld_nc_v4(const uint4* p) {
#if defined(KT2N_HIP)
  return *p;  // AMD: a plain global load (no separate read-only path)
#else
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
#endif
}

}  // namespace kt2n::compat
