// AMD RDNA3 / RDNA4 matrix instructions (WMMA, wave32): bf16 x bf16 -> fp32, one 16 x 16 x 16 tile per wave.
//
// Used only by the HIP build's matrix-instruction kernels (src/hip/); nothing here is compiled into a CUDA build.
//
// Instruction (one per wave, all 32 lanes):
//   gfx11 (RDNA3, RX 7000):  v_wmma_f32_16x16x16_bf16   __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32
//   gfx12 (RDNA4, RX 9000):  v_wmma_f32_16x16x16_bf16   __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12
//   D[16][16] = A[16][16] * B[16][16] + C[16][16]; A, B bf16, C, D fp32.
//
// Operand layout per lane l (0..31) of the wave. "i" is the row of A (M) or the column of B (N), so both operands are
// read the same way from a k-contiguous tile (A stored [m][k], B stored [n][k], i.e. a weight matrix W[n][k]):
//   gfx11 A / B: 8 VGPRs = 16 bf16: lane l holds i = l % 16, k = 0..15 (VGPR j = k 2j | k 2j+1 << 16). Lanes 16..31
//                must hold the same values as lanes 0..15 (the hardware reads both halves).
//   gfx12 A / B: 4 VGPRs = 8 bf16:  lane l holds i = l % 16, k = 8 (l / 16) + 0..7 (VGPR j = k 2j | k 2j+1 << 16).
//   C / D (both): 8 VGPRs fp32: element v (0..7) of lane l is column n = l % 16 of row
//                gfx11: m = 2 v + l / 16        gfx12: m = v + 8 (l / 16)
//
// Numerics: bf16 x bf16 products are exact in fp32. The hardware adds the 16 products of a k-block and C in its own
// order and internal precision, which is not the sequential fmaf chain of the software path (kt2n/cuda_compat.cuh), so
// results differ from it in the last bits; tests/gpu/hip_wmma.cu measures by how much. For a fixed k-block order,
// a row's result depends only on that row of A and that column of B (never on other rows or the tile position), so
// kernels built on it stay batch invariant.
#pragma once

#include <stdint.h>

#include "kt2n/gpu_arch.h"

// KT2N_WMMA_ARCH: 11 (gfx11), 12 (gfx12) in a HIP device pass for those GPUs; 0 everywhere else (host pass, CUDA,
// other AMD targets). Kernels compile their WMMA body only when it is non-zero. The host gate (hip_wmma_arch in
// kt2n/cuda_caps.h) uses the same target list: change both together.
#if defined(KT2N_HIP) && defined(__HIP_DEVICE_COMPILE__) &&                                                    \
    (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) ||            \
     defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__))
#define KT2N_WMMA_ARCH 11
#elif defined(KT2N_HIP) && defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1200__) || defined(__gfx1201__))
#define KT2N_WMMA_ARCH 12
#else
#define KT2N_WMMA_ARCH 0
#endif

namespace kt2n::wmma {

// 32-bit words of one A / B operand per lane.
#if KT2N_WMMA_ARCH == 12
inline constexpr int kWords = 4;
#else
inline constexpr int kWords = 8;
#endif

typedef uint32_t words_t __attribute__((ext_vector_type(kWords)));
typedef float acc_t __attribute__((ext_vector_type(8)));
#if KT2N_WMMA_ARCH == 12
typedef short ab_t __attribute__((ext_vector_type(8)));
#else
typedef short ab_t __attribute__((ext_vector_type(16)));
#endif

#if defined(__HIP_DEVICE_COMPILE__)
__device__ __forceinline__ int lane() { return static_cast<int>(__lane_id()); }
#else
inline int lane() { return 0; }
#endif

// d = a * b + c on the matrix unit. On a target without WMMA the kernel traps (host dispatch never launches it there).
__device__ __forceinline__ acc_t mma(const words_t& a, const words_t& b, const acc_t& c) {
#if KT2N_WMMA_ARCH == 11
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(__builtin_bit_cast(ab_t, a), __builtin_bit_cast(ab_t, b), c);
#elif KT2N_WMMA_ARCH == 12
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(ab_t, a), __builtin_bit_cast(ab_t, b),
                                                           c);
#else
  (void)a;
  (void)b;
#if defined(__HIP_DEVICE_COMPILE__)
  __builtin_trap();
#endif
  return c;
#endif
}

__device__ __forceinline__ acc_t zero_acc() {
  acc_t z;
#pragma unroll
  for (int v = 0; v < 8; ++v) z[v] = 0.f;
  return z;
}

// Index i (row of A / column of B) this lane supplies, and the first k of its operand words (0 or 8 on gfx12).
__device__ __forceinline__ int op_index(int l) { return l & 15; }
__device__ __forceinline__ int op_k0(int l) {
#if KT2N_WMMA_ARCH == 12
  return 8 * (l >> 4);
#else
  (void)l;
  return 0;
#endif
}
// Accumulator element v of lane l: row (M) and column (N) inside the 16 x 16 tile.
__device__ __forceinline__ int acc_row(int v, int l) {
#if KT2N_WMMA_ARCH == 12
  return v + 8 * (l >> 4);
#else
  return 2 * v + (l >> 4);
#endif
}
__device__ __forceinline__ int acc_col(int l) { return l & 15; }

// Operand words from a 16-byte-aligned run of bf16 holding this lane's k values in order: k = op_k0 .. op_k0 + 15
// (gfx11) or op_k0 .. op_k0 + 7 (gfx12), i.e. p = tile + op_index(l) * ld + op_k0(l) of a k-contiguous tile.
__device__ __forceinline__ words_t load_run(const void* p) {
  const uint4* q = static_cast<const uint4*>(p);
  const uint4 v0 = q[0];
  words_t w;
  w[0] = v0.x;
  w[1] = v0.y;
  w[2] = v0.z;
  w[3] = v0.w;
#if KT2N_WMMA_ARCH != 12
  const uint4 v1 = q[1];
  w[4] = v1.x;
  w[5] = v1.y;
  w[6] = v1.z;
  w[7] = v1.w;
#endif
  return w;
}
// Same, zero when `valid` is false (rows past the end are never read).
__device__ __forceinline__ words_t load_run_or_zero(const void* p, bool valid) {
  if (valid) return load_run(p);
  words_t w;
#pragma unroll
  for (int j = 0; j < kWords; ++j) w[j] = 0u;
  return w;
}

// Operand words of a 16 x 16 tile whose k index is the slow one (element (i, k) at p[k * ld + i], e.g. V[key][dim]
// as B with k = key): one 16-bit load per element, for tiles in LDS.
__device__ __forceinline__ words_t load_strided(const uint16_t* tile, int ld, int l) {
  const int i = op_index(l), k0 = op_k0(l);
  words_t w;
#pragma unroll
  for (int j = 0; j < kWords; ++j) {
    const uint32_t lo = tile[(k0 + 2 * j) * ld + i];
    const uint32_t hi = tile[(k0 + 2 * j + 1) * ld + i];
    w[j] = lo | (hi << 16);
  }
  return w;
}

}  // namespace kt2n::wmma
