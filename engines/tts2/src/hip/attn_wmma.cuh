// Decode attention math on the AMD matrix unit (RDNA3 / RDNA4 WMMA): drop-in replacements for the Q.K and P.V steps of
// the decode-step attention kernel (src/cuda/kernels/step_o1.cu attn_kernel), used there only on HIP gfx11 / gfx12 in
// the default mode KT2N_AMD_WMMA_DECODE=2. Include after attn_common.cuh (kt2n_attn namespace).
//
// Geometry of one CTA (unchanged): a (row, K/V head, key split); 4 warps; per 64-key chunk warp w scores keys
// 16w .. 16w+15 and accumulates output dims 32w .. 32w+31. The two query heads of the K/V head are tile rows 0 and 8
// (as in the mma.sync version); every other row is zero.
//   Q.K: one 16 x 16 WMMA tile (rows = query rows, columns = the warp's 16 keys) chained over the 8 16-dim blocks,
//        then handed to the unchanged softmax code as the two m16n8 accumulator fragments it expects (through a
//        per-warp LDS scratch: written in the WMMA layout, read in the mma.sync layout).
//   P.V: P from the CTA's probability array sP (rows 0 / 8, split into bf16 hi + mid + lo exactly as make_pfrag
//        does), V from the swizzled V tile as B ([key][dim]: the k index is the row, so a strided read), 2 WMMA tiles
//        per warp (32 dims), chained per 16-key step hi, mid, lo like pv_step; the row sums l likewise with B = 1.
//        The O / l accumulators stay in the WMMA layout over the chunks (the online-softmax rescale is per row:
//        rows 0 and 8 by a0 / a1) and go back to the m16n8 layout once, after the last chunk.
// Not bitwise equal to the software path (WMMA's own addition order; see kt2n/hip/wmma.cuh).
#pragma once

#include "kt2n/hip/wmma.cuh"
#include "kt2n/hip/wmma_kernels.h"

namespace kt2n_attn::wmma_dec {

namespace wm = kt2n::wmma;

// KT2N_AMD_WMMA_DECODE=2 on a WMMA GPU: decode attention (the decode step's and the attention library's decode-rows
// kernel, which the prefill then uses for every token) on the matrix unit. Host side; env read once.
inline bool enabled() { return kt2n_wmma_decode_mode() == 2; }

// Per-warp LDS scratch for the layout hand-offs: 16 rows x 32 floats.
constexpr int kScratch = 16 * 32;

__device__ __forceinline__ void wave_sync() {
#if KT2N_WMMA_ARCH
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#endif
}

// A operand of the 16 query rows for dims 16 ks .. 16 ks + 15: row 0 = head 0 (q0), row 8 = head 1 (q1), else zero.
__device__ __forceinline__ wm::words_t q_operand(const __nv_bfloat16* q0, const __nv_bfloat16* q1, int ks, int lane) {
  const int i = wm::op_index(lane), k0 = wm::op_k0(lane);
  const __nv_bfloat16* src = i == 0 ? q0 : q1;
  return wm::load_run_or_zero(src + 16 * ks + k0, i == 0 || i == 8);
}

// S for keys 16w .. 16w+15 of the chunk in sK (swizzled 64 x 16 segments) -> s0 (keys 16w + 0..7), s1 (+ 8..15) in
// the mma.sync accumulator layout (lane 4g + t: rows g, g + 8; keys 2t, 2t + 1).
__device__ __forceinline__ void qk16(FragC& s0, FragC& s1, const __nv_bfloat16* q0, const __nv_bfloat16* q1,
                                     const uint4* sK, int warp, int lane, float* scratch) {
#if KT2N_WMMA_ARCH
  const int i = wm::op_index(lane), k0 = wm::op_k0(lane);
  const int key = 16 * warp + i;
  wm::acc_t acc = wm::zero_acc();
#pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    const wm::words_t a = q_operand(q0, q1, ks, lane);
    wm::words_t b;
    const int seg = 2 * ks + (k0 >> 3);  // 8 dims per 16-byte segment
    const uint4 v0 = sK[kswz(key, seg)];
    b[0] = v0.x;
    b[1] = v0.y;
    b[2] = v0.z;
    b[3] = v0.w;
#if KT2N_WMMA_ARCH == 11
    const uint4 v1 = sK[kswz(key, seg + 1)];
    b[4] = v1.x;
    b[5] = v1.y;
    b[6] = v1.z;
    b[7] = v1.w;
#endif
    acc = wm::mma(a, b, acc);
  }
#pragma unroll
  for (int v = 0; v < 8; ++v) scratch[wm::acc_row(v, lane) * 16 + wm::acc_col(lane)] = acc[v];
  wave_sync();
  const int g = lane >> 2, t = lane & 3;
  s0.x[0] = scratch[g * 16 + 2 * t];
  s0.x[1] = scratch[g * 16 + 2 * t + 1];
  s0.x[2] = scratch[(g + 8) * 16 + 2 * t];
  s0.x[3] = scratch[(g + 8) * 16 + 2 * t + 1];
  s1.x[0] = scratch[g * 16 + 8 + 2 * t];
  s1.x[1] = scratch[g * 16 + 8 + 2 * t + 1];
  s1.x[2] = scratch[(g + 8) * 16 + 8 + 2 * t];
  s1.x[3] = scratch[(g + 8) * 16 + 8 + 2 * t + 1];
  wave_sync();  // the scratch is rewritten by the next call
#else
  (void)s0, (void)s1, (void)q0, (void)q1, (void)sK, (void)warp, (void)lane, (void)scratch;
#endif
}

struct Acc {
  wm::acc_t o[2];  // dims 32w + 16 nt + column
  wm::acc_t l;     // row sums (every column)
};

__device__ __forceinline__ void zero(Acc& a) {
  a.o[0] = wm::zero_acc();
  a.o[1] = wm::zero_acc();
  a.l = wm::zero_acc();
}

// Online-softmax rescale: rows 0 (head 0) by a0, 8 (head 1) by a1; the zero rows stay zero.
__device__ __forceinline__ void scale(Acc& a, float a0, float a1, int lane) {
#pragma unroll
  for (int v = 0; v < 8; ++v) {
    const int r = wm::acc_row(v, lane);
    const float f = r == 0 ? a0 : r == 8 ? a1 : 1.f;
    a.o[0][v] = __fmul_rn(a.o[0][v], f);
    a.o[1][v] = __fmul_rn(a.o[1][v], f);
    a.l[v] = __fmul_rn(a.l[v], f);
  }
}

// One chunk of O += P . V and l += P . 1: key steps ks = 0..3 (16 keys each), per step hi, mid, lo.
__device__ __forceinline__ void pv_chunk(Acc& a, const float (*sP)[CHUNK], const uint4* sV, int warp, int lane) {
#if KT2N_WMMA_ARCH
  const int i = wm::op_index(lane), k0 = wm::op_k0(lane);
  const uint16_t* sV16 = reinterpret_cast<const uint16_t*>(sV);
  wm::words_t ones;
#pragma unroll
  for (int j = 0; j < wm::kWords; ++j) ones[j] = 0x3F803F80u;
#pragma unroll 1
  for (int ks = 0; ks < 4; ++ks) {
    // A: this lane's row of P for keys 16 ks + k0 .. (row 0 = head 0, row 8 = head 1, else 0), split in three.
    wm::words_t ph, pm, pl;
#pragma unroll
    for (int j = 0; j < wm::kWords; ++j) {
      float p0 = 0.f, p1 = 0.f;
      if (i == 0 || i == 8) {
        const float* row = sP[i == 0 ? 0 : 1] + 16 * ks + k0 + 2 * j;
        p0 = row[0];
        p1 = row[1];
      }
      const Split3 x = split3(p0), y = split3(p1);
      ph[j] = pack_bf16x2(x.hi, y.hi);
      pm[j] = pack_bf16x2(x.mid, y.mid);
      pl[j] = pack_bf16x2(x.lo, y.lo);
    }
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      // B: column = dim 32w + 16 nt + i, k = key 16 ks + k0 + 0.. (rows of the swizzled V tile).
      const int dim = 32 * warp + 16 * nt + i;
      wm::words_t b;
#pragma unroll
      for (int j = 0; j < wm::kWords; ++j) {
        const int key = 16 * ks + k0 + 2 * j;
        const uint32_t lo = sV16[kswz(key, dim >> 3) * 8 + (dim & 7)];
        const uint32_t hi = sV16[kswz(key + 1, dim >> 3) * 8 + (dim & 7)];
        b[j] = lo | (hi << 16);
      }
      a.o[nt] = wm::mma(ph, b, a.o[nt]);
      a.o[nt] = wm::mma(pm, b, a.o[nt]);
      a.o[nt] = wm::mma(pl, b, a.o[nt]);
    }
    a.l = wm::mma(ph, ones, a.l);
    a.l = wm::mma(pm, ones, a.l);
    a.l = wm::mma(pl, ones, a.l);
  }
#else
  (void)a, (void)sP, (void)sV, (void)warp, (void)lane;
#endif
}

// O / l back to the m16n8 accumulator layout of the epilogue: o[i] = dims 32w + 8i .., lf = row sums.
__device__ __forceinline__ void to_frags(const Acc& a, FragC (&o)[4], FragC& lf, int lane, float* scratch) {
#if KT2N_WMMA_ARCH
#pragma unroll
  for (int nt = 0; nt < 2; ++nt)
#pragma unroll
    for (int v = 0; v < 8; ++v) scratch[wm::acc_row(v, lane) * 32 + 16 * nt + wm::acc_col(lane)] = a.o[nt][v];
  wave_sync();
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    o[i].x[0] = scratch[g * 32 + 8 * i + 2 * t];
    o[i].x[1] = scratch[g * 32 + 8 * i + 2 * t + 1];
    o[i].x[2] = scratch[(g + 8) * 32 + 8 * i + 2 * t];
    o[i].x[3] = scratch[(g + 8) * 32 + 8 * i + 2 * t + 1];
  }
  wave_sync();
#pragma unroll
  for (int v = 0; v < 8; ++v) scratch[wm::acc_row(v, lane) * 16 + wm::acc_col(lane)] = a.l[v];
  wave_sync();
  lf.x[0] = scratch[g * 16 + 2 * t];
  lf.x[1] = scratch[g * 16 + 2 * t + 1];
  lf.x[2] = scratch[(g + 8) * 16 + 2 * t];
  lf.x[3] = scratch[(g + 8) * 16 + 2 * t + 1];
  wave_sync();
#else
  (void)a, (void)o, (void)lf, (void)lane, (void)scratch;
#endif
}

}  // namespace kt2n_attn::wmma_dec
