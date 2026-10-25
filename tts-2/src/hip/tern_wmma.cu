// Ternary projections on the AMD matrix unit (RDNA3 / RDNA4 WMMA, bf16 -> fp32). HIP builds only; the reference is
// kt2n_tern_proj's software path (src/cuda/kernels/tern_proj.cu). See include/kt2n/hip/wmma_kernels.h.
//
// Arithmetic (fixed, part of the result): K is cut into kWK = 8 equal ranges; partial p_j is one WMMA accumulator
// chain over range j, k-blocks of 16 ascending; the output is (((p0 + p1) + p2) + ... + p7) in fp32, rounded once by
// the epilogue. Two launch forms compute exactly that:
//   sequential (large M): a wave owns an (MT x 16) x (NT x 16) tile and runs the 8 chains one after the other,
//     adding each to a running total; a block is kWaves waves side by side along N (same rows: shared cache lines);
//   parallel (small M, where N / 16 tiles are too few waves to fill the GPU): a block is kWK waves on the same tile,
//     wave j runs chain j, and wave 0 adds the others' partials from LDS in order j = 1 .. 7.
// Either way a row's bits never depend on M, the form, the tile shape or the tile the row lands in.
//
// Per k-block of 16 (one WMMA per 16 x 16 tile):
//   A: lane l reads its row's 16 (gfx11) or 8 (gfx12) activations straight from global memory (kt2n::wmma::load_run);
//   B: lane l decodes its column's codes word (16 k, "sel" layout) into exactly the bf16 weights {-s, 0, +s} the
//      reference multiplies (v_perm_b32 form of kt2n_tern::decode_step).
//
// Weight layout (kt2n/kernels/tern_proj.h): codes [N/8 tiles][K/256 chunks][32 lanes][4 words], lane 4g + t of a tile
// holds row 8T + g, k = 256c + 64t + 16w + [0, 16) in word w; scales [N/8][K/256][8 rows][2 groups of 128].
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cstdlib>

#include "kt2n/cuda_caps.h"
#include "kt2n/hip/wmma.cuh"
#include "kt2n/hip/wmma_kernels.h"
#include "kt2n/kernels/tern_proj.h"
#include "tern_proj.cuh"

namespace {

using bf16 = __nv_bfloat16;
namespace wm = kt2n::wmma;

constexpr int kWaves = 4;  // sequential form: waves per block along N
constexpr int kWK = 8;     // K partial sums per output (fixed: it is part of the arithmetic)

struct TernArgs {
  const bf16* x;
  int64_t ldx;
  int m;
  const uint4* codes;
  const uint16_t* scales;
  int n;
  int k;
  bf16* y;
  int64_t ldy;
  const bf16* r;
  int64_t ldr;
};

// Step q (0..3) of a "sel" codes word -> b0 = bf16x2 {w(4q), w(4q+1)}, b1 = {w(4q+2), w(4q+3)}: the same bits as
// kt2n_tern::decode_step, on v_perm_b32 (HIP's __byte_perm is a byte-array emulation that the compiler puts in LDS:
// a 64-bit LDS store and four byte loads per call). v_perm_b32 (S0, S1, sel): byte n of the result is byte sel.n of
// the 8-byte value {S0 : S1} (S1 = bytes 0..3). With y = the four 2-bit codes (byte r = code r) and y4 = y + 4:
//   selector bytes [c0, 4 + c0, c1, 4 + c1] = perm(y4, y, 0x05010400), [c2, 4 + c2, c3, 4 + c3] = perm(y4, y, 0x07030602),
//   weight pair = perm(tb, ta, selector): ta = bytes [0, lo, lo, 0], tb = [0, hi, hi ^ 0x80, 0] (kt2n_tern::scale_tables),
//   so code 0 -> +0.0, 1 -> +s, 2 -> -s, exactly as on NVIDIA. 7 VALU instructions per 4 weights.
[[maybe_unused]] __device__ __forceinline__ void decode_step_amd(uint32_t word, int q, uint32_t ta, uint32_t tb,
                                                                 uint32_t& b0, uint32_t& b1) {
#if KT2N_WMMA_ARCH
  const uint32_t y = (word >> (2 * q)) & 0x03030303u;
  const uint32_t y4 = y + 0x04040404u;
  b0 = __builtin_amdgcn_perm(tb, ta, __builtin_amdgcn_perm(y4, y, 0x05010400u));
  b1 = __builtin_amdgcn_perm(tb, ta, __builtin_amdgcn_perm(y4, y, 0x07030602u));
#else
  kt2n_tern::decode_step(word, q, ta, tb, b0, b1);
#endif
}

// B operand of one 16 x 16 tile from this lane's codes word: gfx11 all 16 k (steps q = 0..3), gfx12 the lane half's
// 8 k (steps q = kh / 4, kh / 4 + 1 with kh = 0 or 8).
[[maybe_unused]] __device__ __forceinline__ wm::words_t decode_b(uint32_t word, uint32_t ta, uint32_t tb, int kh) {
  wm::words_t b;
#if KT2N_WMMA_ARCH == 12
  const int q0 = kh >> 2;
  uint32_t b0, b1, b2, b3;
  decode_step_amd(word, q0, ta, tb, b0, b1);
  decode_step_amd(word, q0 + 1, ta, tb, b2, b3);
  b[0] = b0;
  b[1] = b1;
  b[2] = b2;
  b[3] = b3;
#else
  (void)kh;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    uint32_t b0, b1;
    decode_step_amd(word, q, ta, tb, b0, b1);
    b[2 * q] = b0;
    b[2 * q + 1] = b1;
  }
#endif
  return b;
}

// One partial chain: slabs [s0, s1) (slab = 64 k: one 16-byte codes load per n-tile, 4 k-blocks) into acc. Inside a
// slab the next k-block's activations are loaded before the current block's WMMAs (two register sets); the
// scheduling barrier after each block stops the compiler from hoisting every load of the slab to its top (that
// needs 4 x MT operand sets and spilled).
template <int MT, int NT>
__device__ __forceinline__ void chain(wm::acc_t (&acc)[MT][NT], int s0, int s1, const uint4* const (&wcol)[NT],
                                      const uint16_t* const (&scol)[NT], const bf16* const (&xrow)[MT],
                                      const bool (&rv)[MT], int kh) {
#if KT2N_WMMA_ARCH
#pragma unroll 1
  for (int s = s0; s < s1; ++s) {
    const int c = s >> 2, t = s & 3;
    const int ks = 64 * s;  // = 256 c + 64 t
    uint4 wq[NT];
    uint32_t ta[NT], tb[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      wq[nt] = kt2n_tern::ld_weights(wcol[nt] + c * 32 + t);
      kt2n_tern::scale_tables(scol[nt][c * 16 + (t >> 1)], ta[nt], tb[nt]);
    }
    wm::words_t af[2][MT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) af[0][mt] = wm::load_run_or_zero(xrow[mt] + ks, rv[mt]);
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      if (w < 3) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) af[(w + 1) & 1][mt] = wm::load_run_or_zero(xrow[mt] + ks + 16 * (w + 1), rv[mt]);
      }
      wm::words_t bw[NT];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) bw[nt] = decode_b(kt2n_tern::word_of(wq[nt], w), ta[nt], tb[nt], kh);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) acc[mt][nt] = wm::mma(af[w & 1][mt], bw[nt], acc[mt][nt]);
      __builtin_amdgcn_sched_barrier(0);
    }
  }
#else
  (void)acc, (void)s0, (void)s1, (void)wcol, (void)scol, (void)xrow, (void)rv, (void)kh;
#endif
}

// PAR = false: sequential form, block = kWaves waves along N. PAR = true: parallel form, block = kWK waves on one tile.
template <int MT, int NT, int EPI, bool PAR>
__global__ void __launch_bounds__(PAR ? 32 * kWK : 32 * kWaves) tern_wmma_kernel(TernArgs a) {
#if KT2N_WMMA_ARCH
  const int l = wm::lane();
  const int wave = static_cast<int>(threadIdx.x) >> 5;
  const int i = wm::op_index(l), kh = wm::op_k0(l);
  const int m0 = static_cast<int>(blockIdx.y) * (16 * MT);
  const int n0 = (PAR ? static_cast<int>(blockIdx.x) : static_cast<int>(blockIdx.x) * kWaves + wave) * (16 * NT);
  if (!PAR && n0 >= a.n) return;  // whole wave: n % (16 NT) == 0 (the parallel form's grid is exact)
  const int C = a.k / kt2n_tern::kChunk;
  const int S = 4 * C, SP = S / kWK;  // slabs, slabs per partial

  // This lane's weight column (output n) per n-tile and its activation row per m-tile.
  const uint4* wcol[NT];
  const uint16_t* scol[NT];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int n = n0 + 16 * nt + i, tile = n >> 3, g = n & 7;
    wcol[nt] = a.codes + static_cast<size_t>(tile) * C * 32 + 4 * g;
    scol[nt] = a.scales + static_cast<size_t>(tile) * C * 16 + 2 * g;
  }
  const bf16* xrow[MT];
  bool rv[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int row = m0 + 16 * mt + i;
    rv[mt] = row < a.m;
    xrow[mt] = a.x + static_cast<int64_t>(rv[mt] ? row : 0) * a.ldx + kh;
  }

  wm::acc_t acc[MT][NT];
  if constexpr (PAR) {
    // Wave j: partial j. Waves 1..7 park theirs in LDS ([partial][element][lane]: conflict-free), wave 0 adds them in
    // order onto its own p0.
    __shared__ float red[kWK - 1][MT * NT * 8][32];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[mt][nt] = wm::zero_acc();
    chain<MT, NT>(acc, wave * SP, (wave + 1) * SP, wcol, scol, xrow, rv, kh);
    if (wave > 0) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
          for (int v = 0; v < 8; ++v) red[wave - 1][(mt * NT + nt) * 8 + v][l] = acc[mt][nt][v];
    }
    __syncthreads();
    if (wave != 0) return;
#pragma unroll 1
    for (int j = 1; j < kWK; ++j)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
          for (int v = 0; v < 8; ++v) acc[mt][nt][v] += red[j - 1][(mt * NT + nt) * 8 + v][l];
  } else {
    // total = p0; total += p_j for j = 1 .. 7 (p_j in a fresh accumulator each time).
#pragma unroll 1
    for (int j = 0; j < kWK; ++j) {
      wm::acc_t cur[MT][NT];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) cur[mt][nt] = wm::zero_acc();
      chain<MT, NT>(cur, j * SP, (j + 1) * SP, wcol, scol, xrow, rv, kh);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          if (j == 0) acc[mt][nt] = cur[mt][nt];
          else acc[mt][nt] += cur[mt][nt];
        }
    }
  }

  // Epilogue in the accumulator layout: lane l holds column acc_col(l) of rows acc_row(v, l), v = 0..7.
  const int col = wm::acc_col(l);
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int n = n0 + 16 * nt + col;
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        const int row = m0 + 16 * mt + wm::acc_row(v, l);
        const float sum = acc[mt][nt][v];
        if constexpr (EPI == KT2N_TERN_EPI_SILU_MUL) {
          // gate_i = column 2i (even lane), up_i = column 2i + 1 (the odd neighbour); output column i.
          const float up = __shfl_xor(sum, 1);
          if ((l & 1) == 0 && row < a.m)
            a.y[static_cast<int64_t>(row) * a.ldy + (n >> 1)] = kt2n_tern::silu_mul_bf16(sum, up);
        } else if constexpr (EPI == KT2N_TERN_EPI_RESIDUAL) {
          if (row < a.m) {
            const float res = __bfloat162float(a.r[static_cast<int64_t>(row) * a.ldr + n]);
            a.y[static_cast<int64_t>(row) * a.ldy + n] = __float2bfloat16_rn(res + kt2n_tern::bf16_round(sum));
          }
        } else {
          if (row < a.m) a.y[static_cast<int64_t>(row) * a.ldy + n] = __float2bfloat16_rn(sum);
        }
      }
    }
#else
  (void)a;
#if defined(__HIP_DEVICE_COMPILE__)
  __builtin_trap();
#endif
#endif
}

template <int MT, int NT, bool PAR>
int launch_shape(const TernArgs& a, int epi, cudaStream_t s) {
  if (a.n % (16 * NT) != 0) return KT2N_WMMA_EINVAL;
  const int tiles_n = a.n / (16 * NT);
  const dim3 grid(PAR ? tiles_n : (tiles_n + kWaves - 1) / kWaves, (a.m + 16 * MT - 1) / (16 * MT));
  const dim3 block(PAR ? 32 * kWK : 32 * kWaves);
  if (epi == KT2N_TERN_EPI_STORE) tern_wmma_kernel<MT, NT, KT2N_TERN_EPI_STORE, PAR><<<grid, block, 0, s>>>(a);
  else if (epi == KT2N_TERN_EPI_RESIDUAL) tern_wmma_kernel<MT, NT, KT2N_TERN_EPI_RESIDUAL, PAR><<<grid, block, 0, s>>>(a);
  else tern_wmma_kernel<MT, NT, KT2N_TERN_EPI_SILU_MUL, PAR><<<grid, block, 0, s>>>(a);
  return cudaGetLastError() == cudaSuccess ? KT2N_WMMA_OK : KT2N_WMMA_ECUDA;
}

// KT2N_WMMA_TERN_SHAPE=[1]<MT><NT> (e.g. 22, 111 = parallel form 1 x 1): one launch shape for every call
// (tuning); read once.
int env_shape() {
  static const int v = [] {
    const char* e = std::getenv("KT2N_WMMA_TERN_SHAPE");
    return e ? std::atoi(e) : 0;
  }();
  return v;
}

bool aligned(const void* p, size_t n) { return (reinterpret_cast<uintptr_t>(p) % n) == 0; }

}  // namespace

extern "C" int kt2n_wmma_enabled(void) { return kt2n::cuda::current_caps().wmma != 0 ? 1 : 0; }

extern "C" int kt2n_wmma_decode_mode(void) {
  // Default 2 (decode projections and attention on the matrix unit); KT2N_AMD_WMMA_DECODE=0 / =1 opt out of both /
  // of the attention only. Any other value keeps the default. 0 and 1 are test switches (A/B), not supported modes:
  // the supported fallback is KT2N_AMD_WMMA=0, the whole software path.
  static const int mode = [] {
    const char* e = std::getenv("KT2N_AMD_WMMA_DECODE");
    return (e != nullptr && (e[0] == '0' || e[0] == '1' || e[0] == '2') && e[1] == '\0') ? e[0] - '0' : 2;
  }();
  return kt2n::cuda::current_caps().wmma != 0 ? mode : 0;
}

extern "C" int kt2n_wmma_tern_proj(const void* x, int64_t ldx, int m, const void* codes, const void* scales, int n,
                                   int k, void* y, int64_t ldy, int epi, const void* residual, int64_t ldr, int shape,
                                   cudaStream_t stream) {
  if (!x || !codes || !scales || !y || m < 1 || n <= 0 || n % 64 != 0 || k <= 0 || k % (64 * kWK) != 0 ||
      ldx < k || ldx % 8 != 0 || !aligned(x, 16) || !aligned(codes, 16) || !aligned(scales, 2))
    return KT2N_WMMA_EINVAL;
  if (epi != KT2N_TERN_EPI_STORE && epi != KT2N_TERN_EPI_RESIDUAL && epi != KT2N_TERN_EPI_SILU_MUL)
    return KT2N_WMMA_EINVAL;
  if (epi == KT2N_TERN_EPI_RESIDUAL && (!residual || ldr < n)) return KT2N_WMMA_EINVAL;
  if (ldy < (epi == KT2N_TERN_EPI_SILU_MUL ? n / 2 : n)) return KT2N_WMMA_EINVAL;
  if (kt2n::cuda::current_caps().wmma == 0) return KT2N_WMMA_EINVAL;

  TernArgs a{};
  a.x = static_cast<const bf16*>(x);
  a.ldx = ldx;
  a.m = m;
  a.codes = static_cast<const uint4*>(codes);
  a.scales = static_cast<const uint16_t*>(scales);
  a.n = n;
  a.k = k;
  a.y = static_cast<bf16*>(y);
  a.ldy = ldy;
  a.r = static_cast<const bf16*>(residual);
  a.ldr = ldr;

  if (shape == 0) shape = env_shape();
  if (shape == 0) {
    // [1]MT NT from a 28-layer sweep of all four families on an RX 7900 XTX and an RX 9070 XT (test_gpu_hip_wmma --bench
    // --shape=<code> for each code): the 8 K ranges on 8
    // waves while the GPU is short of tiles, then one wave per tile; RDNA3 keeps 32 x 16 parallel tiles longer.
    const int arch = kt2n::cuda::current_caps().wmma;
    if (m <= 32) shape = 111;
    else if (arch == 11) shape = m <= 192 ? 121 : 22;
    else shape = m <= 96 ? 11 : 22;
  }
  switch (shape) {
    case 11: return launch_shape<1, 1, false>(a, epi, stream);
    case 12: return launch_shape<1, 2, false>(a, epi, stream);
    case 14: return launch_shape<1, 4, false>(a, epi, stream);
    case 22: return launch_shape<2, 2, false>(a, epi, stream);
    case 24: return launch_shape<2, 4, false>(a, epi, stream);
    case 42: return launch_shape<4, 2, false>(a, epi, stream);
    case 111: return launch_shape<1, 1, true>(a, epi, stream);
    case 112: return launch_shape<1, 2, true>(a, epi, stream);
    case 121: return launch_shape<2, 1, true>(a, epi, stream);
    default: return KT2N_WMMA_EINVAL;
  }
}
