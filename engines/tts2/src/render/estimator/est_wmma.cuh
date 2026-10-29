// Flow estimator GEMMs on the AMD matrix unit (RDNA3 / RDNA4 WMMA, bf16 -> fp32). HIP builds only: estimator.cu
// includes this header under KT2N_HIP, so CUDA builds never see it.
//
// est_gemm_wmma_kernel is est_gemm_kernel (est_kernels.cuh) with only its k-loop changed: the same prologue (weight
// prefetch, bias / LayerNorm affine / row metadata, LayerNorm of A into shared memory for QKV / FF1, residual
// prefetch), the same shared-memory staging of A and W tiles, the same k order (BK tiles, 16-wide k-blocks
// ascending; one fp32 chain per output element, so a frame's result does not depend on the bucket, the tile shape or
// its position), but each warp's TM x TN tile is (TM / 16) x (TN / 16) WMMA tiles fed straight from shared memory
// (kt2n::wmma::load_run: A rows and W rows are both k-contiguous there). After the loop the accumulators go through
// shared memory into the original mma.sync m16n8 fragment layout, and the ORIGINAL epilogue runs unchanged, so every
// mode (QKV pool write, FF1 GELU, FINAL Euler step, OUT / FF2 residual) rounds exactly as before given the same sums.
// Not bitwise equal to the software path: the matrix unit adds each k-block's 16 exact products in its own order.
//
// Host side: gemm_wmma() takes the same tile shapes the portable dispatch (gemm_legacy in estimator.cu) takes under
// the 64 KB shared-memory cap (all of them have 16-column warp tiles), est_wmma_prepare() sets their shared-memory
// attributes at load. KT2N_AMD_WMMA_EST=0 (or KT2N_AMD_WMMA=0) keeps the software path.
#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "kt2n/cuda_caps.h"
#include "kt2n/hip/wmma.cuh"

namespace kt2n::render::est {

template <int BM, int BN, int WM, int BK, int ST, int MODE, int NWP = 8>
__global__ void __launch_bounds__(32 * NWP) est_gemm_wmma_kernel(GemmArgs a) {
#if KT2N_WMMA_ARCH
  constexpr int KS = 1;  // no split-K cluster path on AMD
  constexpr bool kLn = (MODE == GM_QKV || MODE == GM_FF1);
  constexpr int NTHR = 32 * NWP;
  constexpr int WN = NWP / WM;
  constexpr int TM = BM / WM;  // warp tile rows
  constexpr int TN = BN / WN;  // warp tile cols
  constexpr int MT = TM / 16;
  constexpr int NT = TN / 8;
  static_assert(MT >= 1 && NT >= 2 && NT % 2 == 0 && TM % 16 == 0, "tile config");
  static_assert(TN % 16 == 0, "WMMA: warp tile columns in 16s");
  constexpr int MW = TM / 16;  // WMMA 16 x 16 tiles per warp along M
  constexpr int NW = TN / 16;  // and along N
  static_assert(size_t(BM) * BN * 4 <= gemm_smem_bytes<BM, BN, BK, ST, MODE>(), "accumulator staging must fit");
  constexpr int LDB = BK + kBPad;
  constexpr int LDA_RES = 256 + kBPad;
  EST_TR_BEGIN
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* as = reinterpret_cast<bf16*>(smem_raw);
  bf16* bs = kLn ? as + BM * LDA_RES : as + ST * BM * LDB;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp / WN, wn = warp % WN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int k_base = (KS > 1) ? int(blockIdx.z) * (a.k / KS) : 0;
  const int n_kt = (a.k / KS) / BK;

  auto load_b = [&](int kt, int stage) {
    bf16* dst = bs + stage * BN * LDB;
#pragma unroll
    for (int chunk = tid; chunk < BN * (BK / 8); chunk += NTHR) {
      const int n = chunk / (BK / 8), kc = (chunk % (BK / 8)) * 8;
      cp_async16(dst + n * LDB + kc, a.w + size_t(n0 + n) * a.k + k_base + kt * BK + kc, true);
    }
  };
  auto load_a = [&](int kt, int stage) {
    bf16* dst = as + stage * BM * LDB;
#pragma unroll
    for (int chunk = tid; chunk < BM * (BK / 8); chunk += NTHR) {
      const int m = chunk / (BK / 8), kc = (chunk % (BK / 8)) * 8;
      cp_async16(dst + m * LDB + kc, a.a + size_t(m0 + m) * a.lda + k_base + kt * BK + kc, true);
    }
  };

  // Weight prefetch (independent of the predecessor) before the dependency wait.
#pragma unroll
  for (int s = 0; s < ST - 1; ++s)
    if (s < n_kt) load_b(s, s);

  // Everything else that does not depend on the predecessor is also read before the wait, so no
  // dependent round trip is left in the prologue / epilogue: bias, LayerNorm affine, and the per-row
  // metadata (frame -> row -> pool page, written by est_prep at the start of the call; every GEMM is
  // >= 3 kernels later, so it is complete when this grid can launch). Same values, same arithmetic.
  const int g = lane >> 2, tq = lane & 3;
  float2 biasv[NT];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int n = n0 + wn * TN + nt * 8 + 2 * tq;
    biasv[nt] = make_float2(0.f, 0.f);
    if constexpr (MODE != GM_QKV) {
      if (MODE != GM_FINAL || n < kMelC)
        biasv[nt] = make_float2(__bfloat162float(a.bias[n]), __bfloat162float(a.bias[n + 1]));
    }
  }
  uint4 lng = make_uint4(0, 0, 0, 0), lnb = make_uint4(0, 0, 0, 0);
  if constexpr (kLn) {
    lng = *reinterpret_cast<const uint4*>(a.ln_g + lane * 8);
    lnb = *reinterpret_cast<const uint4*>(a.ln_b + lane * 8);
  }
  // QKV: element offset of this thread's rows in the layer's pool (-1 = not written); FINAL: row mask.
  int64_t rowoff[MT][2];
  float rowmask[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      rowoff[mt][h] = -1;
      rowmask[mt][h] = 0.f;
      if constexpr (MODE == GM_QKV || MODE == GM_FINAL) {
        const int f = m0 + wm * TM + mt * 16 + g + 8 * h;
        const int r = (f < a.frames_cap) ? a.frame_row[f] : -1;
        if (r >= 0) {
          const EstRow e = a.erows[r];
          const int t = f - e.fr_off;
          if constexpr (MODE == GM_QKV) {
            if (t < e.suffix) {
              const int abs_f = e.past + t;
              const int page = a.page_table[e.page_off + abs_f / kPageFrames];
              rowoff[mt][h] = (int64_t(page) * kPageFrames + (abs_f % kPageFrames)) * (2 * kInner);
            }
          } else {
            rowmask[mt][h] = (t < e.valid) ? 1.f : 0.f;
          }
        }
      }
    }

  pdl_wait();
  if (a.trig_early) pdl_trigger();  // wait -> trigger -> work (dependents' prologues overlap this kernel)
  EST_TR(1);
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if constexpr (!kLn) {
      if (s < n_kt) load_a(s, s);
    }
    cp_async_commit();
  }
  // OUT / FF2: the residual (this thread's epilogue elements of hidden) is read now, overlapping the
  // main loop, instead of after it. hidden was last written >= 2 kernels ago (complete after the wait).
  uint32_t resv[MT][2][NT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        resv[mt][h][nt] = 0u;
        if constexpr (MODE == GM_OUT || MODE == GM_FF2) {
          const int f = m0 + wm * TM + mt * 16 + g + 8 * h;
          const int n = n0 + wn * TN + nt * 8 + 2 * tq;
          if (f < a.frames_cap) resv[mt][h][nt] = *reinterpret_cast<const uint32_t*>(a.out + size_t(f) * a.ld_out + n);
        }
      }

  if constexpr (kLn) {
    // LayerNorm of BM rows (K = 256): warp w owns rows w, w + 8, ...; lane holds 8 channels. All of a
    // warp's row loads are issued before any reduction (independent loads in flight, not one round trip
    // per row).
    constexpr int RPW = BM / NWP;
    constexpr int BATCH = RPW < 8 ? RPW : 8;
    // The BATCH rows' shuffle reductions are interleaved (independent chains in flight instead of
    // one row after another); each row's sums keep exactly the same order, so the bits are unchanged.
    const bf16* gv = reinterpret_cast<const bf16*>(&lng);
    const bf16* bv = reinterpret_cast<const bf16*>(&lnb);
#pragma unroll
    for (int b0 = 0; b0 < RPW; b0 += BATCH) {
      uint4 raw[BATCH];
#pragma unroll
      for (int i = 0; i < BATCH; ++i)
        raw[i] = *reinterpret_cast<const uint4*>(a.a + size_t(m0 + warp + NWP * (b0 + i)) * a.lda + lane * 8);
      float x[BATCH][8], s[BATCH], q[BATCH], mean[BATCH], rstd[BATCH];
#pragma unroll
      for (int i = 0; i < BATCH; ++i) {
        const bf16* hv = reinterpret_cast<const bf16*>(&raw[i]);
        s[i] = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) { x[i][j] = __bfloat162float(hv[j]); s[i] += x[i][j]; }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1)
#pragma unroll
        for (int i = 0; i < BATCH; ++i) s[i] += __shfl_xor_sync(0xffffffffu, s[i], o);
#pragma unroll
      for (int i = 0; i < BATCH; ++i) {
        mean[i] = s[i] * (1.0f / 256.0f);
        q[i] = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) { const float d = x[i][j] - mean[i]; q[i] += d * d; }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1)
#pragma unroll
        for (int i = 0; i < BATCH; ++i) q[i] += __shfl_xor_sync(0xffffffffu, q[i], o);
#pragma unroll
      for (int i = 0; i < BATCH; ++i) {
        const int m = warp + NWP * (b0 + i);
        rstd[i] = rsqrtf(q[i] * (1.0f / 256.0f) + 1e-5f);
        uint32_t packed[4];
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
          const float y0 = (x[i][j] - mean[i]) * rstd[i] * __bfloat162float(gv[j]) + __bfloat162float(bv[j]);
          const float y1 = (x[i][j + 1] - mean[i]) * rstd[i] * __bfloat162float(gv[j + 1]) + __bfloat162float(bv[j + 1]);
          packed[j / 2] = pack_bf16(y0, y1);
        }
        *reinterpret_cast<uint4*>(as + m * LDA_RES + lane * 8) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
      }
    }
    EST_TR(2);  // LN GEMMs: t2 = this thread's LayerNorm rows written (the k loop's first wait is in 'loop')
  }

  // WMMA accumulators: one fp32 chain per output element, k ascending in 16-blocks (the original's order).
  kt2n::wmma::acc_t wacc[MW][NW];
#pragma unroll
  for (int i = 0; i < MW; ++i)
#pragma unroll
    for (int j = 0; j < NW; ++j) wacc[i][j] = kt2n::wmma::zero_acc();
  const int wl = kt2n::wmma::lane();
  const int wi = kt2n::wmma::op_index(wl), wk0 = kt2n::wmma::op_k0(wl);

  for (int kt = 0; kt < n_kt; ++kt) {
    if (kt + ST - 1 < n_kt) {
      load_b(kt + ST - 1, (kt + ST - 1) % ST);
      if constexpr (!kLn) load_a(kt + ST - 1, (kt + ST - 1) % ST);
    }
    cp_async_commit();
    cp_async_wait<ST - 1>();
    __syncthreads();
    if (!kLn && kt == 0) EST_TR(2);
    const bf16* bsk = bs + (kt % ST) * BN * LDB;
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      // A rows wm * TM + 16 mi + op_index (k contiguous in LDS), B = W rows wn * TN + 16 ni (k contiguous).
      kt2n::wmma::words_t af[MW], bfr[NW];
#pragma unroll
      for (int mi = 0; mi < MW; ++mi) {
        const int row = wm * TM + mi * 16 + wi;
        const bf16* p;
        if constexpr (kLn) p = as + row * LDA_RES + kt * BK + kk + wk0;
        else p = as + (kt % ST) * BM * LDB + row * LDB + kk + wk0;
        af[mi] = kt2n::wmma::load_run(p);
      }
#pragma unroll
      for (int ni = 0; ni < NW; ++ni)
        bfr[ni] = kt2n::wmma::load_run(bsk + (wn * TN + ni * 16 + wi) * LDB + kk + wk0);
#pragma unroll
      for (int mi = 0; mi < MW; ++mi)
#pragma unroll
        for (int ni = 0; ni < NW; ++ni) wacc[mi][ni] = kt2n::wmma::mma(af[mi], bfr[ni], wacc[mi][ni]);
    }
    __syncthreads();
  }
  cp_async_wait<0>();
  EST_TR(3);
  if (!a.trig_early) pdl_trigger();
  // Accumulators -> the original m16n8 fragment layout through LDS (the pipeline buffers are free now): this warp's
  // TM x TN tile row-major at warp * TM * TN floats; then lane (g, tq) reads D[16 mt + g + 8 h][8 nt + 2 tq + j] into
  // acc[mt][nt][2 h + j], exactly the registers the original epilogue below consumes.
  (void)KS;
  __syncthreads();
  float* stage = reinterpret_cast<float*>(smem_raw) + size_t(warp) * TM * TN;
#pragma unroll
  for (int mi = 0; mi < MW; ++mi)
#pragma unroll
    for (int ni = 0; ni < NW; ++ni)
#pragma unroll
      for (int v = 0; v < 8; ++v)
        stage[(mi * 16 + kt2n::wmma::acc_row(v, wl)) * TN + ni * 16 + kt2n::wmma::acc_col(wl)] = wacc[mi][ni][v];
  __syncthreads();
  float acc[MT][NT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int j = 0; j < 2; ++j) acc[mt][nt][2 * h + j] = stage[(mt * 16 + g + 8 * h) * TN + nt * 8 + 2 * tq + j];

#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int f = m0 + wm * TM + mt * 16 + g + 8 * h;
      if (f >= a.frames_cap) continue;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int n = n0 + wn * TN + nt * 8 + 2 * tq;
        float y0 = acc[mt][nt][h * 2], y1 = acc[mt][nt][h * 2 + 1];
        if constexpr (MODE == GM_QKV) {
          if (n < 2 * kInner) {
            if (rowoff[mt][h] < 0) continue;
            bf16* dst = a.pool_layer + rowoff[mt][h] + n;
            *reinterpret_cast<uint32_t*>(dst) = pack_bf16(y0, y1);
          } else {
            *reinterpret_cast<uint32_t*>(a.out + size_t(f) * a.ld_out + (n - 2 * kInner)) = pack_bf16(y0, y1);
          }
        } else if constexpr (MODE == GM_FF1) {
          y0 = gelu_f(bfr(y0 + biasv[nt].x));
          y1 = gelu_f(bfr(y1 + biasv[nt].y));
          *reinterpret_cast<uint32_t*>(a.out + size_t(f) * a.ld_out + n) = pack_bf16(y0, y1);
        } else if constexpr (MODE == GM_FINAL) {
          if (n >= kMelC) continue;
          const float maskv = rowmask[mt][h];
          const float yy[2] = {y0, y1};
          const float bb[2] = {biasv[nt].x, biasv[nt].y};
#pragma unroll
          for (int u = 0; u < 2; ++u) {
            const float v = bfr(bfr(yy[u] + bb[u]) * maskv);
            bf16* xp = a.xin + size_t(f) * kIn + n + u;
            const float xn = bfr(__bfloat162float(*xp) + bfr(a.dt * v));
            *xp = __float2bfloat16_rn(xn);
            if (a.deriv_out) a.deriv_out[size_t(f) * kMelC + n + u] = v;
            if (a.mel_out) a.mel_out[size_t(f) * kMelC + n + u] = xn;
          }
        } else {
          bf16* dst = a.out + size_t(f) * a.ld_out + n;
          const __nv_bfloat162 hv = *reinterpret_cast<const __nv_bfloat162*>(&resv[mt][h][nt]);
          y0 = __bfloat162float(hv.x) + bfr(y0 + biasv[nt].x);
          y1 = __bfloat162float(hv.y) + bfr(y1 + biasv[nt].y);
          *reinterpret_cast<uint32_t*>(dst) = pack_bf16(y0, y1);
        }
      }
    }
  }
  EST_TR_END(10 + MODE);
#else
  (void)a;
#if defined(__HIP_DEVICE_COMPILE__)
  __builtin_trap();
#endif
#endif
}

// ---------------------------------------------------------------------------------------------
// Estimator convolutions on the matrix unit. est_conv_wmma_kernel is the portable est_conv_kernel<MODE, 2, true>
// (est_kernels.cuh) with only its k loop on WMMA: same window / residual / weight staging in LDS, same k order
// (k = tap * cin + channel ascending in 16-blocks, then the residual 1x1 conv in its own chain), one 16 x 16 WMMA tile
// per warp (its 16 frames x 16 output channels), accumulators restaged into the original fragment layout so the
// original epilogue (bias, bf16 rounding, portable_values / portable_residual or MODE 2's direct store) runs
// unchanged; est_conv_portable_epilogue<MODE> (LayerNorm, Mish, mask, time embedding / residual) follows as before.
// ---------------------------------------------------------------------------------------------
template <int MODE>
__global__ void __launch_bounds__(kConvThreads) est_conv_wmma_kernel(ConvArgs a) {
#if KT2N_WMMA_ARCH
  constexpr int ST = 2;  // the portable dispatch's weight stages
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int cin = a.c0 + a.c1;
  const int cres = (MODE == 1) ? (a.res_c0 + a.res_c1) : 0;
  const int lda_s = cin + kBPad;
  const int ldr_s = cres + kBPad;
  constexpr int LDB = kConvBK + kBPad;
  bf16* win = reinterpret_cast<bf16*>(smem_raw);                 // [18][cin + pad]
  bf16* rwin = win + size_t(kConvBM + 2) * lda_s;                // [16][cres + pad]
  bf16* bst = rwin + size_t(kConvBM) * ldr_s;                    // [stages][64][BK + pad]

  EST_TR_BEGIN
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int m0 = blockIdx.x * kConvBM;
  const int nb = blockIdx.y * kConvBN;  // this CTA's first output channel
  const int k_conv = 3 * cin;
  const int n_kt = k_conv / kConvBK;
  const int n_kt_res = cres / kConvBK;
  const int n_kt_all = n_kt + n_kt_res;

  auto load_b = [&](int kt, int stage) {
    const bf16* wsrc;
    int ldw, k0;
    if (kt < n_kt) { wsrc = a.w; ldw = k_conv; k0 = kt * kConvBK; }
    else { wsrc = a.res_w; ldw = cres; k0 = (kt - n_kt) * kConvBK; }
    bf16* dst = bst + size_t(stage) * kConvBN * LDB;
#pragma unroll
    for (int i = 0; i < (kConvBN * kConvBK / 8) / kConvThreads; ++i) {
      const int chunk = tid + i * kConvThreads;
      const int n = chunk / (kConvBK / 8), kc = (chunk % (kConvBK / 8)) * 8;
      cp_async16(dst + n * LDB + kc, wsrc + size_t(nb + n) * ldw + k0 + kc, true);
    }
  };
  // Weights do not depend on the predecessor: prefetch the first stages before the dependency wait.
#pragma unroll
  for (int s = 0; s < ST - 1; ++s)
    if (s < n_kt_all) load_b(s, s);
  // Row metadata (est_prep, start of the call) and the epilogue's per-channel constants do not depend
  // on the predecessor: read them before the wait instead of in a dependent chain after it.
  const int r = a.frame_row[m0];
  EstRow e{};
  if (r >= 0) e = a.erows[r];
  const int g = lane >> 2, tq = lane & 3;
  float cb[2][2], cg_[2][2], cbt[2][2], cte[2][2], crb[2][2];
#pragma unroll
  for (int nt = 0; nt < 2; ++nt)
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int n = nb + warp * 16 + nt * 8 + 2 * tq + u;
      cb[nt][u] = __bfloat162float(a.bias[n]);
      cg_[nt][u] = cbt[nt][u] = cte[nt][u] = crb[nt][u] = 0.f;
      if constexpr (MODE != 2) {
        cg_[nt][u] = __bfloat162float(a.ln_g[n]);
        cbt[nt][u] = __bfloat162float(a.ln_b[n]);
      }
      if constexpr (MODE == 0) cte[nt][u] = __bfloat162float(a.temb[n]);
      if constexpr (MODE == 1) crb[nt][u] = __bfloat162float(a.res_b[n]);
    }
  pdl_wait();
  if (a.trig_early) pdl_trigger();  // wait -> trigger -> work (dependents' prologues overlap this kernel)
  EST_TR(1);
  const int t0 = m0 - e.fr_off;  // local frame of the tile's first output frame
  // Input window: rows w = 0..17 hold local frames t0 - 2 + w (masked input).
  for (int idx = tid; idx < (kConvBM + 2) * (cin / 8); idx += kConvThreads) {
    const int w = idx / (cin / 8), c = (idx % (cin / 8)) * 8;
    const int t = t0 - 2 + w;
    bf16* dst = win + w * lda_s + c;
    if (r < 0 || t >= e.valid) {
      *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
    } else if (t >= 0) {
      const int f = e.fr_off + t;
      const bf16* src = (c < a.c0) ? a.in0 + size_t(f) * a.ld0 + c : a.in1 + size_t(f) * a.ld1 + (c - a.c0);
      cp_async16(dst, src, true);
    } else if (e.past > 0) {
      // conv tail: [cin][2] channel-major; t = -2 -> j = 0, t = -1 -> j = 1
      const bf16* tail = a.tail_state + size_t(e.slot) * a.tail_slot_stride;
      const int j = t + 2;
#pragma unroll
      for (int q = 0; q < 8; ++q) dst[q] = tail[(c + q) * 2 + j];
    } else {
      *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
    }
  }
  if constexpr (MODE == 1) {
    for (int idx = tid; idx < kConvBM * (cres / 8); idx += kConvThreads) {
      const int w = idx / (cres / 8), c = (idx % (cres / 8)) * 8;
      const int t = t0 + w;
      bf16* dst = rwin + w * ldr_s + c;
      if (r < 0 || t >= e.valid) {
        *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
      } else {
        const int f = e.fr_off + t;
        const bf16* src = (c < a.res_c0) ? a.res_in0 + size_t(f) * a.res_ld0 + c
                                         : a.res_in1 + size_t(f) * a.res_ld1 + (c - a.res_c0);
        cp_async16(dst, src, true);
      }
    }
  }
  cp_async_commit();  // group 0 = the prefetched weight stages + the window
#pragma unroll
  for (int s = 1; s < ST - 1; ++s) cp_async_commit();  // empty groups keep the count uniform

  // WMMA accumulators of this warp's 16 frames x 16 channels: the conv (k = tap * cin + channel) and, MODE 1, the
  // resnet's 1x1 residual conv; one chain each, k ascending in 16-blocks as in the mma.sync version.
  kt2n::wmma::acc_t wacc = kt2n::wmma::zero_acc();
  kt2n::wmma::acc_t waccr = kt2n::wmma::zero_acc();
  const int wl = kt2n::wmma::lane();
  const int wi = kt2n::wmma::op_index(wl), wk0 = kt2n::wmma::op_k0(wl);

  const int wn0 = warp * 16;  // this warp's 16 output channels within the CTA's 64
  for (int kt = 0; kt < n_kt_all; ++kt) {
    if (kt + ST - 1 < n_kt_all) load_b(kt + ST - 1, (kt + ST - 1) % ST);
    cp_async_commit();
    cp_async_wait<ST - 1>();
    __syncthreads();
    if (kt == 0) EST_TR(2);
    const bf16* bs = bst + size_t(kt % ST) * kConvBN * LDB;
    const bool is_res = kt >= n_kt;
#pragma unroll
    for (int kk = 0; kk < kConvBK; kk += 16) {
      // A: output frame wi of the tile at tap j = window row j + wi, channels c .. (k contiguous in LDS; cin % 16 == 0,
      // so a 16-block never crosses a tap). B: W row wn0 + wi (the output channel), k contiguous.
      kt2n::wmma::words_t af;
      if (!is_res) {
        const int kg = kt * kConvBK + kk;  // global k = j * cin + c
        const int j = kg / cin, c = kg % cin;
        af = kt2n::wmma::load_run(win + (j + wi) * lda_s + c + wk0);
      } else {
        const int c = (kt - n_kt) * kConvBK + kk;
        af = kt2n::wmma::load_run(rwin + wi * ldr_s + c + wk0);
      }
      const kt2n::wmma::words_t bfrag = kt2n::wmma::load_run(bs + (wn0 + wi) * LDB + kk + wk0);
      if (!is_res) wacc = kt2n::wmma::mma(af, bfrag, wacc);
      else waccr = kt2n::wmma::mma(af, bfrag, waccr);
    }
    __syncthreads();
  }
  cp_async_wait<0>();
  EST_TR(3);
  if (!a.trig_early) pdl_trigger();

  // Accumulators -> the original m16n8 fragment registers through LDS (the weight stages are free now): warp w's
  // 16 x 16 tile at bst + w * 256 floats (MODE 1's residual tile 1024 floats further); lane (g, tq) reads
  // D[g + 8 h][8 nt + 2 tq + j] into acc[nt][2 h + j], exactly what the original epilogue below consumes.
  static_assert(size_t(ST) * kConvBN * LDB * 2 >= 2 * 4 * 256 * sizeof(float), "accumulator staging must fit");
  __syncthreads();
  {
    float* sa = reinterpret_cast<float*>(bst) + size_t(warp) * 256;
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      const int rr = kt2n::wmma::acc_row(v, wl), cc = kt2n::wmma::acc_col(wl);
      sa[rr * 16 + cc] = wacc[v];
      if constexpr (MODE == 1) sa[4 * 256 + rr * 16 + cc] = waccr[v];
    }
  }
  __syncthreads();
  float acc[2][4];
  float accr[2][4];
  {
    const float* sa = reinterpret_cast<const float*>(bst) + size_t(warp) * 256;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          acc[nt][2 * h + j] = sa[(g + 8 * h) * 16 + nt * 8 + 2 * tq + j];
          accr[nt][2 * h + j] = (MODE == 1) ? sa[4 * 256 + (g + 8 * h) * 16 + nt * 8 + 2 * tq + j] : 0.f;
        }
  }

  // Tail stage: the tile that owns local frame shift - 1 saves frames shift - 2, shift - 1 of the
  // masked conv input (window rows (shift - 2) - t0 + 2 and + 1). Rank 0 of the cluster writes.
  if (blockIdx.y == 0 && r >= 0 && a.tail_stage != nullptr && e.shift > 0) {
    const int last = e.shift - 1;
    if (last >= t0 && last < t0 + kConvBM) {
      const int w0 = (e.shift - 2) - t0 + 2;
      bf16* stg = a.tail_stage + size_t(r) * a.stage_row_stride;
      for (int c = tid; c < cin; c += kConvThreads) {
        stg[c * 2 + 0] = win[w0 * lda_s + c];
        stg[c * 2 + 1] = win[(w0 + 1) * lda_s + c];
      }
    }
  }

  // Epilogue. Thread rows g, g + 8; columns nb + wn0 + nt * 8 + 2 t, + 1.
  float v[2][4];
#pragma unroll
  for (int nt = 0; nt < 2; ++nt) {
    const float b0 = cb[nt][0], b1 = cb[nt][1];
    v[0][nt * 2 + 0] = bfr(acc[nt][0] + b0);
    v[0][nt * 2 + 1] = bfr(acc[nt][1] + b1);
    v[1][nt * 2 + 0] = bfr(acc[nt][2] + b0);
    v[1][nt * 2 + 1] = bfr(acc[nt][3] + b1);
  }
  float maskv[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) maskv[h] = (r >= 0 && t0 + g + 8 * h < e.valid) ? 1.f : 0.f;

  if constexpr (MODE != 2) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int f = m0 + g + 8 * h;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int n = nb + wn0 + nt * 8 + 2 * tq;
        *reinterpret_cast<uint32_t*>(a.portable_values + size_t(f) * kHid + n) =
            pack_bf16(v[h][nt * 2], v[h][nt * 2 + 1]);
        if constexpr (MODE == 1) {
          const float r0 = bfr(accr[nt][h * 2] + crb[nt][0]);
          const float r1 = bfr(accr[nt][h * 2 + 1] + crb[nt][1]);
          *reinterpret_cast<uint32_t*>(a.portable_residual + size_t(f) * kHid + n) = pack_bf16(r0, r1);
        }
      }
    }
  }

  // (The clustered LayerNorm epilogue of the SM 9.0+ kernel is not used: est_conv_portable_epilogue follows.)
  if constexpr (MODE == 2) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int f = m0 + g + 8 * h;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int n = nb + wn0 + nt * 8 + 2 * tq;
        *reinterpret_cast<uint32_t*>(a.out + size_t(f) * a.ld_out + n) = pack_bf16(v[h][nt * 2], v[h][nt * 2 + 1]);
      }
    }
  }
  EST_TR_END(20 + MODE);
#else
  (void)a;
#if defined(__HIP_DEVICE_COMPILE__)
  __builtin_trap();
#endif
#endif
}

// ---------------------------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------------------------
// Whether the estimator GEMMs run on the matrix unit: a WMMA GPU (DeviceCaps::wmma, which KT2N_AMD_WMMA=0 turns off)
// and not KT2N_AMD_WMMA_EST=0 (read once).
inline bool est_wmma_on() {
  static const bool off = [] {
    const char* e = std::getenv("KT2N_AMD_WMMA_EST");
    return e != nullptr && e[0] == '0' && e[1] == '\0';
  }();
  return !off && kt2n::cuda::current_caps().wmma != 0;
}

inline void est_wmma_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("estimator WMMA GEMM: ") + what + ": " + cudaGetErrorString(e));
}

template <int BM, int BN, int WM, int BK, int ST, int MODE, int NWP = 8>
void est_wmma_launch(cudaStream_t st, const GemmArgs& g, int frames_cap) {
  const dim3 grid(unsigned(g.n / BN), unsigned(frames_cap / BM));
  est_gemm_wmma_kernel<BM, BN, WM, BK, ST, MODE, NWP>
      <<<grid, dim3(32 * NWP), gemm_smem_bytes<BM, BN, BK, ST, MODE>(), st>>>(g);
  est_wmma_check(cudaGetLastError(), "launch");
}
// The LayerNorm-prologue modes (QKV, FF1) of one shape; the streamed-A modes (OUT, FF2, FINAL) of one shape.
template <int BM, int BN, int WM, int BK, int ST>
void est_wmma_ln(cudaStream_t st, int mode, const GemmArgs& g, int F) {
  if (mode == GM_QKV) est_wmma_launch<BM, BN, WM, BK, ST, GM_QKV>(st, g, F);
  else est_wmma_launch<BM, BN, WM, BK, ST, GM_FF1>(st, g, F);
}
template <int BM, int BN, int WM, int BK, int ST, int NWP = 8>
void est_wmma_rest(cudaStream_t st, int mode, const GemmArgs& g, int F) {
  if (mode == GM_OUT) est_wmma_launch<BM, BN, WM, BK, ST, GM_OUT, NWP>(st, g, F);
  else if (mode == GM_FF2) est_wmma_launch<BM, BN, WM, BK, ST, GM_FF2, NWP>(st, g, F);
  else est_wmma_launch<BM, BN, WM, BK, ST, GM_FINAL, NWP>(st, g, F);
}

template <int BM, int BN, int WM, int BK, int ST, int MODE, int NWP = 8>
void est_wmma_attr(int cap) {
  constexpr size_t need = gemm_smem_bytes<BM, BN, BK, ST, MODE>();
  if (need > size_t(cap)) return;  // never launched on this device (gemm_wmma checks the same bound)
  est_wmma_check(cudaFuncSetAttribute(est_gemm_wmma_kernel<BM, BN, WM, BK, ST, MODE, NWP>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, int(need)),
                 "cudaFuncSetAttribute");
}
// Load time (outside any stream capture): the shared-memory attributes of every shape gemm_wmma can launch.
template <int MODE>
void est_conv_wmma_attr(size_t need, int cap) {
  if (need > size_t(cap)) return;
  est_wmma_check(cudaFuncSetAttribute(est_conv_wmma_kernel<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(need)),
                 "cudaFuncSetAttribute(conv)");
}
inline void est_wmma_prepare(int cap) {
  if (!est_wmma_on()) return;
  // convs: the largest launch of each mode, as set_legacy_attrs (estimator.cu) does for the portable kernel
  est_conv_wmma_attr<0>(conv_smem_bytes(512, 0, 2), cap);
  est_conv_wmma_attr<1>(conv_smem_bytes(256, 512, 2), cap);
  est_conv_wmma_attr<2>(conv_smem_bytes(256, 0, 2), cap);
  est_conv_wmma_attr<3>(conv_smem_bytes(256, 0, 2), cap);
  est_wmma_attr<32, 64, 2, 64, 4, GM_QKV>(cap);
  est_wmma_attr<32, 64, 2, 64, 4, GM_FF1>(cap);
  est_wmma_attr<64, 64, 2, 64, 3, GM_QKV>(cap);
  est_wmma_attr<64, 64, 2, 64, 3, GM_FF1>(cap);
  est_wmma_attr<64, 128, 2, 32, 3, GM_QKV>(cap);
  est_wmma_attr<64, 128, 2, 32, 3, GM_FF1>(cap);
  est_wmma_attr<32, 128, 2, 32, 3, GM_QKV>(cap);
  est_wmma_attr<32, 128, 2, 32, 3, GM_FF1>(cap);
  est_wmma_attr<16, 32, 1, 64, 8, GM_OUT, 2>(cap);
  est_wmma_attr<16, 32, 1, 64, 8, GM_FF2, 2>(cap);
  est_wmma_attr<16, 16, 1, 64, 8, GM_OUT, 1>(cap);
  est_wmma_attr<16, 16, 1, 64, 8, GM_FF2, 1>(cap);
  est_wmma_attr<64, 32, 4, 64, 4, GM_OUT>(cap);
  est_wmma_attr<64, 32, 4, 64, 4, GM_FF2>(cap);
  est_wmma_attr<64, 32, 4, 64, 4, GM_FINAL>(cap);
  est_wmma_attr<64, 64, 2, 64, 3, GM_OUT>(cap);
  est_wmma_attr<64, 64, 2, 64, 3, GM_FF2>(cap);
  est_wmma_attr<64, 64, 2, 64, 3, GM_FINAL>(cap);
  est_wmma_attr<64, 128, 2, 32, 3, GM_OUT>(cap);
  est_wmma_attr<64, 128, 2, 32, 3, GM_FF2>(cap);
  est_wmma_attr<64, 128, 2, 32, 3, GM_FINAL>(cap);
}

template <int BM, int BN, int BK, int ST, int MODE>
bool est_wmma_fits(const GemmArgs& g, int F, int cap) {
  return F % BM == 0 && g.n % BN == 0 && gemm_smem_bytes<BM, BN, BK, ST, MODE>() <= size_t(cap);
}
// The portable dispatch's shape choice (estimator.cu gemm_legacy, same order and wave cost) on the WMMA kernel.
// `small_fr` / `sms`: estimator.cu's small_fr() and num_sms(). Returns false (nothing launched) when no shape fits:
// the caller keeps its software path.
inline bool gemm_wmma(cudaStream_t st, int mode, const GemmArgs& g, int F, int cap, int small_fr, int sms) {
  const bool small = F <= small_fr;
  if (mode == GM_QKV || mode == GM_FF1) {
    if (g.k != 256) return false;  // the LayerNorm prologue holds the whole K = 256 row
    if (small && est_wmma_fits<32, 64, 64, 4, GM_QKV>(g, F, cap)) return est_wmma_ln<32, 64, 2, 64, 4>(st, mode, g, F), true;
    if (!small) {
      const int bm[3] = {64, 64, 32}, bn[3] = {64, 128, 128};
      const bool ok[3] = {est_wmma_fits<64, 64, 64, 3, GM_QKV>(g, F, cap), est_wmma_fits<64, 128, 32, 3, GM_QKV>(g, F, cap),
                          est_wmma_fits<32, 128, 32, 3, GM_QKV>(g, F, cap)};
      int best = -1;
      double best_cost = 1e30;
      for (int i = 0; i < 3; ++i) {
        if (!ok[i]) continue;
        const int ctas = (F / bm[i]) * (g.n / bn[i]);
        const int waves = (ctas + sms - 1) / sms;
        const double cost = waves * (3.2 + 0.43 * double(bm[i]) * bn[i] / 1024.0);
        if (cost < best_cost - 1e-9) { best_cost = cost; best = i; }
      }
      if (best == 0) return est_wmma_ln<64, 64, 2, 64, 3>(st, mode, g, F), true;
      if (best == 1) return est_wmma_ln<64, 128, 2, 32, 3>(st, mode, g, F), true;
    }
    if (est_wmma_fits<32, 128, 32, 3, GM_QKV>(g, F, cap)) return est_wmma_ln<32, 128, 2, 32, 3>(st, mode, g, F), true;
    return false;
  }
  if (g.k % 64 != 0) return false;
  if (small && (mode == GM_OUT || mode == GM_FF2)) {
    if (est_wmma_fits<16, 32, 64, 8, GM_OUT>(g, F, cap)) {
      if (mode == GM_OUT) est_wmma_launch<16, 32, 1, 64, 8, GM_OUT, 2>(st, g, F);
      else est_wmma_launch<16, 32, 1, 64, 8, GM_FF2, 2>(st, g, F);
      return true;
    }
    if (est_wmma_fits<16, 16, 64, 8, GM_OUT>(g, F, cap)) {
      if (mode == GM_OUT) est_wmma_launch<16, 16, 1, 64, 8, GM_OUT, 1>(st, g, F);
      else est_wmma_launch<16, 16, 1, 64, 8, GM_FF2, 1>(st, g, F);
      return true;
    }
  }
  if (F <= 128 && est_wmma_fits<64, 32, 64, 4, GM_OUT>(g, F, cap)) return est_wmma_rest<64, 32, 4, 64, 4>(st, mode, g, F), true;
  if (est_wmma_fits<64, 64, 64, 3, GM_OUT>(g, F, cap)) return est_wmma_rest<64, 64, 2, 64, 3>(st, mode, g, F), true;
  if (est_wmma_fits<64, 128, 32, 3, GM_OUT>(g, F, cap)) return est_wmma_rest<64, 128, 2, 32, 3>(st, mode, g, F), true;
  return false;
}

// One estimator conv on the matrix unit, the portable path's launches with the WMMA conv kernel: grid = (F / 16, 4),
// `bytes` = conv_smem_bytes(cin, cres, 2); MODES 0, 1, 3 then run the unchanged est_conv_portable_epilogue.
// Returns false (nothing launched) when the shared memory does not fit: the caller keeps its software path.
inline bool est_conv_wmma(int mode, dim3 grid, size_t bytes, int cap, cudaStream_t st, const ConvArgs& a, int F) {
  // KT2N_AMD_WMMA_EST_CONV=0 keeps only the convs on the software path (A/B); read once.
  static const bool off = [] {
    const char* e = std::getenv("KT2N_AMD_WMMA_EST_CONV");
    return e != nullptr && e[0] == '0' && e[1] == '\0';
  }();
  if (off || bytes > size_t(cap)) return false;
  const dim3 epi_grid(unsigned(F / kConvBM));
  switch (mode) {
    case 0:
      est_conv_wmma_kernel<0><<<grid, dim3(kConvThreads), bytes, st>>>(a);
      est_conv_portable_epilogue<0><<<epi_grid, dim3(512), 0, st>>>(a);
      break;
    case 1:
      est_conv_wmma_kernel<1><<<grid, dim3(kConvThreads), bytes, st>>>(a);
      est_conv_portable_epilogue<1><<<epi_grid, dim3(512), 0, st>>>(a);
      break;
    case 2:
      est_conv_wmma_kernel<2><<<grid, dim3(kConvThreads), bytes, st>>>(a);
      break;
    default:
      est_conv_wmma_kernel<3><<<grid, dim3(kConvThreads), bytes, st>>>(a);
      est_conv_portable_epilogue<3><<<epi_grid, dim3(512), 0, st>>>(a);
  }
  est_wmma_check(cudaGetLastError(), "conv launch");
  return true;
}

}  // namespace kt2n::render::est
