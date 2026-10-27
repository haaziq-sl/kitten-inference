// Flow-estimator attention on the AMD matrix unit (RDNA3 / RDNA4 WMMA). HIP builds only: estimator.cu includes this
// header under KT2N_HIP after est_kernels.cuh.
//
// est_attn2_wmma_kernel is est_attn2_kernel (est_kernels.cuh) with the two matrix products of each 32-key tile on
// WMMA; work split, K/V staging, the online softmax and the cross-warp merge are the original code:
//   S = Q K^T: 2 WMMA tiles (16 queries x 16 keys) over the 4 16-dim blocks (Q and K rows are k-contiguous in LDS);
//     the tile goes through a per-warp LDS buffer into the mma.sync accumulator layout the softmax code reads;
//   O += P V: the softmax's bf16 probabilities (the same values its pack_bf16 gives) go to LDS as the A operand,
//     V is a strided B operand ([key][dim] tile, k = key); O stays in the WMMA accumulator layout, is rescaled per row
//     with the softmax's alpha (passed per row through LDS) and is written straight into the merge buffer.
// Each query row's result depends only on that row (batch invariant). Not bitwise equal to the software path.
// KT2N_AMD_WMMA_EST=0 (or KT2N_AMD_WMMA=0) keeps the software path.
#pragma once

#include <cstdlib>

#include "kt2n/hip/wmma.cuh"

namespace kt2n::render::est {

template <int STG, int MTQ = 1, bool SW = false>
__global__ void __launch_bounds__(128) est_attn2_wmma_kernel(AttnArgs a) {
#if KT2N_WMMA_ARCH
  static_assert(!SW, "WMMA attention: padded (unswizzled) K/V tiles only");
  namespace wm = kt2n::wmma;
  // Per-warp hand-off buffers: S (16 x 32 fp32, WMMA -> mma.sync layout), P (16 x 32 bf16, the A operand of P.V),
  // the per-row rescale factors (16).
  __shared__ __align__(16) float w_s[4][16 * kA2Tile];
  __shared__ __align__(16) bf16 w_p[4][16 * kA2Tile];
  __shared__ float w_alpha[4][16];
  constexpr int QG = 16 * MTQ;
  constexpr int LDK = attn2_kv_ld<SW>();
  constexpr int GPI = 64 / QG;  // groups per 64-query item
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* qs = reinterpret_cast<bf16*>(smem_raw);
  bf16* kvbase = qs + QG * kA2Ld;
  // The work item, its row and the row's page list come from est_prep / the host (complete before
  // this grid can launch), so they are read before the dependency wait: after it, only Q and K/V reads
  // remain, with no page-table round trip in front of each K/V tile.
  __shared__ int pages_s[kA2MaxPages];
  EST_TR_BEGIN
  const int item_idx = blockIdx.x / GPI, grp = blockIdx.x % GPI, h = blockIdx.y;
  if (item_idx >= *a.n_items) return;
  const AttnItem it = a.items[item_idx];
  const EstRow e = a.erows[it.row];
  const int qg0 = it.q0 + grp * QG;  // first local query of this group
  if (qg0 >= e.suffix) return;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int kend_grp = key_end_of(e, min(qg0 + QG - 1, e.suffix - 1));
  const int n_tiles = (kend_grp + kA2Tile - 1) / kA2Tile;
  const int n_pg = (kend_grp + kPageFrames - 1) / kPageFrames;
  const bool pages_in_smem = n_pg <= kA2MaxPages;
  if (pages_in_smem)
    for (int i = tid; i < n_pg; i += 128) pages_s[i] = a.page_table[e.page_off + i];
  __syncthreads();
  pdl_wait();
  if (a.trig_early) pdl_trigger();  // wait -> trigger -> work (dependents' prologues overlap this kernel)
  EST_TR(1);
  bf16* ks = kvbase + size_t(warp) * STG * 2 * kA2Tile * LDK;  // [stage][K|V][32][LDK]

  for (int chunk = tid; chunk < QG * 8; chunk += 128) {
    const int m = chunk >> 3, c = (chunk & 7) * 8;
    const int t = qg0 + m;
    const bool ok = t < e.suffix;
    cp_async16(qs + m * kA2Ld + c, a.q + size_t(e.fr_off + (ok ? t : qg0)) * kInner + h * 64 + c, ok);
  }
  cp_async_commit();
  auto load_kv = [&](int tile, int stage) {
    const int kt0 = tile * kA2Tile;
    const int page = pages_in_smem ? pages_s[kt0 / kPageFrames] : a.page_table[e.page_off + kt0 / kPageFrames];
    const bf16* base = a.pool_layer + (size_t(page) * kPageFrames + (kt0 % kPageFrames)) * (2 * kInner) + h * 64;
    bf16* dk = ks + size_t(stage) * 2 * kA2Tile * LDK;
    bf16* dv = dk + kA2Tile * LDK;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int chunk = lane + i * 32;  // 32 keys x 8 chunks
      const int k = chunk >> 3, c = (chunk & 7) * 8;
      const bool ok = kt0 + k < kend_grp;
      const bf16* src = base + size_t(ok ? k : 0) * (2 * kInner) + c;
      cp_async16(dk + attn2_kv_off<SW>(k, chunk & 7), src, ok);
      cp_async16(dv + attn2_kv_off<SW>(k, chunk & 7), src + kInner, ok);
    }
  };
  int my = warp;  // this warp's next tile
  // prologue: STG - 1 tiles in flight (one for STG = 1), each its own cp.async group after Q's
  constexpr int PRE = STG > 1 ? STG - 1 : 1;
#pragma unroll
  for (int j = 0; j < PRE; ++j) {
    if (my + 4 * j < n_tiles) load_kv(my + 4 * j, j);
    cp_async_commit();
  }
  cp_async_wait<PRE - 1>();  // Q and this warp's first tile have landed (later prologue tiles may not)
  __syncthreads();  // Q visible to every warp
  EST_TR(2);

  const int g = lane >> 2, tq = lane & 3;
  int kend[MTQ][2];
  float m_i[MTQ][2], l_i[MTQ][2];
  float o[MTQ][8][4];
#pragma unroll
  for (int mt = 0; mt < MTQ; ++mt) {
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      kend[mt][hh] = key_end_of(e, min(qg0 + mt * 16 + g + 8 * hh, e.suffix - 1));
      m_i[mt][hh] = -1e30f;
      l_i[mt][hh] = 0.f;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) o[mt][i][j] = 0.f;
  }
  // O on the matrix unit: [mt][16-dim tile], rows = this warp's queries, columns = dims (WMMA accumulator layout).
  wm::acc_t ow[MTQ][4];
#pragma unroll
  for (int mt = 0; mt < MTQ; ++mt)
#pragma unroll
    for (int j = 0; j < 4; ++j) ow[mt][j] = wm::zero_acc();
  const int wl = wm::lane(), wi = wm::op_index(wl), wk0 = wm::op_k0(wl);
  float* ws_s = w_s[warp];
  bf16* ws_p = w_p[warp];
  float* ws_a = w_alpha[warp];
  const float sl2 = a.scale * 1.4426950408889634f;

  int stage = 0, stage_ld = STG - 1;  // stage computed / stage loaded this iteration (rotating)
  for (int it2 = 0; my < n_tiles; ++it2, my += 4) {
    if constexpr (STG >= 2) {
      // the stage loaded now was computed in the previous iteration; the __syncwarp at its end freed it
      if (my + 4 * (STG - 1) < n_tiles) load_kv(my + 4 * (STG - 1), stage_ld);
      cp_async_commit();
      cp_async_wait<STG - 1>();
    } else {
      if (it2 > 0) {
        load_kv(my, 0);
        cp_async_commit();
      }
      cp_async_wait<0>();
    }
    __syncwarp();
    const bf16* kst = ks + size_t(stage) * 2 * kA2Tile * LDK;
    const bf16* vst = kst + kA2Tile * LDK;
    const int kt0 = my * kA2Tile;
#pragma unroll
    for (int mt = 0; mt < MTQ; ++mt) {
      float s[4][4];
      {
        // S = Q K^T for this tile's 32 keys: 2 WMMA tiles (16 keys each), chained over the 4 16-dim blocks in order.
        wm::acc_t sw[2] = {wm::zero_acc(), wm::zero_acc()};
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          const wm::words_t qa = wm::load_run(qs + (mt * 16 + wi) * kA2Ld + kk * 16 + wk0);
#pragma unroll
          for (int np = 0; np < 2; ++np) {
            const wm::words_t kb = wm::load_run(kst + attn2_kv_off<SW>(np * 16 + wi, kk * 2) + wk0);
            sw[np] = wm::mma(qa, kb, sw[np]);
          }
        }
        // -> the mma.sync accumulator layout the softmax below reads: s[nt][u] = S[g + 8 (u >> 1)][8 nt + 2 tq + (u & 1)]
#pragma unroll
        for (int np = 0; np < 2; ++np)
#pragma unroll
          for (int v = 0; v < 8; ++v) ws_s[wm::acc_row(v, wl) * kA2Tile + np * 16 + wm::acc_col(wl)] = sw[np][v];
        __syncwarp();
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
          for (int u = 0; u < 4; ++u) s[nt][u] = ws_s[(g + 8 * (u >> 1)) * kA2Tile + nt * 8 + 2 * tq + (u & 1)];
        __syncwarp();
      }
      float mx[2] = {m_i[mt][0], m_i[mt][1]};
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int hh = u >> 1;
          const int key = kt0 + nt * 8 + 2 * tq + (u & 1);
          const float val = (key < kend[mt][hh]) ? s[nt][u] * sl2 : -1e30f;
          s[nt][u] = val;
          mx[hh] = fmaxf(mx[hh], val);
        }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffffu, mx[hh], 1));
        mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffffu, mx[hh], 2));
      }
      float alpha[2], rs[2] = {0.f, 0.f};
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) alpha[hh] = exp2f(m_i[mt][hh] - mx[hh]);
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int hh = u >> 1;
          const float p = (s[nt][u] <= -1e29f) ? 0.f : exp2f(s[nt][u] - mx[hh]);
          s[nt][u] = p;
          rs[hh] += p;
        }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        rs[hh] += __shfl_xor_sync(0xffffffffu, rs[hh], 1);
        rs[hh] += __shfl_xor_sync(0xffffffffu, rs[hh], 2);
        l_i[mt][hh] = l_i[mt][hh] * alpha[hh] + rs[hh];
        m_i[mt][hh] = mx[hh];
      }
      {
        // Rescale O by alpha of its row (rows g / g + 8 -> alpha[0] / alpha[1] of quad g), P to LDS as bf16 (the
        // values pack_bf16 gives), then O += P V: 2 key blocks x 4 dim tiles of WMMA, V as a strided operand.
        if (tq == 0) {
          ws_a[g] = alpha[0];
          ws_a[g + 8] = alpha[1];
        }
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
          for (int u = 0; u < 4; ++u)
            ws_p[(g + 8 * (u >> 1)) * kA2Tile + nt * 8 + 2 * tq + (u & 1)] = __float2bfloat16_rn(s[nt][u]);
        __syncwarp();
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          const float al = ws_a[wm::acc_row(v, wl)];
#pragma unroll
          for (int dt = 0; dt < 4; ++dt) ow[mt][dt][v] *= al;
        }
        const uint16_t* vst16 = reinterpret_cast<const uint16_t*>(vst);
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          const wm::words_t pa = wm::load_run(ws_p + wi * kA2Tile + kk * 16 + wk0);
#pragma unroll
          for (int dt = 0; dt < 4; ++dt) {
            const int dim = dt * 16 + wi;
            wm::words_t vb;
#pragma unroll
            for (int j = 0; j < wm::kWords; ++j) {
              const int key = kk * 16 + wk0 + 2 * j;
              const uint32_t lo = vst16[attn2_kv_off<SW>(key, dim >> 3) + (dim & 7)];
              const uint32_t hi = vst16[attn2_kv_off<SW>(key + 1, dim >> 3) + (dim & 7)];
              vb[j] = lo | (hi << 16);
            }
            ow[mt][dt] = wm::mma(pa, vb, ow[mt][dt]);
          }
        }
        __syncwarp();
      }
    }
    __syncwarp();
    if constexpr (STG >= 2) {
      stage = (stage + 1 == STG) ? 0 : stage + 1;
      stage_ld = (stage_ld + 1 == STG) ? 0 : stage_ld + 1;
    }
  }
  cp_async_wait<0>();
  EST_TR(3);
  if (!a.trig_early) pdl_trigger();
  // Merge the four warps in order through shared memory (each warp writes into its own K/V region).
  float* po = reinterpret_cast<float*>(ks);   // [QG][64]
  float* pml = po + QG * 64;                  // [QG][2]
  static_assert(size_t(QG) * 66 * 4 <= size_t(STG) * 2 * kA2Tile * LDK * 2, "merge buffer must fit the warp region");
  (void)o;
#pragma unroll
  for (int mt = 0; mt < MTQ; ++mt) {
#pragma unroll
    for (int dt = 0; dt < 4; ++dt)
#pragma unroll
      for (int v = 0; v < 8; ++v) po[(mt * 16 + wm::acc_row(v, wl)) * 64 + dt * 16 + wm::acc_col(wl)] = ow[mt][dt][v];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int row = mt * 16 + g + 8 * hh;
      if (tq == 0) { pml[row * 2] = m_i[mt][hh]; pml[row * 2 + 1] = l_i[mt][hh]; }
    }
  }
  __syncthreads();
  constexpr int TPR = 128 / QG;   // threads per row
  constexpr int DPT = 64 / TPR;   // dims per thread
  const int row = tid / TPR, d0 = (tid % TPR) * DPT;
  const int t = qg0 + row;
  if (t >= e.suffix) return;
  const size_t wstride = size_t(STG) * 2 * kA2Tile * LDK / 2;  // floats per warp region
  const float* wbase = reinterpret_cast<const float*>(kvbase);
  float mm = -1e30f;
#pragma unroll
  for (int w = 0; w < 4; ++w) mm = fmaxf(mm, wbase[w * wstride + QG * 64 + row * 2]);
  float L = 0.f, acc[DPT];
#pragma unroll
  for (int i = 0; i < DPT; ++i) acc[i] = 0.f;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const float* wb = wbase + w * wstride;
    const float wt = exp2f(wb[QG * 64 + row * 2] - mm);
    L += wb[QG * 64 + row * 2 + 1] * wt;
#pragma unroll
    for (int i = 0; i < DPT; ++i) acc[i] += wb[row * 64 + d0 + i] * wt;
  }
  const float inv = 1.0f / fmaxf(L, 1e-30f);
#pragma unroll
  for (int i0 = 0; i0 < DPT; i0 += 8) {
    uint32_t pk[4];
#pragma unroll
    for (int i = 0; i < 8; i += 2) pk[i / 2] = pack_bf16(acc[i0 + i] * inv, acc[i0 + i + 1] * inv);
    *reinterpret_cast<uint4*>(a.out + size_t(e.fr_off + t) * kInner + h * 64 + d0 + i0) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
  }
  EST_TR_END(30);
#else
  (void)a;
#if defined(__HIP_DEVICE_COMPILE__)
  __builtin_trap();
#endif
#endif
}

// KT2N_AMD_WMMA_EST_ATTN=0 keeps only the attention on the software path (A/B); read once.
inline bool est_attn_wmma_off() {
  static const bool off = [] {
    const char* e = std::getenv("KT2N_AMD_WMMA_EST_ATTN");
    return e != nullptr && e[0] == '0' && e[1] == '\0';
  }();
  return off;
}

// The launch shape estimator.cu uses on AMD (STG = 1, 16 queries per CTA): dynamic shared memory as the software kernel,
// plus 12.75 KB static for the hand-off buffers (52 KB in all, under the 64 KB per workgroup).
inline void est_attn_wmma_prepare() {
  (void)cudaFuncSetAttribute(est_attn2_wmma_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             int(attn2_smem_bytes<1>()));
}

}  // namespace kt2n::render::est
