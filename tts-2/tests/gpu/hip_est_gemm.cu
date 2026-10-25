// Flow estimator GEMMs on the AMD matrix unit (src/render/estimator/est_wmma.cuh) against the portable software path
// (est_gemm_kernel, the code GTX 10 / 16 and every HIP build run without WMMA). No model assets required.
//
// For every mode (QKV with the K/V pool write, OUT, FF1, FF2, FINAL with the Euler step) at F = 64 and 256 frames of one
// row, same shape (64 x 64, BK 64, 3 stages) for both kernels:
//   vs software  bitwise-equal fraction and max bf16 ulp (fp32 ulp-scale for FINAL's fp32 outputs); PASS if >= 99.9% bitwise and <= 16 ulp
//   invariance   the WMMA kernel gives the F = 256 bits for frames 0..63 at F = 64, and for every WMMA shape the
//                dispatch can pick (exact memcmp)
//   dispatch     gemm_wmma() picks a shape and launches for every mode at F = 64 .. 1024
//
// Exit status 0 = every check passed (77 = no WMMA GPU: skipped).
#include "kt2n/cuda_caps.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "est_kernels.cuh"
#include "est_wmma.cuh"

namespace est = kt2n::render::est;

#define CHECK(call)                                                                   \
  do {                                                                                \
    const cudaError_t err_ = (call);                                                  \
    if (err_ != cudaSuccess)                                                          \
      throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(err_)); \
  } while (0)

namespace {

template <typename T>
struct Dev {
  T* p = nullptr;
  size_t n = 0;
  explicit Dev(size_t count) : n(count) {
    CHECK(cudaMalloc(&p, (n ? n : 1) * sizeof(T)));
    CHECK(cudaMemset(p, 0, (n ? n : 1) * sizeof(T)));
  }
  explicit Dev(const std::vector<T>& h) : Dev(h.size()) { put(h); }
  ~Dev() { cudaFree(p); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
  void put(const std::vector<T>& h) { CHECK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); }
  std::vector<T> get() const {
    std::vector<T> h(n);
    CHECK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
};

int g_failures = 0;
void report(const std::string& name, bool ok, const std::string& detail) {
  std::printf("%-44s %s  %s\n", name.c_str(), ok ? "PASS" : "FAIL", detail.c_str());
  if (!ok) ++g_failures;
}

uint32_t f2u(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
uint16_t f2bf(float f) {
  const uint32_t u = f2u(f);
  return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
int ulp16(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7fff) : int(v); };
  return std::abs(key(a) - key(b));
}
int64_t ulp32(float a, float b) {
  auto key = [](float v) {
    const int64_t u = f2u(v);
    return (u & 0x80000000) ? -(u & 0x7fffffff) : u;
  };
  return std::llabs(key(a) - key(b));
}

constexpr int kFmax = 1024;
constexpr int kPagesMax = kFmax / kt2n::render::kPageFrames;

// Inputs shared by every case (frames up to kFmax of one row), generated once.
struct Inputs {
  std::mt19937 rng{2026};
  std::vector<uint16_t> x, ao, ff, w_qkv, w_o, w_f1, w_f2, w_fin, b256, b1024, b128, lng, lnb, xin;
  std::vector<uint16_t> rnd(size_t n, float sd) {
    std::normal_distribution<float> d(0.f, sd);
    std::vector<uint16_t> v(n);
    for (auto& e : v) e = f2bf(d(rng));
    return v;
  }
  Inputs() {
    x = rnd(size_t(kFmax) * est::kHid, 1.f);
    ao = rnd(size_t(kFmax) * est::kInner, 1.f);
    ff = rnd(size_t(kFmax) * est::kFfn, 0.5f);
    w_qkv = rnd(size_t(3 * est::kInner) * est::kHid, 1.f / 16);
    w_o = rnd(size_t(est::kHid) * est::kInner, 1.f / 22.6f);
    w_f1 = rnd(size_t(est::kFfn) * est::kHid, 1.f / 16);
    w_f2 = rnd(size_t(est::kHid) * est::kFfn, 1.f / 32);
    w_fin = rnd(size_t(128) * est::kHid, 1.f / 16);
    b256 = rnd(256, 0.1f);
    b1024 = rnd(1024, 0.1f);
    b128 = rnd(128, 0.1f);
    lng = rnd(256, 0.2f);
    for (auto& e : lng) {  // gamma ~ 1 + N(0, 0.2)
      uint32_t u = uint32_t(e) << 16;
      float v;
      std::memcpy(&v, &u, 4);
      e = f2bf(1.f + v);
    }
    lnb = rnd(256, 0.1f);
    xin = rnd(size_t(kFmax) * est::kIn, 1.f);
  }
};

// Device state of one run: all outputs a mode can write, reset from the inputs before each launch.
struct Run {
  const Inputs& in;
  Dev<uint16_t> x, ao, ff, w_qkv, w_o, w_f1, w_f2, w_fin, b256, b1024, b128, lng, lnb, xin, q, pool;
  Dev<float> deriv, mel;
  Dev<est::EstRow> erows;
  Dev<int32_t> frame_row, page_table;
  explicit Run(const Inputs& i)
      : in(i), x(i.x), ao(i.ao), ff(i.ff), w_qkv(i.w_qkv), w_o(i.w_o), w_f1(i.w_f1), w_f2(i.w_f2), w_fin(i.w_fin),
        b256(i.b256), b1024(i.b1024), b128(i.b128), lng(i.lng), lnb(i.lnb), xin(i.xin),
        q(size_t(kFmax) * est::kInner), pool(size_t(kPagesMax) * kt2n::render::kPageFrames * 2 * est::kInner),
        deriv(size_t(kFmax) * est::kMelC), mel(size_t(kFmax) * est::kMelC), erows(1), frame_row(kFmax),
        page_table(kPagesMax) {
    std::vector<int32_t> pt(kPagesMax);
    for (int p = 0; p < kPagesMax; ++p) pt[p] = kPagesMax - 1 - p;  // pages out of order, as a real pool may be
    page_table.put(pt);
  }
  // One row of F frames: suffix F, valid min(F, 250) (FINAL masks frames 250.. of the 256-frame call, and frames
  // 0..63 have the same mask at every F), no past.
  void set_rows(int F) {
    est::EstRow r{};
    r.fr_off = 0;
    r.past = 0;
    r.suffix = F;
    r.valid = std::min(F, 250);
    r.page_off = 0;
    r.slot = 0;
    r.shift = 0;
    r.full = 0;
    erows.put({r});
    std::vector<int32_t> fr(kFmax, -1);
    for (int f = 0; f < F; ++f) fr[f] = 0;
    frame_row.put(fr);
  }
  void reset() {
    x.put(in.x);
    ff.put(in.ff);
    xin.put(in.xin);
    CHECK(cudaMemset(q.p, 0, q.n * 2));
    CHECK(cudaMemset(pool.p, 0, pool.n * 2));
    CHECK(cudaMemset(deriv.p, 0, deriv.n * 4));
    CHECK(cudaMemset(mel.p, 0, mel.n * 4));
  }
  est::GemmArgs args(int mode, int F) {
    est::GemmArgs g{};
    auto bf = [](Dev<uint16_t>& d) { return reinterpret_cast<__nv_bfloat16*>(d.p); };
    g.erows = erows.p;
    g.frame_row = frame_row.p;
    g.page_table = page_table.p;
    g.frames_cap = F;
    g.trig_early = 0;
    switch (mode) {
      case est::GM_QKV:
        g.a = bf(x); g.lda = est::kHid; g.k = est::kHid; g.w = bf(w_qkv); g.bias = nullptr;
        g.ln_g = bf(lng); g.ln_b = bf(lnb); g.n = 3 * est::kInner; g.out = bf(q); g.ld_out = est::kInner;
        g.pool_layer = bf(pool);
        break;
      case est::GM_OUT:
        g.a = bf(ao); g.lda = est::kInner; g.k = est::kInner; g.w = bf(w_o); g.bias = bf(b256);
        g.n = est::kHid; g.out = bf(x); g.ld_out = est::kHid;
        break;
      case est::GM_FF1:
        g.a = bf(x); g.lda = est::kHid; g.k = est::kHid; g.w = bf(w_f1); g.bias = bf(b1024);
        g.ln_g = bf(lng); g.ln_b = bf(lnb); g.n = est::kFfn; g.out = bf(ff); g.ld_out = est::kFfn;
        break;
      case est::GM_FF2:
        g.a = bf(ff); g.lda = est::kFfn; g.k = est::kFfn; g.w = bf(w_f2); g.bias = bf(b256);
        g.n = est::kHid; g.out = bf(x); g.ld_out = est::kHid;
        break;
      default:
        g.a = bf(x); g.lda = est::kHid; g.k = est::kHid; g.w = bf(w_fin); g.bias = bf(b128); g.n = 128;
        g.xin = bf(xin); g.dt = 0.5f; g.deriv_out = deriv.p; g.mel_out = mel.p;
        break;
    }
    return g;
  }
  // What a mode wrote, as comparable bit vectors (bf16 outputs as uint16, fp32 outputs as their bits).
  struct Out {
    std::vector<uint16_t> b;
    std::vector<float> f;
  };
  Out outputs(int mode, int F) {
    Out o;
    auto take = [&](const Dev<uint16_t>& d, size_t n) {
      std::vector<uint16_t> v = d.get();
      o.b.insert(o.b.end(), v.begin(), v.begin() + n);
    };
    switch (mode) {
      case est::GM_QKV: {
        take(q, size_t(F) * est::kInner);
        // pool frames 0..F-1 in frame order (through the page table)
        const std::vector<uint16_t> pl = pool.get();
        for (int f = 0; f < F; ++f) {
          const int page = kPagesMax - 1 - f / kt2n::render::kPageFrames;
          const size_t off = (size_t(page) * kt2n::render::kPageFrames + f % kt2n::render::kPageFrames) * 2 * est::kInner;
          o.b.insert(o.b.end(), pl.begin() + off, pl.begin() + off + 2 * est::kInner);
        }
        break;
      }
      case est::GM_OUT:
      case est::GM_FF2: take(x, size_t(F) * est::kHid); break;
      case est::GM_FF1: take(ff, size_t(F) * est::kFfn); break;
      default: {
        take(xin, size_t(F) * est::kIn);
        const std::vector<float> d = deriv.get(), m = mel.get();
        o.f.insert(o.f.end(), d.begin(), d.begin() + size_t(F) * est::kMelC);
        o.f.insert(o.f.end(), m.begin(), m.begin() + size_t(F) * est::kMelC);
        break;
      }
    }
    return o;
  }
};

template <int MODE>
void launch_sw(const est::GemmArgs& g, int F) {  // the portable est_gemm_kernel, 64 x 64 tiles
  constexpr size_t smem = est::gemm_smem_bytes<64, 64, 64, 3, MODE>();
  CHECK(cudaFuncSetAttribute(est::est_gemm_kernel<64, 64, 2, 64, 3, MODE, 1>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
  est::est_gemm_kernel<64, 64, 2, 64, 3, MODE, 1><<<dim3(g.n / 64, F / 64), dim3(256), smem>>>(g);
  CHECK(cudaGetLastError());
}
template <int MODE>
void launch_wmma(const est::GemmArgs& g, int F) {
  est::est_wmma_attr<64, 64, 2, 64, 3, MODE>(65536);
  est::est_wmma_launch<64, 64, 2, 64, 3, MODE>(nullptr, g, F);
}
void launch_mode(bool wmma, int mode, const est::GemmArgs& g, int F) {
  switch (mode) {
    case est::GM_QKV: wmma ? launch_wmma<est::GM_QKV>(g, F) : launch_sw<est::GM_QKV>(g, F); break;
    case est::GM_OUT: wmma ? launch_wmma<est::GM_OUT>(g, F) : launch_sw<est::GM_OUT>(g, F); break;
    case est::GM_FF1: wmma ? launch_wmma<est::GM_FF1>(g, F) : launch_sw<est::GM_FF1>(g, F); break;
    case est::GM_FF2: wmma ? launch_wmma<est::GM_FF2>(g, F) : launch_sw<est::GM_FF2>(g, F); break;
    default: wmma ? launch_wmma<est::GM_FINAL>(g, F) : launch_sw<est::GM_FINAL>(g, F); break;
  }
  CHECK(cudaDeviceSynchronize());
}

const char* mode_name(int mode) {
  static const char* n[5] = {"QKV", "OUT", "FF1", "FF2", "FINAL"};
  return n[mode];
}

void test_modes(Run& run) {
  for (int mode = 0; mode < 5; ++mode) {
    Run::Out w256;
    for (int F : {256, 64}) {
      run.set_rows(F);
      run.reset();
      launch_mode(false, mode, run.args(mode, F), F);
      const Run::Out sw = run.outputs(mode, F);
      run.reset();
      launch_mode(true, mode, run.args(mode, F), F);
      const Run::Out wm = run.outputs(mode, F);
      size_t same = 0;
      int max16 = 0;
      int64_t max32 = 0;
      for (size_t i = 0; i < wm.b.size(); ++i) {
        same += wm.b[i] == sw.b[i];
        max16 = std::max(max16, ulp16(wm.b[i], sw.b[i]));
      }
      size_t same_f = 0;
      for (size_t i = 0; i < wm.f.size(); ++i) {
        same_f += f2u(wm.f[i]) == f2u(sw.f[i]);
        max32 = std::max(max32, ulp32(wm.f[i], sw.f[i]));
      }
      char name[96], det[200];
      std::snprintf(name, sizeof(name), "%s F=%d vs software path", mode_name(mode), F);
      if (wm.f.empty())
        std::snprintf(det, sizeof(det), "%.4f%% of %zu bf16 outputs bitwise, max %d ulp", 100.0 * same / wm.b.size(),
                      wm.b.size(), max16);
      else
        std::snprintf(det, sizeof(det), "x: %.4f%% bitwise, max %d bf16 ulp; deriv/mel fp32: %.4f%% bitwise, max %lld ulp",
                      100.0 * same / wm.b.size(), max16, 100.0 * same_f / wm.f.size(), (long long)max32);
      // FINAL's fp32 deriv = bf16 values; mel = bf16 values: a 1-2 bf16-ulp change is 2^16-2^17 fp32 ulps.
      // A 1-ulp change of a bf16 pre-activation can move GELU's output by several ulps where its value is small
      // (gfx11's accumulation differs from the software path more often than gfx12's), so: >= 99.9% bitwise and
      // <= 16 ulp anywhere.
      const bool frac_ok = same >= size_t(0.999 * double(wm.b.size())) &&
                           (wm.f.empty() || same_f >= size_t(0.999 * double(wm.f.size())));
      report(name, frac_ok && max16 <= 16 && max32 <= (int64_t(2) << 16), det);
      if (F == 256) {
        w256 = wm;
      } else {
        // frames 0..63 of the F = 256 run (same row content and positions) must be bit-identical
        bool eq = true;
        const size_t per_b = wm.b.size() / 64;  // outputs per frame, in frame-major blocks for every mode
        if (mode == est::GM_QKV) {
          // q block [F][512] then pool [F][1024]
          eq = std::memcmp(wm.b.data(), w256.b.data(), size_t(64) * est::kInner * 2) == 0 &&
               std::memcmp(wm.b.data() + size_t(64) * est::kInner, w256.b.data() + size_t(256) * est::kInner,
                           size_t(64) * 2 * est::kInner * 2) == 0;
        } else if (mode == est::GM_FINAL) {
          eq = std::memcmp(wm.b.data(), w256.b.data(), size_t(64) * est::kIn * 2) == 0 &&
               std::memcmp(wm.f.data(), w256.f.data(), size_t(64) * est::kMelC * 4) == 0 &&
               std::memcmp(wm.f.data() + size_t(64) * est::kMelC, w256.f.data() + size_t(256) * est::kMelC,
                           size_t(64) * est::kMelC * 4) == 0;
        } else {
          eq = std::memcmp(wm.b.data(), w256.b.data(), per_b * 64 * 2) == 0;
        }
        std::snprintf(name, sizeof(name), "%s WMMA F=64 == F=256 frames 0..63", mode_name(mode));
        report(name, eq, eq ? "bitwise" : "differs");
      }
    }
  }
}

// gemm_wmma (the engine's dispatch) at several buckets: launches, and every shape it picks gives the 64 x 64 bits.
void test_dispatch(Run& run) {
  est::est_wmma_prepare(65536);
  for (int mode = 0; mode < 5; ++mode) {
    int bad = 0, runs = 0;
    for (int F : {64, 128, 192, 320, 384, 512, 1024}) {
      run.set_rows(F);
      run.reset();
      launch_mode(true, mode, run.args(mode, F), F);
      const Run::Out ref = run.outputs(mode, F);
      run.reset();
      const bool ok = est::gemm_wmma(nullptr, mode, run.args(mode, F), F, 65536, 320, 96);
      CHECK(cudaDeviceSynchronize());
      ++runs;
      if (!ok) {
        ++bad;
        continue;
      }
      const Run::Out got = run.outputs(mode, F);
      if (got.b != ref.b || std::memcmp(got.f.data(), ref.f.data(), got.f.size() * 4) != 0) ++bad;
    }
    char name[96], det[96];
    std::snprintf(name, sizeof(name), "%s dispatch shapes == 64x64 bits", mode_name(mode));
    std::snprintf(det, sizeof(det), "%d of %d buckets launched and bitwise equal", runs - bad, runs);
    report(name, bad == 0, det);
  }
}

}  // namespace

int main() {
  try {
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    const kt2n::cuda::DeviceCaps caps = kt2n::cuda::device_caps(0);
    std::printf("device %s (%s), %d CUs, wave %d, wmma %d\n", prop.name, prop.gcnArchName, prop.multiProcessorCount,
                prop.warpSize, caps.wmma);
    if (caps.wmma == 0) {
      std::printf("no WMMA on this GPU (or KT2N_AMD_WMMA=0): skipped\n");
      return 77;
    }
    Inputs in;
    Run run(in);
    test_modes(run);
    test_dispatch(run);
  } catch (const std::exception& e) {
    std::printf("ERROR %s\n", e.what());
    return 2;
  }
  std::printf("%s (%d failure%s)\n", g_failures ? "FAILED" : "all estimator WMMA checks passed", g_failures,
              g_failures == 1 ? "" : "s");
  return g_failures ? 1 : 0;
}
