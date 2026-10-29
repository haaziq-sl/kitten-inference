// Flow estimator convolutions on the AMD matrix unit (est_conv_wmma_kernel, src/render/estimator/est_wmma.cuh) against
// the portable software path (est_conv_kernel<MODE, 2, true> + est_conv_portable_epilogue<MODE>, the code every HIP
// build runs without WMMA). No model assets required.
//
// For every mode (0: resnet block1 + time embedding, 1: block2 + 1x1 residual conv, 2: plain down / up conv, 3: final
// block) with the estimator's channel counts (block1 input 320 and 256 + 256 concat; residual input 320 and 256 + 256),
// one row at F = 64 and 256 frames, masked tail frames, a past (conv tail read) and a tail save:
//   vs software  bitwise-equal fraction, max bf16 ulp and max |difference| of `out`; PASS if >= 99.9% bitwise and
//                every element within 4 ulp or within 2^-6 of the output rms (near-zero outputs make ulp counts large)
//   tail stage   the saved conv-input tail equals the software path's (it is a copy of the input window)
//   invariance   frames 0..63 of the WMMA output are bitwise the same at F = 64 and F = 256 (same row content)
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
  ~Dev() { (void)cudaFree(p); }
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
  std::printf("%-52s %s  %s\n", name.c_str(), ok ? "PASS" : "FAIL", detail.c_str());
  if (!ok) ++g_failures;
}

uint16_t f2bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float bf2f(uint16_t b) {
  const uint32_t u = uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
int ulp16(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7fff) : int(v); };
  return std::abs(key(a) - key(b));
}

constexpr int kFmax = 256;
constexpr int kCap = 65536;

std::mt19937 g_rng{4242};
std::vector<uint16_t> rnd(size_t n, float sd, float mean = 0.f) {
  std::normal_distribution<float> d(mean, sd);
  std::vector<uint16_t> v(n);
  for (auto& e : v) e = f2bf(d(g_rng));
  return v;
}

struct Case {
  const char* name;
  int mode;
  int c0, c1;    // conv input channels (in1 = concat)
  int rc0, rc1;  // MODE 1: residual input channels
};

// One conv call's inputs (shared by both paths and every F).
struct Data {
  Case cs;
  int cin, cres;
  Dev<uint16_t> in0, in1, w, bias, lng, lnb, temb, rin0, rin1, rw, rb, tail;
  Data(const Case& c)
      : cs(c), cin(c.c0 + c.c1), cres(c.rc0 + c.rc1),
        in0(rnd(size_t(kFmax) * std::max(c.c0, 1), 1.f)), in1(rnd(size_t(kFmax) * std::max(c.c1, 1), 1.f)),
        w(rnd(size_t(256) * 3 * (c.c0 + c.c1), 1.f / std::sqrt(3.f * (c.c0 + c.c1)))), bias(rnd(256, 0.1f)),
        lng(rnd(256, 0.2f, 1.f)), lnb(rnd(256, 0.1f)), temb(rnd(256, 0.5f)),
        rin0(rnd(size_t(kFmax) * std::max(c.rc0, 1), 1.f)), rin1(rnd(size_t(kFmax) * std::max(c.rc1, 1), 1.f)),
        rw(rnd(size_t(256) * std::max(c.rc0 + c.rc1, 1), 1.f / std::sqrt(float(std::max(c.rc0 + c.rc1, 1))))),
        rb(rnd(256, 0.1f)), tail(rnd(size_t(2) * (c.c0 + c.c1), 1.f)) {}
};

struct Out {
  std::vector<uint16_t> out, stage;
};

// Runs one conv over one row of F frames (valid frames `valid`, `past` frames before it: the tail is read when > 0,
// tail saved at shift = valid - 3) on the software path or the WMMA path.
Out run(const Data& d, int F, int valid, int past, bool wmma) {
  est::EstRow row{};
  row.fr_off = 0;
  row.past = past;
  row.suffix = F;
  row.valid = valid;
  row.page_off = 0;
  row.slot = 0;
  row.shift = valid - 3;
  row.full = 0;
  Dev<est::EstRow> erows(std::vector<est::EstRow>{row});
  Dev<int32_t> frame_row(std::vector<int32_t>(F, 0));
  Dev<uint16_t> out(size_t(F) * 256), pv(size_t(F) * 256), pr(size_t(F) * 256), stage(size_t(2) * d.cin);
  est::ConvArgs a{};
  auto bf = [](const Dev<uint16_t>& v) { return reinterpret_cast<const est::bf16*>(v.p); };
  a.in0 = bf(d.in0); a.c0 = d.cs.c0; a.ld0 = d.cs.c0;
  a.in1 = d.cs.c1 ? bf(d.in1) : nullptr; a.c1 = d.cs.c1; a.ld1 = d.cs.c1;
  a.w = bf(d.w); a.bias = bf(d.bias); a.ln_g = bf(d.lng); a.ln_b = bf(d.lnb); a.temb = bf(d.temb);
  a.res_in0 = d.cs.rc0 ? bf(d.rin0) : nullptr; a.res_c0 = d.cs.rc0; a.res_ld0 = d.cs.rc0;
  a.res_in1 = d.cs.rc1 ? bf(d.rin1) : nullptr; a.res_c1 = d.cs.rc1; a.res_ld1 = d.cs.rc1;
  a.res_w = d.cres ? bf(d.rw) : nullptr; a.res_b = bf(d.rb);
  a.out = reinterpret_cast<est::bf16*>(out.p); a.ld_out = 256;
  a.erows = erows.p; a.frame_row = frame_row.p;
  a.tail_state = bf(d.tail); a.tail_slot_stride = 2 * d.cin;
  a.tail_stage = reinterpret_cast<est::bf16*>(stage.p); a.stage_row_stride = 2 * d.cin;
  a.trig_early = 0;
  a.portable_values = reinterpret_cast<est::bf16*>(pv.p);
  a.portable_residual = reinterpret_cast<est::bf16*>(pr.p);
  const dim3 grid(unsigned(F / est::kConvBM), unsigned(est::kConvCluster));
  const size_t bytes = est::conv_smem_bytes(d.cin, d.cres, 2);
  if (wmma) {
    if (!est::est_conv_wmma(d.cs.mode, grid, bytes, kCap, nullptr, a, F)) throw std::runtime_error("est_conv_wmma refused");
  } else {
    const dim3 eg(unsigned(F / est::kConvBM));
    switch (d.cs.mode) {
      case 0:
        est::est_conv_kernel<0, 2, true><<<grid, dim3(est::kConvThreads), bytes>>>(a);
        est::est_conv_portable_epilogue<0><<<eg, dim3(512)>>>(a);
        break;
      case 1:
        est::est_conv_kernel<1, 2, true><<<grid, dim3(est::kConvThreads), bytes>>>(a);
        est::est_conv_portable_epilogue<1><<<eg, dim3(512)>>>(a);
        break;
      case 2:
        est::est_conv_kernel<2, 2, true><<<grid, dim3(est::kConvThreads), bytes>>>(a);
        break;
      default:
        est::est_conv_kernel<3, 2, true><<<grid, dim3(est::kConvThreads), bytes>>>(a);
        est::est_conv_portable_epilogue<3><<<eg, dim3(512)>>>(a);
    }
    CHECK(cudaGetLastError());
  }
  CHECK(cudaDeviceSynchronize());
  return {out.get(), stage.get()};
}

void set_attrs() {
  auto set = [](auto k) { CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, kCap)); };
  set(est::est_conv_kernel<0, 2, true>);
  set(est::est_conv_kernel<1, 2, true>);
  set(est::est_conv_kernel<2, 2, true>);
  set(est::est_conv_kernel<3, 2, true>);
  est::est_wmma_prepare(kCap);
}

}  // namespace

int main() {
  try {
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    const kt2n::cuda::DeviceCaps caps = kt2n::cuda::device_caps(0);
    std::printf("device %s (%s), wmma %d\n", prop.name, prop.gcnArchName, caps.wmma);
    if (caps.wmma == 0 || !est::est_wmma_on()) {
      std::printf("no WMMA on this GPU (or KT2N_AMD_WMMA=0 / KT2N_AMD_WMMA_EST=0): skipped\n");
      return 77;
    }
    set_attrs();
    const Case cases[] = {{"block1 cin 320", 0, 320, 0, 0, 0},     {"block1 cin 256+256", 0, 256, 256, 0, 0},
                          {"block2 res 320", 1, 256, 0, 320, 0},   {"block2 res 256+256", 1, 256, 0, 256, 256},
                          {"plain 256", 2, 256, 0, 0, 0},          {"final 256", 3, 256, 0, 0, 0}};
    for (const Case& cs : cases) {
      Data d(cs);
      for (int F : {64, 256}) {
        for (int past : {0, 10}) {
          const Out sw = run(d, F, F - 3, past, false);
          const Out wm = run(d, F, F - 3, past, true);
          size_t same = 0;
          int maxu = 0;
          double maxabs = 0, ss = 0, at = 0;
          for (size_t i = 0; i < wm.out.size(); ++i) {
            same += wm.out[i] == sw.out[i];
            const int u = ulp16(wm.out[i], sw.out[i]);
            const double a = bf2f(sw.out[i]), b = bf2f(wm.out[i]);
            ss += a * a;
            if (std::fabs(a - b) > maxabs) {
              maxabs = std::fabs(a - b);
              at = a;
            }
            maxu = std::max(maxu, u);
          }
          const double rms = std::sqrt(ss / double(wm.out.size()));
          // Per element: within 4 bf16 ulp of the software value, or (outputs near zero: Mish, + time embedding,
          // cancellation, where ulp counts explode for tiny changes) within 2^-6 of the output rms.
          size_t bad = 0;
          for (size_t i = 0; i < wm.out.size(); ++i) {
            const double a = bf2f(sw.out[i]), b = bf2f(wm.out[i]);
            if (ulp16(wm.out[i], sw.out[i]) > 4 && std::fabs(a - b) > rms / 64.0) ++bad;
          }
          char name[128], det[200];
          std::snprintf(name, sizeof(name), "%s F=%d past=%d vs software", cs.name, F, past);
          std::snprintf(det, sizeof(det), "%.4f%% of %zu bitwise, max %d ulp, max |d| %.3g (at %.3g) = %.2g x rms",
                        100.0 * same / wm.out.size(), wm.out.size(), maxu, maxabs, at, maxabs / rms);
          report(name, same >= size_t(0.999 * double(wm.out.size())) && bad == 0, det);
          std::snprintf(name, sizeof(name), "%s F=%d past=%d tail stage", cs.name, F, past);
          report(name, wm.stage == sw.stage, wm.stage == sw.stage ? "bitwise" : "differs");
        }
      }
      // Batch invariance: the same row content (valid 61) at F = 64 and F = 256.
      const Out a64 = run(d, 64, 61, 10, true), a256 = run(d, 256, 61, 10, true);
      const bool eq = std::memcmp(a64.out.data(), a256.out.data(), size_t(64) * 256 * 2) == 0;
      report(std::string(cs.name) + " WMMA F=64 == F=256 frames 0..63", eq, eq ? "bitwise" : "differs");
    }
  } catch (const std::exception& e) {
    std::printf("ERROR %s\n", e.what());
    return 2;
  }
  std::printf("%s (%d failure%s)\n", g_failures ? "FAILED" : "all estimator conv WMMA checks passed", g_failures,
              g_failures == 1 ? "" : "s");
  return g_failures ? 1 : 0;
}
