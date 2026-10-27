// CUDA-core ternary decode projections (src/cuda/legacy/tern_dec_cc.cu) against the decode kernel with its software
// MMA, bit for bit, on GPUs without bf16 tensor-core instructions (SM 6.x / 7.5). No model assets.
//
// Every family x epilogue x M in 1..16 x activation pattern (normal, heavy-tailed, tiny products below the fp32
// normal range, cancelling pairs with signed zeros, RMSNorm-like with outliers) and two scale sets (normal, wide
// exponent range with all-zero groups): the output of kt2n_tern_cc::launch must equal kt2n_tern_proj's with
// KT2N_TERN_DECODE=mma (the old kernel), byte for byte, rows >= M untouched; M 1..4 must run the CUDA-core kernel
// and M 5..16 must be declined (the caller keeps the old kernel there). SM 8.0+ GPUs never use the CUDA-core kernel
// (it is not built for them): the test prints "skipped" and passes there. No GPU visible: skipped, or a failure
// with KT2N_REQUIRE_GPU=1.
//
// --dispatch: the default routing of kt2n_tern_proj, as KT2N_TERN_DECODE in the environment sets it (ctest runs it
// unset, =cc and =mma): M 1..4 must take the CUDA-core kernel exactly when the routing allows it (CUDA: on unless
// =mma; HIP: only with =cc; never on SM 8.0+), with output equal to a direct launch byte for byte, and M 5..16 must
// keep the old kernel with rows >= M untouched.
//
// Exit status 0 = every case identical.
#include "kt2n/cuda_caps.h"
#include "kt2n/kernels/tern_proj.h"
#include "../../src/cuda/legacy/tern_dec_cc.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));    \
      std::exit(2);                                                                              \
    }                                                                                            \
  } while (0)

namespace {

uint16_t f2bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7f800000u) == 0x7f800000u) return uint16_t(u >> 16);
  u += 0x7fffu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}
float bf2f(uint16_t h) {
  const uint32_t u = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

struct Buf {
  void* p = nullptr;
  explicit Buf(size_t n) { CK(cudaMalloc(&p, n)); }
  ~Buf() { cudaFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
};

std::vector<uint16_t> activations(int pat, size_t count, std::mt19937& rng) {
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<uint16_t> x(count);
  for (size_t i = 0; i < count; ++i) {
    float v = nd(rng);
    switch (pat) {
      case 1: v = std::ldexp(v, int(rng() % 41) - 20); break;
      case 2: v = std::ldexp(v, -100 - int(rng() % 20)); break;
      case 3: {
        const int r = int(rng() % 8);
        v = r == 0 ? 0.f : r == 1 ? -0.f : (i & 1) ? -bf2f(x[i - 1]) : v;
        break;
      }
      case 4: v = 0.3f * v * ((i % 64) == 0 ? 20.f : 1.f); break;
      default: break;
    }
    x[i] = f2bf(v);
  }
  return x;
}

// Packed weights of one family with random codes and normal scales (the dispatch check).
struct Weights {
  int n, k;
  Buf wc, ws;
  Weights(int fam, std::mt19937& rng)
      : n(fam == KT2N_TERN_FAM_QKV ? 4096 : fam == KT2N_TERN_FAM_GATE_UP ? 12288 : 2048),
        k(fam == KT2N_TERN_FAM_DOWN ? 6144 : 2048),
        wc(kt2n_tern_codes_bytes(n, k)),
        ws(kt2n_tern_scales_bytes(n, k)) {
    std::vector<int32_t> codes(size_t(n) * k / 16);
    std::vector<uint16_t> scales(size_t(n) * k / 128);
    for (auto& v : codes) {
      uint32_t b = 0;
      for (int j = 0; j < 16; ++j) b |= (rng() % 3u) << (2 * j);
      std::memcpy(&v, &b, 4);
    }
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    for (auto& v : scales) v = f2bf(0.002f + 0.03f * u01(rng));
    Buf dc(codes.size() * 4), ds(scales.size() * 2);
    CK(cudaMemcpy(dc.p, codes.data(), codes.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ds.p, scales.data(), scales.size() * 2, cudaMemcpyHostToDevice));
    if (kt2n_tern_repack(static_cast<const int32_t*>(dc.p), ds.p, n, k, wc.p, ws.p, n, 0, 1, nullptr)) std::exit(2);
  }
};

int dispatch(const kt2n::cuda::DeviceCaps& caps) {
  const char* v = std::getenv("KT2N_TERN_DECODE");
  const std::string mode = v ? v : "";
#if defined(KT2N_HIP)
  const bool allowed = mode == "cc";
#else
  const bool allowed = mode != "mma";
#endif
  const bool image = caps.legacy() && kt2n_tern_cc::available();
  long cases = 0, bad = 0, routed = 0;
  const int fams[3][2] = {{KT2N_TERN_FAM_QKV, KT2N_TERN_EPI_STORE},
                          {KT2N_TERN_FAM_GATE_UP, KT2N_TERN_EPI_SILU_MUL},
                          {KT2N_TERN_FAM_DOWN, KT2N_TERN_EPI_RESIDUAL}};
  for (const auto& fe : fams) {
    const int fam = fe[0], epi = fe[1];
    std::mt19937 rng(91u + unsigned(fam));
    const Weights w(fam, rng);
    int wk = 0, cpw = 0;
    kt2n_tern_family_split(fam, &wk, &cpw);
    const int acc = kt2n_tern_family_acc(fam), cols = epi == KT2N_TERN_EPI_SILU_MUL ? w.n / 2 : w.n;
    constexpr int kMaxM = 16;
    Buf dx(size_t(kMaxM) * w.k * 2), ya(size_t(kMaxM) * cols * 2), yb(size_t(kMaxM) * cols * 2);
    const std::vector<uint16_t> x = activations(0, size_t(kMaxM) * w.k, rng);
    const std::vector<uint16_t> r = activations(0, size_t(kMaxM) * cols, rng);
    CK(cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    const bool res = epi == KT2N_TERN_EPI_RESIDUAL;
    for (int m = 1; m <= kMaxM; ++m) {
      std::vector<uint16_t> init(size_t(kMaxM) * cols, 0x7fc1);
      if (res) init.assign(r.begin(), r.begin() + init.size());
      CK(cudaMemcpy(ya.p, init.data(), init.size() * 2, cudaMemcpyHostToDevice));
      const unsigned long before = kt2n_tern_cc::launches();
      const int rc = kt2n_tern_proj(fam, dx.p, w.k, m, w.wc.p, w.ws.p, w.n, w.k, ya.p, cols, epi, res ? ya.p : nullptr,
                                    res ? w.n : 0, 0u, nullptr);
      CK(cudaDeviceSynchronize());
      const long took = long(kt2n_tern_cc::launches() - before);
      const long want = allowed && image && m <= kt2n_tern_cc::kMaxRows ? 1 : 0;
      routed += took;
      std::vector<uint16_t> a(init.size());
      CK(cudaMemcpy(a.data(), ya.p, a.size() * 2, cudaMemcpyDeviceToHost));
      bool ok = rc == 0 && took == want &&
                std::memcmp(a.data() + size_t(m) * cols, init.data() + size_t(m) * cols, (kMaxM - m) * cols * 2) == 0;
      if (ok && took) {  // the routed result is the CUDA-core kernel's, byte for byte
        CK(cudaMemcpy(yb.p, init.data(), init.size() * 2, cudaMemcpyHostToDevice));
        const cudaError_t ce = kt2n_tern_cc::launch(w.k / 256, wk, acc, epi, dx.p, w.k, m, w.wc.p, w.ws.p, w.n, yb.p,
                                                    cols, res ? yb.p : nullptr, res ? w.n : 0, nullptr);
        CK(cudaDeviceSynchronize());
        std::vector<uint16_t> b(init.size());
        CK(cudaMemcpy(b.data(), yb.p, b.size() * 2, cudaMemcpyDeviceToHost));
        ok = ce == cudaSuccess && std::memcmp(a.data(), b.data(), a.size() * 2) == 0;
      }
      ++cases;
      if (!ok) {
        ++bad;
        std::printf("DISPATCH family %d M %d: rc %d, CUDA-core launches %ld (expected %ld)\n", fam, m, rc, took, want);
      }
    }
  }
  std::printf("tern_dec_cc --dispatch: SM %d.%d, KT2N_TERN_DECODE=%s, kernel image %s: %ld cases, %ld on the CUDA-core "
              "kernel, %s\n",
              caps.major, caps.minor, v ? v : "(unset)", image ? "yes" : "no", cases, routed,
              bad ? "ROUTING WRONG" : "routing as expected");
  return bad || cases == 0 ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
    (void)cudaGetLastError();
    const char* req = std::getenv("KT2N_REQUIRE_GPU");
    if (req && std::strcmp(req, "1") == 0) {
      std::printf("tern_dec_cc: no GPU visible and KT2N_REQUIRE_GPU=1: failed\n");
      return 1;
    }
    std::printf("tern_dec_cc: no GPU visible: skipped\n");
    return 0;
  }
  const kt2n::cuda::DeviceCaps caps = kt2n::cuda::current_caps();
  if (argc > 1 && std::strcmp(argv[1], "--dispatch") == 0) return dispatch(caps);
  if (!caps.legacy()) {
    std::printf("tern_dec_cc: SM %d.%d has bf16 tensor-core instructions; the CUDA-core kernel is not used there: skipped\n",
                caps.major, caps.minor);
    return 0;
  }
  if (!kt2n_tern_cc::available()) {
    std::printf("tern_dec_cc: this build has no CUDA-core kernel for SM %d.%d (the decode kernel runs there): skipped\n",
                caps.major, caps.minor);
    return 0;
  }
#ifdef _WIN32
  _putenv_s("KT2N_TERN_DECODE", "mma");  // kt2n_tern_proj = the decode kernel (software MMA)
#else
  setenv("KT2N_TERN_DECODE", "mma", 1);  // kt2n_tern_proj = the decode kernel (software MMA)
#endif
  long cases = 0, bad = 0, declined = 0;
  for (int fam = 0; fam < 4; ++fam) {
    const int k = fam == KT2N_TERN_FAM_DOWN ? 6144 : 2048;
    const int n = fam == KT2N_TERN_FAM_QKV ? 4096 : fam == KT2N_TERN_FAM_GATE_UP ? 12288 : 2048;
    int wk = 0, cpw = 0;
    kt2n_tern_family_split(fam, &wk, &cpw);
    const int acc = kt2n_tern_family_acc(fam);
    for (int spat = 0; spat < 2; ++spat) {
      std::mt19937 rng(77u + 10u * unsigned(fam) + unsigned(spat));
      std::vector<int32_t> codes(size_t(n) * k / 16);
      std::vector<uint16_t> scales(size_t(n) * k / 128);
      for (auto& v : codes) {
        uint32_t b = 0;
        for (int j = 0; j < 16; ++j) b |= (rng() % 3u) << (2 * j);
        std::memcpy(&v, &b, 4);
      }
      std::uniform_real_distribution<float> u01(0.f, 1.f);
      for (auto& v : scales) {
        float s = 0.002f + 0.03f * u01(rng);
        if (spat == 1) s = (rng() % 8 == 0) ? 0.f : std::ldexp(1.f + u01(rng), int(rng() % 60) - 40);
        v = f2bf(s);
      }
      Buf dc(codes.size() * 4), ds(scales.size() * 2), wc(kt2n_tern_codes_bytes(n, k)), ws(kt2n_tern_scales_bytes(n, k));
      CK(cudaMemcpy(dc.p, codes.data(), codes.size() * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(ds.p, scales.data(), scales.size() * 2, cudaMemcpyHostToDevice));
      if (kt2n_tern_repack(static_cast<const int32_t*>(dc.p), ds.p, n, k, wc.p, ws.p, n, 0, 1, nullptr)) return 2;
      constexpr int kMaxM = 16;
      Buf dx(size_t(kMaxM) * k * 2), ya(size_t(kMaxM) * n * 2), yb(size_t(kMaxM) * n * 2);
      for (int pat = 0; pat < 5; ++pat) {
        if (spat == 1 && pat != 0 && pat != 4) continue;
        const std::vector<uint16_t> x = activations(pat, size_t(kMaxM) * k, rng);
        const std::vector<uint16_t> r = activations(0, size_t(kMaxM) * n, rng);
        CK(cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
        for (int epi = 0; epi < 3; ++epi) {
          if (epi == KT2N_TERN_EPI_SILU_MUL && fam != KT2N_TERN_FAM_GATE_UP) continue;
          const int cols = epi == KT2N_TERN_EPI_SILU_MUL ? n / 2 : n;
          for (int m = 1; m <= kMaxM; ++m) {
            // y starts as the residual (RESIDUAL, in place) or a canary; rows >= m must keep it
            std::vector<uint16_t> init(size_t(kMaxM) * cols, 0x7fc1);
            if (epi == KT2N_TERN_EPI_RESIDUAL) init.assign(r.begin(), r.begin() + init.size());
            CK(cudaMemcpy(ya.p, init.data(), init.size() * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(yb.p, init.data(), init.size() * 2, cudaMemcpyHostToDevice));
            const bool res = epi == KT2N_TERN_EPI_RESIDUAL;
            const int rc = kt2n_tern_proj(fam, dx.p, k, m, wc.p, ws.p, n, k, ya.p, cols, epi, res ? ya.p : nullptr,
                                          res ? n : 0, 0u, nullptr);
            const cudaError_t ce = kt2n_tern_cc::launch(k / 256, wk, acc, epi, dx.p, k, m, wc.p, ws.p, n, yb.p, cols,
                                                        res ? yb.p : nullptr, res ? n : 0, nullptr);
            CK(cudaDeviceSynchronize());
            std::vector<uint16_t> a(init.size()), b(init.size());
            CK(cudaMemcpy(a.data(), ya.p, a.size() * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), yb.p, b.size() * 2, cudaMemcpyDeviceToHost));
            if (m > kt2n_tern_cc::kMaxRows) {  // above the row limit the launcher must decline (the caller keeps the old one)
              if (ce == cudaErrorInvalidConfiguration) {
                ++declined;
              } else {
                ++bad;
                std::printf("NOT DECLINED family %d epilogue %d M %d: %s\n", fam, epi, m, cudaGetErrorString(ce));
              }
              continue;
            }
            ++cases;  // M 1..kMaxRows: must run (a decline here is a failure, not a skip)
            const bool same = rc == 0 && ce == cudaSuccess && std::memcmp(a.data(), b.data(), a.size() * 2) == 0;
            if (!same) {
              ++bad;
              size_t d = 0;
              for (size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
              std::printf("DIFF family %d scales %d pattern %d epilogue %d M %d: rc %d / %s, %zu elements differ\n", fam,
                          spat, pat, epi, m, rc, cudaGetErrorString(ce), d);
            }
          }
        }
      }
    }
    std::printf("family %d done (%ld cases, %ld different)\n", fam, cases, bad);
  }
  std::printf("tern_dec_cc: SM %d.%d, %ld cases (%ld row counts left to the old kernel), %s\n", caps.major, caps.minor,
              cases, declined, bad ? "DIFFERENCES FOUND" : "all bitwise identical to the software-MMA decode kernel");
  return bad || cases == 0 ? 1 : 0;
}
