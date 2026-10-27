// AMD WMMA kernels (src/hip/, include/kt2n/hip/) on an RDNA3 / RDNA4 GPU. No model assets required.
//
//   layout   integer-valued A / B / C (every sum exact in fp32, so any summation order gives the same bits):
//            kt2n::wmma operand loaders + accumulator coordinates == a host reference, bit for bit (k-contiguous and
//            strided B, 1 and 8 chained k-blocks)
//   rounding probes of the hardware's internal accumulation (reported, not checked) and random bf16 inputs: WMMA chain
//            and the software path's sequential fmaf chain vs an fp64 reference
//   ternary  kt2n_wmma_tern_proj vs an fp64 reference of the decoded weights and vs kt2n_tern_proj (the software
//            path); batch invariance: every row bitwise equal for M = 1 .. 100, every launch shape and row position
//
// test_gpu_hip_wmma            the checks above; exit status 0 = all passed (77 = no WMMA GPU: skipped)
// test_gpu_hip_wmma --bench    timing of the four LM projection families, WMMA vs software path, M = 1 .. 520
#include "kt2n/cuda_caps.h"
#include "kt2n/hip/wmma.cuh"
#include "kt2n/hip/wmma_kernels.h"
#include "kt2n/kernels/tern_proj.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace wm = kt2n::wmma;

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
void report(const char* name, bool ok, const std::string& detail) {
  std::printf("%-48s %s  %s\n", name, ok ? "PASS" : "FAIL", detail.c_str());
  if (!ok) ++g_failures;
}
void info(const char* name, const std::string& detail) { std::printf("%-48s INFO  %s\n", name, detail.c_str()); }

float u2f(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint32_t f2u(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
uint16_t f2bf(float f) {  // round to nearest even (finite inputs)
  const uint32_t u = f2u(f);
  return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float bf2f(uint16_t b) { return u2f(static_cast<uint32_t>(b) << 16); }

std::string fmt(const char* f, double a, double b = 0, double c = 0, double d = 0) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), f, a, b, c, d);
  return buf;
}

// ------------------------------------------------------------------------------------------------------------------
// Raw WMMA chain on one wave: D = sum over kb of A_kb B_kb + C, 16 x 16 tiles.
//   A [kb][16 m][16 k] k-contiguous; B [kb][16 n][16 k] (k-contiguous) or, with strided_b, [kb][16 k][16 n];
//   C, D [16 m][16 n] fp32.
__global__ void raw_chain_kernel(const uint16_t* A, const uint16_t* B, const float* C, float* D, int kblocks,
                                 int strided_b) {
#if KT2N_WMMA_ARCH
  const int l = wm::lane();
  wm::acc_t acc;
#pragma unroll
  for (int v = 0; v < 8; ++v) acc[v] = C[wm::acc_row(v, l) * 16 + wm::acc_col(l)];
  for (int kb = 0; kb < kblocks; ++kb) {
    const uint16_t* a = A + kb * 256;
    const uint16_t* b = B + kb * 256;
    const wm::words_t af = wm::load_run(a + wm::op_index(l) * 16 + wm::op_k0(l));
    const wm::words_t bf = strided_b ? wm::load_strided(b, 16, l) : wm::load_run(b + wm::op_index(l) * 16 + wm::op_k0(l));
    acc = wm::mma(af, bf, acc);
  }
#pragma unroll
  for (int v = 0; v < 8; ++v) D[wm::acc_row(v, l) * 16 + wm::acc_col(l)] = acc[v];
#else
  (void)A, (void)B, (void)C, (void)D, (void)kblocks, (void)strided_b;
#if defined(__HIP_DEVICE_COMPILE__)
  __builtin_trap();
#endif
#endif
}

struct Chain {
  int kblocks;
  std::vector<uint16_t> A, B;  // B k-contiguous [kb][n][k]
  std::vector<float> C;
};

std::vector<float> run_chain(const Chain& ch, bool strided_b) {
  std::vector<uint16_t> b = ch.B;
  if (strided_b) {  // [kb][n][k] -> [kb][k][n]
    for (int kb = 0; kb < ch.kblocks; ++kb)
      for (int n = 0; n < 16; ++n)
        for (int k = 0; k < 16; ++k) b[kb * 256 + k * 16 + n] = ch.B[kb * 256 + n * 16 + k];
  }
  Dev<uint16_t> dA(ch.A), dB(b);
  Dev<float> dC(ch.C), dD(256);
  raw_chain_kernel<<<1, 32>>>(dA.p, dB.p, dC.p, dD.p, ch.kblocks, strided_b ? 1 : 0);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  return dD.get();
}

// fp64 reference and the software path's order (fmaf chain, k ascending, starting from C).
void host_chain(const Chain& ch, std::vector<double>& exact, std::vector<float>& seq) {
  exact.assign(256, 0.0);
  seq.assign(256, 0.f);
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 16; ++n) {
      double e = ch.C[m * 16 + n];
      float s = ch.C[m * 16 + n];
      for (int kb = 0; kb < ch.kblocks; ++kb)
        for (int k = 0; k < 16; ++k) {
          const float a = bf2f(ch.A[kb * 256 + m * 16 + k]), b = bf2f(ch.B[kb * 256 + n * 16 + k]);
          e += double(a) * double(b);
          s = std::fma(a, b, s);
        }
      exact[m * 16 + n] = e;
      seq[m * 16 + n] = s;
    }
}

// Layout. (a) One-hot: B = a permutation (B[n][k] = 1 when k = perm(n)), so every D element is a single positive
// product plus C = 0, which both generations return exactly: D[m][n] = A[m][perm(n)] checks A's row / k mapping and
// the accumulator mapping; the transposed case (A a permutation) checks B. (b) Random small integers with C: exact on
// gfx12; gfx11's accumulation is not exact under cancellation (see the probes), so there the check is that every
// element is within 2^-18 of max(|C|, sum |a b|), far below the >= 1 error any mapping mistake gives.
void test_layout(std::mt19937& rng) {
  const int arch = kt2n::cuda::device_caps(0).wmma;
  for (int strided : {0, 1}) {
    for (int which : {0, 1}) {  // 0: B is the permutation, 1: A is
      Chain ch{1, std::vector<uint16_t>(256, 0), std::vector<uint16_t>(256, 0), std::vector<float>(256, 0.f)};
      int perm[16];
      for (int i = 0; i < 16; ++i) perm[i] = i;
      std::shuffle(perm, perm + 16, rng);
      std::uniform_int_distribution<int> d(1, 200);
      std::vector<float> val(256);
      for (auto& v : val) v = float(d(rng));
      for (int i = 0; i < 16; ++i)
        for (int k = 0; k < 16; ++k) {
          if (which == 0) {
            ch.A[i * 16 + k] = f2bf(val[i * 16 + k]);
            ch.B[i * 16 + k] = f2bf(k == perm[i] ? 1.f : 0.f);
          } else {
            ch.A[i * 16 + k] = f2bf(k == perm[i] ? 1.f : 0.f);
            ch.B[i * 16 + k] = f2bf(val[i * 16 + k]);
          }
        }
      const std::vector<float> got = run_chain(ch, strided != 0);
      int bad = 0;
      for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n) {
          const float want = which == 0 ? val[m * 16 + perm[n]] : val[n * 16 + perm[m]];
          bad += f2u(got[m * 16 + n]) != f2u(want);
        }
      char name[96];
      std::snprintf(name, sizeof(name), "layout one-hot %s, %s B", which == 0 ? "B" : "A", strided ? "strided" : "k-contiguous");
      report(name, bad == 0, fmt("%.0f of 256 elements wrong", bad));
    }
  }
  for (int kblocks : {1, 8}) {
    for (int strided : {0, 1}) {
      Chain ch{kblocks, std::vector<uint16_t>(kblocks * 256), std::vector<uint16_t>(kblocks * 256),
               std::vector<float>(256)};
      std::uniform_int_distribution<int> d(-8, 8), dc(-1000, 1000);
      for (auto& v : ch.A) v = f2bf(float(d(rng)));
      for (auto& v : ch.B) v = f2bf(float(d(rng)));
      for (auto& v : ch.C) v = float(dc(rng));
      const std::vector<float> got = run_chain(ch, strided != 0);
      std::vector<double> exact;
      std::vector<float> seq;
      host_chain(ch, exact, seq);
      int inexact = 0, bad = 0;
      double worst = 0;
      for (int i = 0; i < 256; ++i) {
        if (f2u(got[i]) == f2u(float(exact[i]))) continue;
        ++inexact;
        double mag = std::fabs(ch.C[i]);
        double absum = 0;
        for (int kb = 0; kb < kblocks; ++kb)
          for (int k = 0; k < 16; ++k)
            absum += std::fabs(double(bf2f(ch.A[kb * 256 + (i / 16) * 16 + k])) * bf2f(ch.B[kb * 256 + (i % 16) * 16 + k]));
        mag = std::max(mag, absum);
        const double rel = std::fabs(got[i] - exact[i]) / mag;
        worst = std::max(worst, rel);
        if (rel > std::ldexp(1.0, -18)) ++bad;
        if (std::getenv("KT2N_WMMA_TEST_VERBOSE"))
          std::printf("    m %2d n %2d: got %.9g exact %.9g (C %.9g)\n", i / 16, i % 16, got[i], exact[i], ch.C[i]);
      }
      char name[96];
      std::snprintf(name, sizeof(name), "integers %s B, %d k-block%s", strided ? "strided" : "k-contiguous", kblocks,
                    kblocks > 1 ? "s" : "");
      const bool ok = arch == 12 ? inexact == 0 : bad == 0;
      report(name, ok, fmt("%.0f of 256 not exact (gfx%.0f), worst error / magnitude %.3g", inexact, arch, worst));
    }
  }
}

// One probe: A row 0 = a[16], B column 0 = b[16], C[0][0] = c; everything else zero. Returns D[0][0].
float probe(const float* a, const float* b, float c) {
  Chain ch{1, std::vector<uint16_t>(256, 0), std::vector<uint16_t>(256, 0), std::vector<float>(256, 0.f)};
  for (int k = 0; k < 16; ++k) {
    ch.A[k] = f2bf(a[k]);
    ch.B[k] = f2bf(b[k]);
  }
  ch.C[0] = c;
  return run_chain(ch, false)[0];
}

void test_rounding(std::mt19937& rng) {
  float a[16], b[16];
  // 1) C = 1 plus sixteen products of 2^-25: a sequential fp32 chain loses every one (1 + 2^-25 rounds to 1); an exact
  //    sum of the products before the add gives 1 + 2^-21.
  for (int k = 0; k < 16; ++k) a[k] = std::ldexp(1.f, -12), b[k] = std::ldexp(1.f, -13);
  float d = probe(a, b, 1.f);
  info("probe: 1 + 16 x 2^-25", fmt("D = 1 + %.3g x 2^-21  (sequential fp32: 0, exact: 1)", (double(d) - 1.0) / std::ldexp(1.0, -21)));
  // 2) C = 0, products 2^24, 1, -2^24, 0...: exact 1; left-to-right fp32 gives 0.
  for (int k = 0; k < 16; ++k) a[k] = 0.f, b[k] = 0.f;
  a[0] = 4096.f, b[0] = 4096.f;
  a[1] = 1.f, b[1] = 1.f;
  a[2] = -4096.f, b[2] = 4096.f;
  d = probe(a, b, 0.f);
  info("probe: 2^24 + 1 - 2^24 (k 0, 1, 2)", fmt("D = %.9g  (sequential fp32: 0, exact: 1)", d));
  // 3) same with the large pair at k 0 and 15
  for (int k = 0; k < 16; ++k) a[k] = 0.f, b[k] = 0.f;
  a[0] = 4096.f, b[0] = 4096.f;
  a[7] = 1.f, b[7] = 1.f;
  a[15] = -4096.f, b[15] = 4096.f;
  d = probe(a, b, 0.f);
  info("probe: 2^24 (k 0) + 1 (k 7) - 2^24 (k 15)", fmt("D = %.9g  (sequential fp32: 0, exact: 1)", d));
  // 4) C = 2^24, one product 1: does the add of C round like fp32?
  for (int k = 0; k < 16; ++k) a[k] = 0.f, b[k] = 0.f;
  a[0] = 1.f, b[0] = 1.f;
  d = probe(a, b, 16777216.f);
  info("probe: C = 2^24, + 1", fmt("D - 2^24 = %.9g  (fp32 round-to-nearest-even: 0)", double(d) - 16777216.0));
  a[1] = 1.f, b[1] = 1.f;
  d = probe(a, b, 16777216.f);
  info("probe: C = 2^24, + 1 + 1", fmt("D - 2^24 = %.9g  (sequential fp32: 0, exact: 2)", double(d) - 16777216.0));
  // 5) denormal products: 2^-70 x 2^-70 = 2^-140 (an fp32 denormal)
  for (int k = 0; k < 16; ++k) a[k] = 0.f, b[k] = 0.f;
  a[0] = std::ldexp(1.f, -70), b[0] = std::ldexp(1.f, -70);
  d = probe(a, b, 0.f);
  info("probe: 2^-70 x 2^-70 (fp32 denormal)", fmt("D = %.6g x 2^-140  (exact: 1, flushed: 0)", double(d) / std::ldexp(1.0, -140)));

  // 5b) cancellation without C, other orders and signs
  {
    const float va[4][16] = {{1.f, 4096.f, -4096.f}, {4096.f, -4096.f, 1.f}, {-4096.f, 4096.f, 1.f},
                             {4096.f, 1.f, -4096.f}};
    for (int t = 0; t < 4; ++t) {
      for (int k = 0; k < 16; ++k) a[k] = va[t][k], b[k] = (va[t][k] == 1.f || va[t][k] == 0.f) ? 1.f : 4096.f;
      d = probe(a, b, 0.f);
      info("probe: three products (order variant)", fmt("variant %.0f: D = %.9g  (exact: 1)", t, d));
    }
    for (int k = 0; k < 16; ++k) a[k] = 0.f, b[k] = 0.f;
    a[0] = 3.f, b[0] = 5.f;
    d = probe(a, b, 0.f);
    info("probe: 3 x 5", fmt("D = %.9g", d));
    a[1] = -7.f, b[1] = 2.f;
    d = probe(a, b, 0.f);
    info("probe: 3 x 5 - 7 x 2", fmt("D = %.9g (exact 1)", d));
    d = probe(a, b, 100.f);
    info("probe: 100 + 3 x 5 - 7 x 2", fmt("D = %.9g (exact 101)", d));
    d = probe(a, b, -100.f);
    info("probe: -100 + 3 x 5 - 7 x 2", fmt("D = %.9g (exact -99)", d));
  }
  // 6) random bf16 N(0, 1) inputs, 128 k-blocks (K = 2048, the LM hidden size): error of each order vs fp64.
  Chain ch{128, std::vector<uint16_t>(128 * 256), std::vector<uint16_t>(128 * 256), std::vector<float>(256, 0.f)};
  std::normal_distribution<float> nd(0.f, 1.f);
  for (auto& v : ch.A) v = f2bf(nd(rng));
  for (auto& v : ch.B) v = f2bf(nd(rng));
  const std::vector<float> got = run_chain(ch, false);
  std::vector<double> exact;
  std::vector<float> seq;
  host_chain(ch, exact, seq);
  double ew = 0, es = 0, ref = 0, mw = 0, ms = 0;
  int same = 0;
  for (int i = 0; i < 256; ++i) {
    ew += (got[i] - exact[i]) * (got[i] - exact[i]);
    es += (seq[i] - exact[i]) * (seq[i] - exact[i]);
    ref += exact[i] * exact[i];
    mw = std::max(mw, std::fabs(got[i] - exact[i]));
    ms = std::max(ms, std::fabs(seq[i] - exact[i]));
    same += f2u(got[i]) == f2u(seq[i]);
  }
  info("random K = 2048: rms rel. error vs fp64", fmt("WMMA %.3g, sequential fmaf %.3g; max abs %.3g / %.3g",
                                                       std::sqrt(ew / ref), std::sqrt(es / ref), mw, ms));
  info("random K = 2048: WMMA == sequential fmaf", fmt("%.0f of 256 elements bitwise", same));
  // Bound check: both within the fp32 accumulation error bound (K * eps * sum |a b|), a loose sanity check.
  double worst = 0;
  for (int i = 0; i < 256; ++i) {
    double absum = 0;
    const int m = i / 16, n = i % 16;
    for (int kb = 0; kb < 128; ++kb)
      for (int k = 0; k < 16; ++k)
        absum += std::fabs(double(bf2f(ch.A[kb * 256 + m * 16 + k])) * bf2f(ch.B[kb * 256 + n * 16 + k]));
    worst = std::max(worst, std::fabs(got[i] - exact[i]) / (2048.0 * std::ldexp(1.0, -24) * absum));
  }
  report("random K = 2048: WMMA within fp32 bound", worst <= 1.0, fmt("max error / (K eps sum|ab|) = %.3g", worst));
}

// ------------------------------------------------------------------------------------------------------------------
// Ternary projections.
struct Tern {
  int n, k;
  Dev<uint8_t> codes, scales;
  std::vector<uint16_t> dense;  // decoded bf16 weights [n][k]
  Tern(int n_, int k_, std::mt19937& rng)
      : n(n_), k(k_), codes(kt2n_tern_codes_bytes(n_, k_)), scales(kt2n_tern_scales_bytes(n_, k_)) {
    std::vector<int32_t> c(size_t(n) * (k / 16));
    std::vector<uint16_t> s(size_t(n) * (k / 128));
    std::uniform_int_distribution<int> code(0, 2);
    std::uniform_real_distribution<float> sc(0.01f, 0.05f);
    for (auto& w : c) {
      uint32_t u = 0;
      for (int o = 0; o < 16; ++o) u |= uint32_t(code(rng)) << (2 * o);
      w = int32_t(u);
    }
    for (auto& v : s) v = f2bf(sc(rng));
    Dev<int32_t> dc(c);
    Dev<uint16_t> ds(s);
    if (kt2n_tern_repack(dc.p, ds.p, n, k, codes.p, scales.p, n, 0, 1, nullptr) != KT2N_TERN_OK)
      throw std::runtime_error("kt2n_tern_repack failed");
    Dev<uint16_t> dd(size_t(n) * k);
    if (kt2n_tern_decode_dense(codes.p, scales.p, n, k, dd.p, nullptr) != KT2N_TERN_OK)
      throw std::runtime_error("kt2n_tern_decode_dense failed");
    CHECK(cudaDeviceSynchronize());
    dense = dd.get();
  }
};

float silu_mul_ref(float gate, float up) {  // the SILU_MUL epilogue on exact accumulators (host, fp32 like the GPU)
  const float gb = bf2f(f2bf(gate)), ub = bf2f(f2bf(up));
  const float act = bf2f(f2bf(gb / (1.0f + std::exp(-gb))));
  return bf2f(f2bf(act * ub));
}

// Output bf16 bits of an (m x n_out) projection; ref = fp64 sums rounded through the same epilogue.
std::vector<uint16_t> ref_proj(const Tern& w, const std::vector<uint16_t>& x, int m, int epi,
                               const std::vector<uint16_t>& res) {
  const int n_out = epi == KT2N_TERN_EPI_SILU_MUL ? w.n / 2 : w.n;
  std::vector<uint16_t> y(size_t(m) * n_out);
  std::vector<double> acc(w.n);
  for (int r = 0; r < m; ++r) {
    for (int n = 0; n < w.n; ++n) {
      double s = 0;
      for (int kk = 0; kk < w.k; ++kk) s += double(bf2f(x[size_t(r) * w.k + kk])) * bf2f(w.dense[size_t(n) * w.k + kk]);
      acc[n] = s;
    }
    for (int o = 0; o < n_out; ++o) {
      float v;
      if (epi == KT2N_TERN_EPI_SILU_MUL) v = silu_mul_ref(float(acc[2 * o]), float(acc[2 * o + 1]));
      else if (epi == KT2N_TERN_EPI_RESIDUAL) v = bf2f(res[size_t(r) * w.n + o]) + bf2f(f2bf(float(acc[o])));
      else v = float(acc[o]);
      y[size_t(r) * n_out + o] = f2bf(v);
    }
  }
  return y;
}

int ulp_diff(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7fff) : int(v); };
  return std::abs(key(a) - key(b));
}

void test_tern(std::mt19937& rng) {
  const int n = 256, k = 2048;  // the software path checks K against the family (2048 for qkv / o / gate_up)
  Tern w(n, k, rng);
  const int mmax = 100;
  std::vector<uint16_t> x(size_t(mmax) * k), res(size_t(mmax) * n);
  std::normal_distribution<float> nd(0.f, 1.f);
  for (auto& v : x) v = f2bf(nd(rng));
  for (auto& v : res) v = f2bf(nd(rng));
  Dev<uint16_t> dx(x), dres(res);
  const char* epi_name[3] = {"STORE", "SILU_MUL", "RESIDUAL"};
  for (int epi : {KT2N_TERN_EPI_STORE, KT2N_TERN_EPI_SILU_MUL, KT2N_TERN_EPI_RESIDUAL}) {
    const int n_out = epi == KT2N_TERN_EPI_SILU_MUL ? n / 2 : n;
    const std::vector<uint16_t> ref = ref_proj(w, x, mmax, epi, res);
    // Full M = 100 call with the automatic shape: vs fp64 reference and vs the software path.
    Dev<uint16_t> dy(size_t(mmax) * n_out), dsw(size_t(mmax) * n_out);
    if (epi == KT2N_TERN_EPI_RESIDUAL) {
      CHECK(cudaMemcpy(dy.p, dres.p, res.size() * 2, cudaMemcpyDeviceToDevice));  // in place, as the LM does
      CHECK(cudaMemcpy(dsw.p, dres.p, res.size() * 2, cudaMemcpyDeviceToDevice));
    }
    const void* rp_w = epi == KT2N_TERN_EPI_RESIDUAL ? dy.p : nullptr;
    const void* rp_s = epi == KT2N_TERN_EPI_RESIDUAL ? dsw.p : nullptr;
    int rc = kt2n_wmma_tern_proj(dx.p, k, mmax, w.codes.p, w.scales.p, n, k, dy.p, n_out, epi, rp_w, n, 0, nullptr);
    const int fam = epi == KT2N_TERN_EPI_SILU_MUL ? KT2N_TERN_FAM_GATE_UP
                    : epi == KT2N_TERN_EPI_RESIDUAL ? KT2N_TERN_FAM_O
                                                    : KT2N_TERN_FAM_QKV;
    int rc2 = kt2n_tern_proj(fam, dx.p, k, mmax, w.codes.p, w.scales.p, n, k, dsw.p, n_out, epi, rp_s, n, 0u, nullptr);
    CHECK(cudaDeviceSynchronize());
    char name[96];
    std::snprintf(name, sizeof(name), "tern %s M=100 launch", epi_name[epi]);
    report(name, rc == 0 && rc2 == 0, fmt("wmma rc %.0f, software rc %.0f", rc, rc2));
    const std::vector<uint16_t> y = dy.get(), sw = dsw.get();
    int same_ref = 0, same_sw = 0, max_ref = 0, max_sw = 0, swref = 0;
    for (size_t i = 0; i < y.size(); ++i) {
      same_ref += y[i] == ref[i];
      same_sw += y[i] == sw[i];
      swref += sw[i] == ref[i];
      max_ref = std::max(max_ref, ulp_diff(y[i], ref[i]));
      max_sw = std::max(max_sw, ulp_diff(y[i], sw[i]));
    }
    std::snprintf(name, sizeof(name), "tern %s vs fp64 reference", epi_name[epi]);
    // gfx11's accumulation is less exact than gfx12's (probes above); SiLU / double rounding can turn a 1-ulp
    // accumulator change into a few output ulps, so: >= 99.9% bitwise and <= 4 ulp anywhere.
    report(name, max_ref <= 4 && same_ref >= 0.999 * y.size(), fmt("%.4f%% bitwise, max %.0f bf16 ulp (software path: %.4f%% bitwise)",
                                   100.0 * same_ref / y.size(), max_ref, 100.0 * swref / y.size()));
    std::snprintf(name, sizeof(name), "tern %s vs software path", epi_name[epi]);
    report(name, max_sw <= 4 && same_sw >= 0.999 * y.size(), fmt("%.4f%% bitwise, max %.0f bf16 ulp", 100.0 * same_sw / y.size(), max_sw));
    // Batch invariance: every M (1 .. 100) and every launch shape gives the M = 100 bits for its rows; a row also
    // gives the same bits at another position (rows shifted by 7 through the x pointer).
    int variant_bad = 0, variants = 0;
    for (int shape : {11, 12, 14, 22, 24, 42, 111, 112, 121}) {
      for (int m : {1, 3, 16, 17, 33, 64, 100}) {
        for (int shift : {0, 7}) {
          if (shift + m > mmax) continue;
          Dev<uint16_t> dz(size_t(m) * n_out);
          if (epi == KT2N_TERN_EPI_RESIDUAL)
            CHECK(cudaMemcpy(dz.p, dres.p + size_t(shift) * n, size_t(m) * n * 2, cudaMemcpyDeviceToDevice));
          rc = kt2n_wmma_tern_proj(dx.p + size_t(shift) * k, k, m, w.codes.p, w.scales.p, n, k, dz.p, n_out, epi,
                                   epi == KT2N_TERN_EPI_RESIDUAL ? dz.p : nullptr, n, shape, nullptr);
          CHECK(cudaDeviceSynchronize());
          ++variants;
          if (rc != 0) {
            ++variant_bad;
            continue;
          }
          const std::vector<uint16_t> z = dz.get();
          if (std::memcmp(z.data(), y.data() + size_t(shift) * n_out, z.size() * 2) != 0) ++variant_bad;
        }
      }
    }
    std::snprintf(name, sizeof(name), "tern %s batch invariance", epi_name[epi]);
    report(name, variant_bad == 0, fmt("%.0f of %.0f (shape, M, row offset) calls differ from M = 100", variant_bad, variants));
  }
}

// ------------------------------------------------------------------------------------------------------------------
float time_ms(const std::function<int()>& f, int iters) {
  cudaEvent_t a, b;
  CHECK(cudaEventCreate(&a));
  CHECK(cudaEventCreate(&b));
  for (int i = 0; i < 3; ++i) f();
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaEventRecord(a));
  for (int i = 0; i < iters; ++i) f();
  CHECK(cudaEventRecord(b));
  CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CHECK(cudaEventElapsedTime(&ms, a, b));
  (void)cudaEventDestroy(a);
  (void)cudaEventDestroy(b);
  return ms / iters;
}

int bench(int argc, char** argv) {
  std::mt19937 rng(7);
  struct Fam {
    const char* name;
    int fam, n, k, epi;
  } fams[4] = {{"qkv", KT2N_TERN_FAM_QKV, 4096, 2048, KT2N_TERN_EPI_STORE},
               {"o", KT2N_TERN_FAM_O, 2048, 2048, KT2N_TERN_EPI_RESIDUAL},
               {"gate_up", KT2N_TERN_FAM_GATE_UP, 12288, 2048, KT2N_TERN_EPI_SILU_MUL},
               {"down", KT2N_TERN_FAM_DOWN, 2048, 6144, KT2N_TERN_EPI_RESIDUAL}};
  std::vector<int> ms = {1, 4, 16, 64, 128, 360, 520};
  int only_shape = 0;
  for (int i = 2; i < argc; ++i)
    if (std::strncmp(argv[i], "--shape=", 8) == 0) only_shape = std::atoi(argv[i] + 8);
  const int mmax = 520;
  Dev<uint16_t> dx(size_t(mmax) * 6144), dy(size_t(mmax) * 12288), dr(size_t(mmax) * 2048);
  {
    std::vector<uint16_t> x(size_t(mmax) * 6144);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = f2bf(nd(rng));
    dx.put(x);
  }
  std::printf("family   M     software_us   wmma_us  speedup  wmma_TFLOPs  (28 layers: software_ms wmma_ms)\n");
  double tot_sw[16] = {0}, tot_wm[16] = {0};
  for (const Fam& f : fams) {
    Tern w(f.n, f.k, rng);
    for (size_t mi = 0; mi < ms.size(); ++mi) {
      const int m = ms[mi];
      const int n_out = f.epi == KT2N_TERN_EPI_SILU_MUL ? f.n / 2 : f.n;
      const void* r = f.epi == KT2N_TERN_EPI_RESIDUAL ? dr.p : nullptr;
      const int iters = m <= 16 ? 50 : 10;
      const float sw = time_ms([&] {
        return kt2n_tern_proj(f.fam, dx.p, f.k, m, w.codes.p, w.scales.p, f.n, f.k, f.epi == KT2N_TERN_EPI_RESIDUAL ? dr.p : dy.p,
                              n_out, f.epi, r, f.n, 0u, nullptr);
      }, iters);
      const float wq = time_ms([&] {
        return kt2n_wmma_tern_proj(dx.p, f.k, m, w.codes.p, w.scales.p, f.n, f.k,
                                   f.epi == KT2N_TERN_EPI_RESIDUAL ? dr.p : dy.p, n_out, f.epi, r, f.n, only_shape,
                                   nullptr);
      }, iters);
      tot_sw[mi] += sw;
      tot_wm[mi] += wq;
      const double tflops = 2.0 * m * f.n * f.k / (wq * 1e-3) / 1e12;
      std::printf("%-8s %-5d %11.1f %9.1f %7.2fx %11.2f\n", f.name, m, sw * 1e3, wq * 1e3, sw / wq, tflops);
    }
  }
  for (size_t mi = 0; mi < ms.size(); ++mi)
    std::printf("all-4 x 28 layers  M %-5d software %8.2f ms  wmma %8.2f ms  speedup %.2fx\n", ms[mi], 28 * tot_sw[mi],
                28 * tot_wm[mi], tot_sw[mi] / tot_wm[mi]);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
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
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) return bench(argc, argv);
    std::mt19937 rng(12345);
    test_layout(rng);
    test_rounding(rng);
    test_tern(rng);
  } catch (const std::exception& e) {
    std::printf("ERROR %s\n", e.what());
    return 2;
  }
  std::printf("%s (%d failure%s)\n", g_failures ? "FAILED" : "all WMMA checks passed", g_failures,
              g_failures == 1 ? "" : "s");
  return g_failures ? 1 : 0;
}
