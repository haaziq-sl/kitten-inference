// The vocoder's noise generator (curand_init + curand_normal4 on Philox4x32-10, src/render/hift/hift.cu) against a
// CPU reference of cuRAND (tests/gpu/philox_ref.h).
//
//   --host    HIP builds only, no GPU needed: rocRAND's generator run on the host (its device functions are also
//             host functions) must give cuRAND's raw uint4 stream bit for bit.
//   (default) on the GPU: the raw stream must match bit for bit; the normal draws must match the reference
//             Box-Muller to within the fast sin/cos approximation (reported: max abs error).
#include <curand_kernel.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "philox_ref.h"

#if defined(KT2N_HIP)
#include <rocrand/rocrand_kernel.h>
#endif

namespace {

struct Case {
  unsigned long long seed, subsequence, offset;
};
const Case kCases[] = {
    {0ull, 0ull, 0ull},
    {42ull, 0ull, 0ull},
    {42ull, 1ull, 0ull},
    {0x123456789abcdefull, 977ull, 0ull},
    {0xffffffffffffffffull, 0xfffffffffull, 0ull},
    {7ull, 3ull, 1ull},
    {7ull, 3ull, 2ull},
    {7ull, 3ull, 3ull},
    {7ull, 3ull, 4ull},
    {7ull, 3ull, 0x1234567ull},
    {99ull, 0xffffffffull, 0xfffffffffffull},
    {1234ull, 255ull, 8ull},
};
constexpr int kCases_n = int(sizeof(kCases) / sizeof(kCases[0]));
constexpr int kDraws = 64;  // curand4 calls per case

__global__ void philox_k(const Case* cases, int n, uint4* raw, float4* normal) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  curandStatePhilox4_32_10_t s;
  curand_init(cases[c].seed, cases[c].subsequence, cases[c].offset, &s);
  for (int i = 0; i < kDraws; ++i) raw[c * kDraws + i] = curand4(&s);
  curand_init(cases[c].seed, cases[c].subsequence, cases[c].offset, &s);
  for (int i = 0; i < kDraws; ++i) normal[c * kDraws + i] = curand_normal4(&s);
}

#define CHECK(x)                                                                        \
  do {                                                                                  \
    const cudaError_t e_ = (x);                                                         \
    if (e_ != cudaSuccess) {                                                            \
      std::printf("%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));    \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

int raw_mismatches(int c, const uint4* got) {
  kt2n_philox_ref::State ref;
  ref.init(kCases[c].seed, kCases[c].subsequence, kCases[c].offset);
  int bad = 0;
  for (int i = 0; i < kDraws; ++i) {
    const kt2n_philox_ref::U4 r = ref.next4();
    const uint4 g = got[i];
    if (g.x != r.x || g.y != r.y || g.z != r.z || g.w != r.w) {
      if (bad < 3)
        std::printf("  case %d draw %d: got %08x %08x %08x %08x, cuRAND %08x %08x %08x %08x\n", c, i, g.x, g.y, g.z,
                    g.w, r.x, r.y, r.z, r.w);
      ++bad;
    }
  }
  return bad;
}

#if defined(KT2N_HIP)
int host_check() {
  int bad = 0;
  std::vector<uint4> got(kDraws);
  for (int c = 0; c < kCases_n; ++c) {
    rocrand_state_philox4x32_10 s;
    rocrand_init(kCases[c].seed, kCases[c].subsequence, kCases[c].offset, &s);
    for (int i = 0; i < kDraws; ++i) got[i] = rocrand4(&s);
    bad += raw_mismatches(c, got.data());
  }
  std::printf("rocRAND Philox4x32-10 (host) vs cuRAND reference: %d cases x %d draws, %d mismatching draws\n",
              kCases_n, kDraws, bad);
  return bad == 0 ? 0 : 1;
}
#endif

int device_check() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    (void)cudaGetLastError();
    if (std::getenv("KT2N_REQUIRE_GPU")) {
      std::printf("no GPU visible and KT2N_REQUIRE_GPU is set\n");
      return 1;
    }
    std::printf("no GPU visible: skipped\n");
    return 0;
  }
  Case* dc = nullptr;
  uint4* draw = nullptr;
  float4* dnorm = nullptr;
  CHECK(cudaMalloc(&dc, sizeof(kCases)));
  CHECK(cudaMalloc(&draw, sizeof(uint4) * kCases_n * kDraws));
  CHECK(cudaMalloc(&dnorm, sizeof(float4) * kCases_n * kDraws));
  CHECK(cudaMemcpy(dc, kCases, sizeof(kCases), cudaMemcpyHostToDevice));
  philox_k<<<1, 32>>>(dc, kCases_n, draw, dnorm);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  std::vector<uint4> raw(size_t(kCases_n) * kDraws);
  std::vector<float4> nrm(size_t(kCases_n) * kDraws);
  CHECK(cudaMemcpy(raw.data(), draw, sizeof(uint4) * raw.size(), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(nrm.data(), dnorm, sizeof(float4) * nrm.size(), cudaMemcpyDeviceToHost));
  int bad = 0;
  double max_err = 0.0;
  for (int c = 0; c < kCases_n; ++c) {
    bad += raw_mismatches(c, raw.data() + size_t(c) * kDraws);
    kt2n_philox_ref::State ref;
    ref.init(kCases[c].seed, kCases[c].subsequence, kCases[c].offset);
    for (int i = 0; i < kDraws; ++i) {
      const kt2n_philox_ref::U4 r = ref.next4();
      float e[4];
      kt2n_philox_ref::box_muller(r.x, r.y, &e[0], &e[1]);
      kt2n_philox_ref::box_muller(r.z, r.w, &e[2], &e[3]);
      const float4 g = nrm[size_t(c) * kDraws + i];
      const float gv[4] = {g.x, g.y, g.z, g.w};
      for (int q = 0; q < 4; ++q) max_err = std::fmax(max_err, std::fabs(double(gv[q]) - double(e[q])));
    }
  }
  cudaFree(dc);
  cudaFree(draw);
  cudaFree(dnorm);
  // The fast __sincosf has an absolute error of about 2^-21 on [-pi, pi] (larger on [pi, 2pi]); times |s| <= ~6.
  const bool normal_ok = max_err < 2e-4;
  std::printf("GPU curand4 vs cuRAND reference: %d mismatching draws; curand_normal4 max abs error %.3g (%s)\n", bad,
              max_err, normal_ok ? "ok" : "too large");
  return bad == 0 && normal_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "--host") == 0) {
#if defined(KT2N_HIP)
    return host_check();
#else
    std::printf("--host: HIP builds only\n");
    return 0;
#endif
  }
  return device_check();
}
