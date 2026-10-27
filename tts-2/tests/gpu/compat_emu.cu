// kt2n::compat (include/kt2n/cuda_compat.cuh) against the hardware instructions it stands in for, and against
// references built on the host or in fp64 from the PTX definitions. No model assets required.
//
//   every GPU   cvt_rna_tf32 over all 2^32 inputs == an fp64 round-half-away reference
//               ldmatrix x1/x2/x4/x4.trans, cp.async cg/ca (src_bytes 0..16), ld_nc_v4 == host expectation
//               mma bf16 m16n8k16 / tf32 m16n8k8: integer-valued inputs exact, random inputs within the fp32
//               accumulation bound of an fp64 reference
//   SM 7.5+     ldmatrix emulation bit-identical to the hardware instruction
//   SM 8.0+     cvt.rna.tf32 (all 2^32), cp.async and both mma shapes on integer-valued inputs (also tf32 with
//               nonzero low bits) bit-identical to the hardware; random-input differences reported
//
// Exit status 0 = every check passed.
#include "kt2n/cuda_caps.h"
#include "kt2n/cuda_compat.cuh"

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

namespace compat = kt2n::compat;

#define CHECK(call)                                                                       \
  do {                                                                                    \
    const cudaError_t err_ = (call);                                                      \
    if (err_ != cudaSuccess)                                                              \
      throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(err_));     \
  } while (0)

template <typename T>
struct Buffer {
  T* p = nullptr;
  size_t n;
  explicit Buffer(size_t count) : n(count) {
    CHECK(cudaMallocManaged(&p, (n ? n : 1) * sizeof(T)));
    std::memset(p, 0, (n ? n : 1) * sizeof(T));
  }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  T& operator[](size_t i) { return p[i]; }
};

int g_failures = 0;
void report(const char* name, bool ok, const std::string& detail) {
  std::printf("%-44s %s  %s\n", name, ok ? "PASS" : "FAIL", detail.c_str());
  if (!ok) ++g_failures;
}

uint32_t f2u(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
float u2f(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
// Values equal, with +0 == -0 (the sign of an exact zero sum depends on the addition order).
bool same_value(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

// ---------------------------------------------------------------------------------------------- cvt.rna.tf32
// Reference: |x| / ulp rounded half away from zero (std::round), times ulp, in fp64. ulp = 2^(exponent - 10),
// at least 2^-136 (fp32 subnormals keep 10 fraction bits too); 2^128 and above is inf; inf / NaN truncated.
__device__ inline uint32_t ref_tf32_bits(uint32_t u) {
  const uint32_t sign = u & 0x80000000u;
  if ((u & 0x7f800000u) == 0x7f800000u) return u & 0xffffe000u;
  if ((u & 0x7fffffffu) == 0u) return u;
  const float x = __uint_as_float(u & 0x7fffffffu);
  int e = 0;
  (void)frexp(double(x), &e);  // x = m * 2^e, m in [0.5, 1): leading bit 2^(e-1)
  const int ue = e - 11 > -136 ? e - 11 : -136;
  const double q = round(double(x) / ldexp(1.0, ue));
  const double r = q * ldexp(1.0, ue);
  const uint32_t mag = r >= 0x1p128 ? 0x7f800000u : __float_as_uint(float(r));
  return sign | mag;
}

__global__ void cvt_kernel(unsigned long long* bad_ref, unsigned long long* bad_hw, uint32_t* first, int* hw_ran) {
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < (1ull << 32); i += stride) {
    const uint32_t u = uint32_t(i);
    const uint32_t em = compat::cvt_rna_tf32(__uint_as_float(u));
    if (em != ref_tf32_bits(u)) {
      if (atomicAdd(bad_ref, 1ull) == 0) first[0] = u;
    }
#if __CUDA_ARCH__ >= 800
    uint32_t hw;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(hw) : "f"(__uint_as_float(u)));
    if (hw != em) {
      if (atomicAdd(bad_hw, 1ull) == 0) first[1] = u;
    }
    if (i == 0) *hw_ran = 1;
#endif
  }
}

void test_cvt(bool sm80) {
  Buffer<unsigned long long> bad_ref(1), bad_hw(1);
  Buffer<uint32_t> first(2);
  Buffer<int> hw_ran(1);
  cvt_kernel<<<2048, 256>>>(bad_ref.p, bad_hw.p, first.p, hw_ran.p);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  char d[128];
  std::snprintf(d, sizeof(d), "mismatches %llu (first input 0x%08x), all 2^32 inputs", bad_ref[0], first[0]);
  report("cvt_rna_tf32 == fp64 reference", bad_ref[0] == 0, d);
  if (sm80) {
    std::snprintf(d, sizeof(d), "mismatches %llu (first input 0x%08x), all 2^32 inputs", bad_hw[0], first[1]);
    report("cvt_rna_tf32 == cvt.rna.tf32.f32", hw_ran[0] == 1 && bad_hw[0] == 0, d);
  }
}

// ---------------------------------------------------------------------------------------------- ldmatrix
// Shared memory: 64 rows of 8 x b16 (16 B); element (row, col) = row * 8 + col + 0x1000 * trial.
// Lane l points at row rows[trial][l]: a random permutation, so the emulation must take each row address
// from the right lane.
constexpr int kLdRows = 64;
constexpr int kLdTrials = 64;
constexpr int kLdOut = 1 + 2 + 4 + 4;  // x1, x2, x4, x4.trans registers per lane

__global__ void ldmatrix_kernel(const int* rows, uint32_t* emu, uint32_t* hw, int* hw_ran) {
  __shared__ __align__(16) uint16_t tile[kLdRows * 8];
  const int trial = blockIdx.x;
  const int lane = threadIdx.x;
  for (int i = lane; i < kLdRows * 8; i += 32) tile[i] = uint16_t(i + 0x1000 * trial);
  __syncthreads();
  const void* p = &tile[rows[trial * 32 + lane] * 8];
  uint32_t* e = emu + (trial * 32 + lane) * kLdOut;
  uint32_t r1[1], r2[2], r4[4], t4[4];
  compat::ldmatrix_x1(r1, p);
  compat::ldmatrix_x2(r2, p);
  compat::ldmatrix_x4(r4, p);
  compat::ldmatrix_x4_trans(t4, p);
  e[0] = r1[0];
  e[1] = r2[0];
  e[2] = r2[1];
  for (int i = 0; i < 4; ++i) e[3 + i] = r4[i];
  for (int i = 0; i < 4; ++i) e[7 + i] = t4[i];
  // The scalar-reference forms must give the same registers.
  uint32_t s0, s1, s2, s3;
  compat::ldmatrix_x4(s0, s1, s2, s3, p);
  if (s0 != r4[0] || s1 != r4[1] || s2 != r4[2] || s3 != r4[3]) e[3] = ~e[3];
  compat::ldmatrix_x4_trans(s0, s1, s2, s3, p);
  if (s0 != t4[0] || s1 != t4[1] || s2 != t4[2] || s3 != t4[3]) e[7] = ~e[7];
  compat::ldmatrix_x2(s0, s1, p);
  if (s0 != r2[0] || s1 != r2[1]) e[1] = ~e[1];
  compat::ldmatrix_x1(s0, p);
  if (s0 != r1[0]) e[0] = ~e[0];
#if __CUDA_ARCH__ >= 750
  uint32_t* h = hw + (trial * 32 + lane) * kLdOut;
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n" : "=r"(h[0]) : "r"(a));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(h[1]), "=r"(h[2]) : "r"(a));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(h[3]), "=r"(h[4]), "=r"(h[5]), "=r"(h[6])
               : "r"(a));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(h[7]), "=r"(h[8]), "=r"(h[9]), "=r"(h[10])
               : "r"(a));
  if (trial == 0 && lane == 0) *hw_ran = 1;
#endif
}

void test_ldmatrix(bool sm75) {
  Buffer<int> rows(kLdTrials * 32);
  Buffer<uint32_t> emu(kLdTrials * 32 * kLdOut), hw(kLdTrials * 32 * kLdOut);
  Buffer<int> hw_ran(1);
  std::mt19937 rng(75);
  for (int t = 0; t < kLdTrials; ++t) {
    std::vector<int> perm(kLdRows);
    for (int i = 0; i < kLdRows; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int l = 0; l < 32; ++l) rows[t * 32 + l] = perm[l];
  }
  ldmatrix_kernel<<<kLdTrials, 32>>>(rows.p, emu.p, hw.p, hw_ran.p);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  auto elem = [](int t, int row, int col) { return uint32_t(uint16_t(row * 8 + col + 0x1000 * t)); };
  long bad = 0, bad_hw = 0;
  for (int t = 0; t < kLdTrials; ++t) {
    for (int l = 0; l < 32; ++l) {
      uint32_t want[kLdOut];
      const int r = l >> 2, c = 2 * (l & 3);
      auto plain = [&](int m) {
        const int row = rows[t * 32 + 8 * m + r];
        return elem(t, row, c) | (elem(t, row, c + 1) << 16);
      };
      auto trans = [&](int m) {
        const int row0 = rows[t * 32 + 8 * m + c], row1 = rows[t * 32 + 8 * m + c + 1];
        return elem(t, row0, r) | (elem(t, row1, r) << 16);
      };
      want[0] = plain(0);
      want[1] = plain(0);
      want[2] = plain(1);
      for (int m = 0; m < 4; ++m) want[3 + m] = plain(m);
      for (int m = 0; m < 4; ++m) want[7 + m] = trans(m);
      for (int k = 0; k < kLdOut; ++k) {
        bad += emu[(t * 32 + l) * kLdOut + k] != want[k];
        if (sm75) bad_hw += hw[(t * 32 + l) * kLdOut + k] != emu[(t * 32 + l) * kLdOut + k];
      }
    }
  }
  report("ldmatrix x1/x2/x4/x4.trans == definition", bad == 0,
         std::to_string(bad) + " of " + std::to_string(kLdTrials * 32 * kLdOut) + " registers differ");
  if (sm75)
    report("ldmatrix x1/x2/x4/x4.trans == hardware", hw_ran[0] == 1 && bad_hw == 0,
           std::to_string(bad_hw) + " registers differ");
}

// ---------------------------------------------------------------------------------------------- cp.async
// Slot i (0..33): src_bytes = i % 17, cg for i < 17 and ca after; destination prefilled with 0xAB.
constexpr int kCpSlots = 34;

__global__ void cp_async_kernel(const uint8_t* src, uint8_t* emu, uint8_t* hw, int* hw_ran) {
  __shared__ __align__(16) uint8_t s_emu[kCpSlots * 16];
  __shared__ __align__(16) uint8_t s_hw[kCpSlots * 16];
  const int i = threadIdx.x;
  for (int j = i; j < kCpSlots * 16; j += blockDim.x) s_emu[j] = s_hw[j] = 0xAB;
  __syncthreads();
  if (i < kCpSlots) {
    const int n = i % 17;
    const bool ca = i >= 17;
    if (ca) compat::cp_async_ca16(&s_emu[16 * i], src + 16 * i, n);
    else compat::cp_async_cg16(&s_emu[16 * i], src + 16 * i, n);
    compat::cp_async_commit();
    compat::cp_async_wait<0>();
    compat::cp_async_wait_all();
#if __CUDA_ARCH__ >= 800
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(&s_hw[16 * i]));
    if (ca)
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(src + 16 * i), "r"(n) : "memory");
    else
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(src + 16 * i), "r"(n) : "memory");
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    if (i == 0) *hw_ran = 1;
#endif
  }
  __syncthreads();
  for (int j = i; j < kCpSlots * 16; j += blockDim.x) {
    emu[j] = s_emu[j];
    hw[j] = s_hw[j];
  }
}

__global__ void ld_nc_kernel(const uint4* src, uint4* dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = compat::ld_nc_v4(src + i);
}

void test_cp_async(bool sm80) {
  Buffer<uint8_t> src(kCpSlots * 16), emu(kCpSlots * 16), hw(kCpSlots * 16);
  Buffer<int> hw_ran(1);
  std::mt19937 rng(80);
  for (int j = 0; j < kCpSlots * 16; ++j) src[j] = uint8_t(rng() | 1u);  // nonzero: a byte that was not copied shows
  cp_async_kernel<<<1, 64>>>(src.p, emu.p, hw.p, hw_ran.p);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  long bad = 0, bad_hw = 0;
  for (int i = 0; i < kCpSlots; ++i) {
    const int n = i % 17;
    for (int b = 0; b < 16; ++b) {
      const uint8_t want = b < n ? src[16 * i + b] : 0;
      bad += emu[16 * i + b] != want;
      if (sm80) bad_hw += hw[16 * i + b] != emu[16 * i + b];
    }
  }
  report("cp.async cg/ca 16, src_bytes 0..16 == def.", bad == 0, std::to_string(bad) + " bytes differ");
  if (sm80) report("cp.async cg/ca 16, src_bytes 0..16 == hw", hw_ran[0] == 1 && bad_hw == 0,
                   std::to_string(bad_hw) + " bytes differ");

  constexpr int n4 = 4096;
  Buffer<uint4> a(n4), b(n4);
  for (int i = 0; i < n4; ++i) a[i] = make_uint4(rng(), rng(), rng(), rng());
  ld_nc_kernel<<<n4 / 256, 256>>>(a.p, b.p, n4);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  report("ld_nc_v4 == plain load", std::memcmp(a.p, b.p, n4 * sizeof(uint4)) == 0, "4096 x 16 B");
}

// ---------------------------------------------------------------------------------------------- mma
// One warp per trial. bf16 m16n8k16: A[16][16], B[16][8] (B[k][n]), C/D[16][8]. tf32 m16n8k8: A[16][8],
// B[8][8]. Each lane loads its fragments by the PTX layout, runs the emulation (and the hardware
// instruction on SM 8.0+), and stores D by the same layout.
template <bool kTf32>
__global__ void mma_kernel(const uint32_t* A, const uint32_t* B, const float* C, float* D_emu, float* D_hw,
                           int trials, int* hw_ran) {
  constexpr int K = kTf32 ? 8 : 16;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= trials) return;  // whole warps only (trials is per warp)
  const int g = lane >> 2, t = lane & 3;
  const uint32_t* a = A + size_t(warp) * 16 * K;  // A[row][k] as raw element bits (bf16 in the low 16 bits)
  const uint32_t* b = B + size_t(warp) * K * 8;   // B[k][n]
  const float* c = C + size_t(warp) * 128;
  uint32_t a0, a1, a2, a3, b0, b1;
  if constexpr (kTf32) {
    a0 = a[g * K + t];
    a1 = a[(g + 8) * K + t];
    a2 = a[g * K + t + 4];
    a3 = a[(g + 8) * K + t + 4];
    b0 = b[t * 8 + g];
    b1 = b[(t + 4) * 8 + g];
  } else {
    auto pk = [](uint32_t lo, uint32_t hi) { return (lo & 0xffffu) | (hi << 16); };
    a0 = pk(a[g * K + 2 * t], a[g * K + 2 * t + 1]);
    a1 = pk(a[(g + 8) * K + 2 * t], a[(g + 8) * K + 2 * t + 1]);
    a2 = pk(a[g * K + 2 * t + 8], a[g * K + 2 * t + 9]);
    a3 = pk(a[(g + 8) * K + 2 * t + 8], a[(g + 8) * K + 2 * t + 9]);
    b0 = pk(b[(2 * t) * 8 + g], b[(2 * t + 1) * 8 + g]);
    b1 = pk(b[(2 * t + 8) * 8 + g], b[(2 * t + 9) * 8 + g]);
  }
  const int o0 = g * 8 + 2 * t, o1 = o0 + 1, o2 = (g + 8) * 8 + 2 * t, o3 = o2 + 1;
  float d[4] = {c[o0], c[o1], c[o2], c[o3]};
  if constexpr (kTf32) compat::mma_m16n8k8_tf32(d, a0, a1, a2, a3, b0, b1);
  else compat::mma_m16n8k16_bf16(d, a0, a1, a2, a3, b0, b1);
  float* de = D_emu + size_t(warp) * 128;
  de[o0] = d[0];
  de[o1] = d[1];
  de[o2] = d[2];
  de[o3] = d[3];
#if __CUDA_ARCH__ >= 800
  float h[4] = {c[o0], c[o1], c[o2], c[o3]};
  if constexpr (kTf32)
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(h[0]), "+f"(h[1]), "+f"(h[2]), "+f"(h[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  else
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(h[0]), "+f"(h[1]), "+f"(h[2]), "+f"(h[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  float* dh = D_hw + size_t(warp) * 128;
  dh[o0] = h[0];
  dh[o1] = h[1];
  dh[o2] = h[2];
  dh[o3] = h[3];
  if (warp == 0 && lane == 0) *hw_ran = 1;
#else
  (void)D_hw;
  (void)hw_ran;
#endif
}

enum class Fill { kInteger, kIntegerLowBits, kRandom };

// Element value as the tensor core reads it (bf16 bits -> fp32; tf32 = fp32 with the low 13 bits unused).
double elem_value(uint32_t bits, bool tf32) {
  return tf32 ? double(u2f(bits & 0xffffe000u)) : double(u2f(bits << 16));
}

template <bool kTf32>
void test_mma(Fill fill, bool sm80, int trials) {
  constexpr int K = kTf32 ? 8 : 16;
  Buffer<uint32_t> A(size_t(trials) * 16 * K), B(size_t(trials) * K * 8);
  Buffer<float> C(size_t(trials) * 128), De(size_t(trials) * 128), Dh(size_t(trials) * 128);
  Buffer<int> hw_ran(1);
  std::mt19937 rng(kTf32 ? 32 : 16);
  std::uniform_int_distribution<int> small(-8, 8);
  std::uniform_real_distribution<float> uni(-1.f, 1.f);
  auto make = [&](bool is_c) -> uint32_t {
    if (fill == Fill::kRandom) {
      const float v = uni(rng);
      if (is_c || kTf32) return f2u(v);              // tf32 operands: full fp32 bits (low 13 bits unused)
      return f2u(v) >> 16;                           // bf16: truncation is fine, any bf16 is a valid input
    }
    const float v = float(is_c ? small(rng) * 8 : small(rng));
    uint32_t u = f2u(v);
    if (is_c) return u;
    if (!kTf32) return u >> 16;                      // small integers are exact in bf16
    if (fill == Fill::kIntegerLowBits && v != 0.f) u |= rng() & 0x1fffu;  // tf32: garbage in the unused bits
    return u;
  };
  for (size_t i = 0; i < A.n; ++i) A[i] = make(false);
  for (size_t i = 0; i < B.n; ++i) B[i] = make(false);
  for (size_t i = 0; i < C.n; ++i) C[i] = u2f(make(true));
  const int threads = 128;
  const int blocks = (trials * 32 + threads - 1) / threads;
  mma_kernel<kTf32><<<blocks, threads>>>(A.p, B.p, C.p, De.p, Dh.p, trials, hw_ran.p);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  long bad_ref = 0, bad_hw = 0;
  double worst_emu = 0, worst_hw = 0;  // |D - ref| / (2^-24 * (|c| + sum |a b|)), random inputs
  for (int w = 0; w < trials; ++w) {
    for (int m = 0; m < 16; ++m) {
      for (int n = 0; n < 8; ++n) {
        const double c = C[size_t(w) * 128 + m * 8 + n];
        double ref = c, mag = std::fabs(c);
        for (int k = 0; k < K; ++k) {
          const double p = elem_value(A[size_t(w) * 16 * K + m * K + k], kTf32) *
                           elem_value(B[size_t(w) * K * 8 + k * 8 + n], kTf32);
          ref += p;
          mag += std::fabs(p);
        }
        const float de = De[size_t(w) * 128 + m * 8 + n];
        const float dh = Dh[size_t(w) * 128 + m * 8 + n];
        if (fill == Fill::kRandom) {
          const double unit = 0x1p-24 * (mag > 0 ? mag : 1.0);
          worst_emu = std::fmax(worst_emu, std::fabs(de - ref) / unit);
          if (sm80) worst_hw = std::fmax(worst_hw, std::fabs(dh - ref) / unit);
        } else {
          bad_ref += !same_value(de, float(ref));  // exact: every partial sum is a small integer
          if (sm80) bad_hw += !same_value(dh, de);
        }
      }
    }
  }
  const char* shape = kTf32 ? "mma m16n8k8 tf32" : "mma m16n8k16 bf16";
  const char* kind = fill == Fill::kInteger ? "integer" : fill == Fill::kIntegerLowBits ? "integer+low bits" : "random";
  const std::string outs = std::to_string(long(trials) * 128) + " outputs";
  char name[96];
  if (fill == Fill::kRandom) {
    // K + 1 fp32 additions: |error| <= (K + 1) * 2^-24 * (|c| + sum |a b|) (first-order bound).
    std::snprintf(name, sizeof(name), "%s %s: within fp32 bound", shape, kind);
    char hw[48] = "no hardware instruction";
    if (sm80) std::snprintf(hw, sizeof(hw), "hardware worst %.2f", worst_hw);
    char d[192];
    std::snprintf(d, sizeof(d), "worst %.2f of %d units (2^-24 x magnitude); %s, %s", worst_emu, K + 1, hw,
                  outs.c_str());
    report(name, worst_emu <= K + 1, d);
  } else {
    std::snprintf(name, sizeof(name), "%s %s: exact", shape, kind);
    report(name, bad_ref == 0, std::to_string(bad_ref) + " of " + outs + " differ from fp64");
    if (sm80) {
      std::snprintf(name, sizeof(name), "%s %s: == hardware", shape, kind);
      report(name, hw_ran[0] == 1 && bad_hw == 0, std::to_string(bad_hw) + " of " + outs + " differ");
    }
  }
}

// ---------------------------------------------------------------------------------------------- arch list
// kt2n::cuda::build_runs_on, used by the engine's create-time check (host only).
void test_build_runs_on() {
  struct Case {
    const char* list;
    int major, minor;
    bool want;
  };
  const char* def = "61-real,75-real,80,86,89,90,100,120";
  const Case cases[] = {
      {def, 6, 1, true},        {def, 6, 2, true},  // the 6.1 cubin runs on 6.2 (create still rejects 6.2)
      {def, 6, 0, false},       {def, 7, 0, false},        {def, 7, 2, false},      {def, 7, 5, true},
      {def, 8, 0, true},        {def, 8, 7, true},         {def, 12, 1, true},      {def, 5, 2, false},
      {"86", 7, 5, false},      {"86", 12, 0, true},       {"86-real", 12, 0, false}, {"86-real", 8, 9, true},
      {"86-virtual", 8, 0, false}, {"86-virtual", 8, 9, true}, {"90a-real", 9, 0, true}, {"90a-real", 10, 0, false},
      {"", 5, 0, true},         {"all", 3, 0, true},       {"75-real,", 7, 5, true}, {"75-real", 6, 1, false},
  };
  int bad = 0;
  for (const Case& c : cases) {
    if (kt2n::cuda::build_runs_on(c.list, c.major, c.minor) != c.want) {
      std::printf("  build_runs_on(\"%s\", %d, %d) != %d\n", c.list, c.major, c.minor, int(c.want));
      ++bad;
    }
  }
  report("build_runs_on (create-time arch list check)", bad == 0,
         std::to_string(bad) + " of " + std::to_string(sizeof(cases) / sizeof(cases[0])) + " cases wrong");
}

// ---------------------------------------------------------------------------------------------- _rn arithmetic
// __fmul_rn / __fadd_rn / __fsub_rn must round at every step, never be fused into an FMA (nvcc guarantees it; the HIP
// build keeps it with include/kt2n/gpu_runtime.h's contract(off) wrappers). Inputs come from memory so nothing folds:
// a = b = 1 + 2^-12, so a * b = 1 + 2^-11 + 2^-24 rounds (tie, to even) to 1 + 2^-11; a fused form keeps the 2^-24.
__global__ void rn_kernel(const float* in, float* out) {
  const float a = in[0], b = in[1], c = in[2], d = in[3];
  out[0] = __fadd_rn(__fmul_rn(a, b), c);  // c = -1
  out[1] = __fsub_rn(__fmul_rn(a, b), d);  // d = +1
  out[2] = __fadd_rn(c, __fmul_rn(a, b));
}

void test_rn_unfused() {
  Buffer<float> in(4), out(3);
  const float a = 1.0f + 0x1p-12f;
  in[0] = a;
  in[1] = a;
  in[2] = -1.0f;
  in[3] = 1.0f;
  rn_kernel<<<1, 1>>>(in.p, out.p);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  volatile float va = a;
  volatile float prod = va * va;  // one rounding: 1 + 2^-11
  const float want = prod - 1.0f;  // 2^-11 exactly
  int bad = 0;
  for (int i = 0; i < 3; ++i) bad += out[i] != want;
  report("__fmul_rn/__fadd_rn/__fsub_rn not fused", bad == 0,
         std::to_string(bad) + " of 3 differ (want 2^-11; a fused FMA gives 2^-11 + 2^-24)");
}

#if defined(KT2N_HIP)
// kt2n::cuda::hip_build_runs_on: the HIP create-time check of gcnArchName against KT2N_HIP_ARCHS.
void test_hip_build_runs_on() {
  struct Case {
    const char* list;
    const char* arch;
    bool want;
  };
  const char* def = "gfx1100,gfx1101,gfx1102,gfx1200,gfx1201";
  const Case cases[] = {
      {def, "gfx1100", true},  {def, "gfx1100:sramecc-:xnack-", true}, {def, "gfx1201", true},
      {def, "gfx1103", false}, {def, "gfx90a:sramecc+:xnack-", false},  {def, "gfx110", false},
      {"gfx1100", "gfx11000", false}, {"", "gfx942", true},             {"gfx1100,", "gfx1100", true},
  };
  int bad = 0;
  for (const Case& c : cases) {
    if (kt2n::cuda::hip_build_runs_on(c.list, c.arch) != c.want) {
      std::printf("  hip_build_runs_on(\"%s\", \"%s\") != %d\n", c.list, c.arch, int(c.want));
      ++bad;
    }
  }
  report("hip_build_runs_on (create-time arch list check)", bad == 0,
         std::to_string(bad) + " of " + std::to_string(sizeof(cases) / sizeof(cases[0])) + " cases wrong");
}
#endif

int main() {
  test_build_runs_on();
#if defined(KT2N_HIP)
  test_hip_build_runs_on();
#endif
  try {
    int dev = 0;
    CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, dev));
#if defined(KT2N_HIP)
    // AMD: no PTX instructions to compare with; the software forms are checked against the references only.
    const int sm = 0;
    std::printf("device %d: %s (%s, wave%d)\n", dev, prop.name, prop.gcnArchName, prop.warpSize);
#else
    const int sm = prop.major * 10 + prop.minor;
    std::printf("device %d: %s (SM %d.%d)\n", dev, prop.name, prop.major, prop.minor);
#endif
    test_rn_unfused();
    test_cvt(sm >= 80);
    test_ldmatrix(sm >= 75);
    test_cp_async(sm >= 80);
    test_mma<false>(Fill::kInteger, sm >= 80, 4096);
    test_mma<false>(Fill::kRandom, sm >= 80, 4096);
    test_mma<true>(Fill::kInteger, sm >= 80, 4096);
    test_mma<true>(Fill::kIntegerLowBits, sm >= 80, 4096);
    test_mma<true>(Fill::kRandom, sm >= 80, 4096);
  } catch (const std::exception& ex) {
    std::printf("FAIL: %s\n", ex.what());
    return 1;
  }
  std::printf("%s (%d failed)\n", g_failures ? "FAILED" : "ALL PASSED", g_failures);
  return g_failures ? 1 : 0;
}
