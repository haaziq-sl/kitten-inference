// CPU reference of cuRAND's Philox4x32-10 device generator (curand_init / curand4 / curand_normal4 as written in
// CUDA 12.8's curand_kernel.h, curand_philox4x32_x.h and curand_normal.h), for checking that the HIP build's
// generator (rocRAND, include/kt2n/gpu_rand.h) and the CUDA build's cuRAND give the same stream.
#pragma once

#include <cmath>
#include <cstdint>

namespace kt2n_philox_ref {

struct U4 {
  uint32_t x, y, z, w;
};

inline U4 round1(U4 c, uint32_t k0, uint32_t k1) {
  const uint64_t p0 = uint64_t(0xD2511F53u) * c.x;
  const uint64_t p1 = uint64_t(0xCD9E8D57u) * c.z;
  return U4{uint32_t(p1 >> 32) ^ c.y ^ k0, uint32_t(p1), uint32_t(p0 >> 32) ^ c.w ^ k1, uint32_t(p0)};
}

inline U4 philox10(U4 c, uint32_t k0, uint32_t k1) {
  for (int r = 0; r < 10; ++r) {
    c = round1(c, k0, k1);
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return c;
}

struct State {
  U4 ctr{0, 0, 0, 0};
  uint32_t k0 = 0, k1 = 0;
  U4 out{0, 0, 0, 0};
  int sub = 0;  // cuRAND's STATE: position inside the current 4-word block

  void incr() {  // Philox_State_Incr(s)
    if (++ctr.x) return;
    if (++ctr.y) return;
    if (++ctr.z) return;
    ++ctr.w;
  }
  void incr(uint64_t n) {  // Philox_State_Incr(s, n)
    const uint32_t nlo = uint32_t(n);
    uint32_t nhi = uint32_t(n >> 32);
    ctr.x += nlo;
    if (ctr.x < nlo) nhi++;
    ctr.y += nhi;
    if (nhi <= ctr.y) return;
    if (++ctr.z) return;
    ++ctr.w;
  }
  void incr_hi(uint64_t n) {  // Philox_State_Incr_hi(s, n)
    const uint32_t nlo = uint32_t(n);
    uint32_t nhi = uint32_t(n >> 32);
    ctr.z += nlo;
    if (ctr.z < nlo) nhi++;
    ctr.w += nhi;
  }
  // curand_init(seed, subsequence, offset, &state)
  void init(uint64_t seed, uint64_t subsequence, uint64_t offset) {
    ctr = U4{0, 0, 0, 0};
    k0 = uint32_t(seed);
    k1 = uint32_t(seed >> 32);
    sub = 0;
    incr_hi(subsequence);  // skipahead_sequence
    out = philox10(ctr, k0, k1);
    sub += int(offset & 3);  // skipahead
    uint64_t n = offset / 4;
    if (sub > 3) {
      n += 1;
      sub -= 4;
    }
    incr(n);
    out = philox10(ctr, k0, k1);
  }
  // curand4(&state)
  U4 next4() {
    const U4 tmp = out;
    incr();
    out = philox10(ctr, k0, k1);
    switch (sub) {
      case 1: return U4{tmp.y, tmp.z, tmp.w, out.x};
      case 2: return U4{tmp.z, tmp.w, out.x, out.y};
      case 3: return U4{tmp.w, out.x, out.y, out.z};
      default: return tmp;
    }
  }
};

// _curand_box_muller in float (the device form uses __sincosf; this one the C library's sinf / cosf).
inline void box_muller(uint32_t x, uint32_t y, float* a, float* b) {
  const float inv = 2.3283064e-10f;
  const float inv_2pi = 2.3283064e-10f * 6.2831855f;
  const float u = std::fma(float(x), inv, inv / 2);
  const float v = std::fma(float(y), inv_2pi, inv_2pi / 2);
  const float s = std::sqrt(-2.0f * std::log(u));
  *a = std::sin(v) * s;
  *b = std::cos(v) * s;
}

}  // namespace kt2n_philox_ref
