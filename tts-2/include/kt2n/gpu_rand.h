// cuRAND's Philox4x32-10 device API (the part the engine uses: curand_init, curand4, curand_normal4) for HIP
// builds, on rocRAND. CUDA builds include <curand_kernel.h> itself and never see this header.
//
// Raw stream: rocRAND's Philox4x32-10 (rocrand_init(seed, subsequence, offset) + rocrand4) produces the same
// uint4 sequence as cuRAND's for the same seed / subsequence / offset: key = seed, counter = (offset / 4,
// subsequence) with the same 10 rounds and key schedule. Checked against a CPU reference written from
// cuRAND's source (tests/gpu/philox_ref.h) by tests/gpu/philox.cu: philox.rocrand_host (host) and gpu.philox.
//
// Normal draws: rocrand_normal4 is NOT cuRAND's transform (rocRAND maps x to u = 2^-32 + x 2^-32, cuRAND to
// u = x 2^-32 + 2^-33, and likewise for the angle), so it would give different numbers from the same raw
// draws. curand_normal4 below is cuRAND's Box-Muller (curand_normal.h, _curand_box_muller) over rocRAND's raw
// stream. The remaining difference from an NVIDIA GPU is the hardware approximation of __sincosf and the last
// bit of logf: the noise values agree to a few float ulps, not bit for bit.
#pragma once

#if defined(KT2N_HIP) && defined(__HIPCC__)

#include <hip/hip_runtime.h>
#include <rocrand/rocrand_kernel.h>

typedef rocrand_state_philox4x32_10 curandStatePhilox4_32_10_t;

__device__ __forceinline__ void curand_init(unsigned long long seed, unsigned long long subsequence,
                                            unsigned long long offset, curandStatePhilox4_32_10_t* state) {
  rocrand_init(seed, subsequence, offset, state);
}

__device__ __forceinline__ uint4 curand4(curandStatePhilox4_32_10_t* state) { return rocrand4(state); }

// cuRAND _curand_box_muller (device form): CURAND_2POW32_INV = 2^-32, CURAND_2POW32_INV_2PI = 2^-32 * 2pi.
__device__ __forceinline__ float2 kt2n_curand_box_muller(unsigned int x, unsigned int y) {
  const float inv = 2.3283064e-10f;
  const float inv_2pi = 2.3283064e-10f * 6.2831855f;
  const float u = x * inv + (inv / 2);
  const float v = y * inv_2pi + (inv_2pi / 2);
  const float s = sqrtf(-2.0f * logf(u));
  float2 r;
  __sincosf(v, &r.x, &r.y);
  r.x *= s;
  r.y *= s;
  return r;
}

__device__ __forceinline__ float4 curand_normal4(curandStatePhilox4_32_10_t* state) {
  const uint4 x = curand4(state);
  const float2 a = kt2n_curand_box_muller(x.x, x.y);
  const float2 b = kt2n_curand_box_muller(x.z, x.w);
  return make_float4(a.x, a.y, b.x, b.y);
}

#endif
