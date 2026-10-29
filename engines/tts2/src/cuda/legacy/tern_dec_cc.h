// CUDA-core ternary decode projections (src/cuda/legacy/tern_dec_cc.cu): internal launcher, not public API.
//
// Built only for the archs without bf16 tensor-core instructions (SM 6.x, SM 7.5; CMake KT2N_CUDA_ARCHS entries
// below 80), so the SM 8.0+ code of the library does not change. kt2n_tern_proj_ex uses it for decode rows
// (m <= 16) on those GPUs instead of the decode kernel's software MMA; KT2N_TERN_DECODE=mma keeps the old kernel.
#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace kt2n_tern_cc {

// Decode rows up to which the kernel beats the software-MMA decode kernel (measured: GTX 1080 Ti for SM 6.x; RTX 2060
// and GTX 1660 Ti for SM 7.5); above it launch() declines and the caller keeps the old kernel (same bits either way).
inline constexpr int kMaxRows = 4;

// One projection with the family's fixed arithmetic (c = K / 256 chunks, wk partial sums, acc accumulators per
// partial), bit-identical to the decode kernel's. Arguments as kt2n_tern_proj, already validated by the caller.
// Returns cudaErrorInvalidConfiguration, with nothing launched, when (c, wk, acc), m or n has no instantiated
// shape, m > kMaxRows, or the kernel has no image for this GPU (a pre-8.0 GPU outside the 6.1 / 7.5 build targets,
// e.g. SM 6.0 / 7.0 in an `all` build): the caller then runs the decode kernel.
cudaError_t launch(int c, int wk, int acc, int epi, const void* x, int ldx, int m, const void* codes,
                   const void* scales, int n, void* y, int ldy, const void* residual, int ldr, cudaStream_t stream);

// Whether this build has the kernel for the current GPU (false: launch() always declines there).
bool available();

// Calls of launch() in this process that ran the CUDA-core kernel (tests: which path kt2n_tern_proj took).
unsigned long launches();

}  // namespace kt2n_tern_cc
