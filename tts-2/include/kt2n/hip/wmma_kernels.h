/* kt2n AMD matrix-instruction (WMMA) kernels: RDNA3 (gfx11) and RDNA4 (gfx12) Radeon GPUs, HIP builds only.
 *
 * These are faster forms of operators whose reference is the portable software path (KT2N_PORTABLE, the code the
 * GTX 10 / 16 series also run). They are NOT bitwise equal to it: the matrix unit forms the same exact bf16 products
 * but adds them in its own order (see kt2n/hip/wmma.cuh). Each operator keeps a fixed per-row arithmetic, so a row's
 * result does not depend on M, the launch shape or the other rows (batch invariance), as in the reference.
 *
 * Selection: kt2n::cuda::DeviceCaps::wmma (kt2n/cuda_caps.h) is 11 / 12 on gfx11 / gfx12 GPUs of a HIP build and 0
 * elsewhere; the environment variable KT2N_AMD_WMMA=0 (read once per process) forces 0, i.e. the software path.
 * Every function returns 0 on success, 1 (EINVAL: nothing launched; the caller keeps its software path) or 2 (launch
 * error). */
#ifndef KT2N_HIP_WMMA_KERNELS_H
#define KT2N_HIP_WMMA_KERNELS_H

#include <stdint.h>

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KT2N_WMMA_OK 0
#define KT2N_WMMA_EINVAL 1
#define KT2N_WMMA_ECUDA 2

/* Ternary projection Y = X W^T on the matrix unit, same arguments and epilogues as kt2n_tern_proj
 * (kt2n/kernels/tern_proj.h): codes / scales in the projection kernel layout, epi KT2N_TERN_EPI_*.
 * Arithmetic: K is cut into 8 equal ranges; each range is one fp32 accumulator chain over k ascending in blocks of 16
 * (one WMMA each); the output is (((p0 + p1) + p2) + ... + p7), rounded once to bf16 by the epilogue. Any m >= 1;
 * n % 64 == 0; k % 512 == 0; x 16-byte aligned with ldx % 8 == 0.
 * shape: 0 = automatic, else P * 100 + MT * 10 + NT (16-row x 16-column tiles per wave; P = 1: the 8 ranges on 8
 * waves of one block, P = 0: one after the other in one wave), for tuning; never changes a bit. */
int kt2n_wmma_tern_proj(const void* x, int64_t ldx, int m, const void* codes, const void* scales, int n, int k,
                        void* y, int64_t ldy, int epi, const void* residual, int64_t ldr, int shape,
                        cudaStream_t stream);

/* 1 when the current device runs the WMMA kernels of this build (DeviceCaps::wmma != 0), else 0. */
int kt2n_wmma_enabled(void);

/* Decode on the matrix unit (environment KT2N_AMD_WMMA_DECODE, read once; default 2): 0 = off (decode on the software
 * path), 1 = the decode step's projections, 2 = projections and attention (the decode step's, and the prefill's
 * attention through the same kernel, so prefill and decode keep one arithmetic, except in SPLIT64 step buckets, which
 * fold the softmax in another order as on NVIDIA). 0 when the device has no WMMA.
 * 0 and 1 are test switches; the supported fallback is KT2N_AMD_WMMA=0 (everything on the software path). */
int kt2n_wmma_decode_mode(void);

#ifdef __cplusplus
}
#endif

#endif /* KT2N_HIP_WMMA_KERNELS_H */
