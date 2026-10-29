// Architecture guards for device code, in one place.
//
// The kernels choose between an SM-specific PTX instruction (mma.sync, ldmatrix, cp.async, griddepcontrol,
// clusters, ...) and its software form in kt2n/cuda_compat.cuh with preprocessor guards. Written directly on
// __CUDA_ARCH__ they break on HIP, where __CUDA_ARCH__ is never defined: "!defined(__CUDA_ARCH__) ||
// __CUDA_ARCH__ >= 800" is then true in the device pass too, so an AMD build would pick the PTX side. These
// three macros keep the CUDA meaning exactly and add KT2N_PORTABLE, which selects the software side
// everywhere (always on for HIP; optional on CUDA, to run the AMD code paths on an NVIDIA GPU):
//
//   KT2N_SM_ASM(n)      the hardware instruction that needs SM n: CUDA device pass for SM >= n, and the CUDA
//                       host pass (as before: the host pass never compiles device bodies). 0 when portable.
//   KT2N_SM_ONLY(n)     code that exists only on SM >= n devices (PDL, clusters, bulk prefetch): CUDA device
//                       pass for SM >= n; 0 in the host pass and when portable.
//   KT2N_SM_EMULATE(n)  the software form of an SM n feature: CUDA device pass below SM n; 1 when portable;
//                       0 in the CUDA host pass.
//
// Without KT2N_PORTABLE each macro expands to the condition the guard had before, so CUDA builds compile the
// same code.
#pragma once

#if defined(KT2N_HIP) && !defined(KT2N_PORTABLE)
#define KT2N_PORTABLE 1
#endif

#if defined(KT2N_PORTABLE)
#define KT2N_SM_ASM(n) 0
#define KT2N_SM_ONLY(n) 0
#define KT2N_SM_EMULATE(n) 1
#elif defined(__CUDA_ARCH__)
#define KT2N_SM_ASM(n) (__CUDA_ARCH__ >= (n))
#define KT2N_SM_ONLY(n) (__CUDA_ARCH__ >= (n))
#define KT2N_SM_EMULATE(n) (__CUDA_ARCH__ < (n))
#else
#define KT2N_SM_ASM(n) 1
#define KT2N_SM_ONLY(n) 0
#define KT2N_SM_EMULATE(n) 0
#endif

// Device pass of either toolchain (nvcc: __CUDA_ARCH__; HIP clang: __HIP_DEVICE_COMPILE__).
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define KT2N_DEVICE_PASS 1
#else
#define KT2N_DEVICE_PASS 0
#endif

// Abort the kernel (unreachable dispatch). PTX "trap" on CUDA; the LLVM trap (s_trap) on AMD.
#if defined(KT2N_HIP)
#define KT2N_TRAP() __builtin_trap()
#else
#define KT2N_TRAP() asm volatile("trap;")
#endif
