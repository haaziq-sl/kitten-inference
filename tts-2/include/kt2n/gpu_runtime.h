// GPU runtime portability layer: the engine is written against the CUDA runtime API, and this header lets the
// same sources build with HIP (ROCm) for AMD Radeon GPUs.
//
//   CUDA build (KT2N_HIP undefined): includes <cuda_runtime.h> and nothing else, so CUDA code is unchanged.
//   HIP build  (KT2N_HIP defined):   includes the HIP runtime and maps every CUDA name the engine uses onto
//                                    its HIP equivalent (cudaMalloc -> hipMalloc, __nv_bfloat16 ->
//                                    __hip_bfloat16, ...), plus small shims where HIP differs:
//     * warp intrinsics: HIP's *_sync forms demand a 64-bit lane mask; the engine only uses full-warp masks
//       (0xffffffffu), so they map to the unmasked forms (HIP waves execute in lockstep).
//     * __ldcg / __ldca / __ldcs / __stcg: cache-hint loads and stores, plain accesses on AMD. Every
//       cross-block use in the engine is ordered by __threadfence(), which on AMD invalidates the
//       non-coherent L0/L1 caches, so plain loads see the other blocks' data.
//     * cudaLaunchKernelEx: programmatic dependent launch (PDL) has no HIP equivalent; the attribute is
//       dropped, which only removes overlap (stream order is stricter). Thread-block clusters are refused
//       (hipErrorNotSupported); the portable host paths never request them.
//     * __cvta_generic_to_shared / __cvta_shared_to_generic: 32-bit LDS addresses via address-space casts.
//
// The build includes this header through tiny <cuda_runtime.h>, <cuda_bf16.h>, ... shims (include/kt2n/
// hip_shim/), which are on the include path of HIP builds only; CUDA builds never see them.
#pragma once

#if !defined(KT2N_HIP)

#include <cuda_runtime.h>

#else  // KT2N_HIP

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif
#if !defined(KT2N_PORTABLE)
#define KT2N_PORTABLE 1
#endif

#include <hip/hip_runtime_api.h>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#endif

#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------------------------------------------
// Types and enums
#define cudaError_t hipError_t
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaGraph_t hipGraph_t
#define cudaGraphExec_t hipGraphExec_t
#define cudaGraphNode_t hipGraphNode_t
#define cudaDeviceProp hipDeviceProp_t
#define cudaFuncAttributes hipFuncAttributes
#define cudaMemcpyKind hipMemcpyKind
#define cudaStreamCaptureMode hipStreamCaptureMode
#define cudaMemcpy3DParms hipMemcpy3DParms
#define cudaPitchedPtr hipPitchedPtr
#define cudaExtent hipExtent
#define cudaPos hipPos
#define make_cudaPitchedPtr make_hipPitchedPtr
#define make_cudaExtent make_hipExtent
#define make_cudaPos make_hipPos

#define cudaSuccess hipSuccess
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaErrorInvalidConfiguration hipErrorInvalidConfiguration
#define cudaErrorLaunchFailure hipErrorLaunchFailure
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorNotSupported hipErrorNotSupported

#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaMemcpyDefault hipMemcpyDefault

#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamDefault hipStreamDefault
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventDefault hipEventDefault
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed

// Pinned host memory. "Mapped" memory is the LM's run-ahead ring, which the host polls while kernels write it:
// it must be coherent (fine-grained) on AMD, so the mapped flag asks for that explicitly.
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaHostAllocPortable hipHostMallocPortable
#define cudaHostAllocMapped (hipHostMallocMapped | hipHostMallocCoherent)

#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributePreferredSharedMemoryCarveout hipFuncAttributePreferredSharedMemoryCarveout
#define cudaSharedmemCarveoutMaxShared 100

#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeSharedMemPerBlockOptin
#define cudaDevAttrMaxThreadsPerMultiProcessor hipDeviceAttributeMaxThreadsPerMultiProcessor
#define cudaDevAttrWarpSize hipDeviceAttributeWarpSize
#define cudaDevAttrTccDriver hipDeviceAttributeTccDriver
// No thread-block clusters on AMD: the query answers 0 (see cudaDeviceGetAttribute below).
enum kt2n_hip_cluster_attr { kt2n_hip_attr_cluster_launch };
#define cudaDevAttrClusterLaunch kt2n_hip_attr_cluster_launch

// ---------------------------------------------------------------------------------------------------------------
// Functions with the same signature
#define cudaGetLastError hipGetLastError
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaGetErrorString hipGetErrorString
#define cudaGetErrorName hipGetErrorName
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaMemGetInfo hipMemGetInfo
#define cudaMalloc hipMalloc
#define cudaMallocManaged hipMallocManaged
#define cudaMallocHost hipHostMalloc
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaHostAlloc hipHostMalloc
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2D hipMemcpy2D
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpy3DAsync hipMemcpy3DAsync
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemcpyFromSymbol hipMemcpyFromSymbol
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamQuery hipStreamQuery
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamGetPriority hipStreamGetPriority
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventQuery hipEventQuery
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphUpload hipGraphUpload
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphGetNodes hipGraphGetNodes
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor

// ---------------------------------------------------------------------------------------------------------------
// Functions whose HIP form differs
#if defined(__cplusplus)
#include <utility>

// Cluster launch support: never on AMD; every other attribute is the device's.
inline hipError_t kt2n_hip_device_get_attribute(int* value, hipDeviceAttribute_t attr, int device) {
  return hipDeviceGetAttribute(value, attr, device);
}
inline hipError_t kt2n_hip_device_get_attribute(int* value, kt2n_hip_cluster_attr, int) {
  *value = 0;
  return hipSuccess;
}
#define cudaDeviceGetAttribute kt2n_hip_device_get_attribute

// CUDA 12 form cudaGraphInstantiate(exec, graph, flags) and the older 5-argument form.
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph, unsigned long long flags = 0) {
  return hipGraphInstantiateWithFlags(exec, graph, flags);
}
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph, hipGraphNode_t* error_node,
                                       char* log, size_t log_size) {
  return hipGraphInstantiate(exec, graph, error_node, log, log_size);
}

// Kernel attributes take the kernel itself in CUDA's C++ API.
// The shared-memory carveout is a hint with nothing to configure on AMD (LDS is not shared with L1); answered as
// done, so no stale error is left for a later cudaGetLastError() to report.
template <class T>
inline hipError_t cudaFuncSetAttribute(T* kernel, hipFuncAttribute attr, int value) {
  if (attr == hipFuncAttributePreferredSharedMemoryCarveout) return hipSuccess;
  return hipFuncSetAttribute(reinterpret_cast<const void*>(kernel), attr, value);
}
template <class T>
inline hipError_t cudaFuncGetAttributes(hipFuncAttributes* attr, T* kernel) {
  return hipFuncGetAttributes(attr, reinterpret_cast<const void*>(kernel));
}

// Extensible launch. Only the two attributes the engine uses exist.
enum cudaLaunchAttributeID {
  cudaLaunchAttributeIgnore = 0,
  cudaLaunchAttributeClusterDimension = 4,
  cudaLaunchAttributeProgrammaticStreamSerialization = 6,
};
struct cudaLaunchAttributeValue {
  int programmaticStreamSerializationAllowed;
  struct {
    unsigned int x, y, z;
  } clusterDim;
};
struct cudaLaunchAttribute {
  cudaLaunchAttributeID id;
  cudaLaunchAttributeValue val;
};
struct cudaLaunchConfig_t {
  dim3 gridDim;
  dim3 blockDim;
  size_t dynamicSmemBytes;
  hipStream_t stream;
  cudaLaunchAttribute* attrs;
  unsigned int numAttrs;
};
template <typename... KernelArgs, typename... Args>
inline hipError_t cudaLaunchKernelEx(const cudaLaunchConfig_t* config, void (*kernel)(KernelArgs...),
                                     Args&&... args) {
  for (unsigned int i = 0; i < config->numAttrs; ++i) {
    const cudaLaunchAttribute& a = config->attrs[i];
    if (a.id == cudaLaunchAttributeClusterDimension &&
        a.val.clusterDim.x * a.val.clusterDim.y * a.val.clusterDim.z > 1u)
      return hipErrorNotSupported;  // thread-block clusters: NVIDIA SM 9.0+ only
    // cudaLaunchAttributeProgrammaticStreamSerialization: dropped (plain stream order).
  }
  hipLaunchConfig_t hc{};
  hc.gridDim = config->gridDim;
  hc.blockDim = config->blockDim;
  hc.dynamicSmemBytes = config->dynamicSmemBytes;
  hc.stream = config->stream;
  hc.attrs = nullptr;
  hc.numAttrs = 0;
  return hipLaunchKernelEx(&hc, kernel, std::forward<Args>(args)...);
}
// Thread-block clusters do not exist on AMD: no cluster shape is ever resident.
template <typename... KernelArgs>
inline hipError_t cudaOccupancyMaxActiveClusters(int* count, void (*)(KernelArgs...), const cudaLaunchConfig_t*) {
  *count = 0;
  return hipErrorNotSupported;
}
#endif  // __cplusplus

// ---------------------------------------------------------------------------------------------------------------
// Device-side names
#if defined(__HIPCC__)
typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;
// HIP's float -> bf16 conversion rounds to nearest even, as CUDA's _rn forms do (NaN payloads may differ).
__host__ __device__ __forceinline__ __hip_bfloat16 __float2bfloat16_rn(float f) { return __float2bfloat16(f); }
__host__ __device__ __forceinline__ __hip_bfloat162 __floats2bfloat162_rn(float lo, float hi) {
  return __hip_bfloat162{__float2bfloat16(lo), __float2bfloat16(hi)};
}

// Round-to-nearest arithmetic that is never fused into an FMA, as nvcc guarantees for __fmul_rn / __fadd_rn /
// __fsub_rn. HIP's own versions are plain "x * y", which -ffp-contract=fast(-honor-pragmas) fuses with a neighbouring
// add; the pragma keeps these out of contraction while the rest of the code may still fuse a * b + c, exactly like
// nvcc's default --fmad=true (the HIP build compiles device code with -ffp-contract=fast-honor-pragmas; see
// cmake/Kt2nCuda.cmake, KT2N_HIP_FP_CONTRACT).
__device__ __forceinline__ float kt2n_fmul_rn(float a, float b) {
#pragma clang fp contract(off)
  return a * b;
}
__device__ __forceinline__ float kt2n_fadd_rn(float a, float b) {
#pragma clang fp contract(off)
  return a + b;
}
__device__ __forceinline__ float kt2n_fsub_rn(float a, float b) {
#pragma clang fp contract(off)
  return a - b;
}
#define __fmul_rn kt2n_fmul_rn
#define __fadd_rn kt2n_fadd_rn
#define __fsub_rn kt2n_fsub_rn

// Full-warp shuffles (every call site passes 0xffffffffu). RDNA runs these kernels as wave32. The default width is
// 32, CUDA's warp, not the device's wavefront size: identical code on wave32, and on a 64-lane wavefront each half
// would still shuffle within its own 32 lanes (the engine refuses wave64 devices at create; see cuda_caps.h).
template <class T>
__device__ __forceinline__ T kt2n_shfl(T v, int src, int width = 32) {
  return __shfl(v, src, width);
}
template <class T>
__device__ __forceinline__ T kt2n_shfl_xor(T v, int mask, int width = 32) {
  return __shfl_xor(v, mask, width);
}
template <class T>
__device__ __forceinline__ T kt2n_shfl_down(T v, unsigned int delta, int width = 32) {
  return __shfl_down(v, delta, width);
}
template <class T>
__device__ __forceinline__ T kt2n_shfl_up(T v, unsigned int delta, int width = 32) {
  return __shfl_up(v, delta, width);
}
#define __shfl_sync(mask, ...) kt2n_shfl(__VA_ARGS__)
#define __shfl_xor_sync(mask, ...) kt2n_shfl_xor(__VA_ARGS__)
#define __shfl_down_sync(mask, ...) kt2n_shfl_down(__VA_ARGS__)
#define __shfl_up_sync(mask, ...) kt2n_shfl_up(__VA_ARGS__)

template <class T>
__device__ __forceinline__ T __ldcg(const T* p) {
  return *p;
}
template <class T>
__device__ __forceinline__ T __ldca(const T* p) {
  return *p;
}
template <class T>
__device__ __forceinline__ T __ldcs(const T* p) {
  return *p;
}
template <class T>
__device__ __forceinline__ void __stcg(T* p, T v) {
  *p = v;
}

// Generic <-> LDS (shared memory) addresses. LDS addresses are 32-bit on AMD (address space 3).
__device__ __forceinline__ size_t __cvta_generic_to_shared(const void* p) {
  return static_cast<size_t>(
      reinterpret_cast<uintptr_t>((__attribute__((address_space(3))) const void*)(p)) & 0xffffffffu);
}
__device__ __forceinline__ void* __cvta_shared_to_generic(size_t a) {
  return (void*)((__attribute__((address_space(3))) void*)(static_cast<uintptr_t>(static_cast<uint32_t>(a))));
}
#endif  // __HIPCC__

#endif  // KT2N_HIP
