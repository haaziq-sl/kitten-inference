#pragma once

#include <cuda_runtime.h>

#include "kt2n/gpu_arch.h"  // KT2N_PORTABLE (implied by KT2N_HIP)

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace kt2n::cuda {

#if defined(KT2N_PORTABLE)
// Portable builds (KT2N_PORTABLE: always for HIP / AMD Radeon; optional on CUDA to run the AMD code paths on an
// NVIDIA GPU) compile the SM 6.1 / 7.5 software forms of every SM-specific instruction for every device
// (kt2n/gpu_arch.h), so host dispatch takes every SM 6.1 / 7.5 choice on every device: legacy() is true, PDL and
// clusters are off (their device code is compiled out), and shared memory per block is capped at 64 KB, the LDS
// one workgroup can use on AMD RDNA3 / RDNA4 (so an NVIDIA GPU picks the same kernel shapes as a Radeon).
inline constexpr int kPortableSmemMax = 65536;
enum class Vendor : int { nvidia = 0, amd = 1 };
#if defined(KT2N_HIP)
inline constexpr Vendor kVendor = Vendor::amd;
#else
inline constexpr Vendor kVendor = Vendor::nvidia;
#endif

struct DeviceCaps {
  int major = 0;
  int minor = 0;
  int shared_bytes = 0;
  bool clusters = false;
  Vendor vendor = kVendor;
  // AMD matrix instructions (HIP builds): 11 = RDNA3 (gfx11) WMMA, 12 = RDNA4 (gfx12) WMMA, 0 = none, i.e. the
  // software path (always 0 on CUDA; KT2N_AMD_WMMA=0 forces 0). See kt2n/hip/wmma_kernels.h.
  int wmma = 0;
  bool pdl() const { return false; }
  bool legacy() const { return major > 0; }
  int smem_optin() const { return shared_bytes; }
};

// Shared memory per block a portable build may use on `device`: the opt-in limit (or the per-block limit when the
// runtime reports no opt-in), at most kPortableSmemMax. cudaDeviceGetAttribute only, so it is safe inside a stream
// capture. 0 when the query fails.
inline int portable_smem_optin(int device) {
  int optin = 0, plain = 0;
  if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) != cudaSuccess) optin = 0;
#if defined(KT2N_HIP)
  if (hipDeviceGetAttribute(&plain, hipDeviceAttributeMaxSharedMemoryPerBlock, device) != hipSuccess) plain = 0;
#endif
  const int bytes = optin > plain ? optin : plain;
  return bytes < kPortableSmemMax ? bytes : kPortableSmemMax;
}
#else
struct DeviceCaps {
  int major = 0;
  int minor = 0;
  int shared_bytes = 0;
  bool clusters = false;
  bool pdl() const { return major >= 9; }
  // Turing / Pascal (SM 7.5 / 6.1): kernels use the kt2n::compat paths and host dispatch must respect a
  // smaller opt-in shared memory. Gate every such fallback on legacy() so SM 8.0+ choices never change.
  // False when the query failed (major 0), so a failed query keeps the SM 8.0+ behaviour.
  bool legacy() const { return major > 0 && major < 8; }
  // Largest dynamic + static shared memory one block may opt in to (cudaDevAttrMaxSharedMemoryPerBlockOptin):
  // SM 6.1 49152, SM 7.5 65536, SM 8.6 / 8.9 / 12.0 101376, SM 8.0 166912, SM 9.0 / 10.0 232448.
  int smem_optin() const { return shared_bytes; }
};
#endif

#if defined(KT2N_HIP)
// WMMA generation of an AMD GPU from its gcnArchName ("gfx1100", "gfx1201:sramecc-:xnack-"): 11 / 12 for exactly the
// targets whose device code has the WMMA kernels (the list of KT2N_WMMA_ARCH in kt2n/hip/wmma.cuh; keep both in step),
// else 0 (software path). KT2N_AMD_WMMA=0 in the environment (read once per process) returns 0 for every GPU.
inline int hip_wmma_arch(const char* gcn_arch) {
  static const bool off = [] {
    const char* e = std::getenv("KT2N_AMD_WMMA");
    return e != nullptr && e[0] == '0' && e[1] == '\0';
  }();
  if (off || gcn_arch == nullptr) return 0;
  static const char* const kGfx11[] = {"gfx1100", "gfx1101", "gfx1102", "gfx1103",
                                       "gfx1150", "gfx1151", "gfx1152", "gfx1153"};
  static const char* const kGfx12[] = {"gfx1200", "gfx1201"};
  const size_t n = std::strcspn(gcn_arch, ":");  // the target name before any feature flags
  for (const char* t : kGfx11)
    if (std::strlen(t) == n && std::strncmp(gcn_arch, t, n) == 0) return 11;
  for (const char* t : kGfx12)
    if (std::strlen(t) == n && std::strncmp(gcn_arch, t, n) == 0) return 12;
  return 0;
}
#endif

// Device properties are immutable. Cache them per host thread without a lock in the launch path, and once per
// process: a thread's first lookup copies the process's entry (under a mutex, once per thread and device) instead
// of querying the driver again. cudaGetDeviceProperties takes milliseconds on some hosts, and the engine's worker
// threads would otherwise each pay it at their first launch, inside the first request (its first audio).
inline DeviceCaps device_caps(int device) {
  static thread_local std::array<DeviceCaps, 64> cache{};
  const bool cached = device >= 0 && device < int(cache.size());
  if (cached && cache[device].major) return cache[device];
  static std::mutex shared_mu;
  static std::array<DeviceCaps, 64> shared{};
  if (cached) {
    std::lock_guard<std::mutex> lk(shared_mu);
    if (shared[device].major) return cache[device] = shared[device];
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return {};
#if defined(KT2N_PORTABLE)
  DeviceCaps caps{prop.major, prop.minor, portable_smem_optin(device), false};
#if defined(KT2N_HIP)
  caps.wmma = hip_wmma_arch(prop.gcnArchName);
#endif
#else
  DeviceCaps caps{prop.major, prop.minor, int(prop.sharedMemPerBlockOptin), false};
  if (prop.major >= 9) {
    int supported = 0;
    if (cudaDeviceGetAttribute(&supported, cudaDevAttrClusterLaunch, device) == cudaSuccess)
      caps.clusters = supported != 0;
  }
#endif
  if (cached) {
    std::lock_guard<std::mutex> lk(shared_mu);
    shared[device] = caps;
    cache[device] = caps;
  }
  return caps;
}

inline DeviceCaps current_caps() {
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) return {};
  return device_caps(device);
}

inline bool pdl_enabled(bool requested) { return requested && current_caps().pdl(); }

// Whether code built for `arch_list` (CMake CUDA architectures, comma separated, e.g. "61-real,75-real,80,86") runs
// on SM major.minor: a cubin ("XY", "XY-real") runs on SM X.z for z >= Y; PTX ("XY", "XY-virtual") is compiled by
// the driver for any SM >= X.Y; an arch-specific cubin ("90a-real") only on that SM. An empty list, or "all",
// "all-major" or "native", is not checked (true).
inline bool build_runs_on(const char* arch_list, int major, int minor) {
  if (arch_list == nullptr || *arch_list == '\0') return true;
  const int sm = major * 10 + minor;
  std::string list(arch_list);
  size_t pos = 0;
  while (pos <= list.size()) {
    const size_t end = std::min(list.find(',', pos), list.size());
    const std::string item = list.substr(pos, end - pos);
    pos = end + 1;
    if (item.empty()) continue;
    if (item == "all" || item == "all-major" || item == "native") return true;
    size_t digits = 0;
    while (digits < item.size() && item[digits] >= '0' && item[digits] <= '9') ++digits;
    if (digits == 0) continue;
    const int n = std::atoi(item.substr(0, digits).c_str());
    std::string rest = item.substr(digits);
    const bool specific = !rest.empty() && (rest[0] == 'a' || rest[0] == 'f');
    if (specific) rest = rest.substr(1);
    const bool cubin = rest != "-virtual";
    const bool ptx = rest != "-real" && !specific;
    if (cubin && n / 10 == major && (specific ? n % 10 == minor : n % 10 <= minor)) return true;
    if (ptx && n <= sm) return true;
  }
  return false;
}

// Create-time GPU check (host only, no kernel referenced): the native engine runs on SM 6.1, SM 7.5 and SM 8.0 or
// newer, where the build has code for the GPU (build_runs_on). Returns 0 when supported; -1 with "this GPU (SM x.y)
// is not supported..." in *why; otherwise the cudaError_t of the failed device query, with its text in *why.
#if defined(KT2N_HIP)
// AMD: whether the build has code for `gcn_arch` ("gfx1100" or "gfx1100:sramecc-:xnack-"; target features after
// the ':' are ignored): arch_list is the comma-separated KT2N_HIP_ARCHS.
inline bool hip_build_runs_on(const char* arch_list, const char* gcn_arch) {
  if (arch_list == nullptr || *arch_list == '\0') return true;
  std::string arch(gcn_arch);
  arch = arch.substr(0, arch.find(':'));
  std::string list(arch_list);
  size_t pos = 0;
  while (pos <= list.size()) {
    const size_t end = std::min(list.find(',', pos), list.size());
    std::string item = list.substr(pos, end - pos);
    pos = end + 1;
    item = item.substr(0, item.find(':'));
    if (!item.empty() && item == arch) return true;
  }
  return false;
}
#endif

inline int arch_check(int device, const char* arch_list, std::string* why) {
  cudaDeviceProp prop{};
  const cudaError_t err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    if (why) *why = cudaGetErrorString(err);
    return static_cast<int>(err);
  }
  char text[256];
#if defined(KT2N_HIP)
  // The kernels' warp code (shuffles, the software MMA fragments) assumes 32 lanes: RDNA's wave32. CDNA (MI-series)
  // and older GCN GPUs run 64-lane wavefronts and are refused rather than computing wrong results.
  if (prop.warpSize != 32) {
    std::snprintf(text, sizeof(text),
                  "this GPU (%s, %s) runs %d-lane wavefronts; the native engine needs a 32-lane (wave32) AMD GPU "
                  "such as Radeon RX 7000 (RDNA3) or RX 9000 (RDNA4)",
                  prop.name, prop.gcnArchName, prop.warpSize);
    if (why) *why = text;
    return -1;
  }
  const bool rdna34 = std::strncmp(prop.gcnArchName, "gfx11", 5) == 0 || std::strncmp(prop.gcnArchName, "gfx12", 5) == 0;
  if (!rdna34) {
    std::snprintf(text, sizeof(text),
                  "this GPU (%s, %s) is not supported; the native engine needs an AMD Radeon RX 7000 (RDNA3) or "
                  "RX 9000 (RDNA4) GPU",
                  prop.name, prop.gcnArchName);
    if (why) *why = text;
    return -1;
  }
  if (!hip_build_runs_on(arch_list, prop.gcnArchName)) {
    std::snprintf(text, sizeof(text),
                  "this GPU (%s, %s) is not supported by this build, which has no kernels for it (built for %s); "
                  "rebuild with KT2N_HIP_ARCHS including it",
                  prop.name, prop.gcnArchName, arch_list);
    if (why) *why = text;
    return -1;
  }
  return 0;
#else
  const bool arch_ok = (prop.major == 6 && prop.minor == 1) || (prop.major == 7 && prop.minor == 5) || prop.major >= 8;
  if (!arch_ok) {
    std::snprintf(text, sizeof(text),
                  "this GPU (SM %d.%d) is not supported; the native engine needs SM 6.1, SM 7.5 or SM 8.0 and newer",
                  prop.major, prop.minor);
    if (why) *why = text;
    return -1;
  }
#if CUDART_VERSION >= 13000
  // CUDA 13 cannot build for SM 6.x, so no CUDA 13 build has code for the GTX 10-series, whatever its architecture
  // list (also "all", "all-major" or "native", which build_runs_on does not check; cmake/Kt2nCuda.cmake).
  if (prop.major < 7) {
    std::snprintf(text, sizeof(text),
                  "this GPU (SM %d.%d) is not supported by this build, which was made with CUDA 13 (built for %s); "
                  "CUDA 13 cannot build for SM %d.%d: build with CUDA 12.8 or 12.9 for this GPU",
                  prop.major, prop.minor, arch_list, prop.major, prop.minor);
    if (why) *why = text;
    return -1;
  }
#endif
  if (!build_runs_on(arch_list, prop.major, prop.minor)) {
    std::snprintf(text, sizeof(text),
                  "this GPU (SM %d.%d) is not supported by this build, which has no kernels for it (built for %s); "
                  "rebuild with KT2N_CUDA_ARCHS including %d%d",
                  prop.major, prop.minor, arch_list, prop.major, prop.minor);
    if (why) *why = text;
    return -1;
  }
  return 0;
#endif
}

}  // namespace kt2n::cuda
