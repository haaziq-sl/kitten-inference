# CUDA selection.
#
#   KT2N_CUDA=AUTO  use CUDA when an nvcc >= 12.8 is found, otherwise build CPU only
#   KT2N_CUDA=ON    require nvcc >= 12.8 (configure fails otherwise)
#   KT2N_CUDA=OFF   CPU only
#
#   KT2N_CUDA_ARCHS default "61-real;75-real;80;86;89;90;100;120" (Pascal, Turing, Ampere, Ada, Hopper and
#   Blackwell). Each plain entry builds a cubin (sm_XX) plus PTX for that arch; a "-real" entry builds only
#   the cubin, so SM 6.1 and 7.5 kernels are never JIT-compiled for other GPUs (the engine rejects
#   SM 6.0 / 6.2 / 7.0 / 7.2 at create). Adding 61-real and 75-real leaves the SM 8.0+ entries, and their
#   code, unchanged. CUDA 13 cannot build 61-real: with the default list it is left out (with a warning), and an
#   explicit list with an entry below 75 fails at configure (below).
#
# Point CMake at a specific toolkit with -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc.
#
# AMD Radeon (HIP / ROCm):
#   KT2N_HIP=ON          build the GPU targets with HIP instead of CUDA (CMake HIP language, ROCm clang). The .cu
#                        sources compile as HIP through include/kt2n/gpu_runtime.h (CUDA names mapped to HIP) and
#                        the <cuda_*.h> shims in include/kt2n/hip_shim/; KT2N_PORTABLE is implied.
#   KT2N_HIP_ARCHS       default "gfx1100;gfx1101;gfx1102;gfx1200;gfx1201" (RDNA3 and RDNA4, wave32).
#   KT2N_HIP_FP_CONTRACT default "fast-honor-pragmas": device code may fuse any a * b + c into an FMA, as nvcc's
#                        default --fmad=true (which the CUDA build uses) does, while __fmul_rn / __fadd_rn /
#                        __fsub_rn are never fused (include/kt2n/gpu_runtime.h wraps them in
#                        "#pragma clang fp contract(off)"; HIP's own versions are plain x * y, which this mode
#                        would fuse). "on" fuses only within one expression; "off" never. Host code never fuses.
#   KT2N_PORTABLE=ON     (with CUDA) compile the software forms of every SM-specific instruction, i.e. the code
#                        paths an AMD build runs, for validating them on an NVIDIA GPU (include/kt2n/gpu_arch.h).
#                        For validation only: slower than the default CUDA build.
#
# Package size (the wheel presets; the defaults leave the build as it is):
#   KT2N_CUDA_COMPRESS_MODE  nvcc --compress-mode for the embedded GPU code (CUDA 12.8+): default, size, speed,
#                        balance or none; empty (default) passes nothing. It changes how the code is stored, not the
#                        code: the driver decompresses it when it loads the library.
#   KT2N_CUDA_LINEINFO   ON (default): -lineinfo, source lines for profilers; OFF leaves them out (about 48 MB less
#                        library with the wheel presets). The SASS is the same either way (measured: 32 compiles of
#                        each give the same set of variants of the one kernel whose PTX nvcc does not write the same
#                        every time, see packaging/sass_manifest.py).
#
# Numerics: never --use_fast_math, never -ftz=true (configure refuses them).
# The CUDA runtime is linked statically (the machine that runs it may have no nvcc and we
# do not want to depend on a particular libcudart.so).

option(KT2N_HIP "Build the GPU engine with HIP (ROCm) for AMD Radeon GPUs instead of CUDA" OFF)
set(KT2N_HIP_ARCHS "gfx1100;gfx1101;gfx1102;gfx1200;gfx1201" CACHE STRING "AMD GPU targets for KT2N_HIP builds")
set(KT2N_HIP_FP_CONTRACT "fast-honor-pragmas" CACHE STRING
  "HIP device -ffp-contract mode: fast-honor-pragmas (as nvcc --fmad=true), on, off")
option(KT2N_PORTABLE "Compile the portable (AMD) software paths instead of SM-specific instructions" OFF)
set(KT2N_CUDA_COMPRESS_MODE "" CACHE STRING "nvcc --compress-mode for the GPU code (default, size, speed, balance, none; empty: not passed)")
option(KT2N_CUDA_LINEINFO "Compile CUDA code with -lineinfo (source lines for profilers)" ON)
if(KT2N_CUDA_COMPRESS_MODE AND NOT KT2N_CUDA_COMPRESS_MODE MATCHES "^(default|size|speed|balance|none)$")
  message(FATAL_ERROR "KT2N_CUDA_COMPRESS_MODE must be default, size, speed, balance or none (got '${KT2N_CUDA_COMPRESS_MODE}')")
endif()
set(KT2N_CUDA "AUTO" CACHE STRING "Build CUDA targets: AUTO, ON or OFF")
set_property(CACHE KT2N_CUDA PROPERTY STRINGS AUTO ON OFF)
set(_kt2n_default_cuda_archs "61-real;75-real;80;86;89;90;100;120")
set(KT2N_CUDA_ARCHS "${_kt2n_default_cuda_archs}" CACHE STRING "CUDA architectures for kt2n_cuda and GPU tests")
set(KT2N_CUDA_MIN_VERSION "12.8")

set(KT2N_HAVE_CUDA OFF)
set(KT2N_NVCC_VERSION "")
string(TOUPPER "${KT2N_CUDA}" _kt2n_cuda_mode)

if(NOT _kt2n_cuda_mode MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "KT2N_CUDA must be AUTO, ON or OFF (got '${KT2N_CUDA}')")
endif()

set(KT2N_HAVE_HIP OFF)
set(KT2N_GPU_RT CUDA::cudart_static)  # the GPU runtime the targets link
set(KT2N_GPU_LANG CUDA)               # LINKER_LANGUAGE of targets with GPU objects
if(KT2N_HIP)
  if(_kt2n_cuda_mode STREQUAL "ON")
    message(FATAL_ERROR "KT2N_HIP=ON and KT2N_CUDA=ON: pick one GPU toolchain")
  endif()
  set(_kt2n_cuda_mode "OFF")
endif()

if(NOT _kt2n_cuda_mode STREQUAL "OFF")
  include(CheckLanguage)
  if(NOT CMAKE_CUDA_COMPILER)
    check_language(CUDA)
  endif()
  if(CMAKE_CUDA_COMPILER)
    execute_process(COMMAND "${CMAKE_CUDA_COMPILER}" --version
      OUTPUT_VARIABLE _kt2n_nvcc_out ERROR_QUIET RESULT_VARIABLE _kt2n_nvcc_rc)
    string(REGEX MATCH "release ([0-9]+\\.[0-9]+)" _kt2n_match "${_kt2n_nvcc_out}")
    set(KT2N_NVCC_VERSION "${CMAKE_MATCH_1}")
    if(NOT _kt2n_nvcc_rc EQUAL 0 OR KT2N_NVCC_VERSION STREQUAL "")
      set(_kt2n_why "could not run '${CMAKE_CUDA_COMPILER} --version'")
    elseif(KT2N_NVCC_VERSION VERSION_LESS KT2N_CUDA_MIN_VERSION)
      set(_kt2n_why "nvcc ${KT2N_NVCC_VERSION} < ${KT2N_CUDA_MIN_VERSION} (sm_100/sm_120 need 12.8)")
    else()
      set(KT2N_HAVE_CUDA ON)
    endif()
  else()
    set(_kt2n_why "no CUDA compiler found")
  endif()

  if(NOT KT2N_HAVE_CUDA)
    if(_kt2n_cuda_mode STREQUAL "ON")
      message(FATAL_ERROR "KT2N_CUDA=ON but ${_kt2n_why}. "
        "Pass -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc or use KT2N_CUDA=AUTO/OFF.")
    else()
      message(STATUS "kt2n: CUDA disabled (${_kt2n_why})")
    endif()
  endif()
endif()

# CUDA 13 no longer compiles for compute capability below 7.5, so it cannot build the default list's 61-real
# (GeForce GTX 10-series). With the default list the build drops that entry and says so: every other GPU gets the
# same kernels, and a GTX 10 GPU is refused at engine start with a message that names CUDA 12.x. An explicit list
# with an entry below 7.5 stops here instead. nvcc 12.x builds are not affected.
if(KT2N_HAVE_CUDA AND KT2N_NVCC_VERSION VERSION_GREATER_EQUAL "13.0")
  set(_kt2n_kept "")
  set(_kt2n_dropped "")
  foreach(_kt2n_arch IN LISTS KT2N_CUDA_ARCHS)
    string(REGEX MATCH "^[0-9]+" _kt2n_arch_num "${_kt2n_arch}")
    if(_kt2n_arch_num AND _kt2n_arch_num LESS 75)
      list(APPEND _kt2n_dropped "${_kt2n_arch}")
    else()
      list(APPEND _kt2n_kept "${_kt2n_arch}")
    endif()
  endforeach()
  if(_kt2n_dropped)
    if(KT2N_CUDA_ARCHS STREQUAL _kt2n_default_cuda_archs)
      message(WARNING "kt2n: CUDA ${KT2N_NVCC_VERSION} cannot build for GeForce GTX 10-series GPUs (compute "
        "capability 6.1), so this build leaves them out (architectures ${_kt2n_kept}) and the engine refuses "
        "them at start. To support GTX 10-series GPUs, build with CUDA 12.8 or 12.9: "
        "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc")
      set(KT2N_CUDA_ARCHS "${_kt2n_kept}")  # this configure only; the cache keeps the default list
    else()
      message(FATAL_ERROR "KT2N_CUDA_ARCHS (${KT2N_CUDA_ARCHS}) includes ${_kt2n_dropped}, which CUDA "
        "${KT2N_NVCC_VERSION} cannot build (CUDA 13 dropped compute capability below 7.5). Remove it, or build "
        "with CUDA 12.8 or 12.9 for GeForce GTX 10-series GPUs: -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc")
    endif()
  endif()
endif()

if(KT2N_HAVE_CUDA)
  # Must be set before enable_language(CUDA), otherwise CMake probes a default arch.
  set(CMAKE_CUDA_ARCHITECTURES "${KT2N_CUDA_ARCHS}")
  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  # nvcc 12.x warns that building for archs below 7.5 is deprecated. The warning is about future toolkits,
  # not this build, and KT2N_WERROR's --Werror=all-warnings would make it fatal, so silence just that one
  # (it does not change code generation).
  set(_kt2n_pre75 OFF)
  foreach(_kt2n_arch IN LISTS KT2N_CUDA_ARCHS)
    string(REGEX MATCH "^[0-9]+" _kt2n_arch_num "${_kt2n_arch}")
    if(_kt2n_arch_num AND _kt2n_arch_num LESS 75)
      set(_kt2n_pre75 ON)
    endif()
  endforeach()
  if(_kt2n_pre75)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
    add_link_options($<$<LINK_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
  endif()

  # The engine's create-time check needs the arch list (kt2n::cuda::arch_check in include/kt2n/cuda_caps.h), as a
  # define on the host-only engine.cc so no kernel object is linked in for it.
  string(REPLACE ";" "," _kt2n_arch_csv "${KT2N_CUDA_ARCHS}")
  set_property(SOURCE src/engine/engine.cc APPEND PROPERTY COMPILE_DEFINITIONS "KT2N_CUDA_ARCH_LIST=\"${_kt2n_arch_csv}\"")

  foreach(_var IN ITEMS CMAKE_CUDA_FLAGS CMAKE_CUDA_FLAGS_RELEASE CMAKE_CUDA_FLAGS_RELWITHDEBINFO
                        CMAKE_CUDA_FLAGS_DEBUG CMAKE_CUDA_FLAGS_MINSIZEREL)
    if("${${_var}}" MATCHES "(use_fast_math|-ftz=true|--ftz=true|-ftz true|prec-div=false|prec-sqrt=false)")
      message(FATAL_ERROR "${_var} contains '${CMAKE_MATCH_1}'; kt2n needs exact numerics")
    endif()
  endforeach()
endif()

if(KT2N_HAVE_CUDA AND KT2N_PORTABLE)
  add_compile_definitions(KT2N_PORTABLE=1)
  message(STATUS "kt2n: KT2N_PORTABLE=ON: software forms of the SM-specific instructions (validation build)")
endif()

if(KT2N_HIP)
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "${KT2N_HIP_ARCHS}")
  endif()
  if(NOT DEFINED CMAKE_HIP_PLATFORM)
    set(CMAKE_HIP_PLATFORM amd)
  endif()
  set(CMAKE_HIP_STANDARD 20)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  set(CMAKE_HIP_EXTENSIONS OFF)
  enable_language(HIP)
  find_package(hip REQUIRED CONFIG)
  set(KT2N_HAVE_HIP ON)
  set(KT2N_HAVE_CUDA ON)  # the GPU engine targets (their .cu sources compile as HIP)
  set(KT2N_GPU_RT hip::host)
  set(KT2N_GPU_LANG HIP)
  # "fast" is refused: it ignores the pragma that keeps __fmul_rn / __fadd_rn / __fsub_rn unfused.
  if(NOT KT2N_HIP_FP_CONTRACT MATCHES "^(fast-honor-pragmas|on|off)$")
    message(FATAL_ERROR "KT2N_HIP_FP_CONTRACT must be fast-honor-pragmas, on or off (got '${KT2N_HIP_FP_CONTRACT}')")
  endif()
  # Every target sees the HIP mapping: the <cuda_runtime.h>-style shims resolve to include/kt2n/gpu_runtime.h.
  # CUstream_st / CUevent_st: the CUDA handle structs that host headers forward-declare (cudaStream_t is a pointer
  # to one) so they compile without the CUDA headers; on HIP the handles point to ihipStream_t / ihipEvent_t.
  add_compile_definitions(KT2N_HIP=1 KT2N_PORTABLE=1 CUstream_st=ihipStream_t CUevent_st=ihipEvent_t)
  include_directories(BEFORE "${PROJECT_SOURCE_DIR}/include/kt2n/hip_shim")
  # hipError_t is [[nodiscard]]; the engine ignores some API results on purpose, as its CUDA build does.
  # (HIP sources get it from kt2n_set_warnings, after -Wall, which clang lets re-enable it.)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-result>)
  string(REPLACE ";" "," _kt2n_arch_csv "${KT2N_HIP_ARCHS}")
  set_property(SOURCE src/engine/engine.cc APPEND PROPERTY COMPILE_DEFINITIONS "KT2N_CUDA_ARCH_LIST=\"${_kt2n_arch_csv}\"")
  foreach(_var IN ITEMS CMAKE_HIP_FLAGS CMAKE_HIP_FLAGS_RELEASE CMAKE_HIP_FLAGS_RELWITHDEBINFO CMAKE_HIP_FLAGS_DEBUG
                        CMAKE_HIP_FLAGS_MINSIZEREL)
    if("${${_var}}" MATCHES
       "(-ffast-math|-Ofast|-fgpu-flush-denormals-to-zero|-funsafe-math-optimizations|-fgpu-approx-transcendentals|-ffp-contract=fast( |$))")
      message(FATAL_ERROR "${_var} contains '${CMAKE_MATCH_1}'; kt2n needs exact numerics")
    endif()
  endforeach()
endif()

# kt2n_cuda_target(<target>): common CUDA settings for a target with .cu sources.
function(kt2n_cuda_target target)
  if(KT2N_HAVE_HIP)
    # The target's .cu sources compile as HIP.
    get_target_property(_srcs ${target} SOURCES)
    get_target_property(_dir ${target} SOURCE_DIR)
    set(_cu "")
    foreach(_s IN LISTS _srcs)
      if(_s MATCHES "\\.cu$" AND NOT _s MATCHES "^\\$<")
        if(NOT IS_ABSOLUTE "${_s}")
          set(_s "${_dir}/${_s}")
        endif()
        list(APPEND _cu "${_s}")
      endif()
    endforeach()
    if(_cu)
      set_source_files_properties(${_cu} TARGET_DIRECTORY ${target} PROPERTIES LANGUAGE HIP)
    endif()
    set_target_properties(${target} PROPERTIES
      HIP_ARCHITECTURES "${KT2N_HIP_ARCHS}"
      HIP_STANDARD 20
      HIP_STANDARD_REQUIRED ON
      HIP_EXTENSIONS OFF)
    # Exact numerics: device FMA contraction as documented above; host code never contracts (as in the CUDA
    # build, whose host compiler runs in ISO C++ mode); IEEE fp32 division and sqrt; fp32 denormals kept.
    target_compile_options(${target} PRIVATE
      "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-Xarch_device -ffp-contract=${KT2N_HIP_FP_CONTRACT}>"
      "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-Xarch_host -ffp-contract=off>"
      "$<$<COMPILE_LANGUAGE:HIP>:-fhip-fp32-correctly-rounded-divide-sqrt;-fno-gpu-flush-denormals-to-zero;-fno-fast-math>")
    return()
  endif()
  set_target_properties(${target} PROPERTIES
    CUDA_ARCHITECTURES "${KT2N_CUDA_ARCHS}"
    CUDA_RUNTIME_LIBRARY Static
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON)
  # -lineinfo for profilers; explicit exact-math flags so nothing can sneak in.
  if(KT2N_CUDA_LINEINFO)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --ftz=false --prec-div=true --prec-sqrt=true>)
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false --prec-div=true --prec-sqrt=true>)
  endif()
  if(KT2N_CUDA_COMPRESS_MODE)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--compress-mode=${KT2N_CUDA_COMPRESS_MODE}>)
  endif()
endfunction()

function(kt2n_print_summary)
  message(STATUS "kt2n ${PROJECT_VERSION}: build type ${CMAKE_BUILD_TYPE}, "
    "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
  if(KT2N_HAVE_HIP)
    message(STATUS "kt2n: HIP ON (${CMAKE_HIP_COMPILER_ID} ${CMAKE_HIP_COMPILER_VERSION}, archs ${KT2N_HIP_ARCHS}, "
      "-ffp-contract=${KT2N_HIP_FP_CONTRACT})")
  elseif(KT2N_HAVE_CUDA)
    message(STATUS "kt2n: CUDA ON (nvcc ${KT2N_NVCC_VERSION}, archs ${KT2N_CUDA_ARCHS}, static cudart)")
  else()
    message(STATUS "kt2n: CUDA OFF (KT2N_CUDA=${KT2N_CUDA})")
  endif()
  message(STATUS "kt2n: tests=${KT2N_BUILD_TESTS} werror=${KT2N_WERROR}")
endfunction()
