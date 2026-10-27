# Native C++20 / CUDA engine for Kitten TTS 2.
cmake_minimum_required(VERSION 3.28)

project(kt2n
  VERSION 0.2.0
  DESCRIPTION "Kitten TTS 2 native inference engine"
  LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(Kt2nOptions)
include(Kt2nWarnings)
include(Kt2nCuda)
include(Kt2nDeps)
include(Kt2nTextPlan)  # text planning: segmentation, budgets, tokenizer (cmake/Kt2nTextPlan.cmake)

# ---------------------------------------------------------------------------
# kt2n_core (CPU)
# ---------------------------------------------------------------------------
add_library(kt2n_core STATIC
  src/core/engine_file.cc
  src/core/json.cc
  src/core/platform.cc
  src/core/ref_kernels.cc
  src/core/sha256.cc
  src/core/verify.cc)
add_library(kt2n::core ALIAS kt2n_core)
target_include_directories(kt2n_core
  PUBLIC "${PROJECT_SOURCE_DIR}/include"
  PRIVATE "${PROJECT_SOURCE_DIR}/src/core")
target_compile_features(kt2n_core PUBLIC cxx_std_20)
target_compile_definitions(kt2n_core PUBLIC KT2N_VERSION_STRING="${PROJECT_VERSION}")
find_package(Threads REQUIRED)
target_link_libraries(kt2n_core PUBLIC Threads::Threads)
if(WIN32)
  target_link_libraries(kt2n_core PUBLIC winmm)  # timeBeginPeriod (src/core/platform.cc)
endif()
kt2n_exact_numerics(kt2n_core)
kt2n_set_warnings(kt2n_core)

# ---------------------------------------------------------------------------
# kt2n_post (CPU): host-audio post-processing ports (include/kt2n/post/).
# ---------------------------------------------------------------------------
add_library(kt2n_post STATIC
  src/post/post.cc
  src/post/atempo.cc
  src/post/response.cc)
add_library(kt2n::post ALIAS kt2n_post)
target_include_directories(kt2n_post
  PUBLIC "${PROJECT_SOURCE_DIR}/include"
  PRIVATE "${PROJECT_SOURCE_DIR}/src/core")
target_compile_features(kt2n_post PUBLIC cxx_std_20)
target_link_libraries(kt2n_post PUBLIC kt2n_core)
kt2n_exact_numerics(kt2n_post)
kt2n_set_warnings(kt2n_post)

# ---------------------------------------------------------------------------
# kt2n_encode (CPU, optional): in-process mp3/aac/opus/flac encoders over the
# FFmpeg libraries the reference ffmpeg CLI uses (include/kt2n/post/encoder.h).
# KT2N_AV=AUTO builds it when pkg-config finds libavcodec/libavformat/libavutil/
# libswresample dev files; ON makes them required. Link the same FFmpeg version
# as the reference to reproduce its bytes (the encoders' output differs between
# FFmpeg versions).
# ---------------------------------------------------------------------------
set(KT2N_AV "AUTO" CACHE STRING "In-process encoders over FFmpeg libraries: AUTO|ON|OFF")
set(KT2N_HAVE_AV OFF)
if(NOT KT2N_AV STREQUAL "OFF")
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(KT2N_AVLIBS QUIET IMPORTED_TARGET libavcodec libavformat libavutil libswresample)
  endif()
  if(KT2N_AVLIBS_FOUND)
    set(KT2N_HAVE_AV ON)
    add_library(kt2n_encode STATIC src/post/encoder.cc)
    add_library(kt2n::encode ALIAS kt2n_encode)
    target_link_libraries(kt2n_encode PUBLIC kt2n_post PRIVATE PkgConfig::KT2N_AVLIBS)
    kt2n_set_warnings(kt2n_encode)
    message(STATUS "kt2n: in-process encoders ON (libavcodec ${KT2N_AVLIBS_libavcodec_VERSION})")
  elseif(KT2N_AV STREQUAL "ON")
    message(FATAL_ERROR "KT2N_AV=ON but the FFmpeg dev files (libavcodec, libavformat, libavutil, libswresample) were not found")
  else()
    message(STATUS "kt2n: in-process encoders OFF (FFmpeg dev files not found)")
  endif()
endif()

# ---------------------------------------------------------------------------
# kt2n_cuda (GPU)
# ---------------------------------------------------------------------------
if(KT2N_HAVE_CUDA)
  # The kernel libraries add src/cuda/kernels/*.cu with extern "C" launchers declared in
  # include/kt2n/kernels/*.h. Re-run cmake (or rely on CONFIGURE_DEPENDS) after
  # adding files.
  file(GLOB KT2N_CUDA_KERNEL_SOURCES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/cuda/kernels/*.cu")
  list(SORT KT2N_CUDA_KERNEL_SOURCES)
  # Native prefill (include/kt2n/prefill.h): model loader + run context over
  # the kernel libraries' launchers; host code plus one small kernel per file.
  file(GLOB KT2N_PREFILL_SOURCES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/prefill/*.cu")
  list(SORT KT2N_PREFILL_SOURCES)
  # Torch-free native LM (include/kt2n/lm.h): engine, stop rules, run-ahead ring.
  file(GLOB KT2N_LM_SOURCES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/lm/*.cu")
  list(SORT KT2N_LM_SOURCES)
  # AMD matrix-instruction (WMMA) kernels: HIP builds only (include/kt2n/hip/wmma_kernels.h).
  set(KT2N_HIP_WMMA_SOURCES "")
  if(KT2N_HAVE_HIP)
    file(GLOB KT2N_HIP_WMMA_SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/hip/*.cu")
    list(SORT KT2N_HIP_WMMA_SOURCES)
  endif()

  # Share compiled CUDA objects between the engine and C++ tests.
  add_library(kt2n_cuda_objs OBJECT src/cuda/smoke.cu ${KT2N_CUDA_KERNEL_SOURCES} ${KT2N_PREFILL_SOURCES}
    ${KT2N_LM_SOURCES} ${KT2N_HIP_WMMA_SOURCES})
  target_include_directories(kt2n_cuda_objs PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
    PRIVATE "${PROJECT_SOURCE_DIR}/src/cuda/kernels" "${PROJECT_SOURCE_DIR}/src/prefill"
    "${PROJECT_SOURCE_DIR}/src/lm")
  target_compile_definitions(kt2n_cuda_objs PUBLIC KT2N_HAVE_CUDA=1)
  target_link_libraries(kt2n_cuda_objs PUBLIC kt2n_core)
  set_target_properties(kt2n_cuda_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
  kt2n_cuda_target(kt2n_cuda_objs)
  kt2n_set_warnings(kt2n_cuda_objs)

  # CUDA-core decode projections for the GPUs without bf16 tensor-core instructions (src/cuda/legacy): built only
  # for the KT2N_CUDA_ARCHS entries below 80, so the SM 8.0+ code of every other file is unchanged.
  set(KT2N_LEGACY_CUDA_ARCHS "")
  if(KT2N_HAVE_HIP OR KT2N_PORTABLE)
    # Portable builds (HIP, or CUDA with KT2N_PORTABLE) take the SM 6.x / 7.5 host choices on every device
    # (DeviceCaps::legacy()), so the kernel must exist for every target.
    set(KT2N_LEGACY_CUDA_ARCHS "${KT2N_CUDA_ARCHS}")
  else()
    set(_kt2n_archs "${KT2N_CUDA_ARCHS}")
    if(_kt2n_archs STREQUAL "native" AND CMAKE_CUDA_ARCHITECTURES_NATIVE)
      set(_kt2n_archs "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
    elseif(_kt2n_archs MATCHES "^all")
      set(_kt2n_archs "61-real;75-real")  # the engine's SM 6.x / 7.5 targets (it refuses SM 6.0 / 6.2 / 7.0 / 7.2)
      if(KT2N_NVCC_VERSION VERSION_GREATER_EQUAL "13.0")
        set(_kt2n_archs "75-real")  # CUDA 13 builds nothing below 7.5 (cmake/Kt2nCuda.cmake)
      endif()
    endif()
    foreach(_kt2n_arch IN LISTS _kt2n_archs)
      string(REGEX MATCH "^[0-9]+" _kt2n_arch_num "${_kt2n_arch}")
      if(_kt2n_arch_num AND _kt2n_arch_num LESS 80)
        list(APPEND KT2N_LEGACY_CUDA_ARCHS "${_kt2n_arch}")
      endif()
    endforeach()
  endif()
  if(NOT KT2N_LEGACY_CUDA_ARCHS)
    message(STATUS "kt2n: no CUDA arch below 8.0 in KT2N_CUDA_ARCHS (${KT2N_CUDA_ARCHS}): CUDA-core decode kernel not built")
  endif()
  set(_kt2n_cuda_legacy_objs "")
  if(KT2N_LEGACY_CUDA_ARCHS)
    add_library(kt2n_cuda_legacy_objs OBJECT src/cuda/legacy/tern_dec_cc.cu)
    target_include_directories(kt2n_cuda_legacy_objs PRIVATE "${PROJECT_SOURCE_DIR}/include"
      "${PROJECT_SOURCE_DIR}/src/cuda/kernels")
    target_compile_definitions(kt2n_cuda_legacy_objs PRIVATE KT2N_HAVE_CUDA=1)
    set_target_properties(kt2n_cuda_legacy_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
    kt2n_cuda_target(kt2n_cuda_legacy_objs)
    set_target_properties(kt2n_cuda_legacy_objs PROPERTIES CUDA_ARCHITECTURES "${KT2N_LEGACY_CUDA_ARCHS}")
    kt2n_set_warnings(kt2n_cuda_legacy_objs)
    set_property(SOURCE src/cuda/kernels/tern_proj.cu APPEND PROPERTY COMPILE_DEFINITIONS KT2N_TERN_CC=1)
    set(_kt2n_cuda_legacy_objs $<TARGET_OBJECTS:kt2n_cuda_legacy_objs>)
  endif()

  add_library(kt2n_cuda STATIC $<TARGET_OBJECTS:kt2n_cuda_objs> ${_kt2n_cuda_legacy_objs})
  add_library(kt2n::cuda ALIAS kt2n_cuda)
  target_include_directories(kt2n_cuda PUBLIC "${PROJECT_SOURCE_DIR}/include")
  target_link_libraries(kt2n_cuda PUBLIC kt2n_core ${KT2N_GPU_RT})
  target_compile_definitions(kt2n_cuda PUBLIC KT2N_HAVE_CUDA=1)
  set_target_properties(kt2n_cuda PROPERTIES LINKER_LANGUAGE ${KT2N_GPU_LANG})
  kt2n_cuda_target(kt2n_cuda)

endif()

# Decoder estimator (src/render/estimator): static library + est_check tool.
if(KT2N_HAVE_CUDA)
  add_subdirectory(src/render/estimator)
endif()

# ---------------------------------------------------------------------------
# Voice files and the voice cloning build: src/voice/CMakeLists.txt
# ---------------------------------------------------------------------------
add_subdirectory(src/voice)

# ---------------------------------------------------------------------------
# kt2n_render (decoder runtime): render state, RNG ports, descriptors, render calls.
# ---------------------------------------------------------------------------
add_subdirectory(src/render)

# Native text normaliser (opt-in: KT2N_BUILD_TEXT, KT2N_TEXT_DEPS).
include(Kt2nText)

# Engine runtime and shared library (cmake/Kt2nEngine.cmake).
include(Kt2nEngine)

# ---------------------------------------------------------------------------
# kitten-tts-native command line (cli/). Its CPU core (argument parser, bundle
# checks, report) always builds so its tests run anywhere; the executable is
# added only when the engine library (TARGET kt2n_engine) exists, and installs
# to bin/ with RPATH $ORIGIN/../lib.
# ---------------------------------------------------------------------------
add_subdirectory(cli)

# ---------------------------------------------------------------------------
# C API examples (examples/CMakeLists.txt).
# ---------------------------------------------------------------------------
option(KT2N_BUILD_EXAMPLES "Build the C API examples" ${PROJECT_IS_TOP_LEVEL})
if(KT2N_BUILD_EXAMPLES)
  if(EXISTS "${PROJECT_SOURCE_DIR}/examples/CMakeLists.txt")
    add_subdirectory(examples)
  else()
    message(STATUS "kt2n: examples/CMakeLists.txt not present; no examples built")
  endif()
endif()

# ---------------------------------------------------------------------------
# Tests (ctest labels: cpu, gpu)
# ---------------------------------------------------------------------------
if(KT2N_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
  add_subdirectory(tests/render)
  add_subdirectory(tests/cli)
  add_subdirectory(tests/consumer)
  if(KT2N_HAVE_CUDA)
    add_executable(test_gpu_portability tests/gpu/portability.cu)
    target_include_directories(test_gpu_portability PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(test_gpu_portability PRIVATE ${KT2N_GPU_RT})
    kt2n_cuda_target(test_gpu_portability)
    add_test(NAME gpu.portability COMMAND test_gpu_portability)
    set_tests_properties(gpu.portability PROPERTIES LABELS gpu TIMEOUT 120)
    # Software versions of the SM 8.0 / 7.5 instructions used on SM 7.5 and 6.1 (include/kt2n/cuda_compat.cuh),
    # checked against the hardware instructions where the GPU has them.
    add_executable(test_gpu_compat_emu tests/gpu/compat_emu.cu)
    target_include_directories(test_gpu_compat_emu PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(test_gpu_compat_emu PRIVATE ${KT2N_GPU_RT})
    kt2n_cuda_target(test_gpu_compat_emu)
    add_test(NAME gpu.compat_emu COMMAND test_gpu_compat_emu)
    set_tests_properties(gpu.compat_emu PROPERTIES LABELS gpu TIMEOUT 300)
    # The vocoder's Philox noise (cuRAND on CUDA, rocRAND on HIP) against a CPU reference of cuRAND.
    add_executable(test_gpu_philox tests/gpu/philox.cu)
    target_include_directories(test_gpu_philox PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(test_gpu_philox PRIVATE ${KT2N_GPU_RT})
    kt2n_cuda_target(test_gpu_philox)
    add_test(NAME gpu.philox COMMAND test_gpu_philox)
    set_tests_properties(gpu.philox PROPERTIES LABELS gpu TIMEOUT 120)
    if(KT2N_HAVE_HIP)
      add_test(NAME philox.rocrand_host COMMAND test_gpu_philox --host)
      set_tests_properties(philox.rocrand_host PROPERTIES LABELS cpu TIMEOUT 60)
    endif()
    # CUDA-core decode projections (SM 6.x / 7.5) bitwise against the software-MMA decode kernel; skipped on SM 8.0+.
    if(TARGET kt2n_cuda_legacy_objs)
      add_executable(test_gpu_tern_dec_cc tests/gpu/tern_dec_cc.cu)
      target_include_directories(test_gpu_tern_dec_cc PRIVATE "${PROJECT_SOURCE_DIR}/include")
      target_link_libraries(test_gpu_tern_dec_cc PRIVATE kt2n_cuda)
      kt2n_cuda_target(test_gpu_tern_dec_cc)
      add_test(NAME gpu.tern_dec_cc COMMAND test_gpu_tern_dec_cc)
      set_tests_properties(gpu.tern_dec_cc PROPERTIES LABELS gpu TIMEOUT 900)
      # Default routing of kt2n_tern_proj (CUDA: on for SM 6.x / 7.5 unless KT2N_TERN_DECODE=mma; HIP: only with =cc).
      add_test(NAME gpu.tern_dec_cc_dispatch COMMAND test_gpu_tern_dec_cc --dispatch)
      add_test(NAME gpu.tern_dec_cc_dispatch_cc COMMAND test_gpu_tern_dec_cc --dispatch)
      add_test(NAME gpu.tern_dec_cc_dispatch_mma COMMAND test_gpu_tern_dec_cc --dispatch)
      set_tests_properties(gpu.tern_dec_cc_dispatch gpu.tern_dec_cc_dispatch_cc gpu.tern_dec_cc_dispatch_mma
                           PROPERTIES LABELS gpu TIMEOUT 300)
      set_tests_properties(gpu.tern_dec_cc_dispatch PROPERTIES ENVIRONMENT_MODIFICATION KT2N_TERN_DECODE=unset:)
      set_tests_properties(gpu.tern_dec_cc_dispatch_cc PROPERTIES ENVIRONMENT KT2N_TERN_DECODE=cc)
      set_tests_properties(gpu.tern_dec_cc_dispatch_mma PROPERTIES ENVIRONMENT KT2N_TERN_DECODE=mma)
    endif()
  endif()
  if(KT2N_HAVE_HIP)
    # AMD WMMA kernels (src/hip/): operand layouts, rounding vs a CPU reference, ternary projection vs the software path.
    add_executable(test_gpu_hip_wmma tests/gpu/hip_wmma.cu)
    target_include_directories(test_gpu_hip_wmma PRIVATE "${PROJECT_SOURCE_DIR}/include"
      "${PROJECT_SOURCE_DIR}/src/cuda/kernels")
    target_link_libraries(test_gpu_hip_wmma PRIVATE kt2n_cuda ${KT2N_GPU_RT})
    kt2n_cuda_target(test_gpu_hip_wmma)
    add_test(NAME gpu.hip_wmma COMMAND test_gpu_hip_wmma)
    set_tests_properties(gpu.hip_wmma PROPERTIES LABELS gpu TIMEOUT 300 SKIP_RETURN_CODE 77)
    # Flow estimator GEMMs on WMMA (src/render/estimator/est_wmma.cuh) vs the portable est_gemm_kernel.
    add_executable(test_gpu_hip_est_gemm tests/gpu/hip_est_gemm.cu)
    target_include_directories(test_gpu_hip_est_gemm PRIVATE "${PROJECT_SOURCE_DIR}/include"
      "${PROJECT_SOURCE_DIR}/src/render/estimator")
    target_link_libraries(test_gpu_hip_est_gemm PRIVATE kt2n_core ${KT2N_GPU_RT})
    kt2n_cuda_target(test_gpu_hip_est_gemm)
    add_test(NAME gpu.hip_est_gemm COMMAND test_gpu_hip_est_gemm)
    set_tests_properties(gpu.hip_est_gemm PROPERTIES LABELS gpu TIMEOUT 300 SKIP_RETURN_CODE 77)
    # Flow estimator convolutions on WMMA (est_wmma.cuh) vs the portable est_conv_kernel + epilogue.
    add_executable(test_gpu_hip_est_conv tests/gpu/hip_est_conv.cu)
    target_include_directories(test_gpu_hip_est_conv PRIVATE "${PROJECT_SOURCE_DIR}/include"
      "${PROJECT_SOURCE_DIR}/src/render/estimator")
    target_link_libraries(test_gpu_hip_est_conv PRIVATE kt2n_core ${KT2N_GPU_RT})
    kt2n_cuda_target(test_gpu_hip_est_conv)
    add_test(NAME gpu.hip_est_conv COMMAND test_gpu_hip_est_conv)
    set_tests_properties(gpu.hip_est_conv PROPERTIES LABELS gpu TIMEOUT 300 SKIP_RETURN_CODE 77)
    # Decode attention on WMMA vs an fp64 reference, and (with the real model) decode == prefill logits under the
    # engine's own per-bucket step options (tests/gpu/hip_wmma_lm.cu).
    add_executable(test_gpu_hip_wmma_lm tests/gpu/hip_wmma_lm.cu)
    target_include_directories(test_gpu_hip_wmma_lm PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(test_gpu_hip_wmma_lm PRIVATE kt2n_cuda ${KT2N_GPU_RT})
    kt2n_cuda_target(test_gpu_hip_wmma_lm)
    add_test(NAME gpu.hip_wmma_lm COMMAND test_gpu_hip_wmma_lm)
    set_tests_properties(gpu.hip_wmma_lm PROPERTIES LABELS gpu TIMEOUT 900 SKIP_RETURN_CODE 77)
  endif()
endif()

kt2n_print_summary()
