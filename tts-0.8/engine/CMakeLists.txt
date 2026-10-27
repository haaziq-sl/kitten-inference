cmake_minimum_required(VERSION 3.18)
if(APPLE)
    project(cnn_inference CXX OBJCXX)
else()
    project(cnn_inference CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_definitions(__restrict__=__restrict)
endif()

# ── CUDA backend (optional) ────────────────────────────────────
option(ENABLE_CUDA "Build the CUDA backend when a CUDA toolkit is available" ON)
set(CUDA_TOOLKIT_ROOT_DIR "" CACHE PATH "CUDA toolkit root")
if(ENABLE_CUDA)
    find_program(NVCC_BIN nvcc
        HINTS
            "${CUDA_TOOLKIT_ROOT_DIR}/bin"
            "$ENV{CUDA_PATH}/bin"
            "/usr/local/cuda/bin"
            "/usr/local/cuda-13.1/bin"
            "$ENV{CONDA_PREFIX}/bin"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin"
    )
endif()
if(ENABLE_CUDA AND NVCC_BIN)
    get_filename_component(_CUDA_BIN_DIR "${NVCC_BIN}" DIRECTORY)
    get_filename_component(_CUDA_ROOT "${_CUDA_BIN_DIR}" DIRECTORY)
    if(NOT CUDA_TOOLKIT_ROOT_DIR)
        set(CUDA_TOOLKIT_ROOT_DIR "${_CUDA_ROOT}" CACHE PATH "CUDA toolkit root" FORCE)
    endif()
    set(CMAKE_CUDA_COMPILER "${NVCC_BIN}" CACHE FILEPATH "CUDA compiler" FORCE)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)  # use libcudart.so, not libcudart_static.a
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "CUDA architectures")
    set(HAVE_CUDA TRUE)
    if(WIN32 AND MSVC)
        string(REGEX MATCH "^[0-9]+" CUDA_DLL_MAJOR "${CMAKE_CUDA_COMPILER_VERSION}")
        set(CUBLAS_DLL_NAME "cublas64_${CUDA_DLL_MAJOR}.dll")
    endif()
    message(STATUS "CUDA enabled, compiler: ${CMAKE_CUDA_COMPILER}")
else()
    set(HAVE_CUDA FALSE)
    if(ENABLE_CUDA)
        message(STATUS "CUDA toolkit not found — building CPU-only")
    else()
        message(STATUS "CUDA disabled — building CPU-only")
    endif()
endif()

if(HAVE_CUDA)
    find_package(CUDAToolkit QUIET)
    # CUDA headers may be in targets/x86_64-linux/include or just include
    if(CUDAToolkit_INCLUDE_DIRS)
        set(CUDA_INCLUDE_DIR "${CUDAToolkit_INCLUDE_DIRS}")
    elseif(EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include/cuda_runtime.h")
        set(CUDA_INCLUDE_DIR "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include")
    else()
        set(CUDA_INCLUDE_DIR "${CUDA_TOOLKIT_ROOT_DIR}/include")
    endif()
    set(CUDA_RUNTIME_LIB CUDA::cudart)
    set(CUDA_BLAS_LIB CUDA::cublas)
    if(NOT TARGET CUDA::cudart OR NOT TARGET CUDA::cublas)
        unset(CUDA_RUNTIME_LIB)
        unset(CUDA_BLAS_LIB)
    endif()
    set(CUDA_LIB_DIRS
        "${CUDA_TOOLKIT_ROOT_DIR}/lib"
        "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
        "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64"
        "$ENV{CUDA_PATH}/lib/x64"
        "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib"
        "/usr/lib/x86_64-linux-gnu")
    if(NOT CUDA_BLAS_LIB)
        find_library(CUBLAS_LIB cublas
            PATHS ${CUDA_LIB_DIRS}
            NO_DEFAULT_PATH)
        set(CUDA_BLAS_LIB "${CUBLAS_LIB}")
    endif()
    if(NOT CUDA_RUNTIME_LIB)
        find_library(CUDART_LIB cudart
            PATHS ${CUDA_LIB_DIRS}
            NO_DEFAULT_PATH)
        set(CUDA_RUNTIME_LIB "${CUDART_LIB}")
    endif()
    if(NOT CUDA_BLAS_LIB OR NOT CUDA_RUNTIME_LIB)
        message(WARNING "cuBLAS or cuDart not found — disabling CUDA backend")
        set(HAVE_CUDA FALSE)
    else()
        # conda-forge nvcc sysroot has linker scripts pointing to /lib64/libm.so.6 (CentOS).
        # Put Ubuntu system lib paths first so the real libm.so is found before the sysroot one.
        # Skip this when the host compiler is itself a conda-forge cross-compiler — its sysroot
        # is self-consistent, and injecting host paths shadows it and breaks glibc linkage.
        if(NOT WIN32 AND NOT CMAKE_CXX_COMPILER MATCHES "conda-linux-gnu")
            set(CMAKE_EXE_LINKER_FLAGS
                "-L/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu ${CMAKE_EXE_LINKER_FLAGS}"
                CACHE STRING "" FORCE)
        endif()
        message(STATUS "cuBLAS: ${CUDA_BLAS_LIB}")
        message(STATUS "cuDart: ${CUDA_RUNTIME_LIB}")
        message(STATUS "CUDA includes: ${CUDA_INCLUDE_DIR}")
        # ── cuDNN (optional, gives 10-30x speedup for FP32 conv via Winograd/implicit-GEMM)
        set(CUDNN_ROOT "" CACHE PATH "cuDNN installation root")
        find_path(CUDNN_INCLUDE cudnn.h
            HINTS
                "${CUDNN_ROOT}/include"
                "$ENV{CUDNN_ROOT}/include"
                "$ENV{CUDA_PATH}/include"
                "$ENV{CONDA_PREFIX}/include")
        find_library(CUDNN_LIB cudnn
            HINTS
                "${CUDNN_ROOT}/lib"
                "${CUDNN_ROOT}/lib/x64"
                "$ENV{CUDNN_ROOT}/lib"
                "$ENV{CUDNN_ROOT}/lib/x64"
                "$ENV{CUDA_PATH}/lib/x64"
                "$ENV{CONDA_PREFIX}/lib")
        if(CUDNN_INCLUDE AND CUDNN_LIB)
            message(STATUS "cuDNN found: ${CUDNN_LIB}")
            set(HAVE_CUDNN TRUE)
        else()
            message(STATUS "cuDNN not found at ${CUDNN_ROOT} — using im2col+cuBLAS for FP32 conv")
            set(HAVE_CUDNN FALSE)
        endif()
        set(CUDA_BACKEND_SOURCES
            src/backend/cuda/cuda_backend.cu
            src/backend/cuda/ops/gemm.cu
            src/backend/cuda/ops/conv.cu
            src/backend/cuda/ops/attention.cu
            src/backend/cuda/ops/pool.cu
            src/backend/cuda/ops/activations.cu
            src/backend/cuda/ops/norm.cu
            src/backend/cuda/ops/tensor.cu
        )
    endif()
endif()

# ── Metal/MPS backend (macOS only) ────────────────────────────
if(APPLE)
    # Check if Metal compiler is available
    execute_process(
        COMMAND xcrun -sdk macosx --find metal
        OUTPUT_VARIABLE METAL_COMPILER
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE METAL_COMPILER_RESULT
    )
    if(METAL_COMPILER_RESULT EQUAL 0 AND EXISTS "${METAL_COMPILER}")
        message(STATUS "macOS detected — enabling Metal/MPS backend (compiler: ${METAL_COMPILER})")
        set(HAVE_METAL TRUE)
    else()
        message(STATUS "macOS detected but Metal compiler not found — disabling Metal backend")
        set(HAVE_METAL FALSE)
    endif()
else()
    set(HAVE_METAL FALSE)
endif()

if(HAVE_METAL)
    set(METAL_BACKEND_SOURCES
        src/backend/metal/metal_backend.mm
        src/backend/metal/ops/gemm.mm
        src/backend/metal/ops/conv.mm
        src/backend/metal/ops/pool.mm
        src/backend/metal/ops/activations.mm
        src/backend/metal/ops/norm.mm
        src/backend/metal/ops/attention.mm
        src/backend/metal/ops/tensor.mm
        src/backend/metal/ops/sequence.mm
    )
    # Enable ARC for Objective-C++ sources
    set_source_files_properties(${METAL_BACKEND_SOURCES} PROPERTIES
        COMPILE_FLAGS "-fobjc-arc")

    # Compile Metal shaders to .metallib at build time
    set(METAL_SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal/shaders/kernels.metal")
    set(METAL_AIR        "${CMAKE_CURRENT_BINARY_DIR}/metal_kernels.air")
    set(METAL_LIB        "${CMAKE_CURRENT_BINARY_DIR}/metal_kernels.metallib")
    add_custom_command(
        OUTPUT  "${METAL_LIB}"
        COMMAND xcrun -sdk macosx metal -c "${METAL_SHADER_SRC}" -o "${METAL_AIR}"
        COMMAND xcrun -sdk macosx metallib "${METAL_AIR}" -o "${METAL_LIB}"
        DEPENDS "${METAL_SHADER_SRC}"
        COMMENT "Compiling Metal shaders → metal_kernels.metallib"
        VERBATIM
    )
    add_custom_target(metal_shaders ALL DEPENDS "${METAL_LIB}")
endif()

# ── Optional C++ examples, benchmarks and tests ───────────────
# Off by default and for wheels. Their sources (examples/, bench/, tests/, tools/) are not
# part of this folder; the option exists for builds from the full engine tree.
option(KITTEN_BUILD_EXTRAS "Build the C++ examples, benchmarks and tests" OFF)
if(KITTEN_BUILD_EXTRAS)
    foreach(_dir examples bench tests)
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_dir}")
            message(FATAL_ERROR "KITTEN_BUILD_EXTRAS=ON needs ${CMAKE_CURRENT_SOURCE_DIR}/${_dir}, which is not part of this folder")
        endif()
    endforeach()
endif()

# ── Debug / diagnostic options ────────────────────────────────
# MODULE_PROBE: builds probe_kitten_15m diagnostic tool only (not applied to bench/run_tests).
option(MODULE_PROBE "Enable module-level tensor probe tool (probe_kitten_15m)" OFF)
if(MODULE_PROBE)
    message(STATUS "MODULE_PROBE enabled — building probe_kitten_15m (GRAPH_DEBUG+MODULE_PROBE)")
endif()

# ── Compiler flags ────────────────────────────────────────────
option(ENABLE_ARM_DOTPROD "Enable ARMv8.2-A dot-product instructions for int8 kernels" ON)
option(ENABLE_NATIVE_ARCH "Tune CPU kernels for the build host with -march=native" ON)
option(ENABLE_X86_AVX2 "Enable AVX2/FMA tuning for non-native x86_64 builds" ON)
option(ENABLE_OPENMP "Enable OpenMP for CPU kernels when available" ON)
set(ARCH_FLAGS)
set(TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    list(GET CMAKE_OSX_ARCHITECTURES 0 TARGET_PROCESSOR)
endif()
string(TOLOWER "${TARGET_PROCESSOR}" TARGET_PROCESSOR_LC)
message(STATUS "Target processor for architecture flags: ${TARGET_PROCESSOR}")
if(MSVC)
    if(TARGET_PROCESSOR_LC MATCHES "^(amd64|x86_64)$")
        if(ENABLE_X86_AVX2)
            message(STATUS "MSVC x86_64 detected — enabling AVX2 tuning")
            list(APPEND ARCH_FLAGS /O2 /arch:AVX2 /fp:fast)
        else()
            message(STATUS "MSVC x86_64 detected — enabling baseline x86_64 tuning")
            list(APPEND ARCH_FLAGS /O2)
        endif()
    elseif(TARGET_PROCESSOR_LC MATCHES "^(arm64|aarch64)$")
        message(STATUS "MSVC ARM64 detected — using optimized ARM64 codegen")
        list(APPEND ARCH_FLAGS /O2)
    else()
        message(STATUS "MSVC detected — using optimized generic codegen")
        list(APPEND ARCH_FLAGS /O2)
    endif()
else()
    if(TARGET_PROCESSOR_LC MATCHES "aarch64|arm64")
        message(STATUS "ARM64 detected — enabling NEON tuning")
        # Note: -fomit-frame-pointer is omitted on arm64 macOS because it can corrupt
        # compact unwind tables and break C++ exception handling (libunwind decodeFDE crash).
        if(ANDROID OR DEFINED ENV{TERMUX_VERSION})
            # Termux/Android Clang does not reliably expand -march=native to include
            # dotprod, but the int8 ARM kernels use vdotq_* intrinsics when building
            # the NEON path. S20 FE-class ARMv8.2 cores support this extension.
            if(ENABLE_ARM_DOTPROD)
                list(APPEND ARCH_FLAGS -march=armv8.2-a+dotprod -O3 -ffast-math -fno-finite-math-only -funroll-loops -ftree-vectorize)
                message(STATUS "ARM dotprod enabled explicitly for Android/Termux")
            else()
                list(APPEND ARCH_FLAGS -march=armv8-a -O3 -ffast-math -fno-finite-math-only -funroll-loops -ftree-vectorize)
                message(STATUS "ARM dotprod disabled")
            endif()
        elseif(APPLE AND ENABLE_ARM_DOTPROD)
            list(APPEND ARCH_FLAGS -mcpu=apple-m1 -O3 -ffast-math -fno-finite-math-only -funroll-loops -ftree-vectorize)
            message(STATUS "ARM dotprod enabled explicitly for Apple Silicon")
        elseif(ENABLE_ARM_DOTPROD)
            list(APPEND ARCH_FLAGS -march=armv8.2-a+dotprod -O3 -ffast-math -fno-finite-math-only -funroll-loops -ftree-vectorize)
            message(STATUS "ARM dotprod enabled explicitly")
        else()
            if(ENABLE_NATIVE_ARCH)
                list(APPEND ARCH_FLAGS -march=native -O3 -ffast-math -fno-finite-math-only -funroll-loops -ftree-vectorize)
            else()
                list(APPEND ARCH_FLAGS -O3 -ffast-math -fno-finite-math-only -funroll-loops -ftree-vectorize)
            endif()
        endif()
    else()
        if(ENABLE_NATIVE_ARCH)
            message(STATUS "x86_64 detected — enabling native AVX tuning")
            list(APPEND ARCH_FLAGS -march=native -O3 -ffast-math -fno-finite-math-only)
        elseif(ENABLE_X86_AVX2)
            message(STATUS "x86_64 detected — enabling portable AVX2/FMA tuning")
            list(APPEND ARCH_FLAGS -mavx2 -mfma -O3 -ffast-math -fno-finite-math-only)
        else()
            message(STATUS "x86_64 detected — enabling baseline x86_64 tuning")
            list(APPEND ARCH_FLAGS -O3 -ffast-math -fno-finite-math-only)
        endif()
    endif()
endif()

foreach(flag IN LISTS ARCH_FLAGS)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${flag}>)
endforeach()

# The instruction sets those flags let the compiler use anywhere in the engine (not only in its vector kernels).
# The Python package checks the CPU for them before it loads the engine (kitten_inference/tts08/cpp_model.py), so an
# older CPU gets an error message instead of "Illegal instruction". "native": tuned for the build machine (not
# checked). Apple Silicon needs no check: every Apple Silicon Mac runs -mcpu=apple-m1 code.
set(KITTEN_CPU_REQUIRES "")
if(MSVC)
    if(TARGET_PROCESSOR_LC MATCHES "^(amd64|x86_64)$" AND ENABLE_X86_AVX2)
        set(KITTEN_CPU_REQUIRES "avx2 fma")
    endif()
elseif(TARGET_PROCESSOR_LC MATCHES "aarch64|arm64")
    if(APPLE AND NOT ANDROID AND NOT DEFINED ENV{TERMUX_VERSION})
        set(KITTEN_CPU_REQUIRES "")
    elseif(ENABLE_ARM_DOTPROD)
        set(KITTEN_CPU_REQUIRES "asimddp")
    elseif(ENABLE_NATIVE_ARCH AND NOT ANDROID AND NOT DEFINED ENV{TERMUX_VERSION})
        set(KITTEN_CPU_REQUIRES "native")
    endif()
elseif(ENABLE_NATIVE_ARCH)
    set(KITTEN_CPU_REQUIRES "native")
elseif(ENABLE_X86_AVX2)
    set(KITTEN_CPU_REQUIRES "avx2 fma")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cpu_requirements.txt" "${KITTEN_CPU_REQUIRES}\n")
message(STATUS "CPU features the engine needs: ${KITTEN_CPU_REQUIRES}")

# ── Source lists ───────────────────────────────────────────────
set(OPS_SOURCES
    src/backend/cpu/ops/profile.cpp
    src/backend/cpu/ops/gemm.cpp
    src/backend/cpu/ops/conv.cpp
    src/backend/cpu/ops/pool.cpp
    src/backend/cpu/ops/add.cpp
    src/backend/cpu/ops/fp32_gemm.cpp
    src/backend/cpu/ops/fp32_conv.cpp
    src/backend/cpu/ops/fp32_conv1d.cpp
    src/backend/cpu/ops/fp32_add.cpp
    src/backend/cpu/ops/fp32_pool.cpp
    src/backend/cpu/ops/activations.cpp
    src/backend/cpu/ops/norm.cpp
    src/backend/cpu/ops/attention.cpp
    src/backend/cpu/ops/signal.cpp
    src/backend/cpu/ops/sequence.cpp
    src/backend/cpu/ops/tensor.cpp
    src/backend/backend.cpp
    src/backend/cpu/cpu_backend.cpp
)

set(LAYER_SOURCES
    src/layers/conv_int8.cpp
    src/layers/conv_fp32.cpp
    src/layers/gemm.cpp
    src/layers/add.cpp
    src/layers/pool.cpp
    src/layers/upsample.cpp
    src/layers/activation.cpp
    src/layers/conv1d.cpp
    src/layers/lstm.cpp
    src/layers/norm.cpp
    src/layers/attention.cpp
    src/layers/embedding.cpp
    src/layers/signal.cpp
    src/layers/tensor.cpp
    src/layers/factory.cpp
)

# ── Engine OBJECT library — every .cpp compiled exactly once ──
#
# OBJECT libraries (unlike STATIC) include all .o files unconditionally at
# link time, so self-registering static initialisers in cpu_backend.cpp and
# cuda_backend.cu are guaranteed to run.  All compilation flags / include
# paths marked PUBLIC propagate to every executable that links this target.
# When MODULE_PROBE=ON, model.cpp is excluded from the engine and compiled
# separately per target (with probe flags for probe tools, without for others).
set(ALL_TARGETS)
if(KITTEN_BUILD_EXTRAS)
if(MODULE_PROBE)
    set(_engine_model_cpp "")
else()
    set(_engine_model_cpp src/engine/model.cpp src/engine/arch_loader.cpp src/engine/weights_loader.cpp src/engine/tensor_pool.cpp src/engine/op_type.cpp)
endif()

add_library(engine OBJECT
    ${OPS_SOURCES}
    ${_engine_model_cpp}
    ${LAYER_SOURCES}
    src/models/image_classifier.cpp
    src/models/kitten.cpp
    src/models/language_model.cpp
)
target_include_directories(engine PUBLIC src)

if(HAVE_CUDA)
    # CUDA sources are kept in a separate OBJECT library (different language).
    # cuda_backend.cu contains another self-registering static initialiser, so
    # OBJECT (not STATIC) is required here for the same reason as above.
    add_library(engine_cuda OBJECT ${CUDA_BACKEND_SOURCES})
    target_include_directories(engine_cuda PRIVATE src "${CUDA_INCLUDE_DIR}")
    target_compile_definitions(engine_cuda PRIVATE HAVE_CUDA)
    if(HAVE_CUDNN)
        target_include_directories(engine_cuda PRIVATE "${CUDNN_INCLUDE}")
        target_compile_definitions(engine_cuda PRIVATE HAVE_CUDNN)
    endif()
    set_target_properties(engine_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

    # Propagate CUDA compile-time definitions to the CXX engine (for headers
    # like cuda_backend.hpp that are included from layer / model code).
    target_compile_definitions(engine PUBLIC HAVE_CUDA)
    target_include_directories(engine PUBLIC "${CUDA_INCLUDE_DIR}")
    if(HAVE_CUDNN)
        target_compile_definitions(engine PUBLIC HAVE_CUDNN)
        target_include_directories(engine PUBLIC "${CUDNN_INCLUDE}")
    endif()
endif()

# ── Executables ───────────────────────────────────────────────
add_executable(image_classify  examples/image_classify/main.cpp)
add_executable(test_multiio    examples/test_multiio/main.cpp)
add_executable(kitten_demo     examples/kitten_demo/main.cpp)
add_executable(lm_demo         examples/lm_demo/main.cpp)

add_executable(bench_image   bench/image_bench.cpp)
add_executable(bench_lm      bench/lm_bench.cpp)
add_executable(bench_kitten  bench/kitten_bench.cpp)
add_executable(bench_all
    bench/image_bench.cpp bench/kitten_bench.cpp
    bench/lm_bench.cpp bench/bench_all.cpp)
target_compile_definitions(bench_all PRIVATE BENCH_ALL_BUILD)
add_executable(run_tests
    tests/run_tests.cpp tests/test_layers.cpp tests/test_models.cpp)

# ── Diagnostic probe tools (built only when MODULE_PROBE=ON) ──────
# GRAPH_DEBUG and MODULE_PROBE compile flags are applied ONLY to probe targets,
# never to bench_kitten or run_tests (which would be unusably slow with per-layer dumps).
if(MODULE_PROBE)
    add_executable(probe_kitten_15m  tools/probe_kitten_15m.cpp)
    add_executable(probe_decoder_15m tools/probe_decoder_15m.cpp)
    target_compile_definitions(probe_kitten_15m  PRIVATE GRAPH_DEBUG MODULE_PROBE)
    # probe_decoder_15m only needs MODULE_PROBE for its own output dumping,
    # but the decoder-only model also benefits from GRAPH_DEBUG for diagnostics.
    target_compile_definitions(probe_decoder_15m PRIVATE GRAPH_DEBUG MODULE_PROBE)

    # model.cpp compiled WITH probe flags — linked only into probe executables.
    # engine OBJECT library excludes model.cpp when MODULE_PROBE=ON (see above).
    add_library(engine_model_probe OBJECT src/engine/model.cpp src/engine/arch_loader.cpp src/engine/weights_loader.cpp src/engine/tensor_pool.cpp src/engine/op_type.cpp)
    target_include_directories(engine_model_probe PUBLIC src)
    target_compile_definitions(engine_model_probe PRIVATE GRAPH_DEBUG MODULE_PROBE)
    target_link_libraries(probe_kitten_15m  PRIVATE engine_model_probe)
    target_link_libraries(probe_decoder_15m PRIVATE engine_model_probe)

    # Regular (non-probe) targets compile model.cpp WITHOUT probe flags.
    foreach(tgt image_classify test_multiio kitten_demo lm_demo
                bench_image bench_lm bench_kitten bench_all run_tests)
        target_sources(${tgt} PRIVATE src/engine/model.cpp src/engine/arch_loader.cpp src/engine/weights_loader.cpp src/engine/tensor_pool.cpp src/engine/op_type.cpp)
    endforeach()
endif()

set(ALL_TARGETS
    image_classify test_multiio kitten_demo lm_demo
    bench_image bench_lm bench_kitten bench_all run_tests)
if(MODULE_PROBE)
    list(APPEND ALL_TARGETS probe_kitten_15m probe_decoder_15m)
endif()

# Every target links the engine OBJECT library (and the CUDA one if available)
# plus any runtime shared libraries.
foreach(tgt ${ALL_TARGETS})
    target_link_libraries(${tgt} PRIVATE engine)
    target_include_directories(${tgt} PRIVATE src)
    if(HAVE_METAL)
        target_sources(${tgt} PRIVATE ${METAL_BACKEND_SOURCES})
        target_compile_definitions(${tgt} PRIVATE HAVE_METAL)
        target_link_libraries(${tgt} PRIVATE
            "-framework Metal"
            "-framework Foundation"
            "-framework MetalPerformanceShaders")
        add_dependencies(${tgt} metal_shaders)
    endif()
    if(HAVE_CUDA)
        target_link_libraries(${tgt} PRIVATE engine_cuda ${CUDA_BLAS_LIB} ${CUDA_RUNTIME_LIB})
        if(HAVE_CUDNN)
            target_link_libraries(${tgt} PRIVATE "${CUDNN_LIB}")
        endif()
    endif()
endforeach()
endif()  # KITTEN_BUILD_EXTRAS


# ── OpenMP ────────────────────────────────────────────────────
if(ENABLE_OPENMP)
    find_package(OpenMP)
else()
    message(STATUS "OpenMP disabled by ENABLE_OPENMP=OFF")
endif()
if(ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    message(STATUS "OpenMP ${OpenMP_CXX_VERSION} found — multi-threaded GEMM enabled")
    # Link to engine so compile flags (-fopenmp) propagate to all consumers,
    # and add the link library to every executable explicitly.
    if(TARGET engine)
        target_link_libraries(engine PUBLIC OpenMP::OpenMP_CXX)
    endif()
    foreach(tgt ${ALL_TARGETS})
        target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
    endforeach()
else()
    message(STATUS "OpenMP not found — single-threaded mode")
endif()

# ── LTO ───────────────────────────────────────────────────────
# Disable LTO when CUDA is enabled: nvcc uses a different toolchain (conda-forge gcc)
# that has hardcoded /lib64 paths conflicting with the Ubuntu host at link-time.
if(NOT HAVE_CUDA)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported)
    if(ipo_supported)
        message(STATUS "LTO enabled")
        if(TARGET engine)
            set_property(TARGET engine PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        endif()
        foreach(tgt ${ALL_TARGETS})
            set_property(TARGET ${tgt} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        endforeach()
    endif()
else()
    message(STATUS "LTO disabled (CUDA build)")
endif()

# ── Python bindings (optional) ────────────────────────────────
if(CMAKE_CROSSCOMPILING)
    find_package(Python3 COMPONENTS Development.Module)
else()
    find_package(Python3 COMPONENTS Interpreter Development.Module)
endif()
if(Python3_FOUND AND Python3_Development.Module_FOUND)
    set(PYTHON_MODULE_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('MULTIARCH') or '')"
            OUTPUT_VARIABLE PYTHON_MULTIARCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()
    if(PYTHON_MULTIARCH)
        set(PYTHON_MULTIARCH_INCLUDE_DIR
            "/usr/include/${PYTHON_MULTIARCH}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
        if(EXISTS "${PYTHON_MULTIARCH_INCLUDE_DIR}/pyconfig.h")
            list(APPEND PYTHON_MODULE_INCLUDE_DIRS "${PYTHON_MULTIARCH_INCLUDE_DIR}")
            set(PYTHON_MULTIARCH_PARENT_INCLUDE_DIR "/usr/include")
        endif()
    endif()

    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND AND Python3_Interpreter_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(PYBIND11_CMAKE_DIR)
            find_package(pybind11 QUIET HINTS "${PYBIND11_CMAKE_DIR}")
        endif()
    endif()
    if(pybind11_FOUND)
        message(STATUS "pybind11 ${pybind11_VERSION} found — building Python bindings")
        # Compile sources directly into the module (not via engine) so PIC is
        # applied automatically by pybind11_add_module without affecting the
        # non-PIC engine OBJECT library used by the main executables.
        pybind11_add_module(_engine
            python/bindings.cpp
            ${OPS_SOURCES}
            src/engine/model.cpp
            src/engine/arch_loader.cpp
            src/engine/weights_loader.cpp
            src/engine/tensor_pool.cpp
            src/engine/op_type.cpp
            ${LAYER_SOURCES}
            src/models/image_classifier.cpp
            src/models/kitten.cpp
            src/models/language_model.cpp
        )
        target_include_directories(_engine PRIVATE src)
        target_include_directories(_engine SYSTEM PRIVATE ${PYTHON_MODULE_INCLUDE_DIRS})
        if(ANDROID)
            # Android extension modules are linked with --no-undefined, unlike
            # desktop Python extension modules where Python C-API symbols are
            # commonly resolved by the host interpreter at load time. Link the
            # target Python shared library explicitly so pybind11 symbols such
            # as PyErr_Fetch/PyObject_Str resolve during the Android link step.
            set(_python_android_library_dirs ${Python3_LIBRARY_DIRS})
            foreach(_python_include_dir IN LISTS Python3_INCLUDE_DIRS)
                get_filename_component(_python_include_parent "${_python_include_dir}" DIRECTORY)
                get_filename_component(_python_prefix "${_python_include_parent}" DIRECTORY)
                if(EXISTS "${_python_prefix}/lib")
                    list(APPEND _python_android_library_dirs "${_python_prefix}/lib")
                endif()
            endforeach()
            find_library(PYTHON_ANDROID_LIBRARY
                NAMES
                    "python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}"
                    "python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}"
                PATHS ${_python_android_library_dirs}
                NO_DEFAULT_PATH)
            if(PYTHON_ANDROID_LIBRARY)
                message(STATUS "Android Python library found: ${PYTHON_ANDROID_LIBRARY}")
                target_link_libraries(_engine PRIVATE "${PYTHON_ANDROID_LIBRARY}")
            else()
                message(WARNING "Android Python library not found; Android link may fail with unresolved Python C-API symbols")
            endif()
        endif()
        if(PYTHON_MULTIARCH_PARENT_INCLUDE_DIR AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(_engine PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-idirafter>
                $<$<COMPILE_LANGUAGE:CXX>:${PYTHON_MULTIARCH_PARENT_INCLUDE_DIR}>)
        endif()
        if(HAVE_CUDA)
            target_sources(_engine PRIVATE ${CUDA_BACKEND_SOURCES})
            target_include_directories(_engine PRIVATE "${CUDA_INCLUDE_DIR}")
            target_link_libraries(_engine PRIVATE ${CUDA_BLAS_LIB} ${CUDA_RUNTIME_LIB})
            if(WIN32 AND MSVC AND CUBLAS_DLL_NAME)
                target_link_libraries(_engine PRIVATE delayimp)
                target_link_options(_engine PRIVATE "/DELAYLOAD:${CUBLAS_DLL_NAME}")
            endif()
            target_compile_definitions(_engine PRIVATE HAVE_CUDA)
            if(HAVE_CUDNN)
                target_include_directories(_engine PRIVATE "${CUDNN_INCLUDE}")
                target_link_libraries(_engine PRIVATE "${CUDNN_LIB}")
                target_compile_definitions(_engine PRIVATE HAVE_CUDNN)
            endif()
        endif()
        if(HAVE_METAL)
            target_sources(_engine PRIVATE ${METAL_BACKEND_SOURCES})
            target_compile_definitions(_engine PRIVATE HAVE_METAL)
            target_link_libraries(_engine PRIVATE
                "-framework Metal"
                "-framework Foundation"
                "-framework MetalPerformanceShaders")
            add_dependencies(_engine metal_shaders)
        endif()
        if(ENABLE_OPENMP AND OpenMP_CXX_FOUND)
            target_link_libraries(_engine PRIVATE OpenMP::OpenMP_CXX)
        endif()
        if(NOT HAVE_CUDA)
            set_property(TARGET _engine PROPERTY INTERPROCEDURAL_OPTIMIZATION ${ipo_supported})
        endif()
    else()
        message(STATUS "pybind11 not found — skipping Python bindings (run: pip install pybind11)")
    endif()
else()
    message(STATUS "Python3 development headers not found — skipping Python bindings")
endif()

# ── Install ───────────────────────────────────────────────────
# The extension and the Metal shader library go next to the Python package files
# (kitten_inference/tts08/). The model definitions ship with the Python package.
set(KITTEN_PY_DEST "kitten_inference/tts08")
if(TARGET _engine)
    install(TARGETS _engine
            LIBRARY DESTINATION ${KITTEN_PY_DEST} COMPONENT python
            RUNTIME DESTINATION ${KITTEN_PY_DEST} COMPONENT python
            ARCHIVE DESTINATION ${KITTEN_PY_DEST} COMPONENT python)
    if(HAVE_METAL)
        install(FILES "${METAL_LIB}" DESTINATION ${KITTEN_PY_DEST} COMPONENT python)
    endif()
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cpu_requirements.txt" DESTINATION ${KITTEN_PY_DEST}
            RENAME _cpu_requirements.txt COMPONENT python)
endif()

if(KITTEN_BUILD_EXTRAS)
    install(TARGETS image_classify DESTINATION bin)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Architecture compile options: ${ARCH_FLAGS}")
