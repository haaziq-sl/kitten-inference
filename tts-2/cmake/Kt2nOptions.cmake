# Build options, language standard and compiler floor.

option(KT2N_WERROR "Treat compiler warnings as errors (the dev preset turns this on)" OFF)
option(KT2N_BUILD_TESTS "Build the C++ tests (GoogleTest)" ${PROJECT_IS_TOP_LEVEL})
option(KT2N_BUILD_CLONING "Build on-demand voice cloning networks" OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Static libraries link into the shared engine library.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Floor: gcc 11 (Ubuntu 22.04, the CUDA 12.8 container and the test machine).
# Code must not use std::format / std::expected / other post-gcc-11 library parts.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
  message(FATAL_ERROR "kt2n needs gcc >= 11 (found ${CMAKE_CXX_COMPILER_VERSION})")
endif()

# MSVC (Windows): sources are UTF-8 (string literals in the text code), large translation units need /bigobj,
# and the Windows headers are kept from defining min/max. CUDA files pass the same to the host compiler.
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8> $<$<COMPILE_LANGUAGE:C,CXX>:/bigobj>
                      $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
                      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8,/bigobj,/Zc:__cplusplus>)
  # _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR: since Visual Studio 2022 17.10 a std::mutex built with the default
  # constructor crashes on its first lock() in a process that has loaded an older msvcp140.dll (Python, Java and
  # Electron hosts often have); this keeps the older, compatible constructor.
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS
                          _CRT_NONSTDC_NO_WARNINGS _WIN32_WINNT=0x0A00 _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
  # Programs get 8 MB thread stacks, as on Linux (Windows' default is 1 MB): the text normaliser recurses once per
  # token of a long unpunctuated sentence.
  add_link_options("$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:LINKER:/STACK:8388608>")
endif()

# Exact numerics: refuse flags that change floating-point results.
foreach(_var IN ITEMS CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELWITHDEBINFO
                      CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_MINSIZEREL)
  if("${${_var}}" MATCHES "(-ffast-math|-Ofast|-funsafe-math-optimizations|-ffinite-math-only)")
    message(FATAL_ERROR "${_var} contains '${CMAKE_MATCH_1}'; kt2n needs exact IEEE numerics")
  endif()
endforeach()

# kt2n_exact_numerics(<target>): no FP contraction on the CPU side, so the CPU
# reference kernels give the same bits with gcc and clang at any -O level.
function(kt2n_exact_numerics target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  elseif(MSVC)
    # /fp:precise without /fp:contract: no FMA contraction (Visual Studio 2022 and later).
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/fp:precise>)
  endif()
endfunction()
