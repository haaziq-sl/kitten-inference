# C API examples (plain C11, public header kt2n/engine.h only).
#
# Built with the engine when KT2N_BUILD_EXAMPLES is ON (the default for a
# top-level build); the programs land in <build>/examples/. They also build on
# their own against an installed package:
#
#   cmake -S examples -B build-examples -DCMAKE_PREFIX_PATH=<install prefix>
#   cmake --build build-examples
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.28)
  project(kt2n_examples LANGUAGES C)
  find_package(kt2n 0.2 REQUIRED)
elseif(NOT TARGET kt2n::engine)
  message(STATUS "kt2n: examples skipped (they need the CUDA engine library)")
  return()
endif()

# One program per name: <name>.c builds the executable <name>.
set(KT2N_EXAMPLES stream complete clone session voice_cache)

foreach(_example IN LISTS KT2N_EXAMPLES)
  set(_target kt2n_example_${_example})
  add_executable(${_target} ${_example}.c example_util.h)
  target_link_libraries(${_target} PRIVATE kt2n::engine)
  set_target_properties(${_target} PROPERTIES
    OUTPUT_NAME ${_example}
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF)
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${_target} PRIVATE
      -Wall -Wextra -Wpedantic -Wshadow -Wformat=2 -Wundef -Wstrict-prototypes -Wmissing-prototypes)
    if(KT2N_WERROR)
      target_compile_options(${_target} PRIVATE -Werror)
    endif()
  endif()
endforeach()
