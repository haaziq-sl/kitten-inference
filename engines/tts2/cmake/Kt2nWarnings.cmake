# kt2n_set_warnings(<target> [THIRD_PARTY_HEADERS]): warning set for our own
# targets only (never for fetched dependencies). KT2N_WERROR=ON turns warnings
# into errors. THIRD_PARTY_HEADERS drops -Wshadow / -Wcast-qual, which some
# third-party headers trigger.

function(kt2n_set_warnings target)
  cmake_parse_arguments(ARG "THIRD_PARTY_HEADERS" "" "" ${ARGN})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wnon-virtual-dtor
                                -Woverloaded-virtual -Wformat=2 -Wimplicit-fallthrough -Wundef>)
    if(NOT ARG_THIRD_PARTY_HEADERS)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wshadow -Wcast-qual>)
    endif()
    if(KT2N_WERROR)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
    endif()
  elseif(MSVC)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W3>)
    if(KT2N_WERROR)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/WX>)
    endif()
  endif()
  if(KT2N_HAVE_HIP)
    # HIP: clang compiles host and device code; ignored hipError_t results are as in the CUDA code.
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:HIP>:-Wall -Wextra -Wno-unknown-pragmas -Wno-unused-result>)
    if(KT2N_WERROR)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-Werror>)
    endif()
  elseif(KT2N_HAVE_CUDA AND MSVC)
    # nvcc with MSVC as the host compiler: the GCC warning flags below do not apply.
    if(KT2N_WERROR)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)
    endif()
  elseif(KT2N_HAVE_CUDA)
    # -Wno-unknown-pragmas: `#pragma unroll` in __host__ __device__ code is
    # valid CUDA but unknown to the host compiler.
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wno-unknown-pragmas>)
    if(KT2N_WERROR)
      target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings -Xcompiler=-Werror>)
      # GCC's -Wmaybe-uninitialized depends on the GCC version and on inlining.
      # GCC 13 reports it for an LM launch record whose sequence number is copied
      # before it is assigned but is always assigned before it is read
      # (src/lm/lm_engine.cu). It stays a warning, not an error.
      if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${target} PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-error=maybe-uninitialized>)
      endif()
    endif()
  endif()
endfunction()
