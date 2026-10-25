# Native normalization uses the dependencies built by tools/build_text_deps.py.
option(KT2N_BUILD_TEXT "Build the native text normaliser (needs OpenFst + PCRE2 under KT2N_TEXT_DEPS)" OFF)
option(KT2N_BUILD_VOICE_NORMALIZATION "Build frozen multilingual voice normalization (ICU >=72)" OFF)
set(KT2N_TEXT_DEPS "" CACHE PATH "Prefix with include/fst, include/pcre2.h, lib/libfst.a, lib/libpcre2-8.a")

if(WIN32)  # MSVC static libraries from tools/build_text_deps.py
  set(_kt2n_fst_lib "${KT2N_TEXT_DEPS}/lib/fst.lib")
  set(_kt2n_pcre2_lib "${KT2N_TEXT_DEPS}/lib/pcre2-8.lib")
else()
  set(_kt2n_fst_lib "${KT2N_TEXT_DEPS}/lib/libfst.a")
  set(_kt2n_pcre2_lib "${KT2N_TEXT_DEPS}/lib/libpcre2-8.a")
endif()

if(KT2N_BUILD_TEXT)
  if(NOT EXISTS "${_kt2n_fst_lib}" OR NOT EXISTS "${_kt2n_pcre2_lib}")
    message(FATAL_ERROR "Run python3 tools/build_text_deps.py --prefix <prefix>, then configure with -DKT2N_TEXT_DEPS=<prefix>")
  endif()
  set(_kt2n_text_sources
    src/text/unicode.cc src/text/pyre.cc src/text/nemo.cc src/text/normalizer.cc
    src/text/incremental.cc src/text/fast_tagger.cc src/text/c_api.cc)
  add_library(kt2n_text STATIC ${_kt2n_text_sources})
  target_include_directories(kt2n_text
    PUBLIC "${PROJECT_SOURCE_DIR}/include"
    PRIVATE "${PROJECT_SOURCE_DIR}/src/text" "${KT2N_TEXT_DEPS}/include")
  if(WIN32)
    # pcre2.h: static library (no dllimport); OpenFst built without dlopen() (tools/build_text_deps.py).
    target_compile_definitions(kt2n_text PUBLIC PCRE2_STATIC FST_NO_DYNAMIC_LINKING)
  else()
    target_compile_options(kt2n_text PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
  endif()
  target_link_libraries(kt2n_text PUBLIC "${_kt2n_fst_lib}" "${_kt2n_pcre2_lib}"
    ${CMAKE_DL_LIBS} Threads::Threads)

  # libkt2n_text.so: the C ABI (include/kt2n/text.h) with OpenFst/PCRE2 hidden. Not built on Windows (the engine
  # and the command line tool use the static library).
  set(_kt2n_text_targets kt2n_text)
  if(NOT WIN32)
    list(APPEND _kt2n_text_targets kt2n_text_shared)
    add_library(kt2n_text_shared SHARED ${_kt2n_text_sources})
    set_target_properties(kt2n_text_shared PROPERTIES OUTPUT_NAME kt2n_text
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
    target_include_directories(kt2n_text_shared
      PUBLIC "${PROJECT_SOURCE_DIR}/include"
      PRIVATE "${PROJECT_SOURCE_DIR}/src/text" "${KT2N_TEXT_DEPS}/include")
    target_compile_options(kt2n_text_shared PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
    target_link_libraries(kt2n_text_shared PRIVATE "${_kt2n_fst_lib}" "${_kt2n_pcre2_lib}"
      ${CMAKE_DL_LIBS} Threads::Threads)
    target_link_options(kt2n_text_shared PRIVATE -Wl,--exclude-libs,ALL -Wl,-Bsymbolic)
  endif()
  if(KT2N_BUILD_VOICE_NORMALIZATION)
    find_package(ICU 72 REQUIRED COMPONENTS i18n uc)
    foreach(_target IN LISTS _kt2n_text_targets)
      target_sources(${_target} PRIVATE src/text/voice/voice_normalizer.cpp)
      target_link_libraries(${_target} PUBLIC ICU::i18n ICU::uc)
      target_compile_definitions(${_target} PUBLIC KT2N_TEXT_WITH_VOICE=1)
    endforeach()
  endif()

  add_executable(kt2n_prepare_text_data tools/prepare_text_data.cc)
  target_include_directories(kt2n_prepare_text_data PRIVATE "${KT2N_TEXT_DEPS}/include")
  target_link_libraries(kt2n_prepare_text_data PRIVATE "${_kt2n_fst_lib}" ${CMAKE_DL_LIBS} Threads::Threads)
  if(WIN32)
    target_compile_definitions(kt2n_prepare_text_data PRIVATE FST_NO_DYNAMIC_LINKING)
    # UTF-8 arguments and file names: it opens its two paths with the narrow C runtime calls, which otherwise read
    # them in the ANSI code page, so a path outside it (C:\Users\Łukasz under cp1252) could not be opened.
    target_sources(kt2n_prepare_text_data PRIVATE tools/kt2n_prepare_text_data.manifest)
  endif()
  install(TARGETS kt2n_prepare_text_data RUNTIME DESTINATION bin)
endif()
