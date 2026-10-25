# kitten-tts-native command line.
#   kt2n_cli_core      CPU static library: argument parser, bundle checks, run report (unit tested anywhere)
#   kitten-tts-native  the executable, built when the engine library (kt2n_engine) is part of the build

add_library(kt2n_cli_core STATIC cli_core.cc ffmpeg_decode.cc)
target_include_directories(kt2n_cli_core
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src/core")
target_link_libraries(kt2n_cli_core PUBLIC kt2n_core PRIVATE kt2n_voice_blob)
target_compile_features(kt2n_cli_core PUBLIC cxx_std_20)
kt2n_set_warnings(kt2n_cli_core)

if(TARGET kt2n_engine)
  include(GNUInstallDirs)
  add_executable(kitten-tts-native kitten_tts_native.cc)
  target_link_libraries(kitten-tts-native PRIVATE kt2n_cli_core kt2n_engine)
  if(WIN32)
    target_sources(kitten-tts-native PRIVATE kitten-tts-native.manifest)  # UTF-8 arguments and file names
    # The engine DLL is loaded at start from next to the program or, in the Python wheel's layout
    # (bin\kitten-tts-native.exe, lib\kt2n_engine.dll), from ..\lib (load_engine_dll() in kitten_tts_native.cc).
    target_link_options(kitten-tts-native PRIVATE "/DELAYLOAD:$<TARGET_FILE_NAME:kt2n_engine>")
    target_link_libraries(kitten-tts-native PRIVATE delayimp)
  endif()
  set_target_properties(kitten-tts-native PROPERTIES
    BUILD_RPATH_USE_ORIGIN ON
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
  kt2n_set_warnings(kitten-tts-native)
  install(TARGETS kitten-tts-native RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
