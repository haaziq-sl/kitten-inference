# kitten-tts-native tests.
#   cpu: argument parser, bundle checks and report (test_cli_core), the stats reader, and
#        `kitten-tts-native --check-assets` against small fake bundles (test_bundle.py).
#   gpu: end-to-end synthesis with a prepared bundle (e2e.sh; KT2N_TEST_ASSETS names the bundle
#        directory; without it the test is skipped, or fails when KT2N_REQUIRE_GPU=1 as in
#        `ctest --preset gpu`).

add_executable(test_cli_core cli_core_test.cc)
target_link_libraries(test_cli_core PRIVATE kt2n_cli_core GTest::gtest_main)
kt2n_set_warnings(test_cli_core)
gtest_discover_tests(test_cli_core DISCOVERY_MODE PRE_TEST PROPERTIES LABELS cpu)

# The reference-audio decode with the system's ffmpeg program (skips the decode cases when ffmpeg is not on PATH).
add_executable(test_cli_ffmpeg ffmpeg_decode_test.cc)
target_link_libraries(test_cli_ffmpeg PRIVATE kt2n_cli_core GTest::gtest_main)
kt2n_set_warnings(test_cli_ffmpeg)
gtest_discover_tests(test_cli_ffmpeg DISCOVERY_MODE PRE_TEST PROPERTIES LABELS cpu)

# Links no engine: the test provides its own kt2n_stats_json.
add_executable(test_cli_stats_reader stats_reader_test.cc)
target_include_directories(test_cli_stats_reader PRIVATE "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/cli")
target_compile_features(test_cli_stats_reader PRIVATE cxx_std_20)
kt2n_set_warnings(test_cli_stats_reader)
add_test(NAME cli.stats_reader COMMAND test_cli_stats_reader)
set_tests_properties(cli.stats_reader PROPERTIES LABELS cpu)

if(TARGET kitten-tts-native)
  add_test(NAME cli.bundle COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_bundle.py")
  set_tests_properties(cli.bundle PROPERTIES LABELS cpu TIMEOUT 300 SKIP_RETURN_CODE 77
    ENVIRONMENT "KT2N_CLI_BINARY=$<TARGET_FILE:kitten-tts-native>")
  add_test(NAME cli.poll_modes COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/poll_modes.py")
  set_tests_properties(cli.poll_modes PROPERTIES LABELS gpu TIMEOUT 1200 SKIP_RETURN_CODE 77
    ENVIRONMENT "KT2N_CLI_BINARY=$<TARGET_FILE:kitten-tts-native>")
  if(NOT WIN32)  # a bash script; on Windows tests/cli/e2e.sh is not run (it also needs a GPU with more than 24 GB)
    add_test(NAME cli.e2e COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/e2e.sh")
    set_tests_properties(cli.e2e PROPERTIES LABELS gpu TIMEOUT 3600 SKIP_RETURN_CODE 77
      ENVIRONMENT "KT2N_CLI_BINARY=$<TARGET_FILE:kitten-tts-native>")
  endif()
endif()
