# Install test: installs the package into a scratch prefix, then configures,
# builds and runs a separate project that finds it with find_package(kt2n), the
# way an application would. A second project builds the examples against the
# same installed package. Neither needs a GPU.
if(NOT TARGET kt2n_engine)
  return()
endif()

set(_kt2n_prefix "${CMAKE_CURRENT_BINARY_DIR}/prefix")
include(GNUInstallDirs)

get_property(_kt2n_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(_kt2n_multi_config)
  set(_kt2n_config_args "-DCONFIG=$<CONFIG>")
  set(_kt2n_build_config --build-config "$<CONFIG>")
else()
  set(_kt2n_config_args "")
  set(_kt2n_build_config "")
endif()

add_test(NAME consumer.install
  COMMAND "${CMAKE_COMMAND}"
    "-DBUILD_DIR=${PROJECT_BINARY_DIR}" "-DPREFIX=${_kt2n_prefix}" "-DLIBDIR=${CMAKE_INSTALL_LIBDIR}"
    ${_kt2n_config_args} -P "${CMAKE_CURRENT_SOURCE_DIR}/install.cmake")
set_tests_properties(consumer.install PROPERTIES FIXTURES_SETUP kt2n_installed LABELS cpu TIMEOUT 300)

# The same compilers and generator as this build.
set(_kt2n_build_and_test
  --build-generator "${CMAKE_GENERATOR}"
  --build-makeprogram "${CMAKE_MAKE_PROGRAM}"
  ${_kt2n_build_config})
set(_kt2n_options
  "-DCMAKE_PREFIX_PATH=${_kt2n_prefix}"
  "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
  "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
if(NOT _kt2n_multi_config AND CMAKE_BUILD_TYPE)
  list(APPEND _kt2n_options "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
endif()

add_test(NAME consumer.find_package
  COMMAND "${CMAKE_CTEST_COMMAND}"
    --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/app" "${CMAKE_CURRENT_BINARY_DIR}/app"
    ${_kt2n_build_and_test}
    --build-project kt2n_consumer
    --build-options ${_kt2n_options} "-DKT2N_EXPECTED_VERSION=${PROJECT_VERSION}"
    --test-command "${CMAKE_CTEST_COMMAND}" --output-on-failure ${_kt2n_build_config})
set_tests_properties(consumer.find_package PROPERTIES FIXTURES_REQUIRED kt2n_installed LABELS cpu TIMEOUT 300)

if(WIN32)
  return()  # the C examples are POSIX programs (poll on stdin and the event fd, sigaction); Linux only for now
endif()
add_test(NAME consumer.examples
  COMMAND "${CMAKE_CTEST_COMMAND}"
    --build-and-test "${PROJECT_SOURCE_DIR}/examples" "${CMAKE_CURRENT_BINARY_DIR}/examples"
    ${_kt2n_build_and_test}
    --build-project kt2n_examples
    --build-options ${_kt2n_options} -DKT2N_WERROR=ON)
set_tests_properties(consumer.examples PROPERTIES FIXTURES_REQUIRED kt2n_installed LABELS cpu TIMEOUT 300)
