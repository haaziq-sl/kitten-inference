# A minimal application that uses the installed package.
cmake_minimum_required(VERSION 3.28)
project(kt2n_consumer LANGUAGES C CXX)

find_package(kt2n REQUIRED)
if(KT2N_EXPECTED_VERSION AND NOT kt2n_VERSION VERSION_EQUAL KT2N_EXPECTED_VERSION)
  message(FATAL_ERROR "found kt2n ${kt2n_VERSION}, expected ${KT2N_EXPECTED_VERSION}")
endif()

# main.c checks the header from C11, header_cxx.cc from C++17.
add_executable(kt2n_consumer main.c header_cxx.cc)
target_link_libraries(kt2n_consumer PRIVATE kt2n::engine)
set_target_properties(kt2n_consumer PROPERTIES
  C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(kt2n_consumer PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(WIN32)  # Windows finds DLLs next to the program: copy the engine DLL there
  add_custom_command(TARGET kt2n_consumer POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different $<TARGET_RUNTIME_DLLS:kt2n_consumer> $<TARGET_FILE_DIR:kt2n_consumer>
    COMMAND_EXPAND_LISTS)
endif()

enable_testing()
add_test(NAME consumer.run COMMAND kt2n_consumer)
