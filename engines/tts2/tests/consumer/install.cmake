# cmake -DBUILD_DIR=<build> -DPREFIX=<prefix> -DLIBDIR=<lib dir> [-DCONFIG=<config>] -P install.cmake
#
# Installs a finished build into an empty PREFIX and checks the files an
# application needs: the header, the library and the CMake package files.
foreach(_var BUILD_DIR PREFIX LIBDIR)
  if(NOT DEFINED ${_var})
    message(FATAL_ERROR "install.cmake needs -D${_var}=...")
  endif()
endforeach()

file(REMOVE_RECURSE "${PREFIX}")
set(_args --install "${BUILD_DIR}" --prefix "${PREFIX}")
if(CONFIG)
  list(APPEND _args --config "${CONFIG}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" ${_args} RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "cmake --install failed (${_rc}); build the project before running this test")
endif()

if(CMAKE_HOST_WIN32)
  set(_kt2n_engine_files bin/kt2n_engine.dll ${LIBDIR}/kt2n_engine.lib)  # the DLL and its import library
else()
  set(_kt2n_engine_files ${LIBDIR}/libkt2n_engine.so)
endif()
foreach(_file
    include/kt2n/engine.h
    ${_kt2n_engine_files}
    ${LIBDIR}/cmake/kt2n/kt2nConfig.cmake
    ${LIBDIR}/cmake/kt2n/kt2nConfigVersion.cmake
    ${LIBDIR}/cmake/kt2n/kt2nTargets.cmake)
  if(NOT EXISTS "${PREFIX}/${_file}")
    message(FATAL_ERROR "the installed package has no ${_file}")
  endif()
endforeach()
message(STATUS "installed kt2n into ${PREFIX}")
