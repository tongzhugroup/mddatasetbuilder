cmake_minimum_required(VERSION 3.15...3.28)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)

find_package(Cython REQUIRED)

cython_compile_pyx(${CMAKE_CURRENT_BINARY_DIR}/dps.pyx OUTPUT_VAR
                   _dps_source_files)

python_add_library(dps MODULE ${_dps_source_files}
                   ${CMAKE_CURRENT_BINARY_DIR}/c_stack.cpp WITH_SOABI)
target_compile_definitions(dps PRIVATE CYTHON_LIMITED_API=1
                                       Py_LIMITED_API=0x03070000)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(dps PRIVATE CYTHON_TRACE=1 CYTHON_TRACE_NOGIL=1)
endif()

install(TARGETS dps DESTINATION .)
