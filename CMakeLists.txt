cmake_minimum_required(VERSION 3.15...3.28)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)

add_custom_command(
  OUTPUT dps.cpp
  COMMENT
    "Making ${CMAKE_CURRENT_BINARY_DIR}/dps.cpp from ${CMAKE_CURRENT_SOURCE_DIR}/mddatasetbuilder/dps.pyx"
  COMMAND
    Python::Interpreter -m cython
    "${CMAKE_CURRENT_SOURCE_DIR}/mddatasetbuilder/dps.pyx" --output-file dps.cpp
  DEPENDS mddatasetbuilder/dps.pyx
  VERBATIM)

python_add_library(
  dps MODULE ${CMAKE_CURRENT_BINARY_DIR}/dps.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mddatasetbuilder/c_stack.cpp WITH_SOABI)
target_include_directories(dps
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/mddatasetbuilder)
target_compile_definitions(dps PRIVATE CYTHON_LIMITED_API=1
                                       Py_LIMITED_API=0x03070000)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(dps PRIVATE CYTHON_TRACE=1 CYTHON_TRACE_NOGIL=1)
endif()

install(TARGETS dps DESTINATION mddatasetbuilder/)
