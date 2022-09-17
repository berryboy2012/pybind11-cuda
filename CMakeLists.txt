# use cmake version >= 3.12 so that to benefit from
# the new FindPython3 module
# cmake >= 3.18 for modern CUDA support
cmake_minimum_required(VERSION 3.18)

message("Using CMake version ${CMAKE_VERSION}")
if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
    message("Using cmake policy CMP0104 OLD")
    cmake_policy(SET CMP0104 OLD)
else()
    message("Using cmake policy CMP0104 NEW")
    cmake_policy(SET CMP0104 NEW)
endif()

project(pybind11-cuda-test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# custom cmake macros location
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda")

# Prevent from build in source tree
include(preventBuildInSource)

# Init build type: Release, Debug, ...
include(initBuildType)

# useful for debugging cmake
include(print_target_properties)

# always export compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#-------------------------------------------------------------------#
# Cuda and Python configuration
#-------------------------------------------------------------------#


if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# make sure to find python from conda, if a conda virtual env is activated
set(Python3_FIND_VIRTUALENV "STANDARD")
# Don't use FindPython here, pybind11 takes care of things much better
#find_package(Python3 COMPONENTS Interpreter Development)
#find_package(PythonInterp 3.6 REQUIRED)
#find_package(PythonLibs 3.6 REQUIRED)

# we also need extra stuff to make sure compile flags are correctly
# passed to nvcc / host compiler
include(protect_nvcc_flags)
include(protect_pthread_flag)

#-------------------------------------------------------------------#
# build some Cuda library
#-------------------------------------------------------------------#

add_subdirectory(src)

#-------------------------------------------------------------------#
# Configuration summary
#-------------------------------------------------------------------#
message("//===================================================")
message("  ${PROJECT_NAME} build configuration:")
message("//===================================================")
message("  CUDA compiler ID      : ${CMAKE_CUDA_COMPILER_ID}")
message("  CUDA compiler Version : ${CMAKE_CUDA_COMPILER_VERSION}")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
  "${CMAKE_CXX_COMPILER_VERSION} "
  "${CMAKE_CXX_COMPILER_WRAPPER}")
message("    ${CMAKE_CXX_COMPILER}")
message("  C++ Compiler flags : ${CMAKE_CXX_FLAGS}")
message("  CUDA Compiler      : ${CMAKE_CUDA_COMPILER}")
message("  CUDA Compiler exec : ${CUDA_NVCC_EXECUTABLE}")
message("  CUDA Compile flags : ${CMAKE_CUDA_FLAGS}")
message("  CUDA toolkit inc   : ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
message("  CUDA arch targets  : ${CMAKE_CUDA_ARCHITECTURES}")
