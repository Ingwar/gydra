project(gydra)

cmake_minimum_required(VERSION 2.8.7)

#Project-wide option for enabling/disabling compilation of tests
option(build_gydra_tests "Build tests for Gydra." ON)

# Enable warnings and optimisation.
# Warnings for unused local typedefs are suppressed because Thrust headers
# generate a lot of them.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-local-typedefs -O2")

find_package(CUDA REQUIRED)

# Minimum architecture required for usage of assertions in the device code
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=compute_20")

# Add the project top-level ("public") headers to the include paths
include_directories(${CMAKE_SOURCE_DIR}/src/include)

# Add library for octree creation
add_subdirectory(src/octree)

# Compile tests if they are enabled
if (build_gydra_tests)
  add_subdirectory(test)
endif()
