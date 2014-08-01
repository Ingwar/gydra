cmake_minimum_required(VERSION 2.8.7)

project(gydra)

# Enable warnings and optimisation.
# Warnings for unused local typedefs are suppressed because Thrust headers
# generate a lot of them.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-local-typedefs -O2")

find_package(CUDA REQUIRED)

# Minimum architecture required for usage of assertions in the device code
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=compute_20")

# Add the project top-level ("public") headers to the include paths
include_directories(${CMAKE_SOURCE_DIR}/src/include)

# Compile tests
add_subdirectory(test)
