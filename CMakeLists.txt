cmake_minimum_required(VERSION 2.8.7)

project(gydra)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-local-typedefs -O2")

find_package(CUDA REQUIRED)

add_subdirectory(test)
