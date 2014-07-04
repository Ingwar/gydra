cmake_minimum_required(VERSION 2.8.7 FATAL_ERROR)

project(tests)

#Minimum architecture required for usage of assertions in the device code
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=compute_20")

include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src/include)

#Add Google Test
include(GTest.cmake)

set(TEST_SOURCES ${PROJECT_SOURCE_DIR}/test_runner.cpp  ${PROJECT_SOURCE_DIR}/octree/find_enclosing_volume_test.cu)

# Add test runner
cuda_add_executable(test_runner ${TEST_SOURCES} OPTIONS ${CUDAOPTIONS})

# Create dependency of test runner on googletest
add_dependencies(test_runner googletest)

target_link_libraries(test_runner ${GTEST_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
