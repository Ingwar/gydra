project(tests)

include_directories(${CMAKE_SOURCE_DIR})

# Add Google Test
include(GTest.cmake)

set(TEST_SOURCES ${PROJECT_SOURCE_DIR}/test_runner.cpp  ${PROJECT_SOURCE_DIR}/octree/find_enclosing_volume_test.cu)

# Add test runner
cuda_add_executable(test_runner ${TEST_SOURCES} OPTIONS ${CUDAOPTIONS})

# Create dependency of test runner on googletest
add_dependencies(test_runner googletest)

target_link_libraries(test_runner ${GTEST_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
