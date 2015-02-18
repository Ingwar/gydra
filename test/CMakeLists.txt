project(tests)

include_directories(${CMAKE_SOURCE_DIR})

# Add Google Mock
include(GMock.cmake)

add_subdirectory(utils)

set(
  MORTON_TESTS
  octree/morton_tests/matchers/morton_order.cu
  octree/morton_tests/property/morton_order.cu
  octree/morton_tests/utils.cpp
  octree/morton_tests/helpers_test.cu
  octree/morton_tests/short_dilation_test.cu octree/morton_tests/dilation_test.cu
  octree/morton_tests/short_undilation_test.cu octree/morton_tests/undilation_test.cu
  octree/morton_tests/short_dilation_and_undilation_test.cu octree/morton_tests/dilation_and_undilation_test.cu
  octree/morton_tests/morton_order_calculation_test.cu
  octree/morton_tests/morton_order_reversion_test.cu
  octree/morton_tests/morton_order_revertibility_test.cu
)

set(OCTREE_TESTS octree/find_enclosing_volume_test.cu ${MORTON_TESTS})

set(TEST_SOURCES test_runner.cpp ${OCTREE_TESTS})

# Add test runner
cuda_add_executable(test_runner ${TEST_SOURCES} OPTIONS ${CUDAOPTIONS})

# Create dependency of test runner on googlemock
add_dependencies(test_runner googlemock)

target_link_libraries(test_runner ${GMOCK_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} octree testing_utils)
