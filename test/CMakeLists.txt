project(tests)

include_directories(${CMAKE_SOURCE_DIR})

# Add Google Mock
include(GMock.cmake)

add_subdirectory(utils)

set(
  BOUNDING_BOX_TESTS
  octree/bounding_box/test_bounding_box.cu
  octree/bounding_box/find_bounding_box_test.cu
  octree/bounding_box/operators.cu
)

set(
  MORTON_TESTS
  octree/morton/matchers/morton_code.cu
  octree/morton/property/morton_code.cu
  octree/morton/helpers_test.cu
  octree/morton/short_dilation_test.cu octree/morton/dilation_test.cu
  octree/morton/short_undilation_test.cu octree/morton/undilation_test.cu
  octree/morton/short_dilation_and_undilation_test.cu octree/morton/dilation_and_undilation_test.cu
  octree/morton/morton_code_calculation_test.cu
  octree/morton/morton_code_reversion_test.cu
  octree/morton/morton_code_revertibility_test.cu
)

set(OCTREE_TESTS ${BOUNDING_BOX_TESTS} ${MORTON_TESTS})

set(TEST_SOURCES test_runner.cpp ${OCTREE_TESTS})

# Add test runner
cuda_add_executable(test_runner ${TEST_SOURCES} OPTIONS ${CUDAOPTIONS})

# Create dependency of test runner on googlemock
add_dependencies(test_runner googlemock)

target_link_libraries(test_runner ${GMOCK_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} octree testing_utils)
