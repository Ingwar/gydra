project(tests)

include_directories(${CMAKE_SOURCE_DIR})

# Add Google Mock
include(GMock.cmake)

set(TEST_SOURCES test_runner.cpp octree/find_enclosing_volume_test.cu)

# Add test runner
cuda_add_executable(test_runner ${TEST_SOURCES} OPTIONS ${CUDAOPTIONS})

# Create dependency of test runner on googlemock
add_dependencies(test_runner googlemock)

target_link_libraries(test_runner ${GMOCK_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
