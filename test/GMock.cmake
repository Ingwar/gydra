# CMake script for including Google Test framework to the project

#Required for Google test
find_package(Threads)

# Enable ExternalProject CMake module
include(ExternalProject)

# Set default ExternalProject root directory
set_directory_properties(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/ThirdParty)

ExternalProject_Add(
    googlemock
    URL http://googlemock.googlecode.com/files/gmock-1.7.0.zip
    # Disable install step
    INSTALL_COMMAND ""
    # Wrap download, configure and build steps in a sсript to log output
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON)

# Specify include dir
ExternalProject_Get_Property(googlemock source_dir)
set(GTEST_INCLUDE_DIR ${source_dir}/gtest/include)
set(GMOCK_INCLUDE_DIR ${source_dir}/include)
include_directories(${GTEST_INCLUDE_DIR})
include_directories(${GMOCK_INCLUDE_DIR})

# GTest Library
ExternalProject_Get_Property(googlemock binary_dir)
set(GTEST_LIBRARY_PATH ${binary_dir}/gtest/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.a)
set(GTEST_LIBRARY gtest)

add_library(${GTEST_LIBRARY} UNKNOWN IMPORTED)
set_property(TARGET ${GTEST_LIBRARY} PROPERTY IMPORTED_LOCATION ${GTEST_LIBRARY_PATH} )

add_dependencies(${GTEST_LIBRARY} googlemock)

# GMock Library
set(GMOCK_LIBRARY_PATH ${binary_dir}/${CMAKE_FIND_LIBRARY_PREFIXES}gmock.a)
set(GMOCK_LIBRARY gmock)

add_library(${GMOCK_LIBRARY} UNKNOWN IMPORTED)
set_property(TARGET ${GMOCK_LIBRARY} PROPERTY IMPORTED_LOCATION ${GMOCK_LIBRARY_PATH} )

add_dependencies(${GMOCK_LIBRARY} googlemock)
