# CMake script for including Google Test framework to the project

#Required for Google test
find_package(Threads)

# Enable ExternalProject CMake module
include(ExternalProject)

# Set default ExternalProject root directory
set_directory_properties(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/ThirdParty)

ExternalProject_Add(
    googletest
    URL http://googletest.googlecode.com/files/gtest-1.7.0.zip
    # Disable install step
    INSTALL_COMMAND ""
    # Wrap download, configure and build steps in a sсript to log output
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON)

# Specify include dir
ExternalProject_Get_Property(googletest source_dir)
set(GTEST_INCLUDE_DIR ${source_dir}/include)
include_directories(${GTEST_INCLUDE_DIR})

# Library
ExternalProject_Get_Property(googletest binary_dir)
set(GTEST_LIBRARY_PATH ${binary_dir}/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.a)
set(GTEST_LIBRARY gtest)

add_library(${GTEST_LIBRARY} UNKNOWN IMPORTED)
set_property(TARGET ${GTEST_LIBRARY} PROPERTY IMPORTED_LOCATION ${GTEST_LIBRARY_PATH} )

add_dependencies(${GTEST_LIBRARY} googletest)
