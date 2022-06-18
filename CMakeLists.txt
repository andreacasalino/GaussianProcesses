cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

option(LIB_OPT "Compile shared libraries (ON) or static (OFF)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/MakeLibrary.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FecthJSONlib.cmake)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

project(Gaussian-Process)
add_subdirectory(src)

option(BUILD_GaussianProcess_SAMPLES "Build the samples showing how to use the package" ON)
if(BUILD_GaussianProcess_SAMPLES)
	project(GaussianProcess-Samples)
    add_subdirectory(samples)
endif()

option(BUILD_GaussianProcess_KERNEL_VIZ "Build applications that visualizes kernel functions shape" ON)
if(BUILD_GaussianProcess_KERNEL_VIZ)
    project(GaussianProcessKernel-Visualizations)
    add_subdirectory(kernels_viz)
endif()

option(BUILD_GaussianProcess_TESTS "" OFF)
if(BUILD_GaussianProcess_TESTS)
	project(GaussianProcess-Tests)
    add_subdirectory(tests)
endif()
