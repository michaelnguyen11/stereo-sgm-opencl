cmake_minimum_required(VERSION 3.10)

project(stereo_sgm_opencl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

option(BUILD_EXAMPLES "Build sgm ocl example" ON)
set(CL_TARGET_OPENCL_VERSION 120 CACHE STRING "OpenCL target version")
message(STATUS "OpenCL target verison: ${CL_TARGET_OPENCL_VERSION}")

add_subdirectory(src)
add_subdirectory(camera)

find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)

file(GLOB SOURCES "*.cpp")

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIR})

add_executable(stereo_sgm_opencl ${SOURCES})

target_link_libraries(stereo_sgm_opencl libsgm_ocl ocams_camera ${OpenCL_LIBRARIES} ${OpenCV_LIBS})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
