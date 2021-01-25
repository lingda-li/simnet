cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

project(TensorRTSample)

find_package(Threads)
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)

add_executable(trt_static code_static.cpp)
add_executable(trt_dynamic code_dynamic.cpp)
target_include_directories(trt_static PUBLIC ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_include_directories(trt_dynamic PUBLIC ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(trt_static PUBLIC ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
target_link_libraries(trt_dynamic PUBLIC ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})

