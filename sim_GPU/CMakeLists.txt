cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})
project(TensorRTSample LANGUAGES CXX CUDA)

find_package(OpenMP)
find_package(Threads)
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)

add_executable(parallel_simulator gpu_simulator_parallel.cu)
target_include_directories(parallel_simulator PUBLIC ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(parallel_simulator PUBLIC ${OpenMP_CXX_FLAGS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})

