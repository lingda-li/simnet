cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(sim)

find_package(Torch REQUIRED)

add_executable(simulator simulator.cpp)
target_link_libraries(simulator "${TORCH_LIBRARIES}")
set_property(TARGET simulator PROPERTY CXX_STANDARD 11)
