cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(sim)

find_package(Torch REQUIRED)

add_executable(simulator simulator.cpp)
add_executable(simulator_parallel parall_simnet.cpp)
add_executable(simulator_ground_truth simulator.cpp)
add_executable(simulator_cla simulator.cpp)

target_link_libraries(simulator "${TORCH_LIBRARIES}")
target_link_libraries(simulator_parallel "${TORCH_LIBRARIES}")
target_link_libraries(simulator_ground_truth "${TORCH_LIBRARIES}")
target_link_libraries(simulator_cla "${TORCH_LIBRARIES}")

set_property(TARGET simulator PROPERTY CXX_STANDARD 11)
set_property(TARGET simulator_parallel PROPERTY CXX_STANDARD 11)
set_property(TARGET simulator_ground_truth  PROPERTY CXX_STANDARD 11)
set_property(TARGET simulator_cla PROPERTY CXX_STANDARD 11)

target_compile_definitions(simulator_ground_truth PRIVATE RUN_TRUTH)
target_compile_definitions(simulator_cla PRIVATE CLASSIFY)
