cmake_minimum_required(VERSION 3.14)
project(TDMuZero LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TDMZ_ENABLE_TORCH "Build LibTorch network components" OFF)

include_directories(include)

# Core Library (No LibTorch Dependency)
file(GLOB CORE_SRC "src/core/*.cpp" "src/mcts/*.cpp")
add_library(tdmz_core ${CORE_SRC})
target_include_directories(tdmz_core PUBLIC include)
set_property(TARGET tdmz_core PROPERTY CXX_STANDARD 17)

if(TDMZ_ENABLE_TORCH)
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    file(GLOB NN_SRC "src/nn/*.cpp")
    add_library(tdmz_nn ${NN_SRC})
    target_include_directories(tdmz_nn PUBLIC include)
    target_link_libraries(tdmz_nn PUBLIC tdmz_core "${TORCH_LIBRARIES}")
    set_property(TARGET tdmz_nn PROPERTY CXX_STANDARD 17)
endif()

# Tests
enable_testing()
add_subdirectory(tests)

# Apps
add_executable(export_cpp_trace apps/export_cpp_trace.cpp)
target_link_libraries(export_cpp_trace tdmz_core)

add_executable(bench_engine apps/bench_engine.cpp)
target_link_libraries(bench_engine tdmz_core)

add_executable(bench_mcts apps/bench_mcts.cpp)
target_link_libraries(bench_mcts tdmz_core)
