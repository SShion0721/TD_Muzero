cmake_minimum_required(VERSION 3.14)
project(TDMuZero LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# Core Library (No LibTorch Dependency)
file(GLOB CORE_SRC "src/core/*.cpp")
add_library(tdmz_core ${CORE_SRC})
target_include_directories(tdmz_core PUBLIC include)
set_property(TARGET tdmz_core PROPERTY CXX_STANDARD 17)

# Tests
enable_testing()
add_subdirectory(tests)

# Apps
add_executable(export_cpp_trace apps/export_cpp_trace.cpp)
target_link_libraries(export_cpp_trace tdmz_core)

add_executable(bench_engine apps/bench_engine.cpp)
target_link_libraries(bench_engine tdmz_core)
