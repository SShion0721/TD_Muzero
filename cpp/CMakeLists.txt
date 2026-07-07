cmake_minimum_required(VERSION 3.14)
project(TDMuZero LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point to LibTorch from the micromamba env
set(CMAKE_PREFIX_PATH "E:/Micromamba/envs/mamba_env/Lib/site-packages/torch/share/cmake" ${CMAKE_PREFIX_PATH})

find_package(Torch REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})
include_directories(include)

# Core Library
file(GLOB CORE_SRC "src/core/*.cpp")
add_library(tdmz_core ${CORE_SRC})
target_include_directories(tdmz_core PUBLIC include)
target_link_libraries(tdmz_core "${TORCH_LIBRARIES}")
set_property(TARGET tdmz_core PROPERTY CXX_STANDARD 17)

# Tests
enable_testing()
# We will define test targets here later

# Apps
add_executable(export_cpp_trace apps/export_cpp_trace.cpp)
target_link_libraries(export_cpp_trace tdmz_core)
add_executable(test_rng apps/test_rng.cpp)
target_link_libraries(test_rng tdmz_core)
