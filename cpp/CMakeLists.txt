cmake_minimum_required(VERSION 3.14)
project(TDMuZero LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TDMZ_ENABLE_TORCH "Build LibTorch network components" OFF)

include_directories(include)

file(GLOB CORE_SRC "src/core/*.cpp" "src/mcts/*.cpp" "src/selfplay/*.cpp" "src/balance/*.cpp")
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

enable_testing()
add_subdirectory(tests)

add_executable(export_cpp_trace apps/export_cpp_trace.cpp)
target_link_libraries(export_cpp_trace tdmz_core)

add_executable(export_golden_traces apps/export_golden_traces.cpp)
target_link_libraries(export_golden_traces tdmz_core)

add_executable(export_defense_capacity apps/export_defense_capacity.cpp)
target_link_libraries(export_defense_capacity tdmz_core)

add_executable(export_attack_budget apps/export_attack_budget.cpp)
target_link_libraries(export_attack_budget tdmz_core)

add_executable(export_budgeted_wave apps/export_budgeted_wave.cpp)
target_link_libraries(export_budgeted_wave tdmz_core)

add_executable(export_engine_wave_modes apps/export_engine_wave_modes.cpp)
target_link_libraries(export_engine_wave_modes tdmz_core)

add_executable(export_wave_mode_ab apps/export_wave_mode_ab.cpp)
target_link_libraries(export_wave_mode_ab tdmz_core)

add_executable(bench_engine apps/bench_engine.cpp)
target_link_libraries(bench_engine tdmz_core)

add_executable(bench_observation apps/bench_observation.cpp)
target_link_libraries(bench_observation tdmz_core)

add_executable(bench_selfplay apps/bench_selfplay.cpp)
target_link_libraries(bench_selfplay tdmz_core)

add_executable(bench_replay_loader apps/bench_replay_loader.cpp)
target_link_libraries(bench_replay_loader tdmz_core)

add_executable(bench_batch_sampler apps/bench_batch_sampler.cpp)
target_link_libraries(bench_batch_sampler tdmz_core)

add_executable(bench_mcts apps/bench_mcts.cpp)
target_link_libraries(bench_mcts tdmz_core)

add_executable(generate_selfplay_dummy apps/generate_selfplay_dummy.cpp)
target_link_libraries(generate_selfplay_dummy tdmz_core)

add_executable(generate_selfplay_shards apps/generate_selfplay_shards.cpp)
target_link_libraries(generate_selfplay_shards tdmz_core)
