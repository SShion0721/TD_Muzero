cmake_minimum_required(VERSION 3.14)
project(TDMuZero LANGUAGES CXX)

option(TDMZ_ENABLE_TORCH "Build LibTorch network components" OFF)
option(TDMZ_BUILD_ENGINE_DEBUG_GUI "Build the Windows engine debug GUI" ON)

set(TDMZ_CORE_SOURCES
    src/core/action.cpp
    src/core/board_tables.cpp
    src/core/compatibility.cpp
    src/core/enemy.cpp
    src/core/engine.cpp
    src/core/golden_trace.cpp
    src/core/observation.cpp
    src/core/pathfinding.cpp
    src/core/rng.cpp
    src/core/tower.cpp
    src/mcts/mcts.cpp
    src/persistence/checkpoint_manifest.cpp
    src/selfplay/replay_dataset_v2.cpp
    src/selfplay/selfplay_runner.cpp
    src/selfplay/trajectory_writer.cpp
    src/selfplay/transition_shard.cpp
    src/balance/attack_budget.cpp
    src/balance/budgeted_wave_generator.cpp
    src/balance/defense_capacity.cpp
)

add_library(tdmz_core ${TDMZ_CORE_SOURCES})
target_compile_features(tdmz_core PUBLIC cxx_std_17)
target_include_directories(tdmz_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(TDMZ_ENABLE_TORCH)
    find_package(Torch REQUIRED)

    set(TDMZ_NN_SOURCES
        src/nn/action_encoder.cpp
        src/nn/dynamics.cpp
        src/nn/libtorch_evaluator.cpp
        src/nn/muzero_network.cpp
        src/nn/prediction.cpp
        src/nn/representation.cpp
    )

    add_library(tdmz_torch_options INTERFACE)
    if(TORCH_CXX_FLAGS)
        separate_arguments(TDMZ_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
        target_compile_options(tdmz_torch_options INTERFACE ${TDMZ_TORCH_CXX_FLAGS})
    endif()

    add_library(tdmz_nn ${TDMZ_NN_SOURCES})
    target_compile_features(tdmz_nn PUBLIC cxx_std_17)
    target_include_directories(tdmz_nn PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(tdmz_nn PUBLIC tdmz_core tdmz_torch_options ${TORCH_LIBRARIES})
endif()

enable_testing()
add_subdirectory(tests)

function(tdmz_add_core_app target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE tdmz_core)
endfunction()

tdmz_add_core_app(export_cpp_trace apps/export_cpp_trace.cpp)
tdmz_add_core_app(export_golden_traces apps/export_golden_traces.cpp)
tdmz_add_core_app(export_defense_capacity apps/export_defense_capacity.cpp)
tdmz_add_core_app(export_attack_budget apps/export_attack_budget.cpp)
tdmz_add_core_app(export_budgeted_wave apps/export_budgeted_wave.cpp)
tdmz_add_core_app(export_engine_wave_modes apps/export_engine_wave_modes.cpp)
tdmz_add_core_app(export_wave_mode_ab apps/export_wave_mode_ab.cpp)
tdmz_add_core_app(bench_engine apps/bench_engine.cpp)
tdmz_add_core_app(bench_observation apps/bench_observation.cpp)
tdmz_add_core_app(bench_selfplay apps/bench_selfplay.cpp)
tdmz_add_core_app(bench_replay_loader apps/bench_replay_loader.cpp)
tdmz_add_core_app(bench_batch_sampler apps/bench_batch_sampler.cpp)
tdmz_add_core_app(bench_mcts apps/bench_mcts.cpp)
tdmz_add_core_app(generate_selfplay_dummy apps/generate_selfplay_dummy.cpp)
tdmz_add_core_app(generate_selfplay_shards apps/generate_selfplay_shards_v2.cpp)

if(WIN32 AND TDMZ_BUILD_ENGINE_DEBUG_GUI)
    add_executable(tdmz_engine_debug_gui WIN32 apps/engine_debug_gui_win32.cpp)
    target_compile_features(tdmz_engine_debug_gui PRIVATE cxx_std_17)
    target_link_libraries(tdmz_engine_debug_gui PRIVATE tdmz_core user32 gdi32)
endif()
