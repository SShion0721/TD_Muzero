add_executable(test_engine test_engine.cpp)
target_link_libraries(test_engine tdmz_core)
add_test(NAME test_engine COMMAND test_engine)

add_executable(test_board_tables test_board_tables.cpp)
target_link_libraries(test_board_tables tdmz_core)
add_test(NAME test_board_tables COMMAND test_board_tables)

add_executable(test_game_logic test_game_logic.cpp)
target_link_libraries(test_game_logic tdmz_core)
add_test(NAME test_game_logic COMMAND test_game_logic)

add_executable(test_golden_trace test_golden_trace.cpp)
target_link_libraries(test_golden_trace tdmz_core)
add_test(NAME test_golden_trace COMMAND test_golden_trace)

add_executable(test_defense_capacity test_defense_capacity.cpp)
target_link_libraries(test_defense_capacity tdmz_core)
add_test(NAME test_defense_capacity COMMAND test_defense_capacity)

add_executable(test_attack_budget test_attack_budget.cpp)
target_link_libraries(test_attack_budget tdmz_core)
add_test(NAME test_attack_budget COMMAND test_attack_budget)

add_executable(test_budgeted_wave_generator test_budgeted_wave_generator.cpp)
target_link_libraries(test_budgeted_wave_generator tdmz_core)
add_test(NAME test_budgeted_wave_generator COMMAND test_budgeted_wave_generator)

add_executable(test_engine_wave_modes test_engine_wave_modes.cpp)
target_link_libraries(test_engine_wave_modes tdmz_core)
add_test(NAME test_engine_wave_modes COMMAND test_engine_wave_modes)

add_executable(test_observation_cache test_observation_cache.cpp)
target_link_libraries(test_observation_cache tdmz_core)
add_test(NAME test_observation_cache COMMAND test_observation_cache)

add_executable(test_binary_trajectory test_binary_trajectory.cpp)
target_link_libraries(test_binary_trajectory tdmz_core)
add_test(NAME test_binary_trajectory COMMAND test_binary_trajectory)

add_executable(test_mcts test_mcts.cpp)
target_link_libraries(test_mcts tdmz_core)
add_test(NAME test_mcts COMMAND test_mcts)

add_executable(test_selfplay test_selfplay.cpp)
target_link_libraries(test_selfplay tdmz_core)
add_test(NAME test_selfplay COMMAND test_selfplay)

if(TDMZ_ENABLE_TORCH)
    add_executable(test_network_shape test_network_shape.cpp)
    target_link_libraries(test_network_shape tdmz_nn)
    add_test(NAME test_network_shape COMMAND test_network_shape)

    add_executable(test_mcts_libtorch_smoke test_mcts_libtorch_smoke.cpp)
    target_link_libraries(test_mcts_libtorch_smoke tdmz_nn)
    add_test(NAME test_mcts_libtorch_smoke COMMAND test_mcts_libtorch_smoke)

    add_executable(test_selfplay_torch test_selfplay_libtorch.cpp)
    target_link_libraries(test_selfplay_torch tdmz_nn)
    add_test(NAME test_selfplay_torch COMMAND test_selfplay_torch)
endif()
