add_executable(test_engine test_engine.cpp)
target_link_libraries(test_engine tdmz_core)
add_test(NAME test_engine COMMAND test_engine)

add_executable(test_mcts test_mcts.cpp)
target_link_libraries(test_mcts tdmz_core)
add_test(NAME test_mcts COMMAND test_mcts)

if(TDMZ_ENABLE_TORCH)
    add_executable(test_network_shape test_network_shape.cpp)
    target_link_libraries(test_network_shape tdmz_nn)
    add_test(NAME test_network_shape COMMAND test_network_shape)

    add_executable(test_mcts_libtorch_smoke test_mcts_libtorch_smoke.cpp)
    target_link_libraries(test_mcts_libtorch_smoke tdmz_nn)
    add_test(NAME test_mcts_libtorch_smoke COMMAND test_mcts_libtorch_smoke)
endif()
