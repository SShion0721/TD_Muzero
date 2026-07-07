add_executable(test_engine test_engine.cpp)
target_link_libraries(test_engine tdmz_core)
add_test(NAME test_engine COMMAND test_engine)

add_executable(test_mcts test_mcts.cpp)
target_link_libraries(test_mcts tdmz_core)
add_test(NAME test_mcts COMMAND test_mcts)
