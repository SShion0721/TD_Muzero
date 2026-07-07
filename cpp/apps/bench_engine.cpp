#include "tdmz/core/engine.hpp"
#include <iostream>
#include <chrono>

using namespace tdmz;

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    int num_games = 50;
    int total_steps = 0;
    
    PythonRNG rng(42);
    
    for (int i = 0; i < num_games; ++i) {
        TDEngine env(11, 11, i);
        while (!env.game_over() && env.time() < 1000.0f) {
            auto legal = env.legal_actions();
            int action = legal[rng.randrange(legal.size())];
            env.step_action(action);
            total_steps++;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    
    std::cout << "Ran " << num_games << " games in " << diff.count() << " seconds.\n";
    std::cout << "Total steps: " << total_steps << "\n";
    std::cout << "Steps per second: " << total_steps / diff.count() << "\n";
    
    return 0;
}
