#include "tdmz/core/engine.hpp"
#include "tdmz/core/observation.hpp"
#include "tdmz/mcts/mcts.hpp"
#include "tdmz/mcts/dummy_network.hpp"
#include <iostream>
#include <chrono>

int main() {
    tdmz::TDEngine env(11, 11, 0);
    auto obs = tdmz::make_observation_v1(env);
    auto legal = env.legal_actions();

    tdmz::MCTSConfig cfg;
    cfg.num_simulations = 64;
    cfg.latent_top_k = 32;

    tdmz::DummyNetwork net;
    tdmz::MCTS mcts(cfg);

    int num_searches = 1000;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_searches; ++i) {
        mcts.search_single(net, obs, legal);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    
    std::cout << "Ran " << num_searches << " searches in " << diff.count() << " seconds.\n";
    std::cout << "Simulations per second: " << (num_searches * cfg.num_simulations) / diff.count() << "\n";
    
    return 0;
}
