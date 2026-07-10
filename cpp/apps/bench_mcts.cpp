#include "tdmz/core/engine.hpp"
#include "tdmz/core/observation.hpp"
#include "tdmz/mcts/dummy_network.hpp"
#include "tdmz/mcts/mcts.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int parse_positive(const char* value, const char* name) {
    const int parsed = std::stoi(value);
    if (parsed <= 0) throw std::runtime_error(std::string(name) + " must be positive");
    return parsed;
}

} // namespace

int main(int argc, char** argv) {
    try {
        int searches = 1000;
        int simulations = 64;
        int latent_top_k = 32;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--searches") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --searches");
                searches = parse_positive(argv[++i], "--searches");
            } else if (argument == "--simulations") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --simulations");
                simulations = parse_positive(argv[++i], "--simulations");
            } else if (argument == "--latent-top-k") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --latent-top-k");
                latent_top_k = parse_positive(argv[++i], "--latent-top-k");
            } else if (argument == "--help" || argument == "-h") {
                std::cout
                    << "Usage: " << argv[0] << " [options]\n"
                    << "  --searches N       Timed searches. Default: 1000\n"
                    << "  --simulations N    Simulations per search. Default: 64\n"
                    << "  --latent-top-k N   Latent branching. Default: 32\n";
                return 0;
            } else {
                throw std::runtime_error("Unknown argument: " + argument);
            }
        }

        tdmz::TDEngine environment(11, 11, 0);
        const auto observation = tdmz::make_observation_v1(environment);
        const auto legal_actions = environment.legal_actions();

        tdmz::MCTSConfig config;
        config.num_simulations = simulations;
        config.latent_top_k = latent_top_k;
        config.max_nodes = 1
            + static_cast<int>(legal_actions.size())
            + simulations * std::min(latent_top_k, tdmz::kActionSpaceSize);

        tdmz::DummyNetwork network;
        tdmz::MCTS mcts(config);

        const tdmz::RootSearchOutput warmup = mcts.search_single(
            network, observation, legal_actions);

        uint64_t created_nodes = 0;
        uint64_t reused_nodes = 0;
        uint64_t node_buffer_growths = 0;
        uint64_t scratch_growths = 0;
        int max_depth = 0;
        double checksum = 0.0;

        const auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < searches; ++i) {
            const tdmz::RootSearchOutput output = mcts.search_single(
                network, observation, legal_actions);
            created_nodes += static_cast<uint64_t>(output.debug.node_objects_created);
            reused_nodes += static_cast<uint64_t>(output.debug.node_objects_reused);
            node_buffer_growths += static_cast<uint64_t>(output.debug.node_buffer_growth_events);
            scratch_growths += static_cast<uint64_t>(output.debug.scratch_capacity_growth_events);
            max_depth = std::max(max_depth, output.debug.max_search_depth);
            checksum += output.action * 0.125;
            checksum += output.root_value * 3.0;
            checksum += output.debug.total_nodes * 0.001;
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();
        const uint64_t total_simulations = static_cast<uint64_t>(searches)
            * static_cast<uint64_t>(simulations);

        std::cout << std::fixed << std::setprecision(6)
                  << "{\"case\":\"mcts_repeated_search\""
                  << ",\"searches\":" << searches
                  << ",\"simulations_per_search\":" << simulations
                  << ",\"latent_top_k\":" << latent_top_k
                  << ",\"root_actions\":" << legal_actions.size()
                  << ",\"warmup_nodes_created\":" << warmup.debug.node_objects_created
                  << ",\"warmup_node_buffer_growths\":"
                  << warmup.debug.node_buffer_growth_events
                  << ",\"warmup_scratch_growths\":"
                  << warmup.debug.scratch_capacity_growth_events
                  << ",\"timed_nodes_created\":" << created_nodes
                  << ",\"timed_nodes_reused\":" << reused_nodes
                  << ",\"timed_node_buffer_growths\":" << node_buffer_growths
                  << ",\"timed_scratch_growths\":" << scratch_growths
                  << ",\"max_search_depth\":" << max_depth
                  << ",\"seconds\":" << seconds
                  << ",\"searches_per_second\":"
                  << (seconds > 0.0 ? searches / seconds : 0.0)
                  << ",\"simulations_per_second\":"
                  << (seconds > 0.0 ? total_simulations / seconds : 0.0)
                  << ",\"checksum\":" << checksum
                  << "}" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
