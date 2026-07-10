#include "tdmz/core/engine.hpp"
#include "tdmz/core/observation.hpp"
#include "tdmz/mcts/dummy_network.hpp"
#include "tdmz/mcts/edge_pool.hpp"
#include "tdmz/mcts/mcts.hpp"
#include "tdmz/mcts/node_pool.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace tdmz;

namespace {

void check_true(bool condition, const char* expression, int line) {
    if (!condition) {
        throw std::runtime_error(
            std::string("check failed at line ") + std::to_string(line) + ": " + expression);
    }
}
#define CHECK_TRUE(expression) check_true(static_cast<bool>(expression), #expression, __LINE__)

template <typename Fn>
void check_throws_out_of_range(Fn&& function) {
    bool threw = false;
    try {
        function();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK_TRUE(threw);
}

void test_node_pool_retains_node_objects() {
    NodePool pool(4);
    const int first_id = pool.allocate();
    CHECK_TRUE(first_id == 0);
    pool.get(first_id).parent = 17;
    pool.get(first_id).first_edge = 5;
    pool.get(first_id).edge_count = 3;

    CHECK_TRUE(pool.new_nodes_this_search() == 1);
    CHECK_TRUE(pool.reused_nodes_this_search() == 0);
    CHECK_TRUE(pool.retained_node_count() == 1);

    pool.clear();
    CHECK_TRUE(pool.size() == 0);
    CHECK_TRUE(pool.remaining() == 4);
    CHECK_TRUE(pool.retained_node_count() == 1);
    check_throws_out_of_range([&] { (void)pool.get(0); });

    const int reused_id = pool.allocate();
    CHECK_TRUE(reused_id == 0);
    CHECK_TRUE(pool.new_nodes_this_search() == 0);
    CHECK_TRUE(pool.reused_nodes_this_search() == 1);
    CHECK_TRUE(pool.get(0).parent == -1);
    CHECK_TRUE(pool.get(0).first_edge == -1);
    CHECK_TRUE(pool.get(0).edge_count == 0);
}

void test_edge_pool_allocates_contiguous_reusable_ranges() {
    EdgePool pool(8);
    const int first = pool.allocate_range(3);
    const int second = pool.allocate_range(2);
    CHECK_TRUE(first == 0);
    CHECK_TRUE(second == 3);
    CHECK_TRUE(pool.size() == 5);
    CHECK_TRUE(pool.created_edges_this_search() == 5);
    CHECK_TRUE(pool.reused_edges_this_search() == 0);
    pool.get(0).action = 7;
    pool.get(0).child = 11;

    pool.clear();
    CHECK_TRUE(pool.size() == 0);
    CHECK_TRUE(pool.retained_edge_count() == 5);
    check_throws_out_of_range([&] { (void)pool.get(0); });

    const int reused = pool.allocate_range(5);
    CHECK_TRUE(reused == 0);
    CHECK_TRUE(pool.created_edges_this_search() == 0);
    CHECK_TRUE(pool.reused_edges_this_search() == 5);
    CHECK_TRUE(pool.get(0).action == 7);
    CHECK_TRUE(pool.get(0).child == 11);
}

void check_search_outputs_equal(
    const RootSearchOutput& first,
    const RootSearchOutput& second
) {
    CHECK_TRUE(first.action == second.action);
    CHECK_TRUE(std::fabs(first.root_value - second.root_value) <= 1e-7f);
    CHECK_TRUE(first.root_actions == second.root_actions);
    CHECK_TRUE(first.root_priors == second.root_priors);
    CHECK_TRUE(first.visit_counts == second.visit_counts);
    CHECK_TRUE(first.policy_full == second.policy_full);
    CHECK_TRUE(first.debug.total_nodes == second.debug.total_nodes);
    CHECK_TRUE(first.debug.total_edges == second.debug.total_edges);
    CHECK_TRUE(first.debug.max_root_branching == second.debug.max_root_branching);
    CHECK_TRUE(first.debug.max_latent_branching == second.debug.max_latent_branching);
    CHECK_TRUE(first.debug.max_search_depth == second.debug.max_search_depth);
}

void test_repeated_search_reuses_all_retained_storage() {
    TDEngine environment(11, 11, 42);
    const auto observation = make_observation_v1(environment);
    const auto legal_actions = environment.legal_actions();

    MCTSConfig config;
    config.num_simulations = 64;
    config.latent_top_k = 16;
    config.max_nodes = 4096;
    config.add_root_noise = false;

    DummyNetwork network;
    MCTS mcts(config);

    const RootSearchOutput first = mcts.search_single(
        network, observation, legal_actions);
    const RootSearchOutput second = mcts.search_single(
        network, observation, legal_actions);
    check_search_outputs_equal(first, second);

    CHECK_TRUE(first.debug.total_edges == first.debug.total_nodes - 1);
    CHECK_TRUE(first.debug.node_objects_created == first.debug.total_nodes);
    CHECK_TRUE(first.debug.node_objects_reused == 0);
    CHECK_TRUE(first.debug.edge_objects_created == first.debug.total_edges);
    CHECK_TRUE(first.debug.edge_objects_reused == 0);
    CHECK_TRUE(first.debug.scratch_capacity_growth_events <= 1);

    CHECK_TRUE(second.debug.node_objects_created == 0);
    CHECK_TRUE(second.debug.node_objects_reused == second.debug.total_nodes);
    CHECK_TRUE(second.debug.edge_objects_created == 0);
    CHECK_TRUE(second.debug.edge_objects_reused == second.debug.total_edges);
    CHECK_TRUE(second.debug.scratch_capacity_growth_events == 0);
    CHECK_TRUE(second.debug.max_search_depth > 0);
}

void test_larger_observation_only_grows_scratch_once() {
    MCTSConfig config;
    config.num_simulations = 1;
    config.latent_top_k = 1;
    config.max_nodes = 3;

    DummyNetwork network;
    MCTS mcts(config);
    std::vector<float> first_observation(4, 0.0f);
    std::vector<float> larger_observation(64, 0.0f);

    const RootSearchOutput first = mcts.search_single(network, first_observation, {0});
    const RootSearchOutput larger = mcts.search_single(network, larger_observation, {0});
    const RootSearchOutput repeated = mcts.search_single(network, larger_observation, {0});

    CHECK_TRUE(first.debug.scratch_capacity_growth_events == 1);
    CHECK_TRUE(larger.debug.scratch_capacity_growth_events == 1);
    CHECK_TRUE(repeated.debug.scratch_capacity_growth_events == 0);
    CHECK_TRUE(repeated.debug.node_objects_created == 0);
    CHECK_TRUE(repeated.debug.edge_objects_created == 0);
    CHECK_TRUE(repeated.debug.total_edges == repeated.debug.total_nodes - 1);
}

} // namespace

int main() {
    try {
        test_node_pool_retains_node_objects();
        test_edge_pool_allocates_contiguous_reusable_ranges();
        test_repeated_search_reuses_all_retained_storage();
        test_larger_observation_only_grows_scratch_once();
        std::cout << "MCTS contiguous edge reuse tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
