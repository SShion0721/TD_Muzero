#include "tdmz/core/engine.hpp"
#include "tdmz/core/observation.hpp"
#include "tdmz/mcts/mcts.hpp"
#include "tdmz/mcts/dummy_network.hpp"
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

using namespace tdmz;

static void check_true(bool ok, const char* expr, int line) {
    if (!ok) {
        throw std::runtime_error(std::string("check failed at line ") + std::to_string(line) + ": " + expr);
    }
}
#define CHECK_TRUE(x) check_true(static_cast<bool>(x), #x, __LINE__)

static RootSearchOutput run_search(int sims, int topk) {
    TDEngine env(11, 11, 0);
    auto obs = make_observation_v1(env);
    auto legal = env.legal_actions();
    MCTSConfig cfg;
    cfg.num_simulations = sims;
    cfg.latent_top_k = topk;
    cfg.max_nodes = 8192;
    DummyNetwork net;
    MCTS mcts(cfg);
    return mcts.search_single(net, obs, legal);
}

void test_node_pool() {
    NodePool pool(2);
    CHECK_TRUE(pool.allocate() == 0);
    CHECK_TRUE(pool.allocate() == 1);
    bool threw = false;
    try { pool.allocate(); } catch (const std::runtime_error&) { threw = true; }
    CHECK_TRUE(threw);
}

void test_root_policy_and_legality() {
    TDEngine env(11, 11, 0);
    auto legal = env.legal_actions();
    auto out = run_search(10, 5);
    CHECK_TRUE(out.root_actions == legal);
    CHECK_TRUE(out.policy_full.size() == kActionSpaceSize);
    std::set<int> legal_set(legal.begin(), legal.end());
    float sum = 0.0f;
    for (int a = 0; a < kActionSpaceSize; ++a) {
        sum += out.policy_full[a];
        if (legal_set.find(a) == legal_set.end()) CHECK_TRUE(out.policy_full[a] == 0.0f);
    }
    CHECK_TRUE(std::abs(sum - 1.0f) < 1e-4f);
}

void test_visit_counts_and_debug() {
    MCTSConfig defaults;
    auto out = run_search(defaults.num_simulations, 5);
    int total = 0;
    for (int vc : out.visit_counts) total += vc;
    CHECK_TRUE(total == defaults.num_simulations);
    CHECK_TRUE(out.debug.total_nodes > 0);
    CHECK_TRUE(out.debug.max_root_branching == static_cast<int>(out.root_actions.size()));
    CHECK_TRUE(out.debug.max_latent_branching > 0);
    CHECK_TRUE(out.debug.max_latent_branching <= 5);
}

void test_best_action_stable_and_legal() {
    TDEngine env(11, 11, 0);
    auto legal = env.legal_actions();
    auto out1 = run_search(32, 8);
    auto out2 = run_search(32, 8);
    CHECK_TRUE(out1.action == out2.action);
    CHECK_TRUE(out1.action != -1);
    std::set<int> legal_set(legal.begin(), legal.end());
    CHECK_TRUE(legal_set.find(out1.action) != legal_set.end());
}

void test_minmax() {
    MinMaxStats stats(0.01f);
    stats.update(1.0f);
    stats.update(3.0f);
    CHECK_TRUE(std::abs(stats.normalize(2.0f) - (1.0f / 2.01f)) < 1e-5f);
}

int main() {
    test_node_pool();
    test_root_policy_and_legality();
    test_visit_counts_and_debug();
    test_best_action_stable_and_legal();
    test_minmax();
    std::cout << "All MCTS tests passed!" << std::endl;
    return 0;
}
