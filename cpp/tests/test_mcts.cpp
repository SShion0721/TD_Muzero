#include "tdmz/core/engine.hpp"
#include "tdmz/core/observation.hpp"
#include "tdmz/mcts/mcts.hpp"
#include "tdmz/mcts/dummy_network.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace tdmz;

void test_node_pool_allocate() {
    NodePool pool;
    int id1 = pool.allocate();
    int id2 = pool.allocate();
    assert(id1 == 0 && id2 == 1);
    assert(pool.size() == 2);
}

void test_mcts_root_expansion() {
    TDEngine env(11, 11, 0);
    auto obs = make_observation_v1(env);
    auto legal = env.legal_actions();

    MCTSConfig cfg;
    cfg.num_simulations = 10;
    cfg.latent_top_k = 5;

    DummyNetwork net;
    MCTS mcts(cfg);

    auto out = mcts.search_single(net, obs, legal);
    
    // test_policy_full_size_727
    assert(out.policy_full.size() == kActionSpaceSize);
    
    // sum(policy_full) ≈ 1
    float sum = 0.0f;
    for (float p : out.policy_full) sum += p;
    assert(std::abs(sum - 1.0f) < 1e-4f);

    // root_actions should exactly match legal
    assert(out.root_actions == legal);

    // test_visit_count_sum
    int total_vc = 0;
    for (int vc : out.visit_counts) total_vc += vc;
    assert(total_vc == cfg.num_simulations);
    
    // test_best_action_deterministic
    // Because DummyNetwork outputs 1.0f for kFlatWaitOffset and 0.2f for a%17==0, 
    // it's deterministic which action gets most visits
    assert(out.action != -1);
}

void test_minmax() {
    MinMaxStats stats(0.01f);
    stats.update(1.0f);
    stats.update(3.0f);
    
    float norm = stats.normalize(2.0f);
    // minimum=1.0, maximum=3.0, delta=0.01 -> max-min+delta = 2.01
    // (2.0 - 1.0) / 2.01 = 1.0 / 2.01 ≈ 0.4975
    assert(std::abs(norm - (1.0f / 2.01f)) < 1e-5f);
}

int main() {
    test_node_pool_allocate();
    test_mcts_root_expansion();
    test_minmax();
    
    std::cout << "All MCTS tests passed!" << std::endl;
    return 0;
}
