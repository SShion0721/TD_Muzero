#include "tdmz/core/engine.hpp"
#include "tdmz/core/observation.hpp"
#include "tdmz/mcts/mcts.hpp"
#include "tdmz/nn/libtorch_evaluator.hpp"
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

using namespace tdmz;

static void check_true(bool ok, const char* expr, int line) {
    if (!ok) throw std::runtime_error(std::string("check failed at line ") + std::to_string(line) + ": " + expr);
}
#define CHECK_TRUE(x) check_true(static_cast<bool>(x), #x, __LINE__)

int main() {
    TDEngine env(11, 11, 0);
    auto obs = make_observation_v1(env);
    auto legal = env.legal_actions();

    NetworkConfig net_cfg;
    MuZeroNetwork network(net_cfg);
    LibTorchEvaluator evaluator(network, torch::kCPU);

    MCTSConfig mcts_cfg;
    mcts_cfg.num_simulations = 8;
    mcts_cfg.latent_top_k = 8;
    mcts_cfg.max_nodes = 8192;

    MCTS mcts(mcts_cfg);
    auto out = mcts.search_single(evaluator, obs, legal);

    CHECK_TRUE(out.policy_full.size() == kActionSpaceSize);
    CHECK_TRUE(out.action != -1);
    CHECK_TRUE(std::set<int>(legal.begin(), legal.end()).count(out.action) == 1);
    CHECK_TRUE(out.debug.total_nodes > 0);
    CHECK_TRUE(out.debug.max_latent_branching <= mcts_cfg.latent_top_k);

    std::cout << "MCTS LibTorch smoke test passed!" << std::endl;
    return 0;
}
