#include "tdmz/core/engine.hpp"
#include "tdmz/core/observation.hpp"
#include "tdmz/mcts/mcts.hpp"
#include "tdmz/nn/libtorch_evaluator.hpp"
#include <cmath>
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

    bool root_batch_rejected = false;
    try {
        (void)evaluator.initial_inference({obs, obs});
    } catch (const std::invalid_argument&) {
        root_batch_rejected = true;
    }
    CHECK_TRUE(root_batch_rejected);

    const EvalOutput initial = evaluator.initial_inference({obs});
    CHECK_TRUE(initial.values.size() == 1);
    CHECK_TRUE(initial.rewards.size() == 1);
    CHECK_TRUE(initial.policy_logits.size() == 1);
    CHECK_TRUE(initial.legality_logits.size() == 1);
    CHECK_TRUE(initial.policy_logits[0].size() == kActionSpaceSize);
    CHECK_TRUE(initial.legality_logits[0].size() == kActionSpaceSize);
    for (float logit : initial.legality_logits[0]) CHECK_TRUE(std::isfinite(logit));

    MCTSConfig mcts_cfg;
    mcts_cfg.num_simulations = 8;
    mcts_cfg.latent_top_k = 8;
    mcts_cfg.max_nodes = 8192;

    MCTS mcts(mcts_cfg);
    auto out = mcts.search_single(evaluator, obs, legal);

    CHECK_TRUE(out.policy_full.size() == kActionSpaceSize);
    CHECK_TRUE(out.root_priors.size() == out.root_actions.size());
    CHECK_TRUE(out.action != -1);
    CHECK_TRUE(std::set<int>(legal.begin(), legal.end()).count(out.action) == 1);
    CHECK_TRUE(out.debug.total_nodes > 0);
    CHECK_TRUE(out.debug.max_latent_branching <= mcts_cfg.latent_top_k);

    EvalInput invalid_input;
    invalid_input.batch_size = 1;
    invalid_input.parent_node_ids = {0};
    invalid_input.target_node_ids = {1};
    bool recurrent_shape_rejected = false;
    try {
        (void)evaluator.recurrent_inference(invalid_input);
    } catch (const std::invalid_argument&) {
        recurrent_shape_rejected = true;
    }
    CHECK_TRUE(recurrent_shape_rejected);

    std::cout << "MCTS LibTorch smoke test passed!" << std::endl;
    return 0;
}
