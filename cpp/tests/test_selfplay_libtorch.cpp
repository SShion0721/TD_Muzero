#include "tdmz/nn/libtorch_evaluator.hpp"
#include "tdmz/selfplay/selfplay_runner.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace tdmz;

static void check_true(bool ok, const char* expr, int line) {
    if (!ok) throw std::runtime_error(std::string("check failed at line ") + std::to_string(line) + ": " + expr);
}
#define CHECK_TRUE(x) check_true(static_cast<bool>(x), #x, __LINE__)

int main() {
    SelfPlayConfig cfg;
    cfg.seed = 0;
    cfg.max_steps = 2;
    cfg.mcts.num_simulations = 4;
    cfg.mcts.latent_top_k = 4;
    cfg.mcts.max_nodes = 2048;

    NetworkConfig net_cfg;
    MuZeroNetwork network(net_cfg);
    LibTorchEvaluator evaluator(network, torch::kCPU);

    SelfPlayRunner runner(cfg);
    auto history = runner.run(evaluator);

    CHECK_TRUE(!history.steps.empty());
    CHECK_TRUE(history.steps.size() <= static_cast<size_t>(cfg.max_steps));
    for (const auto& step : history.steps) {
        CHECK_TRUE(step.action >= 0);
        CHECK_TRUE(step.action < kActionSpaceSize);
        CHECK_TRUE(step.policy_target.size() == kActionSpaceSize);
        CHECK_TRUE(step.legal_mask.size() == kActionSpaceSize);
        CHECK_TRUE(step.legal_mask[step.action] == 1);
    }

    std::cout << "LibTorch self-play test passed!" << std::endl;
    return 0;
}
