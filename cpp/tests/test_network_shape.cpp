#include "tdmz/core/action.hpp"
#include "tdmz/core/observation.hpp"
#include "tdmz/nn/muzero_network.hpp"
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
    NetworkConfig cfg;
    MuZeroNetwork net(cfg);
    net->eval();

    auto obs = torch::zeros({2, cfg.observation_channels, cfg.board_h, cfg.board_w});
    auto init = net->initial_inference(obs);

    CHECK_TRUE(init.latent_state.sizes() == torch::IntArrayRef({2, cfg.latent_channels, cfg.board_h, cfg.board_w}));
    CHECK_TRUE(init.value.sizes() == torch::IntArrayRef({2, cfg.value_dim}));
    CHECK_TRUE(init.reward.sizes() == torch::IntArrayRef({2, cfg.reward_dim}));
    CHECK_TRUE(init.policy_logits.sizes() == torch::IntArrayRef({2, kActionSpaceSize}));
    CHECK_TRUE(init.legality_logits.sizes() == torch::IntArrayRef({2, kActionSpaceSize}));

    std::vector<int> actions = {kFlatWaitOffset, encode_action(Action{ActionType::BuildBasic, 1, 1, 1})};
    auto rec = net->recurrent_inference(init.latent_state, actions);

    CHECK_TRUE(rec.latent_state.sizes() == torch::IntArrayRef({2, cfg.latent_channels, cfg.board_h, cfg.board_w}));
    CHECK_TRUE(rec.value.sizes() == torch::IntArrayRef({2, cfg.value_dim}));
    CHECK_TRUE(rec.reward.sizes() == torch::IntArrayRef({2, cfg.reward_dim}));
    CHECK_TRUE(rec.policy_logits.sizes() == torch::IntArrayRef({2, kActionSpaceSize}));
    CHECK_TRUE(rec.legality_logits.sizes() == torch::IntArrayRef({2, kActionSpaceSize}));

    std::cout << "Network shape tests passed!" << std::endl;
    return 0;
}
