#include "tdmz/core/action.hpp"
#include "tdmz/core/observation.hpp"
#include "tdmz/nn/muzero_network.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

using namespace tdmz;

static void check_true(bool ok, const char* expr, int line) {
    if (!ok) throw std::runtime_error(std::string("check failed at line ") + std::to_string(line) + ": " + expr);
}
#define CHECK_TRUE(x) check_true(static_cast<bool>(x), #x, __LINE__)

static void check_4d(const torch::Tensor& t, int b, int c, int h, int w) {
    CHECK_TRUE(t.dim() == 4);
    CHECK_TRUE(t.size(0) == b);
    CHECK_TRUE(t.size(1) == c);
    CHECK_TRUE(t.size(2) == h);
    CHECK_TRUE(t.size(3) == w);
}

static void check_2d(const torch::Tensor& t, int b, int d) {
    CHECK_TRUE(t.dim() == 2);
    CHECK_TRUE(t.size(0) == b);
    CHECK_TRUE(t.size(1) == d);
}

int main() {
    NetworkConfig cfg;
    MuZeroNetwork net(cfg);
    net->eval();

    auto obs = torch::zeros({2, cfg.observation_channels, cfg.board_h, cfg.board_w});
    auto init = net->initial_inference(obs);

    check_4d(init.latent_state, 2, cfg.latent_channels, cfg.board_h, cfg.board_w);
    check_2d(init.value, 2, cfg.value_dim);
    check_2d(init.reward, 2, cfg.reward_dim);
    check_2d(init.policy_logits, 2, kActionSpaceSize);
    check_2d(init.legality_logits, 2, kActionSpaceSize);

    std::vector<int> actions = {kFlatWaitOffset, encode_action(Action{ActionType::BuildBasic, 1, 1, 1})};
    auto rec = net->recurrent_inference(init.latent_state, actions);

    check_4d(rec.latent_state, 2, cfg.latent_channels, cfg.board_h, cfg.board_w);
    check_2d(rec.value, 2, cfg.value_dim);
    check_2d(rec.reward, 2, cfg.reward_dim);
    check_2d(rec.policy_logits, 2, kActionSpaceSize);
    check_2d(rec.legality_logits, 2, kActionSpaceSize);

    std::cout << "Network shape tests passed!" << std::endl;
    return 0;
}
