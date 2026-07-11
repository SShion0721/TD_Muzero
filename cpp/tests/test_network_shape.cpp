#include "tdmz/core/action.hpp"
#include "tdmz/core/observation.hpp"
#include "tdmz/nn/action_encoder.hpp"
#include "tdmz/nn/muzero_network.hpp"
#include "tdmz/nn/prediction.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

static void test_action_roundtrip_and_spatial_encoding() {
    NetworkConfig cfg;
    ActionEncoder encoder(cfg);

    std::vector<int> actions;
    actions.reserve(kActionSpaceSize);
    for (int id = 0; id < kActionSpaceSize; ++id) {
        const Action decoded = decode_action(id);
        CHECK_TRUE(encode_action(decoded) == id);
        actions.push_back(id);
    }

    auto encoded = encoder->forward(actions, torch::kCPU).contiguous();
    check_4d(encoded, kActionSpaceSize, kSpatialActionPlanes, kBoardH, kBoardW);
    auto acc = encoded.accessor<float, 4>();

    for (int id = 0; id < kActionSpaceSize; ++id) {
        const Action action = decode_action(id);
        const int expected_plane = static_cast<int>(action.type);
        float sum = 0.0f;
        for (int plane = 0; plane < kSpatialActionPlanes; ++plane) {
            for (int y = 0; y < kBoardH; ++y) {
                for (int x = 0; x < kBoardW; ++x) {
                    const float value = acc[id][plane][y][x];
                    sum += value;
                    const bool expected = action.type == ActionType::Wait1
                        ? plane == expected_plane
                        : plane == expected_plane && x == action.x && y == action.y;
                    CHECK_TRUE(value == (expected ? 1.0f : 0.0f));
                }
            }
        }
        CHECK_TRUE(sum == (action.type == ActionType::Wait1 ? static_cast<float>(kCells) : 1.0f));
    }
}

static void test_prediction_flatten_mapping() {
    auto spatial = torch::zeros({2, kSpatialPolicyPlanes, kBoardH, kBoardW});
    auto spatial_acc = spatial.accessor<float, 4>();
    for (int b = 0; b < 2; ++b) {
        for (int plane = 0; plane < kSpatialPolicyPlanes; ++plane) {
            for (int y = 0; y < kBoardH; ++y) {
                for (int x = 0; x < kBoardW; ++x) {
                    const int cell = y * kBoardW + x;
                    spatial_acc[b][plane][y][x] = static_cast<float>(
                        b * 1000 + plane * kCells + cell);
                }
            }
        }
    }
    auto wait = torch::tensor({{726.0f}, {1726.0f}});
    auto flattened = flatten_spatial_action_logits(spatial, wait).contiguous();
    check_2d(flattened, 2, kActionSpaceSize);
    auto flat_acc = flattened.accessor<float, 2>();
    for (int b = 0; b < 2; ++b) {
        for (int action = 0; action < kActionSpaceSize; ++action) {
            CHECK_TRUE(flat_acc[b][action] == static_cast<float>(b * 1000 + action));
        }
    }
}

static void test_network_shapes() {
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

    std::vector<int> actions = {
        kFlatWaitOffset,
        encode_action(Action{ActionType::BuildBasic, 1, 1, 1})
    };
    auto rec = net->recurrent_inference(init.latent_state, actions);

    check_4d(rec.latent_state, 2, cfg.latent_channels, cfg.board_h, cfg.board_w);
    check_2d(rec.value, 2, cfg.value_dim);
    check_2d(rec.reward, 2, cfg.reward_dim);
    check_2d(rec.policy_logits, 2, kActionSpaceSize);
    check_2d(rec.legality_logits, 2, kActionSpaceSize);

    CHECK_TRUE(torch::isfinite(init.policy_logits).all().item<bool>());
    CHECK_TRUE(torch::isfinite(rec.policy_logits).all().item<bool>());
}

int main() {
    test_action_roundtrip_and_spatial_encoding();
    test_prediction_flatten_mapping();
    test_network_shapes();
    std::cout << "Network shape tests passed!" << std::endl;
    return 0;
}
