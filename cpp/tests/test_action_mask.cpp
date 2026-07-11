#include "tdmz/core/action.hpp"
#include "tdmz/core/engine.hpp"
#include "tdmz/core/observation.hpp"
#include "tdmz/mcts/action_mask.hpp"
#include "tdmz/mcts/dummy_network.hpp"
#include "tdmz/mcts/mcts.hpp"
#include <cmath>
#include <exception>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tdmz;

namespace {

void check_true(bool ok, const char* expr, int line) {
    if (!ok) {
        throw std::runtime_error(
            std::string("check failed at line ") + std::to_string(line) + ": " + expr
        );
    }
}

#define CHECK_TRUE(x) check_true(static_cast<bool>(x), #x, __LINE__)

void test_mask_to_actions_is_sorted_and_exact() {
    std::vector<uint8_t> mask(kActionSpaceSize, 0u);
    mask[3] = 1u;
    mask[17] = 1u;
    mask[kFlatWaitOffset] = 1u;

    auto actions = legal_actions_from_mask(mask);

    CHECK_TRUE(actions.size() == 3);
    CHECK_TRUE(actions[0] == 3);
    CHECK_TRUE(actions[1] == 17);
    CHECK_TRUE(actions[2] == kFlatWaitOffset);
    CHECK_TRUE(count_legal_actions(mask) == 3);
}

void test_masked_softmax_zeroes_invalid_actions() {
    std::vector<float> logits(kActionSpaceSize, 1000.0f);
    std::vector<uint8_t> mask(kActionSpaceSize, 0u);
    mask[5] = 1u;
    mask[9] = 1u;
    logits[5] = 1.0f;
    logits[9] = 2.0f;

    auto probabilities = masked_softmax(logits, mask);

    CHECK_TRUE(probabilities.size() == logits.size());
    float sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0f);
    CHECK_TRUE(std::fabs(sum - 1.0f) < 1e-6f);
    CHECK_TRUE(probabilities[5] > 0.0f);
    CHECK_TRUE(probabilities[9] > probabilities[5]);
    for (int action = 0; action < kActionSpaceSize; ++action) {
        if (mask[static_cast<size_t>(action)] == 0u) {
            CHECK_TRUE(probabilities[static_cast<size_t>(action)] == 0.0f);
        }
    }
}

void test_mcts_mask_api_never_expands_invalid_root_actions() {
    TDEngine env(11, 11, 0);
    auto observation = make_observation_v1(env);
    auto mask = env.legal_action_mask();

    MCTSConfig cfg;
    cfg.num_simulations = 16;
    cfg.latent_top_k = 8;
    cfg.max_nodes = 8192;
    DummyNetwork network;
    MCTS mcts(cfg);

    auto output = mcts.search_single_masked(network, observation, mask);

    CHECK_TRUE(output.action >= 0);
    CHECK_TRUE(mask[static_cast<size_t>(output.action)] == 1u);
    CHECK_TRUE(output.policy_full.size() == static_cast<size_t>(kActionSpaceSize));
    for (int action = 0; action < kActionSpaceSize; ++action) {
        if (mask[static_cast<size_t>(action)] == 0u) {
            CHECK_TRUE(output.policy_full[static_cast<size_t>(action)] == 0.0f);
        }
    }
    for (int action : output.root_actions) {
        CHECK_TRUE(action >= 0 && action < kActionSpaceSize);
        CHECK_TRUE(mask[static_cast<size_t>(action)] == 1u);
    }
}

void test_invalid_masks_are_rejected() {
    bool empty_threw = false;
    try {
        (void)legal_actions_from_mask({});
    } catch (const std::invalid_argument&) {
        empty_threw = true;
    }
    CHECK_TRUE(empty_threw);

    bool no_legal_threw = false;
    try {
        (void)masked_softmax({0.0f, 0.0f}, {0u, 0u});
    } catch (const std::invalid_argument&) {
        no_legal_threw = true;
    }
    CHECK_TRUE(no_legal_threw);

    bool bad_value_threw = false;
    try {
        (void)legal_actions_from_mask({1u, 2u});
    } catch (const std::invalid_argument&) {
        bad_value_threw = true;
    }
    CHECK_TRUE(bad_value_threw);
}

} // namespace

int main() {
    try {
        test_mask_to_actions_is_sorted_and_exact();
        test_masked_softmax_zeroes_invalid_actions();
        test_mcts_mask_api_never_expands_invalid_root_actions();
        test_invalid_masks_are_rejected();
        std::cout << "Action mask tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
