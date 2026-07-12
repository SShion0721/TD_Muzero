#include "tdmz/core/action.hpp"
#include "tdmz/core/engine.hpp"
#include "tdmz/core/observation.hpp"
#include "tdmz/mcts/mcts.hpp"
#include "tdmz/mcts/network_evaluator.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tdmz;

namespace {

void check(bool condition, const char* expression, int line) {
    if (!condition) {
        throw std::runtime_error(
            std::string("check failed at line ") + std::to_string(line) + ": " + expression);
    }
}
#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

class PatternLegalityNetwork final : public INetworkEvaluator {
public:
    explicit PatternLegalityNetwork(bool invert_legality)
        : invert_legality_(invert_legality) {}

protected:
    EvalOutput initial_inference_impl(
        const std::vector<std::vector<float>>& observations
    ) override {
        return make_output(static_cast<int>(observations.size()));
    }

    EvalOutput recurrent_inference_impl(const EvalInput& input) override {
        return make_output(input.batch_size);
    }

private:
    EvalOutput make_output(int batch_size) const {
        EvalOutput output;
        output.values.assign(static_cast<size_t>(batch_size), 0.125f);
        output.rewards.assign(static_cast<size_t>(batch_size), -0.025f);
        output.policy_logits.resize(
            static_cast<size_t>(batch_size),
            std::vector<float>(kActionSpaceSize, 0.0f));
        output.legality_logits.resize(
            static_cast<size_t>(batch_size),
            std::vector<float>(kActionSpaceSize, 0.0f));

        for (int batch = 0; batch < batch_size; ++batch) {
            for (int action = 0; action < kActionSpaceSize; ++action) {
                output.policy_logits[static_cast<size_t>(batch)][static_cast<size_t>(action)]
                    = static_cast<float>((action * 17 + batch * 5) % 97) * 0.01f;
                const float legality = static_cast<float>((action * 11 + batch) % 53) * 0.02f - 0.5f;
                output.legality_logits[static_cast<size_t>(batch)][static_cast<size_t>(action)]
                    = invert_legality_ ? -legality : legality;
            }
        }
        return output;
    }

    bool invert_legality_;
};

void check_search_equal(const RootSearchOutput& left, const RootSearchOutput& right) {
    CHECK(left.action == right.action);
    CHECK(std::fabs(left.root_value - right.root_value) <= 1e-7f);
    CHECK(left.root_actions == right.root_actions);
    CHECK(left.root_priors == right.root_priors);
    CHECK(left.visit_counts == right.visit_counts);
    CHECK(left.policy_full == right.policy_full);
    CHECK(left.debug.total_nodes == right.debug.total_nodes);
    CHECK(left.debug.total_edges == right.debug.total_edges);
    CHECK(left.debug.max_latent_branching == right.debug.max_latent_branching);
    CHECK(left.debug.recurrent_evaluator_calls == right.debug.recurrent_evaluator_calls);
    CHECK(left.debug.recurrent_evaluator_samples == right.debug.recurrent_evaluator_samples);
}

void test_evaluator_exposes_legality_logits() {
    PatternLegalityNetwork network(false);
    TDEngine environment(11, 11, 123);
    const auto observation = make_observation_v2(environment);
    const EvalOutput output = network.initial_inference({observation});

    CHECK(output.values.size() == 1);
    CHECK(output.rewards.size() == 1);
    CHECK(output.policy_logits.size() == 1);
    CHECK(output.legality_logits.size() == 1);
    CHECK(output.policy_logits[0].size() == kActionSpaceSize);
    CHECK(output.legality_logits[0].size() == kActionSpaceSize);
    for (float logit : output.legality_logits[0]) CHECK(std::isfinite(logit));
}

void test_legality_is_transport_only_and_does_not_change_search() {
    TDEngine environment(11, 11, 321);
    const auto observation = make_observation_v2(environment);
    const auto legal_actions = environment.legal_actions();

    MCTSConfig config;
    config.num_simulations = 32;
    config.latent_top_k = 12;
    config.max_nodes = 4096;
    config.recurrent_batch_size = 4;
    config.add_root_noise = false;

    PatternLegalityNetwork positive(false);
    PatternLegalityNetwork inverted(true);
    MCTS first(config);
    MCTS second(config);
    const RootSearchOutput first_output = first.search_single(
        positive, observation, legal_actions);
    const RootSearchOutput second_output = second.search_single(
        inverted, observation, legal_actions);

    check_search_equal(first_output, second_output);
}

} // namespace

int main() {
    try {
        test_evaluator_exposes_legality_logits();
        test_legality_is_transport_only_and_does_not_change_search();
        std::cout << "Legality transport tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
