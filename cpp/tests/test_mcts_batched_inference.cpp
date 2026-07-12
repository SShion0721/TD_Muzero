#include "tdmz/core/action.hpp"
#include "tdmz/core/engine.hpp"
#include "tdmz/core/observation.hpp"
#include "tdmz/mcts/mcts.hpp"
#include "tdmz/mcts/network_evaluator.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tdmz;

namespace {

void check_true(bool condition, const char* expression, int line) {
    if (!condition) {
        throw std::runtime_error(
            std::string("check failed at line ") + std::to_string(line) + ": " + expression);
    }
}
#define CHECK_TRUE(expression) check_true(static_cast<bool>(expression), #expression, __LINE__)

template <typename Fn>
void check_invalid_argument(Fn&& function) {
    bool threw = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK_TRUE(threw);
}

class RecordingBatchNetwork final : public INetworkEvaluator {
public:
    int initial_calls = 0;
    int recurrent_calls = 0;
    int recurrent_samples = 0;
    int max_batch = 0;
    std::vector<int> batch_sizes;

protected:
    EvalOutput initial_inference_impl(
        const std::vector<std::vector<float>>& observations
    ) override {
        ++initial_calls;
        return make_output(static_cast<int>(observations.size()), nullptr);
    }

    EvalOutput recurrent_inference_impl(const EvalInput& input) override {
        ++recurrent_calls;
        recurrent_samples += input.batch_size;
        max_batch = std::max(max_batch, input.batch_size);
        batch_sizes.push_back(input.batch_size);

        CHECK_TRUE(input.batch_size > 0);
        CHECK_TRUE(input.parent_node_ids.size() == static_cast<size_t>(input.batch_size));
        CHECK_TRUE(input.target_node_ids.size() == static_cast<size_t>(input.batch_size));
        CHECK_TRUE(input.actions.size() == static_cast<size_t>(input.batch_size));
        const std::set<int> unique_targets(
            input.target_node_ids.begin(), input.target_node_ids.end());
        CHECK_TRUE(unique_targets.size() == input.target_node_ids.size());

        return make_output(input.batch_size, &input);
    }

private:
    EvalOutput make_output(int batch_size, const EvalInput* input) {
        EvalOutput output;
        output.values.resize(static_cast<size_t>(batch_size));
        output.rewards.resize(static_cast<size_t>(batch_size));
        output.policy_logits.resize(
            static_cast<size_t>(batch_size),
            std::vector<float>(kActionSpaceSize, -4.0f));
        output.legality_logits.resize(
            static_cast<size_t>(batch_size),
            std::vector<float>(kActionSpaceSize, 0.0f));

        for (int batch = 0; batch < batch_size; ++batch) {
            const int target = input ? input->target_node_ids[static_cast<size_t>(batch)] : 0;
            const int action = input ? input->actions[static_cast<size_t>(batch)] : 0;
            output.values[static_cast<size_t>(batch)]
                = 0.05f + static_cast<float>(target % 11) * 0.001f;
            output.rewards[static_cast<size_t>(batch)]
                = static_cast<float>((action + 3) % 7) * 0.002f;

            auto& logits = output.policy_logits[static_cast<size_t>(batch)];
            auto& legality = output.legality_logits[static_cast<size_t>(batch)];
            for (int candidate = 0; candidate < kActionSpaceSize; ++candidate) {
                logits[static_cast<size_t>(candidate)]
                    = static_cast<float>((candidate * 13 + target * 5 + action) % 101) * 0.01f;
                legality[static_cast<size_t>(candidate)]
                    = static_cast<float>((candidate * 7 + target + action) % 29) * 0.01f;
            }
        }
        return output;
    }
};

void check_outputs_equal(const RootSearchOutput& left, const RootSearchOutput& right) {
    CHECK_TRUE(left.action == right.action);
    CHECK_TRUE(std::fabs(left.root_value - right.root_value) <= 1e-7f);
    CHECK_TRUE(left.root_actions == right.root_actions);
    CHECK_TRUE(left.root_priors == right.root_priors);
    CHECK_TRUE(left.visit_counts == right.visit_counts);
    CHECK_TRUE(left.policy_full == right.policy_full);
    CHECK_TRUE(left.debug.total_nodes == right.debug.total_nodes);
    CHECK_TRUE(left.debug.total_edges == right.debug.total_edges);
    CHECK_TRUE(left.debug.recurrent_evaluator_calls == right.debug.recurrent_evaluator_calls);
    CHECK_TRUE(left.debug.recurrent_evaluator_samples == right.debug.recurrent_evaluator_samples);
    CHECK_TRUE(left.debug.max_recurrent_batch_size == right.debug.max_recurrent_batch_size);
    CHECK_TRUE(left.debug.leaf_batch_collision_stops == right.debug.leaf_batch_collision_stops);
}

int visit_sum(const RootSearchOutput& output) {
    int total = 0;
    for (int visits : output.visit_counts) total += visits;
    return total;
}

void test_default_batch_one_is_explicitly_stable() {
    TDEngine environment(11, 11, 71);
    const auto observation = make_observation_v1(environment);
    const auto legal_actions = environment.legal_actions();

    MCTSConfig implicit_config;
    implicit_config.num_simulations = 24;
    implicit_config.latent_top_k = 8;
    implicit_config.max_nodes = 2048;

    MCTSConfig explicit_config = implicit_config;
    explicit_config.recurrent_batch_size = 1;

    RecordingBatchNetwork first_network;
    RecordingBatchNetwork second_network;
    MCTS implicit_mcts(implicit_config);
    MCTS explicit_mcts(explicit_config);
    const RootSearchOutput implicit_output = implicit_mcts.search_single(
        first_network, observation, legal_actions);
    const RootSearchOutput explicit_output = explicit_mcts.search_single(
        second_network, observation, legal_actions);

    check_outputs_equal(implicit_output, explicit_output);
    CHECK_TRUE(implicit_output.debug.recurrent_evaluator_calls == implicit_config.num_simulations);
    CHECK_TRUE(implicit_output.debug.recurrent_evaluator_samples == implicit_config.num_simulations);
    CHECK_TRUE(implicit_output.debug.max_recurrent_batch_size == 1);
    CHECK_TRUE(implicit_output.debug.leaf_batch_collision_stops == 0);
    CHECK_TRUE(visit_sum(implicit_output) == implicit_config.num_simulations);
}

void test_wide_tree_batches_unique_leaves() {
    TDEngine environment(11, 11, 72);
    const auto observation = make_observation_v1(environment);
    const auto legal_actions = environment.legal_actions();

    MCTSConfig config;
    config.num_simulations = 48;
    config.latent_top_k = 16;
    config.max_nodes = 4096;
    config.recurrent_batch_size = 8;

    RecordingBatchNetwork network;
    MCTS mcts(config);
    const RootSearchOutput output = mcts.search_single(
        network, observation, legal_actions);

    CHECK_TRUE(network.initial_calls == 1);
    CHECK_TRUE(network.recurrent_samples == config.num_simulations);
    CHECK_TRUE(network.recurrent_calls == output.debug.recurrent_evaluator_calls);
    CHECK_TRUE(network.max_batch == output.debug.max_recurrent_batch_size);
    CHECK_TRUE(output.debug.recurrent_evaluator_samples == config.num_simulations);
    CHECK_TRUE(output.debug.recurrent_evaluator_calls < config.num_simulations);
    CHECK_TRUE(output.debug.max_recurrent_batch_size > 1);
    CHECK_TRUE(output.debug.max_recurrent_batch_size <= config.recurrent_batch_size);
    CHECK_TRUE(visit_sum(output) == config.num_simulations);
    CHECK_TRUE(output.debug.total_edges == output.debug.total_nodes - 1);
}

void test_narrow_tree_falls_back_without_duplicate_targets() {
    std::vector<float> observation(4, 0.0f);
    MCTSConfig config;
    config.num_simulations = 12;
    config.latent_top_k = 1;
    config.max_nodes = 14;
    config.recurrent_batch_size = 8;

    RecordingBatchNetwork network;
    MCTS mcts(config);
    const RootSearchOutput output = mcts.search_single(network, observation, {0});

    CHECK_TRUE(output.debug.recurrent_evaluator_samples == config.num_simulations);
    CHECK_TRUE(output.debug.recurrent_evaluator_calls == config.num_simulations);
    CHECK_TRUE(output.debug.max_recurrent_batch_size == 1);
    CHECK_TRUE(output.debug.leaf_batch_collision_stops > 0);
    CHECK_TRUE(visit_sum(output) == config.num_simulations);
}

void test_batched_schedule_is_deterministic() {
    TDEngine environment(11, 11, 73);
    const auto observation = make_observation_v1(environment);
    const auto legal_actions = environment.legal_actions();

    MCTSConfig config;
    config.num_simulations = 40;
    config.latent_top_k = 12;
    config.max_nodes = 4096;
    config.recurrent_batch_size = 6;

    RecordingBatchNetwork first_network;
    RecordingBatchNetwork second_network;
    MCTS first_mcts(config);
    MCTS second_mcts(config);
    const RootSearchOutput first = first_mcts.search_single(
        first_network, observation, legal_actions);
    const RootSearchOutput second = second_mcts.search_single(
        second_network, observation, legal_actions);
    check_outputs_equal(first, second);
    CHECK_TRUE(first_network.batch_sizes == second_network.batch_sizes);
}

void test_invalid_recurrent_batch_size_is_rejected() {
    MCTSConfig config;
    config.recurrent_batch_size = 0;
    check_invalid_argument([&] { MCTS invalid(config); });
}

} // namespace

int main() {
    try {
        test_default_batch_one_is_explicitly_stable();
        test_wide_tree_batches_unique_leaves();
        test_narrow_tree_falls_back_without_duplicate_targets();
        test_batched_schedule_is_deterministic();
        test_invalid_recurrent_batch_size_is_rejected();
        std::cout << "MCTS batched inference tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
