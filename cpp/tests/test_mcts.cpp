#include "tdmz/core/engine.hpp"
#include "tdmz/core/observation.hpp"
#include "tdmz/mcts/mcts.hpp"
#include "tdmz/mcts/dummy_network.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tdmz;

static void check_true(bool ok, const char* expr, int line) {
    if (!ok) {
        throw std::runtime_error(std::string("check failed at line ") + std::to_string(line) + ": " + expr);
    }
}
#define CHECK_TRUE(x) check_true(static_cast<bool>(x), #x, __LINE__)

static void check_close(float actual, float expected, float tolerance, const char* label) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            std::string("float mismatch for ") + label
            + ": actual=" + std::to_string(actual)
            + " expected=" + std::to_string(expected));
    }
}

template <typename Fn>
static void check_throws_runtime(Fn&& fn, const char* label) {
    bool threw = false;
    try {
        fn();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    if (!threw) throw std::runtime_error(std::string("expected runtime_error: ") + label);
}

template <typename Fn>
static void check_throws_invalid(Fn&& fn, const char* label) {
    bool threw = false;
    try {
        fn();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    if (!threw) throw std::runtime_error(std::string("expected invalid_argument: ") + label);
}

static EvalOutput make_eval_output(float value = 0.0f, float reward = 0.0f) {
    return {{value}, {reward}, {std::vector<float>(kActionSpaceSize, 0.0f)}};
}

static RootSearchOutput run_search(MCTSConfig cfg) {
    TDEngine env(11, 11, 0);
    auto obs = make_observation_v1(env);
    auto legal = env.legal_actions();
    DummyNetwork net;
    MCTS mcts(cfg);
    return mcts.search_single(net, obs, legal);
}

static RootSearchOutput run_search(int sims, int topk) {
    MCTSConfig cfg;
    cfg.num_simulations = sims;
    cfg.latent_top_k = topk;
    cfg.max_nodes = 8192;
    return run_search(cfg);
}

class InvalidPolicyNetwork : public INetworkEvaluator {
public:
    EvalOutput initial_inference(const std::vector<std::vector<float>>&) override {
        return {{0.0f}, {0.0f}, {{0.0f}}};
    }

    EvalOutput recurrent_inference(const EvalInput&) override {
        return {{0.0f}, {0.0f}, {{0.0f}}};
    }
};

class CountingNetwork : public INetworkEvaluator {
public:
    EvalOutput initial_inference(const std::vector<std::vector<float>>&) override {
        ++initial_calls;
        return make_eval_output();
    }

    EvalOutput recurrent_inference(const EvalInput&) override {
        ++recurrent_calls;
        return make_eval_output();
    }

    int initial_calls = 0;
    int recurrent_calls = 0;
};

class FixedReturnNetwork : public INetworkEvaluator {
public:
    FixedReturnNetwork(float value, float reward) : value_(value), reward_(reward) {}

    EvalOutput initial_inference(const std::vector<std::vector<float>>&) override {
        return make_eval_output();
    }

    EvalOutput recurrent_inference(const EvalInput&) override {
        return make_eval_output(value_, reward_);
    }

private:
    float value_;
    float reward_;
};

enum class EvalStage { Initial, Recurrent };
enum class EvalField { Value, Reward, Policy };

class NonFiniteNetwork : public INetworkEvaluator {
public:
    NonFiniteNetwork(EvalStage stage, EvalField field, float bad_value)
        : stage_(stage), field_(field), bad_value_(bad_value) {}

    EvalOutput initial_inference(const std::vector<std::vector<float>>&) override {
        return make(stage_ == EvalStage::Initial);
    }

    EvalOutput recurrent_inference(const EvalInput&) override {
        return make(stage_ == EvalStage::Recurrent);
    }

private:
    EvalOutput make(bool inject) const {
        EvalOutput output = make_eval_output();
        if (!inject) return output;
        switch (field_) {
            case EvalField::Value:
                output.values[0] = bad_value_;
                break;
            case EvalField::Reward:
                output.rewards[0] = bad_value_;
                break;
            case EvalField::Policy:
                output.policy_logits[0][17] = bad_value_;
                break;
        }
        return output;
    }

    EvalStage stage_;
    EvalField field_;
    float bad_value_;
};

void test_node_pool() {
    check_throws_invalid([] { NodePool invalid(0); }, "zero node capacity");

    NodePool pool(2);
    CHECK_TRUE(pool.capacity() == 2);
    CHECK_TRUE(pool.remaining() == 2);
    CHECK_TRUE(pool.allocate() == 0);
    CHECK_TRUE(pool.remaining() == 1);
    CHECK_TRUE(pool.allocate() == 1);
    CHECK_TRUE(pool.remaining() == 0);
    check_throws_runtime([&] { (void)pool.allocate(); }, "node pool exhausted");

    pool.clear();
    CHECK_TRUE(pool.size() == 0);
    CHECK_TRUE(pool.remaining() == 2);
}

void test_root_policy_and_legality() {
    TDEngine env(11, 11, 0);
    auto legal = env.legal_actions();
    auto out = run_search(10, 5);
    CHECK_TRUE(out.root_actions == legal);
    CHECK_TRUE(out.root_priors.size() == legal.size());
    CHECK_TRUE(out.policy_full.size() == kActionSpaceSize);
    std::set<int> legal_set(legal.begin(), legal.end());
    float sum = 0.0f;
    for (int a = 0; a < kActionSpaceSize; ++a) {
        sum += out.policy_full[a];
        if (legal_set.find(a) == legal_set.end()) CHECK_TRUE(out.policy_full[a] == 0.0f);
    }
    CHECK_TRUE(std::abs(sum - 1.0f) < 1e-4f);
}

void test_visit_counts_and_debug() {
    MCTSConfig defaults;
    auto out = run_search(defaults.num_simulations, 5);
    int total = 0;
    for (int vc : out.visit_counts) total += vc;
    CHECK_TRUE(total == defaults.num_simulations);
    CHECK_TRUE(out.debug.total_nodes > 0);
    CHECK_TRUE(out.debug.max_root_branching == static_cast<int>(out.root_actions.size()));
    CHECK_TRUE(out.debug.max_latent_branching > 0);
    CHECK_TRUE(out.debug.max_latent_branching <= 5);
}

void test_best_action_stable_and_legal() {
    TDEngine env(11, 11, 0);
    auto legal = env.legal_actions();
    auto out1 = run_search(32, 8);
    auto out2 = run_search(32, 8);
    CHECK_TRUE(out1.action == out2.action);
    CHECK_TRUE(out1.action != -1);
    std::set<int> legal_set(legal.begin(), legal.end());
    CHECK_TRUE(legal_set.find(out1.action) != legal_set.end());
}

void test_root_noise_changes_priors_and_is_reproducible() {
    MCTSConfig base;
    base.num_simulations = 8;
    base.latent_top_k = 8;
    base.max_nodes = 8192;
    base.random_seed = 1234567;

    auto without_noise = run_search(base);

    MCTSConfig noisy = base;
    noisy.add_root_noise = true;
    noisy.root_dirichlet_alpha = 0.3f;
    noisy.root_noise_weight = 0.25f;
    auto with_noise_a = run_search(noisy);
    auto with_noise_b = run_search(noisy);

    CHECK_TRUE(with_noise_a.root_priors.size() == without_noise.root_priors.size());
    CHECK_TRUE(with_noise_a.root_priors == with_noise_b.root_priors);

    float prior_sum = std::accumulate(
        with_noise_a.root_priors.begin(), with_noise_a.root_priors.end(), 0.0f);
    CHECK_TRUE(std::abs(prior_sum - 1.0f) < 1e-4f);

    bool changed = false;
    for (size_t i = 0; i < with_noise_a.root_priors.size(); ++i) {
        if (std::abs(with_noise_a.root_priors[i] - without_noise.root_priors[i]) > 1e-6f) {
            changed = true;
            break;
        }
    }
    CHECK_TRUE(changed);
}

void test_multiplayer_mode_is_rejected() {
    MCTSConfig cfg;
    cfg.single_player = false;
    check_throws_invalid([&] { MCTS mcts(cfg); }, "multiplayer mode");
}

void test_invalid_network_output_is_rejected() {
    TDEngine env(11, 11, 0);
    InvalidPolicyNetwork net;
    MCTSConfig cfg;
    MCTS mcts(cfg);
    check_throws_runtime(
        [&] { (void)mcts.search_single(net, make_observation_v1(env), env.legal_actions()); },
        "invalid policy width");
}

void test_non_finite_initial_and_recurrent_outputs_are_rejected() {
    const float bad_values[] = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
    };
    const EvalStage stages[] = {EvalStage::Initial, EvalStage::Recurrent};
    const EvalField fields[] = {EvalField::Value, EvalField::Reward, EvalField::Policy};

    MCTSConfig cfg;
    cfg.num_simulations = 1;
    cfg.latent_top_k = 1;
    cfg.max_nodes = 3;

    for (EvalStage stage : stages) {
        for (EvalField field : fields) {
            for (float bad_value : bad_values) {
                NonFiniteNetwork net(stage, field, bad_value);
                MCTS mcts(cfg);
                check_throws_runtime(
                    [&] { (void)mcts.search_single(net, {}, {0}); },
                    "non-finite evaluator output");
            }
        }
    }
}

void test_node_capacity_is_checked_before_network_inference() {
    MCTSConfig cfg;
    cfg.num_simulations = 2;
    cfg.latent_top_k = 3;
    cfg.max_nodes = 8; // Exact requirement is 1 root + 2 root children + 2*3 latent children = 9.

    CountingNetwork too_small_net;
    MCTS too_small(cfg);
    check_throws_invalid(
        [&] { (void)too_small.search_single(too_small_net, {}, {0, 1}); },
        "insufficient MCTS max_nodes");
    CHECK_TRUE(too_small_net.initial_calls == 0);
    CHECK_TRUE(too_small_net.recurrent_calls == 0);

    cfg.max_nodes = 9;
    CountingNetwork exact_net;
    MCTS exact(cfg);
    RootSearchOutput output = exact.search_single(exact_net, {}, {0, 1});
    CHECK_TRUE(output.debug.total_nodes == 9);
    CHECK_TRUE(exact_net.initial_calls == 1);
    CHECK_TRUE(exact_net.recurrent_calls == 2);
}

void test_single_player_backup_preserves_reward_and_value_signs() {
    MCTSConfig cfg;
    cfg.num_simulations = 1;
    cfg.latent_top_k = 1;
    cfg.max_nodes = 3;
    cfg.discount = 0.8f;

    const float recurrent_value = -0.5f;
    const float transition_reward = 0.25f;
    FixedReturnNetwork net(recurrent_value, transition_reward);
    MCTS mcts(cfg);
    RootSearchOutput output = mcts.search_single(net, {}, {7});

    const float expected_root_value = transition_reward + cfg.discount * recurrent_value;
    check_close(output.root_value, expected_root_value, 1e-6f, "single-player backup");
    CHECK_TRUE(output.root_value < 0.0f);
    CHECK_TRUE(output.visit_counts.size() == 1);
    CHECK_TRUE(output.visit_counts[0] == 1);
}

void test_duplicate_root_actions_are_rejected() {
    TDEngine env(11, 11, 0);
    DummyNetwork net;
    MCTSConfig cfg;
    MCTS mcts(cfg);
    auto legal = env.legal_actions();
    legal.push_back(legal.front());

    check_throws_invalid(
        [&] { (void)mcts.search_single(net, make_observation_v1(env), legal); },
        "duplicate root actions");
}

void test_minmax() {
    MinMaxStats stats(0.01f);
    stats.update(1.0f);
    stats.update(3.0f);
    CHECK_TRUE(std::abs(stats.normalize(2.0f) - (1.0f / 2.01f)) < 1e-5f);
}

int main() {
    test_node_pool();
    test_root_policy_and_legality();
    test_visit_counts_and_debug();
    test_best_action_stable_and_legal();
    test_root_noise_changes_priors_and_is_reproducible();
    test_multiplayer_mode_is_rejected();
    test_invalid_network_output_is_rejected();
    test_non_finite_initial_and_recurrent_outputs_are_rejected();
    test_node_capacity_is_checked_before_network_inference();
    test_single_player_backup_preserves_reward_and_value_signs();
    test_duplicate_root_actions_are_rejected();
    test_minmax();
    std::cout << "All MCTS tests passed!" << std::endl;
    return 0;
}
