#include "tdmz/mcts/mcts.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace tdmz {

MCTS::MCTS(MCTSConfig cfg) : cfg_(cfg), rng_(cfg.random_seed) {
    if (cfg_.num_simulations <= 0) {
        throw std::invalid_argument("MCTS num_simulations must be positive");
    }
    if (!std::isfinite(cfg_.discount) || cfg_.discount < 0.0f || cfg_.discount > 1.0f) {
        throw std::invalid_argument("MCTS discount must be finite and in [0, 1]");
    }
    if (!std::isfinite(cfg_.pb_c_base) || cfg_.pb_c_base <= 0.0f) {
        throw std::invalid_argument("MCTS pb_c_base must be finite and positive");
    }
    if (!std::isfinite(cfg_.pb_c_init) || cfg_.pb_c_init < 0.0f) {
        throw std::invalid_argument("MCTS pb_c_init must be finite and non-negative");
    }
    if (cfg_.latent_top_k <= 0) {
        throw std::invalid_argument("MCTS latent_top_k must be positive");
    }
    if (cfg_.max_nodes <= 1) {
        throw std::invalid_argument("MCTS max_nodes must be greater than one");
    }
    if (!std::isfinite(cfg_.value_delta_max) || cfg_.value_delta_max < 0.0f) {
        throw std::invalid_argument("MCTS value_delta_max must be finite and non-negative");
    }
    if (!cfg_.single_player) {
        throw std::invalid_argument("MCTS multiplayer backup semantics are not implemented");
    }
    if (cfg_.add_root_noise) {
        if (!std::isfinite(cfg_.root_dirichlet_alpha) || cfg_.root_dirichlet_alpha <= 0.0f) {
            throw std::invalid_argument("MCTS root_dirichlet_alpha must be finite and positive");
        }
        if (!std::isfinite(cfg_.root_noise_weight) ||
            cfg_.root_noise_weight < 0.0f || cfg_.root_noise_weight > 1.0f) {
            throw std::invalid_argument("MCTS root_noise_weight must be finite and in [0, 1]");
        }
    }
}

RootSearchOutput MCTS::search_single(
    INetworkEvaluator& net,
    const std::vector<float>& observation,
    const std::vector<int>& legal_actions
) {
    if (legal_actions.empty()) {
        throw std::invalid_argument("MCTS requires at least one legal root action");
    }

    std::vector<uint8_t> seen(kActionSpaceSize, 0);
    for (int action : legal_actions) {
        if (action < 0 || action >= kActionSpaceSize) {
            throw std::invalid_argument("MCTS legal root action is out of range");
        }
        if (seen[action]) {
            throw std::invalid_argument("MCTS legal root actions contain duplicates");
        }
        seen[action] = 1;
    }

    NodePool pool(cfg_.max_nodes);
    MinMaxStats minmax(cfg_.value_delta_max);

    int root_id = pool.allocate();

    auto init_out = net.initial_inference({observation});
    validate_eval_output(init_out, 1, "initial inference");
    expand_node(pool, root_id, -1, -1, 0.0f, 1.0f, legal_actions, init_out.policy_logits[0]);
    if (cfg_.add_root_noise && cfg_.root_noise_weight > 0.0f) {
        apply_root_noise(pool, root_id);
    }

    SearchDebugStats debug;
    debug.max_root_branching = static_cast<int>(legal_actions.size());

    for (int sim = 0; sim < cfg_.num_simulations; ++sim) {
        int current = root_id;
        std::vector<int> search_path = {current};

        while (pool.get(current).expanded()) {
            int next_child = select_child(pool, current, minmax);
            if (next_child < 0) {
                throw std::runtime_error("MCTS failed to select a child from an expanded node");
            }
            current = pool.get(current).children[next_child];
            search_path.push_back(current);
        }

        Node& leaf = pool.get(current);

        EvalInput eval_in;
        eval_in.batch_size = 1;
        eval_in.parent_node_ids = {leaf.parent};
        eval_in.target_node_ids = {current};
        eval_in.actions = {leaf.action_from_parent};

        auto rec_out = net.recurrent_inference(eval_in);
        validate_eval_output(rec_out, 1, "recurrent inference");

        float value = rec_out.values[0];
        float reward = rec_out.rewards[0];
        auto topk_actions = select_topk_candidates(rec_out.policy_logits[0], cfg_.latent_top_k);

        debug.max_latent_branching = std::max(
            debug.max_latent_branching, static_cast<int>(topk_actions.size()));

        expand_node(pool, current, leaf.parent, leaf.action_from_parent, reward, leaf.prior, topk_actions, rec_out.policy_logits[0]);
        backup(pool, search_path, value, minmax);
    }

    debug.total_nodes = pool.size();

    RootSearchOutput out;
    out.root_value = pool.get(root_id).value();
    out.debug = debug;

    int best_action = -1;
    int best_visit_count = -1;

    const Node& root_node = pool.get(root_id);
    out.root_actions = root_node.actions;
    out.root_priors.resize(root_node.actions.size());
    out.visit_counts.resize(root_node.actions.size());
    out.policy_full.assign(kActionSpaceSize, 0.0f);

    float total_visits = 0.0f;
    for (int i = 0; i < static_cast<int>(root_node.actions.size()); ++i) {
        int child_id = root_node.children[i];
        const Node& child = pool.get(child_id);
        int vc = child.visit_count;
        out.root_priors[i] = child.prior;
        out.visit_counts[i] = vc;
        total_visits += static_cast<float>(vc);
        if (vc > best_visit_count) {
            best_visit_count = vc;
            best_action = root_node.actions[i];
        }
    }

    out.action = best_action;

    if (total_visits > 0.0f) {
        for (int i = 0; i < static_cast<int>(root_node.actions.size()); ++i) {
            out.policy_full[root_node.actions[i]] = out.visit_counts[i] / total_visits;
        }
    }

    return out;
}

int MCTS::expand_node(
    NodePool& pool,
    int node_id,
    int parent,
    int action_from_parent,
    float reward,
    float prior,
    const std::vector<int>& candidate_actions,
    const std::vector<float>& policy_logits
) {
    Node& node = pool.get(node_id);
    node.parent = parent;
    node.action_from_parent = action_from_parent;
    node.reward = reward;
    node.prior = prior;

    node.actions = candidate_actions;
    node.children.resize(candidate_actions.size());

    auto priors = softmax_over_actions(policy_logits, candidate_actions);

    for (int i = 0; i < static_cast<int>(candidate_actions.size()); ++i) {
        int child_id = pool.allocate();
        Node& child = pool.get(child_id);

        child.parent = node_id;
        child.action_from_parent = candidate_actions[i];
        child.prior = priors[i];

        pool.get(node_id).children[i] = child_id;
    }

    return node_id;
}

void MCTS::apply_root_noise(NodePool& pool, int root_id) {
    Node& root = pool.get(root_id);
    if (root.children.empty()) return;

    std::gamma_distribution<double> gamma(
        static_cast<double>(cfg_.root_dirichlet_alpha), 1.0);
    std::vector<double> noise(root.children.size(), 0.0);
    double sum = 0.0;
    for (double& value : noise) {
        value = gamma(rng_);
        sum += value;
    }

    if (!std::isfinite(sum) || sum <= 0.0) {
        const double uniform = 1.0 / static_cast<double>(noise.size());
        std::fill(noise.begin(), noise.end(), uniform);
    } else {
        for (double& value : noise) value /= sum;
    }

    const float keep = 1.0f - cfg_.root_noise_weight;
    for (size_t i = 0; i < root.children.size(); ++i) {
        Node& child = pool.get(root.children[i]);
        child.prior = keep * child.prior +
                      cfg_.root_noise_weight * static_cast<float>(noise[i]);
    }
}

void MCTS::validate_eval_output(
    const EvalOutput& output,
    int expected_batch_size,
    const char* stage
) const {
    if (expected_batch_size <= 0 ||
        output.values.size() != static_cast<size_t>(expected_batch_size) ||
        output.rewards.size() != static_cast<size_t>(expected_batch_size) ||
        output.policy_logits.size() != static_cast<size_t>(expected_batch_size)) {
        throw std::runtime_error(std::string("Invalid ") + stage + " batch dimensions");
    }

    for (int i = 0; i < expected_batch_size; ++i) {
        if (!std::isfinite(output.values[i]) || !std::isfinite(output.rewards[i])) {
            throw std::runtime_error(std::string("Non-finite value or reward from ") + stage);
        }
        const auto& logits = output.policy_logits[i];
        if (logits.size() != kActionSpaceSize) {
            throw std::runtime_error(std::string("Invalid policy size from ") + stage);
        }
        for (float logit : logits) {
            if (!std::isfinite(logit)) {
                throw std::runtime_error(std::string("Non-finite policy logit from ") + stage);
            }
        }
    }
}

int MCTS::select_child(
    const NodePool& pool,
    int node_id,
    const MinMaxStats& minmax
) const {
    const Node& node = pool.get(node_id);
    int best_index = -1;
    float best_score = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < static_cast<int>(node.children.size()); ++i) {
        const Node& child = pool.get(node.children[i]);
        float score = ucb_score(node, child, minmax);
        if (score > best_score) {
            best_score = score;
            best_index = i;
        }
    }
    return best_index;
}

float MCTS::ucb_score(
    const Node& parent,
    const Node& child,
    const MinMaxStats& minmax
) const {
    float pb_c = std::log(
        (parent.visit_count + cfg_.pb_c_base + 1.0f) / cfg_.pb_c_base
    ) + cfg_.pb_c_init;

    pb_c *= std::sqrt(static_cast<float>(std::max(1, parent.visit_count))) /
            (child.visit_count + 1.0f);

    float prior_score = pb_c * child.prior;
    float value_score = minmax.normalize(child.reward + cfg_.discount * child.value());

    return prior_score + value_score;
}

void MCTS::backup(
    NodePool& pool,
    const std::vector<int>& search_path,
    float value,
    MinMaxStats& minmax
) const {
    float bootstrap = value;

    for (int i = static_cast<int>(search_path.size()) - 1; i >= 0; --i) {
        Node& node = pool.get(search_path[i]);

        node.value_sum += bootstrap;
        node.visit_count += 1;

        float q = node.reward + cfg_.discount * node.value();
        minmax.update(q);

        bootstrap = node.reward + cfg_.discount * bootstrap;
    }
}

std::vector<int> MCTS::select_topk_candidates(
    const std::vector<float>& policy_logits,
    int k
) const {
    std::vector<std::pair<float, int>> scored_actions;
    scored_actions.reserve(policy_logits.size());
    for (int i = 0; i < static_cast<int>(policy_logits.size()); ++i) {
        scored_actions.push_back({policy_logits[i], i});
    }

    int actual_k = std::min(k, static_cast<int>(scored_actions.size()));
    std::partial_sort(
        scored_actions.begin(), scored_actions.begin() + actual_k, scored_actions.end(),
        [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });

    std::vector<int> topk;
    topk.reserve(actual_k);
    for (int i = 0; i < actual_k; ++i) {
        topk.push_back(scored_actions[i].second);
    }
    return topk;
}

std::vector<float> MCTS::softmax_over_actions(
    const std::vector<float>& logits,
    const std::vector<int>& actions
) const {
    if (actions.empty()) return {};

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int action : actions) {
        if (action < 0 || action >= static_cast<int>(logits.size())) {
            throw std::runtime_error("MCTS candidate action is outside policy logits");
        }
        max_logit = std::max(max_logit, logits[action]);
    }

    std::vector<float> exps;
    exps.reserve(actions.size());
    double sum_exp = 0.0;
    for (int action : actions) {
        float value = std::exp(logits[action] - max_logit);
        exps.push_back(value);
        sum_exp += value;
    }

    if (!std::isfinite(sum_exp) || sum_exp <= 0.0) {
        throw std::runtime_error("MCTS policy softmax normalization failed");
    }
    for (float& value : exps) {
        value = static_cast<float>(value / sum_exp);
    }

    return exps;
}

} // namespace tdmz
