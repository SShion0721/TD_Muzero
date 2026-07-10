#include "tdmz/mcts/mcts.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tdmz {

namespace {

uint64_t required_nodes_for_search(
    int num_simulations,
    int latent_top_k,
    size_t root_action_count
) {
    const uint64_t latent_width = static_cast<uint64_t>(
        std::min(latent_top_k, kActionSpaceSize));
    return 1ULL
        + static_cast<uint64_t>(root_action_count)
        + static_cast<uint64_t>(num_simulations) * latent_width;
}

} // namespace

MCTS::MCTS(MCTSConfig cfg)
    : cfg_(cfg), rng_(cfg.random_seed), node_pool_(cfg.max_nodes) {
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
        if (!std::isfinite(cfg_.root_noise_weight)
            || cfg_.root_noise_weight < 0.0f
            || cfg_.root_noise_weight > 1.0f) {
            throw std::invalid_argument("MCTS root_noise_weight must be finite and in [0, 1]");
        }
    }

    seen_actions_scratch_.assign(kActionSpaceSize, 0u);
    initial_observations_scratch_.resize(1);
    recurrent_input_scratch_.batch_size = 1;
    recurrent_input_scratch_.parent_node_ids.resize(1);
    recurrent_input_scratch_.target_node_ids.resize(1);
    recurrent_input_scratch_.actions.resize(1);

    search_path_scratch_.reserve(static_cast<size_t>(cfg_.num_simulations) + 1u);
    scored_actions_scratch_.reserve(kActionSpaceSize);
    topk_actions_scratch_.reserve(static_cast<size_t>(std::min(cfg_.latent_top_k, kActionSpaceSize)));
    priors_scratch_.reserve(kActionSpaceSize);
    root_noise_scratch_.reserve(kActionSpaceSize);
}

RootSearchOutput MCTS::search_single(
    INetworkEvaluator& net,
    const std::vector<float>& observation,
    const std::vector<int>& legal_actions
) {
    if (legal_actions.empty()) {
        throw std::invalid_argument("MCTS requires at least one legal root action");
    }

    std::fill(seen_actions_scratch_.begin(), seen_actions_scratch_.end(), 0u);
    for (int action : legal_actions) {
        if (action < 0 || action >= kActionSpaceSize) {
            throw std::invalid_argument("MCTS legal root action is out of range");
        }
        if (seen_actions_scratch_[static_cast<size_t>(action)] != 0u) {
            throw std::invalid_argument("MCTS legal root actions contain duplicates");
        }
        seen_actions_scratch_[static_cast<size_t>(action)] = 1u;
    }

    const uint64_t required_nodes = required_nodes_for_search(
        cfg_.num_simulations, cfg_.latent_top_k, legal_actions.size());
    if (required_nodes > static_cast<uint64_t>(cfg_.max_nodes)) {
        throw std::invalid_argument(
            "MCTS max_nodes is too small for the configured simulations and branching");
    }

    node_pool_.clear();
    scratch_capacity_growth_events_ = 0;
    node_buffer_growth_events_ = 0;
    max_search_depth_ = 0;
    MinMaxStats minmax(cfg_.value_delta_max);

    auto& initial_observation = initial_observations_scratch_[0];
    if (initial_observation.capacity() < observation.size()) {
        ++scratch_capacity_growth_events_;
    }
    initial_observation.assign(observation.begin(), observation.end());

    const int root_id = node_pool_.allocate();
    EvalOutput init_out = net.initial_inference(initial_observations_scratch_);
    validate_eval_output(init_out, 1, "initial inference");
    expand_node(
        node_pool_, root_id, -1, -1, 0.0f, 1.0f,
        legal_actions, init_out.policy_logits[0]);
    if (cfg_.add_root_noise && cfg_.root_noise_weight > 0.0f) {
        apply_root_noise(node_pool_, root_id);
    }

    SearchDebugStats debug;
    debug.max_root_branching = static_cast<int>(legal_actions.size());

    for (int simulation = 0; simulation < cfg_.num_simulations; ++simulation) {
        int current = root_id;
        search_path_scratch_.clear();
        search_path_scratch_.push_back(current);

        while (node_pool_.get(current).expanded()) {
            const int next_child = select_child(node_pool_, current, minmax);
            if (next_child < 0) {
                throw std::runtime_error("MCTS failed to select a child from an expanded node");
            }
            current = node_pool_.get(current).children[static_cast<size_t>(next_child)];
            if (search_path_scratch_.size() == search_path_scratch_.capacity()) {
                ++scratch_capacity_growth_events_;
            }
            search_path_scratch_.push_back(current);
        }
        max_search_depth_ = std::max(
            max_search_depth_, static_cast<int>(search_path_scratch_.size()));

        Node& leaf = node_pool_.get(current);
        recurrent_input_scratch_.parent_node_ids[0] = leaf.parent;
        recurrent_input_scratch_.target_node_ids[0] = current;
        recurrent_input_scratch_.actions[0] = leaf.action_from_parent;

        EvalOutput recurrent_output = net.recurrent_inference(recurrent_input_scratch_);
        validate_eval_output(recurrent_output, 1, "recurrent inference");

        const float value = recurrent_output.values[0];
        const float reward = recurrent_output.rewards[0];
        const auto& topk_actions = select_topk_candidates(
            recurrent_output.policy_logits[0], cfg_.latent_top_k);

        debug.max_latent_branching = std::max(
            debug.max_latent_branching, static_cast<int>(topk_actions.size()));
        expand_node(
            node_pool_, current, leaf.parent, leaf.action_from_parent,
            reward, leaf.prior, topk_actions, recurrent_output.policy_logits[0]);
        backup(node_pool_, search_path_scratch_, value, minmax);
    }

    debug.total_nodes = node_pool_.size();
    debug.max_search_depth = max_search_depth_;
    debug.node_objects_created = node_pool_.new_nodes_this_search();
    debug.node_objects_reused = node_pool_.reused_nodes_this_search();
    debug.node_buffer_growth_events = node_buffer_growth_events_;
    debug.scratch_capacity_growth_events = scratch_capacity_growth_events_;

    RootSearchOutput output;
    output.root_value = node_pool_.get(root_id).value();
    output.debug = debug;

    int best_action = -1;
    int best_visit_count = -1;
    const Node& root_node = node_pool_.get(root_id);
    output.root_actions = root_node.actions;
    output.root_priors.resize(root_node.actions.size());
    output.visit_counts.resize(root_node.actions.size());
    output.policy_full.assign(kActionSpaceSize, 0.0f);

    float total_visits = 0.0f;
    for (int i = 0; i < static_cast<int>(root_node.actions.size()); ++i) {
        const int child_id = root_node.children[static_cast<size_t>(i)];
        const Node& child = node_pool_.get(child_id);
        const int visit_count = child.visit_count;
        output.root_priors[static_cast<size_t>(i)] = child.prior;
        output.visit_counts[static_cast<size_t>(i)] = visit_count;
        total_visits += static_cast<float>(visit_count);
        if (visit_count > best_visit_count) {
            best_visit_count = visit_count;
            best_action = root_node.actions[static_cast<size_t>(i)];
        }
    }

    output.action = best_action;
    if (total_visits > 0.0f) {
        for (int i = 0; i < static_cast<int>(root_node.actions.size()); ++i) {
            output.policy_full[static_cast<size_t>(root_node.actions[static_cast<size_t>(i)])]
                = output.visit_counts[static_cast<size_t>(i)] / total_visits;
        }
    }
    return output;
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
    if (candidate_actions.size() > static_cast<size_t>(pool.remaining())) {
        throw std::runtime_error("NodePool capacity is insufficient for atomic node expansion");
    }

    Node& node = pool.get(node_id);
    node.parent = parent;
    node.action_from_parent = action_from_parent;
    node.reward = reward;
    node.prior = prior;

    if (node.actions.capacity() < candidate_actions.size()) {
        ++node_buffer_growth_events_;
    }
    node.actions.assign(candidate_actions.begin(), candidate_actions.end());
    if (node.children.capacity() < candidate_actions.size()) {
        ++node_buffer_growth_events_;
    }
    node.children.resize(candidate_actions.size());

    const auto& priors = softmax_over_actions(policy_logits, candidate_actions);
    for (int i = 0; i < static_cast<int>(candidate_actions.size()); ++i) {
        const int child_id = pool.allocate();
        Node& child = pool.get(child_id);
        child.parent = node_id;
        child.action_from_parent = candidate_actions[static_cast<size_t>(i)];
        child.prior = priors[static_cast<size_t>(i)];
        pool.get(node_id).children[static_cast<size_t>(i)] = child_id;
    }
    return node_id;
}

void MCTS::apply_root_noise(NodePool& pool, int root_id) {
    Node& root = pool.get(root_id);
    if (root.children.empty()) return;

    if (root_noise_scratch_.capacity() < root.children.size()) {
        ++scratch_capacity_growth_events_;
    }
    root_noise_scratch_.resize(root.children.size());
    std::gamma_distribution<double> gamma(
        static_cast<double>(cfg_.root_dirichlet_alpha), 1.0);
    double sum = 0.0;
    for (double& value : root_noise_scratch_) {
        value = gamma(rng_);
        sum += value;
    }

    if (!std::isfinite(sum) || sum <= 0.0) {
        const double uniform = 1.0 / static_cast<double>(root_noise_scratch_.size());
        std::fill(root_noise_scratch_.begin(), root_noise_scratch_.end(), uniform);
    } else {
        for (double& value : root_noise_scratch_) value /= sum;
    }

    const float keep = 1.0f - cfg_.root_noise_weight;
    for (size_t i = 0; i < root.children.size(); ++i) {
        Node& child = pool.get(root.children[i]);
        child.prior = keep * child.prior
            + cfg_.root_noise_weight * static_cast<float>(root_noise_scratch_[i]);
    }
}

void MCTS::validate_eval_output(
    const EvalOutput& output,
    int expected_batch_size,
    const char* stage
) const {
    if (expected_batch_size <= 0
        || output.values.size() != static_cast<size_t>(expected_batch_size)
        || output.rewards.size() != static_cast<size_t>(expected_batch_size)
        || output.policy_logits.size() != static_cast<size_t>(expected_batch_size)) {
        throw std::runtime_error(std::string("Invalid ") + stage + " batch dimensions");
    }

    for (int i = 0; i < expected_batch_size; ++i) {
        if (!std::isfinite(output.values[static_cast<size_t>(i)])
            || !std::isfinite(output.rewards[static_cast<size_t>(i)])) {
            throw std::runtime_error(std::string("Non-finite value or reward from ") + stage);
        }
        const auto& logits = output.policy_logits[static_cast<size_t>(i)];
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
        const Node& child = pool.get(node.children[static_cast<size_t>(i)]);
        const float score = ucb_score(node, child, minmax);
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
        (parent.visit_count + cfg_.pb_c_base + 1.0f) / cfg_.pb_c_base)
        + cfg_.pb_c_init;
    pb_c *= std::sqrt(static_cast<float>(std::max(1, parent.visit_count)))
        / (child.visit_count + 1.0f);

    const float prior_score = pb_c * child.prior;
    const float value_score = minmax.normalize(
        child.reward + cfg_.discount * child.value());
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
        Node& node = pool.get(search_path[static_cast<size_t>(i)]);
        node.value_sum += bootstrap;
        node.visit_count += 1;
        const float q = node.reward + cfg_.discount * node.value();
        minmax.update(q);
        bootstrap = node.reward + cfg_.discount * bootstrap;
    }
}

const std::vector<int>& MCTS::select_topk_candidates(
    const std::vector<float>& policy_logits,
    int k
) {
    scored_actions_scratch_.clear();
    if (scored_actions_scratch_.capacity() < policy_logits.size()) {
        ++scratch_capacity_growth_events_;
    }
    for (int i = 0; i < static_cast<int>(policy_logits.size()); ++i) {
        scored_actions_scratch_.push_back({policy_logits[static_cast<size_t>(i)], i});
    }

    const int actual_k = std::min(k, static_cast<int>(scored_actions_scratch_.size()));
    std::partial_sort(
        scored_actions_scratch_.begin(),
        scored_actions_scratch_.begin() + actual_k,
        scored_actions_scratch_.end(),
        [](const auto& left, const auto& right) {
            if (left.first != right.first) return left.first > right.first;
            return left.second < right.second;
        });

    topk_actions_scratch_.clear();
    if (topk_actions_scratch_.capacity() < static_cast<size_t>(actual_k)) {
        ++scratch_capacity_growth_events_;
    }
    for (int i = 0; i < actual_k; ++i) {
        topk_actions_scratch_.push_back(
            scored_actions_scratch_[static_cast<size_t>(i)].second);
    }
    return topk_actions_scratch_;
}

const std::vector<float>& MCTS::softmax_over_actions(
    const std::vector<float>& logits,
    const std::vector<int>& actions
) {
    priors_scratch_.clear();
    if (actions.empty()) return priors_scratch_;

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int action : actions) {
        if (action < 0 || action >= static_cast<int>(logits.size())) {
            throw std::runtime_error("MCTS candidate action is outside policy logits");
        }
        max_logit = std::max(max_logit, logits[static_cast<size_t>(action)]);
    }

    if (priors_scratch_.capacity() < actions.size()) {
        ++scratch_capacity_growth_events_;
    }
    double sum_exp = 0.0;
    for (int action : actions) {
        const float value = std::exp(logits[static_cast<size_t>(action)] - max_logit);
        priors_scratch_.push_back(value);
        sum_exp += value;
    }
    if (!std::isfinite(sum_exp) || sum_exp <= 0.0) {
        throw std::runtime_error("MCTS policy softmax normalization failed");
    }
    for (float& value : priors_scratch_) {
        value = static_cast<float>(value / sum_exp);
    }
    return priors_scratch_;
}

} // namespace tdmz
