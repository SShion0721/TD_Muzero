#include "tdmz/mcts/mcts.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

namespace tdmz {

MCTS::MCTS(MCTSConfig cfg) : cfg_(cfg) {}

RootSearchOutput MCTS::search_single(
    INetworkEvaluator& net,
    const std::vector<float>& observation,
    const std::vector<int>& legal_actions
) {
    NodePool pool(cfg_.max_nodes);
    MinMaxStats minmax(cfg_.value_delta_max);

    int root_id = pool.allocate();
    Node& root = pool.get(root_id);

    auto init_out = net.initial_inference({observation});
    float root_value = init_out.values[0];
    
    expand_node(pool, root_id, -1, -1, 0.0f, 1.0f, legal_actions, init_out.policy_logits[0]);

    for (int sim = 0; sim < cfg_.num_simulations; ++sim) {
        int current = root_id;
        std::vector<int> search_path = {current};

        while (pool.get(current).expanded()) {
            int next_child = select_child(pool, current, minmax);
            current = pool.get(current).children[next_child];
            search_path.push_back(current);
        }

        Node& leaf = pool.get(current);

        EvalInput eval_in;
        eval_in.batch_size = 1;
        eval_in.latent_ids = {leaf.parent}; 
        eval_in.actions = {leaf.action_from_parent};
        
        auto rec_out = net.recurrent_inference(eval_in);
        
        float value = rec_out.values[0];
        float reward = rec_out.rewards[0];

        auto topk_actions = select_topk_candidates(rec_out.policy_logits[0], cfg_.latent_top_k);
        expand_node(pool, current, leaf.parent, leaf.action_from_parent, reward, leaf.prior, topk_actions, rec_out.policy_logits[0]);

        backup(pool, search_path, value, minmax);
    }

    RootSearchOutput out;
    out.root_value = root_value;
    
    int best_action = -1;
    int best_visit_count = -1;
    
    const Node& root_node = pool.get(root_id);
    out.root_actions = root_node.actions;
    out.visit_counts.resize(root_node.actions.size());
    out.policy_full.assign(kActionSpaceSize, 0.0f);

    float total_visits = 0.0f;
    for (int i = 0; i < static_cast<int>(root_node.actions.size()); ++i) {
        int child_id = root_node.children[i];
        int vc = pool.get(child_id).visit_count;
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
    std::partial_sort(scored_actions.begin(), scored_actions.begin() + actual_k, scored_actions.end(),
                      [](const auto& a, const auto& b) {
                          return a.first > b.first;
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
    for (int a : actions) {
        max_logit = std::max(max_logit, logits[a]);
    }
    
    std::vector<float> exps;
    exps.reserve(actions.size());
    float sum_exp = 0.0f;
    for (int a : actions) {
        float e = std::exp(logits[a] - max_logit);
        exps.push_back(e);
        sum_exp += e;
    }
    
    for (float& e : exps) {
        e /= sum_exp;
    }
    
    return exps;
}

} // namespace tdmz
