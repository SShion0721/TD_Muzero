#pragma once
#include <random>
#include <string>
#include <vector>
#include "tdmz/core/action.hpp"
#include "tdmz/mcts/mcts_config.hpp"
#include "tdmz/mcts/node_pool.hpp"
#include "tdmz/mcts/minmax.hpp"
#include "tdmz/mcts/network_evaluator.hpp"
#include "tdmz/mcts/search_output.hpp"

namespace tdmz {

class MCTS {
public:
    explicit MCTS(MCTSConfig cfg);

    RootSearchOutput search_single(
        INetworkEvaluator& net,
        const std::vector<float>& observation,
        const std::vector<int>& legal_actions
    );

private:
    MCTSConfig cfg_;
    std::mt19937_64 rng_;

    int expand_node(
        NodePool& pool,
        int node_id,
        int parent,
        int action_from_parent,
        float reward,
        float prior,
        const std::vector<int>& candidate_actions,
        const std::vector<float>& policy_logits
    );

    void apply_root_noise(NodePool& pool, int root_id);

    void validate_eval_output(
        const EvalOutput& output,
        int expected_batch_size,
        const char* stage
    ) const;

    int select_child(
        const NodePool& pool,
        int node_id,
        const MinMaxStats& minmax
    ) const;

    float ucb_score(
        const Node& parent,
        const Node& child,
        const MinMaxStats& minmax
    ) const;

    void backup(
        NodePool& pool,
        const std::vector<int>& search_path,
        float value,
        MinMaxStats& minmax
    ) const;

    std::vector<int> select_topk_candidates(
        const std::vector<float>& policy_logits,
        int k
    ) const;

    std::vector<float> softmax_over_actions(
        const std::vector<float>& logits,
        const std::vector<int>& actions
    ) const;
};

} // namespace tdmz
