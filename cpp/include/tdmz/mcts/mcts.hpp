#pragma once
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "tdmz/core/action.hpp"
#include "tdmz/mcts/edge_pool.hpp"
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
    NodePool node_pool_;
    EdgePool edge_pool_;

    std::vector<uint8_t> seen_actions_scratch_;
    std::vector<std::vector<float>> initial_observations_scratch_;
    EvalInput recurrent_input_scratch_;
    std::vector<std::pair<float, int>> scored_actions_scratch_;
    std::vector<int> topk_actions_scratch_;
    std::vector<float> priors_scratch_;
    std::vector<double> root_noise_scratch_;

    std::vector<std::vector<int>> pending_paths_scratch_;
    std::vector<int> pending_leaf_ids_scratch_;
    std::vector<uint32_t> leaf_reservation_marks_;
    uint32_t leaf_reservation_epoch_ = 0;

    int scratch_capacity_growth_events_ = 0;
    int max_search_depth_ = 0;

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

    int select_child_edge(
        const NodePool& pool,
        int node_id,
        const MinMaxStats& minmax
    ) const;

    int select_leaf_path(
        const NodePool& pool,
        int root_id,
        const MinMaxStats& minmax,
        std::vector<int>& path
    );

    void apply_virtual_visits(NodePool& pool, const std::vector<int>& path) const;
    void undo_virtual_visits(NodePool& pool, const std::vector<int>& path) const;
    void begin_leaf_reservation_epoch();

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

    const std::vector<int>& select_topk_candidates(
        const std::vector<float>& policy_logits,
        int k
    );

    const std::vector<float>& softmax_over_actions(
        const std::vector<float>& logits,
        const std::vector<int>& actions
    );
};

} // namespace tdmz
