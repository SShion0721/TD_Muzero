#pragma once
#include <vector>

namespace tdmz {

struct EvalInput {
    int batch_size = 0;
    // TODO(Phase 3): Replace with proper LatentRef when LibTorch latent store is introduced.
    // Currently these are node IDs of the parent nodes, NOT latent state indices.
    std::vector<int> parent_node_ids;
    std::vector<int> actions;
};

struct EvalOutput {
    std::vector<float> values;                         // [B]
    std::vector<float> rewards;                        // [B]
    std::vector<std::vector<float>> policy_logits;     // [B][A]
};

class INetworkEvaluator {
public:
    virtual ~INetworkEvaluator() = default;

    virtual EvalOutput initial_inference(
        const std::vector<std::vector<float>>& observations
    ) = 0;

    virtual EvalOutput recurrent_inference(
        const EvalInput& input
    ) = 0;
};

} // namespace tdmz
