#pragma once
#include <vector>

namespace tdmz {

struct EvalInput {
    int batch_size = 0;
    // Parent node IDs are used by evaluators with a latent store to fetch the
    // latent state before applying the selected action.
    std::vector<int> parent_node_ids;
    // Current/target node IDs are where recurrent evaluators should store the
    // newly produced latent state. Dummy evaluators may ignore this field.
    std::vector<int> target_node_ids;
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
