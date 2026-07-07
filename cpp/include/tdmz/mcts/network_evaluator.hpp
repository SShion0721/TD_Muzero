#pragma once
#include <vector>

namespace tdmz {

struct EvalInput {
    int batch_size = 0;
    std::vector<int> latent_ids;
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
