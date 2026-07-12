#pragma once
#include "tdmz/mcts/network_evaluator.hpp"
#include "tdmz/core/action.hpp"

namespace tdmz {

class DummyNetwork : public INetworkEvaluator {
protected:
    EvalOutput initial_inference_impl(
        const std::vector<std::vector<float>>& observations
    ) override {
        int b = static_cast<int>(observations.size());
        return make_output(b);
    }

    EvalOutput recurrent_inference_impl(
        const EvalInput& input
    ) override {
        return make_output(input.batch_size);
    }

private:
    EvalOutput make_output(int batch_size) {
        EvalOutput out;
        out.values.assign(batch_size, 0.5f);
        out.rewards.assign(batch_size, 0.0f);
        out.policy_logits.resize(batch_size, std::vector<float>(kActionSpaceSize, -10.0f));
        out.legality_logits.resize(batch_size, std::vector<float>(kActionSpaceSize, 0.0f));

        for (int i = 0; i < batch_size; ++i) {
            out.policy_logits[i][kFlatWaitOffset] = 1.0f;
            out.legality_logits[i][kFlatWaitOffset] = 1.0f;
            for (int a = 0; a < kActionSpaceSize; ++a) {
                if (a % 17 == 0) out.policy_logits[i][a] = 0.2f;
            }
        }
        return out;
    }
};

} // namespace tdmz
