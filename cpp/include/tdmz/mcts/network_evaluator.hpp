#pragma once
#include "tdmz/core/action.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
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
    std::vector<std::vector<float>> legality_logits;   // [B][A]
};

class INetworkEvaluator {
public:
    virtual ~INetworkEvaluator() = default;

    EvalOutput initial_inference(
        const std::vector<std::vector<float>>& observations
    ) {
        return validate_output(
            initial_inference_impl(observations),
            static_cast<int>(observations.size()),
            "initial inference");
    }

    EvalOutput recurrent_inference(const EvalInput& input) {
        return validate_output(
            recurrent_inference_impl(input),
            input.batch_size,
            "recurrent inference");
    }

protected:
    virtual EvalOutput initial_inference_impl(
        const std::vector<std::vector<float>>& observations
    ) = 0;

    virtual EvalOutput recurrent_inference_impl(
        const EvalInput& input
    ) = 0;

private:
    static EvalOutput validate_output(
        EvalOutput output,
        int expected_batch_size,
        const char* stage
    ) {
        if (expected_batch_size <= 0
            || output.values.size() != static_cast<size_t>(expected_batch_size)
            || output.rewards.size() != static_cast<size_t>(expected_batch_size)
            || output.policy_logits.size() != static_cast<size_t>(expected_batch_size)
            || output.legality_logits.size() != static_cast<size_t>(expected_batch_size)) {
            throw std::runtime_error(std::string("Invalid ") + stage + " batch dimensions");
        }

        for (int batch = 0; batch < expected_batch_size; ++batch) {
            const size_t index = static_cast<size_t>(batch);
            if (!std::isfinite(output.values[index]) || !std::isfinite(output.rewards[index])) {
                throw std::runtime_error(
                    std::string("Non-finite value or reward from ") + stage);
            }

            const auto& policy = output.policy_logits[index];
            const auto& legality = output.legality_logits[index];
            if (policy.size() != kActionSpaceSize) {
                throw std::runtime_error(std::string("Invalid policy size from ") + stage);
            }
            if (legality.size() != kActionSpaceSize) {
                throw std::runtime_error(std::string("Invalid legality size from ") + stage);
            }

            for (float logit : policy) {
                if (!std::isfinite(logit)) {
                    throw std::runtime_error(
                        std::string("Non-finite policy logit from ") + stage);
                }
            }
            for (float logit : legality) {
                if (!std::isfinite(logit)) {
                    throw std::runtime_error(
                        std::string("Non-finite legality logit from ") + stage);
                }
            }
        }
        return output;
    }
};

} // namespace tdmz
