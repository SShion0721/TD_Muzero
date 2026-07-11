#include "tdmz/mcts/action_mask.hpp"
#include "tdmz/mcts/mcts.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tdmz {

namespace {

void validate_mask(const std::vector<uint8_t>& legal_mask) {
    if (legal_mask.empty()) {
        throw std::invalid_argument("Action mask must not be empty");
    }
    bool any_legal = false;
    for (uint8_t value : legal_mask) {
        if (value > 1u) {
            throw std::invalid_argument("Action mask values must be 0 or 1");
        }
        any_legal = any_legal || value != 0u;
    }
    if (!any_legal) {
        throw std::invalid_argument("Action mask must contain at least one legal action");
    }
}

} // namespace

std::size_t count_legal_actions(const std::vector<uint8_t>& legal_mask) {
    validate_mask(legal_mask);
    std::size_t count = 0;
    for (uint8_t value : legal_mask) count += value != 0u ? 1u : 0u;
    return count;
}

std::vector<int> legal_actions_from_mask(const std::vector<uint8_t>& legal_mask) {
    validate_mask(legal_mask);
    std::vector<int> actions;
    actions.reserve(count_legal_actions(legal_mask));
    for (std::size_t i = 0; i < legal_mask.size(); ++i) {
        if (legal_mask[i] != 0u) actions.push_back(static_cast<int>(i));
    }
    return actions;
}

std::vector<float> masked_softmax(
    const std::vector<float>& logits,
    const std::vector<uint8_t>& legal_mask
) {
    if (logits.size() != legal_mask.size()) {
        throw std::invalid_argument("Masked softmax logits and mask sizes differ");
    }
    validate_mask(legal_mask);

    float max_logit = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (legal_mask[i] == 0u) continue;
        if (!std::isfinite(logits[i])) {
            throw std::invalid_argument("Masked softmax received a non-finite legal logit");
        }
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    std::vector<float> probabilities(logits.size(), 0.0f);
    double sum = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (legal_mask[i] == 0u) continue;
        const double value = std::exp(static_cast<double>(logits[i] - max_logit));
        probabilities[i] = static_cast<float>(value);
        sum += value;
    }
    if (!std::isfinite(sum) || sum <= 0.0) {
        throw std::runtime_error("Masked softmax normalization failed");
    }

    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        if (legal_mask[i] != 0u) {
            probabilities[i] = static_cast<float>(probabilities[i] / sum);
        }
    }
    return probabilities;
}

RootSearchOutput MCTS::search_single_masked(
    INetworkEvaluator& net,
    const std::vector<float>& observation,
    const std::vector<uint8_t>& legal_mask
) {
    if (legal_mask.size() != static_cast<std::size_t>(kActionSpaceSize)) {
        throw std::invalid_argument("MCTS root action mask has an invalid size");
    }
    return search_single(net, observation, legal_actions_from_mask(legal_mask));
}

} // namespace tdmz
