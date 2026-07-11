#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tdmz {

std::size_t count_legal_actions(const std::vector<uint8_t>& legal_mask);

// Convert a full state-dependent mask into ascending action IDs.
// The mask must contain only 0/1 values and at least one legal action.
std::vector<int> legal_actions_from_mask(const std::vector<uint8_t>& legal_mask);

// Stable full-width masked softmax. Invalid actions receive exactly 0.0f and
// the probabilities over valid actions sum to one.
std::vector<float> masked_softmax(
    const std::vector<float>& logits,
    const std::vector<uint8_t>& legal_mask
);

} // namespace tdmz
