#pragma once

#include "tdmz/core/action.hpp"

#include <cstdint>
#include <vector>

namespace tdmz {

struct LegalityMetrics {
    uint64_t samples = 0;
    uint64_t total_actions = 0;

    uint64_t true_legal_count = 0;
    uint64_t true_illegal_count = 0;
    uint64_t predicted_legal_count = 0;
    uint64_t predicted_illegal_count = 0;

    uint64_t true_positive = 0;
    uint64_t false_positive = 0;
    uint64_t true_negative = 0;
    uint64_t false_negative = 0;

    uint64_t wait_legal_count = 0;
    uint64_t wait_true_positive = 0;

    double legal_recall = 0.0;
    double legal_precision = 0.0;
    // Among actions predicted illegal, the fraction that are truly illegal.
    double illegal_precision = 0.0;
    double false_negative_rate = 0.0;
    double false_positive_rate = 0.0;
    double wait_recall = 0.0;

    bool hard_pruning_ready(double minimum_legal_recall = 0.995) const;
};

// Converts one exact engine legal mask into floating BCE targets [727].
// The mask must be binary and Wait1 must be legal.
std::vector<float> make_legality_targets(
    const std::vector<uint8_t>& exact_legal_mask
);

// Converts flat exact engine masks [B,727] into floating BCE targets [B,727].
std::vector<float> make_legality_targets(
    const std::vector<uint8_t>& flat_exact_legal_masks,
    int batch_size
);

class LegalityMetricsAccumulator {
public:
    void reset();

    // A logit >= threshold is classified as legal. The default threshold 0 is
    // equivalent to sigmoid(logit) >= 0.5.
    void update(
        const std::vector<float>& legality_logits,
        const std::vector<uint8_t>& exact_legal_mask,
        float threshold_logit = 0.0f
    );

    void update_batch(
        const std::vector<float>& flat_legality_logits,
        const std::vector<uint8_t>& flat_exact_legal_masks,
        int batch_size,
        float threshold_logit = 0.0f
    );

    LegalityMetrics metrics() const;

private:
    uint64_t samples_ = 0;
    uint64_t true_legal_count_ = 0;
    uint64_t predicted_legal_count_ = 0;
    uint64_t true_positive_ = 0;
    uint64_t false_positive_ = 0;
    uint64_t true_negative_ = 0;
    uint64_t false_negative_ = 0;
    uint64_t wait_legal_count_ = 0;
    uint64_t wait_true_positive_ = 0;
};

} // namespace tdmz
