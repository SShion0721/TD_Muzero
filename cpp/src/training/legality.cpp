#include "tdmz/training/legality.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace tdmz {

namespace {

void validate_batch_size(int batch_size) {
    if (batch_size <= 0) {
        throw std::invalid_argument("Legality batch size must be positive");
    }
}

void validate_mask_row(const uint8_t* mask, int row) {
    for (int action = 0; action < kActionSpaceSize; ++action) {
        const uint8_t value = mask[action];
        if (value > 1u) {
            throw std::invalid_argument(
                "Legality mask must be binary at row " + std::to_string(row));
        }
    }
    if (mask[kFlatWaitOffset] == 0u) {
        throw std::invalid_argument(
            "Exact engine legality mask must keep Wait1 legal at row " +
            std::to_string(row));
    }
}

double ratio_or_zero(uint64_t numerator, uint64_t denominator) {
    if (denominator == 0u) return 0.0;
    return static_cast<double>(numerator) / static_cast<double>(denominator);
}

} // namespace

bool LegalityMetrics::hard_pruning_ready(double minimum_legal_recall) const {
    if (!std::isfinite(minimum_legal_recall)
        || minimum_legal_recall < 0.0
        || minimum_legal_recall > 1.0) {
        throw std::invalid_argument(
            "Minimum legal recall must be finite and in [0,1]");
    }
    return samples > 0u
        && legal_recall >= minimum_legal_recall
        && wait_recall >= 1.0;
}

std::vector<float> make_legality_targets(
    const std::vector<uint8_t>& exact_legal_mask
) {
    return make_legality_targets(exact_legal_mask, 1);
}

std::vector<float> make_legality_targets(
    const std::vector<uint8_t>& flat_exact_legal_masks,
    int batch_size
) {
    validate_batch_size(batch_size);
    const size_t expected = static_cast<size_t>(batch_size) * kActionSpaceSize;
    if (flat_exact_legal_masks.size() != expected) {
        throw std::invalid_argument(
            "Legality mask payload does not match [B,727]");
    }

    std::vector<float> targets(expected, 0.0f);
    for (int batch = 0; batch < batch_size; ++batch) {
        const size_t offset = static_cast<size_t>(batch) * kActionSpaceSize;
        const uint8_t* mask = flat_exact_legal_masks.data() + offset;
        validate_mask_row(mask, batch);
        for (int action = 0; action < kActionSpaceSize; ++action) {
            targets[offset + static_cast<size_t>(action)] =
                mask[action] != 0u ? 1.0f : 0.0f;
        }
    }
    return targets;
}

void LegalityMetricsAccumulator::reset() {
    samples_ = 0;
    true_legal_count_ = 0;
    predicted_legal_count_ = 0;
    true_positive_ = 0;
    false_positive_ = 0;
    true_negative_ = 0;
    false_negative_ = 0;
    wait_legal_count_ = 0;
    wait_true_positive_ = 0;
}

void LegalityMetricsAccumulator::update(
    const std::vector<float>& legality_logits,
    const std::vector<uint8_t>& exact_legal_mask,
    float threshold_logit
) {
    update_batch(legality_logits, exact_legal_mask, 1, threshold_logit);
}

void LegalityMetricsAccumulator::update_batch(
    const std::vector<float>& flat_legality_logits,
    const std::vector<uint8_t>& flat_exact_legal_masks,
    int batch_size,
    float threshold_logit
) {
    validate_batch_size(batch_size);
    if (!std::isfinite(threshold_logit)) {
        throw std::invalid_argument("Legality threshold must be finite");
    }

    const size_t expected = static_cast<size_t>(batch_size) * kActionSpaceSize;
    if (flat_legality_logits.size() != expected) {
        throw std::invalid_argument(
            "Legality logit payload does not match [B,727]");
    }
    if (flat_exact_legal_masks.size() != expected) {
        throw std::invalid_argument(
            "Legality mask payload does not match [B,727]");
    }

    // Accumulate into local counters first so malformed input cannot partially
    // mutate a long-lived validation accumulator.
    uint64_t add_samples = 0;
    uint64_t add_true_legal = 0;
    uint64_t add_predicted_legal = 0;
    uint64_t add_true_positive = 0;
    uint64_t add_false_positive = 0;
    uint64_t add_true_negative = 0;
    uint64_t add_false_negative = 0;
    uint64_t add_wait_legal = 0;
    uint64_t add_wait_true_positive = 0;

    for (int batch = 0; batch < batch_size; ++batch) {
        const size_t offset = static_cast<size_t>(batch) * kActionSpaceSize;
        const uint8_t* mask = flat_exact_legal_masks.data() + offset;
        validate_mask_row(mask, batch);

        ++add_samples;
        ++add_wait_legal;
        for (int action = 0; action < kActionSpaceSize; ++action) {
            const float logit = flat_legality_logits[
                offset + static_cast<size_t>(action)];
            if (!std::isfinite(logit)) {
                throw std::invalid_argument(
                    "Legality logits must be finite at row " +
                    std::to_string(batch));
            }

            const bool truth_legal = mask[action] != 0u;
            const bool predicted_legal = logit >= threshold_logit;
            if (truth_legal) ++add_true_legal;
            if (predicted_legal) ++add_predicted_legal;

            if (truth_legal && predicted_legal) {
                ++add_true_positive;
            } else if (!truth_legal && predicted_legal) {
                ++add_false_positive;
            } else if (!truth_legal && !predicted_legal) {
                ++add_true_negative;
            } else {
                ++add_false_negative;
            }

            if (action == kFlatWaitOffset && predicted_legal) {
                ++add_wait_true_positive;
            }
        }
    }

    samples_ += add_samples;
    true_legal_count_ += add_true_legal;
    predicted_legal_count_ += add_predicted_legal;
    true_positive_ += add_true_positive;
    false_positive_ += add_false_positive;
    true_negative_ += add_true_negative;
    false_negative_ += add_false_negative;
    wait_legal_count_ += add_wait_legal;
    wait_true_positive_ += add_wait_true_positive;
}

LegalityMetrics LegalityMetricsAccumulator::metrics() const {
    LegalityMetrics output;
    output.samples = samples_;
    output.total_actions = samples_ * static_cast<uint64_t>(kActionSpaceSize);
    output.true_legal_count = true_legal_count_;
    output.true_illegal_count = output.total_actions - true_legal_count_;
    output.predicted_legal_count = predicted_legal_count_;
    output.predicted_illegal_count = output.total_actions - predicted_legal_count_;
    output.true_positive = true_positive_;
    output.false_positive = false_positive_;
    output.true_negative = true_negative_;
    output.false_negative = false_negative_;
    output.wait_legal_count = wait_legal_count_;
    output.wait_true_positive = wait_true_positive_;

    output.legal_recall = ratio_or_zero(
        true_positive_, true_positive_ + false_negative_);
    output.legal_precision = ratio_or_zero(
        true_positive_, true_positive_ + false_positive_);
    output.illegal_precision = ratio_or_zero(
        true_negative_, true_negative_ + false_negative_);
    output.false_negative_rate = ratio_or_zero(
        false_negative_, true_positive_ + false_negative_);
    output.false_positive_rate = ratio_or_zero(
        false_positive_, true_negative_ + false_positive_);
    output.wait_recall = ratio_or_zero(
        wait_true_positive_, wait_legal_count_);
    return output;
}

} // namespace tdmz
