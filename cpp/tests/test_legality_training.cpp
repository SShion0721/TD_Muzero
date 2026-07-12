#include "tdmz/training/legality.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tdmz;

namespace {

void check(bool condition, const char* expression, int line) {
    if (!condition) {
        throw std::runtime_error(
            std::string("check failed at line ") + std::to_string(line) + ": " + expression);
    }
}
#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

void check_close(double actual, double expected, double tolerance, const char* label) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            std::string(label) + " actual=" + std::to_string(actual)
            + " expected=" + std::to_string(expected));
    }
}

template <typename Fn>
void check_invalid(Fn&& fn) {
    bool threw = false;
    try {
        fn();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

std::vector<uint8_t> make_mask() {
    std::vector<uint8_t> mask(kActionSpaceSize, 0u);
    mask[0] = 1u;
    mask[10] = 1u;
    mask[kFlatWaitOffset] = 1u;
    return mask;
}

void test_exact_target_conversion() {
    const auto mask = make_mask();
    const auto targets = make_legality_targets(mask);
    CHECK(targets.size() == kActionSpaceSize);
    for (int action = 0; action < kActionSpaceSize; ++action) {
        CHECK(targets[static_cast<size_t>(action)]
            == (mask[static_cast<size_t>(action)] != 0u ? 1.0f : 0.0f));
    }

    std::vector<uint8_t> batch_masks;
    batch_masks.insert(batch_masks.end(), mask.begin(), mask.end());
    batch_masks.insert(batch_masks.end(), mask.begin(), mask.end());
    const auto batch_targets = make_legality_targets(batch_masks, 2);
    CHECK(batch_targets.size() == static_cast<size_t>(2 * kActionSpaceSize));
    CHECK(batch_targets[kFlatWaitOffset] == 1.0f);
    CHECK(batch_targets[kActionSpaceSize + kFlatWaitOffset] == 1.0f);
}

void test_invalid_masks_are_rejected() {
    check_invalid([] {
        (void)make_legality_targets(std::vector<uint8_t>(5, 0u));
    });

    auto non_binary = make_mask();
    non_binary[5] = 2u;
    check_invalid([&] { (void)make_legality_targets(non_binary); });

    auto missing_wait = make_mask();
    missing_wait[kFlatWaitOffset] = 0u;
    check_invalid([&] { (void)make_legality_targets(missing_wait); });

    check_invalid([&] { (void)make_legality_targets(make_mask(), 0); });
}

void test_metrics_confusion_counts_and_rates() {
    const auto mask = make_mask();
    std::vector<float> logits(kActionSpaceSize, -1.0f);
    logits[0] = 1.0f;                  // true positive
    logits[10] = -1.0f;                // false negative
    logits[20] = 1.0f;                 // false positive
    logits[kFlatWaitOffset] = 2.0f;    // true positive / Wait recall

    LegalityMetricsAccumulator accumulator;
    accumulator.update(logits, mask);
    const auto metrics = accumulator.metrics();

    CHECK(metrics.samples == 1u);
    CHECK(metrics.total_actions == static_cast<uint64_t>(kActionSpaceSize));
    CHECK(metrics.true_legal_count == 3u);
    CHECK(metrics.true_illegal_count == 724u);
    CHECK(metrics.predicted_legal_count == 3u);
    CHECK(metrics.predicted_illegal_count == 724u);
    CHECK(metrics.true_positive == 2u);
    CHECK(metrics.false_negative == 1u);
    CHECK(metrics.false_positive == 1u);
    CHECK(metrics.true_negative == 723u);
    CHECK(metrics.wait_legal_count == 1u);
    CHECK(metrics.wait_true_positive == 1u);

    check_close(metrics.legal_recall, 2.0 / 3.0, 1e-12, "legal recall");
    check_close(metrics.legal_precision, 2.0 / 3.0, 1e-12, "legal precision");
    check_close(metrics.illegal_precision, 723.0 / 724.0, 1e-12, "illegal precision");
    check_close(metrics.false_negative_rate, 1.0 / 3.0, 1e-12, "false negative rate");
    check_close(metrics.false_positive_rate, 1.0 / 724.0, 1e-12, "false positive rate");
    check_close(metrics.wait_recall, 1.0, 1e-12, "wait recall");
    CHECK(!metrics.hard_pruning_ready());
}

void test_batch_aggregation_and_pruning_gate() {
    const auto mask = make_mask();
    std::vector<uint8_t> masks;
    masks.insert(masks.end(), mask.begin(), mask.end());
    masks.insert(masks.end(), mask.begin(), mask.end());

    std::vector<float> perfect(static_cast<size_t>(2 * kActionSpaceSize), -10.0f);
    for (int batch = 0; batch < 2; ++batch) {
        const size_t offset = static_cast<size_t>(batch) * kActionSpaceSize;
        for (int action = 0; action < kActionSpaceSize; ++action) {
            if (mask[static_cast<size_t>(action)] != 0u) {
                perfect[offset + static_cast<size_t>(action)] = 10.0f;
            }
        }
    }

    LegalityMetricsAccumulator accumulator;
    accumulator.update_batch(perfect, masks, 2);
    auto metrics = accumulator.metrics();
    CHECK(metrics.samples == 2u);
    CHECK(metrics.legal_recall == 1.0);
    CHECK(metrics.wait_recall == 1.0);
    CHECK(metrics.hard_pruning_ready());

    perfect[kFlatWaitOffset] = -10.0f;
    accumulator.reset();
    accumulator.update_batch(perfect, masks, 2);
    metrics = accumulator.metrics();
    CHECK(metrics.wait_recall == 0.5);
    CHECK(!metrics.hard_pruning_ready());

    const auto before_invalid = accumulator.metrics();
    check_invalid([&] {
        accumulator.update(
            std::vector<float>(kActionSpaceSize, std::numeric_limits<float>::quiet_NaN()),
            mask);
    });
    const auto after_invalid = accumulator.metrics();
    CHECK(after_invalid.samples == before_invalid.samples);
    CHECK(after_invalid.true_positive == before_invalid.true_positive);
    CHECK(after_invalid.false_positive == before_invalid.false_positive);
    CHECK(after_invalid.true_negative == before_invalid.true_negative);
    CHECK(after_invalid.false_negative == before_invalid.false_negative);
    CHECK(after_invalid.wait_true_positive == before_invalid.wait_true_positive);
}

} // namespace

int main() {
    try {
        test_exact_target_conversion();
        test_invalid_masks_are_rejected();
        test_metrics_confusion_counts_and_rates();
        test_batch_aggregation_and_pruning_gate();
        std::cout << "Legality training utility tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
