#include "tdmz/core/action.hpp"
#include "tdmz/nn/legality_loss.hpp"
#include "tdmz/training/legality.hpp"

#include <cmath>
#include <iostream>
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

torch::Tensor make_targets() {
    std::vector<uint8_t> mask(kActionSpaceSize, 0u);
    mask[0] = 1u;
    mask[100] = 1u;
    mask[kFlatWaitOffset] = 1u;
    const auto values = make_legality_targets(mask);
    return torch::tensor(values, torch::TensorOptions().dtype(torch::kFloat32))
        .view({1, kActionSpaceSize});
}

void test_zero_logits_equal_log_two() {
    const auto targets = make_targets();
    const auto logits = torch::zeros_like(targets);
    const auto loss = legality_bce_with_logits(logits, targets);
    CHECK(loss.dim() == 0);
    CHECK(std::fabs(loss.item<double>() - std::log(2.0)) < 1e-6);
}

void test_confident_correct_logits_have_small_loss() {
    const auto targets = make_targets();
    const auto logits = targets * 40.0f - 20.0f;
    const auto loss = legality_bce_with_logits(logits, targets.to(torch::kUInt8));
    CHECK(loss.item<double>() < 1e-7);
}

void test_loss_backpropagates_finite_gradients() {
    const auto targets = make_targets();
    auto logits = torch::zeros(
        {1, kActionSpaceSize},
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
    const auto loss = legality_bce_with_logits(logits, targets);
    loss.backward();

    CHECK(logits.grad().defined());
    CHECK(logits.grad().sizes() == logits.sizes());
    CHECK(torch::isfinite(logits.grad()).all().item<bool>());
    CHECK(logits.grad().abs().sum().item<double>() > 0.0);
}

void test_invalid_shapes_are_rejected() {
    const auto targets = make_targets();
    check_invalid([&] {
        (void)legality_bce_with_logits(
            torch::zeros({1, kActionSpaceSize - 1}),
            torch::zeros({1, kActionSpaceSize - 1}));
    });
    check_invalid([&] {
        (void)legality_bce_with_logits(
            torch::zeros({1, kActionSpaceSize}),
            torch::zeros({2, kActionSpaceSize}));
    });
    check_invalid([&] {
        (void)legality_bce_with_logits(
            torch::zeros({1, kActionSpaceSize}, torch::TensorOptions().dtype(torch::kInt32)),
            targets);
    });
}

} // namespace

int main() {
    try {
        test_zero_logits_equal_log_two();
        test_confident_correct_logits_have_small_loss();
        test_loss_backpropagates_finite_gradients();
        test_invalid_shapes_are_rejected();
        std::cout << "Legality loss tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
