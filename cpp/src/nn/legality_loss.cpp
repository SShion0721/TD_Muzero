#include "tdmz/nn/legality_loss.hpp"
#include "tdmz/core/action.hpp"

#include <stdexcept>

namespace tdmz {

torch::Tensor legality_bce_with_logits(
    const torch::Tensor& legality_logits,
    const torch::Tensor& legality_targets
) {
    if (!legality_logits.defined() || !legality_targets.defined()) {
        throw std::invalid_argument("Legality logits and targets must be defined");
    }
    if (legality_logits.dim() != 2
        || legality_logits.size(1) != kActionSpaceSize) {
        throw std::invalid_argument("Legality logits must have shape [B,727]");
    }
    if (legality_targets.sizes() != legality_logits.sizes()) {
        throw std::invalid_argument("Legality targets must match logits shape [B,727]");
    }
    if (!legality_logits.is_floating_point()) {
        throw std::invalid_argument("Legality logits must use a floating dtype");
    }
    if (legality_logits.size(0) <= 0) {
        throw std::invalid_argument("Legality batch must not be empty");
    }

    const auto targets = legality_targets.to(legality_logits.options());
    // max(x,0) - x*y + log(1 + exp(-abs(x)))
    const auto element_loss = torch::relu(legality_logits)
        - legality_logits * targets
        + torch::log1p(torch::exp(-torch::abs(legality_logits)));
    return element_loss.mean();
}

} // namespace tdmz
