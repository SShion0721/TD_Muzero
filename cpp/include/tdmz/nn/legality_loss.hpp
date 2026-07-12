#pragma once

#include <torch/torch.h>

namespace tdmz {

// Numerically stable binary cross entropy over [B,727] legality logits.
// Targets are expected to contain exact engine labels in {0,1}; integer masks
// are accepted and converted to the logits' device and floating dtype.
torch::Tensor legality_bce_with_logits(
    const torch::Tensor& legality_logits,
    const torch::Tensor& legality_targets
);

} // namespace tdmz
