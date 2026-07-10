#pragma once
#include <torch/torch.h>

namespace tdmz {

struct NetworkOutput {
    torch::Tensor latent_state;
    torch::Tensor value;
    torch::Tensor reward;
    torch::Tensor policy_logits;
    torch::Tensor legality_logits;
};

} // namespace tdmz
