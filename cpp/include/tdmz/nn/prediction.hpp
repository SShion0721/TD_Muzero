#pragma once
#include <torch/torch.h>
#include "tdmz/nn/network_config.hpp"

namespace tdmz {

struct PredictionOutput {
    torch::Tensor value;
    torch::Tensor policy_logits;
    torch::Tensor legality_logits;
};

// Flattens six [H,W] spatial action planes in channel-major order and appends
// one wait scalar, preserving the public 727-action ABI exactly.
torch::Tensor flatten_spatial_action_logits(
    torch::Tensor spatial_logits,
    torch::Tensor wait_logits
);

struct PredictionNetworkImpl : torch::nn::Module {
    explicit PredictionNetworkImpl(const NetworkConfig& cfg);
    PredictionOutput forward(torch::Tensor latent);

    NetworkConfig cfg_;
    torch::nn::Sequential trunk{nullptr};
    torch::nn::AdaptiveAvgPool2d pool{nullptr};
    torch::nn::Linear value{nullptr};
    torch::nn::Conv2d policy_spatial{nullptr};
    torch::nn::Linear policy_wait{nullptr};
    torch::nn::Conv2d legality_spatial{nullptr};
    torch::nn::Linear legality_wait{nullptr};
};
TORCH_MODULE(PredictionNetwork);

} // namespace tdmz
