#pragma once
#include <torch/torch.h>
#include "tdmz/nn/network_config.hpp"

namespace tdmz {

struct PredictionOutput {
    torch::Tensor value;
    torch::Tensor policy_logits;
    torch::Tensor legality_logits;
};

struct PredictionNetworkImpl : torch::nn::Module {
    explicit PredictionNetworkImpl(const NetworkConfig& cfg);
    PredictionOutput forward(torch::Tensor latent);
    NetworkConfig cfg_;
    torch::nn::AdaptiveAvgPool2d pool{nullptr};
    torch::nn::Linear fc{nullptr};
    torch::nn::Linear value{nullptr};
    torch::nn::Linear policy{nullptr};
    torch::nn::Linear legality{nullptr};
};
TORCH_MODULE(PredictionNetwork);

} // namespace tdmz
