#pragma once
#include <torch/torch.h>
#include "tdmz/nn/action_encoder.hpp"
#include "tdmz/nn/network_config.hpp"

namespace tdmz {

struct DynamicsOutput {
    torch::Tensor latent_state;
    torch::Tensor reward;
};

struct DynamicsNetworkImpl : torch::nn::Module {
    explicit DynamicsNetworkImpl(const NetworkConfig& cfg);
    DynamicsOutput forward(torch::Tensor latent, const std::vector<int>& actions);

    NetworkConfig cfg_;
    ActionEncoder action_encoder{nullptr};
    torch::nn::Sequential conv{nullptr};
    torch::nn::AdaptiveAvgPool2d pool{nullptr};
    torch::nn::Linear reward_head{nullptr};
};
TORCH_MODULE(DynamicsNetwork);

} // namespace tdmz
