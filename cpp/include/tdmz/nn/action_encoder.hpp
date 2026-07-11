#pragma once
#include <torch/torch.h>
#include <vector>
#include "tdmz/nn/network_config.hpp"

namespace tdmz {

struct ActionEncoderImpl : torch::nn::Module {
    explicit ActionEncoderImpl(const NetworkConfig& cfg);

    // Returns [B, 7, H, W]. Build/upgrade/sell actions are one-hot at their
    // target cell. Wait is represented by a full constant wait plane.
    torch::Tensor forward(const std::vector<int>& actions, torch::Device device);

    NetworkConfig cfg_;
};
TORCH_MODULE(ActionEncoder);

} // namespace tdmz
