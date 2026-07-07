#pragma once
#include <torch/torch.h>
#include <vector>
#include "tdmz/nn/network_config.hpp"

namespace tdmz {

struct ActionEncoderImpl : torch::nn::Module {
    explicit ActionEncoderImpl(const NetworkConfig& cfg);

    torch::Tensor forward(const std::vector<int>& actions, torch::Device device);

    NetworkConfig cfg_;
    torch::nn::Linear fc{nullptr};
};
TORCH_MODULE(ActionEncoder);

} // namespace tdmz
