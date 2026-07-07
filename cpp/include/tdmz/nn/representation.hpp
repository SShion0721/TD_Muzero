#pragma once
#include <torch/torch.h>
#include "tdmz/nn/network_config.hpp"

namespace tdmz {

struct RepresentationNetworkImpl : torch::nn::Module {
    explicit RepresentationNetworkImpl(const NetworkConfig& cfg);
    torch::Tensor forward(torch::Tensor observation);

    NetworkConfig cfg_;
    torch::nn::Sequential net{nullptr};
};
TORCH_MODULE(RepresentationNetwork);

} // namespace tdmz
