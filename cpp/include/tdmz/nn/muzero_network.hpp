#pragma once
#include <torch/torch.h>
#include "tdmz/nn/dynamics.hpp"
#include "tdmz/nn/network_config.hpp"
#include "tdmz/nn/network_output.hpp"
#include "tdmz/nn/prediction.hpp"
#include "tdmz/nn/representation.hpp"

namespace tdmz {

struct MuZeroNetworkImpl : torch::nn::Module {
    explicit MuZeroNetworkImpl(const NetworkConfig& cfg = NetworkConfig{});

    NetworkOutput initial_inference(torch::Tensor observation);
    NetworkOutput recurrent_inference(torch::Tensor latent, const std::vector<int>& actions);

    NetworkConfig cfg_;
    RepresentationNetwork representation{nullptr};
    DynamicsNetwork dynamics{nullptr};
    PredictionNetwork prediction{nullptr};
};
TORCH_MODULE(MuZeroNetwork);

} // namespace tdmz
