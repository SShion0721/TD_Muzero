#include "tdmz/nn/prediction.hpp"
#include <stdexcept>
#include <vector>

namespace tdmz {

torch::Tensor flatten_spatial_action_logits(
    torch::Tensor spatial_logits,
    torch::Tensor wait_logits
) {
    if (!spatial_logits.defined() || spatial_logits.dim() != 4 ||
        spatial_logits.size(1) != kSpatialPolicyPlanes ||
        spatial_logits.size(2) != kBoardH ||
        spatial_logits.size(3) != kBoardW) {
        throw std::invalid_argument("Spatial action logits have an invalid shape");
    }
    if (!wait_logits.defined() || wait_logits.dim() != 2 ||
        wait_logits.size(0) != spatial_logits.size(0) ||
        wait_logits.size(1) != 1) {
        throw std::invalid_argument("Wait logits have an invalid shape");
    }

    auto spatial_flat = spatial_logits.contiguous().view({spatial_logits.size(0), kFlatWaitOffset});
    return torch::cat({spatial_flat, wait_logits}, 1);
}

PredictionNetworkImpl::PredictionNetworkImpl(const NetworkConfig& cfg) : cfg_(cfg) {
    if (cfg_.board_h != kBoardH || cfg_.board_w != kBoardW ||
        cfg_.policy_planes != kSpatialPolicyPlanes ||
        cfg_.action_space_size != kActionSpaceSize) {
        throw std::invalid_argument("PredictionNetwork configuration does not match the action ABI");
    }

    trunk = register_module("trunk", torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(
            cfg_.latent_channels, cfg_.hidden_channels, 3).padding(1)),
        torch::nn::ReLU()
    ));
    pool = register_module("pool", torch::nn::AdaptiveAvgPool2d(
        torch::nn::AdaptiveAvgPool2dOptions(std::vector<int64_t>{1, 1})));
    value = register_module("value", torch::nn::Linear(cfg_.hidden_channels, cfg_.value_dim));
    policy_spatial = register_module("policy_spatial", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(cfg_.hidden_channels, cfg_.policy_planes, 1)));
    policy_wait = register_module("policy_wait", torch::nn::Linear(cfg_.hidden_channels, 1));
    legality_spatial = register_module("legality_spatial", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(cfg_.hidden_channels, cfg_.policy_planes, 1)));
    legality_wait = register_module("legality_wait", torch::nn::Linear(cfg_.hidden_channels, 1));
}

PredictionOutput PredictionNetworkImpl::forward(torch::Tensor latent) {
    auto features = trunk->forward(latent);
    auto pooled = pool->forward(features).flatten(1);
    auto policy_logits = flatten_spatial_action_logits(
        policy_spatial->forward(features),
        policy_wait->forward(pooled));
    auto legality_logits = flatten_spatial_action_logits(
        legality_spatial->forward(features),
        legality_wait->forward(pooled));
    return {value->forward(pooled), policy_logits, legality_logits};
}

} // namespace tdmz
