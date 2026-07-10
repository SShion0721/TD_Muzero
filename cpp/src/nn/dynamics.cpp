#include "tdmz/nn/dynamics.hpp"
#include <vector>

namespace tdmz {

DynamicsNetworkImpl::DynamicsNetworkImpl(const NetworkConfig& cfg) : cfg_(cfg) {
    action_encoder = register_module("action_encoder", ActionEncoder(cfg_));
    conv = register_module("conv", torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(cfg_.latent_channels + cfg_.action_embedding_dim, cfg_.hidden_channels, 3).padding(1)),
        torch::nn::ReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(cfg_.hidden_channels, cfg_.latent_channels, 3).padding(1)),
        torch::nn::ReLU()
    ));
    pool = register_module("pool", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(std::vector<int64_t>{1, 1})));
    reward_head = register_module("reward_head", torch::nn::Linear(cfg_.latent_channels, cfg_.reward_dim));
}

DynamicsOutput DynamicsNetworkImpl::forward(torch::Tensor latent, const std::vector<int>& actions) {
    auto emb = action_encoder->forward(actions, latent.device());
    emb = emb.view({emb.size(0), emb.size(1), 1, 1}).expand({-1, -1, cfg_.board_h, cfg_.board_w});
    auto next_latent = conv->forward(torch::cat({latent, emb}, 1));
    auto reward = reward_head->forward(pool->forward(next_latent).flatten(1));
    return {next_latent, reward};
}

} // namespace tdmz
