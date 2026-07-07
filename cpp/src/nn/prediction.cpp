#include "tdmz/nn/prediction.hpp"

namespace tdmz {

PredictionNetworkImpl::PredictionNetworkImpl(const NetworkConfig& cfg) : cfg_(cfg) {
    pool = register_module("pool", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(std::vector<int64_t>{1, 1})));
    fc = register_module("fc", torch::nn::Linear(cfg_.latent_channels, cfg_.hidden_channels));
    value = register_module("value", torch::nn::Linear(cfg_.hidden_channels, cfg_.value_dim));
    policy = register_module("policy", torch::nn::Linear(cfg_.hidden_channels, cfg_.action_space_size));
    legality = register_module("legality", torch::nn::Linear(cfg_.hidden_channels, cfg_.action_space_size));
}

PredictionOutput PredictionNetworkImpl::forward(torch::Tensor latent) {
    auto x = pool->forward(latent).flatten(1);
    x = torch::relu(fc->forward(x));
    return {value->forward(x), policy->forward(x), legality->forward(x)};
}

} // namespace tdmz
