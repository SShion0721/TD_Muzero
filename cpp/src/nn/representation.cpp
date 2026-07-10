#include "tdmz/nn/representation.hpp"

namespace tdmz {

RepresentationNetworkImpl::RepresentationNetworkImpl(const NetworkConfig& cfg) : cfg_(cfg) {
    net = register_module("net", torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(cfg_.observation_channels, cfg_.hidden_channels, 3).padding(1)),
        torch::nn::ReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(cfg_.hidden_channels, cfg_.latent_channels, 3).padding(1)),
        torch::nn::ReLU()
    ));
}

torch::Tensor RepresentationNetworkImpl::forward(torch::Tensor observation) {
    return net->forward(observation);
}

} // namespace tdmz
