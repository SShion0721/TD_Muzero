#include "tdmz/nn/muzero_network.hpp"

namespace tdmz {

MuZeroNetworkImpl::MuZeroNetworkImpl(const NetworkConfig& cfg) : cfg_(cfg) {
    representation = register_module("representation", RepresentationNetwork(cfg_));
    dynamics = register_module("dynamics", DynamicsNetwork(cfg_));
    prediction = register_module("prediction", PredictionNetwork(cfg_));
}

NetworkOutput MuZeroNetworkImpl::initial_inference(torch::Tensor observation) {
    auto latent = representation->forward(observation);
    auto pred = prediction->forward(latent);
    auto reward = torch::zeros({observation.size(0), cfg_.reward_dim}, observation.options());
    return {latent, pred.value, reward, pred.policy_logits, pred.legality_logits};
}

NetworkOutput MuZeroNetworkImpl::recurrent_inference(torch::Tensor latent, const std::vector<int>& actions) {
    auto dyn = dynamics->forward(latent, actions);
    auto pred = prediction->forward(dyn.latent_state);
    return {dyn.latent_state, pred.value, dyn.reward, pred.policy_logits, pred.legality_logits};
}

} // namespace tdmz
