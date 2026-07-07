#include "tdmz/nn/action_encoder.hpp"
#include "tdmz/core/action.hpp"

namespace tdmz {

ActionEncoderImpl::ActionEncoderImpl(const NetworkConfig& cfg) : cfg_(cfg) {
    fc = register_module("fc", torch::nn::Linear(4, cfg_.action_embedding_dim));
}

torch::Tensor ActionEncoderImpl::forward(const std::vector<int>& actions, torch::Device device) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor features = torch::zeros({static_cast<long long>(actions.size()), 4}, opts);
    auto acc = features.accessor<float, 2>();

    for (int i = 0; i < static_cast<int>(actions.size()); ++i) {
        Action a = decode_action(actions[i]);
        acc[i][0] = static_cast<float>(static_cast<int>(a.type)) / 6.0f;
        acc[i][1] = a.x < 0 ? 1.0f : static_cast<float>(a.x) / static_cast<float>(kBoardW - 1);
        acc[i][2] = a.y < 0 ? 1.0f : static_cast<float>(a.y) / static_cast<float>(kBoardH - 1);
        acc[i][3] = static_cast<float>(a.wait_steps);
    }

    return torch::relu(fc->forward(features.to(device)));
}

} // namespace tdmz
