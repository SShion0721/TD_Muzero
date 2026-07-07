#include "tdmz/nn/libtorch_evaluator.hpp"
#include "tdmz/core/action.hpp"
#include "tdmz/core/observation.hpp"
#include <stdexcept>

namespace tdmz {

LibTorchEvaluator::LibTorchEvaluator(MuZeroNetwork network, torch::Device device)
    : network_(network), device_(device) {
    network_->to(device_);
    network_->eval();
}

void LibTorchEvaluator::clear_latents() { latent_by_node_.clear(); }

bool LibTorchEvaluator::has_latent(int node_id) const {
    return latent_by_node_.find(node_id) != latent_by_node_.end();
}

torch::Tensor LibTorchEvaluator::observations_to_tensor(const std::vector<std::vector<float>>& observations) const {
    int batch = static_cast<int>(observations.size());
    int expected = OBS_CHANNELS * kBoardH * kBoardW;
    std::vector<float> flat;
    flat.reserve(batch * expected);
    for (const auto& obs : observations) {
        if (static_cast<int>(obs.size()) != expected) throw std::invalid_argument("bad observation size");
        flat.insert(flat.end(), obs.begin(), obs.end());
    }
    return torch::tensor(flat, torch::TensorOptions().dtype(torch::kFloat32))
        .view({batch, OBS_CHANNELS, kBoardH, kBoardW}).to(device_);
}

EvalOutput LibTorchEvaluator::to_eval_output(const NetworkOutput& out) const {
    auto values = out.value.detach().to(torch::kCPU).view({-1});
    auto rewards = out.reward.detach().to(torch::kCPU).view({-1});
    auto logits = out.policy_logits.detach().to(torch::kCPU).contiguous();
    int batch = static_cast<int>(values.size(0));
    EvalOutput eval;
    eval.values.resize(batch);
    eval.rewards.resize(batch);
    eval.policy_logits.resize(batch, std::vector<float>(kActionSpaceSize, 0.0f));
    auto v = values.accessor<float, 1>();
    auto r = rewards.accessor<float, 1>();
    auto l = logits.accessor<float, 2>();
    for (int b = 0; b < batch; ++b) {
        eval.values[b] = v[b];
        eval.rewards[b] = r[b];
        for (int a = 0; a < kActionSpaceSize; ++a) eval.policy_logits[b][a] = l[b][a];
    }
    return eval;
}

EvalOutput LibTorchEvaluator::initial_inference(const std::vector<std::vector<float>>& observations) {
    torch::NoGradGuard guard;
    clear_latents();
    auto out = network_->initial_inference(observations_to_tensor(observations));
    if (observations.size() == 1) latent_by_node_[0] = out.latent_state[0].unsqueeze(0).detach();
    return to_eval_output(out);
}

EvalOutput LibTorchEvaluator::recurrent_inference(const EvalInput& input) {
    torch::NoGradGuard guard;
    if (input.batch_size <= 0) throw std::invalid_argument("bad batch size");
    std::vector<torch::Tensor> latents;
    for (int i = 0; i < input.batch_size; ++i) {
        auto it = latent_by_node_.find(input.parent_node_ids[i]);
        if (it == latent_by_node_.end()) throw std::runtime_error("missing parent latent");
        latents.push_back(it->second);
    }
    auto out = network_->recurrent_inference(torch::cat(latents, 0).to(device_), input.actions);
    if (static_cast<int>(input.target_node_ids.size()) == input.batch_size) {
        for (int i = 0; i < input.batch_size; ++i) {
            latent_by_node_[input.target_node_ids[i]] = out.latent_state[i].unsqueeze(0).detach();
        }
    }
    return to_eval_output(out);
}

} // namespace tdmz
