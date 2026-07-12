#include "tdmz/nn/libtorch_evaluator.hpp"
#include "tdmz/core/action.hpp"
#include "tdmz/core/observation.hpp"
#include <cmath>
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
    if (observations.empty()) throw std::invalid_argument("observation batch must not be empty");
    int batch = static_cast<int>(observations.size());
    int expected = OBS_CHANNELS * kBoardH * kBoardW;
    std::vector<float> flat;
    flat.reserve(static_cast<size_t>(batch) * expected);
    for (const auto& obs : observations) {
        if (static_cast<int>(obs.size()) != expected) throw std::invalid_argument("bad observation size");
        flat.insert(flat.end(), obs.begin(), obs.end());
    }
    return torch::tensor(flat, torch::TensorOptions().dtype(torch::kFloat32))
        .view({batch, OBS_CHANNELS, kBoardH, kBoardW}).to(device_);
}

EvalOutput LibTorchEvaluator::to_eval_output(const NetworkOutput& out) const {
    if (!out.value.defined()
        || !out.reward.defined()
        || !out.policy_logits.defined()
        || !out.legality_logits.defined()) {
        throw std::runtime_error("network returned an undefined output tensor");
    }
    if (out.policy_logits.dim() != 2 || out.policy_logits.size(1) != kActionSpaceSize) {
        throw std::runtime_error("network policy tensor has an invalid shape");
    }
    if (out.legality_logits.dim() != 2 || out.legality_logits.size(1) != kActionSpaceSize) {
        throw std::runtime_error("network legality tensor has an invalid shape");
    }
    if (out.legality_logits.size(0) != out.policy_logits.size(0)) {
        throw std::runtime_error("network legality batch does not match policy batch");
    }

    auto values = out.value.detach().to(torch::kCPU).view({-1}).contiguous();
    auto rewards = out.reward.detach().to(torch::kCPU).view({-1}).contiguous();
    auto policy = out.policy_logits.detach().to(torch::kCPU).contiguous();
    auto legality = out.legality_logits.detach().to(torch::kCPU).contiguous();
    const int batch = static_cast<int>(policy.size(0));
    if (values.size(0) != batch || rewards.size(0) != batch) {
        throw std::runtime_error("network value/reward batch does not match policy batch");
    }

    EvalOutput eval;
    eval.values.resize(batch);
    eval.rewards.resize(batch);
    eval.policy_logits.resize(batch, std::vector<float>(kActionSpaceSize, 0.0f));
    eval.legality_logits.resize(batch, std::vector<float>(kActionSpaceSize, 0.0f));
    auto v = values.accessor<float, 1>();
    auto r = rewards.accessor<float, 1>();
    auto p = policy.accessor<float, 2>();
    auto l = legality.accessor<float, 2>();
    for (int b = 0; b < batch; ++b) {
        if (!std::isfinite(v[b]) || !std::isfinite(r[b])) {
            throw std::runtime_error("network returned a non-finite value or reward");
        }
        eval.values[b] = v[b];
        eval.rewards[b] = r[b];
        for (int a = 0; a < kActionSpaceSize; ++a) {
            if (!std::isfinite(p[b][a])) {
                throw std::runtime_error("network returned a non-finite policy logit");
            }
            if (!std::isfinite(l[b][a])) {
                throw std::runtime_error("network returned a non-finite legality logit");
            }
            eval.policy_logits[b][a] = p[b][a];
            eval.legality_logits[b][a] = l[b][a];
        }
    }
    return eval;
}

EvalOutput LibTorchEvaluator::initial_inference_impl(
    const std::vector<std::vector<float>>& observations
) {
    torch::NoGradGuard guard;
    if (observations.size() != 1) {
        throw std::invalid_argument(
            "LibTorchEvaluator initial_inference currently supports one root; batch roots need explicit node IDs");
    }

    clear_latents();
    auto out = network_->initial_inference(observations_to_tensor(observations));
    if (!out.latent_state.defined() || out.latent_state.dim() != 4 || out.latent_state.size(0) != 1) {
        throw std::runtime_error("network latent tensor has an invalid initial shape");
    }
    latent_by_node_[0] = out.latent_state[0].unsqueeze(0).detach();
    return to_eval_output(out);
}

EvalOutput LibTorchEvaluator::recurrent_inference_impl(const EvalInput& input) {
    torch::NoGradGuard guard;
    if (input.batch_size <= 0) throw std::invalid_argument("bad batch size");
    const size_t batch = static_cast<size_t>(input.batch_size);
    if (input.parent_node_ids.size() != batch ||
        input.target_node_ids.size() != batch ||
        input.actions.size() != batch) {
        throw std::invalid_argument("recurrent input vector sizes do not match batch size");
    }

    std::vector<torch::Tensor> latents;
    latents.reserve(batch);
    for (int i = 0; i < input.batch_size; ++i) {
        auto it = latent_by_node_.find(input.parent_node_ids[i]);
        if (it == latent_by_node_.end()) throw std::runtime_error("missing parent latent");
        latents.push_back(it->second);
    }

    auto out = network_->recurrent_inference(torch::cat(latents, 0).to(device_), input.actions);
    if (!out.latent_state.defined() || out.latent_state.dim() != 4 || out.latent_state.size(0) != input.batch_size) {
        throw std::runtime_error("network latent tensor has an invalid recurrent shape");
    }
    for (int i = 0; i < input.batch_size; ++i) {
        latent_by_node_[input.target_node_ids[i]] = out.latent_state[i].unsqueeze(0).detach();
    }
    return to_eval_output(out);
}

} // namespace tdmz
