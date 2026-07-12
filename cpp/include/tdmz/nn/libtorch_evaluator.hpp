#pragma once
#include <torch/torch.h>
#include <unordered_map>
#include <vector>
#include "tdmz/mcts/network_evaluator.hpp"
#include "tdmz/nn/muzero_network.hpp"

namespace tdmz {

class LibTorchEvaluator : public INetworkEvaluator {
public:
    explicit LibTorchEvaluator(MuZeroNetwork network, torch::Device device = torch::kCPU);

    void clear_latents();
    bool has_latent(int node_id) const;

protected:
    EvalOutput initial_inference_impl(
        const std::vector<std::vector<float>>& observations
    ) override;
    EvalOutput recurrent_inference_impl(const EvalInput& input) override;

private:
    torch::Tensor observations_to_tensor(const std::vector<std::vector<float>>& observations) const;
    EvalOutput to_eval_output(const NetworkOutput& out) const;

    MuZeroNetwork network_;
    torch::Device device_;
    std::unordered_map<int, torch::Tensor> latent_by_node_;
};

} // namespace tdmz
