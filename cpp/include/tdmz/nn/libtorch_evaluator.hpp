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

    EvalOutput initial_inference(const std::vector<std::vector<float>>& observations) override;
    EvalOutput recurrent_inference(const EvalInput& input) override;

    void clear_latents();
    bool has_latent(int node_id) const;

private:
    torch::Tensor observations_to_tensor(const std::vector<std::vector<float>>& observations) const;
    EvalOutput to_eval_output(const NetworkOutput& out) const;

    MuZeroNetwork network_;
    torch::Device device_;
    std::unordered_map<int, torch::Tensor> latent_by_node_;
};

} // namespace tdmz
