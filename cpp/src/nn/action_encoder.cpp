#include "tdmz/nn/action_encoder.hpp"
#include "tdmz/core/action.hpp"
#include <stdexcept>

namespace tdmz {

ActionEncoderImpl::ActionEncoderImpl(const NetworkConfig& cfg) : cfg_(cfg) {
    if (cfg_.board_h != kBoardH || cfg_.board_w != kBoardW) {
        throw std::invalid_argument("ActionEncoder requires the fixed 11x11 board");
    }
    if (cfg_.action_planes != kSpatialActionPlanes) {
        throw std::invalid_argument("ActionEncoder action plane count does not match the action ABI");
    }
}

torch::Tensor ActionEncoderImpl::forward(const std::vector<int>& actions, torch::Device device) {
    auto planes = torch::zeros(
        {static_cast<long long>(actions.size()), cfg_.action_planes, cfg_.board_h, cfg_.board_w},
        torch::TensorOptions().dtype(torch::kFloat32));
    auto acc = planes.accessor<float, 4>();

    for (int i = 0; i < static_cast<int>(actions.size()); ++i) {
        const Action action = decode_action(actions[static_cast<size_t>(i)]);
        const int plane = static_cast<int>(action.type);
        if (plane < 0 || plane >= cfg_.action_planes) {
            throw std::runtime_error("Decoded action has an invalid spatial plane");
        }

        if (action.type == ActionType::Wait1) {
            for (int y = 0; y < cfg_.board_h; ++y) {
                for (int x = 0; x < cfg_.board_w; ++x) {
                    acc[i][plane][y][x] = 1.0f;
                }
            }
        } else {
            if (action.x < 0 || action.x >= cfg_.board_w ||
                action.y < 0 || action.y >= cfg_.board_h) {
                throw std::runtime_error("Decoded spatial action cell is out of bounds");
            }
            acc[i][plane][action.y][action.x] = 1.0f;
        }
    }

    return planes.to(device);
}

} // namespace tdmz
