#include "tdmz/mcts/mcts.hpp"
#include "tdmz/mcts/action_mask.hpp"
#include "tdmz/core/action.hpp"
#include <stdexcept>

namespace tdmz {

RootSearchOutput MCTS::search_single(
    INetworkEvaluator& net,
    const std::vector<float>& observation,
    const std::vector<uint8_t>& legal_mask
) {
    if (legal_mask.size() != static_cast<std::size_t>(kActionSpaceSize)) {
        throw std::invalid_argument("MCTS root action mask has an invalid size");
    }
    return search_single(net, observation, legal_actions_from_mask(legal_mask));
}

} // namespace tdmz
