#pragma once
#include <cstdint>
#include "tdmz/mcts/mcts_config.hpp"

namespace tdmz {

struct SelfPlayConfig {
    uint64_t seed = 0;
    int max_steps = 256;
    bool stop_on_terminal = true;
    bool save_observations = true;
    bool save_legal_mask = true;
    MCTSConfig mcts;
};

} // namespace tdmz
