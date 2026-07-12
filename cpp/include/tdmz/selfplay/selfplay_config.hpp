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
    // When a game reaches max_steps without terminating, run one additional
    // root search at the cutoff state and persist its observation/value/policy/
    // legal mask for unbiased truncated n-step bootstrapping.
    bool save_bootstrap_state = false;
    MCTSConfig mcts;
};

} // namespace tdmz
