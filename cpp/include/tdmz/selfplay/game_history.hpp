#pragma once
#include <cstdint>
#include <optional>
#include <vector>
#include "tdmz/core/observation.hpp"
#include "tdmz/core/wave_mode.hpp"

namespace tdmz {

struct TrajectoryStep {
    int step_index = 0;
    int action = -1;
    float reward = 0.0f;
    float root_value = 0.0f;
    bool done = false;

    int money = 0;
    int base_hp = 0;
    int wave = 0;
    float time = 0.0f;

    Observation observation;
    std::vector<float> policy_target;
    std::vector<uint8_t> legal_mask;
};

// Real root state captured after the final stored transition of a truncated
// episode. It supplies an explicit cutoff bootstrap instead of treating the
// time limit as a zero-value terminal state.
struct BootstrapState {
    float root_value = 0.0f;
    Observation observation;
    std::vector<float> policy_target;
    std::vector<uint8_t> legal_mask;
};

struct GameHistory {
    uint64_t seed = 0;
    int max_steps = 0;
    bool terminal = false;
    bool truncated = false;
    float total_reward = 0.0f;
    WaveMode wave_mode = WaveMode::Unknown;
    std::vector<TrajectoryStep> steps;
    std::optional<BootstrapState> bootstrap_state;

    bool completed() const { return terminal || truncated; }
};

} // namespace tdmz
