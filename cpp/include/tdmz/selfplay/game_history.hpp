#pragma once
#include <cstdint>
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

struct GameHistory {
    uint64_t seed = 0;
    int max_steps = 0;
    bool terminal = false;
    float total_reward = 0.0f;
    WaveMode wave_mode = WaveMode::Unknown;
    std::vector<TrajectoryStep> steps;
};

} // namespace tdmz
