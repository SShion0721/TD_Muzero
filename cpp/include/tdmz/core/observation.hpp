#pragma once
#include <array>
#include <vector>
#include "tdmz/core/engine.hpp"

namespace tdmz {

enum ObsChannel {
    CH_BLOCKED = 0,
    CH_SPAWN,
    CH_BASE,

    CH_TOWER_BASIC,
    CH_TOWER_SNIPER,
    CH_TOWER_AOE,
    CH_TOWER_SLOW,

    CH_TOWER_LEVEL,
    CH_TOWER_COOLDOWN,

    CH_ENEMY_HP,
    CH_ENEMY_DENSITY,
    CH_ENEMY_SPEED,
    CH_ENEMY_SLOW_TIMER,

    CH_BASE_HP,
    CH_MONEY,
    CH_WAVE,
    CH_SPAWN_TIMER,
    CH_TO_SPAWN_COUNT,

    CH_DISTANCE_TO_BASE,
    CH_PLACEABLE_MASK,

    OBS_CHANNELS
};

using Observation = std::vector<float>; // C * H * W, contiguous CHW

Observation make_observation_v1(const TDEngine& env);
std::array<int, 3> observation_shape_v1(); // {OBS_CHANNELS, 11, 11}

// For Phase 1 parity check, we also need the exact Python observation
Observation make_observation_python_parity(const TDEngine& env);
std::array<int, 3> observation_shape_python_parity(); // {5, 11, 11}

} // namespace tdmz
