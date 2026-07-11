#pragma once
#include <array>
#include <vector>
#include "tdmz/core/engine.hpp"

namespace tdmz {

// Observation schema v2. All planes are CHW-contiguous and use bounded or
// logarithmically scaled values unless noted otherwise. Spatial enemy features
// are bilinearly splatted from continuous positions instead of rounded cells.
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

    // Structural placement ignores money and temporary enemy occupancy.
    CH_STRUCTURAL_PLACEABLE,
    CH_DISTANCE_TO_BASE,

    CH_ENEMY_DENSITY,
    CH_ENEMY_HP_RATIO,
    CH_ENEMY_CURRENT_HP,
    CH_ENEMY_MAX_HP,
    CH_ENEMY_SPEED,
    CH_ENEMY_SLOW_TIMER,
    CH_ENEMY_REWARD,
    CH_ENEMY_LEAK_DAMAGE,
    CH_ENEMY_REMAINING_DISTANCE,
    CH_ENEMY_DIR_RIGHT,
    CH_ENEMY_DIR_LEFT,
    CH_ENEMY_DIR_DOWN,
    CH_ENEMY_DIR_UP,
    CH_ENEMY_OCCUPIED,

    CH_BASE_HP,
    CH_MONEY,
    CH_WAVE,
    CH_SPAWN_TIMER,
    CH_PENDING_COUNT,
    CH_PENDING_TOTAL_HP,
    CH_PENDING_TOTAL_REWARD,
    CH_PENDING_TOTAL_LEAK_DAMAGE,
    CH_PENDING_MEAN_SPEED,
    CH_PENDING_MAX_SPEED,
    CH_NEXT_ENEMY_PRESENT,
    CH_NEXT_ENEMY_HP,
    CH_NEXT_ENEMY_SPEED,
    CH_NEXT_ENEMY_REWARD,
    CH_NEXT_ENEMY_LEAK_DAMAGE,

    OBS_CHANNELS = 40,

    // Source-compatibility aliases. Their v2 semantics are documented above.
    CH_PLACEABLE_MASK = CH_STRUCTURAL_PLACEABLE,
    CH_TO_SPAWN_COUNT = CH_PENDING_COUNT,
    CH_ENEMY_HP = CH_ENEMY_HP_RATIO,
};

static_assert(OBS_CHANNELS == 40, "Observation schema v2 channel count changed");

using Observation = std::vector<float>; // C * H * W, contiguous CHW

void make_observation_v2_into(const TDEngine& env, Observation& output);
Observation make_observation_v2(const TDEngine& env);
std::array<int, 3> observation_shape_v2(); // {OBS_CHANNELS, 11, 11}

// Transitional source-compatible names. These now produce schema v2 and should
// not be used to identify serialized data; compatibility metadata is authoritative.
inline void make_observation_v1_into(const TDEngine& env, Observation& output) {
    make_observation_v2_into(env, output);
}
inline Observation make_observation_v1(const TDEngine& env) {
    return make_observation_v2(env);
}
inline std::array<int, 3> observation_shape_v1() {
    return observation_shape_v2();
}

// Exact legacy Python observation retained only as a migration oracle.
Observation make_observation_python_parity(const TDEngine& env);
std::array<int, 3> observation_shape_python_parity(); // {5, 11, 11}

} // namespace tdmz
