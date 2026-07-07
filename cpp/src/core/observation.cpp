#include "tdmz/core/observation.hpp"
#include <cmath>

namespace tdmz {

Observation make_observation_python_parity(const TDEngine& env) {
    int w = env.width();
    int h = env.height();
    Observation obs(5 * h * w, 0.0f);
    
    auto at = [&](int c, int y, int x) -> float& {
        return obs[(c * h + y) * w + x];
    };
    
    // Channel 0: Towers
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (env.grid()[y][x] == 1) {
                at(0, y, x) = 1.0f;
            }
        }
    }
    
    // Channel 1: Start and End map
    at(1, env.spawn_y(), env.spawn_x()) = 1.0f;
    at(1, env.base_y(), env.base_x()) = 1.0f;
    
    // Channel 2: Enemies HP map
    for (const auto& enemy : env.enemies()) {
        int grid_x = static_cast<int>(enemy.x); // Python uses int(enemy['x'])
        int grid_y = static_cast<int>(enemy.y); // Python uses int(enemy['y'])
        
        if (grid_x >= 0 && grid_x < w && grid_y >= 0 && grid_y < h) {
            float added_hp = enemy.hp / std::max(1.0f, enemy.max_hp);
            at(2, grid_y, grid_x) = std::min(1.0f, at(2, grid_y, grid_x) + added_hp);
        }
    }
    
    // Channel 3: Base HP
    float base_hp_val = env.base_hp() / 100.0f;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            at(3, y, x) = base_hp_val;
        }
    }
    
    // Channel 4: Money
    float money_val = std::min(1.0f, env.money() / 1000.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            at(4, y, x) = money_val;
        }
    }
    
    return obs;
}

std::array<int, 3> observation_shape_python_parity() {
    return {5, kBoardH, kBoardW};
}

Observation make_observation_v1(const TDEngine& env) {
    int w = env.width();
    int h = env.height();
    Observation obs(OBS_CHANNELS * h * w, 0.0f);
    // (Implementation of v1 omitted for brevity during Phase 1 parity check, will implement when switching network)
    return obs;
}

std::array<int, 3> observation_shape_v1() {
    return {OBS_CHANNELS, kBoardH, kBoardW};
}

} // namespace tdmz
