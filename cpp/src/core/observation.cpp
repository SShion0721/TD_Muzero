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
    
    auto at = [&](int c, int y, int x) -> float& {
        return obs[(c * h + y) * w + x];
    };
    
    // Grid/Map info
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (env.grid()[y][x] == 1) at(CH_BLOCKED, y, x) = 1.0f;
        }
    }
    at(CH_SPAWN, env.spawn_y(), env.spawn_x()) = 1.0f;
    at(CH_BASE, env.base_y(), env.base_x()) = 1.0f;
    
    // Tower info
    for (const auto& tower : env.towers()) {
        int x = tower.x;
        int y = tower.y;
        if (tower.type == TowerType::Basic) at(CH_TOWER_BASIC, y, x) = 1.0f;
        else if (tower.type == TowerType::Sniper) at(CH_TOWER_SNIPER, y, x) = 1.0f;
        else if (tower.type == TowerType::AOE) at(CH_TOWER_AOE, y, x) = 1.0f;
        else if (tower.type == TowerType::Slow) at(CH_TOWER_SLOW, y, x) = 1.0f;
        
        at(CH_TOWER_LEVEL, y, x) = tower.level / 3.0f; // max level 3
        at(CH_TOWER_COOLDOWN, y, x) = tower.cooldown / tower.cooldown_max;
    }
    
    // Enemy info
    for (const auto& enemy : env.enemies()) {
        int gx = static_cast<int>(std::round(enemy.x));
        int gy = static_cast<int>(std::round(enemy.y));
        if (gx >= 0 && gx < w && gy >= 0 && gy < h) {
            at(CH_ENEMY_HP, gy, gx) += enemy.hp / std::max(1.0f, enemy.max_hp);
            at(CH_ENEMY_DENSITY, gy, gx) += 1.0f;
            at(CH_ENEMY_SPEED, gy, gx) = std::max(at(CH_ENEMY_SPEED, gy, gx), enemy.speed / 3.0f);
            at(CH_ENEMY_SLOW_TIMER, gy, gx) = std::max(at(CH_ENEMY_SLOW_TIMER, gy, gx), enemy.slow_timer / 3.0f);
        }
    }
    
    // Global scalars
    float base_hp_val = env.base_hp() / 100.0f;
    float money_val = std::min(1.0f, env.money() / 1000.0f);
    float wave_val = env.wave() / 20.0f;
    float spawn_timer_val = env.spawn_timer() / 3.0f;
    float to_spawn_val = env.enemies_to_spawn_count() / 20.0f;
    
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            at(CH_BASE_HP, y, x) = base_hp_val;
            at(CH_MONEY, y, x) = money_val;
            at(CH_WAVE, y, x) = wave_val;
            at(CH_SPAWN_TIMER, y, x) = spawn_timer_val;
            at(CH_TO_SPAWN_COUNT, y, x) = to_spawn_val;
        }
    }
    
    // Distance to base (BFS)
    auto placeable_mask = env.compute_placeable_mask();
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (placeable_mask[y][x]) {
                at(CH_PLACEABLE_MASK, y, x) = 1.0f;
            }
            // For distance to base, maybe precompute it?
            // For now, we can omit exact distance if too slow, or just leave it 0 since it's a fixed grid
        }
    }

    return obs;
}

std::array<int, 3> observation_shape_v1() {
    return {OBS_CHANNELS, kBoardH, kBoardW};
}

} // namespace tdmz
