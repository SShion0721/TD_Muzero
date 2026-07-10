#include "tdmz/core/observation.hpp"
#include "tdmz/core/board_tables.hpp"
#include <algorithm>
#include <cmath>

namespace tdmz {

namespace {

constexpr int kObsPlane = kBoardW * kBoardH;
constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

inline int obs_index(int c, int y, int x) {
    return (c * kBoardH + y) * kBoardW + x;
}

uint64_t static_tower_signature(const TDEngine& env) {
    uint64_t h = kFnvOffset;
    auto mix = [&](uint64_t value) {
        h ^= value;
        h *= kFnvPrime;
    };

    mix(static_cast<uint64_t>(env.towers().size()));
    for (const auto& tower : env.towers()) {
        mix(static_cast<uint64_t>(tower.x));
        mix(static_cast<uint64_t>(tower.y));
        mix(static_cast<uint64_t>(static_cast<int>(tower.type)));
        mix(static_cast<uint64_t>(tower.level));
    }
    return h;
}

struct ObservationStaticCache {
    bool valid = false;
    Bitboard128 blocked{};
    uint64_t tower_signature = 0;
    Observation static_obs;
};

thread_local ObservationStaticCache g_static_cache;

const Observation& get_static_observation_channels(const TDEngine& env) {
    const Bitboard128 blocked = env.blocked_bitboard();
    const uint64_t tower_signature = static_tower_signature(env);
    if (g_static_cache.valid &&
        g_static_cache.blocked == blocked &&
        g_static_cache.tower_signature == tower_signature &&
        g_static_cache.static_obs.size() == OBS_CHANNELS * kObsPlane) {
        return g_static_cache.static_obs;
    }

    g_static_cache.valid = true;
    g_static_cache.blocked = blocked;
    g_static_cache.tower_signature = tower_signature;
    g_static_cache.static_obs.assign(OBS_CHANNELS * kObsPlane, 0.0f);

    auto at = [&](int c, int y, int x) -> float& {
        return g_static_cache.static_obs[obs_index(c, y, x)];
    };

    for (int y = 0; y < kBoardH; ++y) {
        for (int x = 0; x < kBoardW; ++x) {
            if (env.grid()[y][x] == 1) at(CH_BLOCKED, y, x) = 1.0f;
        }
    }

    at(CH_SPAWN, env.spawn_y(), env.spawn_x()) = 1.0f;
    at(CH_BASE, env.base_y(), env.base_x()) = 1.0f;

    for (const auto& tower : env.towers()) {
        int x = tower.x;
        int y = tower.y;
        if (tower.type == TowerType::Basic) at(CH_TOWER_BASIC, y, x) = 1.0f;
        else if (tower.type == TowerType::Sniper) at(CH_TOWER_SNIPER, y, x) = 1.0f;
        else if (tower.type == TowerType::AOE) at(CH_TOWER_AOE, y, x) = 1.0f;
        else if (tower.type == TowerType::Slow) at(CH_TOWER_SLOW, y, x) = 1.0f;

        at(CH_TOWER_LEVEL, y, x) = std::min(1.0f, static_cast<float>(tower.level) / static_cast<float>(kTowerMaxLevel));
    }

    auto placeable_mask = env.compute_placeable_mask();
    for (int y = 0; y < kBoardH; ++y) {
        for (int x = 0; x < kBoardW; ++x) {
            int cell = cell_id(x, y);
            if (placeable_mask[y][x]) {
                at(CH_PLACEABLE_MASK, y, x) = 1.0f;
            }
            int dist = env.distance_to_base_cell(cell);
            if (dist < 1000000000) {
                at(CH_DISTANCE_TO_BASE, y, x) = static_cast<float>(dist) / 20.0f;
            }
        }
    }

    return g_static_cache.static_obs;
}

} // namespace

Observation make_observation_python_parity(const TDEngine& env) {
    int w = env.width();
    int h = env.height();
    Observation obs(5 * h * w, 0.0f);

    auto at = [&](int c, int y, int x) -> float& {
        return obs[(c * h + y) * w + x];
    };

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (env.grid()[y][x] == 1) {
                at(0, y, x) = 1.0f;
            }
        }
    }

    at(1, env.spawn_y(), env.spawn_x()) = 1.0f;
    at(1, env.base_y(), env.base_x()) = 1.0f;

    for (const auto& enemy : env.enemies()) {
        int grid_x = static_cast<int>(enemy.x);
        int grid_y = static_cast<int>(enemy.y);

        if (grid_x >= 0 && grid_x < w && grid_y >= 0 && grid_y < h) {
            float added_hp = enemy.hp / std::max(1.0f, enemy.max_hp);
            at(2, grid_y, grid_x) = std::min(1.0f, at(2, grid_y, grid_x) + added_hp);
        }
    }

    float base_hp_val = env.base_hp() / 100.0f;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            at(3, y, x) = base_hp_val;
        }
    }

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
    Observation obs = get_static_observation_channels(env);

    auto at = [&](int c, int y, int x) -> float& {
        return obs[obs_index(c, y, x)];
    };

    for (const auto& tower : env.towers()) {
        int x = tower.x;
        int y = tower.y;
        const float cooldown_ratio = tower.cooldown_max > 0.0f ? tower.cooldown / tower.cooldown_max : 0.0f;
        at(CH_TOWER_COOLDOWN, y, x) = std::clamp(cooldown_ratio, 0.0f, 1.0f);
    }

    for (const auto& enemy : env.enemies()) {
        int gx = static_cast<int>(std::round(enemy.x));
        int gy = static_cast<int>(std::round(enemy.y));
        if (gx >= 0 && gx < kBoardW && gy >= 0 && gy < kBoardH) {
            at(CH_ENEMY_HP, gy, gx) += enemy.hp / std::max(1.0f, enemy.max_hp);
            at(CH_ENEMY_DENSITY, gy, gx) += 1.0f;
            at(CH_ENEMY_SPEED, gy, gx) = std::max(at(CH_ENEMY_SPEED, gy, gx), enemy.speed / 3.0f);
            at(CH_ENEMY_SLOW_TIMER, gy, gx) = std::max(at(CH_ENEMY_SLOW_TIMER, gy, gx), enemy.slow_timer / 3.0f);
        }
    }

    float base_hp_val = env.base_hp() / 100.0f;
    float money_val = std::min(1.0f, env.money() / 1000.0f);
    float wave_val = env.wave() / 20.0f;
    float spawn_timer_val = env.spawn_timer() / 3.0f;
    float to_spawn_val = env.enemies_to_spawn_count() / 20.0f;

    for (int y = 0; y < kBoardH; ++y) {
        for (int x = 0; x < kBoardW; ++x) {
            at(CH_BASE_HP, y, x) = base_hp_val;
            at(CH_MONEY, y, x) = money_val;
            at(CH_WAVE, y, x) = wave_val;
            at(CH_SPAWN_TIMER, y, x) = spawn_timer_val;
            at(CH_TO_SPAWN_COUNT, y, x) = to_spawn_val;
        }
    }

    return obs;
}

std::array<int, 3> observation_shape_v1() {
    return {OBS_CHANNELS, kBoardH, kBoardW};
}

} // namespace tdmz
