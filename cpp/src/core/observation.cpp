#include "tdmz/core/observation.hpp"
#include <algorithm>
#include <cmath>

namespace tdmz {

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
    int w = env.width();
    int h = env.height();
    Observation obs(OBS_CHANNELS * h * w, 0.0f);

    auto at = [&](int c, int y, int x) -> float& {
        return obs[(c * h + y) * w + x];
    };

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
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
        at(CH_TOWER_COOLDOWN, y, x) = tower.cooldown / tower.cooldown_max;
    }

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

    auto placeable_mask = env.compute_placeable_mask();

    std::vector<std::vector<int>> dist(h, std::vector<int>(w, 1000000000));
    std::vector<std::pair<int, int>> queue;
    int head = 0;

    dist[env.base_y()][env.base_x()] = 0;
    queue.push_back({env.base_x(), env.base_y()});

    const int dx[] = {0, 1, 0, -1};
    const int dy[] = {1, 0, -1, 0};

    while (head < static_cast<int>(queue.size())) {
        auto [cx, cy] = queue[head++];
        for (int i = 0; i < 4; ++i) {
            int nx = cx + dx[i];
            int ny = cy + dy[i];
            if (nx >= 0 && nx < w && ny >= 0 && ny < h && env.grid()[ny][nx] == 0) {
                if (dist[ny][nx] > dist[cy][cx] + 1) {
                    dist[ny][nx] = dist[cy][cx] + 1;
                    queue.push_back({nx, ny});
                }
            }
        }
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (placeable_mask[y][x]) {
                at(CH_PLACEABLE_MASK, y, x) = 1.0f;
            }
            if (dist[y][x] < 1000000000) {
                at(CH_DISTANCE_TO_BASE, y, x) = static_cast<float>(dist[y][x]) / 20.0f;
            }
        }
    }

    return obs;
}

std::array<int, 3> observation_shape_v1() {
    return {OBS_CHANNELS, kBoardH, kBoardW};
}

} // namespace tdmz
