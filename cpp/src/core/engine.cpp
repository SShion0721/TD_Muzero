#include "tdmz/core/engine.hpp"
#include "tdmz/core/pathfinding.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace tdmz {

TDEngine::TDEngine(int width, int height, uint64_t seed)
    : width_(width), height_(height),
      spawn_x_(0), spawn_y_(height / 2),
      base_x_(width - 1), base_y_(height / 2),
      seed_(seed), rng_(seed) {
    if (width != kBoardW || height != kBoardH) {
        throw std::invalid_argument("TDEngine currently supports only 11x11 boards");
    }
    reset(seed);
}

void TDEngine::reset(uint64_t seed) {
    seed_ = seed;
    rng_.seed(seed_);

    money_ = 200;
    base_hp_ = 100;

    for (auto& row : grid_) {
        row.fill(0);
    }

    towers_.clear();
    enemies_.clear();
    enemy_id_counter_ = 0;

    wave_ = 1;
    enemies_to_spawn_ = get_wave_enemies();
    spawn_timer_ = 0.0f;
    time_ = 0.0f;
    game_over_ = false;
}

std::vector<EnemySpec> TDEngine::get_wave_enemies() {
    std::vector<EnemySpec> format;
    float base_hp = 20.0f + wave_ * 15.0f + (wave_ * wave_) * 2.0f;

    int swarm_count = wave_ * 2;
    for (int i = 0; i < swarm_count; ++i) {
        format.push_back({base_hp * 0.3f, 2.8f, 5});
    }

    int reg_count = 5 + wave_ * 2;
    for (int i = 0; i < reg_count; ++i) {
        format.push_back({base_hp, 1.5f, 10});
    }

    if (wave_ >= 3) {
        int tank_count = wave_;
        for (int i = 0; i < tank_count; ++i) {
            format.push_back({base_hp * 3.5f, 0.8f, 30});
        }
    }

    if (wave_ >= 5 && wave_ % 2 == 1) {
        int boss_count = 1 + (wave_ / 10);
        for (int i = 0; i < boss_count; ++i) {
            format.push_back({base_hp * 10.0f, 0.6f, 100});
        }
    }

    rng_.shuffle(format);
    return format;
}

std::vector<std::pair<int,int>> TDEngine::find_path(int sx, int sy, int ex, int ey) const {
    return find_shortest_path(sx, sy, ex, ey, grid_);
}

bool TDEngine::in_bounds(int x, int y) const {
    return x >= 0 && x < width_ && y >= 0 && y < height_;
}

bool TDEngine::can_place_tower(int x, int y, TowerType type) const {
    if (!in_bounds(x, y)) return false;
    if (grid_[y][x] != 0) return false;
    if (x == spawn_x_ && y == spawn_y_) return false;
    if (x == base_x_ && y == base_y_) return false;

    auto stats = tower_stats(type);
    if (money_ < stats.cost) return false;

    auto temp_grid = grid_;
    temp_grid[y][x] = 1;
    auto path = find_shortest_path(spawn_x_, spawn_y_, base_x_, base_y_, temp_grid);

    return !path.empty();
}

bool TDEngine::place_tower(int x, int y, TowerType type) {
    if (can_place_tower(x, y, type)) {
        auto stats = tower_stats(type);
        money_ -= stats.cost;
        grid_[y][x] = 1;
        towers_.emplace_back(x, y, type);

        for (auto& enemy : enemies_) {
            int curr_grid_x = static_cast<int>(std::round(enemy.x));
            int curr_grid_y = static_cast<int>(std::round(enemy.y));
            auto path = find_path(curr_grid_x, curr_grid_y, base_x_, base_y_);
            if (!path.empty()) {
                enemy.set_path(path);
            }
        }
        return true;
    }
    return false;
}

bool TDEngine::upgrade_tower(int x, int y) {
    for (auto& tower : towers_) {
        if (tower.x == x && tower.y == y) {
            if (money_ >= tower.upgrade_cost) {
                money_ -= tower.upgrade_cost;
                tower.upgrade();
                return true;
            }
            break;
        }
    }
    return false;
}

bool TDEngine::sell_tower(int x, int y) {
    for (auto it = towers_.begin(); it != towers_.end(); ++it) {
        if (it->x == x && it->y == y) {
            int refund = static_cast<int>(it->total_spent * 0.8f);
            money_ += refund;
            towers_.erase(it);
            grid_[y][x] = 0;

            for (auto& enemy : enemies_) {
                int curr_grid_x = static_cast<int>(std::round(enemy.x));
                int curr_grid_y = static_cast<int>(std::round(enemy.y));
                auto path = find_path(curr_grid_x, curr_grid_y, base_x_, base_y_);
                if (!path.empty()) {
                    enemy.set_path(path);
                }
            }
            return true;
        }
    }
    return false;
}

std::array<std::array<bool, kBoardW>, kBoardH> TDEngine::compute_placeable_mask() const {
    std::array<std::array<bool, kBoardW>, kBoardH> mask;
    for (auto& row : mask) row.fill(false);

    auto base_path = find_path(spawn_x_, spawn_y_, base_x_, base_y_);
    if (base_path.empty()) return mask;

    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            if (grid_[y][x] == 0 && (x != spawn_x_ || y != spawn_y_) && (x != base_x_ || y != base_y_)) {
                auto temp_grid = grid_;
                temp_grid[y][x] = 1;
                auto p = find_shortest_path(spawn_x_, spawn_y_, base_x_, base_y_, temp_grid);
                if (!p.empty()) {
                    mask[y][x] = true;
                }
            }
        }
    }
    return mask;
}

std::vector<int> TDEngine::legal_actions() const {
    std::vector<int> actions;
    auto placeable = compute_placeable_mask();

    for (int t = 0; t < 4; ++t) {
        auto type = static_cast<TowerType>(t);
        if (money_ >= tower_stats(type).cost) {
            for (int y = 0; y < height_; ++y) {
                for (int x = 0; x < width_; ++x) {
                    if (placeable[y][x]) {
                        Action a{static_cast<ActionType>(t), x, y, 1, -1};
                        actions.push_back(encode_action(a));
                    }
                }
            }
        }
    }

    for (const auto& tower : towers_) {
        if (money_ >= tower.upgrade_cost) {
            actions.push_back(encode_action(Action{ActionType::Upgrade, tower.x, tower.y}));
        }
        actions.push_back(encode_action(Action{ActionType::Sell, tower.x, tower.y}));
    }

    actions.push_back(encode_action(Action{ActionType::Wait1, -1, -1, 1}));
    return actions;
}

std::vector<uint8_t> TDEngine::legal_action_mask() const {
    std::vector<uint8_t> mask(kActionSpaceSize, 0);
    for (int a : legal_actions()) {
        mask[a] = 1;
    }
    return mask;
}

StepResult TDEngine::step_wait(int wait_steps) {
    StepResult r;
    for (int i = 0; i < wait_steps; ++i) {
        auto step_r = step_time(1.0f);
        r.reward += step_r.reward;
        if (step_r.done) {
            r.done = true;
            break;
        }
    }
    return r;
}

StepResult TDEngine::step_action(int flat_action) {
    Action a = decode_action(flat_action);
    float reward = 0.0f;

    switch (a.type) {
        case ActionType::BuildBasic:
        case ActionType::BuildSniper:
        case ActionType::BuildAOE:
        case ActionType::BuildSlow:
            if (!place_tower(a.x, a.y, static_cast<TowerType>(static_cast<int>(a.type)))) {
                reward -= 5.0f;
            }
            break;

        case ActionType::Upgrade:
            if (!upgrade_tower(a.x, a.y)) reward -= 5.0f;
            break;

        case ActionType::Sell:
            if (!sell_tower(a.x, a.y)) reward -= 5.0f;
            break;

        case ActionType::Wait1:
            break;
    }

    StepResult r = step_wait(a.wait_steps);
    r.reward += reward;
    return r;
}

StepResult TDEngine::step_time(float dt) {
    if (game_over_) return {0.0f, true};

    float step_reward = 0.0f;

    spawn_timer_ -= dt;
    if (spawn_timer_ <= 0.0f && !enemies_to_spawn_.empty()) {
        auto e_data = enemies_to_spawn_.front();
        enemies_to_spawn_.erase(enemies_to_spawn_.begin());

        enemies_.emplace_back(enemy_id_counter_++, static_cast<float>(spawn_x_), static_cast<float>(spawn_y_),
                              e_data.hp, e_data.speed, e_data.reward);

        auto path = find_path(spawn_x_, spawn_y_, base_x_, base_y_);
        enemies_.back().set_path(path);

        spawn_timer_ = 1.0f;
    }

    std::vector<int> reached_base_ids;
    for (auto& enemy : enemies_) {
        enemy.step(dt);
        if (std::hypot(enemy.x - static_cast<float>(base_x_), enemy.y - static_cast<float>(base_y_)) < 0.2f) {
            reached_base_ids.push_back(enemy.id);
        }
    }

    for (int id : reached_base_ids) {
        auto it = std::find_if(enemies_.begin(), enemies_.end(), [id](const Enemy& e) { return e.id == id; });
        if (it != enemies_.end()) {
            base_hp_ -= static_cast<int>(it->max_hp) / 10;
            enemies_.erase(it);
            step_reward -= 50.0f;
        }
    }

    if (base_hp_ <= 0) {
        game_over_ = true;
        step_reward -= 1000.0f;
        return {step_reward, true};
    }

    for (auto& tower : towers_) {
        tower.step(dt);
        if (tower.can_shoot()) {
            Enemy* best_target = nullptr;
            float best_dist = 1e9f;

            for (auto& enemy : enemies_) {
                float dist = std::hypot(enemy.x - static_cast<float>(tower.x), enemy.y - static_cast<float>(tower.y));
                if (dist <= tower.range && dist < best_dist) {
                    best_dist = dist;
                    best_target = &enemy;
                }
            }

            if (best_target != nullptr) {
                if (tower.aoe_radius > 0.0f) {
                    for (auto& e : enemies_) {
                        if (std::hypot(e.x - best_target->x, e.y - best_target->y) <= tower.aoe_radius) {
                            e.hp -= tower.damage;
                            e.apply_slow(tower.slow_factor, tower.slow_duration);
                        }
                    }
                } else {
                    best_target->hp -= tower.damage;
                    best_target->apply_slow(tower.slow_factor, tower.slow_duration);
                }
                tower.shoot();
            }
        }
    }

    std::vector<int> dead_ids;
    for (const auto& e : enemies_) {
        if (e.hp <= 0.0f) dead_ids.push_back(e.id);
    }

    for (int id : dead_ids) {
        auto it = std::find_if(enemies_.begin(), enemies_.end(), [id](const Enemy& e) { return e.id == id; });
        if (it != enemies_.end()) {
            money_ += it->reward;
            step_reward += it->reward;
            enemies_.erase(it);
        }
    }

    if (enemies_.empty() && enemies_to_spawn_.empty()) {
        wave_ += 1;
        enemies_to_spawn_ = get_wave_enemies();
        spawn_timer_ = 3.0f;
        step_reward += 100.0f;
    }

    time_ += dt;
    return {step_reward, false};
}

} // namespace tdmz
