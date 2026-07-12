#include "tdmz/core/engine.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace tdmz {

namespace {
constexpr int kBucketTargetingEnemyThreshold = 16;
constexpr int kMaxTowerShotsPerTick = 64;

int enemy_rounded_cell(const Enemy& enemy) {
    const int x = static_cast<int>(std::round(enemy.x));
    const int y = static_cast<int>(std::round(enemy.y));
    if (!valid_cell_xy(x, y)) return -1;
    return cell_id(x, y);
}

} // namespace

StepResult TDEngine::step_one_tick() {
    if (game_over_) return {0.0f, true};
    episode_started_ = true;

    constexpr float dt = 1.0f;
    float step_reward = 0.0f;
    bool enemies_changed = false;

    spawn_timer_ -= dt;
    if (spawn_timer_ <= 0.0f && !enemies_to_spawn_.empty()) {
        const EnemySpec enemy_spec = enemies_to_spawn_.front();
        enemies_to_spawn_.erase(enemies_to_spawn_.begin());

        enemies_.emplace_back(
            enemy_id_counter_++,
            static_cast<float>(spawn_x_),
            static_cast<float>(spawn_y_),
            enemy_spec.hp,
            enemy_spec.speed,
            enemy_spec.reward);
        enemies_.back().set_path(find_path(spawn_x_, spawn_y_, base_x_, base_y_));

        spawn_timer_ = 1.0f;
        enemies_changed = true;
    }

    for (auto& enemy : enemies_) {
        enemy.step(dt);
        enemies_changed = true;
    }

    const auto reached_base_end = std::remove_if(
        enemies_.begin(),
        enemies_.end(),
        [&](const Enemy& enemy) {
            const float base_distance = std::hypot(
                enemy.x - static_cast<float>(base_x_),
                enemy.y - static_cast<float>(base_y_));
            if (base_distance >= 0.2f) return false;

            base_hp_ -= static_cast<int>(enemy.max_hp) / 10;
            step_reward -= 50.0f;
            enemies_changed = true;
            return true;
        });
    enemies_.erase(reached_base_end, enemies_.end());

    if (base_hp_ <= 0) {
        game_over_ = true;
        legal_cache_valid_ = false;
        if (enemies_changed) mark_enemies_changed();
        step_reward -= 1000.0f;
        time_ += dt;
        return {step_reward, true};
    }

    const auto& tables = board_tables();
    const int enemy_count = static_cast<int>(enemies_.size());
    const bool use_bucket_targeting =
        enemy_count >= kBucketTargetingEnemyThreshold && !towers_.empty();

    std::array<int, kCells> bucket_head;
    Bitboard128 enemy_occupancy = Bitboard128::zero();
    bool has_offboard_enemy = false;

    if (use_bucket_targeting) {
        bucket_head.fill(-1);
        if (static_cast<int>(enemy_bucket_next_scratch_.size()) < enemy_count) {
            enemy_bucket_next_scratch_.resize(static_cast<size_t>(enemy_count));
        }
        std::fill(
            enemy_bucket_next_scratch_.begin(),
            enemy_bucket_next_scratch_.begin() + enemy_count,
            -1);

        ++perf_.enemy_bucket_recompute;
        for (int enemy_index = 0; enemy_index < enemy_count; ++enemy_index) {
            const int cell = enemy_rounded_cell(enemies_[enemy_index]);
            if (cell >= 0) {
                enemy_bucket_next_scratch_[enemy_index] = bucket_head[cell];
                bucket_head[cell] = enemy_index;
                enemy_occupancy |= tables.cell_bb[cell];
            } else {
                has_offboard_enemy = true;
            }
        }
    }

    for (auto& tower : towers_) {
        float remaining_dt = dt;
        int shots_this_tick = 0;

        while (tower.advance_until_ready(remaining_dt)) {
            int best_index = -1;
            float best_distance_squared = 1e30f;
            const float range_squared = tower.range * tower.range;

            if (enemy_count > 0) {
                if (!use_bucket_targeting) {
                    for (int enemy_index = 0; enemy_index < enemy_count; ++enemy_index) {
                        Enemy& enemy = enemies_[enemy_index];
                        if (enemy.hp <= 0.0f) continue;

                        const float dx = enemy.x - static_cast<float>(tower.x);
                        const float dy = enemy.y - static_cast<float>(tower.y);
                        const float distance_squared = dx * dx + dy * dy;
                        ++perf_.tower_exact_distance_checks;
                        if (distance_squared <= range_squared &&
                            distance_squared < best_distance_squared) {
                            best_distance_squared = distance_squared;
                            best_index = enemy_index;
                        }
                    }
                } else {
                    const int tower_cell = cell_id(tower.x, tower.y);
                    const int tower_type = static_cast<int>(tower.type);
                    const int tower_level = std::max(
                        1,
                        std::min(kTowerMaxLevel, tower.level));
                    Bitboard128 candidate_cells =
                        tables.range_mask_expanded8[tower_type][tower_level][tower_cell] &
                        enemy_occupancy;
                    perf_.tower_candidate_cells +=
                        static_cast<uint64_t>(candidate_cells.popcount());

                    while (candidate_cells) {
                        const int cell = candidate_cells.pop_lsb();
                        for (int enemy_index = bucket_head[cell];
                             enemy_index >= 0;
                             enemy_index = enemy_bucket_next_scratch_[enemy_index]) {
                            Enemy& enemy = enemies_[enemy_index];
                            if (enemy.hp <= 0.0f) continue;

                            const float dx = enemy.x - static_cast<float>(tower.x);
                            const float dy = enemy.y - static_cast<float>(tower.y);
                            const float distance_squared = dx * dx + dy * dy;
                            ++perf_.tower_exact_distance_checks;
                            if (distance_squared <= range_squared &&
                                (distance_squared < best_distance_squared ||
                                 (distance_squared == best_distance_squared &&
                                  (best_index < 0 || enemy_index < best_index)))) {
                                best_distance_squared = distance_squared;
                                best_index = enemy_index;
                            }
                        }
                    }

                    if (best_index < 0 && has_offboard_enemy) {
                        for (int enemy_index = 0; enemy_index < enemy_count; ++enemy_index) {
                            Enemy& enemy = enemies_[enemy_index];
                            if (enemy.hp <= 0.0f || enemy_rounded_cell(enemy) >= 0) continue;

                            const float dx = enemy.x - static_cast<float>(tower.x);
                            const float dy = enemy.y - static_cast<float>(tower.y);
                            const float distance_squared = dx * dx + dy * dy;
                            ++perf_.tower_exact_distance_checks;
                            if (distance_squared <= range_squared &&
                                distance_squared < best_distance_squared) {
                                best_distance_squared = distance_squared;
                                best_index = enemy_index;
                            }
                        }
                    }
                }
            }

            if (best_index < 0) break;

            Enemy* best_target = &enemies_[best_index];
            if (tower.aoe_radius > 0.0f) {
                const float aoe_squared = tower.aoe_radius * tower.aoe_radius;
                for (auto& enemy : enemies_) {
                    if (enemy.hp <= 0.0f) continue;

                    const float dx = enemy.x - best_target->x;
                    const float dy = enemy.y - best_target->y;
                    ++perf_.tower_aoe_distance_checks;
                    if (dx * dx + dy * dy <= aoe_squared) {
                        enemy.hp -= tower.damage;
                        enemy.apply_slow(tower.slow_factor, tower.slow_duration);
                        enemies_changed = true;
                    }
                }
            } else {
                best_target->hp -= tower.damage;
                best_target->apply_slow(tower.slow_factor, tower.slow_duration);
                enemies_changed = true;
            }

            tower.shoot();
            ++shots_this_tick;
            if (shots_this_tick > kMaxTowerShotsPerTick) {
                throw std::runtime_error("Tower exceeded the per-tick shot safety limit");
            }
        }
    }

    bool money_changed = false;
    const auto dead_end = std::remove_if(
        enemies_.begin(),
        enemies_.end(),
        [&](const Enemy& enemy) {
            if (enemy.hp > 0.0f) return false;
            money_ += enemy.reward;
            step_reward += enemy.reward;
            money_changed = true;
            enemies_changed = true;
            return true;
        });
    enemies_.erase(dead_end, enemies_.end());
    if (money_changed) mark_money_changed();

    if (enemies_.empty() && enemies_to_spawn_.empty()) {
        ++wave_;
        enemies_to_spawn_ = get_wave_enemies();
        spawn_timer_ = 3.0f;
        step_reward += 100.0f;
        mark_wave_changed();
    }

    if (enemies_changed) mark_enemies_changed();
    time_ += dt;
    return {step_reward, false};
}

} // namespace tdmz
