#include "tdmz/balance/defense_capacity.hpp"
#include "tdmz/core/board_tables.hpp"
#include <algorithm>
#include <cmath>

namespace tdmz {

namespace {

float sanitize_nonnegative(float v) {
    if (!std::isfinite(v) || v < 0.0f) return 0.0f;
    return v;
}

int count_placeable_cells(const std::array<std::array<bool, kBoardW>, kBoardH>& mask) {
    int count = 0;
    for (const auto& row : mask) {
        for (bool v : row) {
            if (v) ++count;
        }
    }
    return count;
}

TowerStats stats_after_level(TowerType type, int level) {
    TowerStats s = tower_stats(type);
    int target_level = std::max(1, std::min(level, kTowerMaxLevel));
    for (int lv = 1; lv < target_level; ++lv) {
        s.damage *= 1.5f;
        s.range *= 1.1f;
        s.cooldown = std::max(0.1f, s.cooldown * 0.9f);
    }
    return s;
}

float estimate_stats_damage_capacity(const TowerStats& s, TowerType type, float window_seconds, float aoe_multiplier, float slow_multiplier) {
    float window = sanitize_nonnegative(window_seconds);
    float cooldown = std::max(0.1f, s.cooldown);
    int shots = static_cast<int>(std::floor(window / cooldown));
    if (window > 0.0f && shots < 1) shots = 1;

    float cap = static_cast<float>(shots) * std::max(0.0f, s.damage);
    if (type == TowerType::AOE) {
        cap *= std::max(1.0f, aoe_multiplier);
    }
    if (type == TowerType::Slow) {
        cap *= std::max(1.0f, slow_multiplier);
    }
    return cap;
}

} // namespace

float estimate_tower_damage_capacity(TowerType type, int level, float window_seconds, float aoe_multiplier, float slow_multiplier) {
    TowerStats s = stats_after_level(type, level);
    return estimate_stats_damage_capacity(s, type, window_seconds, aoe_multiplier, slow_multiplier);
}

float estimate_existing_tower_damage_capacity(const Tower& tower, float window_seconds, float aoe_multiplier, float slow_multiplier) {
    float window = sanitize_nonnegative(window_seconds);
    float cooldown = std::max(0.1f, tower.cooldown_max);
    float first_shot_delay = std::max(0.0f, tower.cooldown);
    if (window <= 0.0f || first_shot_delay > window) {
        return 0.0f;
    }

    int shots = 1 + static_cast<int>(std::floor((window - first_shot_delay) / cooldown));
    float cap = static_cast<float>(shots) * std::max(0.0f, tower.damage);
    if (tower.type == TowerType::AOE) {
        cap *= std::max(1.0f, aoe_multiplier);
    }
    if (tower.type == TowerType::Slow) {
        cap *= std::max(1.0f, slow_multiplier);
    }
    return cap;
}

DefenseCapacityResult estimate_defense_capacity(const TDEngine& env, const DefenseCapacityConfig& cfg) {
    DefenseCapacityResult result;

    const float window = sanitize_nonnegative(cfg.wave_window_seconds);
    const float safety = std::max(0.0f, cfg.safety_factor);
    const float aoe_mult = std::max(1.0f, cfg.aoe_multiplier);
    const float slow_mult = std::max(1.0f, cfg.slow_multiplier);

    for (const auto& tower : env.towers()) {
        result.current_tower_damage_cap += estimate_existing_tower_damage_capacity(tower, window, aoe_mult, slow_mult);
    }

    auto placeable = env.compute_placeable_mask();
    result.placeable_count = count_placeable_cells(placeable);

    if (cfg.include_spendable_money && result.placeable_count > 0 && env.money() > 0) {
        float best_ratio = 0.0f;
        TowerType best_type = TowerType::Basic;
        int best_cost = tower_stats(TowerType::Basic).cost;
        float best_cap = 0.0f;

        for (int t = 0; t < kBuildTypes; ++t) {
            auto type = static_cast<TowerType>(t);
            TowerStats s = tower_stats(type);
            if (s.cost <= 0) continue;
            float cap = estimate_tower_damage_capacity(type, 1, window, aoe_mult, slow_mult);
            float ratio = cap / static_cast<float>(s.cost);
            if (ratio > best_ratio) {
                best_ratio = ratio;
                best_type = type;
                best_cost = s.cost;
                best_cap = cap;
            }
        }

        int affordable = env.money() / std::max(1, best_cost);
        result.selected_new_tower_count = std::max(0, std::min(result.placeable_count, affordable));
        result.selected_new_tower_type = best_type;
        result.spendable_money_damage_cap = static_cast<float>(result.selected_new_tower_count) * best_cap;
    }

    int virtual_hp = std::max(0, cfg.virtual_base_hp);
    result.leak_hp_cap = static_cast<float>(std::max(0, virtual_hp - 1) * 10);

    result.raw_hp_cap = result.current_tower_damage_cap + result.spendable_money_damage_cap + result.leak_hp_cap;
    result.allowed_attack_hp = safety * result.raw_hp_cap;
    return result;
}

} // namespace tdmz
