#pragma once
#include "tdmz/core/engine.hpp"
#include "tdmz/core/tower.hpp"

namespace tdmz {

struct DefenseCapacityConfig {
    float wave_window_seconds = 15.0f;
    float safety_factor = 0.90f;
    float aoe_multiplier = 1.50f;
    float slow_multiplier = 1.10f;
    int virtual_base_hp = 100;
    bool include_spendable_money = true;
};

struct DefenseCapacityResult {
    float current_tower_damage_cap = 0.0f;
    float spendable_money_damage_cap = 0.0f;
    float leak_hp_cap = 0.0f;
    float raw_hp_cap = 0.0f;
    float allowed_attack_hp = 0.0f;

    int placeable_count = 0;
    int selected_new_tower_count = 0;
    TowerType selected_new_tower_type = TowerType::Basic;
};

float estimate_tower_damage_capacity(TowerType type, int level, float window_seconds, float aoe_multiplier, float slow_multiplier);
float estimate_existing_tower_damage_capacity(const Tower& tower, float window_seconds, float aoe_multiplier, float slow_multiplier);
DefenseCapacityResult estimate_defense_capacity(const TDEngine& env, const DefenseCapacityConfig& cfg = DefenseCapacityConfig{});

} // namespace tdmz
