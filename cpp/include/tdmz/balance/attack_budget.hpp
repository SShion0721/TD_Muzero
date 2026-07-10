#pragma once
#include "tdmz/balance/defense_capacity.hpp"

namespace tdmz {

struct AttackBudgetConfig {
    DefenseCapacityConfig defense;

    bool enable_wave_cap = true;
    float wave_base_hp = 300.0f;
    float wave_linear_hp = 180.0f;
    float wave_quadratic_hp = 35.0f;
    float wave_cap_multiplier = 1.0f;

    int tank_unlock_wave = 3;
    int boss_unlock_wave = 5;
    bool boss_odd_waves_only = true;

    float max_fast_ratio = 0.35f;
    float max_tank_ratio = 0.45f;
    float max_boss_ratio = 0.25f;
};

struct AttackBudgetResult {
    DefenseCapacityResult defense;

    int wave = 0;
    float defense_allowed_attack_hp = 0.0f;
    float wave_cap_hp = 0.0f;
    float allowed_attack_hp = 0.0f;

    float regular_hp_cap = 0.0f;
    float fast_hp_cap = 0.0f;
    float tank_hp_cap = 0.0f;
    float boss_hp_cap = 0.0f;

    bool wave_cap_applied = false;
    bool tank_unlocked = false;
    bool boss_unlocked = false;
};

float estimate_wave_attack_cap_hp(int wave, const AttackBudgetConfig& cfg);
bool attack_budget_tank_unlocked(int wave, const AttackBudgetConfig& cfg);
bool attack_budget_boss_unlocked(int wave, const AttackBudgetConfig& cfg);
AttackBudgetResult estimate_attack_budget(const TDEngine& env, const AttackBudgetConfig& cfg = AttackBudgetConfig{});

} // namespace tdmz
