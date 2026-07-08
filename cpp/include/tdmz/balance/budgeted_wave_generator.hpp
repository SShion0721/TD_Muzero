#pragma once
#include "tdmz/balance/attack_budget.hpp"
#include "tdmz/core/enemy.hpp"
#include <vector>

namespace tdmz {

struct BudgetedWaveConfig {
    float regular_ratio = 0.55f;
    float fast_ratio = 0.25f;
    float tank_ratio = 0.15f;
    float boss_ratio = 0.05f;

    int min_regular_count = 3;
    int max_enemy_count = 64;
    bool deterministic_greedy = true;

    bool enable_elite_scaling = true;
    float max_regular_elite_multiplier = 4.0f;
    float max_tank_elite_multiplier = 2.0f;
    float max_boss_elite_multiplier = 1.5f;
    bool scale_elite_reward = true;
};

struct EnemyArchetypeSpec {
    EnemySpec spec{};
    float unit_hp = 0.0f;
};

struct BudgetedWaveResult {
    std::vector<EnemySpec> enemies;

    int wave = 0;
    float requested_budget_hp = 0.0f;
    float used_budget_hp = 0.0f;
    float unused_budget_hp = 0.0f;

    float regular_hp = 0.0f;
    float fast_hp = 0.0f;
    float tank_hp = 0.0f;
    float boss_hp = 0.0f;

    float elite_bonus_hp = 0.0f;
    int elite_scaled_count = 0;
    float max_enemy_hp = 0.0f;

    int regular_count = 0;
    int fast_count = 0;
    int tank_count = 0;
    int boss_count = 0;
};

EnemyArchetypeSpec regular_archetype_for_wave(int wave);
EnemyArchetypeSpec fast_archetype_for_wave(int wave);
EnemyArchetypeSpec tank_archetype_for_wave(int wave);
EnemyArchetypeSpec boss_archetype_for_wave(int wave);

BudgetedWaveResult generate_budgeted_wave(const AttackBudgetResult& budget, const BudgetedWaveConfig& cfg = BudgetedWaveConfig{});

} // namespace tdmz
