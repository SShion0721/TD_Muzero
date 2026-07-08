#include "tdmz/balance/budgeted_wave_generator.hpp"
#include <algorithm>
#include <cmath>

namespace tdmz {

namespace {

float nonnegative_finite(float v) {
    if (!std::isfinite(v) || v < 0.0f) return 0.0f;
    return v;
}

float clamp_ratio(float v) {
    if (!std::isfinite(v)) return 0.0f;
    return std::max(0.0f, std::min(1.0f, v));
}

float sane_multiplier(float v, float fallback) {
    if (!std::isfinite(v) || v < 1.0f) return fallback;
    return v;
}

float base_hp_for_wave(int wave) {
    int w = std::max(1, wave);
    return static_cast<float>(20 + w * 15 + w * w * 2);
}

void push_enemy(BudgetedWaveResult& result, const EnemyArchetypeSpec& arch, int kind) {
    result.enemies.push_back(arch.spec);
    result.used_budget_hp += arch.unit_hp;
    result.max_enemy_hp = std::max(result.max_enemy_hp, arch.unit_hp);
    switch (kind) {
        case 0: ++result.regular_count; result.regular_hp += arch.unit_hp; break;
        case 1: ++result.fast_count; result.fast_hp += arch.unit_hp; break;
        case 2: ++result.tank_count; result.tank_hp += arch.unit_hp; break;
        case 3: ++result.boss_count; result.boss_hp += arch.unit_hp; break;
    }
}

int add_by_budget(BudgetedWaveResult& result, const EnemyArchetypeSpec& arch, float budget_hp, int max_count_left, int kind) {
    if (arch.unit_hp <= 0.0f || budget_hp < arch.unit_hp || max_count_left <= 0) return 0;
    int count = static_cast<int>(std::floor(budget_hp / arch.unit_hp));
    count = std::max(0, std::min(count, max_count_left));
    for (int i = 0; i < count; ++i) {
        push_enemy(result, arch, kind);
    }
    return count;
}

float remaining_total_budget(const BudgetedWaveResult& result, float total) {
    return std::max(0.0f, total - result.used_budget_hp);
}

int max_count_left(const BudgetedWaveResult& result, int max_count) {
    return std::max(0, max_count - static_cast<int>(result.enemies.size()));
}

void apply_elite_bonus_to_enemy(BudgetedWaveResult& result, int index, float base_hp, float bonus_hp, bool scale_reward, int kind) {
    if (index < 0 || index >= static_cast<int>(result.enemies.size()) || bonus_hp <= 0.0f || base_hp <= 0.0f) return;
    EnemySpec& e = result.enemies[index];
    float old_hp = e.hp;
    e.hp += bonus_hp;
    result.used_budget_hp += bonus_hp;
    result.elite_bonus_hp += bonus_hp;
    result.elite_scaled_count += 1;
    result.max_enemy_hp = std::max(result.max_enemy_hp, e.hp);

    switch (kind) {
        case 0: result.regular_hp += bonus_hp; break;
        case 1: result.fast_hp += bonus_hp; break;
        case 2: result.tank_hp += bonus_hp; break;
        case 3: result.boss_hp += bonus_hp; break;
    }

    if (scale_reward && old_hp > 0.0f) {
        float reward_scale = e.hp / old_hp;
        e.reward = std::max(e.reward, static_cast<int>(std::lround(static_cast<float>(e.reward) * reward_scale)));
    }
}

void absorb_leftover_with_elites(BudgetedWaveResult& result,
                                 float total,
                                 const EnemyArchetypeSpec& regular,
                                 const EnemyArchetypeSpec& tank,
                                 const EnemyArchetypeSpec& boss,
                                 const BudgetedWaveConfig& cfg) {
    if (!cfg.enable_elite_scaling) return;

    float remaining = remaining_total_budget(result, total);
    if (remaining <= 0.0f) return;

    const float regular_max = sane_multiplier(cfg.max_regular_elite_multiplier, 4.0f);
    const float tank_max = sane_multiplier(cfg.max_tank_elite_multiplier, 2.0f);
    const float boss_max = sane_multiplier(cfg.max_boss_elite_multiplier, 1.5f);

    // Put most overflow into regulars first. This preserves special-archetype ratio caps;
    // tank/boss scaling is only a fallback when no regular backbone exists.
    for (int i = 0; i < static_cast<int>(result.enemies.size()) && remaining > 1e-5f; ++i) {
        EnemySpec& e = result.enemies[i];
        if (std::fabs(e.base_speed - 1.5f) > 1e-4f) continue;
        float cap_hp = regular.unit_hp * regular_max;
        float add = std::min(remaining, std::max(0.0f, cap_hp - e.hp));
        apply_elite_bonus_to_enemy(result, i, regular.unit_hp, add, cfg.scale_elite_reward, 0);
        remaining = remaining_total_budget(result, total);
    }

    for (int i = 0; i < static_cast<int>(result.enemies.size()) && remaining > 1e-5f; ++i) {
        EnemySpec& e = result.enemies[i];
        if (std::fabs(e.base_speed - 0.8f) > 1e-4f) continue;
        float cap_hp = tank.unit_hp * tank_max;
        float add = std::min(remaining, std::max(0.0f, cap_hp - e.hp));
        apply_elite_bonus_to_enemy(result, i, tank.unit_hp, add, cfg.scale_elite_reward, 2);
        remaining = remaining_total_budget(result, total);
    }

    for (int i = 0; i < static_cast<int>(result.enemies.size()) && remaining > 1e-5f; ++i) {
        EnemySpec& e = result.enemies[i];
        if (std::fabs(e.base_speed - 0.6f) > 1e-4f) continue;
        float cap_hp = boss.unit_hp * boss_max;
        float add = std::min(remaining, std::max(0.0f, cap_hp - e.hp));
        apply_elite_bonus_to_enemy(result, i, boss.unit_hp, add, cfg.scale_elite_reward, 3);
        remaining = remaining_total_budget(result, total);
    }
}

} // namespace

EnemyArchetypeSpec regular_archetype_for_wave(int wave) {
    float base = base_hp_for_wave(wave);
    EnemyArchetypeSpec a;
    a.spec = EnemySpec{base, 1.5f, 10};
    a.unit_hp = base;
    return a;
}

EnemyArchetypeSpec fast_archetype_for_wave(int wave) {
    float base = base_hp_for_wave(wave);
    EnemyArchetypeSpec a;
    a.spec = EnemySpec{base * 0.3f, 2.8f, 5};
    a.unit_hp = base * 0.3f;
    return a;
}

EnemyArchetypeSpec tank_archetype_for_wave(int wave) {
    float base = base_hp_for_wave(wave);
    EnemyArchetypeSpec a;
    a.spec = EnemySpec{base * 3.5f, 0.8f, 30};
    a.unit_hp = base * 3.5f;
    return a;
}

EnemyArchetypeSpec boss_archetype_for_wave(int wave) {
    float base = base_hp_for_wave(wave);
    EnemyArchetypeSpec a;
    a.spec = EnemySpec{base * 10.0f, 0.6f, 100};
    a.unit_hp = base * 10.0f;
    return a;
}

BudgetedWaveResult generate_budgeted_wave(const AttackBudgetResult& budget, const BudgetedWaveConfig& cfg) {
    BudgetedWaveResult result;
    result.wave = std::max(1, budget.wave);
    result.requested_budget_hp = nonnegative_finite(budget.allowed_attack_hp);

    int max_count = std::max(0, cfg.max_enemy_count);
    if (result.requested_budget_hp <= 0.0f || max_count <= 0) {
        result.unused_budget_hp = result.requested_budget_hp;
        return result;
    }

    const auto regular = regular_archetype_for_wave(result.wave);
    const auto fast = fast_archetype_for_wave(result.wave);
    const auto tank = tank_archetype_for_wave(result.wave);
    const auto boss = boss_archetype_for_wave(result.wave);

    float regular_ratio = clamp_ratio(cfg.regular_ratio);
    float fast_ratio = clamp_ratio(cfg.fast_ratio);
    float tank_ratio = clamp_ratio(cfg.tank_ratio);
    float boss_ratio = clamp_ratio(cfg.boss_ratio);
    float ratio_sum = regular_ratio + fast_ratio + tank_ratio + boss_ratio;
    if (ratio_sum <= 0.0f) {
        regular_ratio = 1.0f;
        fast_ratio = tank_ratio = boss_ratio = 0.0f;
        ratio_sum = 1.0f;
    }

    regular_ratio /= ratio_sum;
    fast_ratio /= ratio_sum;
    tank_ratio /= ratio_sum;
    boss_ratio /= ratio_sum;

    const float total = result.requested_budget_hp;
    float fast_budget = std::min(total * fast_ratio, nonnegative_finite(budget.fast_hp_cap));
    float tank_budget = budget.tank_unlocked ? std::min(total * tank_ratio, nonnegative_finite(budget.tank_hp_cap)) : 0.0f;
    float boss_budget = budget.boss_unlocked ? std::min(total * boss_ratio, nonnegative_finite(budget.boss_hp_cap)) : 0.0f;
    float regular_budget = total * regular_ratio;

    if (!budget.tank_unlocked) regular_budget += total * tank_ratio;
    if (!budget.boss_unlocked) regular_budget += total * boss_ratio;

    float assigned = regular_budget + fast_budget + tank_budget + boss_budget;
    if (assigned < total) regular_budget += (total - assigned);

    // Slot-aware deterministic greedy:
    // 1. reserve a small regular backbone first;
    // 2. add rare high-value archetypes;
    // 3. spend the main regular budget before low-HP fast fillers;
    // 4. absorb leftover budget with regulars when count-limited special/fast budgets leave holes;
    // 5. if count-limited, absorb remaining HP by scaling regular elites.
    int min_regular = std::max(0, cfg.min_regular_count);
    float regular_reserved_budget = std::min(regular_budget, total);
    while (result.regular_count < min_regular && max_count_left(result, max_count) > 0 &&
           regular_reserved_budget >= regular.unit_hp && result.used_budget_hp + regular.unit_hp <= total) {
        push_enemy(result, regular, 0);
        regular_reserved_budget -= regular.unit_hp;
        regular_budget = std::max(0.0f, regular_budget - regular.unit_hp);
    }

    add_by_budget(result, boss, std::min(boss_budget, remaining_total_budget(result, total)), max_count_left(result, max_count), 3);
    add_by_budget(result, tank, std::min(tank_budget, remaining_total_budget(result, total)), max_count_left(result, max_count), 2);
    add_by_budget(result, regular, std::min(regular_budget, remaining_total_budget(result, total)), max_count_left(result, max_count), 0);
    add_by_budget(result, fast, std::min(fast_budget, remaining_total_budget(result, total)), max_count_left(result, max_count), 1);

    while (max_count_left(result, max_count) > 0 && result.used_budget_hp + regular.unit_hp <= total) {
        push_enemy(result, regular, 0);
    }

    if (result.regular_count == 0 && max_count_left(result, max_count) > 0 && result.used_budget_hp + regular.unit_hp <= total) {
        push_enemy(result, regular, 0);
    }

    absorb_leftover_with_elites(result, total, regular, tank, boss, cfg);

    result.used_budget_hp = nonnegative_finite(result.used_budget_hp);
    if (result.used_budget_hp > total) {
        // Should not happen; keep the invariant explicit for release builds.
        result.used_budget_hp = total;
    }
    result.unused_budget_hp = std::max(0.0f, total - result.used_budget_hp);
    return result;
}

} // namespace tdmz
