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

float base_hp_for_wave(int wave) {
    int w = std::max(1, wave);
    return static_cast<float>(20 + w * 15 + w * w * 2);
}

void push_enemy(BudgetedWaveResult& result, const EnemyArchetypeSpec& arch, int kind) {
    result.enemies.push_back(arch.spec);
    result.used_budget_hp += arch.unit_hp;
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

    int max_left = max_count;

    add_by_budget(result, boss, boss_budget, max_left, 3);
    max_left = max_count - static_cast<int>(result.enemies.size());
    add_by_budget(result, tank, tank_budget, max_left, 2);
    max_left = max_count - static_cast<int>(result.enemies.size());
    add_by_budget(result, fast, fast_budget, max_left, 1);
    max_left = max_count - static_cast<int>(result.enemies.size());

    float used_before_regular = result.used_budget_hp;
    float budget_left_for_regular = std::max(0.0f, total - used_before_regular);
    float regular_budget_capped = std::min(regular_budget, budget_left_for_regular);
    int regular_added = add_by_budget(result, regular, regular_budget_capped, max_left, 0);
    max_left = max_count - static_cast<int>(result.enemies.size());

    int min_regular = std::max(0, cfg.min_regular_count);
    while (result.regular_count < min_regular && max_left > 0 && result.used_budget_hp + regular.unit_hp <= total) {
        push_enemy(result, regular, 0);
        --max_left;
        ++regular_added;
    }

    if (result.regular_count == 0 && max_left > 0 && result.used_budget_hp + regular.unit_hp <= total) {
        push_enemy(result, regular, 0);
        --max_left;
    }

    result.used_budget_hp = nonnegative_finite(result.used_budget_hp);
    if (result.used_budget_hp > total) {
        // Should not happen; keep the invariant explicit for release builds.
        result.used_budget_hp = total;
    }
    result.unused_budget_hp = std::max(0.0f, total - result.used_budget_hp);
    return result;
}

} // namespace tdmz
