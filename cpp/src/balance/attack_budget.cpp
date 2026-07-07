#include "tdmz/balance/attack_budget.hpp"
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

} // namespace

float estimate_wave_attack_cap_hp(int wave, const AttackBudgetConfig& cfg) {
    int w = std::max(1, wave);
    float wf = static_cast<float>(w);
    float cap = cfg.wave_base_hp + cfg.wave_linear_hp * wf + cfg.wave_quadratic_hp * wf * wf;
    cap *= std::max(0.0f, cfg.wave_cap_multiplier);
    return nonnegative_finite(cap);
}

bool attack_budget_tank_unlocked(int wave, const AttackBudgetConfig& cfg) {
    return wave >= std::max(1, cfg.tank_unlock_wave);
}

bool attack_budget_boss_unlocked(int wave, const AttackBudgetConfig& cfg) {
    int unlock = std::max(1, cfg.boss_unlock_wave);
    if (wave < unlock) return false;
    if (cfg.boss_odd_waves_only && (wave % 2 == 0)) return false;
    return true;
}

AttackBudgetResult estimate_attack_budget(const TDEngine& env, const AttackBudgetConfig& cfg) {
    AttackBudgetResult result;
    result.wave = env.wave();
    result.defense = estimate_defense_capacity(env, cfg.defense);
    result.defense_allowed_attack_hp = nonnegative_finite(result.defense.allowed_attack_hp);
    result.wave_cap_hp = estimate_wave_attack_cap_hp(result.wave, cfg);

    if (cfg.enable_wave_cap) {
        result.allowed_attack_hp = std::min(result.defense_allowed_attack_hp, result.wave_cap_hp);
        result.wave_cap_applied = result.wave_cap_hp < result.defense_allowed_attack_hp;
    } else {
        result.allowed_attack_hp = result.defense_allowed_attack_hp;
        result.wave_cap_applied = false;
    }

    result.allowed_attack_hp = nonnegative_finite(result.allowed_attack_hp);
    result.tank_unlocked = attack_budget_tank_unlocked(result.wave, cfg);
    result.boss_unlocked = attack_budget_boss_unlocked(result.wave, cfg);

    result.regular_hp_cap = result.allowed_attack_hp;
    result.fast_hp_cap = result.allowed_attack_hp * clamp_ratio(cfg.max_fast_ratio);
    result.tank_hp_cap = result.tank_unlocked ? result.allowed_attack_hp * clamp_ratio(cfg.max_tank_ratio) : 0.0f;
    result.boss_hp_cap = result.boss_unlocked ? result.allowed_attack_hp * clamp_ratio(cfg.max_boss_ratio) : 0.0f;
    return result;
}

} // namespace tdmz
