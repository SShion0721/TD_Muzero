#include "tdmz/balance/attack_budget.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace tdmz;

static void check_true(bool ok, const char* expr, int line) {
    if (!ok) throw std::runtime_error(std::string("check failed at line ") + std::to_string(line) + ": " + expr);
}
#define CHECK_TRUE(x) check_true(static_cast<bool>(x), #x, __LINE__)

static void check_close(float a, float b, float eps, const char* expr, int line) {
    if (std::fabs(a - b) > eps) throw std::runtime_error(std::string("check failed at line ") + std::to_string(line) + ": " + expr);
}
#define CHECK_CLOSE(a, b, eps) check_close((a), (b), (eps), #a " ~= " #b, __LINE__)

void test_wave_cap_formula_and_unlocks() {
    AttackBudgetConfig cfg;
    CHECK_TRUE(estimate_wave_attack_cap_hp(2, cfg) > estimate_wave_attack_cap_hp(1, cfg));
    CHECK_TRUE(!attack_budget_tank_unlocked(1, cfg));
    CHECK_TRUE(attack_budget_tank_unlocked(3, cfg));
    CHECK_TRUE(!attack_budget_boss_unlocked(4, cfg));
    CHECK_TRUE(attack_budget_boss_unlocked(5, cfg));
    CHECK_TRUE(!attack_budget_boss_unlocked(6, cfg));
}

void test_empty_budget_is_wave_clamped() {
    TDEngine env(11, 11, 0);
    AttackBudgetConfig cfg;
    auto r = estimate_attack_budget(env, cfg);
    CHECK_TRUE(r.defense_allowed_attack_hp > r.wave_cap_hp);
    CHECK_TRUE(r.wave_cap_applied);
    CHECK_CLOSE(r.allowed_attack_hp, r.wave_cap_hp, 1e-5f);
    CHECK_TRUE(r.fast_hp_cap <= r.allowed_attack_hp);
    CHECK_TRUE(r.tank_hp_cap == 0.0f);
    CHECK_TRUE(r.boss_hp_cap == 0.0f);
}

void test_defense_increases_budget_when_unclamped() {
    AttackBudgetConfig cfg;
    cfg.enable_wave_cap = false;

    TDEngine empty(11, 11, 0);
    TDEngine defended(11, 11, 0);
    CHECK_TRUE(defended.place_tower(1, 4, TowerType::Basic));

    auto a = estimate_attack_budget(empty, cfg);
    auto b = estimate_attack_budget(defended, cfg);
    CHECK_TRUE(b.allowed_attack_hp > a.allowed_attack_hp);
}

void test_virtual_base_hp_reduces_budget_when_unclamped() {
    TDEngine env(11, 11, 0);
    AttackBudgetConfig high;
    high.enable_wave_cap = false;
    high.defense.virtual_base_hp = 100;
    AttackBudgetConfig low = high;
    low.defense.virtual_base_hp = 20;

    auto rh = estimate_attack_budget(env, high);
    auto rl = estimate_attack_budget(env, low);
    CHECK_TRUE(rl.allowed_attack_hp < rh.allowed_attack_hp);
}

void test_ratio_caps() {
    TDEngine env(11, 11, 0);
    AttackBudgetConfig cfg;
    cfg.enable_wave_cap = false;
    cfg.max_fast_ratio = 0.25f;
    cfg.max_tank_ratio = 0.50f;
    cfg.max_boss_ratio = 0.75f;
    cfg.tank_unlock_wave = 1;
    cfg.boss_unlock_wave = 1;
    cfg.boss_odd_waves_only = true;

    auto r = estimate_attack_budget(env, cfg);
    CHECK_CLOSE(r.fast_hp_cap, r.allowed_attack_hp * 0.25f, 1e-5f);
    CHECK_CLOSE(r.tank_hp_cap, r.allowed_attack_hp * 0.50f, 1e-5f);
    CHECK_CLOSE(r.boss_hp_cap, r.allowed_attack_hp * 0.75f, 1e-5f);
}

void test_negative_free_config() {
    TDEngine env(11, 11, 0);
    AttackBudgetConfig cfg;
    cfg.wave_base_hp = -100.0f;
    cfg.wave_linear_hp = -100.0f;
    cfg.wave_quadratic_hp = -100.0f;
    cfg.wave_cap_multiplier = -1.0f;
    cfg.max_fast_ratio = -1.0f;
    cfg.max_tank_ratio = -1.0f;
    cfg.max_boss_ratio = -1.0f;
    auto r = estimate_attack_budget(env, cfg);
    CHECK_TRUE(r.wave_cap_hp >= 0.0f);
    CHECK_TRUE(r.allowed_attack_hp >= 0.0f);
    CHECK_TRUE(r.fast_hp_cap >= 0.0f);
    CHECK_TRUE(r.tank_hp_cap >= 0.0f);
    CHECK_TRUE(r.boss_hp_cap >= 0.0f);
}

int main() {
    test_wave_cap_formula_and_unlocks();
    test_empty_budget_is_wave_clamped();
    test_defense_increases_budget_when_unclamped();
    test_virtual_base_hp_reduces_budget_when_unclamped();
    test_ratio_caps();
    test_negative_free_config();
    std::cout << "Attack budget tests passed!" << std::endl;
    return 0;
}
