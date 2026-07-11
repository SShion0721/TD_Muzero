#include "tdmz/balance/defense_capacity.hpp"
#include "tdmz/core/action.hpp"
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

void test_empty_defense_has_leak_cap() {
    TDEngine env(11, 11, 0);
    DefenseCapacityConfig cfg;
    cfg.include_spendable_money = false;
    auto r = estimate_defense_capacity(env, cfg);
    CHECK_CLOSE(r.current_tower_damage_cap, 0.0f, 1e-5f);
    CHECK_CLOSE(r.spendable_money_damage_cap, 0.0f, 1e-5f);
    CHECK_CLOSE(r.leak_hp_cap, 990.0f, 1e-5f);
    CHECK_CLOSE(r.allowed_attack_hp, 891.0f, 1e-5f);
    CHECK_TRUE(r.placeable_count > 0);
    CHECK_TRUE(r.path_len > 0);
}

void test_half_open_shot_timing() {
    CHECK_CLOSE(
        estimate_tower_damage_capacity(TowerType::Basic, 1, 1.0f, 1.0f, 1.0f),
        10.0f,
        1e-5f);
    CHECK_CLOSE(
        estimate_tower_damage_capacity(TowerType::Basic, 1, 2.0f, 1.0f, 1.0f),
        20.0f,
        1e-5f);
    CHECK_CLOSE(
        estimate_tower_damage_capacity(TowerType::AOE, 1, 2.0f, 1.0f, 1.0f),
        30.0f,
        1e-5f);

    Tower tower(0, 0, TowerType::Basic);
    tower.cooldown = 1.0f;
    CHECK_CLOSE(
        estimate_existing_tower_damage_capacity(tower, 1.0f, 1.0f, 1.0f),
        0.0f,
        1e-5f);
    tower.cooldown = 0.5f;
    CHECK_CLOSE(
        estimate_existing_tower_damage_capacity(tower, 1.0f, 1.0f, 1.0f),
        10.0f,
        1e-5f);
}

void test_path_tower_cap_increases() {
    TDEngine empty(11, 11, 0);
    DefenseCapacityConfig cfg;
    cfg.include_spendable_money = false;
    auto r0 = estimate_defense_capacity(empty, cfg);

    TDEngine basic(11, 11, 0);
    CHECK_TRUE(basic.place_tower(1, 4, TowerType::Basic));
    auto r1 = estimate_defense_capacity(basic, cfg);
    CHECK_TRUE(r1.current_tower_damage_cap > r0.current_tower_damage_cap);
    CHECK_TRUE(r1.allowed_attack_hp > r0.allowed_attack_hp);
    CHECK_TRUE(r1.path_coverage_factor_sum > 0.0f);
}

void test_off_path_tower_is_discounted() {
    TDEngine env(11, 11, 0);
    CHECK_TRUE(env.place_tower(1, 1, TowerType::Basic));

    DefenseCapacityConfig path_aware;
    path_aware.include_spendable_money = false;
    path_aware.path_aware = true;

    DefenseCapacityConfig static_cfg = path_aware;
    static_cfg.path_aware = false;

    auto rp = estimate_defense_capacity(env, path_aware);
    auto rs = estimate_defense_capacity(env, static_cfg);
    CHECK_TRUE(rs.current_tower_damage_cap > 0.0f);
    CHECK_TRUE(rp.current_tower_damage_cap < rs.current_tower_damage_cap);
    CHECK_CLOSE(rp.current_tower_damage_cap, 0.0f, 1e-5f);
}

void test_sniper_cap_beats_basic_on_path() {
    DefenseCapacityConfig cfg;
    cfg.include_spendable_money = false;

    TDEngine basic(11, 11, 0);
    TDEngine sniper(11, 11, 0);
    CHECK_TRUE(basic.place_tower(1, 4, TowerType::Basic));
    CHECK_TRUE(sniper.place_tower(1, 4, TowerType::Sniper));

    auto rb = estimate_defense_capacity(basic, cfg);
    auto rs = estimate_defense_capacity(sniper, cfg);
    CHECK_TRUE(rs.current_tower_damage_cap > rb.current_tower_damage_cap);
}

void test_virtual_base_hp_and_safety_factor() {
    TDEngine env(11, 11, 0);
    DefenseCapacityConfig high;
    high.include_spendable_money = false;
    high.virtual_base_hp = 100;
    high.safety_factor = 1.0f;

    DefenseCapacityConfig low = high;
    low.virtual_base_hp = 20;
    auto rh = estimate_defense_capacity(env, high);
    auto rl = estimate_defense_capacity(env, low);
    CHECK_TRUE(rl.leak_hp_cap < rh.leak_hp_cap);

    DefenseCapacityConfig half = high;
    half.safety_factor = 0.5f;
    auto rhalf = estimate_defense_capacity(env, half);
    CHECK_CLOSE(rhalf.allowed_attack_hp, rh.allowed_attack_hp * 0.5f, 1e-5f);
}

void test_path_aware_spendable_money_adds_capacity() {
    TDEngine env(11, 11, 0);
    DefenseCapacityConfig without;
    without.include_spendable_money = false;
    DefenseCapacityConfig with;
    with.include_spendable_money = true;

    auto r0 = estimate_defense_capacity(env, without);
    auto r1 = estimate_defense_capacity(env, with);
    CHECK_TRUE(r1.spendable_money_damage_cap > 0.0f);
    CHECK_TRUE(r1.selected_new_tower_count > 0 || r1.selected_upgrade_count > 0);
    CHECK_TRUE(r1.build_candidate_count > 0);
    CHECK_TRUE(r1.best_build_value_per_cost > 0.0f);
    CHECK_TRUE(r1.raw_hp_cap > r0.raw_hp_cap);
    CHECK_TRUE(r1.allowed_attack_hp > r0.allowed_attack_hp);
}

void test_upgrade_candidate_is_detected() {
    TDEngine env(11, 11, 0);
    CHECK_TRUE(env.place_tower(1, 4, TowerType::Basic));
    DefenseCapacityConfig cfg;
    cfg.include_spendable_money = true;
    cfg.include_upgrade_candidates = true;
    auto r = estimate_defense_capacity(env, cfg);
    CHECK_TRUE(r.upgrade_candidate_count > 0);
    CHECK_TRUE(r.best_upgrade_value_per_cost > 0.0f);
}

void test_negative_free_result() {
    TDEngine env(11, 11, 0);
    DefenseCapacityConfig cfg;
    cfg.wave_window_seconds = -1.0f;
    cfg.safety_factor = -1.0f;
    cfg.virtual_base_hp = -10;
    auto r = estimate_defense_capacity(env, cfg);
    CHECK_TRUE(r.current_tower_damage_cap >= 0.0f);
    CHECK_TRUE(r.spendable_money_damage_cap >= 0.0f);
    CHECK_TRUE(r.leak_hp_cap >= 0.0f);
    CHECK_TRUE(r.raw_hp_cap >= 0.0f);
    CHECK_TRUE(r.allowed_attack_hp >= 0.0f);
}

int main() {
    test_empty_defense_has_leak_cap();
    test_half_open_shot_timing();
    test_path_tower_cap_increases();
    test_off_path_tower_is_discounted();
    test_sniper_cap_beats_basic_on_path();
    test_virtual_base_hp_and_safety_factor();
    test_path_aware_spendable_money_adds_capacity();
    test_upgrade_candidate_is_detected();
    test_negative_free_result();
    std::cout << "Defense capacity tests passed!" << std::endl;
    return 0;
}
