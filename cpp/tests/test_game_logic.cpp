#include "tdmz/core/action.hpp"
#include "tdmz/core/enemy.hpp"
#include "tdmz/core/engine.hpp"
#include "tdmz/core/tower.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace tdmz;

static void check_true(bool ok, const char* expr, int line) {
    if (!ok) throw std::runtime_error(std::string("check failed at line ") + std::to_string(line) + ": " + expr);
}
#define CHECK_TRUE(x) check_true(static_cast<bool>(x), #x, __LINE__)

void test_constructor_rejects_non_11x11() {
    bool threw = false;
    try {
        TDEngine env(9, 9, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK_TRUE(threw);
}

void test_path_on_empty_grid() {
    TDEngine env(11, 11, 0);
    auto path = env.find_path(env.spawn_x(), env.spawn_y(), env.base_x(), env.base_y());
    CHECK_TRUE(path.size() == 10);
    CHECK_TRUE(path.front().first == 1);
    CHECK_TRUE(path.front().second == env.spawn_y());
    CHECK_TRUE(path.back().first == env.base_x());
    CHECK_TRUE(path.back().second == env.base_y());
}

void test_fast_enemy_consumes_full_movement_distance() {
    Enemy enemy(1, 0.0f, 0.0f, 10.0f, 2.8f, 1);
    enemy.set_path({{1, 0}, {2, 0}, {3, 0}, {4, 0}});

    enemy.step(1.0f);

    CHECK_TRUE(std::abs(enemy.x - 2.8f) < 1e-5f);
    CHECK_TRUE(std::abs(enemy.y) < 1e-5f);
    CHECK_TRUE(enemy.path.size() == 2);
    CHECK_TRUE(std::abs(enemy.target_x - 3.0f) < 1e-5f);
    CHECK_TRUE(std::abs(enemy.target_y) < 1e-5f);
}

void test_slow_duration_spans_full_time_window() {
    Enemy enemy(2, 0.0f, 0.0f, 10.0f, 2.0f, 1);
    enemy.set_path({{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}});
    enemy.apply_slow(0.5f, 2.0f);

    enemy.step(1.0f);
    CHECK_TRUE(std::abs(enemy.x - 1.0f) < 1e-5f);
    CHECK_TRUE(std::abs(enemy.slow_timer - 1.0f) < 1e-5f);
    CHECK_TRUE(std::abs(enemy.speed - 1.0f) < 1e-5f);

    enemy.step(1.0f);
    CHECK_TRUE(std::abs(enemy.x - 2.0f) < 1e-5f);
    CHECK_TRUE(enemy.slow_timer == 0.0f);
    CHECK_TRUE(enemy.speed == enemy.base_speed);

    enemy.step(1.0f);
    CHECK_TRUE(std::abs(enemy.x - 4.0f) < 1e-5f);
}

void test_slow_expiry_splits_large_dt() {
    Enemy enemy(3, 0.0f, 0.0f, 10.0f, 2.0f, 1);
    enemy.set_path({{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}});
    enemy.apply_slow(0.5f, 1.0f);

    enemy.step(2.0f);

    CHECK_TRUE(std::abs(enemy.x - 3.0f) < 1e-5f);
    CHECK_TRUE(enemy.slow_timer == 0.0f);
    CHECK_TRUE(enemy.speed == enemy.base_speed);
}

void test_place_upgrade_sell_economy() {
    TDEngine env(11, 11, 0);
    CHECK_TRUE(env.place_tower(1, 1, TowerType::Basic));
    CHECK_TRUE(env.money() == 150);
    CHECK_TRUE(env.grid()[1][1] == 1);
    CHECK_TRUE(env.towers().size() == 1);

    CHECK_TRUE(env.upgrade_tower(1, 1));
    CHECK_TRUE(env.money() == 75);
    CHECK_TRUE(env.towers()[0].level == 2);
    CHECK_TRUE(env.towers()[0].damage > 10.0f);

    CHECK_TRUE(env.sell_tower(1, 1));
    CHECK_TRUE(env.money() == 175);
    CHECK_TRUE(env.grid()[1][1] == 0);
    CHECK_TRUE(env.towers().empty());
}

void test_tower_upgrade_caps_at_level_five() {
    Tower tower(0, 0, TowerType::Basic);
    for (int i = 0; i < 20; ++i) tower.upgrade();
    CHECK_TRUE(tower.level == kTowerMaxLevel);
    CHECK_TRUE(!tower.can_upgrade());
}

void test_invalid_action_penalty() {
    TDEngine env(11, 11, 0);
    int upgrade_empty = encode_action(Action{ActionType::Upgrade, 0, 0, 1});
    auto result = env.step_action(upgrade_empty);
    CHECK_TRUE(!result.done);
    CHECK_TRUE(std::abs(result.reward + 5.0f) < 1e-5f);
}

void test_spawn_and_enemy_movement() {
    TDEngine env(11, 11, 0);
    auto result = env.step_wait(1);
    CHECK_TRUE(!result.done);
    CHECK_TRUE(!env.enemies().empty());
    const auto& enemy = env.enemies().front();
    CHECK_TRUE(enemy.x >= 0.9f);
    CHECK_TRUE(std::abs(enemy.y - static_cast<float>(env.spawn_y())) < 1e-5f);
    CHECK_TRUE(enemy.path.size() > 0);
}

void test_base_takes_damage_without_towers() {
    TDEngine env(11, 11, 0);
    env.step_wait(20);
    CHECK_TRUE(env.base_hp() < 100 || env.game_over());
}

void test_sniper_shoots_and_enters_cooldown() {
    TDEngine env(11, 11, 0);
    int build_sniper = encode_action(Action{ActionType::BuildSniper, 5, 4, 1});
    auto result = env.step_action(build_sniper);
    CHECK_TRUE(!result.done);
    CHECK_TRUE(env.towers().size() == 1);
    CHECK_TRUE(env.towers()[0].cooldown > 0.0f);
}

void test_slow_tower_applies_slow() {
    TDEngine env(11, 11, 0);
    int build_slow = encode_action(Action{ActionType::BuildSlow, 1, 4, 1});
    auto result = env.step_action(build_slow);
    CHECK_TRUE(!result.done);
    CHECK_TRUE(!env.enemies().empty());

    bool slowed = false;
    for (const auto& enemy : env.enemies()) {
        if (enemy.speed < enemy.base_speed || enemy.slow_timer > 0.0f) slowed = true;
    }
    CHECK_TRUE(slowed);
}

void test_build_legality_basics() {
    TDEngine env(11, 11, 0);
    CHECK_TRUE(!env.can_place_tower(env.spawn_x(), env.spawn_y(), TowerType::Basic));
    CHECK_TRUE(!env.can_place_tower(env.base_x(), env.base_y(), TowerType::Basic));
    CHECK_TRUE(!env.can_place_tower(-1, 0, TowerType::Basic));
    CHECK_TRUE(env.can_place_tower(1, 1, TowerType::Basic));
    CHECK_TRUE(env.place_tower(1, 1, TowerType::Basic));
    CHECK_TRUE(!env.can_place_tower(1, 1, TowerType::Basic));
}

void test_placeable_and_legal_actions_are_cached() {
    TDEngine env(11, 11, 0);
    env.reset_perf_counters();

    auto mask1 = env.compute_placeable_mask();
    auto mask2 = env.compute_placeable_mask();
    CHECK_TRUE(mask1 == mask2);
    CHECK_TRUE(env.perf_counters().placeable_recompute == 1);

    auto legal1 = env.legal_actions();
    auto legal2 = env.legal_actions();
    CHECK_TRUE(legal1 == legal2);
    CHECK_TRUE(env.perf_counters().legal_recompute == 1);

    CHECK_TRUE(env.place_tower(1, 1, TowerType::Basic));
    auto legal3 = env.legal_actions();
    CHECK_TRUE(!legal3.empty());
    CHECK_TRUE(env.perf_counters().legal_recompute == 2);
    CHECK_TRUE(env.perf_counters().placeable_recompute == 2);
}

int main() {
    test_constructor_rejects_non_11x11();
    test_path_on_empty_grid();
    test_fast_enemy_consumes_full_movement_distance();
    test_slow_duration_spans_full_time_window();
    test_slow_expiry_splits_large_dt();
    test_place_upgrade_sell_economy();
    test_tower_upgrade_caps_at_level_five();
    test_invalid_action_penalty();
    test_spawn_and_enemy_movement();
    test_base_takes_damage_without_towers();
    test_sniper_shoots_and_enters_cooldown();
    test_slow_tower_applies_slow();
    test_build_legality_basics();
    test_placeable_and_legal_actions_are_cached();
    std::cout << "Game logic tests passed!" << std::endl;
    return 0;
}
