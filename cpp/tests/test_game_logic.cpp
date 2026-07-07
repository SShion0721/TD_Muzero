#include "tdmz/core/action.hpp"
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

void test_path_on_empty_grid() {
    TDEngine env(11, 11, 0);
    auto path = env.find_path(env.spawn_x(), env.spawn_y(), env.base_x(), env.base_y());
    CHECK_TRUE(path.size() == 10);
    CHECK_TRUE(path.front().first == 1);
    CHECK_TRUE(path.front().second == env.spawn_y());
    CHECK_TRUE(path.back().first == env.base_x());
    CHECK_TRUE(path.back().second == env.base_y());
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

int main() {
    test_path_on_empty_grid();
    test_place_upgrade_sell_economy();
    test_invalid_action_penalty();
    test_spawn_and_enemy_movement();
    test_base_takes_damage_without_towers();
    test_sniper_shoots_and_enters_cooldown();
    test_slow_tower_applies_slow();
    test_build_legality_basics();
    std::cout << "Game logic tests passed!" << std::endl;
    return 0;
}
