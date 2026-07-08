#include "tdmz/core/action.hpp"
#include "tdmz/core/observation.hpp"
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace tdmz;

static void check_true(bool ok, const char* expr, int line) {
    if (!ok) throw std::runtime_error(std::string("check failed at line ") + std::to_string(line) + ": " + expr);
}
#define CHECK_TRUE(x) check_true(static_cast<bool>(x), #x, __LINE__)

static Observation make_observation_v1_reference(const TDEngine& env) {
    int w = env.width();
    int h = env.height();
    Observation obs(OBS_CHANNELS * h * w, 0.0f);

    auto at = [&](int c, int y, int x) -> float& {
        return obs[(c * h + y) * w + x];
    };

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (env.grid()[y][x] == 1) at(CH_BLOCKED, y, x) = 1.0f;
        }
    }
    at(CH_SPAWN, env.spawn_y(), env.spawn_x()) = 1.0f;
    at(CH_BASE, env.base_y(), env.base_x()) = 1.0f;

    for (const auto& tower : env.towers()) {
        int x = tower.x;
        int y = tower.y;
        if (tower.type == TowerType::Basic) at(CH_TOWER_BASIC, y, x) = 1.0f;
        else if (tower.type == TowerType::Sniper) at(CH_TOWER_SNIPER, y, x) = 1.0f;
        else if (tower.type == TowerType::AOE) at(CH_TOWER_AOE, y, x) = 1.0f;
        else if (tower.type == TowerType::Slow) at(CH_TOWER_SLOW, y, x) = 1.0f;

        at(CH_TOWER_LEVEL, y, x) = std::min(1.0f, static_cast<float>(tower.level) / static_cast<float>(kTowerMaxLevel));
        at(CH_TOWER_COOLDOWN, y, x) = tower.cooldown / tower.cooldown_max;
    }

    for (const auto& enemy : env.enemies()) {
        int gx = static_cast<int>(std::round(enemy.x));
        int gy = static_cast<int>(std::round(enemy.y));
        if (gx >= 0 && gx < w && gy >= 0 && gy < h) {
            at(CH_ENEMY_HP, gy, gx) += enemy.hp / std::max(1.0f, enemy.max_hp);
            at(CH_ENEMY_DENSITY, gy, gx) += 1.0f;
            at(CH_ENEMY_SPEED, gy, gx) = std::max(at(CH_ENEMY_SPEED, gy, gx), enemy.speed / 3.0f);
            at(CH_ENEMY_SLOW_TIMER, gy, gx) = std::max(at(CH_ENEMY_SLOW_TIMER, gy, gx), enemy.slow_timer / 3.0f);
        }
    }

    float base_hp_val = env.base_hp() / 100.0f;
    float money_val = std::min(1.0f, env.money() / 1000.0f);
    float wave_val = env.wave() / 20.0f;
    float spawn_timer_val = env.spawn_timer() / 3.0f;
    float to_spawn_val = env.enemies_to_spawn_count() / 20.0f;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            at(CH_BASE_HP, y, x) = base_hp_val;
            at(CH_MONEY, y, x) = money_val;
            at(CH_WAVE, y, x) = wave_val;
            at(CH_SPAWN_TIMER, y, x) = spawn_timer_val;
            at(CH_TO_SPAWN_COUNT, y, x) = to_spawn_val;
        }
    }

    auto placeable_mask = env.compute_placeable_mask();
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int cell = cell_id(x, y);
            if (placeable_mask[y][x]) {
                at(CH_PLACEABLE_MASK, y, x) = 1.0f;
            }
            int dist = env.distance_to_base_cell(cell);
            if (dist < 1000000000) {
                at(CH_DISTANCE_TO_BASE, y, x) = static_cast<float>(dist) / 20.0f;
            }
        }
    }

    return obs;
}

static void assert_obs_equal(const TDEngine& env, const char* label) {
    Observation cached = make_observation_v1(env);
    Observation ref = make_observation_v1_reference(env);
    CHECK_TRUE(cached.size() == ref.size());
    for (size_t i = 0; i < cached.size(); ++i) {
        if (std::fabs(cached[i] - ref[i]) > 1e-6f) {
            throw std::runtime_error(std::string("observation mismatch at ") + label + " index " + std::to_string(i));
        }
    }
}

void test_static_cache_matches_reference_across_versions() {
    TDEngine env(11, 11, 0);
    assert_obs_equal(env, "initial");

    CHECK_TRUE(env.place_tower(1, 4, TowerType::Basic));
    assert_obs_equal(env, "after_build_basic");

    env.step_wait(3);
    assert_obs_equal(env, "after_enemy_steps");
    assert_obs_equal(env, "repeat_same_versions");

    CHECK_TRUE(env.place_tower(5, 4, TowerType::Sniper));
    assert_obs_equal(env, "after_build_sniper");

    while (env.money() < env.towers().front().upgrade_cost && !env.game_over()) {
        env.step_wait(1);
    }
    if (!env.game_over() && env.towers().front().can_upgrade()) {
        env.upgrade_tower(1, 4);
        assert_obs_equal(env, "after_upgrade");
    }

    env.step_wait(5);
    assert_obs_equal(env, "after_more_dynamic_steps");
}

void test_cache_switches_between_env_instances() {
    TDEngine a(11, 11, 0);
    TDEngine b(11, 11, 1);
    CHECK_TRUE(a.place_tower(1, 4, TowerType::Basic));
    CHECK_TRUE(b.place_tower(1, 6, TowerType::Slow));

    assert_obs_equal(a, "env_a_first");
    assert_obs_equal(b, "env_b_second");
    assert_obs_equal(a, "env_a_again_after_b");
}

int main() {
    try {
        test_static_cache_matches_reference_across_versions();
        test_cache_switches_between_env_instances();
        std::cout << "Observation cache tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
