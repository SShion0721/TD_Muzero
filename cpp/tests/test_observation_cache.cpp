#include "tdmz/core/action.hpp"
#include "tdmz/core/observation.hpp"
#include <algorithm>
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
        const float cooldown_ratio = tower.cooldown_max > 0.0f ? tower.cooldown / tower.cooldown_max : 0.0f;
        at(CH_TOWER_COOLDOWN, y, x) = std::clamp(cooldown_ratio, 0.0f, 1.0f);
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

static void assert_vectors_equal(const Observation& actual, const Observation& expected, const char* label) {
    CHECK_TRUE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > 1e-6f) {
            throw std::runtime_error(std::string("observation mismatch at ") + label + " index " + std::to_string(i));
        }
    }
}

static void assert_obs_equal(const TDEngine& env, const char* label) {
    Observation cached = make_observation_v1(env);
    Observation reused;
    make_observation_v1_into(env, reused);
    Observation ref = make_observation_v1_reference(env);
    assert_vectors_equal(cached, ref, label);
    assert_vectors_equal(reused, ref, label);
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

void test_exact_key_distinguishes_same_blocked_cells_with_different_towers() {
    TDEngine basic(11, 11, 0);
    TDEngine slow(11, 11, 0);
    CHECK_TRUE(basic.place_tower(1, 1, TowerType::Basic));
    CHECK_TRUE(slow.place_tower(1, 1, TowerType::Slow));
    CHECK_TRUE(basic.blocked_bitboard() == slow.blocked_bitboard());

    Observation basic_obs;
    Observation slow_obs;
    make_observation_v1_into(basic, basic_obs);
    make_observation_v1_into(slow, slow_obs);

    const int basic_index = (CH_TOWER_BASIC * kBoardH + 1) * kBoardW + 1;
    const int slow_index = (CH_TOWER_SLOW * kBoardH + 1) * kBoardW + 1;
    CHECK_TRUE(basic_obs[basic_index] == 1.0f);
    CHECK_TRUE(basic_obs[slow_index] == 0.0f);
    CHECK_TRUE(slow_obs[basic_index] == 0.0f);
    CHECK_TRUE(slow_obs[slow_index] == 1.0f);
}

void test_reused_buffer_keeps_allocation_and_matches_wrapper() {
    TDEngine env(11, 11, 7);
    CHECK_TRUE(env.place_tower(1, 4, TowerType::Basic));

    Observation reused;
    make_observation_v1_into(env, reused);
    CHECK_TRUE(reused.size() == static_cast<size_t>(OBS_CHANNELS * kBoardH * kBoardW));
    const float* first_data = reused.data();
    const size_t first_capacity = reused.capacity();

    for (int i = 0; i < 20; ++i) {
        env.step_wait(1);
        make_observation_v1_into(env, reused);
        CHECK_TRUE(reused.data() == first_data);
        CHECK_TRUE(reused.capacity() == first_capacity);
        assert_vectors_equal(reused, make_observation_v1(env), "reused_buffer");
        if (env.game_over()) env.reset(static_cast<uint64_t>(100 + i));
    }
}

void test_reused_buffer_clears_previous_dynamic_channels() {
    TDEngine active(11, 11, 0);
    active.step_wait(1);

    Observation reused;
    make_observation_v1_into(active, reused);
    float density_sum = 0.0f;
    for (int i = 0; i < kBoardW * kBoardH; ++i) {
        density_sum += reused[CH_ENEMY_DENSITY * kBoardW * kBoardH + i];
    }
    CHECK_TRUE(density_sum > 0.0f);

    TDEngine empty(11, 11, 0);
    make_observation_v1_into(empty, reused);
    for (int channel = CH_ENEMY_HP; channel <= CH_ENEMY_SLOW_TIMER; ++channel) {
        for (int i = 0; i < kBoardW * kBoardH; ++i) {
            CHECK_TRUE(reused[channel * kBoardW * kBoardH + i] == 0.0f);
        }
    }
}

void test_cooldown_channel_is_clamped() {
    TDEngine env(11, 11, 0);
    CHECK_TRUE(env.place_tower(1, 1, TowerType::Basic));

    auto& towers = const_cast<std::vector<Tower>&>(env.towers());
    towers.front().cooldown = -0.5f;

    Observation obs;
    make_observation_v1_into(env, obs);
    const int index = (CH_TOWER_COOLDOWN * kBoardH + 1) * kBoardW + 1;
    CHECK_TRUE(obs[index] == 0.0f);

    towers.front().cooldown = towers.front().cooldown_max * 2.0f;
    make_observation_v1_into(env, obs);
    CHECK_TRUE(obs[index] == 1.0f);
}

int main() {
    try {
        test_static_cache_matches_reference_across_versions();
        test_cache_switches_between_env_instances();
        test_exact_key_distinguishes_same_blocked_cells_with_different_towers();
        test_reused_buffer_keeps_allocation_and_matches_wrapper();
        test_reused_buffer_clears_previous_dynamic_channels();
        test_cooldown_channel_is_clamped();
        std::cout << "Observation cache tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
