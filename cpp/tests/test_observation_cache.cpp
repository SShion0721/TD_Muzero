#include "tdmz/core/action.hpp"
#include "tdmz/core/observation.hpp"
#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace tdmz;

namespace {

void check_true(bool ok, const char* expr, int line) {
    if (!ok) {
        throw std::runtime_error(
            std::string("check failed at line ") + std::to_string(line) + ": " + expr);
    }
}

#define CHECK_TRUE(x) check_true(static_cast<bool>(x), #x, __LINE__)

constexpr int kPlane = kBoardW * kBoardH;

float obs_at(const Observation& obs, int channel, int x, int y) {
    return obs[static_cast<std::size_t>((channel * kBoardH + y) * kBoardW + x)];
}

float plane_sum(const Observation& obs, int channel) {
    float total = 0.0f;
    const std::size_t offset = static_cast<std::size_t>(channel * kPlane);
    for (int i = 0; i < kPlane; ++i) total += obs[offset + static_cast<std::size_t>(i)];
    return total;
}

void assert_vectors_equal(const Observation& actual, const Observation& expected, const char* label) {
    CHECK_TRUE(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > 1e-6f) {
            throw std::runtime_error(
                std::string("observation mismatch at ") + label + " index " + std::to_string(i));
        }
    }
}

void test_schema_shape_and_bounded_values() {
    TDEngine env(11, 11, 0);
    const Observation obs = make_observation_v2(env);
    const auto shape = observation_shape_v2();

    CHECK_TRUE(shape[0] == OBS_CHANNELS);
    CHECK_TRUE(shape[1] == kBoardH);
    CHECK_TRUE(shape[2] == kBoardW);
    CHECK_TRUE(obs.size() == static_cast<std::size_t>(OBS_CHANNELS * kPlane));
    CHECK_TRUE(OBS_CHANNELS == 40);

    for (float value : obs) {
        CHECK_TRUE(std::isfinite(value));
        CHECK_TRUE(value >= 0.0f);
        CHECK_TRUE(value <= 1.0f);
    }

    CHECK_TRUE(obs_at(obs, CH_NEXT_ENEMY_PRESENT, 0, 0) == 1.0f);
    CHECK_TRUE(obs_at(obs, CH_PENDING_COUNT, 0, 0) > 0.0f);
    CHECK_TRUE(obs_at(obs, CH_NEXT_ENEMY_HP, 0, 0) > 0.0f);
    CHECK_TRUE(obs_at(obs, CH_NEXT_ENEMY_SPEED, 0, 0) > 0.0f);
    CHECK_TRUE(obs_at(obs, CH_NEXT_ENEMY_REWARD, 0, 0) > 0.0f);
}

void test_static_cache_switches_between_env_instances() {
    TDEngine basic(11, 11, 0);
    TDEngine slow(11, 11, 0);
    CHECK_TRUE(basic.place_tower(1, 1, TowerType::Basic));
    CHECK_TRUE(slow.place_tower(1, 1, TowerType::Slow));
    CHECK_TRUE(basic.blocked_bitboard() == slow.blocked_bitboard());

    Observation basic_obs;
    Observation slow_obs;
    make_observation_v2_into(basic, basic_obs);
    make_observation_v2_into(slow, slow_obs);

    CHECK_TRUE(obs_at(basic_obs, CH_TOWER_BASIC, 1, 1) == 1.0f);
    CHECK_TRUE(obs_at(basic_obs, CH_TOWER_SLOW, 1, 1) == 0.0f);
    CHECK_TRUE(obs_at(slow_obs, CH_TOWER_BASIC, 1, 1) == 0.0f);
    CHECK_TRUE(obs_at(slow_obs, CH_TOWER_SLOW, 1, 1) == 1.0f);

    Observation basic_again;
    make_observation_v2_into(basic, basic_again);
    assert_vectors_equal(basic_again, basic_obs, "cache_switch_back");
}

void test_reused_buffer_keeps_allocation_and_matches_wrapper() {
    TDEngine env(11, 11, 7);
    CHECK_TRUE(env.place_tower(1, 4, TowerType::Basic));

    Observation reused;
    make_observation_v2_into(env, reused);
    const float* first_data = reused.data();
    const std::size_t first_capacity = reused.capacity();

    for (int i = 0; i < 20; ++i) {
        env.step_wait(1);
        make_observation_v2_into(env, reused);
        CHECK_TRUE(reused.data() == first_data);
        CHECK_TRUE(reused.capacity() == first_capacity);
        assert_vectors_equal(reused, make_observation_v2(env), "reused_buffer");
        if (env.game_over()) env.reset(static_cast<uint64_t>(100 + i));
    }
}

void test_dynamic_channels_clear_between_environments() {
    TDEngine active(11, 11, 0);
    active.step_wait(1);

    Observation reused;
    make_observation_v2_into(active, reused);
    CHECK_TRUE(plane_sum(reused, CH_ENEMY_DENSITY) > 0.0f);
    CHECK_TRUE(plane_sum(reused, CH_ENEMY_CURRENT_HP) > 0.0f);

    TDEngine empty(11, 11, 0);
    make_observation_v2_into(empty, reused);
    for (int channel = CH_ENEMY_DENSITY; channel <= CH_ENEMY_OCCUPIED; ++channel) {
        CHECK_TRUE(plane_sum(reused, channel) == 0.0f);
    }
}

void test_absolute_hp_breaks_ratio_aliasing() {
    TDEngine small(11, 11, 0);
    TDEngine large(11, 11, 0);
    small.step_wait(1);
    large.step_wait(1);
    CHECK_TRUE(small.enemies().size() == 1);
    CHECK_TRUE(large.enemies().size() == 1);

    auto& small_enemy = const_cast<std::vector<Enemy>&>(small.enemies()).front();
    auto& large_enemy = const_cast<std::vector<Enemy>&>(large.enemies()).front();
    small_enemy.hp = 10.0f;
    small_enemy.max_hp = 20.0f;
    large_enemy.hp = 500.0f;
    large_enemy.max_hp = 1000.0f;

    const Observation small_obs = make_observation_v2(small);
    const Observation large_obs = make_observation_v2(large);

    CHECK_TRUE(std::fabs(
        plane_sum(small_obs, CH_ENEMY_HP_RATIO) -
        plane_sum(large_obs, CH_ENEMY_HP_RATIO)) < 1e-5f);
    CHECK_TRUE(plane_sum(large_obs, CH_ENEMY_CURRENT_HP) >
               plane_sum(small_obs, CH_ENEMY_CURRENT_HP));
    CHECK_TRUE(plane_sum(large_obs, CH_ENEMY_MAX_HP) >
               plane_sum(small_obs, CH_ENEMY_MAX_HP));
    CHECK_TRUE(plane_sum(large_obs, CH_ENEMY_LEAK_DAMAGE) >
               plane_sum(small_obs, CH_ENEMY_LEAK_DAMAGE));
}

void test_continuous_position_is_bilinearly_splatted() {
    TDEngine env(11, 11, 0);
    env.step_wait(1);
    CHECK_TRUE(env.enemies().size() == 1);

    auto& enemy = const_cast<std::vector<Enemy>&>(env.enemies()).front();
    enemy.x = 2.25f;
    enemy.y = 5.0f;
    enemy.target_x = 3.0f;
    enemy.target_y = 5.0f;

    const Observation obs = make_observation_v2(env);
    CHECK_TRUE(obs_at(obs, CH_ENEMY_DENSITY, 2, 5) > 0.0f);
    CHECK_TRUE(obs_at(obs, CH_ENEMY_DENSITY, 3, 5) > 0.0f);
    CHECK_TRUE(obs_at(obs, CH_ENEMY_DIR_RIGHT, 2, 5) > 0.0f);
    CHECK_TRUE(obs_at(obs, CH_ENEMY_DIR_LEFT, 2, 5) == 0.0f);

    // Legal occupancy intentionally keeps the engine's rounded-cell semantics.
    CHECK_TRUE(obs_at(obs, CH_ENEMY_OCCUPIED, 2, 5) == 1.0f);
    CHECK_TRUE(obs_at(obs, CH_ENEMY_OCCUPIED, 3, 5) == 0.0f);
}

void test_structural_placeability_is_distinct_from_enemy_occupancy() {
    TDEngine env(11, 11, 0);
    env.step_wait(1);
    CHECK_TRUE(!env.enemies().empty());

    const Enemy& enemy = env.enemies().front();
    const int x = static_cast<int>(std::round(enemy.x));
    const int y = static_cast<int>(std::round(enemy.y));
    const Observation obs = make_observation_v2(env);

    CHECK_TRUE(obs_at(obs, CH_STRUCTURAL_PLACEABLE, x, y) == 1.0f);
    CHECK_TRUE(obs_at(obs, CH_ENEMY_OCCUPIED, x, y) == 1.0f);
    CHECK_TRUE(!env.can_place_tower(x, y, TowerType::Basic));

    const auto legal_mask = env.legal_action_mask();
    const int action = encode_action(Action{ActionType::BuildBasic, x, y, 1});
    CHECK_TRUE(legal_mask[static_cast<std::size_t>(action)] == 0u);
}

void test_pending_features_clear_when_queue_is_empty() {
    TDEngine env(11, 11, 0);
    for (int i = 0; i < 64 && env.enemies_to_spawn_count() > 0 && !env.game_over(); ++i) {
        env.step_wait(1);
    }
    CHECK_TRUE(env.enemies_to_spawn_count() == 0);

    const Observation obs = make_observation_v2(env);
    CHECK_TRUE(obs_at(obs, CH_NEXT_ENEMY_PRESENT, 0, 0) == 0.0f);
    CHECK_TRUE(obs_at(obs, CH_NEXT_ENEMY_HP, 0, 0) == 0.0f);
    CHECK_TRUE(obs_at(obs, CH_NEXT_ENEMY_SPEED, 0, 0) == 0.0f);
    CHECK_TRUE(obs_at(obs, CH_SPAWN_TIMER, 0, 0) == 0.0f);
    CHECK_TRUE(obs_at(obs, CH_PENDING_COUNT, 0, 0) == 0.0f);
}

void test_cooldown_channel_is_clamped() {
    TDEngine env(11, 11, 0);
    CHECK_TRUE(env.place_tower(1, 1, TowerType::Basic));

    auto& tower = const_cast<std::vector<Tower>&>(env.towers()).front();
    tower.cooldown = -0.5f;
    Observation obs = make_observation_v2(env);
    CHECK_TRUE(obs_at(obs, CH_TOWER_COOLDOWN, 1, 1) == 0.0f);

    tower.cooldown = tower.cooldown_max * 2.0f;
    obs = make_observation_v2(env);
    CHECK_TRUE(obs_at(obs, CH_TOWER_COOLDOWN, 1, 1) == 1.0f);
}

} // namespace

int main() {
    try {
        test_schema_shape_and_bounded_values();
        test_static_cache_switches_between_env_instances();
        test_reused_buffer_keeps_allocation_and_matches_wrapper();
        test_dynamic_channels_clear_between_environments();
        test_absolute_hp_breaks_ratio_aliasing();
        test_continuous_position_is_bilinearly_splatted();
        test_structural_placeability_is_distinct_from_enemy_occupancy();
        test_pending_features_clear_when_queue_is_empty();
        test_cooldown_channel_is_clamped();
        std::cout << "Observation schema v2 tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
