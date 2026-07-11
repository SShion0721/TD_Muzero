#include "tdmz/core/engine.hpp"
#include "tdmz/core/tower.hpp"
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tdmz;

namespace {

void check_true(bool ok, const char* expr, int line) {
    if (!ok) {
        throw std::runtime_error(
            std::string("check failed at line ") + std::to_string(line) + ": " + expr
        );
    }
}

#define CHECK_TRUE(x) check_true(static_cast<bool>(x), #x, __LINE__)

bool near_float(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

int consume_shot_schedule(Tower& tower, float dt) {
    float remaining = dt;
    int shots = 0;
    while (tower.advance_until_ready(remaining)) {
        ++shots;
        if (shots > 256) throw std::runtime_error("tower timing test exceeded shot guard");
        tower.shoot();
    }
    return shots;
}

void test_default_one_second_cooldown_fires_once() {
    Tower tower(0, 0, TowerType::Basic);
    CHECK_TRUE(near_float(tower.cooldown_max, 1.0f));
    CHECK_TRUE(consume_shot_schedule(tower, 1.0f) == 1);
    CHECK_TRUE(near_float(tower.cooldown, 0.0f));
}

void test_half_second_cooldown_fires_twice() {
    Tower tower(0, 0, TowerType::Basic);
    tower.cooldown_max = 0.5f;
    CHECK_TRUE(consume_shot_schedule(tower, 1.0f) == 2);
    CHECK_TRUE(near_float(tower.cooldown, 0.0f));
}

void test_fractional_cooldown_preserves_phase() {
    Tower tower(0, 0, TowerType::Basic);
    tower.cooldown_max = 0.9f;
    CHECK_TRUE(consume_shot_schedule(tower, 1.0f) == 2);
    CHECK_TRUE(near_float(tower.cooldown, 0.8f));

    CHECK_TRUE(consume_shot_schedule(tower, 1.0f) == 1);
    CHECK_TRUE(near_float(tower.cooldown, 0.7f));
}

void test_upgraded_tower_has_real_rate_gain() {
    Tower base(0, 0, TowerType::Basic);
    Tower upgraded(0, 0, TowerType::Basic);
    upgraded.upgrade();

    const int base_shots = consume_shot_schedule(base, 9.0f);
    const int upgraded_shots = consume_shot_schedule(upgraded, 9.0f);

    CHECK_TRUE(base_shots == 9);
    CHECK_TRUE(upgraded_shots == 10);
    CHECK_TRUE(upgraded_shots > base_shots);
}

void test_batched_and_split_cooldown_windows_match() {
    Tower batched(0, 0, TowerType::Basic);
    Tower split(0, 0, TowerType::Basic);
    batched.cooldown_max = 0.9f;
    split.cooldown_max = 0.9f;

    const int batched_shots = consume_shot_schedule(batched, 10.0f);
    int split_shots = 0;
    for (int i = 0; i < 10; ++i) {
        split_shots += consume_shot_schedule(split, 1.0f);
    }

    CHECK_TRUE(batched_shots == split_shots);
    CHECK_TRUE(near_float(batched.cooldown, split.cooldown));
}

void test_terminal_tick_advances_engine_time() {
    TDEngine env(11, 11, 123);
    bool observed_terminal = false;

    for (int tick = 0; tick < 5000; ++tick) {
        const float before = env.time();
        const StepResult result = env.step_wait(1);
        if (result.done) {
            CHECK_TRUE(env.game_over());
            CHECK_TRUE(near_float(env.time(), before + 1.0f));
            observed_terminal = true;
            break;
        }
    }

    CHECK_TRUE(observed_terminal);
}

struct EngineSnapshot {
    int money = 0;
    int base_hp = 0;
    int wave = 0;
    float spawn_timer = 0.0f;
    float time = 0.0f;
    bool game_over = false;
    bool use_budgeted_waves = false;
    uint64_t seed = 0;
    uint64_t grid_version = 0;
    uint64_t tower_version = 0;
    uint64_t money_version = 0;
    uint64_t enemy_version = 0;
    uint64_t wave_version = 0;
    std::array<std::array<int, kBoardW>, kBoardH> grid{};
    Bitboard128 blocked{};
    std::vector<Tower> towers;
    std::vector<Enemy> enemies;
    std::vector<EnemySpec> pending;
};

EngineSnapshot capture(const TDEngine& env) {
    EngineSnapshot snapshot;
    snapshot.money = env.money();
    snapshot.base_hp = env.base_hp();
    snapshot.wave = env.wave();
    snapshot.spawn_timer = env.spawn_timer();
    snapshot.time = env.time();
    snapshot.game_over = env.game_over();
    snapshot.use_budgeted_waves = env.use_budgeted_waves();
    snapshot.seed = env.seed();
    snapshot.grid_version = env.grid_version();
    snapshot.tower_version = env.tower_version();
    snapshot.money_version = env.money_version();
    snapshot.enemy_version = env.enemy_version();
    snapshot.wave_version = env.wave_version();
    snapshot.grid = env.grid();
    snapshot.blocked = env.blocked_bitboard();
    snapshot.towers = env.towers();
    snapshot.enemies = env.enemies();
    snapshot.pending.assign(env.pending_spawns().begin(), env.pending_spawns().end());
    return snapshot;
}

bool same_tower(const Tower& left, const Tower& right) {
    return left.x == right.x && left.y == right.y && left.type == right.type &&
           left.cost == right.cost && left.damage == right.damage &&
           left.range == right.range && left.cooldown_max == right.cooldown_max &&
           left.aoe_radius == right.aoe_radius &&
           left.slow_factor == right.slow_factor &&
           left.slow_duration == right.slow_duration &&
           left.cooldown == right.cooldown && left.level == right.level &&
           left.total_spent == right.total_spent &&
           left.upgrade_cost == right.upgrade_cost;
}

bool same_enemy(const Enemy& left, const Enemy& right) {
    return left.id == right.id && left.x == right.x && left.y == right.y &&
           left.target_x == right.target_x && left.target_y == right.target_y &&
           left.hp == right.hp && left.max_hp == right.max_hp &&
           left.speed == right.speed && left.base_speed == right.base_speed &&
           left.slow_timer == right.slow_timer && left.reward == right.reward &&
           left.path == right.path && left.path_cursor == right.path_cursor;
}

void check_same_snapshot(const EngineSnapshot& left, const EngineSnapshot& right) {
    CHECK_TRUE(left.money == right.money);
    CHECK_TRUE(left.base_hp == right.base_hp);
    CHECK_TRUE(left.wave == right.wave);
    CHECK_TRUE(left.spawn_timer == right.spawn_timer);
    CHECK_TRUE(left.time == right.time);
    CHECK_TRUE(left.game_over == right.game_over);
    CHECK_TRUE(left.use_budgeted_waves == right.use_budgeted_waves);
    CHECK_TRUE(left.seed == right.seed);
    CHECK_TRUE(left.grid_version == right.grid_version);
    CHECK_TRUE(left.tower_version == right.tower_version);
    CHECK_TRUE(left.money_version == right.money_version);
    CHECK_TRUE(left.enemy_version == right.enemy_version);
    CHECK_TRUE(left.wave_version == right.wave_version);
    CHECK_TRUE(left.grid == right.grid);
    CHECK_TRUE(left.blocked == right.blocked);
    CHECK_TRUE(left.towers.size() == right.towers.size());
    CHECK_TRUE(left.enemies.size() == right.enemies.size());
    CHECK_TRUE(left.pending.size() == right.pending.size());

    for (size_t i = 0; i < left.towers.size(); ++i) {
        CHECK_TRUE(same_tower(left.towers[i], right.towers[i]));
    }
    for (size_t i = 0; i < left.enemies.size(); ++i) {
        CHECK_TRUE(same_enemy(left.enemies[i], right.enemies[i]));
    }
    for (size_t i = 0; i < left.pending.size(); ++i) {
        CHECK_TRUE(left.pending[i].hp == right.pending[i].hp);
        CHECK_TRUE(left.pending[i].speed == right.pending[i].speed);
        CHECK_TRUE(left.pending[i].reward == right.pending[i].reward);
    }
}

void test_terminal_state_is_fully_absorbing() {
    TDEngine env(11, 11, 321);
    CHECK_TRUE(env.place_tower(1, 1, TowerType::Basic));

    bool observed_terminal = false;
    for (int tick = 0; tick < 5000; ++tick) {
        if (env.step_wait(1).done) {
            observed_terminal = true;
            break;
        }
    }
    CHECK_TRUE(observed_terminal);
    CHECK_TRUE(env.game_over());

    const EngineSnapshot terminal = capture(env);

    const auto legal = env.legal_actions();
    CHECK_TRUE(legal.size() == 1);
    CHECK_TRUE(legal.front() == kFlatWaitOffset);
    const auto legal_mask = env.legal_action_mask();
    CHECK_TRUE(legal_mask.size() == kActionSpaceSize);
    CHECK_TRUE(legal_mask[kFlatWaitOffset] == 1u);
    int legal_count = 0;
    for (uint8_t value : legal_mask) legal_count += value != 0u ? 1 : 0;
    CHECK_TRUE(legal_count == 1);

    CHECK_TRUE(!env.can_place_tower(2, 1, TowerType::Basic));
    CHECK_TRUE(!env.place_tower(2, 1, TowerType::Basic));
    CHECK_TRUE(!env.upgrade_tower(1, 1));
    CHECK_TRUE(!env.sell_tower(1, 1));
    check_same_snapshot(terminal, capture(env));

    const int build_action = encode_action(Action{ActionType::BuildBasic, 2, 1, 1});
    const int upgrade_action = encode_action(Action{ActionType::Upgrade, 1, 1, 1});
    const int sell_action = encode_action(Action{ActionType::Sell, 1, 1, 1});
    for (int action : {build_action, upgrade_action, sell_action, kFlatWaitOffset, -1, kActionSpaceSize}) {
        const StepResult result = env.step_action(action);
        CHECK_TRUE(result.done);
        CHECK_TRUE(result.reward == 0.0f);
        check_same_snapshot(terminal, capture(env));
    }

    const StepResult wait_result = env.step_wait(20);
    CHECK_TRUE(wait_result.done);
    CHECK_TRUE(wait_result.reward == 0.0f);
    check_same_snapshot(terminal, capture(env));

    const StepResult time_result = env.step_time(20.0f);
    CHECK_TRUE(time_result.done);
    CHECK_TRUE(time_result.reward == 0.0f);
    check_same_snapshot(terminal, capture(env));

    env.set_use_budgeted_waves(!terminal.use_budgeted_waves, true);
    check_same_snapshot(terminal, capture(env));
}

} // namespace

int main() {
    try {
        test_default_one_second_cooldown_fires_once();
        test_half_second_cooldown_fires_twice();
        test_fractional_cooldown_preserves_phase();
        test_upgraded_tower_has_real_rate_gain();
        test_batched_and_split_cooldown_windows_match();
        test_terminal_tick_advances_engine_time();
        test_terminal_state_is_fully_absorbing();
        std::cout << "Timing semantics tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
