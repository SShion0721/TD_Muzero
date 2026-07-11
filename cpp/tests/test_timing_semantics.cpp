#include "tdmz/core/engine.hpp"
#include "tdmz/core/tower.hpp"
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

} // namespace

int main() {
    try {
        test_default_one_second_cooldown_fires_once();
        test_half_second_cooldown_fires_twice();
        test_fractional_cooldown_preserves_phase();
        test_upgraded_tower_has_real_rate_gain();
        test_batched_and_split_cooldown_windows_match();
        test_terminal_tick_advances_engine_time();
        std::cout << "Timing semantics tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
