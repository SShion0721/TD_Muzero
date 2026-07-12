#include "tdmz/balance/attack_budget.hpp"
#include "tdmz/core/engine.hpp"
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

static void check_close(float a, float b, float eps, const char* expr, int line) {
    if (std::fabs(a - b) > eps) throw std::runtime_error(std::string("check failed at line ") + std::to_string(line) + ": " + expr);
}
#define CHECK_CLOSE(a, b, eps) check_close((a), (b), (eps), #a " ~= " #b, __LINE__)

template <typename Exception, typename Fn>
void check_throws(Fn&& function) {
    bool threw = false;
    try {
        function();
    } catch (const Exception&) {
        threw = true;
    }
    CHECK_TRUE(threw);
}

void test_default_fixed_wave_mode() {
    TDEngine env(11, 11, 0);
    CHECK_TRUE(!env.use_budgeted_waves());
    CHECK_TRUE(env.wave_mode() == WaveMode::Fixed);
    CHECK_TRUE(!env.episode_started());
    CHECK_TRUE(env.enemies_to_spawn_count() == 9);
    CHECK_CLOSE(env.pending_spawn_total_hp(), 281.2f, 1e-3f);
}

void test_constructor_budgeted_wave_mode() {
    TDEngine env(11, 11, 0, true);
    CHECK_TRUE(env.use_budgeted_waves());
    CHECK_TRUE(env.wave_mode() == WaveMode::Budgeted);
    CHECK_TRUE(env.enemies_to_spawn_count() > 9);
    CHECK_TRUE(env.pending_spawn_total_hp() > 500.0f);

    AttackBudgetConfig cfg;
    auto budget = estimate_attack_budget(env, cfg);
    CHECK_TRUE(env.pending_spawn_total_hp() <= budget.allowed_attack_hp + 1e-3f);
}

void test_runtime_regenerate_wave_mode_before_start() {
    TDEngine env(11, 11, 0);
    float fixed_hp = env.pending_spawn_total_hp();
    int fixed_count = env.enemies_to_spawn_count();

    check_throws<std::invalid_argument>([&] {
        env.set_use_budgeted_waves(true, false);
    });
    CHECK_TRUE(env.wave_mode() == WaveMode::Fixed);

    env.set_use_budgeted_waves(true, true);
    CHECK_TRUE(env.wave_mode() == WaveMode::Budgeted);
    CHECK_TRUE(!env.episode_started());
    CHECK_TRUE(env.enemies_to_spawn_count() != fixed_count);
    CHECK_TRUE(env.pending_spawn_total_hp() > fixed_hp);

    env.set_use_budgeted_waves(false, true);
    CHECK_TRUE(env.wave_mode() == WaveMode::Fixed);
    CHECK_TRUE(env.enemies_to_spawn_count() == fixed_count);
    CHECK_CLOSE(env.pending_spawn_total_hp(), fixed_hp, 1e-3f);
}

void test_mode_change_rejected_after_time_starts() {
    TDEngine env(11, 11, 0, false);
    CHECK_TRUE(!env.episode_started());
    (void)env.step_wait(1);
    CHECK_TRUE(env.episode_started());

    check_throws<std::logic_error>([&] {
        env.set_use_budgeted_waves(true, true);
    });
    check_throws<std::logic_error>([&] {
        env.set_use_budgeted_waves(false, true);
    });
    CHECK_TRUE(env.wave_mode() == WaveMode::Fixed);
}

void test_mode_change_rejected_after_direct_mutation() {
    TDEngine env(11, 11, 0, false);
    CHECK_TRUE(env.place_tower(1, 4, TowerType::Basic));
    CHECK_TRUE(env.episode_started());

    check_throws<std::logic_error>([&] {
        env.set_use_budgeted_waves(true, true);
    });
    CHECK_TRUE(env.wave_mode() == WaveMode::Fixed);
}

void test_reset_reopens_mode_selection() {
    TDEngine env(11, 11, 0, false);
    (void)env.step_wait(1);
    CHECK_TRUE(env.episode_started());

    env.reset(7);
    CHECK_TRUE(!env.episode_started());
    CHECK_TRUE(env.wave_mode() == WaveMode::Fixed);
    env.set_use_budgeted_waves(true, true);
    CHECK_TRUE(env.wave_mode() == WaveMode::Budgeted);
}

void test_fixed_and_budgeted_modes_diverge() {
    TDEngine fixed(11, 11, 0, false);
    TDEngine budgeted(11, 11, 0, true);
    CHECK_TRUE(fixed.enemies_to_spawn_count() != budgeted.enemies_to_spawn_count());
    CHECK_TRUE(fixed.pending_spawn_total_hp() < budgeted.pending_spawn_total_hp());
}

int main() {
    try {
        test_default_fixed_wave_mode();
        test_constructor_budgeted_wave_mode();
        test_runtime_regenerate_wave_mode_before_start();
        test_mode_change_rejected_after_time_starts();
        test_mode_change_rejected_after_direct_mutation();
        test_reset_reopens_mode_selection();
        test_fixed_and_budgeted_modes_diverge();
        std::cout << "Engine wave mode tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
