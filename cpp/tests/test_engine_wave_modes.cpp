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

void test_default_fixed_wave_mode() {
    TDEngine env(11, 11, 0);
    CHECK_TRUE(!env.use_budgeted_waves());
    CHECK_TRUE(env.enemies_to_spawn_count() == 9);
    CHECK_CLOSE(env.pending_spawn_total_hp(), 281.2f, 1e-3f);
}

void test_constructor_budgeted_wave_mode() {
    TDEngine env(11, 11, 0, true);
    CHECK_TRUE(env.use_budgeted_waves());
    CHECK_TRUE(env.enemies_to_spawn_count() > 9);
    CHECK_TRUE(env.pending_spawn_total_hp() > 500.0f);

    AttackBudgetConfig cfg;
    auto budget = estimate_attack_budget(env, cfg);
    CHECK_TRUE(env.pending_spawn_total_hp() <= budget.allowed_attack_hp + 1e-3f);
}

void test_runtime_regenerate_wave_mode() {
    TDEngine env(11, 11, 0);
    float fixed_hp = env.pending_spawn_total_hp();
    int fixed_count = env.enemies_to_spawn_count();

    env.set_use_budgeted_waves(true, true);
    CHECK_TRUE(env.use_budgeted_waves());
    CHECK_TRUE(env.enemies_to_spawn_count() != fixed_count);
    CHECK_TRUE(env.pending_spawn_total_hp() > fixed_hp);

    env.set_use_budgeted_waves(false, true);
    CHECK_TRUE(!env.use_budgeted_waves());
    CHECK_TRUE(env.enemies_to_spawn_count() == fixed_count);
    CHECK_CLOSE(env.pending_spawn_total_hp(), fixed_hp, 1e-3f);
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
        test_runtime_regenerate_wave_mode();
        test_fixed_and_budgeted_modes_diverge();
        std::cout << "Engine wave mode tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
