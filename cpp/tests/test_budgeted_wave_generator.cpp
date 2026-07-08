#include "tdmz/balance/budgeted_wave_generator.hpp"
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

static AttackBudgetResult make_budget(int wave, float hp) {
    AttackBudgetResult b;
    b.wave = wave;
    b.allowed_attack_hp = hp;
    b.regular_hp_cap = hp;
    b.fast_hp_cap = hp * 0.35f;
    b.tank_hp_cap = hp * 0.45f;
    b.boss_hp_cap = hp * 0.25f;
    b.tank_unlocked = wave >= 3;
    b.boss_unlocked = wave >= 5 && (wave % 2 == 1);
    return b;
}

static float total_enemy_hp(const BudgetedWaveResult& r) {
    float sum = 0.0f;
    for (const auto& e : r.enemies) sum += e.hp;
    return sum;
}

void test_budget_not_exceeded() {
    auto b = make_budget(1, 515.0f);
    auto r = generate_budgeted_wave(b);
    CHECK_TRUE(r.used_budget_hp <= b.allowed_attack_hp + 1e-4f);
    CHECK_CLOSE(r.used_budget_hp, total_enemy_hp(r), 1e-4f);
    CHECK_CLOSE(r.unused_budget_hp, b.allowed_attack_hp - r.used_budget_hp, 1e-4f);
    CHECK_TRUE(static_cast<int>(r.enemies.size()) == r.regular_count + r.fast_count + r.tank_count + r.boss_count);
}

void test_wave_one_locks_tank_boss() {
    auto b = make_budget(1, 1000.0f);
    auto r = generate_budgeted_wave(b);
    CHECK_TRUE(r.tank_count == 0);
    CHECK_TRUE(r.boss_count == 0);
    CHECK_TRUE(r.regular_count > 0);
    CHECK_TRUE(r.fast_count > 0);
}

void test_wave_three_allows_tank() {
    auto b = make_budget(3, 2500.0f);
    auto r = generate_budgeted_wave(b);
    CHECK_TRUE(r.tank_count > 0);
    CHECK_TRUE(r.boss_count == 0);
    CHECK_TRUE(r.tank_hp <= b.tank_hp_cap + 1e-4f);
}

void test_wave_five_allows_boss_when_budget_can_afford_one() {
    auto b = make_budget(5, 30000.0f);
    auto r = generate_budgeted_wave(b);
    CHECK_TRUE(r.boss_count > 0);
    CHECK_TRUE(r.tank_count > 0);
    CHECK_TRUE(r.regular_count > 0);
    CHECK_TRUE(r.used_budget_hp > 10000.0f);
    CHECK_TRUE(r.boss_hp <= b.boss_hp_cap + 1e-4f);
    CHECK_TRUE(r.tank_hp <= b.tank_hp_cap + 1e-4f);
}

void test_ratio_caps_are_respected() {
    auto b = make_budget(5, 10000.0f);
    b.fast_hp_cap = 1000.0f;
    b.tank_hp_cap = 1200.0f;
    b.boss_hp_cap = 1500.0f;
    auto r = generate_budgeted_wave(b);
    CHECK_TRUE(r.fast_hp <= b.fast_hp_cap + 1e-4f);
    CHECK_TRUE(r.tank_hp <= b.tank_hp_cap + 1e-4f);
    CHECK_TRUE(r.boss_hp <= b.boss_hp_cap + 1e-4f);
}

void test_monotonic_budget() {
    auto small = generate_budgeted_wave(make_budget(3, 1000.0f));
    auto large = generate_budgeted_wave(make_budget(3, 3000.0f));
    CHECK_TRUE(large.used_budget_hp >= small.used_budget_hp);
    CHECK_TRUE(large.enemies.size() >= small.enemies.size());
}

void test_deterministic() {
    auto b = make_budget(5, 8000.0f);
    auto a = generate_budgeted_wave(b);
    auto c = generate_budgeted_wave(b);
    CHECK_TRUE(a.enemies.size() == c.enemies.size());
    CHECK_CLOSE(a.used_budget_hp, c.used_budget_hp, 1e-5f);
    CHECK_TRUE(a.regular_count == c.regular_count);
    CHECK_TRUE(a.fast_count == c.fast_count);
    CHECK_TRUE(a.tank_count == c.tank_count);
    CHECK_TRUE(a.boss_count == c.boss_count);
}

void test_max_enemy_count() {
    auto b = make_budget(1, 100000.0f);
    BudgetedWaveConfig cfg;
    cfg.max_enemy_count = 7;
    auto r = generate_budgeted_wave(b, cfg);
    CHECK_TRUE(r.enemies.size() <= 7);
}

int main() {
    try {
        test_budget_not_exceeded();
        test_wave_one_locks_tank_boss();
        test_wave_three_allows_tank();
        test_wave_five_allows_boss_when_budget_can_afford_one();
        test_ratio_caps_are_respected();
        test_monotonic_budget();
        test_deterministic();
        test_max_enemy_count();
        std::cout << "Budgeted wave generator tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
