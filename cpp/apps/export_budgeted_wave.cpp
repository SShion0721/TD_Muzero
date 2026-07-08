#include "tdmz/balance/attack_budget.hpp"
#include "tdmz/balance/budgeted_wave_generator.hpp"
#include <iomanip>
#include <iostream>

using namespace tdmz;

static void print_result(const char* name, const BudgetedWaveResult& r) {
    std::cout << "{"
              << "\"case\":\"" << name << "\","
              << "\"wave\":" << r.wave << ","
              << "\"requested_budget_hp\":" << std::fixed << std::setprecision(3) << r.requested_budget_hp << ","
              << "\"used_budget_hp\":" << r.used_budget_hp << ","
              << "\"unused_budget_hp\":" << r.unused_budget_hp << ","
              << "\"enemy_count\":" << r.enemies.size() << ","
              << "\"regular_count\":" << r.regular_count << ","
              << "\"fast_count\":" << r.fast_count << ","
              << "\"tank_count\":" << r.tank_count << ","
              << "\"boss_count\":" << r.boss_count << ","
              << "\"regular_hp\":" << r.regular_hp << ","
              << "\"fast_hp\":" << r.fast_hp << ","
              << "\"tank_hp\":" << r.tank_hp << ","
              << "\"boss_hp\":" << r.boss_hp
              << "}" << std::endl;
}

static AttackBudgetResult synthetic_budget(int wave, float hp) {
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

int main() {
    AttackBudgetConfig cfg;
    TDEngine empty(11, 11, 0);
    auto budget_empty = estimate_attack_budget(empty, cfg);
    print_result("budget_empty_seed0", generate_budgeted_wave(budget_empty));

    TDEngine defended(11, 11, 0);
    defended.place_tower(1, 4, TowerType::Basic);
    defended.place_tower(5, 4, TowerType::Sniper);
    auto budget_defended = estimate_attack_budget(defended, cfg);
    print_result("budget_path_basic_sniper_seed0", generate_budgeted_wave(budget_defended));

    print_result("synthetic_wave3_budget2500", generate_budgeted_wave(synthetic_budget(3, 2500.0f)));
    print_result("synthetic_wave5_budget30000", generate_budgeted_wave(synthetic_budget(5, 30000.0f)));
    return 0;
}
