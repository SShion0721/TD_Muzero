#include "tdmz/balance/attack_budget.hpp"
#include <iomanip>
#include <iostream>

using namespace tdmz;

static void print_result(const char* name, const AttackBudgetResult& r) {
    std::cout << "{"
              << "\"case\":\"" << name << "\","
              << "\"wave\":" << r.wave << ","
              << "\"defense_allowed_attack_hp\":" << std::fixed << std::setprecision(3) << r.defense_allowed_attack_hp << ","
              << "\"wave_cap_hp\":" << r.wave_cap_hp << ","
              << "\"allowed_attack_hp\":" << r.allowed_attack_hp << ","
              << "\"wave_cap_applied\":" << (r.wave_cap_applied ? "true" : "false") << ","
              << "\"regular_hp_cap\":" << r.regular_hp_cap << ","
              << "\"fast_hp_cap\":" << r.fast_hp_cap << ","
              << "\"tank_hp_cap\":" << r.tank_hp_cap << ","
              << "\"boss_hp_cap\":" << r.boss_hp_cap << ","
              << "\"tank_unlocked\":" << (r.tank_unlocked ? "true" : "false") << ","
              << "\"boss_unlocked\":" << (r.boss_unlocked ? "true" : "false") << ","
              << "\"defense_current_cap\":" << r.defense.current_tower_damage_cap << ","
              << "\"defense_spendable_cap\":" << r.defense.spendable_money_damage_cap << ","
              << "\"defense_leak_cap\":" << r.defense.leak_hp_cap
              << "}" << std::endl;
}

int main() {
    AttackBudgetConfig cfg;

    TDEngine empty(11, 11, 0);
    print_result("empty_seed0", estimate_attack_budget(empty, cfg));

    TDEngine basic(11, 11, 0);
    basic.place_tower(1, 4, TowerType::Basic);
    print_result("path_basic_seed0", estimate_attack_budget(basic, cfg));

    TDEngine mixed(11, 11, 0);
    mixed.place_tower(1, 4, TowerType::Basic);
    mixed.place_tower(5, 4, TowerType::Sniper);
    print_result("path_basic_sniper_seed0", estimate_attack_budget(mixed, cfg));

    AttackBudgetConfig unclamped = cfg;
    unclamped.enable_wave_cap = false;
    print_result("unclamped_path_basic_sniper_seed0", estimate_attack_budget(mixed, unclamped));

    AttackBudgetConfig low_base = cfg;
    low_base.enable_wave_cap = false;
    low_base.defense.virtual_base_hp = 20;
    print_result("low_virtual_base_seed0", estimate_attack_budget(empty, low_base));

    return 0;
}
