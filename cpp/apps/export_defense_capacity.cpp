#include "tdmz/balance/defense_capacity.hpp"
#include <iomanip>
#include <iostream>
#include <string>

using namespace tdmz;

static const char* tower_type_name(TowerType t) {
    switch (t) {
        case TowerType::Basic: return "Basic";
        case TowerType::Sniper: return "Sniper";
        case TowerType::AOE: return "AOE";
        case TowerType::Slow: return "Slow";
    }
    return "Unknown";
}

static void print_result(const char* name, const DefenseCapacityResult& r) {
    std::cout << "{"
              << "\"case\":\"" << name << "\","
              << "\"current_tower_damage_cap\":" << std::fixed << std::setprecision(3) << r.current_tower_damage_cap << ","
              << "\"static_current_tower_damage_cap\":" << r.static_current_tower_damage_cap << ","
              << "\"spendable_money_damage_cap\":" << r.spendable_money_damage_cap << ","
              << "\"selected_build_cap\":" << r.selected_build_cap << ","
              << "\"selected_upgrade_cap\":" << r.selected_upgrade_cap << ","
              << "\"leak_hp_cap\":" << r.leak_hp_cap << ","
              << "\"raw_hp_cap\":" << r.raw_hp_cap << ","
              << "\"allowed_attack_hp\":" << r.allowed_attack_hp << ","
              << "\"path_len\":" << r.path_len << ","
              << "\"placeable_count\":" << r.placeable_count << ","
              << "\"build_candidate_count\":" << r.build_candidate_count << ","
              << "\"upgrade_candidate_count\":" << r.upgrade_candidate_count << ","
              << "\"selected_new_tower_count\":" << r.selected_new_tower_count << ","
              << "\"selected_upgrade_count\":" << r.selected_upgrade_count << ","
              << "\"best_build_value_per_cost\":" << r.best_build_value_per_cost << ","
              << "\"best_upgrade_value_per_cost\":" << r.best_upgrade_value_per_cost << ","
              << "\"selected_new_tower_type\":\"" << tower_type_name(r.selected_new_tower_type) << "\""
              << "}" << std::endl;
}

int main() {
    DefenseCapacityConfig cfg;

    TDEngine empty(11, 11, 0);
    print_result("empty_seed0", estimate_defense_capacity(empty, cfg));

    TDEngine off_path(11, 11, 0);
    off_path.place_tower(1, 1, TowerType::Basic);
    print_result("off_path_basic_seed0", estimate_defense_capacity(off_path, cfg));

    TDEngine basic(11, 11, 0);
    basic.place_tower(1, 4, TowerType::Basic);
    print_result("one_path_basic_seed0", estimate_defense_capacity(basic, cfg));

    TDEngine mixed(11, 11, 0);
    mixed.place_tower(1, 4, TowerType::Basic);
    mixed.place_tower(5, 4, TowerType::Sniper);
    print_result("path_basic_sniper_seed0", estimate_defense_capacity(mixed, cfg));

    DefenseCapacityConfig static_cfg;
    static_cfg.path_aware = false;
    print_result("static_path_basic_sniper_seed0", estimate_defense_capacity(mixed, static_cfg));

    return 0;
}
