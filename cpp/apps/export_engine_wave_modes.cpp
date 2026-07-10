#include "tdmz/balance/attack_budget.hpp"
#include "tdmz/core/engine.hpp"
#include <iomanip>
#include <iostream>

using namespace tdmz;

static void print_engine_wave(const char* name, const TDEngine& env) {
    AttackBudgetConfig cfg;
    auto budget = estimate_attack_budget(env, cfg);
    std::cout << "{"
              << "\"case\":\"" << name << "\","
              << "\"use_budgeted_waves\":" << (env.use_budgeted_waves() ? "true" : "false") << ","
              << "\"wave\":" << env.wave() << ","
              << "\"pending_count\":" << env.enemies_to_spawn_count() << ","
              << "\"pending_total_hp\":" << std::fixed << std::setprecision(3) << env.pending_spawn_total_hp() << ","
              << "\"attack_budget_allowed_hp\":" << budget.allowed_attack_hp << ","
              << "\"wave_cap_hp\":" << budget.wave_cap_hp << ","
              << "\"defense_allowed_hp\":" << budget.defense_allowed_attack_hp
              << "}" << std::endl;
}

int main() {
    TDEngine fixed(11, 11, 0, false);
    print_engine_wave("fixed_wave_seed0", fixed);

    TDEngine budgeted(11, 11, 0, true);
    print_engine_wave("budgeted_wave_seed0", budgeted);

    TDEngine toggled(11, 11, 0, false);
    toggled.set_use_budgeted_waves(true, true);
    print_engine_wave("toggled_budgeted_wave_seed0", toggled);

    toggled.set_use_budgeted_waves(false, true);
    print_engine_wave("toggled_fixed_wave_seed0", toggled);
    return 0;
}
