#include "tdmz/core/engine.hpp"
#include "tdmz/core/tower.hpp"
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace tdmz;

namespace {

struct EpisodeSummary {
    int seed = 0;
    bool budgeted = false;
    int steps = 0;
    float total_reward = 0.0f;
    int final_wave = 0;
    int base_hp = 0;
    int money = 0;
    bool game_over = false;
    int active_enemy_count = 0;
    int pending_count = 0;
    float active_enemy_hp = 0.0f;
    float pending_enemy_hp = 0.0f;
    int tower_count = 0;
};

struct BuildCandidate {
    int x;
    int y;
    TowerType type;
};

float active_enemy_hp(const TDEngine& env) {
    float total = 0.0f;
    for (const auto& e : env.enemies()) total += e.hp;
    return total;
}

void scripted_economy_action(TDEngine& env) {
    const std::vector<BuildCandidate> plan = {
        {1, 4, TowerType::Basic},
        {1, 6, TowerType::Basic},
        {5, 4, TowerType::Sniper},
        {5, 6, TowerType::Sniper},
        {3, 4, TowerType::AOE},
        {3, 6, TowerType::Slow},
        {7, 4, TowerType::Sniper},
        {7, 6, TowerType::Sniper},
        {2, 3, TowerType::Basic},
        {2, 7, TowerType::Basic},
    };

    for (const auto& c : plan) {
        if (env.can_place_tower(c.x, c.y, c.type)) {
            env.place_tower(c.x, c.y, c.type);
            return;
        }
    }

    for (const auto& t : env.towers()) {
        if (t.can_upgrade() && env.money() >= t.upgrade_cost) {
            env.upgrade_tower(t.x, t.y);
            return;
        }
    }
}

EpisodeSummary run_episode(int seed, bool budgeted, int max_steps) {
    TDEngine env(11, 11, static_cast<uint64_t>(seed), budgeted);
    EpisodeSummary s;
    s.seed = seed;
    s.budgeted = budgeted;

    for (int step = 0; step < max_steps && !env.game_over(); ++step) {
        scripted_economy_action(env);
        auto r = env.step_wait(1);
        s.total_reward += r.reward;
        s.steps = step + 1;
        if (r.done) break;
    }

    s.final_wave = env.wave();
    s.base_hp = env.base_hp();
    s.money = env.money();
    s.game_over = env.game_over();
    s.active_enemy_count = static_cast<int>(env.enemies().size());
    s.pending_count = env.enemies_to_spawn_count();
    s.active_enemy_hp = active_enemy_hp(env);
    s.pending_enemy_hp = env.pending_spawn_total_hp();
    s.tower_count = static_cast<int>(env.towers().size());
    return s;
}

void print_summary(const EpisodeSummary& s) {
    std::cout << "{"
              << "\"seed\":" << s.seed << ","
              << "\"mode\":\"" << (s.budgeted ? "budgeted" : "fixed") << "\","
              << "\"steps\":" << s.steps << ","
              << "\"total_reward\":" << std::fixed << std::setprecision(3) << s.total_reward << ","
              << "\"final_wave\":" << s.final_wave << ","
              << "\"base_hp\":" << s.base_hp << ","
              << "\"money\":" << s.money << ","
              << "\"game_over\":" << (s.game_over ? "true" : "false") << ","
              << "\"tower_count\":" << s.tower_count << ","
              << "\"active_enemy_count\":" << s.active_enemy_count << ","
              << "\"pending_count\":" << s.pending_count << ","
              << "\"active_enemy_hp\":" << s.active_enemy_hp << ","
              << "\"pending_enemy_hp\":" << s.pending_enemy_hp
              << "}" << std::endl;
}

} // namespace

int main() {
    constexpr int max_steps = 300;
    const std::vector<int> seeds = {0, 1, 2, 3, 4};
    for (int seed : seeds) {
        print_summary(run_episode(seed, false, max_steps));
        print_summary(run_episode(seed, true, max_steps));
    }
    return 0;
}
