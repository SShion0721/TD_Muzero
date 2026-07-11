#include "tdmz/core/board_tables.hpp"
#include "tdmz/core/engine.hpp"
#include "tdmz/core/observation.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

using namespace tdmz;

namespace {

struct BenchResult {
    std::string name;
    int iterations = 0;
    double seconds = 0.0;
    double checksum = 0.0;
};

void seed_scripted_towers(TDEngine& env) {
    env.place_tower(1, 4, TowerType::Basic);
    env.place_tower(1, 6, TowerType::Basic);
    env.place_tower(5, 4, TowerType::Sniper);
    env.place_tower(3, 6, TowerType::Slow);
}

void maybe_scripted_build(TDEngine& env) {
    struct Plan { int x; int y; TowerType type; };
    static const Plan plans[] = {
        {1, 4, TowerType::Basic},
        {1, 6, TowerType::Basic},
        {5, 4, TowerType::Sniper},
        {5, 6, TowerType::Sniper},
        {3, 4, TowerType::AOE},
        {3, 6, TowerType::Slow},
        {7, 4, TowerType::Sniper},
        {7, 6, TowerType::Sniper},
    };

    for (const auto& p : plans) {
        if (env.can_place_tower(p.x, p.y, p.type)) {
            env.place_tower(p.x, p.y, p.type);
            return;
        }
    }

    for (const auto& tower : env.towers()) {
        if (tower.can_upgrade() && env.money() >= tower.upgrade_cost) {
            env.upgrade_tower(tower.x, tower.y);
            return;
        }
    }
}

inline double sample_observation(const Observation& obs) {
    double sum = 0.0;
    for (size_t i = 0; i < obs.size(); i += 17) {
        sum += static_cast<double>(obs[i]) * static_cast<double>((i % 131) + 1);
    }
    return sum;
}

BenchResult bench_static_cached(const std::string& name, bool budgeted, int iterations, bool reuse_buffer) {
    TDEngine env(11, 11, 7, budgeted);
    seed_scripted_towers(env);
    env.step_wait(12);

    Observation buffer;
    if (reuse_buffer) {
        make_observation_v2_into(env, buffer);
    } else {
        buffer = make_observation_v2(env);
    }
    volatile double warm = sample_observation(buffer);
    (void)warm;

    double checksum = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (reuse_buffer) {
            make_observation_v2_into(env, buffer);
            checksum += sample_observation(buffer);
        } else {
            Observation obs = make_observation_v2(env);
            checksum += sample_observation(obs);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return {name, iterations, diff.count(), checksum};
}

BenchResult bench_dynamic_steps(const std::string& name, bool budgeted, int iterations, bool reuse_buffer) {
    TDEngine env(11, 11, 11, budgeted);
    Observation buffer;
    if (reuse_buffer) make_observation_v2_into(env, buffer);
    else (void)make_observation_v2(env);

    double checksum = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (env.game_over()) {
            env.reset(static_cast<uint64_t>(11 + i));
        }
        maybe_scripted_build(env);
        env.step_wait(1);
        if (reuse_buffer) {
            make_observation_v2_into(env, buffer);
            checksum += sample_observation(buffer);
        } else {
            Observation obs = make_observation_v2(env);
            checksum += sample_observation(obs);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return {name, iterations, diff.count(), checksum};
}

void print_result(const BenchResult& r) {
    double obs_per_sec = r.iterations / r.seconds;
    std::cout << std::fixed << std::setprecision(6)
              << "{\"case\":\"" << r.name << "\""
              << ",\"iterations\":" << r.iterations
              << ",\"seconds\":" << r.seconds
              << ",\"observations_per_second\":" << obs_per_sec
              << ",\"checksum\":" << r.checksum
              << "}" << std::endl;
}

} // namespace

int main() {
    (void)board_tables();
    {
        TDEngine warmup(11, 11, 999);
        (void)warmup.legal_actions();
        Observation buffer;
        make_observation_v2_into(warmup, buffer);
    }

    print_result(bench_static_cached("fixed_static_v2_allocating", false, 50000, false));
    print_result(bench_static_cached("fixed_static_v2_reused", false, 50000, true));
    print_result(bench_static_cached("budgeted_static_v2_allocating", true, 50000, false));
    print_result(bench_static_cached("budgeted_static_v2_reused", true, 50000, true));
    print_result(bench_dynamic_steps("fixed_dynamic_v2_allocating", false, 10000, false));
    print_result(bench_dynamic_steps("fixed_dynamic_v2_reused", false, 10000, true));
    print_result(bench_dynamic_steps("budgeted_dynamic_v2_allocating", true, 10000, false));
    print_result(bench_dynamic_steps("budgeted_dynamic_v2_reused", true, 10000, true));

    return 0;
}
