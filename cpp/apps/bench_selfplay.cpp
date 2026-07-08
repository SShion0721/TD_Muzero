#include "tdmz/core/board_tables.hpp"
#include "tdmz/mcts/dummy_network.hpp"
#include "tdmz/selfplay/selfplay_runner.hpp"
#include "tdmz/selfplay/trajectory_writer.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace tdmz;

namespace {

struct BenchResult {
    std::string name;
    int games = 0;
    int steps = 0;
    int simulations = 0;
    int observation_calls = 0;
    int trajectory_records = 0;
    double seconds = 0.0;
    double checksum = 0.0;
    uint64_t bytes_written = 0;
};

uint64_t file_size_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return 0;
    return static_cast<uint64_t>(in.tellg());
}

double checksum_history(const GameHistory& history) {
    double sum = static_cast<double>(history.steps.size()) * 17.0 + static_cast<double>(history.total_reward);
    for (const auto& step : history.steps) {
        sum += step.action * 0.125;
        sum += step.reward * 3.0;
        sum += step.root_value * 5.0;
        sum += step.money * 0.01;
        sum += step.base_hp * 0.02;
        sum += step.wave * 0.5;
        if (!step.policy_target.empty()) sum += step.policy_target[step.action >= 0 ? step.action : 0] * 11.0;
        for (size_t i = 0; i < step.observation.size(); i += 97) {
            sum += step.observation[i] * static_cast<double>((i % 29) + 1);
        }
    }
    return sum;
}

BenchResult run_case(const std::string& name, bool budgeted, bool write_binary, bool write_jsonl, int games, int max_steps, int simulations) {
    DummyNetwork net;
    SelfPlayConfig cfg;
    cfg.max_steps = max_steps;
    cfg.mcts.num_simulations = simulations;
    cfg.mcts.latent_top_k = 16;
    cfg.mcts.max_nodes = 8192;
    cfg.save_observations = true;
    cfg.save_legal_mask = true;

    BenchResult result;
    result.name = name;
    result.games = games;

    auto start = std::chrono::high_resolution_clock::now();
    for (int g = 0; g < games; ++g) {
        cfg.seed = static_cast<uint64_t>(1000 + g);
        SelfPlayRunner runner(cfg);
        TDEngine env(11, 11, cfg.seed, budgeted);
        GameHistory history = runner.run(env, net);

        int steps = static_cast<int>(history.steps.size());
        result.steps += steps;
        result.simulations += steps * simulations;
        result.observation_calls += steps;
        result.trajectory_records += steps;
        result.checksum += checksum_history(history);

        if (write_binary) {
            std::string path = "bench_selfplay_" + name + "_" + std::to_string(g) + ".tdmzspb";
            write_history_binary(history, path);
            result.bytes_written += file_size_bytes(path);
        }
        if (write_jsonl) {
            std::string path = "bench_selfplay_" + name + "_" + std::to_string(g) + ".jsonl";
            write_history_jsonl(history, path);
            result.bytes_written += file_size_bytes(path);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    result.seconds = diff.count();
    return result;
}

void print_result(const BenchResult& r) {
    double games_per_sec = r.games / r.seconds;
    double steps_per_sec = r.steps / r.seconds;
    double sims_per_sec = r.simulations / r.seconds;
    double obs_per_sec = r.observation_calls / r.seconds;
    double records_per_sec = r.trajectory_records / r.seconds;
    double mb_written = static_cast<double>(r.bytes_written) / (1024.0 * 1024.0);
    double mb_per_sec = mb_written / r.seconds;

    std::cout << std::fixed << std::setprecision(6)
              << "{\"case\":\"" << r.name << "\""
              << ",\"games\":" << r.games
              << ",\"steps\":" << r.steps
              << ",\"simulations\":" << r.simulations
              << ",\"seconds\":" << r.seconds
              << ",\"games_per_second\":" << games_per_sec
              << ",\"steps_per_second\":" << steps_per_sec
              << ",\"simulations_per_second\":" << sims_per_sec
              << ",\"observation_calls_per_second\":" << obs_per_sec
              << ",\"trajectory_records_per_second\":" << records_per_sec
              << ",\"mb_written\":" << mb_written
              << ",\"mb_per_second\":" << mb_per_sec
              << ",\"checksum\":" << r.checksum
              << "}" << std::endl;
}

} // namespace

int main() {
    // Measure steady-state self-play throughput, not one-time static table setup.
    (void)board_tables();
    {
        TDEngine warmup(11, 11, 999);
        (void)warmup.legal_actions();
    }

    const int games = 8;
    const int max_steps = 64;
    const int simulations = 32;

    print_result(run_case("fixed_no_write", false, false, false, games, max_steps, simulations));
    print_result(run_case("budgeted_no_write", true, false, false, games, max_steps, simulations));
    print_result(run_case("fixed_binary_write", false, true, false, games, max_steps, simulations));
    print_result(run_case("budgeted_binary_write", true, true, false, games, max_steps, simulations));
    print_result(run_case("fixed_jsonl_write", false, false, true, 2, max_steps, simulations));

    return 0;
}
