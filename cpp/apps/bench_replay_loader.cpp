#include "tdmz/selfplay/replay_dataset.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace tdmz;

namespace {

struct BenchResult {
    std::string name;
    int shards = 0;
    int games = 0;
    int steps = 0;
    uint64_t bytes = 0;
    double seconds = 0.0;
    double checksum = 0.0;
};

struct XorShift64 {
    uint64_t state;
    explicit XorShift64(uint64_t seed)
        : state(seed ? seed : 0x9e3779b97f4a7c15ULL) {}

    uint64_t next() {
        uint64_t value = state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        state = value;
        return value;
    }
};

uint64_t file_size_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return 0;
    return static_cast<uint64_t>(in.tellg());
}

uint64_t logical_history_payload_bytes(const GameHistory& history) {
    uint64_t bytes = 0;
    for (const auto& step : history.steps) {
        bytes += 36;
        bytes += static_cast<uint64_t>(step.observation.size()) * sizeof(float);
        bytes += static_cast<uint64_t>(step.policy_target.size()) * sizeof(float);
        bytes += static_cast<uint64_t>(step.legal_mask.size()) * sizeof(uint8_t);
    }
    return bytes;
}

double checksum_history(const GameHistory& history) {
    double sum = static_cast<double>(history.steps.size()) * 17.0
        + static_cast<double>(history.total_reward);
    for (const auto& step : history.steps) {
        sum += step.action * 0.125;
        sum += step.reward * 3.0;
        sum += step.root_value * 5.0;
        sum += step.money * 0.01;
        sum += step.base_hp * 0.02;
        sum += step.wave * 0.5;
        if (!step.policy_target.empty()) {
            const int action = std::max(
                0,
                std::min(step.action, static_cast<int>(step.policy_target.size()) - 1));
            sum += step.policy_target[static_cast<size_t>(action)] * 11.0;
        }
        for (size_t i = 0; i < step.observation.size(); i += 97) {
            sum += step.observation[i] * static_cast<double>((i % 29) + 1);
        }
    }
    return sum;
}

BenchResult bench_sequential_scan(ReplayDataset& dataset) {
    BenchResult result;
    result.name = "sequential_scan";
    result.shards = static_cast<int>(dataset.shard_count());
    for (const auto& path : dataset.shard_paths()) result.bytes += file_size_bytes(path);

    const auto start = std::chrono::high_resolution_clock::now();
    for (size_t game = 0; game < dataset.game_count(); ++game) {
        GameHistory history = dataset.read_game(game);
        ++result.games;
        result.steps += static_cast<int>(history.steps.size());
        result.checksum += checksum_history(history);
    }
    result.seconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start).count();
    return result;
}

BenchResult bench_random_read(ReplayDataset& dataset, int samples, uint64_t seed) {
    XorShift64 random(seed);
    BenchResult result;
    result.name = "random_game_read";
    result.shards = static_cast<int>(dataset.shard_count());

    const auto start = std::chrono::high_resolution_clock::now();
    for (int sample = 0; sample < samples; ++sample) {
        const size_t game = static_cast<size_t>(random.next() % dataset.game_count());
        GameHistory history = dataset.read_game(game);
        ++result.games;
        result.steps += static_cast<int>(history.steps.size());
        result.bytes += logical_history_payload_bytes(history);
        result.checksum += checksum_history(history);
    }
    result.seconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start).count();
    return result;
}

void print_result(const BenchResult& result) {
    const double megabytes = static_cast<double>(result.bytes) / (1024.0 * 1024.0);
    std::cout << std::fixed << std::setprecision(6)
              << "{\"case\":\"" << result.name << "\""
              << ",\"shards\":" << result.shards
              << ",\"games\":" << result.games
              << ",\"steps\":" << result.steps
              << ",\"seconds\":" << result.seconds
              << ",\"games_per_second\":"
              << (result.seconds > 0.0 ? result.games / result.seconds : 0.0)
              << ",\"steps_per_second\":"
              << (result.seconds > 0.0 ? result.steps / result.seconds : 0.0)
              << ",\"logical_mb_read\":" << megabytes
              << ",\"logical_mb_per_second\":"
              << (result.seconds > 0.0 ? megabytes / result.seconds : 0.0)
              << ",\"checksum\":" << result.checksum
              << "}" << std::endl;
}

void print_help(const char* executable) {
    std::cout
        << "Usage: " << executable
        << " --index data/selfplay/train.index.json [options]\n\n"
        << "Options:\n"
        << "  --index PATH           Shard index JSON from generate_selfplay_shards.\n"
        << "  --random-samples N    Number of random whole-game reads. Default: 1024\n"
        << "  --seed N              Random-read seed. Default: 1234567\n"
        << "  --allow-legacy-index  Explicitly allow a reviewed v1/unversioned index.\n"
        << "  --help                Show this help.\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string index_path = "data/selfplay/train.index.json";
        int random_samples = 1024;
        uint64_t seed = 1234567;
        bool allow_legacy_index = false;

        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--index") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --index");
                index_path = argv[++i];
            } else if (argument == "--random-samples") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --random-samples");
                random_samples = std::stoi(argv[++i]);
            } else if (argument == "--seed") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --seed");
                seed = static_cast<uint64_t>(std::stoull(argv[++i]));
            } else if (argument == "--allow-legacy-index") {
                allow_legacy_index = true;
            } else if (argument == "--help" || argument == "-h") {
                print_help(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("Unknown argument: " + argument);
            }
        }
        if (random_samples <= 0) throw std::runtime_error("--random-samples must be positive");

        const LegacyReplayIndexPolicy legacy_policy = allow_legacy_index
            ? LegacyReplayIndexPolicy::AllowV1
            : LegacyReplayIndexPolicy::Reject;
        const auto open_start = std::chrono::high_resolution_clock::now();
        ReplayDataset dataset = ReplayDataset::from_index_json(index_path, legacy_policy);
        const double open_seconds = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - open_start).count();

        std::cout << std::fixed << std::setprecision(6)
                  << "{\"case\":\"open_dataset\""
                  << ",\"shards\":" << dataset.shard_count()
                  << ",\"games\":" << dataset.game_count()
                  << ",\"index_version\":" << dataset.index_metadata().index_version
                  << ",\"replay_format_version\":" << dataset.replay_format_version()
                  << ",\"transition_indexed\":"
                  << (dataset.transition_indexed() ? "true" : "false")
                  << ",\"legacy_index\":"
                  << (dataset.index_metadata().legacy_v1 ? "true" : "false")
                  << ",\"seconds\":" << open_seconds
                  << "}" << std::endl;

        dataset.reset_game_read_count();
        print_result(bench_sequential_scan(dataset));
        print_result(bench_random_read(dataset, random_samples, seed));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
