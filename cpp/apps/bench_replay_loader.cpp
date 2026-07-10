#include "tdmz/selfplay/replay_dataset.hpp"
#include "tdmz/selfplay/trajectory_writer.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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
        uint64_t x = state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state = x;
        return x;
    }
};

uint64_t file_size_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return 0;
    return static_cast<uint64_t>(in.tellg());
}

uint64_t history_payload_bytes(const GameHistory& history) {
    uint64_t bytes = 48; // BinaryHistoryHeader
    for (const auto& step : history.steps) {
        bytes += 36; // BinaryStepHeader
        bytes += static_cast<uint64_t>(step.observation.size()) * sizeof(float);
        bytes += static_cast<uint64_t>(step.policy_target.size()) * sizeof(float);
        bytes += static_cast<uint64_t>(step.legal_mask.size()) * sizeof(uint8_t);
    }
    return bytes;
}

double checksum_history(const GameHistory& history) {
    double sum = static_cast<double>(history.steps.size()) * 17.0 +
                 static_cast<double>(history.total_reward);
    for (const auto& step : history.steps) {
        sum += step.action * 0.125;
        sum += step.reward * 3.0;
        sum += step.root_value * 5.0;
        sum += step.money * 0.01;
        sum += step.base_hp * 0.02;
        sum += step.wave * 0.5;
        if (!step.policy_target.empty()) {
            int action = std::max(
                0,
                std::min(
                    step.action,
                    static_cast<int>(step.policy_target.size()) - 1));
            sum += step.policy_target[static_cast<size_t>(action)] * 11.0;
        }
        for (size_t i = 0; i < step.observation.size(); i += 97) {
            sum += step.observation[i] * static_cast<double>((i % 29) + 1);
        }
    }
    return sum;
}

BenchResult bench_open_index(const ReplayIndexInfo& index) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::unique_ptr<BinaryShardReader>> readers;
    readers.reserve(index.shard_paths.size());
    int games = 0;
    for (const auto& path : index.shard_paths) {
        auto reader = std::make_unique<BinaryShardReader>(path);
        games += static_cast<int>(reader->history_count());
        readers.push_back(std::move(reader));
    }
    auto end = std::chrono::high_resolution_clock::now();

    BenchResult result;
    result.name = "open_index";
    result.shards = static_cast<int>(index.shard_paths.size());
    result.games = games;
    result.seconds = std::chrono::duration<double>(end - start).count();
    return result;
}

BenchResult bench_sequential_scan(const ReplayIndexInfo& index) {
    BenchResult result;
    result.name = "sequential_scan";
    result.shards = static_cast<int>(index.shard_paths.size());
    for (const auto& path : index.shard_paths) {
        result.bytes += file_size_bytes(path);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& path : index.shard_paths) {
        BinaryShardReader reader(path);
        for (size_t i = 0; i < reader.history_count(); ++i) {
            GameHistory history = reader.read_at(i);
            ++result.games;
            result.steps += static_cast<int>(history.steps.size());
            result.checksum += checksum_history(history);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    result.seconds = std::chrono::duration<double>(end - start).count();
    return result;
}

BenchResult bench_random_read(
    const ReplayIndexInfo& index,
    int samples,
    uint64_t seed
) {
    std::vector<std::unique_ptr<BinaryShardReader>> readers;
    readers.reserve(index.shard_paths.size());
    std::vector<size_t> cumulative;
    cumulative.reserve(index.shard_paths.size());

    size_t total_games = 0;
    for (const auto& path : index.shard_paths) {
        auto reader = std::make_unique<BinaryShardReader>(path);
        total_games += reader->history_count();
        cumulative.push_back(total_games);
        readers.push_back(std::move(reader));
    }
    if (total_games == 0) throw std::runtime_error("No games in replay index");

    XorShift64 rng(seed);
    BenchResult result;
    result.name = "random_read";
    result.shards = static_cast<int>(readers.size());

    auto start = std::chrono::high_resolution_clock::now();
    for (int sample = 0; sample < samples; ++sample) {
        size_t global_index = static_cast<size_t>(rng.next() % total_games);
        auto it = std::upper_bound(cumulative.begin(), cumulative.end(), global_index);
        size_t shard_id = static_cast<size_t>(it - cumulative.begin());
        size_t shard_base = shard_id == 0 ? 0 : cumulative[shard_id - 1];
        size_t local_index = global_index - shard_base;

        GameHistory history = readers[shard_id]->read_at(local_index);
        ++result.games;
        result.steps += static_cast<int>(history.steps.size());
        result.bytes += history_payload_bytes(history);
        result.checksum += checksum_history(history);
    }
    auto end = std::chrono::high_resolution_clock::now();
    result.seconds = std::chrono::duration<double>(end - start).count();
    return result;
}

void print_result(const BenchResult& result) {
    double mb = static_cast<double>(result.bytes) / (1024.0 * 1024.0);
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
              << ",\"mb_read\":" << mb
              << ",\"mb_per_second\":"
              << (result.seconds > 0.0 ? mb / result.seconds : 0.0)
              << ",\"checksum\":" << result.checksum
              << "}" << std::endl;
}

void print_index_metadata(const ReplayIndexInfo& index) {
    std::cout << "{\"case\":\"index_metadata\""
              << ",\"index_version\":" << index.metadata.index_version
              << ",\"legacy_index\":"
              << (index.metadata.legacy_v1 ? "true" : "false")
              << ",\"replay_format_version\":"
              << index.metadata.compatibility.replay_format_version
              << ",\"environment_rule_version\":"
              << index.metadata.compatibility.environment_rule_version
              << ",\"observation_schema_version\":"
              << index.metadata.compatibility.observation_schema_version
              << ",\"action_space_version\":"
              << index.metadata.compatibility.action_space_version
              << ",\"network_architecture_version\":"
              << index.metadata.compatibility.network_architecture_version
              << "}" << std::endl;
}

void print_help(const char* argv0) {
    std::cout
        << "Usage: " << argv0
        << " --index data/selfplay/train.index.json [options]\n"
        << "\n"
        << "Options:\n"
        << "  --index PATH           Shard index JSON from generate_selfplay_shards.\n"
        << "  --random-samples N    Number of random GameHistory reads. Default: 1024\n"
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
            std::string arg = argv[i];
            if (arg == "--index") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("Missing value for --index");
                }
                index_path = argv[++i];
            } else if (arg == "--random-samples") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("Missing value for --random-samples");
                }
                random_samples = std::stoi(argv[++i]);
            } else if (arg == "--seed") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("Missing value for --seed");
                }
                seed = static_cast<uint64_t>(std::stoull(argv[++i]));
            } else if (arg == "--allow-legacy-index") {
                allow_legacy_index = true;
            } else if (arg == "--help" || arg == "-h") {
                print_help(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("Unknown argument: " + arg);
            }
        }
        if (random_samples <= 0) {
            throw std::runtime_error("--random-samples must be positive");
        }

        const LegacyReplayIndexPolicy legacy_policy = allow_legacy_index
            ? LegacyReplayIndexPolicy::AllowV1
            : LegacyReplayIndexPolicy::Reject;
        ReplayIndexInfo index = load_replay_index_json(index_path, legacy_policy);
        print_index_metadata(index);
        print_result(bench_open_index(index));
        print_result(bench_sequential_scan(index));
        print_result(bench_random_read(index, random_samples, seed));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
