#include "tdmz/selfplay/trajectory_writer.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tdmz;

namespace {

struct ReplayIndex {
    std::string path;
    std::vector<std::string> shard_paths;
};

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
    explicit XorShift64(uint64_t seed) : state(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
    uint64_t next() {
        uint64_t x = state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state = x;
        return x;
    }
};

std::string read_text_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open index JSON: " + path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string unescape_json_string(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char n = s[++i];
            switch (n) {
                case '\\': out.push_back('\\'); break;
                case '"': out.push_back('"'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                default: out.push_back(n); break;
            }
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

ReplayIndex load_replay_index(const std::string& path) {
    std::string text = read_text_file(path);
    ReplayIndex index;
    index.path = path;

    std::regex path_re("\\\"path\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
    auto begin = std::sregex_iterator(text.begin(), text.end(), path_re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        index.shard_paths.push_back(unescape_json_string((*it)[1].str()));
    }
    if (index.shard_paths.empty()) {
        throw std::runtime_error("No shard paths found in index JSON: " + path);
    }
    return index;
}

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
    double sum = static_cast<double>(history.steps.size()) * 17.0 + static_cast<double>(history.total_reward);
    for (const auto& step : history.steps) {
        sum += step.action * 0.125;
        sum += step.reward * 3.0;
        sum += step.root_value * 5.0;
        sum += step.money * 0.01;
        sum += step.base_hp * 0.02;
        sum += step.wave * 0.5;
        if (!step.policy_target.empty()) {
            int action = std::max(0, std::min(step.action, static_cast<int>(step.policy_target.size()) - 1));
            sum += step.policy_target[static_cast<size_t>(action)] * 11.0;
        }
        for (size_t i = 0; i < step.observation.size(); i += 97) {
            sum += step.observation[i] * static_cast<double>((i % 29) + 1);
        }
    }
    return sum;
}

BenchResult bench_open_index(const ReplayIndex& index) {
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

    BenchResult r;
    r.name = "open_index";
    r.shards = static_cast<int>(index.shard_paths.size());
    r.games = games;
    r.seconds = std::chrono::duration<double>(end - start).count();
    return r;
}

BenchResult bench_sequential_scan(const ReplayIndex& index) {
    BenchResult r;
    r.name = "sequential_scan";
    r.shards = static_cast<int>(index.shard_paths.size());
    for (const auto& path : index.shard_paths) {
        r.bytes += file_size_bytes(path);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& path : index.shard_paths) {
        BinaryShardReader reader(path);
        for (size_t i = 0; i < reader.history_count(); ++i) {
            GameHistory history = reader.read_at(i);
            ++r.games;
            r.steps += static_cast<int>(history.steps.size());
            r.checksum += checksum_history(history);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    r.seconds = std::chrono::duration<double>(end - start).count();
    return r;
}

BenchResult bench_random_read(const ReplayIndex& index, int samples, uint64_t seed) {
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
    BenchResult r;
    r.name = "random_read";
    r.shards = static_cast<int>(readers.size());

    auto start = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < samples; ++s) {
        size_t global_index = static_cast<size_t>(rng.next() % total_games);
        auto it = std::upper_bound(cumulative.begin(), cumulative.end(), global_index);
        size_t shard_id = static_cast<size_t>(it - cumulative.begin());
        size_t shard_base = shard_id == 0 ? 0 : cumulative[shard_id - 1];
        size_t local_index = global_index - shard_base;

        GameHistory history = readers[shard_id]->read_at(local_index);
        ++r.games;
        r.steps += static_cast<int>(history.steps.size());
        r.bytes += history_payload_bytes(history);
        r.checksum += checksum_history(history);
    }
    auto end = std::chrono::high_resolution_clock::now();
    r.seconds = std::chrono::duration<double>(end - start).count();
    return r;
}

void print_result(const BenchResult& r) {
    double mb = static_cast<double>(r.bytes) / (1024.0 * 1024.0);
    std::cout << std::fixed << std::setprecision(6)
              << "{\"case\":\"" << r.name << "\""
              << ",\"shards\":" << r.shards
              << ",\"games\":" << r.games
              << ",\"steps\":" << r.steps
              << ",\"seconds\":" << r.seconds
              << ",\"games_per_second\":" << (r.seconds > 0.0 ? r.games / r.seconds : 0.0)
              << ",\"steps_per_second\":" << (r.seconds > 0.0 ? r.steps / r.seconds : 0.0)
              << ",\"mb_read\":" << mb
              << ",\"mb_per_second\":" << (r.seconds > 0.0 ? mb / r.seconds : 0.0)
              << ",\"checksum\":" << r.checksum
              << "}" << std::endl;
}

void print_help(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " --index data/selfplay/train.index.json [options]\n"
        << "\n"
        << "Options:\n"
        << "  --index PATH          Shard index JSON from generate_selfplay_shards.\n"
        << "  --random-samples N   Number of random GameHistory reads. Default: 1024\n"
        << "  --seed N             Random-read seed. Default: 1234567\n"
        << "  --help               Show this help.\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string index_path = "data/selfplay/train.index.json";
        int random_samples = 1024;
        uint64_t seed = 1234567;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--index") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --index");
                index_path = argv[++i];
            } else if (arg == "--random-samples") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --random-samples");
                random_samples = std::stoi(argv[++i]);
            } else if (arg == "--seed") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --seed");
                seed = static_cast<uint64_t>(std::stoull(argv[++i]));
            } else if (arg == "--help" || arg == "-h") {
                print_help(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("Unknown argument: " + arg);
            }
        }
        if (random_samples <= 0) throw std::runtime_error("--random-samples must be positive");

        ReplayIndex index = load_replay_index(index_path);
        print_result(bench_open_index(index));
        print_result(bench_sequential_scan(index));
        print_result(bench_random_read(index, random_samples, seed));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
