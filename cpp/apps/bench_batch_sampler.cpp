#include "tdmz/selfplay/replay_dataset.hpp"

#include <chrono>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace tdmz;

namespace {

struct BenchResult {
    std::string name;
    int batches = 0;
    int samples = 0;
    int batch_size = 0;
    int observation_size = 0;
    int policy_size = 0;
    int legal_mask_size = 0;
    bool direct_transition_reads = false;
    bool io_stats_exact = false;
    uint64_t packed_bytes = 0;
    uint64_t physical_bytes = 0;
    uint64_t physical_read_operations = 0;
    uint64_t game_reads = 0;
    uint64_t unique_games_sum = 0;
    uint64_t unique_shards_sum = 0;
    double seconds = 0.0;
    double checksum = 0.0;
};

void print_result(const BenchResult& result) {
    const double packed_mb = static_cast<double>(result.packed_bytes) / (1024.0 * 1024.0);
    const double physical_mb = static_cast<double>(result.physical_bytes) / (1024.0 * 1024.0);
    const double samples_per_game_read = result.game_reads > 0
        ? static_cast<double>(result.samples) / static_cast<double>(result.game_reads)
        : 0.0;
    const double samples_per_physical_read = result.physical_read_operations > 0
        ? static_cast<double>(result.samples) / static_cast<double>(result.physical_read_operations)
        : 0.0;
    const double read_amplification = result.packed_bytes > 0
        ? static_cast<double>(result.physical_bytes) / static_cast<double>(result.packed_bytes)
        : 0.0;
    const double average_unique_games = result.batches > 0
        ? static_cast<double>(result.unique_games_sum) / result.batches
        : 0.0;
    const double average_unique_shards = result.batches > 0
        ? static_cast<double>(result.unique_shards_sum) / result.batches
        : 0.0;

    std::cout << std::fixed << std::setprecision(6)
              << "{\"case\":\"" << result.name << "\""
              << ",\"batches\":" << result.batches
              << ",\"batch_size\":" << result.batch_size
              << ",\"samples\":" << result.samples
              << ",\"observation_size\":" << result.observation_size
              << ",\"policy_size\":" << result.policy_size
              << ",\"legal_mask_size\":" << result.legal_mask_size
              << ",\"direct_transition_reads\":"
              << (result.direct_transition_reads ? "true" : "false")
              << ",\"io_stats_exact\":" << (result.io_stats_exact ? "true" : "false")
              << ",\"game_reads\":" << result.game_reads
              << ",\"physical_read_operations\":" << result.physical_read_operations
              << ",\"samples_per_game_read\":" << samples_per_game_read
              << ",\"samples_per_physical_read\":" << samples_per_physical_read
              << ",\"average_unique_games\":" << average_unique_games
              << ",\"average_unique_shards\":" << average_unique_shards
              << ",\"seconds\":" << result.seconds
              << ",\"batches_per_second\":"
              << (result.seconds > 0.0 ? result.batches / result.seconds : 0.0)
              << ",\"samples_per_second\":"
              << (result.seconds > 0.0 ? result.samples / result.seconds : 0.0)
              << ",\"mb_packed\":" << packed_mb
              << ",\"physical_mb_read\":" << physical_mb
              << ",\"read_amplification\":" << read_amplification
              << ",\"packed_mb_per_second\":"
              << (result.seconds > 0.0 ? packed_mb / result.seconds : 0.0)
              << ",\"physical_mb_per_second\":"
              << (result.seconds > 0.0 ? physical_mb / result.seconds : 0.0)
              << ",\"checksum\":" << result.checksum
              << "}" << std::endl;
}

void print_help(const char* executable) {
    std::cout
        << "Usage: " << executable << " --index data/selfplay/train.index.json [options]\n\n"
        << "Options:\n"
        << "  --index PATH           Shard index JSON from generate_selfplay_shards.\n"
        << "  --batch-size N         Samples per batch. Default: 256\n"
        << "  --batches N            Number of batches to sample. Default: 256\n"
        << "  --seed N               Sampler seed. Default: 1234567\n"
        << "  --allow-legacy-index   Explicitly allow a reviewed v1/unversioned index.\n"
        << "  --help                 Show this help.\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string index_path = "data/selfplay/train.index.json";
        int batch_size = 256;
        int batches = 256;
        uint64_t seed = 1234567;
        bool allow_legacy_index = false;

        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--index") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --index");
                index_path = argv[++i];
            } else if (argument == "--batch-size") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --batch-size");
                batch_size = std::stoi(argv[++i]);
            } else if (argument == "--batches") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --batches");
                batches = std::stoi(argv[++i]);
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

        if (batch_size <= 0) throw std::runtime_error("--batch-size must be positive");
        if (batches <= 0) throw std::runtime_error("--batches must be positive");

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

        ReplayBatchSampler sampler(dataset, seed);
        BenchResult result;
        result.name = "batch_sample_pack";
        result.batch_size = batch_size;
        result.batches = batches;
        result.samples = batch_size * batches;

        dataset.reset_game_read_count();
        dataset.reset_io_stats();
        const auto start = std::chrono::high_resolution_clock::now();
        for (int batch_index = 0; batch_index < batches; ++batch_index) {
            ReplayBatch batch = sampler.sample_batch(batch_size);
            if (batch_index == 0) {
                result.observation_size = batch.observation_size;
                result.policy_size = batch.policy_size;
                result.legal_mask_size = batch.legal_mask_size;
                result.direct_transition_reads = batch.direct_transition_reads;
                result.io_stats_exact = batch.io_stats_exact;
            } else if (batch.direct_transition_reads != result.direct_transition_reads
                       || batch.io_stats_exact != result.io_stats_exact) {
                throw std::runtime_error("Replay batch I/O mode changed during benchmark");
            }
            result.packed_bytes += replay_batch_payload_bytes(batch);
            result.physical_bytes += batch.physical_bytes_read;
            result.physical_read_operations += batch.physical_read_operations;
            result.unique_games_sum += static_cast<uint64_t>(batch.unique_games_touched);
            result.unique_shards_sum += static_cast<uint64_t>(batch.unique_shards_touched);
            result.checksum += replay_batch_checksum(batch);
        }
        result.seconds = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();
        result.game_reads = dataset.game_read_count();
        print_result(result);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
