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
    uint64_t bytes = 0;
    uint64_t game_reads = 0;
    double seconds = 0.0;
    double checksum = 0.0;
};

void print_result(const BenchResult& r) {
    double mb = static_cast<double>(r.bytes) / (1024.0 * 1024.0);
    double samples_per_game_read = r.game_reads > 0
        ? static_cast<double>(r.samples) / static_cast<double>(r.game_reads)
        : 0.0;
    std::cout << std::fixed << std::setprecision(6)
              << "{\"case\":\"" << r.name << "\""
              << ",\"batches\":" << r.batches
              << ",\"batch_size\":" << r.batch_size
              << ",\"samples\":" << r.samples
              << ",\"observation_size\":" << r.observation_size
              << ",\"policy_size\":" << r.policy_size
              << ",\"legal_mask_size\":" << r.legal_mask_size
              << ",\"game_reads\":" << r.game_reads
              << ",\"samples_per_game_read\":" << samples_per_game_read
              << ",\"seconds\":" << r.seconds
              << ",\"batches_per_second\":" << (r.seconds > 0.0 ? r.batches / r.seconds : 0.0)
              << ",\"samples_per_second\":" << (r.seconds > 0.0 ? r.samples / r.seconds : 0.0)
              << ",\"mb_packed\":" << mb
              << ",\"mb_per_second\":" << (r.seconds > 0.0 ? mb / r.seconds : 0.0)
              << ",\"checksum\":" << r.checksum
              << "}" << std::endl;
}

void print_help(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " --index data/selfplay/train.index.json [options]\n"
        << "\n"
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
            std::string arg = argv[i];
            if (arg == "--index") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --index");
                index_path = argv[++i];
            } else if (arg == "--batch-size") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --batch-size");
                batch_size = std::stoi(argv[++i]);
            } else if (arg == "--batches") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --batches");
                batches = std::stoi(argv[++i]);
            } else if (arg == "--seed") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --seed");
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

        if (batch_size <= 0) throw std::runtime_error("--batch-size must be positive");
        if (batches <= 0) throw std::runtime_error("--batches must be positive");

        const LegacyReplayIndexPolicy legacy_policy = allow_legacy_index
            ? LegacyReplayIndexPolicy::AllowV1
            : LegacyReplayIndexPolicy::Reject;

        auto open_start = std::chrono::high_resolution_clock::now();
        ReplayDataset dataset = ReplayDataset::from_index_json(index_path, legacy_policy);
        auto open_end = std::chrono::high_resolution_clock::now();
        double open_seconds = std::chrono::duration<double>(open_end - open_start).count();
        std::cout << std::fixed << std::setprecision(6)
                  << "{\"case\":\"open_dataset\""
                  << ",\"shards\":" << dataset.shard_count()
                  << ",\"games\":" << dataset.game_count()
                  << ",\"index_version\":" << dataset.index_metadata().index_version
                  << ",\"legacy_index\":" << (dataset.index_metadata().legacy_v1 ? "true" : "false")
                  << ",\"seconds\":" << open_seconds
                  << "}" << std::endl;

        ReplayBatchSampler sampler(dataset, seed);
        BenchResult r;
        r.name = "batch_sample_pack";
        r.batch_size = batch_size;
        r.batches = batches;
        r.samples = batch_size * batches;

        dataset.reset_game_read_count();
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < batches; ++i) {
            ReplayBatch batch = sampler.sample_batch(batch_size);
            if (i == 0) {
                r.observation_size = batch.observation_size;
                r.policy_size = batch.policy_size;
                r.legal_mask_size = batch.legal_mask_size;
            }
            r.bytes += replay_batch_payload_bytes(batch);
            r.checksum += replay_batch_checksum(batch);
        }
        auto end = std::chrono::high_resolution_clock::now();
        r.seconds = std::chrono::duration<double>(end - start).count();
        r.game_reads = dataset.game_read_count();
        print_result(r);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
