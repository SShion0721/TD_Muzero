#include "tdmz/selfplay/training_replay.hpp"
#include "tdmz/training/direct_sequence_replay.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace tdmz;

namespace {

void print_help(const char* executable) {
    std::cout
        << "Usage: " << executable << " --index PATH [options]\n\n"
        << "Options:\n"
        << "  --index PATH       Replay v3 index JSON.\n"
        << "  --batch-size N     Samples per batch. Default: 64\n"
        << "  --batches N        Number of batches. Default: 128\n"
        << "  --unroll N         MuZero unroll K. Default: 5\n"
        << "  --td-steps N       TD horizon n. Default: 10\n"
        << "  --discount X       Discount. Default: 0.997\n"
        << "  --seed N           Sampler seed. Default: 1234567\n"
        << "  --help             Show this help.\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string index_path = "data/selfplay/train.index.json";
        int batch_size = 64;
        int batches = 128;
        uint64_t seed = 1234567;
        SequenceTargetConfig config;

        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            auto require_value = [&](const char* name) -> std::string {
                if (index + 1 >= argc) {
                    throw std::runtime_error(std::string("Missing value for ") + name);
                }
                return argv[++index];
            };

            if (argument == "--index") {
                index_path = require_value("--index");
            } else if (argument == "--batch-size") {
                batch_size = std::stoi(require_value("--batch-size"));
            } else if (argument == "--batches") {
                batches = std::stoi(require_value("--batches"));
            } else if (argument == "--unroll") {
                config.unroll_steps = std::stoi(require_value("--unroll"));
            } else if (argument == "--td-steps") {
                config.td_steps = std::stoi(require_value("--td-steps"));
            } else if (argument == "--discount") {
                config.discount = std::stof(require_value("--discount"));
            } else if (argument == "--seed") {
                seed = static_cast<uint64_t>(std::stoull(require_value("--seed")));
            } else if (argument == "--help" || argument == "-h") {
                print_help(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("Unknown argument: " + argument);
            }
        }

        if (batch_size <= 0) throw std::runtime_error("--batch-size must be positive");
        if (batches <= 0) throw std::runtime_error("--batches must be positive");

        const auto open_start = std::chrono::high_resolution_clock::now();
        TrainingReplayDataset training = TrainingReplayDataset::from_index_json(index_path);
        DirectSequenceReplayDataset direct_dataset(training);
        DirectKStepReplaySampler sampler(direct_dataset, config, seed);
        const double open_seconds = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - open_start).count();

        uint64_t packed_bytes = 0;
        uint64_t physical_bytes = 0;
        uint64_t physical_reads = 0;
        uint64_t requested_records = 0;
        uint64_t unique_records = 0;
        uint64_t bootstrap_records = 0;
        uint64_t unique_games_sum = 0;
        uint64_t unique_shards_sum = 0;
        double checksum = 0.0;
        double reference_checksum = 0.0;

        direct_dataset.reset_io_stats();
        training.replay().reset_game_read_count();
        const auto start = std::chrono::high_resolution_clock::now();
        for (int batch_index = 0; batch_index < batches; ++batch_index) {
            const std::vector<ReplaySampleRef> refs = sampler.sample_refs(batch_size);
            DirectSequenceBatchStats stats;
            const KStepBatch batch = build_direct_k_step_batch(
                direct_dataset, refs, config, &stats);
            packed_bytes += k_step_batch_payload_bytes(batch);
            physical_bytes += stats.physical_bytes_read;
            physical_reads += stats.physical_read_operations;
            requested_records += stats.requested_step_records;
            unique_records += stats.unique_step_records_read;
            bootstrap_records += stats.bootstrap_records_read;
            unique_games_sum += static_cast<uint64_t>(stats.unique_games_touched);
            unique_shards_sum += static_cast<uint64_t>(stats.unique_shards_touched);
            checksum += k_step_batch_checksum(batch);

            if (batch_index == 0) {
                const KStepBatch reference = build_reference_k_step_batch(
                    training, refs, config);
                reference_checksum = k_step_batch_checksum(reference);
                if (std::fabs(reference_checksum - k_step_batch_checksum(batch)) > 1e-9) {
                    throw std::runtime_error(
                        "Direct/reference sequence checksum mismatch on benchmark gate batch");
                }
            }
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();

        const uint64_t samples = static_cast<uint64_t>(batch_size)
            * static_cast<uint64_t>(batches);
        const double packed_mb = static_cast<double>(packed_bytes) / (1024.0 * 1024.0);
        const double physical_mb = static_cast<double>(physical_bytes) / (1024.0 * 1024.0);
        const double dedup_ratio = requested_records > 0
            ? static_cast<double>(unique_records) / static_cast<double>(requested_records)
            : 0.0;
        const double read_amplification = packed_bytes > 0
            ? static_cast<double>(physical_bytes) / static_cast<double>(packed_bytes)
            : 0.0;

        std::cout << std::fixed << std::setprecision(6)
                  << "{\"case\":\"direct_sequence_replay\""
                  << ",\"index\":\"" << index_path << "\""
                  << ",\"open_seconds\":" << open_seconds
                  << ",\"shards\":" << direct_dataset.shard_count()
                  << ",\"games\":" << direct_dataset.game_count()
                  << ",\"positions\":" << sampler.position_index().position_count()
                  << ",\"batch_size\":" << batch_size
                  << ",\"batches\":" << batches
                  << ",\"samples\":" << samples
                  << ",\"unroll_steps\":" << config.unroll_steps
                  << ",\"td_steps\":" << config.td_steps
                  << ",\"discount\":" << config.discount
                  << ",\"seconds\":" << seconds
                  << ",\"batches_per_second\":"
                  << (seconds > 0.0 ? batches / seconds : 0.0)
                  << ",\"samples_per_second\":"
                  << (seconds > 0.0 ? samples / seconds : 0.0)
                  << ",\"packed_mb\":" << packed_mb
                  << ",\"physical_mb_read\":" << physical_mb
                  << ",\"physical_read_operations\":" << physical_reads
                  << ",\"requested_step_records\":" << requested_records
                  << ",\"unique_step_records_read\":" << unique_records
                  << ",\"bootstrap_records_read\":" << bootstrap_records
                  << ",\"unique_record_ratio\":" << dedup_ratio
                  << ",\"read_amplification\":" << read_amplification
                  << ",\"average_unique_games\":"
                  << static_cast<double>(unique_games_sum) / batches
                  << ",\"average_unique_shards\":"
                  << static_cast<double>(unique_shards_sum) / batches
                  << ",\"reference_game_reads\":" << training.replay().game_read_count()
                  << ",\"reference_gate_checksum\":" << reference_checksum
                  << ",\"checksum\":" << checksum
                  << "}" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
