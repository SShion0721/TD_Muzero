#pragma once

#include "tdmz/core/compatibility.hpp"
#include "tdmz/core/wave_mode.hpp"
#include "tdmz/selfplay/game_history.hpp"
#include "tdmz/selfplay/transition_shard.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tdmz {

class ReplayShardReaderBase;

struct ReplayBatch {
    int batch_size = 0;
    int observation_size = 0;
    int policy_size = 0;
    int legal_mask_size = 0;

    std::vector<float> observations;
    std::vector<float> policy_targets;
    std::vector<float> values;
    std::vector<float> rewards;
    std::vector<int> actions;
    std::vector<uint8_t> legal_masks;
    std::vector<uint8_t> dones;

    bool direct_transition_reads = false;
    bool io_stats_exact = false;
    int unique_games_touched = 0;
    int unique_shards_touched = 0;
    uint64_t physical_read_operations = 0;
    uint64_t physical_bytes_read = 0;
};

struct ReplaySampleRef {
    size_t global_game_index = 0;
    size_t shard_index = 0;
    size_t local_game_index = 0;
    size_t step_index = 0;
    uint64_t physical_offset = 0;
};

enum class LegacyReplayIndexPolicy {
    Reject,
    AllowV1,
};

struct ReplayIndexMetadata {
    std::string format;
    uint32_t index_version = 0;
    bool legacy_v1 = false;
    CompatibilityMetadata compatibility = current_compatibility_metadata();
    WaveMode wave_mode = WaveMode::Unknown;
};

struct ReplayIndexInfo {
    std::string index_path;
    std::vector<std::string> shard_paths;
    ReplayIndexMetadata metadata;
};

ReplayIndexInfo load_replay_index_json(
    const std::string& index_path,
    LegacyReplayIndexPolicy legacy_policy = LegacyReplayIndexPolicy::Reject
);

class ReplayDataset {
public:
    // Direct shard construction has no index from which to infer provenance,
    // so callers must state the single wave mode for the complete shard set.
    ReplayDataset(std::vector<std::string> shard_paths, WaveMode wave_mode);
    ~ReplayDataset();

    ReplayDataset(const ReplayDataset&) = delete;
    ReplayDataset& operator=(const ReplayDataset&) = delete;
    ReplayDataset(ReplayDataset&&) noexcept;
    ReplayDataset& operator=(ReplayDataset&&) noexcept;

    static ReplayDataset from_index_json(
        const std::string& index_path,
        LegacyReplayIndexPolicy legacy_policy = LegacyReplayIndexPolicy::Reject
    );

    size_t shard_count() const;
    size_t game_count() const;
    const std::vector<std::string>& shard_paths() const;
    bool has_index_metadata() const { return has_index_metadata_; }
    const ReplayIndexMetadata& index_metadata() const { return index_metadata_; }
    bool transition_indexed() const { return transition_indexed_; }
    uint32_t replay_format_version() const { return replay_format_version_; }
    int observation_size() const { return observation_size_; }
    int policy_size() const { return policy_size_; }
    int legal_mask_size() const { return legal_mask_size_; }
    WaveMode wave_mode() const { return wave_mode_; }

    ReplaySampleRef locate_game(size_t global_game_index) const;
    GameHistory read_game(size_t global_game_index);
    ReplaySampleRef sample_ref(uint64_t random_u64);
    TrajectoryStep sample_step(uint64_t random_u64);

    size_t step_count(size_t global_game_index) const;
    void read_step_into_batch(
        const ReplaySampleRef& ref,
        ReplayBatch& batch,
        size_t output_index
    );

    ReplayIoStats io_stats() const;
    void reset_io_stats();

    uint64_t game_read_count() const { return game_read_count_; }
    void reset_game_read_count() { game_read_count_ = 0; }

private:
    ReplayDataset(
        std::vector<std::string> shard_paths,
        ReplayIndexMetadata index_metadata
    );

    void open_shards();

    std::vector<std::string> shard_paths_;
    std::vector<std::unique_ptr<ReplayShardReaderBase>> readers_;
    std::vector<size_t> cumulative_games_;
    ReplayIndexMetadata index_metadata_;
    bool has_index_metadata_ = false;
    bool transition_indexed_ = false;
    uint32_t replay_format_version_ = 0;
    int observation_size_ = 0;
    int policy_size_ = 0;
    int legal_mask_size_ = 0;
    WaveMode wave_mode_ = WaveMode::Unknown;
    uint64_t game_read_count_ = 0;
};

class ReplayBatchSampler {
public:
    ReplayBatchSampler(ReplayDataset& dataset, uint64_t seed = 1234567);

    ReplayBatch sample_batch(int batch_size);

private:
    uint64_t next_u64();

    ReplayDataset& dataset_;
    uint64_t rng_state_;
};

uint64_t replay_batch_payload_bytes(const ReplayBatch& batch);
double replay_batch_checksum(const ReplayBatch& batch);

} // namespace tdmz
