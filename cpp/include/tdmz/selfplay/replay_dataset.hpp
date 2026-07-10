#pragma once
#include "tdmz/core/compatibility.hpp"
#include "tdmz/selfplay/game_history.hpp"
#include "tdmz/selfplay/trajectory_writer.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tdmz {

struct ReplayBatch {
    int batch_size = 0;
    int observation_size = 0;
    int policy_size = 0;
    int legal_mask_size = 0;

    // Flat contiguous training buffers.
    // observations:    [B, observation_size]
    // policy_targets:  [B, policy_size]
    // values:          [B]
    // rewards:         [B]
    // actions:         [B]
    // legal_masks:     [B, legal_mask_size]
    // dones:           [B]
    std::vector<float> observations;
    std::vector<float> policy_targets;
    std::vector<float> values;
    std::vector<float> rewards;
    std::vector<int> actions;
    std::vector<uint8_t> legal_masks;
    std::vector<uint8_t> dones;
};

struct ReplaySampleRef {
    size_t shard_index = 0;
    size_t local_game_index = 0;
    size_t step_index = 0;
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
    explicit ReplayDataset(std::vector<std::string> shard_paths);

    static ReplayDataset from_index_json(
        const std::string& index_path,
        LegacyReplayIndexPolicy legacy_policy = LegacyReplayIndexPolicy::Reject
    );

    size_t shard_count() const;
    size_t game_count() const;
    const std::vector<std::string>& shard_paths() const;
    bool has_index_metadata() const { return has_index_metadata_; }
    const ReplayIndexMetadata& index_metadata() const { return index_metadata_; }

    GameHistory read_game(size_t global_game_index);
    ReplaySampleRef sample_ref(uint64_t random_u64);
    TrajectoryStep sample_step(uint64_t random_u64);

    uint64_t game_read_count() const { return game_read_count_; }
    void reset_game_read_count() { game_read_count_ = 0; }

private:
    ReplayDataset(
        std::vector<std::string> shard_paths,
        ReplayIndexMetadata index_metadata
    );

    void open_shards();

    std::vector<std::string> shard_paths_;
    std::vector<std::unique_ptr<BinaryShardReader>> readers_;
    std::vector<size_t> cumulative_games_;
    ReplayIndexMetadata index_metadata_;
    bool has_index_metadata_ = false;
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
