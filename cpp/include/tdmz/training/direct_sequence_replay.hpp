#pragma once

#include "tdmz/selfplay/training_replay.hpp"
#include "tdmz/selfplay/transition_shard.hpp"
#include "tdmz/training/sequence_replay.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tdmz {

// Training-only v3 reader. It opens transition shards directly and never
// deserializes a complete GameHistory for sequence sampling.
class DirectSequenceReplayDataset {
public:
    explicit DirectSequenceReplayDataset(const TrainingReplayDataset& source);
    ~DirectSequenceReplayDataset();

    DirectSequenceReplayDataset(const DirectSequenceReplayDataset&) = delete;
    DirectSequenceReplayDataset& operator=(const DirectSequenceReplayDataset&) = delete;
    DirectSequenceReplayDataset(DirectSequenceReplayDataset&&) noexcept;
    DirectSequenceReplayDataset& operator=(DirectSequenceReplayDataset&&) noexcept;

    size_t shard_count() const;
    size_t game_count() const;
    WaveMode wave_mode() const { return wave_mode_; }

    ReplaySampleRef locate_game(size_t global_game_index) const;
    ReplayGameMetadata game_metadata(size_t global_game_index) const;
    TrajectoryStep read_step(size_t global_game_index, size_t step_index);
    std::vector<TrajectoryStep> read_step_range(
        size_t global_game_index,
        size_t start_step,
        size_t count
    );
    BootstrapState read_bootstrap_state(size_t global_game_index);

    ReplayIoStats io_stats() const;
    void reset_io_stats();

private:
    std::vector<std::unique_ptr<TransitionShardReader>> readers_;
    std::vector<size_t> cumulative_games_;
    WaveMode wave_mode_ = WaveMode::Unknown;
};

// Every persisted root is sampleable. A truncated game contributes its real
// cutoff bootstrap root in addition to its transition roots.
class DirectPositionIndex {
public:
    explicit DirectPositionIndex(const DirectSequenceReplayDataset& dataset);

    size_t position_count() const { return total_positions_; }
    ReplaySampleRef locate(size_t global_position_index) const;
    ReplaySampleRef sample(uint64_t random_u64) const;

private:
    const DirectSequenceReplayDataset* dataset_ = nullptr;
    std::vector<size_t> cumulative_positions_by_game_;
    size_t total_positions_ = 0;
};

struct DirectSequenceBatchStats {
    int unique_games_touched = 0;
    int unique_shards_touched = 0;
    uint64_t requested_step_records = 0;
    uint64_t unique_step_records_read = 0;
    uint64_t bootstrap_records_read = 0;
    uint64_t physical_read_operations = 0;
    uint64_t physical_bytes_read = 0;
};

// Whole-game oracle retained for parity tests only.
KStepBatch build_reference_k_step_batch(
    TrainingReplayDataset& dataset,
    const std::vector<ReplaySampleRef>& refs,
    const SequenceTargetConfig& config = SequenceTargetConfig{}
);

// Direct v3 path. Overlapping windows in the same batch share cached step and
// cutoff records; no complete GameHistory is read from disk.
KStepBatch build_direct_k_step_batch(
    DirectSequenceReplayDataset& dataset,
    const std::vector<ReplaySampleRef>& refs,
    const SequenceTargetConfig& config = SequenceTargetConfig{},
    DirectSequenceBatchStats* stats = nullptr
);

class DirectKStepReplaySampler {
public:
    DirectKStepReplaySampler(
        DirectSequenceReplayDataset& dataset,
        SequenceTargetConfig config = SequenceTargetConfig{},
        uint64_t seed = 1234567
    );

    std::vector<ReplaySampleRef> sample_refs(int batch_size);
    KStepBatch sample_batch(int batch_size);

    const DirectPositionIndex& position_index() const { return positions_; }
    const DirectSequenceBatchStats& last_stats() const { return last_stats_; }

private:
    uint64_t next_u64();

    DirectSequenceReplayDataset& dataset_;
    SequenceTargetConfig config_;
    DirectPositionIndex positions_;
    uint64_t rng_state_;
    DirectSequenceBatchStats last_stats_;
};

} // namespace tdmz
