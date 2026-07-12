#pragma once

#include "tdmz/selfplay/game_history.hpp"
#include "tdmz/selfplay/training_replay.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tdmz {

struct SequenceTargetConfig {
    int unroll_steps = 5;
    int td_steps = 10;
    float discount = 0.997f;
};

struct KStepSample {
    size_t global_game_index = 0;
    size_t start_step = 0;
    int unroll_steps = 0;
    int observation_size = 0;
    int policy_size = 0;
    int legal_mask_size = 0;

    bool episode_terminal = false;
    bool episode_truncated = false;

    // Initial real observation [observation_size].
    std::vector<float> initial_observation;

    // K recurrent transition targets.
    std::vector<int> actions;                  // [K]
    std::vector<float> reward_targets;         // [K]
    std::vector<uint8_t> transition_valid;     // [K]

    // K+1 state targets. Invalid tail states are zero padded.
    std::vector<float> value_targets;          // [K+1]
    std::vector<uint8_t> value_valid;          // [K+1]
    std::vector<float> policy_targets;         // [K+1, policy_size]
    std::vector<uint8_t> legal_masks;           // [K+1, legal_mask_size]
    std::vector<uint8_t> state_valid;           // [K+1]
};

struct KStepBatch {
    int batch_size = 0;
    int unroll_steps = 0;
    int observation_size = 0;
    int policy_size = 0;
    int legal_mask_size = 0;

    std::vector<float> initial_observations;    // [B, observation_size]
    std::vector<int> actions;                   // [B, K]
    std::vector<float> reward_targets;          // [B, K]
    std::vector<uint8_t> transition_valid;      // [B, K]
    std::vector<float> value_targets;           // [B, K+1]
    std::vector<uint8_t> value_valid;           // [B, K+1]
    std::vector<float> policy_targets;           // [B, K+1, policy_size]
    std::vector<uint8_t> legal_masks;            // [B, K+1, legal_mask_size]
    std::vector<uint8_t> state_valid;            // [B, K+1]
    std::vector<uint8_t> episode_terminal;       // [B]
    std::vector<uint8_t> episode_truncated;      // [B]
};

// Builds one MuZero K-step sample from a completed terminal or truncated game.
KStepSample build_k_step_sample(
    const GameHistory& history,
    size_t start_step,
    const SequenceTargetConfig& config = SequenceTargetConfig{}
);

KStepBatch pack_k_step_samples(const std::vector<KStepSample>& samples);

// Position-uniform index: every stored root state has equal probability,
// independent of the length of its game.
class UniformPositionIndex {
public:
    explicit UniformPositionIndex(const TrainingReplayDataset& dataset);

    size_t position_count() const { return total_positions_; }
    ReplaySampleRef locate(size_t global_position_index) const;
    ReplaySampleRef sample(uint64_t random_u64) const;

private:
    const TrainingReplayDataset* dataset_ = nullptr;
    std::vector<size_t> cumulative_positions_by_game_;
    size_t total_positions_ = 0;
};

// Correct reference sampler for Phase D. This first implementation reads one
// complete game per unique sampled game. A later direct-range reader may replace
// the I/O path without changing the target semantics or public batch layout.
class KStepReplaySampler {
public:
    KStepReplaySampler(
        TrainingReplayDataset& dataset,
        SequenceTargetConfig config = SequenceTargetConfig{},
        uint64_t seed = 1234567
    );

    KStepBatch sample_batch(int batch_size);
    const UniformPositionIndex& position_index() const { return positions_; }

private:
    uint64_t next_u64();

    TrainingReplayDataset& dataset_;
    SequenceTargetConfig config_;
    UniformPositionIndex positions_;
    uint64_t rng_state_;
};

uint64_t k_step_batch_payload_bytes(const KStepBatch& batch);
double k_step_batch_checksum(const KStepBatch& batch);

} // namespace tdmz
