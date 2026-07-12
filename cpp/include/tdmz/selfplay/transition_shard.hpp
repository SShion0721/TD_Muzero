#pragma once

#include "tdmz/selfplay/game_history.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tdmz {

inline constexpr uint32_t kTransitionShardFormatVersion = 3;

struct ReplayStepDestination {
    float* observation = nullptr;
    size_t observation_size = 0;
    float* policy_target = nullptr;
    size_t policy_size = 0;
    uint8_t* legal_mask = nullptr;
    size_t legal_mask_size = 0;

    float* value = nullptr;
    float* reward = nullptr;
    int* action = nullptr;
    uint8_t* done = nullptr;

    int* step_index = nullptr;
    int* money = nullptr;
    int* base_hp = nullptr;
    int* wave = nullptr;
    float* time = nullptr;
};

struct ReplayGameMetadata {
    uint64_t seed = 0;
    int max_steps = 0;
    bool terminal = false;
    bool truncated = false;
    WaveMode wave_mode = WaveMode::Unknown;
    float total_reward = 0.0f;
    size_t step_count = 0;
    bool has_bootstrap_state = false;
};

struct ReplayIoStats {
    uint64_t physical_read_operations = 0;
    uint64_t physical_bytes_read = 0;
};

// Kept for source compatibility with the original v2 detector. It returns true
// for indexed transition shards (legacy v2 or current v3); the reader then emits
// an explicit version error for unsupported v2 data.
bool is_transition_shard_v2(const std::string& path);

class TransitionShardReader {
public:
    explicit TransitionShardReader(const std::string& path);
    ~TransitionShardReader();

    TransitionShardReader(const TransitionShardReader&) = delete;
    TransitionShardReader& operator=(const TransitionShardReader&) = delete;
    TransitionShardReader(TransitionShardReader&&) noexcept;
    TransitionShardReader& operator=(TransitionShardReader&&) noexcept;

    size_t history_count() const;
    size_t step_count(size_t game_index) const;
    uint64_t step_physical_offset(size_t game_index, size_t step_index) const;

    int observation_size() const;
    int policy_size() const;
    int legal_mask_size() const;
    const std::string& path() const;

    ReplayGameMetadata game_metadata(size_t game_index) const;
    GameHistory read_at(size_t game_index);
    TrajectoryStep read_step(size_t game_index, size_t step_index);
    BootstrapState read_bootstrap_state(size_t game_index);
    void read_step_into(
        size_t game_index,
        size_t step_index,
        const ReplayStepDestination& destination
    );

    ReplayIoStats io_stats() const;
    void reset_io_stats();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class TransitionShardWriter {
public:
    TransitionShardWriter(const std::string& path, size_t expected_history_count);
    ~TransitionShardWriter();

    TransitionShardWriter(const TransitionShardWriter&) = delete;
    TransitionShardWriter& operator=(const TransitionShardWriter&) = delete;

    void write_history(const GameHistory& history);
    void close();

    size_t expected_history_count() const;
    size_t written_count() const;
    bool closed() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

enum class AsyncTransitionShardWriterState {
    Open,
    Closing,
    Closed,
    Failed,
};

class AsyncTransitionShardWriter {
public:
    AsyncTransitionShardWriter(
        const std::string& path,
        size_t expected_history_count,
        size_t max_queue_size = 4
    );
    ~AsyncTransitionShardWriter();

    AsyncTransitionShardWriter(const AsyncTransitionShardWriter&) = delete;
    AsyncTransitionShardWriter& operator=(const AsyncTransitionShardWriter&) = delete;

    void write_history(GameHistory history);
    void close();

    size_t expected_history_count() const;
    size_t enqueued_count() const;
    size_t max_queue_size() const;
    AsyncTransitionShardWriterState state() const;
    bool closed() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

void write_histories_transition_shard(
    const std::vector<GameHistory>& histories,
    const std::string& path
);

} // namespace tdmz
