#pragma once
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "tdmz/selfplay/game_history.hpp"

namespace tdmz {

void write_history_jsonl(const GameHistory& history, const std::string& path);
void write_history_summary_json(const GameHistory& history, const std::string& path);

// Binary format intended for high-throughput training-data generation.
// File layout is POD headers plus contiguous float/byte arrays:
// header, then per-step fixed metadata, observation floats, policy floats, legal-mask bytes.
void write_history_binary(const GameHistory& history, const std::string& path);
GameHistory read_history_binary(const std::string& path);

// Multi-game binary shard. This is the preferred training-data container.
// The shard stores a small header plus an offset table, then concatenated per-history
// binary records, so a future loader can either scan sequentially or seek directly to
// one game by index.
void write_histories_binary_shard(const std::vector<GameHistory>& histories, const std::string& path);
std::vector<GameHistory> read_histories_binary_shard(const std::string& path);
GameHistory read_history_binary_shard_at(const std::string& path, size_t index);

// Cached reader for high-throughput replay / dataset loading.
// It keeps the shard file open and caches the offset table, avoiding per-sample
// reopen and offset-table parsing overhead.
class BinaryShardReader {
public:
    explicit BinaryShardReader(const std::string& path);
    ~BinaryShardReader();

    BinaryShardReader(const BinaryShardReader&) = delete;
    BinaryShardReader& operator=(const BinaryShardReader&) = delete;

    size_t history_count() const;
    const std::string& path() const;
    GameHistory read_at(size_t index);
    std::vector<GameHistory> read_all();

private:
    std::string path_;
    std::ifstream in_;
    std::vector<uint64_t> offsets_;
    uint64_t file_size_ = 0;
};

// Streaming shard writer for production self-play generation.
// Data is written to a temporary file in the destination directory and published
// only after the complete offset table and payload have been flushed successfully.
class BinaryShardWriter {
public:
    BinaryShardWriter(const std::string& path, size_t expected_history_count);
    ~BinaryShardWriter();

    BinaryShardWriter(const BinaryShardWriter&) = delete;
    BinaryShardWriter& operator=(const BinaryShardWriter&) = delete;

    void write_history(const GameHistory& history);
    void close();

    size_t expected_history_count() const { return expected_history_count_; }
    size_t written_count() const { return written_count_; }
    bool closed() const { return closed_; }

private:
    std::string path_;
    std::string temp_path_;
    size_t expected_history_count_ = 0;
    size_t written_count_ = 0;
    bool closed_ = true;
    std::ofstream out_;
    std::vector<uint64_t> offsets_;
};

enum class AsyncBinaryShardWriterState {
    Open,
    Closing,
    Closed,
    Failed,
};

// Asynchronous shard writer for actor-style self-play generation.
// Producers enqueue completed GameHistory objects into a bounded memory queue;
// a dedicated writer thread serializes and writes them to disk. This hides most
// disk latency from the self-play hot path while bounding memory use.
class AsyncBinaryShardWriter {
public:
    AsyncBinaryShardWriter(
        const std::string& path,
        size_t expected_history_count,
        size_t max_queue_size = 4
    );
    ~AsyncBinaryShardWriter();

    AsyncBinaryShardWriter(const AsyncBinaryShardWriter&) = delete;
    AsyncBinaryShardWriter& operator=(const AsyncBinaryShardWriter&) = delete;

    void write_history(GameHistory history);
    void close();

    size_t expected_history_count() const;
    size_t enqueued_count() const;
    size_t max_queue_size() const;
    AsyncBinaryShardWriterState state() const;
    bool closed() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tdmz
