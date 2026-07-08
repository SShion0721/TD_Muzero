#pragma once
#include <fstream>
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

// Streaming shard writer for production self-play generation.
// This avoids keeping all generated games in RAM before writing a shard.
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
    size_t expected_history_count_ = 0;
    size_t written_count_ = 0;
    bool closed_ = true;
    std::ofstream out_;
    std::vector<uint64_t> offsets_;
};

} // namespace tdmz
