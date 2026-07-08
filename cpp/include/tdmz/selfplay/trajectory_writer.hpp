#pragma once
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

} // namespace tdmz
