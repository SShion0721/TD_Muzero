#pragma once
#include <string>
#include "tdmz/selfplay/game_history.hpp"

namespace tdmz {

void write_history_jsonl(const GameHistory& history, const std::string& path);
void write_history_summary_json(const GameHistory& history, const std::string& path);

// Binary format intended for high-throughput training-data generation.
// File layout is POD headers plus contiguous float/byte arrays:
// header, then per-step fixed metadata, observation floats, policy floats, legal-mask bytes.
void write_history_binary(const GameHistory& history, const std::string& path);
GameHistory read_history_binary(const std::string& path);

} // namespace tdmz
