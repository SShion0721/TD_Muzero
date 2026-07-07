#pragma once
#include <string>
#include "tdmz/selfplay/game_history.hpp"

namespace tdmz {

void write_history_jsonl(const GameHistory& history, const std::string& path);
void write_history_summary_json(const GameHistory& history, const std::string& path);

} // namespace tdmz
