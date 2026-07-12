#pragma once

#include "tdmz/core/wave_mode.hpp"
#include "tdmz/selfplay/replay_dataset.hpp"

#include <string>
#include <vector>

namespace tdmz {

// Training-facing replay wrapper. Unlike the generic diagnostic readers, this
// class requires one explicit wave mode and the exact current 40/727/727 tensor
// contract before exposing data to a trainer.
class TrainingReplayDataset {
public:
    TrainingReplayDataset(std::vector<std::string> shard_paths, WaveMode wave_mode);

    static TrainingReplayDataset from_index_json(const std::string& index_path);

    ReplayDataset& replay() { return replay_; }
    const ReplayDataset& replay() const { return replay_; }
    WaveMode wave_mode() const { return wave_mode_; }

    size_t shard_count() const { return replay_.shard_count(); }
    size_t game_count() const { return replay_.game_count(); }
    GameHistory read_game(size_t game_index);

private:
    TrainingReplayDataset(ReplayDataset replay, WaveMode wave_mode);
    void validate_current_training_contract() const;

    ReplayDataset replay_;
    WaveMode wave_mode_ = WaveMode::Unknown;
};

} // namespace tdmz
