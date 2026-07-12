#include "tdmz/selfplay/training_replay.hpp"

#include "tdmz/core/compatibility.hpp"

#include <fstream>
#include <iterator>
#include <regex>
#include <stdexcept>
#include <utility>

namespace tdmz {

namespace {

WaveMode read_index_wave_mode(const std::string& index_path) {
    std::ifstream input(index_path);
    if (!input) {
        throw std::runtime_error("Failed to open replay index for wave-mode provenance: " + index_path);
    }
    const std::string text(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());

    const std::regex explicit_mode("\\\"wave_mode\\\"\\s*:\\s*\\\"(fixed|budgeted)\\\"");
    std::smatch match;
    if (std::regex_search(text, match, explicit_mode)) {
        return parse_wave_mode(match[1].str());
    }

    const std::regex budgeted_mode("\\\"budgeted\\\"\\s*:\\s*(true|false)");
    if (std::regex_search(text, match, budgeted_mode)) {
        return match[1].str() == "true" ? WaveMode::Budgeted : WaveMode::Fixed;
    }

    throw std::runtime_error(
        "Replay index is missing explicit wave-mode provenance ('wave_mode' or 'budgeted')");
}

} // namespace

TrainingReplayDataset::TrainingReplayDataset(
    std::vector<std::string> shard_paths,
    WaveMode wave_mode
) : replay_(std::move(shard_paths)), wave_mode_(wave_mode) {
    validate_current_training_contract();
}

TrainingReplayDataset::TrainingReplayDataset(
    ReplayDataset replay,
    WaveMode wave_mode
) : replay_(std::move(replay)), wave_mode_(wave_mode) {
    validate_current_training_contract();
}

TrainingReplayDataset TrainingReplayDataset::from_index_json(
    const std::string& index_path
) {
    const WaveMode mode = read_index_wave_mode(index_path);
    ReplayDataset replay = ReplayDataset::from_index_json(index_path);
    return TrainingReplayDataset(std::move(replay), mode);
}

void TrainingReplayDataset::validate_current_training_contract() const {
    if (!wave_mode_is_known(wave_mode_)) {
        throw std::runtime_error("Training replay requires an explicit fixed or budgeted wave mode");
    }
    if (!replay_.transition_indexed()) {
        throw std::runtime_error("Training replay requires transition-indexed v3 shards");
    }
    if (replay_.replay_format_version() != kReplayBinaryFormatVersion) {
        throw std::runtime_error("Training replay format version is not current v3");
    }
    if (replay_.observation_size() != kObservationSize) {
        throw std::runtime_error("Training replay observation_size is not the current 40x11x11 schema");
    }
    if (replay_.policy_size() != kActionSpaceSize) {
        throw std::runtime_error("Training replay policy_size is not the current 727-action schema");
    }
    if (replay_.legal_mask_size() != kActionSpaceSize) {
        throw std::runtime_error("Training replay legal_mask_size is not the current 727-action schema");
    }
}

GameHistory TrainingReplayDataset::read_game(size_t game_index) {
    GameHistory history = replay_.read_game(game_index);
    if (history.wave_mode != wave_mode_) {
        throw std::runtime_error(
            "Training replay shard wave mode disagrees with index/direct provenance");
    }
    if (history.terminal && history.truncated) {
        throw std::runtime_error("Training replay episode is both terminal and truncated");
    }
    if (!history.completed()) {
        throw std::runtime_error("Training replay episode is neither terminal nor truncated");
    }
    if (history.truncated && !history.bootstrap_state) {
        throw std::runtime_error(
            "Current v3 truncated training replay requires an explicit cutoff bootstrap state");
    }
    if (history.terminal && history.bootstrap_state) {
        throw std::runtime_error("Terminal training replay must not contain a cutoff bootstrap state");
    }
    return history;
}

} // namespace tdmz
