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

    // Current generator indexes already contain an explicit boolean mode. Keep
    // accepting it while future producers migrate to the clearer string field.
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
        throw std::runtime_error("Training replay requires transition-indexed v2 shards");
    }
    if (replay_.replay_format_version() != kReplayBinaryFormatVersion) {
        throw std::runtime_error("Training replay format version is not current");
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
    history.wave_mode = wave_mode_;
    return history;
}

} // namespace tdmz
