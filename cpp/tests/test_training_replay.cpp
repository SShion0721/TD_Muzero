#include "tdmz/core/compatibility.hpp"
#include "tdmz/selfplay/training_replay.hpp"
#include "tdmz/selfplay/transition_shard.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace tdmz;

namespace {

void check(bool condition, const char* expression, int line) {
    if (!condition) {
        throw std::runtime_error(
            std::string("check failed at line ") + std::to_string(line) + ": " + expression);
    }
}
#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

template <typename Fn>
void check_throws_contains(Fn&& function, const std::string& needle) {
    bool threw = false;
    try {
        function();
    } catch (const std::runtime_error& error) {
        threw = true;
        CHECK(std::string(error.what()).find(needle) != std::string::npos);
    }
    CHECK(threw);
}

GameHistory make_history(
    uint64_t seed,
    WaveMode mode,
    size_t observation_size = kObservationSize,
    size_t policy_size = kActionSpaceSize,
    size_t legal_size = kActionSpaceSize
) {
    GameHistory history;
    history.seed = seed;
    history.max_steps = 1;
    history.wave_mode = mode;

    TrajectoryStep step;
    step.step_index = 0;
    step.action = kFlatWaitOffset;
    step.observation.assign(observation_size, 0.0f);
    step.policy_target.assign(policy_size, 0.0f);
    if (policy_size > static_cast<size_t>(kFlatWaitOffset)) {
        step.policy_target[static_cast<size_t>(kFlatWaitOffset)] = 1.0f;
    }
    step.legal_mask.assign(legal_size, 0u);
    if (legal_size > static_cast<size_t>(kFlatWaitOffset)) {
        step.legal_mask[static_cast<size_t>(kFlatWaitOffset)] = 1u;
    }
    history.steps.push_back(std::move(step));
    return history;
}

void write_index(
    const std::string& path,
    const std::string& shard_path,
    const char* mode_field,
    const char* mode_value
) {
    const CompatibilityMetadata metadata = current_compatibility_metadata();
    std::ofstream out(path);
    if (!out) throw std::runtime_error("failed to write training replay test index");
    out << "{\n";
    out << "  \"format\": \"" << kReplayIndexFormat << "\",\n";
    out << "  \"version\": " << kReplayIndexVersion << ",\n";
    out << "  \"replay_format_version\": " << metadata.replay_format_version << ",\n";
    out << "  \"environment_rule_version\": " << metadata.environment_rule_version << ",\n";
    out << "  \"observation_schema_version\": " << metadata.observation_schema_version << ",\n";
    out << "  \"action_space_version\": " << metadata.action_space_version << ",\n";
    out << "  \"reward_transform_version\": " << metadata.reward_transform_version << ",\n";
    out << "  \"network_architecture_version\": " << metadata.network_architecture_version << ",\n";
    out << "  \"board_width\": " << metadata.board_width << ",\n";
    out << "  \"board_height\": " << metadata.board_height << ",\n";
    out << "  \"observation_channels\": " << metadata.observation_channels << ",\n";
    out << "  \"observation_size\": " << metadata.observation_size << ",\n";
    out << "  \"action_space_size\": " << metadata.action_space_size << ",\n";
    out << "  \"policy_size\": " << metadata.policy_size << ",\n";
    out << "  \"legal_mask_size\": " << metadata.legal_mask_size << ",\n";
    if (mode_field && mode_value) {
        out << "  \"" << mode_field << "\": " << mode_value << ",\n";
    }
    out << "  \"shards\": [{\"path\":\"" << shard_path << "\"}]\n";
    out << "}\n";
}

void test_direct_current_contract() {
    const std::string shard = "test_training_replay_direct.tdmzshd";
    write_histories_transition_shard(
        {make_history(1, WaveMode::Fixed)},
        shard);

    TrainingReplayDataset dataset({shard}, WaveMode::Fixed);
    CHECK(dataset.wave_mode() == WaveMode::Fixed);
    CHECK(dataset.game_count() == 1);
    CHECK(dataset.replay().observation_size() == kObservationSize);
    CHECK(dataset.read_game(0).wave_mode == WaveMode::Fixed);
}

void test_direct_requires_known_mode() {
    const std::string shard = "test_training_replay_unknown.tdmzshd";
    write_histories_transition_shard(
        {make_history(2, WaveMode::Fixed)},
        shard);
    check_throws_contains(
        [&] { TrainingReplayDataset dataset({shard}, WaveMode::Unknown); },
        "explicit");
}

void test_direct_rejects_old_observation_shape() {
    const std::string shard = "test_training_replay_old_shape.tdmzshd";
    write_histories_transition_shard(
        {make_history(3, WaveMode::Fixed, 20 * kBoardH * kBoardW)},
        shard);
    check_throws_contains(
        [&] { TrainingReplayDataset dataset({shard}, WaveMode::Fixed); },
        "observation_size");
}

void test_index_parses_boolean_mode() {
    const std::string shard = "test_training_replay_budgeted.tdmzshd";
    const std::string index = "test_training_replay_budgeted.index.json";
    write_histories_transition_shard(
        {make_history(4, WaveMode::Budgeted)},
        shard);
    write_index(index, shard, "budgeted", "true");

    TrainingReplayDataset dataset = TrainingReplayDataset::from_index_json(index);
    CHECK(dataset.wave_mode() == WaveMode::Budgeted);
    CHECK(dataset.read_game(0).wave_mode == WaveMode::Budgeted);
}

void test_index_parses_string_mode() {
    const std::string shard = "test_training_replay_fixed.tdmzshd";
    const std::string index = "test_training_replay_fixed.index.json";
    write_histories_transition_shard(
        {make_history(5, WaveMode::Fixed)},
        shard);
    write_index(index, shard, "wave_mode", "\"fixed\"");

    TrainingReplayDataset dataset = TrainingReplayDataset::from_index_json(index);
    CHECK(dataset.wave_mode() == WaveMode::Fixed);
}

void test_index_rejects_missing_mode() {
    const std::string shard = "test_training_replay_missing.tdmzshd";
    const std::string index = "test_training_replay_missing.index.json";
    write_histories_transition_shard(
        {make_history(6, WaveMode::Fixed)},
        shard);
    write_index(index, shard, nullptr, nullptr);

    check_throws_contains(
        [&] { (void)TrainingReplayDataset::from_index_json(index); },
        "wave-mode provenance");
}

} // namespace

int main() {
    try {
        test_direct_current_contract();
        test_direct_requires_known_mode();
        test_direct_rejects_old_observation_shape();
        test_index_parses_boolean_mode();
        test_index_parses_string_mode();
        test_index_rejects_missing_mode();
        std::cout << "Training replay tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
