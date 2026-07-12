#include "tdmz/core/compatibility.hpp"
#include "tdmz/mcts/dummy_network.hpp"
#include "tdmz/selfplay/replay_dataset.hpp"
#include "tdmz/selfplay/selfplay_runner.hpp"
#include "tdmz/selfplay/trajectory_writer.hpp"
#include "tdmz/selfplay/transition_shard.hpp"

#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tdmz;

namespace {

void check_true(bool condition, const char* expression, int line) {
    if (!condition) {
        throw std::runtime_error(
            std::string("check failed at line ") + std::to_string(line) + ": " + expression);
    }
}
#define CHECK_TRUE(expression) check_true(static_cast<bool>(expression), #expression, __LINE__)

template <typename Fn>
void check_throws_contains(Fn&& function, const std::string& needle) {
    bool threw = false;
    try {
        function();
    } catch (const std::runtime_error& error) {
        threw = true;
        CHECK_TRUE(std::string(error.what()).find(needle) != std::string::npos);
    }
    CHECK_TRUE(threw);
}

GameHistory make_tiny_history(uint64_t seed, int max_steps) {
    SelfPlayConfig config;
    config.seed = seed;
    config.max_steps = max_steps;
    config.save_bootstrap_state = true;
    config.mcts.num_simulations = 8;
    config.mcts.latent_top_k = 8;
    config.mcts.max_nodes = 1024;

    DummyNetwork network;
    SelfPlayRunner runner(config);
    GameHistory history = runner.run(network);
    CHECK_TRUE(!history.steps.empty());
    CHECK_TRUE(history.completed());
    if (history.truncated) CHECK_TRUE(history.bootstrap_state.has_value());
    return history;
}

GameHistory make_single_step_history(uint64_t seed, size_t tensor_size) {
    GameHistory history;
    history.seed = seed;
    history.max_steps = 1;
    history.truncated = true;
    history.wave_mode = WaveMode::Fixed;

    TrajectoryStep step;
    step.action = kFlatWaitOffset;
    step.observation.assign(tensor_size, 1.0f);
    step.policy_target.assign(
        tensor_size,
        tensor_size == 0 ? 0.0f : 1.0f / static_cast<float>(tensor_size));
    step.legal_mask.assign(tensor_size, 1u);
    history.steps.push_back(step);

    BootstrapState bootstrap;
    bootstrap.root_value = 0.0f;
    bootstrap.observation = step.observation;
    bootstrap.policy_target = step.policy_target;
    bootstrap.legal_mask = step.legal_mask;
    history.bootstrap_state = std::move(bootstrap);
    return history;
}

void write_current_index(
    const std::string& index_path,
    const std::vector<std::string>& shard_paths,
    uint32_t replay_format_version,
    int observation_size = kObservationSize,
    uint32_t environment_rule_version = kEnvironmentRuleVersion,
    uint32_t index_version = kReplayIndexVersion
) {
    CompatibilityMetadata metadata = current_compatibility_metadata();
    metadata.replay_format_version = replay_format_version;
    std::ofstream out(index_path);
    if (!out) throw std::runtime_error("failed to write test replay index");
    out << "{\n";
    out << "  \"format\": \"" << kReplayIndexFormat << "\",\n";
    out << "  \"version\": " << index_version << ",\n";
    out << "  \"replay_format_version\": " << metadata.replay_format_version << ",\n";
    out << "  \"environment_rule_version\": " << environment_rule_version << ",\n";
    out << "  \"observation_schema_version\": " << metadata.observation_schema_version << ",\n";
    out << "  \"action_space_version\": " << metadata.action_space_version << ",\n";
    out << "  \"reward_transform_version\": " << metadata.reward_transform_version << ",\n";
    out << "  \"network_architecture_version\": " << metadata.network_architecture_version << ",\n";
    out << "  \"board_width\": " << metadata.board_width << ",\n";
    out << "  \"board_height\": " << metadata.board_height << ",\n";
    out << "  \"observation_channels\": " << metadata.observation_channels << ",\n";
    out << "  \"observation_size\": " << observation_size << ",\n";
    out << "  \"action_space_size\": " << metadata.action_space_size << ",\n";
    out << "  \"policy_size\": " << metadata.policy_size << ",\n";
    out << "  \"legal_mask_size\": " << metadata.legal_mask_size << ",\n";
    out << "  \"sampling_distribution\": \"uniform_game_then_uniform_step\",\n";
    out << "  \"shards\": [\n";
    for (size_t i = 0; i < shard_paths.size(); ++i) {
        out << "    {\"worker\": " << i
            << ", \"path\": \"" << shard_paths[i] << "\", \"games\": 2}";
        out << (i + 1 == shard_paths.size() ? "" : ",") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

void write_legacy_index(
    const std::string& index_path,
    const std::vector<std::string>& shard_paths,
    bool include_format_and_version
) {
    std::ofstream out(index_path);
    if (!out) throw std::runtime_error("failed to write legacy replay index");
    out << "{\n";
    if (include_format_and_version) {
        out << "  \"format\": \"tdmz_selfplay_shard_index\",\n";
        out << "  \"version\": 1,\n";
    }
    out << "  \"shards\": [\n";
    for (size_t i = 0; i < shard_paths.size(); ++i) {
        out << "    {\"path\":\"" << shard_paths[i] << "\"}";
        out << (i + 1 == shard_paths.size() ? "" : ",") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

void check_batch_equal(const ReplayBatch& left, const ReplayBatch& right) {
    CHECK_TRUE(left.batch_size == right.batch_size);
    CHECK_TRUE(left.observation_size == right.observation_size);
    CHECK_TRUE(left.policy_size == right.policy_size);
    CHECK_TRUE(left.legal_mask_size == right.legal_mask_size);
    CHECK_TRUE(left.observations == right.observations);
    CHECK_TRUE(left.policy_targets == right.policy_targets);
    CHECK_TRUE(left.values == right.values);
    CHECK_TRUE(left.rewards == right.rewards);
    CHECK_TRUE(left.actions == right.actions);
    CHECK_TRUE(left.legal_masks == right.legal_masks);
    CHECK_TRUE(left.dones == right.dones);
}

std::vector<GameHistory> make_histories() {
    return {
        make_tiny_history(400, 8),
        make_tiny_history(401, 9),
        make_tiny_history(402, 10),
        make_tiny_history(403, 11),
    };
}

void test_v1_replay_dataset_batch_sampler() {
    const std::vector<GameHistory> histories = make_histories();
    const std::string shard0_path = "test_replay_v1_w0.tdmzshd";
    const std::string shard1_path = "test_replay_v1_w1.tdmzshd";
    const std::string index_path = "test_replay_v1.index.json";
    write_histories_binary_shard({histories[0], histories[1]}, shard0_path);
    write_histories_binary_shard({histories[2], histories[3]}, shard1_path);
    write_current_index(
        index_path,
        {shard0_path, shard1_path},
        kLegacyReplayBinaryFormatVersion);

    ReplayDataset dataset = ReplayDataset::from_index_json(index_path);
    CHECK_TRUE(dataset.shard_count() == 2);
    CHECK_TRUE(dataset.game_count() == 4);
    CHECK_TRUE(dataset.replay_format_version() == kLegacyReplayBinaryFormatVersion);
    CHECK_TRUE(!dataset.transition_indexed());

    dataset.reset_game_read_count();
    ReplayBatchSampler sampler(dataset, 1234567);
    ReplayBatch batch = sampler.sample_batch(8);
    CHECK_TRUE(dataset.game_read_count() > 0);
    CHECK_TRUE(dataset.game_read_count() <= dataset.game_count());
    CHECK_TRUE(dataset.game_read_count() < static_cast<uint64_t>(batch.batch_size));
    CHECK_TRUE(!batch.direct_transition_reads);
    CHECK_TRUE(!batch.io_stats_exact);
    CHECK_TRUE(batch.unique_games_touched > 0);
    CHECK_TRUE(batch.unique_shards_touched > 0);
    CHECK_TRUE(replay_batch_payload_bytes(batch) > 0);
    CHECK_TRUE(std::isfinite(replay_batch_checksum(batch)));
}

void test_v2_direct_batch_sampler() {
    const std::vector<GameHistory> histories = make_histories();
    const std::string shard0_path = "test_replay_v2_w0.tdmzshd";
    const std::string shard1_path = "test_replay_v2_w1.tdmzshd";
    const std::string index_path = "test_replay_v2.index.json";
    write_histories_transition_shard({histories[0], histories[1]}, shard0_path);
    write_histories_transition_shard({histories[2], histories[3]}, shard1_path);
    write_current_index(
        index_path,
        {shard0_path, shard1_path},
        kReplayBinaryFormatVersion);

    ReplayDataset dataset = ReplayDataset::from_index_json(index_path);
    CHECK_TRUE(dataset.transition_indexed());
    CHECK_TRUE(dataset.replay_format_version() == kReplayBinaryFormatVersion);
    CHECK_TRUE(dataset.observation_size() == kObservationSize);
    CHECK_TRUE(dataset.policy_size() == kActionSpaceSize);
    CHECK_TRUE(dataset.legal_mask_size() == kActionSpaceSize);

    dataset.reset_game_read_count();
    dataset.reset_io_stats();
    ReplayBatchSampler sampler(dataset, 1234567);
    ReplayBatch batch = sampler.sample_batch(64);
    CHECK_TRUE(dataset.game_read_count() == 0);
    CHECK_TRUE(batch.direct_transition_reads);
    CHECK_TRUE(batch.io_stats_exact);
    CHECK_TRUE(batch.physical_read_operations > 0);
    CHECK_TRUE(batch.physical_read_operations <= static_cast<uint64_t>(batch.batch_size));
    CHECK_TRUE(batch.physical_bytes_read > 0);
    CHECK_TRUE(batch.unique_games_touched > 0);
    CHECK_TRUE(batch.unique_shards_touched == 2);
    CHECK_TRUE(batch.observations.size()
        == static_cast<size_t>(batch.batch_size) * batch.observation_size);
    CHECK_TRUE(replay_batch_payload_bytes(batch) > batch.physical_bytes_read);
}

void test_v1_v2_sampling_distribution_parity() {
    const std::vector<GameHistory> histories = make_histories();
    const std::string v1_path = "test_replay_parity_v1.tdmzshd";
    const std::string v2_path = "test_replay_parity_v2.tdmzshd";
    const std::string v1_index = "test_replay_parity_v1.index.json";
    const std::string v2_index = "test_replay_parity_v2.index.json";
    write_histories_binary_shard(histories, v1_path);
    write_histories_transition_shard(histories, v2_path);
    write_current_index(v1_index, {v1_path}, kLegacyReplayBinaryFormatVersion);
    write_current_index(v2_index, {v2_path}, kReplayBinaryFormatVersion);

    ReplayDataset v1_dataset = ReplayDataset::from_index_json(v1_index);
    ReplayDataset v2_dataset = ReplayDataset::from_index_json(v2_index);
    ReplayBatchSampler v1_sampler(v1_dataset, 987654321);
    ReplayBatchSampler v2_sampler(v2_dataset, 987654321);
    ReplayBatch v1_batch = v1_sampler.sample_batch(128);
    ReplayBatch v2_batch = v2_sampler.sample_batch(128);
    check_batch_equal(v1_batch, v2_batch);
    CHECK_TRUE(replay_batch_checksum(v1_batch) == replay_batch_checksum(v2_batch));
}

void test_zero_length_first_sample_does_not_hide_shape_mismatch() {
    const std::string shard_path = "test_replay_zero_shape.tdmzshd";
    write_histories_binary_shard(
        {make_single_step_history(500, 0), make_single_step_history(501, 4)},
        shard_path);

    ReplayDataset dataset({shard_path});
    ReplayBatchSampler sampler(dataset, 2);
    bool threw = false;
    try {
        (void)sampler.sample_batch(2);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK_TRUE(threw);
}

void test_index_relative_shard_path() {
    const std::filesystem::path directory = "test_replay_relative_dir";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    const std::filesystem::path shard_path = directory / "data.tdmzshd";
    const std::filesystem::path index_path = directory / "index.json";
    GameHistory history = make_tiny_history(600, 8);
    write_histories_transition_shard({history}, shard_path.string());
    write_current_index(
        index_path.string(),
        {"data.tdmzshd"},
        kReplayBinaryFormatVersion);

    {
        ReplayDataset dataset = ReplayDataset::from_index_json(index_path.string());
        CHECK_TRUE(dataset.game_count() == 1);
        CHECK_TRUE(dataset.read_game(0).seed == 600);
    }
    std::filesystem::remove_all(directory);
}

void test_legacy_indexes_require_explicit_policy() {
    const std::string shard_path = "test_replay_legacy.tdmzshd";
    write_histories_binary_shard({make_single_step_history(700, 3)}, shard_path);

    const std::string versioned = "test_replay_legacy.index.json";
    write_legacy_index(versioned, {shard_path}, true);
    check_throws_contains(
        [&] { (void)ReplayDataset::from_index_json(versioned); },
        "legacy");
    ReplayDataset versioned_dataset = ReplayDataset::from_index_json(
        versioned,
        LegacyReplayIndexPolicy::AllowV1);
    CHECK_TRUE(versioned_dataset.index_metadata().legacy_v1);
    CHECK_TRUE(versioned_dataset.replay_format_version() == kLegacyReplayBinaryFormatVersion);

    const std::string unversioned = "test_replay_unversioned.index.json";
    write_legacy_index(unversioned, {shard_path}, false);
    check_throws_contains(
        [&] { (void)ReplayDataset::from_index_json(unversioned); },
        "format/version");
    ReplayDataset unversioned_dataset = ReplayDataset::from_index_json(
        unversioned,
        LegacyReplayIndexPolicy::AllowV1);
    CHECK_TRUE(unversioned_dataset.index_metadata().legacy_v1);
}

void test_index_rejects_compatibility_mismatches() {
    const std::string shard_path = "test_replay_mismatch.tdmzshd";
    write_histories_transition_shard({make_tiny_history(800, 8)}, shard_path);

    const std::string observation_index = "test_replay_bad_observation.index.json";
    write_current_index(
        observation_index,
        {shard_path},
        kReplayBinaryFormatVersion,
        kObservationSize + 1);
    check_throws_contains(
        [&] { (void)ReplayDataset::from_index_json(observation_index); },
        "observation_size");

    const std::string environment_index = "test_replay_bad_environment.index.json";
    write_current_index(
        environment_index,
        {shard_path},
        kReplayBinaryFormatVersion,
        kObservationSize,
        kEnvironmentRuleVersion + 1);
    check_throws_contains(
        [&] { (void)ReplayDataset::from_index_json(environment_index); },
        "environment_rule_version");
}

void test_index_rejects_future_version() {
    const std::string shard_path = "test_replay_future.tdmzshd";
    const std::string index_path = "test_replay_future.index.json";
    write_histories_transition_shard({make_tiny_history(900, 8)}, shard_path);
    write_current_index(
        index_path,
        {shard_path},
        kReplayBinaryFormatVersion,
        kObservationSize,
        kEnvironmentRuleVersion,
        kReplayIndexVersion + 1);
    check_throws_contains(
        [&] { (void)ReplayDataset::from_index_json(index_path); },
        "version");
}

void test_rejects_mixed_and_mismatched_shard_formats() {
    const std::vector<GameHistory> histories = make_histories();
    const std::string v1_path = "test_replay_mixed_v1.tdmzshd";
    const std::string v2_path = "test_replay_mixed_v2.tdmzshd";
    write_histories_binary_shard({histories[0]}, v1_path);
    write_histories_transition_shard({histories[1]}, v2_path);

    check_throws_contains(
        [&] { ReplayDataset dataset({v1_path, v2_path}); },
        "mixed");

    const std::string bad_index = "test_replay_format_mismatch.index.json";
    write_current_index(bad_index, {v1_path}, kReplayBinaryFormatVersion);
    check_throws_contains(
        [&] { (void)ReplayDataset::from_index_json(bad_index); },
        "format mismatch");
}

} // namespace

int main() {
    try {
        test_v1_replay_dataset_batch_sampler();
        test_v2_direct_batch_sampler();
        test_v1_v2_sampling_distribution_parity();
        test_zero_length_first_sample_does_not_hide_shape_mismatch();
        test_index_relative_shard_path();
        test_legacy_indexes_require_explicit_policy();
        test_index_rejects_compatibility_mismatches();
        test_index_rejects_future_version();
        test_rejects_mixed_and_mismatched_shard_formats();
        std::cout << "Replay dataset tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
