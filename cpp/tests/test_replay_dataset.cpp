#include "tdmz/core/compatibility.hpp"
#include "tdmz/mcts/dummy_network.hpp"
#include "tdmz/selfplay/replay_dataset.hpp"
#include "tdmz/selfplay/selfplay_runner.hpp"
#include "tdmz/selfplay/trajectory_writer.hpp"
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tdmz;

static void check_true(bool ok, const char* expr, int line) {
    if (!ok) throw std::runtime_error(std::string("check failed at line ") + std::to_string(line) + ": " + expr);
}
#define CHECK_TRUE(x) check_true(static_cast<bool>(x), #x, __LINE__)

template <typename Fn>
static void check_throws_contains(Fn&& fn, const std::string& needle) {
    bool threw = false;
    try {
        fn();
    } catch (const std::runtime_error& e) {
        threw = true;
        CHECK_TRUE(std::string(e.what()).find(needle) != std::string::npos);
    }
    CHECK_TRUE(threw);
}

static GameHistory make_tiny_history(uint64_t seed, int max_steps) {
    SelfPlayConfig cfg;
    cfg.seed = seed;
    cfg.max_steps = max_steps;
    cfg.mcts.num_simulations = 8;
    cfg.mcts.latent_top_k = 8;
    cfg.mcts.max_nodes = 1024;

    DummyNetwork net;
    SelfPlayRunner runner(cfg);
    GameHistory history = runner.run(net);
    CHECK_TRUE(!history.steps.empty());
    return history;
}

static GameHistory make_single_step_history(uint64_t seed, size_t tensor_size) {
    GameHistory history;
    history.seed = seed;
    history.max_steps = 1;

    TrajectoryStep step;
    step.action = kFlatWaitOffset;
    step.observation.assign(tensor_size, 1.0f);
    step.policy_target.assign(tensor_size, tensor_size == 0 ? 0.0f : 1.0f / static_cast<float>(tensor_size));
    step.legal_mask.assign(tensor_size, 1u);
    history.steps.push_back(std::move(step));
    return history;
}

static void write_current_index(
    const std::string& index_path,
    const std::vector<std::string>& shard_paths,
    int observation_size = kObservationSize,
    uint32_t environment_rule_version = kEnvironmentRuleVersion,
    uint32_t index_version = kReplayIndexVersion
) {
    const CompatibilityMetadata metadata = current_compatibility_metadata();
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
    out << "  \"shards\": [\n";
    for (size_t i = 0; i < shard_paths.size(); ++i) {
        out << "    {\"worker\": " << i << ", \"path\": \"" << shard_paths[i] << "\", \"games\": 2}";
        out << (i + 1 == shard_paths.size() ? "" : ",") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

static void write_legacy_index(
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

void test_replay_dataset_batch_sampler() {
    std::vector<GameHistory> shard0;
    shard0.push_back(make_tiny_history(400, 8));
    shard0.push_back(make_tiny_history(401, 9));
    std::vector<GameHistory> shard1;
    shard1.push_back(make_tiny_history(402, 10));
    shard1.push_back(make_tiny_history(403, 11));

    const std::string shard0_path = "test_replay_dataset_w0.tdmzshd";
    const std::string shard1_path = "test_replay_dataset_w1.tdmzshd";
    const std::string index_path = "test_replay_dataset.index.json";
    write_histories_binary_shard(shard0, shard0_path);
    write_histories_binary_shard(shard1, shard1_path);
    write_current_index(index_path, {shard0_path, shard1_path});

    ReplayDataset dataset = ReplayDataset::from_index_json(index_path);
    CHECK_TRUE(dataset.shard_count() == 2);
    CHECK_TRUE(dataset.game_count() == 4);
    CHECK_TRUE(dataset.has_index_metadata());
    CHECK_TRUE(!dataset.index_metadata().legacy_v1);
    CHECK_TRUE(dataset.index_metadata().index_version == kReplayIndexVersion);

    GameHistory g0 = dataset.read_game(0);
    GameHistory g3 = dataset.read_game(3);
    CHECK_TRUE(g0.seed == shard0[0].seed);
    CHECK_TRUE(g3.seed == shard1[1].seed);

    dataset.reset_game_read_count();
    ReplayBatchSampler sampler(dataset, 1234567);
    ReplayBatch batch = sampler.sample_batch(8);
    CHECK_TRUE(dataset.game_read_count() > 0);
    CHECK_TRUE(dataset.game_read_count() <= dataset.game_count());
    CHECK_TRUE(dataset.game_read_count() < static_cast<uint64_t>(batch.batch_size));

    CHECK_TRUE(batch.batch_size == 8);
    CHECK_TRUE(batch.observation_size > 0);
    CHECK_TRUE(batch.policy_size > 0);
    CHECK_TRUE(batch.legal_mask_size > 0);
    CHECK_TRUE(batch.observations.size() == static_cast<size_t>(batch.batch_size) * batch.observation_size);
    CHECK_TRUE(batch.policy_targets.size() == static_cast<size_t>(batch.batch_size) * batch.policy_size);
    CHECK_TRUE(batch.legal_masks.size() == static_cast<size_t>(batch.batch_size) * batch.legal_mask_size);
    CHECK_TRUE(batch.values.size() == static_cast<size_t>(batch.batch_size));
    CHECK_TRUE(batch.rewards.size() == static_cast<size_t>(batch.batch_size));
    CHECK_TRUE(batch.actions.size() == static_cast<size_t>(batch.batch_size));
    CHECK_TRUE(batch.dones.size() == static_cast<size_t>(batch.batch_size));
    CHECK_TRUE(replay_batch_payload_bytes(batch) > 0);
    CHECK_TRUE(std::isfinite(replay_batch_checksum(batch)));
}

void test_zero_length_first_sample_does_not_hide_shape_mismatch() {
    const std::string shard_path = "test_replay_zero_shape.tdmzshd";
    write_histories_binary_shard(
        {make_single_step_history(500, 0), make_single_step_history(501, 4)},
        shard_path
    );

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
    const std::filesystem::path dir = "test_replay_relative_dir";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    const std::filesystem::path shard_path = dir / "data.tdmzshd";
    const std::filesystem::path index_path = dir / "index.json";
    write_histories_binary_shard({make_single_step_history(600, 3)}, shard_path.string());
    write_current_index(index_path.string(), {"data.tdmzshd"});

    {
        ReplayDataset dataset = ReplayDataset::from_index_json(index_path.string());
        CHECK_TRUE(dataset.game_count() == 1);
        CHECK_TRUE(dataset.read_game(0).seed == 600);
    }

    std::filesystem::remove_all(dir);
}

void test_legacy_index_requires_explicit_policy() {
    const std::string shard_path = "test_replay_legacy.tdmzshd";
    const std::string index_path = "test_replay_legacy.index.json";
    write_histories_binary_shard({make_single_step_history(700, 3)}, shard_path);
    write_legacy_index(index_path, {shard_path}, true);

    check_throws_contains(
        [&] { (void)ReplayDataset::from_index_json(index_path); },
        "legacy");

    ReplayDataset dataset = ReplayDataset::from_index_json(
        index_path,
        LegacyReplayIndexPolicy::AllowV1);
    CHECK_TRUE(dataset.index_metadata().legacy_v1);
    CHECK_TRUE(dataset.game_count() == 1);
}

void test_unversioned_index_requires_explicit_policy() {
    const std::string shard_path = "test_replay_unversioned.tdmzshd";
    const std::string index_path = "test_replay_unversioned.index.json";
    write_histories_binary_shard({make_single_step_history(701, 3)}, shard_path);
    write_legacy_index(index_path, {shard_path}, false);

    check_throws_contains(
        [&] { (void)ReplayDataset::from_index_json(index_path); },
        "format/version");

    ReplayDataset dataset = ReplayDataset::from_index_json(
        index_path,
        LegacyReplayIndexPolicy::AllowV1);
    CHECK_TRUE(dataset.index_metadata().legacy_v1);
}

void test_index_rejects_compatibility_mismatches() {
    const std::string shard_path = "test_replay_mismatch.tdmzshd";
    write_histories_binary_shard({make_single_step_history(800, 3)}, shard_path);

    const std::string observation_index = "test_replay_bad_observation.index.json";
    write_current_index(observation_index, {shard_path}, kObservationSize + 1);
    check_throws_contains(
        [&] { (void)ReplayDataset::from_index_json(observation_index); },
        "observation_size");

    const std::string environment_index = "test_replay_bad_environment.index.json";
    write_current_index(
        environment_index,
        {shard_path},
        kObservationSize,
        kEnvironmentRuleVersion + 1);
    check_throws_contains(
        [&] { (void)ReplayDataset::from_index_json(environment_index); },
        "environment_rule_version");
}

void test_index_rejects_future_version() {
    const std::string shard_path = "test_replay_future.tdmzshd";
    const std::string index_path = "test_replay_future.index.json";
    write_histories_binary_shard({make_single_step_history(900, 3)}, shard_path);
    write_current_index(
        index_path,
        {shard_path},
        kObservationSize,
        kEnvironmentRuleVersion,
        kReplayIndexVersion + 1);

    check_throws_contains(
        [&] { (void)ReplayDataset::from_index_json(index_path); },
        "version");
}

int main() {
    try {
        test_replay_dataset_batch_sampler();
        test_zero_length_first_sample_does_not_hide_shape_mismatch();
        test_index_relative_shard_path();
        test_legacy_index_requires_explicit_policy();
        test_unversioned_index_requires_explicit_policy();
        test_index_rejects_compatibility_mismatches();
        test_index_rejects_future_version();
        std::cout << "Replay dataset tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
