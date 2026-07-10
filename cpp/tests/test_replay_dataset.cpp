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

static void write_test_index(const std::string& index_path, const std::vector<std::string>& shard_paths) {
    std::ofstream out(index_path);
    if (!out) throw std::runtime_error("failed to write test replay index");
    out << "{\n";
    out << "  \"format\": \"tdmz_selfplay_shard_index\",\n";
    out << "  \"version\": 1,\n";
    out << "  \"shards\": [\n";
    for (size_t i = 0; i < shard_paths.size(); ++i) {
        out << "    {\"worker\": " << i << ", \"path\": \"" << shard_paths[i] << "\", \"games\": 2}";
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
    write_test_index(index_path, {shard0_path, shard1_path});

    ReplayDataset dataset = ReplayDataset::from_index_json(index_path);
    CHECK_TRUE(dataset.shard_count() == 2);
    CHECK_TRUE(dataset.game_count() == 4);

    GameHistory g0 = dataset.read_game(0);
    GameHistory g3 = dataset.read_game(3);
    CHECK_TRUE(g0.seed == shard0[0].seed);
    CHECK_TRUE(g3.seed == shard1[1].seed);

    ReplayBatchSampler sampler(dataset, 1234567);
    ReplayBatch batch = sampler.sample_batch(8);
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

    std::ofstream out(index_path);
    CHECK_TRUE(out.good());
    out << "{\"shards\":[{\"path\":\"data.tdmzshd\"}]}\n";
    out.close();

    {
        ReplayDataset dataset = ReplayDataset::from_index_json(index_path.string());
        CHECK_TRUE(dataset.game_count() == 1);
        CHECK_TRUE(dataset.read_game(0).seed == 600);
    }

    std::filesystem::remove_all(dir);
}

int main() {
    try {
        test_replay_dataset_batch_sampler();
        test_zero_length_first_sample_does_not_hide_shape_mismatch();
        test_index_relative_shard_path();
        std::cout << "Replay dataset tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
