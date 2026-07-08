#include "tdmz/mcts/dummy_network.hpp"
#include "tdmz/selfplay/selfplay_runner.hpp"
#include "tdmz/selfplay/trajectory_writer.hpp"
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tdmz;

static void check_true(bool ok, const char* expr, int line) {
    if (!ok) throw std::runtime_error(std::string("check failed at line ") + std::to_string(line) + ": " + expr);
}
#define CHECK_TRUE(x) check_true(static_cast<bool>(x), #x, __LINE__)

static void check_close(float a, float b, const char* label) {
    if (std::fabs(a - b) > 1e-6f) {
        throw std::runtime_error(std::string("float mismatch: ") + label);
    }
}

static void assert_history_equal(const GameHistory& a, const GameHistory& b) {
    CHECK_TRUE(a.seed == b.seed);
    CHECK_TRUE(a.max_steps == b.max_steps);
    CHECK_TRUE(a.terminal == b.terminal);
    check_close(a.total_reward, b.total_reward, "total_reward");
    CHECK_TRUE(a.steps.size() == b.steps.size());

    for (size_t i = 0; i < a.steps.size(); ++i) {
        const auto& x = a.steps[i];
        const auto& y = b.steps[i];
        CHECK_TRUE(x.step_index == y.step_index);
        CHECK_TRUE(x.action == y.action);
        check_close(x.reward, y.reward, "reward");
        check_close(x.root_value, y.root_value, "root_value");
        CHECK_TRUE(x.done == y.done);
        CHECK_TRUE(x.money == y.money);
        CHECK_TRUE(x.base_hp == y.base_hp);
        CHECK_TRUE(x.wave == y.wave);
        check_close(x.time, y.time, "time");
        CHECK_TRUE(x.observation.size() == y.observation.size());
        CHECK_TRUE(x.policy_target.size() == y.policy_target.size());
        CHECK_TRUE(x.legal_mask.size() == y.legal_mask.size());
        for (size_t j = 0; j < x.observation.size(); ++j) check_close(x.observation[j], y.observation[j], "observation");
        for (size_t j = 0; j < x.policy_target.size(); ++j) check_close(x.policy_target[j], y.policy_target[j], "policy_target");
        for (size_t j = 0; j < x.legal_mask.size(); ++j) CHECK_TRUE(x.legal_mask[j] == y.legal_mask[j]);
    }
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

void test_binary_roundtrip_selfplay() {
    GameHistory history = make_tiny_history(123, 12);

    const std::string path = "test_binary_trajectory.tdmzspb";
    write_history_binary(history, path);
    GameHistory loaded = read_history_binary(path);
    assert_history_equal(history, loaded);
}

void test_binary_roundtrip_empty_history() {
    GameHistory history;
    history.seed = 7;
    history.max_steps = 0;
    history.terminal = false;
    history.total_reward = 0.0f;

    const std::string path = "test_binary_empty.tdmzspb";
    write_history_binary(history, path);
    GameHistory loaded = read_history_binary(path);
    assert_history_equal(history, loaded);
}

void test_binary_shard_roundtrip() {
    std::vector<GameHistory> histories;
    histories.push_back(make_tiny_history(200, 8));
    histories.push_back(make_tiny_history(201, 9));
    histories.push_back(make_tiny_history(202, 10));

    const std::string path = "test_binary_shard.tdmzshd";
    write_histories_binary_shard(histories, path);

    auto loaded = read_histories_binary_shard(path);
    CHECK_TRUE(loaded.size() == histories.size());
    for (size_t i = 0; i < histories.size(); ++i) {
        assert_history_equal(histories[i], loaded[i]);
    }

    GameHistory loaded_one = read_history_binary_shard_at(path, 1);
    assert_history_equal(histories[1], loaded_one);
}

void test_async_binary_shard_roundtrip() {
    std::vector<GameHistory> histories;
    histories.push_back(make_tiny_history(300, 8));
    histories.push_back(make_tiny_history(301, 9));
    histories.push_back(make_tiny_history(302, 10));
    histories.push_back(make_tiny_history(303, 11));

    const std::string path = "test_async_binary_shard.tdmzshd";
    {
        // Queue size 1 forces producer/writer backpressure in the test.
        AsyncBinaryShardWriter writer(path, histories.size(), 1);
        for (auto& history : histories) {
            writer.write_history(history);
        }
        CHECK_TRUE(writer.enqueued_count() == histories.size());
        writer.close();
        CHECK_TRUE(writer.closed());
    }

    auto loaded = read_histories_binary_shard(path);
    CHECK_TRUE(loaded.size() == histories.size());
    for (size_t i = 0; i < histories.size(); ++i) {
        assert_history_equal(histories[i], loaded[i]);
    }

    GameHistory loaded_one = read_history_binary_shard_at(path, 2);
    assert_history_equal(histories[2], loaded_one);
}

int main() {
    try {
        test_binary_roundtrip_selfplay();
        test_binary_roundtrip_empty_history();
        test_binary_shard_roundtrip();
        test_async_binary_shard_roundtrip();
        std::cout << "Binary trajectory tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
