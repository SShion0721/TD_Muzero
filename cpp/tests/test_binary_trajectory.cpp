#include "tdmz/mcts/dummy_network.hpp"
#include "tdmz/selfplay/selfplay_runner.hpp"
#include "tdmz/selfplay/trajectory_writer.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace tdmz;
namespace fs = std::filesystem;

static void check_true(bool ok, const char* expr, int line) {
    if (!ok) throw std::runtime_error(std::string("check failed at line ") + std::to_string(line) + ": " + expr);
}
#define CHECK_TRUE(x) check_true(static_cast<bool>(x), #x, __LINE__)

static void check_close(float a, float b, const char* label) {
    if (std::fabs(a - b) > 1e-6f) {
        throw std::runtime_error(std::string("float mismatch: ") + label);
    }
}

template <typename Fn>
static void check_throws(Fn&& fn, const char* label) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception&) {
        threw = true;
    }
    if (!threw) throw std::runtime_error(std::string("expected exception: ") + label);
}

static void remove_if_exists(const std::string& path) {
    std::error_code ec;
    fs::remove(path, ec);
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

template <typename T>
static void overwrite_scalar(const std::string& path, std::streamoff offset, const T& value) {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!file) throw std::runtime_error("failed to open corruption target");
    file.seekp(offset, std::ios::beg);
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!file) throw std::runtime_error("failed to corrupt binary test file");
}

void test_binary_roundtrip_selfplay() {
    GameHistory history = make_tiny_history(123, 12);

    const std::string path = "test_binary_trajectory.tdmzspb";
    remove_if_exists(path);
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
    remove_if_exists(path);
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
    remove_if_exists(path);
    write_histories_binary_shard(histories, path);

    auto loaded = read_histories_binary_shard(path);
    CHECK_TRUE(loaded.size() == histories.size());
    for (size_t i = 0; i < histories.size(); ++i) {
        assert_history_equal(histories[i], loaded[i]);
    }

    GameHistory loaded_one = read_history_binary_shard_at(path, 1);
    assert_history_equal(histories[1], loaded_one);

    BinaryShardReader reader(path);
    CHECK_TRUE(reader.history_count() == histories.size());
    CHECK_TRUE(reader.path() == path);
    assert_history_equal(histories[0], reader.read_at(0));
    assert_history_equal(histories[2], reader.read_at(2));
    auto cached_loaded = reader.read_all();
    CHECK_TRUE(cached_loaded.size() == histories.size());
    for (size_t i = 0; i < histories.size(); ++i) {
        assert_history_equal(histories[i], cached_loaded[i]);
    }
}

void test_atomic_shard_publication() {
    const std::string path = "test_atomic_publish.tdmzshd";
    remove_if_exists(path);
    GameHistory history = make_tiny_history(250, 8);

    BinaryShardWriter writer(path, 1);
    CHECK_TRUE(!fs::exists(path));
    writer.write_history(history);
    CHECK_TRUE(!fs::exists(path));
    writer.close();
    CHECK_TRUE(fs::exists(path));
    assert_history_equal(history, read_history_binary_shard_at(path, 0));
}

void test_async_binary_shard_roundtrip_and_concurrent_close() {
    std::vector<GameHistory> histories;
    histories.push_back(make_tiny_history(300, 8));
    histories.push_back(make_tiny_history(301, 9));
    histories.push_back(make_tiny_history(302, 10));
    histories.push_back(make_tiny_history(303, 11));

    const std::string path = "test_async_binary_shard.tdmzshd";
    remove_if_exists(path);
    {
        AsyncBinaryShardWriter writer(path, histories.size(), 1);
        for (auto& history : histories) writer.write_history(history);
        CHECK_TRUE(writer.enqueued_count() == histories.size());

        std::exception_ptr first_error;
        std::exception_ptr second_error;
        std::thread first([&] {
            try { writer.close(); } catch (...) { first_error = std::current_exception(); }
        });
        std::thread second([&] {
            try { writer.close(); } catch (...) { second_error = std::current_exception(); }
        });
        first.join();
        second.join();
        CHECK_TRUE(!first_error);
        CHECK_TRUE(!second_error);
        CHECK_TRUE(writer.state() == AsyncBinaryShardWriterState::Closed);
        CHECK_TRUE(writer.closed());
    }

    auto loaded = read_histories_binary_shard(path);
    CHECK_TRUE(loaded.size() == histories.size());
    for (size_t i = 0; i < histories.size(); ++i) assert_history_equal(histories[i], loaded[i]);
}

void test_async_rejects_write_after_closing_begins() {
    const std::string path = "test_async_closing_reject.tdmzshd";
    remove_if_exists(path);
    GameHistory first = make_tiny_history(350, 8);
    GameHistory second = make_tiny_history(351, 8);

    AsyncBinaryShardWriter writer(path, 2, 1);
    writer.write_history(first);
    std::exception_ptr close_error;
    std::thread closer([&] {
        try { writer.close(); } catch (...) { close_error = std::current_exception(); }
    });

    for (int i = 0; i < 10000 && writer.state() == AsyncBinaryShardWriterState::Open; ++i) {
        std::this_thread::yield();
    }
    CHECK_TRUE(writer.state() != AsyncBinaryShardWriterState::Open);
    check_throws([&] { writer.write_history(std::move(second)); }, "write after closing begins");
    closer.join();
    CHECK_TRUE(close_error != nullptr);
    CHECK_TRUE(writer.state() == AsyncBinaryShardWriterState::Failed);
    CHECK_TRUE(!fs::exists(path));
}

void test_reader_rejects_invalid_offset_table() {
    const std::string path = "test_bad_offset.tdmzshd";
    remove_if_exists(path);
    write_histories_binary_shard({make_tiny_history(400, 8)}, path);
    const uint64_t invalid_offset = 1;
    overwrite_scalar(path, 24, invalid_offset);
    check_throws([&] { BinaryShardReader reader(path); }, "invalid shard offset");
}

void test_reader_rejects_truncated_history() {
    const std::string path = "test_truncated_shard.tdmzshd";
    remove_if_exists(path);
    write_histories_binary_shard({make_tiny_history(410, 8)}, path);
    const auto size = fs::file_size(path);
    CHECK_TRUE(size > 0);
    fs::resize_file(path, size - 1);
    BinaryShardReader reader(path);
    check_throws([&] { (void)reader.read_at(0); }, "truncated shard history");
}

void test_reader_rejects_oversized_tensor_header_before_allocation() {
    const std::string path = "test_oversized_tensor.tdmzspb";
    remove_if_exists(path);
    write_history_binary(make_tiny_history(420, 8), path);
    const uint32_t huge = std::numeric_limits<uint32_t>::max();
    overwrite_scalar(path, 16, huge);
    check_throws([&] { (void)read_history_binary(path); }, "oversized tensor header");
}

int main() {
    try {
        test_binary_roundtrip_selfplay();
        test_binary_roundtrip_empty_history();
        test_binary_shard_roundtrip();
        test_atomic_shard_publication();
        test_async_binary_shard_roundtrip_and_concurrent_close();
        test_async_rejects_write_after_closing_begins();
        test_reader_rejects_invalid_offset_table();
        test_reader_rejects_truncated_history();
        test_reader_rejects_oversized_tensor_header_before_allocation();
        std::cout << "Binary trajectory tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
