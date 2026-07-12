#include "tdmz/selfplay/transition_shard.hpp"

#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace tdmz;
namespace fs = std::filesystem;

namespace {

void check_true(bool condition, const char* expression, int line) {
    if (!condition) {
        throw std::runtime_error(
            std::string("check failed at line ") + std::to_string(line) + ": " + expression);
    }
}
#define CHECK_TRUE(expression) check_true(static_cast<bool>(expression), #expression, __LINE__)

void check_close(float left, float right, const char* label) {
    if (std::fabs(left - right) > 1e-6f) {
        throw std::runtime_error(std::string("float mismatch: ") + label);
    }
}

template <typename Fn>
void check_throws(Fn&& function, const char* label) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    if (!threw) throw std::runtime_error(std::string("expected exception: ") + label);
}

void remove_if_exists(const std::string& path) {
    std::error_code error;
    fs::remove(path, error);
}

TrajectoryStep make_step(
    int index,
    size_t observation_size,
    size_t policy_size,
    size_t legal_size,
    bool done
) {
    TrajectoryStep step;
    step.step_index = index;
    step.action = index + 3;
    step.reward = 0.25f * static_cast<float>(index + 1);
    step.root_value = -0.1f * static_cast<float>(index + 1);
    step.done = done;
    step.money = 100 - index;
    step.base_hp = 20 - index;
    step.wave = 1 + index / 2;
    step.time = static_cast<float>(index + 1);
    step.observation.resize(observation_size);
    step.policy_target.resize(policy_size);
    step.legal_mask.resize(legal_size);
    for (size_t i = 0; i < observation_size; ++i) {
        step.observation[i] = static_cast<float>(index * 100 + static_cast<int>(i)) * 0.01f;
    }
    for (size_t i = 0; i < policy_size; ++i) {
        step.policy_target[i] = static_cast<float>(index + 1) / static_cast<float>(policy_size + i + 1);
    }
    for (size_t i = 0; i < legal_size; ++i) {
        step.legal_mask[i] = static_cast<uint8_t>((i + static_cast<size_t>(index)) % 3 != 0);
    }
    return step;
}

BootstrapState make_bootstrap(
    int index,
    size_t observation_size,
    size_t policy_size,
    size_t legal_size
) {
    const TrajectoryStep state = make_step(
        index, observation_size, policy_size, legal_size, false);
    BootstrapState bootstrap;
    bootstrap.root_value = state.root_value;
    bootstrap.observation = state.observation;
    bootstrap.policy_target = state.policy_target;
    bootstrap.legal_mask = state.legal_mask;
    return bootstrap;
}

GameHistory make_terminal_history(
    uint64_t seed,
    int step_count,
    size_t obs = 7,
    size_t policy = 5,
    size_t legal = 5
) {
    GameHistory history;
    history.seed = seed;
    history.max_steps = 32;
    history.terminal = true;
    history.wave_mode = WaveMode::Fixed;
    history.total_reward = static_cast<float>(step_count) * 0.75f;
    for (int step = 0; step < step_count; ++step) {
        history.steps.push_back(make_step(
            step, obs, policy, legal, step + 1 == step_count));
    }
    return history;
}

GameHistory make_truncated_history(
    uint64_t seed,
    int step_count,
    size_t obs = 7,
    size_t policy = 5,
    size_t legal = 5
) {
    GameHistory history;
    history.seed = seed;
    history.max_steps = step_count;
    history.truncated = true;
    history.wave_mode = WaveMode::Budgeted;
    history.total_reward = static_cast<float>(step_count) * 0.5f;
    for (int step = 0; step < step_count; ++step) {
        history.steps.push_back(make_step(step, obs, policy, legal, false));
    }
    history.bootstrap_state = make_bootstrap(step_count, obs, policy, legal);
    return history;
}

void assert_step_equal(const TrajectoryStep& expected, const TrajectoryStep& actual) {
    CHECK_TRUE(expected.step_index == actual.step_index);
    CHECK_TRUE(expected.action == actual.action);
    check_close(expected.reward, actual.reward, "reward");
    check_close(expected.root_value, actual.root_value, "root_value");
    CHECK_TRUE(expected.done == actual.done);
    CHECK_TRUE(expected.money == actual.money);
    CHECK_TRUE(expected.base_hp == actual.base_hp);
    CHECK_TRUE(expected.wave == actual.wave);
    check_close(expected.time, actual.time, "time");
    CHECK_TRUE(expected.observation == actual.observation);
    CHECK_TRUE(expected.policy_target == actual.policy_target);
    CHECK_TRUE(expected.legal_mask == actual.legal_mask);
}

void assert_bootstrap_equal(const BootstrapState& expected, const BootstrapState& actual) {
    check_close(expected.root_value, actual.root_value, "bootstrap root_value");
    CHECK_TRUE(expected.observation == actual.observation);
    CHECK_TRUE(expected.policy_target == actual.policy_target);
    CHECK_TRUE(expected.legal_mask == actual.legal_mask);
}

void assert_history_equal(const GameHistory& expected, const GameHistory& actual) {
    CHECK_TRUE(expected.seed == actual.seed);
    CHECK_TRUE(expected.max_steps == actual.max_steps);
    CHECK_TRUE(expected.terminal == actual.terminal);
    CHECK_TRUE(expected.truncated == actual.truncated);
    CHECK_TRUE(expected.wave_mode == actual.wave_mode);
    check_close(expected.total_reward, actual.total_reward, "total_reward");
    CHECK_TRUE(expected.steps.size() == actual.steps.size());
    CHECK_TRUE(expected.bootstrap_state.has_value() == actual.bootstrap_state.has_value());
    for (size_t i = 0; i < expected.steps.size(); ++i) {
        assert_step_equal(expected.steps[i], actual.steps[i]);
    }
    if (expected.bootstrap_state) {
        assert_bootstrap_equal(*expected.bootstrap_state, *actual.bootstrap_state);
    }
}

template <typename T>
void overwrite_scalar(const std::string& path, std::streamoff offset, const T& value) {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!file) throw std::runtime_error("failed to open transition corruption target");
    file.seekp(offset, std::ios::beg);
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!file) throw std::runtime_error("failed to corrupt transition shard");
}

#pragma pack(push, 1)
struct TestTransitionHeader {
    char magic[8];
    uint32_t version;
    uint32_t history_count;
    uint32_t observation_size;
    uint32_t policy_size;
    uint32_t legal_mask_size;
    uint32_t reserved;
    uint64_t total_step_count;
    uint64_t game_table_offset;
    uint64_t step_table_offset;
    uint64_t file_size;
};
#pragma pack(pop)
static_assert(sizeof(TestTransitionHeader) == 64, "test header layout mismatch");

TestTransitionHeader read_header(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open transition shard header");
    TestTransitionHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in) throw std::runtime_error("failed to read transition shard header");
    return header;
}

void test_transition_roundtrip_and_direct_read() {
    const std::string path = "test_transition_roundtrip.tdmzshd";
    remove_if_exists(path);
    const std::vector<GameHistory> histories = {
        make_terminal_history(100, 3),
        make_truncated_history(101, 2),
        make_terminal_history(102, 4),
    };
    write_histories_transition_shard(histories, path);

    CHECK_TRUE(is_transition_shard_v2(path));
    TransitionShardReader reader(path);
    CHECK_TRUE(reader.history_count() == histories.size());
    CHECK_TRUE(reader.observation_size() == 7);
    CHECK_TRUE(reader.policy_size() == 5);
    CHECK_TRUE(reader.legal_mask_size() == 5);
    CHECK_TRUE(reader.step_count(0) == 3);
    CHECK_TRUE(reader.step_count(2) == 4);
    CHECK_TRUE(reader.step_physical_offset(0, 1) < reader.step_physical_offset(2, 0));

    const ReplayGameMetadata metadata = reader.game_metadata(1);
    CHECK_TRUE(!metadata.terminal);
    CHECK_TRUE(metadata.truncated);
    CHECK_TRUE(metadata.wave_mode == WaveMode::Budgeted);
    CHECK_TRUE(metadata.has_bootstrap_state);
    CHECK_TRUE(metadata.step_count == 2u);
    assert_bootstrap_equal(
        *histories[1].bootstrap_state,
        reader.read_bootstrap_state(1));

    for (size_t game = 0; game < histories.size(); ++game) {
        assert_history_equal(histories[game], reader.read_at(game));
    }

    reader.reset_io_stats();
    std::vector<float> observation(7);
    std::vector<float> policy(5);
    std::vector<uint8_t> legal(5);
    float value = 0.0f;
    float reward = 0.0f;
    int action = -1;
    uint8_t done = 0;
    int step_index = -1;
    int money = 0;
    int base_hp = 0;
    int wave = 0;
    float time = 0.0f;

    ReplayStepDestination destination;
    destination.observation = observation.data();
    destination.observation_size = observation.size();
    destination.policy_target = policy.data();
    destination.policy_size = policy.size();
    destination.legal_mask = legal.data();
    destination.legal_mask_size = legal.size();
    destination.value = &value;
    destination.reward = &reward;
    destination.action = &action;
    destination.done = &done;
    destination.step_index = &step_index;
    destination.money = &money;
    destination.base_hp = &base_hp;
    destination.wave = &wave;
    destination.time = &time;
    reader.read_step_into(1, 1, destination);

    const TrajectoryStep& expected = histories[1].steps[1];
    CHECK_TRUE(observation == expected.observation);
    CHECK_TRUE(policy == expected.policy_target);
    CHECK_TRUE(legal == expected.legal_mask);
    check_close(value, expected.root_value, "direct value");
    check_close(reward, expected.reward, "direct reward");
    CHECK_TRUE(action == expected.action);
    CHECK_TRUE(done == static_cast<uint8_t>(expected.done));
    CHECK_TRUE(step_index == expected.step_index);
    CHECK_TRUE(money == expected.money);
    CHECK_TRUE(base_hp == expected.base_hp);
    CHECK_TRUE(wave == expected.wave);
    check_close(time, expected.time, "direct time");

    const ReplayIoStats stats = reader.io_stats();
    CHECK_TRUE(stats.physical_read_operations == 1);
    CHECK_TRUE(stats.physical_bytes_read > 0);
    CHECK_TRUE(stats.physical_bytes_read < fs::file_size(path));
}

void test_transition_atomic_publication() {
    const std::string path = "test_transition_atomic.tdmzshd";
    remove_if_exists(path);
    TransitionShardWriter writer(path, 1);
    CHECK_TRUE(!fs::exists(path));
    writer.write_history(make_terminal_history(200, 2));
    CHECK_TRUE(!fs::exists(path));
    writer.close();
    CHECK_TRUE(fs::exists(path));
    CHECK_TRUE(writer.closed());
}

void test_async_transition_roundtrip_and_close() {
    const std::string path = "test_transition_async.tdmzshd";
    remove_if_exists(path);
    const std::vector<GameHistory> histories = {
        make_terminal_history(300, 2),
        make_truncated_history(301, 3),
        make_terminal_history(302, 1),
    };

    AsyncTransitionShardWriter writer(path, histories.size(), 1);
    for (const auto& history : histories) writer.write_history(history);
    std::exception_ptr first_error;
    std::exception_ptr second_error;
    std::thread first([&] { try { writer.close(); } catch (...) { first_error = std::current_exception(); } });
    std::thread second([&] { try { writer.close(); } catch (...) { second_error = std::current_exception(); } });
    first.join();
    second.join();
    CHECK_TRUE(!first_error);
    CHECK_TRUE(!second_error);
    CHECK_TRUE(writer.state() == AsyncTransitionShardWriterState::Closed);
    CHECK_TRUE(writer.closed());

    TransitionShardReader reader(path);
    for (size_t i = 0; i < histories.size(); ++i) {
        assert_history_equal(histories[i], reader.read_at(i));
    }
}

void test_transition_rejects_future_version() {
    const std::string path = "test_transition_future.tdmzshd";
    remove_if_exists(path);
    write_histories_transition_shard({make_terminal_history(400, 2)}, path);
    const uint32_t future = 99;
    overwrite_scalar(path, 8, future);
    check_throws([&] { TransitionShardReader reader(path); }, "future transition version");
}

void test_transition_rejects_bad_step_offset() {
    const std::string path = "test_transition_bad_offset.tdmzshd";
    remove_if_exists(path);
    write_histories_transition_shard({make_terminal_history(410, 2)}, path);
    const TestTransitionHeader header = read_header(path);
    const uint64_t invalid_offset = 1;
    overwrite_scalar(path, static_cast<std::streamoff>(header.step_table_offset), invalid_offset);
    check_throws([&] { TransitionShardReader reader(path); }, "bad transition step offset");
}

void test_transition_rejects_truncation() {
    const std::string path = "test_transition_truncated.tdmzshd";
    remove_if_exists(path);
    write_histories_transition_shard({make_terminal_history(420, 2)}, path);
    const uint64_t size = fs::file_size(path);
    CHECK_TRUE(size > 1);
    fs::resize_file(path, size - 1);
    check_throws([&] { TransitionShardReader reader(path); }, "truncated transition shard");
}

void test_transition_rejects_shape_mismatch() {
    const std::string path = "test_transition_shape_mismatch.tdmzshd";
    remove_if_exists(path);
    TransitionShardWriter writer(path, 2);
    writer.write_history(make_terminal_history(500, 1, 7, 5, 5));
    check_throws(
        [&] { writer.write_history(make_terminal_history(501, 1, 8, 5, 5)); },
        "transition shape mismatch");
}

} // namespace

int main() {
    try {
        test_transition_roundtrip_and_direct_read();
        test_transition_atomic_publication();
        test_async_transition_roundtrip_and_close();
        test_transition_rejects_future_version();
        test_transition_rejects_bad_step_offset();
        test_transition_rejects_truncation();
        test_transition_rejects_shape_mismatch();
        std::cout << "Transition shard tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
