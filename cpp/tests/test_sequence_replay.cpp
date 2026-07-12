#include "tdmz/core/compatibility.hpp"
#include "tdmz/selfplay/training_replay.hpp"
#include "tdmz/selfplay/transition_shard.hpp"
#include "tdmz/training/sequence_replay.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
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

void check_close(float actual, float expected, float tolerance, const char* label) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            std::string(label) + " actual=" + std::to_string(actual)
            + " expected=" + std::to_string(expected));
    }
}

template <typename Fn>
void check_invalid(Fn&& fn) {
    bool threw = false;
    try {
        fn();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

TrajectoryStep make_step(int index, float reward, float root_value, bool done) {
    TrajectoryStep step;
    step.step_index = index;
    step.action = kFlatWaitOffset;
    step.reward = reward;
    step.root_value = root_value;
    step.done = done;
    step.money = 200 + index;
    step.base_hp = 100;
    step.wave = 1;
    step.time = static_cast<float>(index);
    step.observation.assign(kObservationSize, static_cast<float>(index) * 0.01f);
    step.policy_target.assign(kActionSpaceSize, 0.0f);
    step.policy_target[kFlatWaitOffset] = 1.0f;
    step.legal_mask.assign(kActionSpaceSize, 0u);
    step.legal_mask[kFlatWaitOffset] = 1u;
    return step;
}

GameHistory make_terminal_history() {
    GameHistory history;
    history.seed = 11;
    history.max_steps = 10;
    history.terminal = true;
    history.wave_mode = WaveMode::Fixed;
    for (int index = 0; index < 5; ++index) {
        history.steps.push_back(make_step(
            index,
            static_cast<float>(index + 1),
            static_cast<float>(10 + index),
            index == 4));
        history.total_reward += history.steps.back().reward;
    }
    return history;
}

GameHistory make_truncated_history(int step_count, uint64_t seed = 22) {
    GameHistory history;
    history.seed = seed;
    history.max_steps = step_count;
    history.truncated = true;
    history.wave_mode = WaveMode::Fixed;
    for (int index = 0; index < step_count; ++index) {
        history.steps.push_back(make_step(
            index,
            static_cast<float>(index + 1),
            static_cast<float>(20 + index),
            false));
        history.total_reward += history.steps.back().reward;
    }
    return history;
}

void test_terminal_n_step_targets() {
    SequenceTargetConfig config;
    config.unroll_steps = 3;
    config.td_steps = 2;
    config.discount = 0.5f;

    const KStepSample sample = build_k_step_sample(
        make_terminal_history(), 1, config);
    CHECK(sample.episode_terminal);
    CHECK(!sample.episode_truncated);
    CHECK(sample.state_valid == std::vector<uint8_t>({1u, 1u, 1u, 1u}));
    CHECK(sample.transition_valid == std::vector<uint8_t>({1u, 1u, 1u}));
    CHECK(sample.actions == std::vector<int>({kFlatWaitOffset, kFlatWaitOffset, kFlatWaitOffset}));
    CHECK(sample.reward_targets == std::vector<float>({2.0f, 3.0f, 4.0f}));

    // s1: r1 + .5*r2 + .25*V(s3) = 2 + 1.5 + 3.25
    check_close(sample.value_targets[0], 6.75f, 1e-6f, "terminal s1 target");
    // s2: r2 + .5*r3 + .25*V(s4) = 3 + 2 + 3.5
    check_close(sample.value_targets[1], 8.5f, 1e-6f, "terminal s2 target");
    // s3 includes the terminal transition at step 4, so no bootstrap.
    check_close(sample.value_targets[2], 6.5f, 1e-6f, "terminal s3 target");
    check_close(sample.value_targets[3], 5.0f, 1e-6f, "terminal s4 target");
    CHECK(sample.value_valid == std::vector<uint8_t>({1u, 1u, 1u, 1u}));
}

void test_terminal_tail_padding() {
    SequenceTargetConfig config;
    config.unroll_steps = 3;
    config.td_steps = 10;
    config.discount = 0.9f;

    const KStepSample sample = build_k_step_sample(
        make_terminal_history(), 4, config);
    CHECK(sample.state_valid == std::vector<uint8_t>({1u, 0u, 0u, 0u}));
    CHECK(sample.transition_valid == std::vector<uint8_t>({1u, 0u, 0u}));
    CHECK(sample.value_valid == std::vector<uint8_t>({1u, 0u, 0u, 0u}));
    check_close(sample.value_targets[0], 5.0f, 1e-6f, "terminal tail target");
    CHECK(sample.actions[1] == kFlatWaitOffset);
    CHECK(sample.reward_targets[1] == 0.0f);
}

void test_truncated_tail_is_not_terminal() {
    SequenceTargetConfig config;
    config.unroll_steps = 2;
    config.td_steps = 2;
    config.discount = 0.5f;

    const KStepSample sample = build_k_step_sample(
        make_truncated_history(4), 1, config);
    CHECK(!sample.episode_terminal);
    CHECK(sample.episode_truncated);

    // s1: r1 + .5*r2 + .25*V(s3) = 2 + 1.5 + 5.75.
    check_close(sample.value_targets[0], 9.25f, 1e-6f, "truncated interior target");
    CHECK(sample.value_valid[0] == 1u);

    // s2 and s3 reach the max-step cutoff without a stored post-cutoff value.
    CHECK(sample.value_valid[1] == 0u);
    CHECK(sample.value_valid[2] == 0u);
    CHECK(sample.value_targets[1] == 0.0f);
    CHECK(sample.value_targets[2] == 0.0f);
}

void test_invalid_episode_semantics_are_rejected() {
    GameHistory both = make_terminal_history();
    both.truncated = true;
    check_invalid([&] { (void)build_k_step_sample(both, 0); });

    GameHistory incomplete = make_truncated_history(4);
    incomplete.truncated = false;
    incomplete.max_steps = 10;
    check_invalid([&] { (void)build_k_step_sample(incomplete, 0); });

    GameHistory bad_wait = make_truncated_history(4);
    bad_wait.steps[0].legal_mask[kFlatWaitOffset] = 0u;
    check_invalid([&] { (void)build_k_step_sample(bad_wait, 0); });
}

void test_uniform_position_index_and_sampler() {
    const std::string path = "test_sequence_replay.tdmzshd";
    std::filesystem::remove(path);

    const GameHistory short_game = make_truncated_history(2, 100);
    const GameHistory long_game = make_truncated_history(5, 101);
    write_histories_transition_shard({short_game, long_game}, path);

    // Windows does not allow deleting a file while TransitionShardReader still owns
    // an open handle. Keep all replay readers inside a nested scope so their destructors
    // close the shard before cleanup.
    {
        TrainingReplayDataset dataset({path}, WaveMode::Fixed);
        UniformPositionIndex positions(dataset);
        CHECK(positions.position_count() == 7u);

        CHECK(positions.locate(0).global_game_index == 0u);
        CHECK(positions.locate(0).step_index == 0u);
        CHECK(positions.locate(1).global_game_index == 0u);
        CHECK(positions.locate(1).step_index == 1u);
        CHECK(positions.locate(2).global_game_index == 1u);
        CHECK(positions.locate(2).step_index == 0u);
        CHECK(positions.locate(6).global_game_index == 1u);
        CHECK(positions.locate(6).step_index == 4u);

        SequenceTargetConfig config;
        config.unroll_steps = 2;
        config.td_steps = 2;
        config.discount = 0.5f;
        KStepReplaySampler sampler(dataset, config, 1234567);
        const KStepBatch batch = sampler.sample_batch(8);
        CHECK(batch.batch_size == 8);
        CHECK(batch.unroll_steps == 2);
        CHECK(batch.initial_observations.size() == static_cast<size_t>(8 * kObservationSize));
        CHECK(batch.actions.size() == 16u);
        CHECK(batch.value_targets.size() == 24u);
        CHECK(batch.policy_targets.size() == static_cast<size_t>(24 * kActionSpaceSize));
        CHECK(batch.legal_masks.size() == static_cast<size_t>(24 * kActionSpaceSize));
        CHECK(k_step_batch_payload_bytes(batch) > 0u);
        CHECK(std::isfinite(k_step_batch_checksum(batch)));
    }

    std::filesystem::remove(path);
}

} // namespace

int main() {
    try {
        test_terminal_n_step_targets();
        test_terminal_tail_padding();
        test_truncated_tail_is_not_terminal();
        test_invalid_episode_semantics_are_rejected();
        test_uniform_position_index_and_sampler();
        std::cout << "Sequence replay tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
