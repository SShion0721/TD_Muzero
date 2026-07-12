#include "tdmz/selfplay/training_replay.hpp"
#include "tdmz/selfplay/transition_shard.hpp"
#include "tdmz/training/direct_sequence_replay.hpp"

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
            std::string("check failed at line ") + std::to_string(line)
            + ": " + expression);
    }
}
#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

TrajectoryStep make_step(int index, bool done) {
    TrajectoryStep step;
    step.step_index = index;
    step.action = kFlatWaitOffset;
    step.reward = static_cast<float>(index + 1);
    step.root_value = static_cast<float>(10 + index);
    step.done = done;
    step.money = 200 + index;
    step.base_hp = 100 - index;
    step.wave = 1;
    step.time = static_cast<float>(index);
    step.observation.assign(kObservationSize, static_cast<float>(index) * 0.01f);
    step.policy_target.assign(kActionSpaceSize, 0.0f);
    step.policy_target[kFlatWaitOffset] = 1.0f;
    step.legal_mask.assign(kActionSpaceSize, 0u);
    step.legal_mask[kFlatWaitOffset] = 1u;
    return step;
}

BootstrapState make_bootstrap(int index, float root_value) {
    const TrajectoryStep root = make_step(index, false);
    BootstrapState bootstrap;
    bootstrap.root_value = root_value;
    bootstrap.observation = root.observation;
    bootstrap.policy_target = root.policy_target;
    bootstrap.legal_mask = root.legal_mask;
    return bootstrap;
}

GameHistory make_terminal_history(uint64_t seed, int step_count) {
    GameHistory history;
    history.seed = seed;
    history.max_steps = 32;
    history.terminal = true;
    history.wave_mode = WaveMode::Fixed;
    for (int step = 0; step < step_count; ++step) {
        history.steps.push_back(make_step(step, step + 1 == step_count));
        history.total_reward += history.steps.back().reward;
    }
    return history;
}

GameHistory make_truncated_history(uint64_t seed, int step_count) {
    GameHistory history;
    history.seed = seed;
    history.max_steps = step_count;
    history.truncated = true;
    history.wave_mode = WaveMode::Fixed;
    for (int step = 0; step < step_count; ++step) {
        history.steps.push_back(make_step(step, false));
        history.total_reward += history.steps.back().reward;
    }
    history.bootstrap_state = make_bootstrap(step_count, 100.0f + step_count);
    return history;
}

void check_float_vectors(
    const std::vector<float>& left,
    const std::vector<float>& right,
    float tolerance,
    const char* label
) {
    if (left.size() != right.size()) {
        throw std::runtime_error(std::string(label) + " size mismatch");
    }
    for (size_t index = 0; index < left.size(); ++index) {
        if (std::fabs(left[index] - right[index]) > tolerance) {
            throw std::runtime_error(
                std::string(label) + " mismatch at " + std::to_string(index));
        }
    }
}

void check_batches_equal(const KStepBatch& reference, const KStepBatch& direct) {
    CHECK(reference.batch_size == direct.batch_size);
    CHECK(reference.unroll_steps == direct.unroll_steps);
    CHECK(reference.observation_size == direct.observation_size);
    CHECK(reference.policy_size == direct.policy_size);
    CHECK(reference.legal_mask_size == direct.legal_mask_size);
    check_float_vectors(
        reference.initial_observations,
        direct.initial_observations,
        0.0f,
        "initial observations");
    CHECK(reference.actions == direct.actions);
    check_float_vectors(reference.reward_targets, direct.reward_targets, 0.0f, "rewards");
    CHECK(reference.transition_valid == direct.transition_valid);
    check_float_vectors(reference.value_targets, direct.value_targets, 1e-6f, "values");
    CHECK(reference.value_valid == direct.value_valid);
    check_float_vectors(reference.policy_targets, direct.policy_targets, 0.0f, "policies");
    CHECK(reference.legal_masks == direct.legal_masks);
    CHECK(reference.state_valid == direct.state_valid);
    CHECK(reference.episode_terminal == direct.episode_terminal);
    CHECK(reference.episode_truncated == direct.episode_truncated);
}

void test_direct_sequence_parity_and_deduplication() {
    const std::string shard0 = "test_direct_sequence_w0.tdmzshd";
    const std::string shard1 = "test_direct_sequence_w1.tdmzshd";
    std::filesystem::remove(shard0);
    std::filesystem::remove(shard1);

    write_histories_transition_shard({make_terminal_history(10, 12)}, shard0);
    write_histories_transition_shard({make_truncated_history(20, 5)}, shard1);

    {
        TrainingReplayDataset training({shard0, shard1}, WaveMode::Fixed);
        DirectSequenceReplayDataset direct_dataset(training);
        DirectPositionIndex positions(direct_dataset);

        CHECK(direct_dataset.shard_count() == 2u);
        CHECK(direct_dataset.game_count() == 2u);
        CHECK(positions.position_count() == 18u);
        CHECK(positions.locate(0).global_game_index == 0u);
        CHECK(positions.locate(11).step_index == 11u);
        CHECK(positions.locate(12).global_game_index == 1u);
        CHECK(positions.locate(17).step_index == 5u);

        const ReplayGameMetadata terminal = direct_dataset.game_metadata(0);
        const ReplayGameMetadata truncated = direct_dataset.game_metadata(1);
        CHECK(terminal.terminal && !terminal.truncated);
        CHECK(!terminal.has_bootstrap_state);
        CHECK(!truncated.terminal && truncated.truncated);
        CHECK(truncated.has_bootstrap_state);

        std::vector<ReplaySampleRef> refs;
        auto add_ref = [&](size_t game, size_t step) {
            ReplaySampleRef ref = direct_dataset.locate_game(game);
            ref.step_index = step;
            refs.push_back(ref);
        };
        add_ref(0, 0);  // synthetic local cutoff before the real terminal
        add_ref(0, 1);  // overlapping synthetic window
        add_ref(0, 1);  // exact duplicate
        add_ref(0, 11); // real terminal tail
        add_ref(1, 0);
        add_ref(1, 2);
        add_ref(1, 5);  // persisted cutoff root

        SequenceTargetConfig config;
        config.unroll_steps = 3;
        config.td_steps = 4;
        config.discount = 0.5f;

        const KStepBatch reference = build_reference_k_step_batch(
            training, refs, config);
        training.replay().reset_game_read_count();
        direct_dataset.reset_io_stats();

        DirectSequenceBatchStats stats;
        const KStepBatch direct = build_direct_k_step_batch(
            direct_dataset, refs, config, &stats);
        check_batches_equal(reference, direct);

        CHECK(training.replay().game_read_count() == 0u);
        CHECK(stats.unique_games_touched == 2);
        CHECK(stats.unique_shards_touched == 2);
        CHECK(stats.requested_step_records == 33u);
        CHECK(stats.unique_step_records_read == 15u);
        CHECK(stats.bootstrap_records_read == 1u);
        CHECK(stats.physical_read_operations == 16u);
        CHECK(stats.physical_bytes_read > 0u);
        CHECK(stats.unique_step_records_read < stats.requested_step_records);

        const size_t cutoff_batch = 6u;
        const size_t state_count = static_cast<size_t>(config.unroll_steps) + 1u;
        const size_t state_base = cutoff_batch * state_count;
        CHECK(direct.state_valid[state_base] == 1u);
        CHECK(direct.state_valid[state_base + 1u] == 0u);
        CHECK(direct.value_valid[state_base] == 1u);
        CHECK(std::fabs(direct.value_targets[state_base] - 105.0f) < 1e-6f);
        CHECK(direct.transition_valid[cutoff_batch * config.unroll_steps] == 0u);
        CHECK(direct.episode_truncated[cutoff_batch] == 1u);
    }

    std::filesystem::remove(shard0);
    std::filesystem::remove(shard1);
}

void test_direct_sampler_smoke() {
    const std::string shard = "test_direct_sequence_sampler.tdmzshd";
    std::filesystem::remove(shard);
    write_histories_transition_shard({make_truncated_history(30, 8)}, shard);

    {
        TrainingReplayDataset training({shard}, WaveMode::Fixed);
        DirectSequenceReplayDataset direct_dataset(training);
        SequenceTargetConfig config;
        config.unroll_steps = 5;
        config.td_steps = 10;
        config.discount = 0.997f;
        DirectKStepReplaySampler sampler(direct_dataset, config, 1234567);
        const KStepBatch batch = sampler.sample_batch(16);
        CHECK(batch.batch_size == 16);
        CHECK(batch.unroll_steps == 5);
        CHECK(batch.initial_observations.size()
            == static_cast<size_t>(16 * kObservationSize));
        CHECK(sampler.position_index().position_count() == 9u);
        CHECK(sampler.last_stats().physical_read_operations > 0u);
        CHECK(std::isfinite(k_step_batch_checksum(batch)));
    }

    std::filesystem::remove(shard);
}

} // namespace

int main() {
    try {
        test_direct_sequence_parity_and_deduplication();
        test_direct_sampler_smoke();
        std::cout << "Direct sequence replay tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
