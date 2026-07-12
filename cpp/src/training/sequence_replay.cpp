#include "tdmz/training/sequence_replay.hpp"

#include "tdmz/core/action.hpp"
#include "tdmz/core/compatibility.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace tdmz {
namespace {

void validate_config(const SequenceTargetConfig& config) {
    if (config.unroll_steps < 0) {
        throw std::invalid_argument("MuZero unroll_steps must be non-negative");
    }
    if (config.td_steps <= 0) {
        throw std::invalid_argument("MuZero td_steps must be positive");
    }
    if (!std::isfinite(config.discount)
        || config.discount < 0.0f
        || config.discount > 1.0f) {
        throw std::invalid_argument("MuZero discount must be finite and in [0,1]");
    }
}

void validate_state_payload(
    float root_value,
    const Observation& observation,
    const std::vector<float>& policy_target,
    const std::vector<uint8_t>& legal_mask,
    const char* label
) {
    if (!std::isfinite(root_value)) {
        throw std::invalid_argument(std::string(label) + " root value is non-finite");
    }
    if (observation.size() != static_cast<size_t>(kObservationSize)) {
        throw std::invalid_argument(
            std::string(label) + " observation does not match the current 40x11x11 schema");
    }
    if (policy_target.size() != static_cast<size_t>(kActionSpaceSize)) {
        throw std::invalid_argument(
            std::string(label) + " policy target does not match the 727-action schema");
    }
    if (legal_mask.size() != static_cast<size_t>(kActionSpaceSize)) {
        throw std::invalid_argument(
            std::string(label) + " legal mask does not match the 727-action schema");
    }

    double policy_sum = 0.0;
    for (float probability : policy_target) {
        if (!std::isfinite(probability) || probability < 0.0f) {
            throw std::invalid_argument(
                std::string(label) + " policy target must be finite and non-negative");
        }
        policy_sum += probability;
    }
    if (!std::isfinite(policy_sum) || std::fabs(policy_sum - 1.0) > 1e-4) {
        throw std::invalid_argument(std::string(label) + " policy target must sum to one");
    }
    for (uint8_t legal : legal_mask) {
        if (legal > 1u) {
            throw std::invalid_argument(std::string(label) + " legal mask must be binary");
        }
    }
    if (legal_mask[kFlatWaitOffset] == 0u) {
        throw std::invalid_argument(
            std::string(label) + " exact legal mask must keep Wait1 legal");
    }
}

void validate_history(const GameHistory& history) {
    if (history.terminal && history.truncated) {
        throw std::invalid_argument("Replay episode cannot be both terminal and truncated");
    }
    if (!history.completed()) {
        throw std::invalid_argument("K-step targets require a completed terminal or truncated episode");
    }
    if (history.max_steps < 0) {
        throw std::invalid_argument("Replay max_steps must be non-negative");
    }
    if (history.truncated
        && history.steps.size() != static_cast<size_t>(history.max_steps)) {
        throw std::invalid_argument("Truncated episode step count must equal max_steps");
    }
    if (history.bootstrap_state && !history.truncated) {
        throw std::invalid_argument("Only truncated episodes may contain a bootstrap state");
    }
    if (history.terminal && history.steps.empty()) {
        throw std::invalid_argument("Terminal replay episode must contain a final transition");
    }
    if (history.steps.empty() && !history.bootstrap_state) {
        throw std::invalid_argument("K-step targets require at least one stored root state");
    }

    int done_count = 0;
    for (size_t index = 0; index < history.steps.size(); ++index) {
        const TrajectoryStep& step = history.steps[index];
        if (step.step_index != static_cast<int>(index)) {
            throw std::invalid_argument("Replay step_index must be contiguous from zero");
        }
        if (step.action < 0 || step.action >= kActionSpaceSize) {
            throw std::invalid_argument("Replay action is outside the 727-action ABI");
        }
        if (!std::isfinite(step.reward) || !std::isfinite(step.time)) {
            throw std::invalid_argument("Replay step contains a non-finite transition scalar");
        }
        validate_state_payload(
            step.root_value,
            step.observation,
            step.policy_target,
            step.legal_mask,
            "Replay step");
        if (step.legal_mask[static_cast<size_t>(step.action)] == 0u) {
            throw std::invalid_argument("Replay selected action is masked illegal");
        }

        if (step.done) {
            ++done_count;
            if (index + 1 != history.steps.size()) {
                throw std::invalid_argument("Terminal done flag may appear only on the last transition");
            }
        }
    }

    if (history.terminal && done_count != 1) {
        throw std::invalid_argument("Terminal episode must end with exactly one done transition");
    }
    if (history.truncated && done_count != 0) {
        throw std::invalid_argument("Truncated episode must not contain a done transition");
    }
    if (history.bootstrap_state) {
        validate_state_payload(
            history.bootstrap_state->root_value,
            history.bootstrap_state->observation,
            history.bootstrap_state->policy_target,
            history.bootstrap_state->legal_mask,
            "Replay bootstrap state");
    }
}

size_t logical_state_count(const GameHistory& history) {
    return history.steps.size() + (history.bootstrap_state ? 1u : 0u);
}

const Observation& state_observation(const GameHistory& history, size_t state_index) {
    if (state_index < history.steps.size()) {
        return history.steps[state_index].observation;
    }
    if (state_index == history.steps.size() && history.bootstrap_state) {
        return history.bootstrap_state->observation;
    }
    throw std::out_of_range("Replay state index is outside the episode");
}

std::pair<float, uint8_t> n_step_value_target(
    const GameHistory& history,
    size_t state_index,
    const SequenceTargetConfig& config
) {
    const size_t step_count = history.steps.size();
    if (state_index == step_count && history.bootstrap_state) {
        return {history.bootstrap_state->root_value, 1u};
    }
    if (state_index >= step_count) {
        return {0.0f, 0u};
    }

    double target = 0.0;
    double discount_power = 1.0;
    bool terminal_reached = false;

    for (int offset = 0; offset < config.td_steps; ++offset) {
        const size_t transition_index = state_index + static_cast<size_t>(offset);
        if (transition_index >= step_count) break;

        const TrajectoryStep& transition = history.steps[transition_index];
        target += discount_power * static_cast<double>(transition.reward);
        if (transition.done) {
            terminal_reached = true;
            break;
        }
        discount_power *= static_cast<double>(config.discount);
    }

    if (terminal_reached) {
        return {static_cast<float>(target), 1u};
    }

    const size_t bootstrap_index = state_index + static_cast<size_t>(config.td_steps);
    if (bootstrap_index < step_count) {
        target += discount_power
            * static_cast<double>(history.steps[bootstrap_index].root_value);
        return {static_cast<float>(target), 1u};
    }
    if (bootstrap_index == step_count && history.bootstrap_state) {
        target += discount_power
            * static_cast<double>(history.bootstrap_state->root_value);
        return {static_cast<float>(target), 1u};
    }

    // A truncated tail without a persisted cutoff bootstrap remains explicitly
    // invalid instead of being treated as a zero-value terminal state.
    return {0.0f, 0u};
}

void checked_add_size(size_t& value, size_t increment, const char* label) {
    if (increment > std::numeric_limits<size_t>::max() - value) {
        throw std::overflow_error(std::string(label) + " size overflow");
    }
    value += increment;
}

} // namespace

KStepSample build_k_step_sample(
    const GameHistory& history,
    size_t start_step,
    const SequenceTargetConfig& config
) {
    validate_config(config);
    validate_history(history);
    const size_t available_states = logical_state_count(history);
    if (start_step >= available_states) {
        throw std::out_of_range("K-step start position is outside the episode");
    }

    KStepSample sample;
    sample.start_step = start_step;
    sample.unroll_steps = config.unroll_steps;
    sample.observation_size = kObservationSize;
    sample.policy_size = kActionSpaceSize;
    sample.legal_mask_size = kActionSpaceSize;
    sample.episode_terminal = history.terminal;
    sample.episode_truncated = history.truncated;
    sample.initial_observation = state_observation(history, start_step);

    const size_t transition_count = static_cast<size_t>(config.unroll_steps);
    const size_t state_count = transition_count + 1u;
    sample.actions.assign(transition_count, kFlatWaitOffset);
    sample.reward_targets.assign(transition_count, 0.0f);
    sample.transition_valid.assign(transition_count, 0u);
    sample.value_targets.assign(state_count, 0.0f);
    sample.value_valid.assign(state_count, 0u);
    sample.state_valid.assign(state_count, 0u);
    sample.policy_targets.assign(state_count * kActionSpaceSize, 0.0f);
    sample.legal_masks.assign(state_count * kActionSpaceSize, 0u);

    for (size_t offset = 0; offset < transition_count; ++offset) {
        const size_t transition_index = start_step + offset;
        if (transition_index >= history.steps.size()) break;
        const TrajectoryStep& transition = history.steps[transition_index];
        sample.actions[offset] = transition.action;
        sample.reward_targets[offset] = transition.reward;
        sample.transition_valid[offset] = 1u;
    }

    for (size_t offset = 0; offset < state_count; ++offset) {
        const size_t state_index = start_step + offset;
        if (state_index >= available_states) break;

        sample.state_valid[offset] = 1u;
        if (state_index < history.steps.size()) {
            const TrajectoryStep& state = history.steps[state_index];
            std::copy(
                state.policy_target.begin(),
                state.policy_target.end(),
                sample.policy_targets.begin() + offset * kActionSpaceSize);
            std::copy(
                state.legal_mask.begin(),
                state.legal_mask.end(),
                sample.legal_masks.begin() + offset * kActionSpaceSize);
        } else {
            const BootstrapState& state = *history.bootstrap_state;
            std::copy(
                state.policy_target.begin(),
                state.policy_target.end(),
                sample.policy_targets.begin() + offset * kActionSpaceSize);
            std::copy(
                state.legal_mask.begin(),
                state.legal_mask.end(),
                sample.legal_masks.begin() + offset * kActionSpaceSize);
        }

        const auto value = n_step_value_target(history, state_index, config);
        sample.value_targets[offset] = value.first;
        sample.value_valid[offset] = value.second;
    }

    return sample;
}

KStepBatch pack_k_step_samples(const std::vector<KStepSample>& samples) {
    if (samples.empty()) {
        throw std::invalid_argument("K-step batch requires at least one sample");
    }

    const KStepSample& first = samples.front();
    KStepBatch batch;
    batch.batch_size = static_cast<int>(samples.size());
    batch.unroll_steps = first.unroll_steps;
    batch.observation_size = first.observation_size;
    batch.policy_size = first.policy_size;
    batch.legal_mask_size = first.legal_mask_size;

    const size_t state_count = static_cast<size_t>(batch.unroll_steps) + 1u;
    for (const KStepSample& sample : samples) {
        if (sample.unroll_steps != batch.unroll_steps
            || sample.observation_size != batch.observation_size
            || sample.policy_size != batch.policy_size
            || sample.legal_mask_size != batch.legal_mask_size) {
            throw std::invalid_argument("K-step samples have incompatible dimensions");
        }
        if (sample.initial_observation.size() != static_cast<size_t>(batch.observation_size)
            || sample.actions.size() != static_cast<size_t>(batch.unroll_steps)
            || sample.reward_targets.size() != static_cast<size_t>(batch.unroll_steps)
            || sample.transition_valid.size() != static_cast<size_t>(batch.unroll_steps)
            || sample.value_targets.size() != state_count
            || sample.value_valid.size() != state_count
            || sample.state_valid.size() != state_count
            || sample.policy_targets.size() != state_count * static_cast<size_t>(batch.policy_size)
            || sample.legal_masks.size() != state_count * static_cast<size_t>(batch.legal_mask_size)) {
            throw std::invalid_argument("K-step sample payload does not match its declared dimensions");
        }

        batch.initial_observations.insert(
            batch.initial_observations.end(),
            sample.initial_observation.begin(),
            sample.initial_observation.end());
        batch.actions.insert(batch.actions.end(), sample.actions.begin(), sample.actions.end());
        batch.reward_targets.insert(
            batch.reward_targets.end(), sample.reward_targets.begin(), sample.reward_targets.end());
        batch.transition_valid.insert(
            batch.transition_valid.end(), sample.transition_valid.begin(), sample.transition_valid.end());
        batch.value_targets.insert(
            batch.value_targets.end(), sample.value_targets.begin(), sample.value_targets.end());
        batch.value_valid.insert(
            batch.value_valid.end(), sample.value_valid.begin(), sample.value_valid.end());
        batch.policy_targets.insert(
            batch.policy_targets.end(), sample.policy_targets.begin(), sample.policy_targets.end());
        batch.legal_masks.insert(
            batch.legal_masks.end(), sample.legal_masks.begin(), sample.legal_masks.end());
        batch.state_valid.insert(
            batch.state_valid.end(), sample.state_valid.begin(), sample.state_valid.end());
        batch.episode_terminal.push_back(sample.episode_terminal ? 1u : 0u);
        batch.episode_truncated.push_back(sample.episode_truncated ? 1u : 0u);
    }
    return batch;
}

UniformPositionIndex::UniformPositionIndex(const TrainingReplayDataset& dataset)
    : dataset_(&dataset) {
    cumulative_positions_by_game_.reserve(dataset.game_count());
    for (size_t game = 0; game < dataset.game_count(); ++game) {
        // D2.1 keeps the reference sampler position set on transition roots.
        // D2.2 will use shard metadata to include the persisted cutoff root
        // without deserializing every game during index construction.
        const size_t count = dataset.replay().step_count(game);
        checked_add_size(total_positions_, count, "Replay position index");
        cumulative_positions_by_game_.push_back(total_positions_);
    }
    if (total_positions_ == 0) {
        throw std::runtime_error("Training replay contains zero positions");
    }
}

ReplaySampleRef UniformPositionIndex::locate(size_t global_position_index) const {
    if (global_position_index >= total_positions_) {
        throw std::out_of_range("Global replay position is out of range");
    }
    const auto iterator = std::upper_bound(
        cumulative_positions_by_game_.begin(),
        cumulative_positions_by_game_.end(),
        global_position_index);
    const size_t global_game = static_cast<size_t>(
        iterator - cumulative_positions_by_game_.begin());
    const size_t game_base = global_game == 0
        ? 0
        : cumulative_positions_by_game_[global_game - 1];

    ReplaySampleRef ref = dataset_->replay().locate_game(global_game);
    ref.step_index = global_position_index - game_base;
    return ref;
}

ReplaySampleRef UniformPositionIndex::sample(uint64_t random_u64) const {
    return locate(static_cast<size_t>(random_u64 % total_positions_));
}

KStepReplaySampler::KStepReplaySampler(
    TrainingReplayDataset& dataset,
    SequenceTargetConfig config,
    uint64_t seed
) : dataset_(dataset),
    config_(config),
    positions_(dataset),
    rng_state_(seed ? seed : 0x9e3779b97f4a7c15ULL) {
    validate_config(config_);
}

uint64_t KStepReplaySampler::next_u64() {
    uint64_t value = rng_state_;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    rng_state_ = value;
    return value;
}

KStepBatch KStepReplaySampler::sample_batch(int batch_size) {
    if (batch_size <= 0) {
        throw std::invalid_argument("K-step replay batch size must be positive");
    }

    std::vector<ReplaySampleRef> refs;
    refs.reserve(static_cast<size_t>(batch_size));
    for (int index = 0; index < batch_size; ++index) {
        refs.push_back(positions_.sample(next_u64()));
    }

    std::map<size_t, GameHistory> games;
    std::vector<KStepSample> samples;
    samples.reserve(static_cast<size_t>(batch_size));
    for (const ReplaySampleRef& ref : refs) {
        auto iterator = games.find(ref.global_game_index);
        if (iterator == games.end()) {
            iterator = games.emplace(
                ref.global_game_index,
                dataset_.read_game(ref.global_game_index)).first;
        }
        if (ref.step_index >= iterator->second.steps.size()) {
            throw std::runtime_error("Uniform position index disagrees with replay game length");
        }
        KStepSample sample = build_k_step_sample(iterator->second, ref.step_index, config_);
        sample.global_game_index = ref.global_game_index;
        samples.push_back(std::move(sample));
    }
    return pack_k_step_samples(samples);
}

uint64_t k_step_batch_payload_bytes(const KStepBatch& batch) {
    uint64_t bytes = 0;
    bytes += static_cast<uint64_t>(batch.initial_observations.size()) * sizeof(float);
    bytes += static_cast<uint64_t>(batch.actions.size()) * sizeof(int);
    bytes += static_cast<uint64_t>(batch.reward_targets.size()) * sizeof(float);
    bytes += static_cast<uint64_t>(batch.transition_valid.size()) * sizeof(uint8_t);
    bytes += static_cast<uint64_t>(batch.value_targets.size()) * sizeof(float);
    bytes += static_cast<uint64_t>(batch.value_valid.size()) * sizeof(uint8_t);
    bytes += static_cast<uint64_t>(batch.policy_targets.size()) * sizeof(float);
    bytes += static_cast<uint64_t>(batch.legal_masks.size()) * sizeof(uint8_t);
    bytes += static_cast<uint64_t>(batch.state_valid.size()) * sizeof(uint8_t);
    bytes += static_cast<uint64_t>(batch.episode_terminal.size()) * sizeof(uint8_t);
    bytes += static_cast<uint64_t>(batch.episode_truncated.size()) * sizeof(uint8_t);
    return bytes;
}

double k_step_batch_checksum(const KStepBatch& batch) {
    double checksum = static_cast<double>(batch.batch_size) * 17.0
        + static_cast<double>(batch.unroll_steps) * 19.0;
    for (size_t index = 0; index < batch.initial_observations.size(); index += 101) {
        checksum += batch.initial_observations[index]
            * static_cast<double>((index % 23) + 1);
    }
    for (size_t index = 0; index < batch.policy_targets.size(); index += 67) {
        checksum += batch.policy_targets[index]
            * static_cast<double>((index % 29) + 1);
    }
    for (size_t index = 0; index < batch.value_targets.size(); ++index) {
        checksum += batch.value_targets[index] * 7.0;
        checksum += batch.value_valid[index] ? 0.5 : 0.0;
        checksum += batch.state_valid[index] ? 0.25 : 0.0;
    }
    for (size_t index = 0; index < batch.actions.size(); ++index) {
        checksum += batch.actions[index] * 0.03125;
        checksum += batch.reward_targets[index] * 3.0;
        checksum += batch.transition_valid[index] ? 0.125 : 0.0;
    }
    return checksum;
}

} // namespace tdmz
