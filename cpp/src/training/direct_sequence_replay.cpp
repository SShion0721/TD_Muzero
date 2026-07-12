#include "tdmz/training/direct_sequence_replay.hpp"

#include "tdmz/core/compatibility.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace tdmz {
namespace {

void validate_config(const SequenceTargetConfig& config) {
    if (config.unroll_steps < 0) {
        throw std::invalid_argument("Direct sequence unroll_steps must be non-negative");
    }
    if (config.td_steps <= 0) {
        throw std::invalid_argument("Direct sequence td_steps must be positive");
    }
    if (!(config.discount >= 0.0f && config.discount <= 1.0f)) {
        throw std::invalid_argument("Direct sequence discount must be in [0,1]");
    }
}

size_t checked_add(size_t left, size_t right, const char* label) {
    if (right > std::numeric_limits<size_t>::max() - left) {
        throw std::overflow_error(std::string(label) + " size overflow");
    }
    return left + right;
}

size_t window_record_count(
    size_t start_step,
    size_t step_count,
    const SequenceTargetConfig& config
) {
    if (start_step > step_count) {
        throw std::out_of_range("Direct sequence start is outside the game");
    }
    const size_t lookahead = checked_add(
        checked_add(
            static_cast<size_t>(config.unroll_steps),
            static_cast<size_t>(config.td_steps),
            "Direct sequence lookahead"),
        1u,
        "Direct sequence lookahead");
    const size_t remaining = step_count - start_step;
    return std::min(remaining, lookahead);
}

BootstrapState bootstrap_from_step(const TrajectoryStep& step) {
    BootstrapState bootstrap;
    bootstrap.root_value = step.root_value;
    bootstrap.observation = step.observation;
    bootstrap.policy_target = step.policy_target;
    bootstrap.legal_mask = step.legal_mask;
    return bootstrap;
}

struct CachedGameWindow {
    ReplayGameMetadata metadata;
    std::map<size_t, TrajectoryStep> steps;
    std::optional<BootstrapState> bootstrap;
};

KStepSample build_sample_from_cache(
    const CachedGameWindow& cache,
    size_t global_game_index,
    size_t start_step,
    const SequenceTargetConfig& config
) {
    const size_t step_count = cache.metadata.step_count;
    if (start_step > step_count
        || (start_step == step_count && !cache.metadata.has_bootstrap_state)) {
        throw std::out_of_range("Direct sequence sample position is outside the persisted roots");
    }

    GameHistory window;
    window.seed = cache.metadata.seed;
    window.wave_mode = cache.metadata.wave_mode;
    window.total_reward = cache.metadata.total_reward;

    if (start_step == step_count) {
        if (!cache.bootstrap) {
            throw std::runtime_error("Direct sequence cache is missing the cutoff bootstrap root");
        }
        window.max_steps = 0;
        window.truncated = true;
        window.bootstrap_state = *cache.bootstrap;
    } else {
        const size_t record_count = window_record_count(
            start_step, step_count, config);
        const size_t end_step = start_step + record_count;
        std::vector<TrajectoryStep> records;
        records.reserve(record_count);
        for (size_t step_index = start_step; step_index < end_step; ++step_index) {
            const auto iterator = cache.steps.find(step_index);
            if (iterator == cache.steps.end()) {
                throw std::runtime_error("Direct sequence cache is missing a requested step");
            }
            records.push_back(iterator->second);
        }

        const bool reaches_episode_end = end_step == step_count;
        if (reaches_episode_end) {
            window.terminal = cache.metadata.terminal;
            window.truncated = cache.metadata.truncated;
            window.max_steps = window.truncated
                ? static_cast<int>(records.size())
                : cache.metadata.max_steps;
            window.steps = std::move(records);
            if (window.truncated) {
                if (!cache.bootstrap) {
                    throw std::runtime_error(
                        "Direct truncated sequence is missing its cutoff bootstrap root");
                }
                window.bootstrap_state = *cache.bootstrap;
            }
        } else {
            if (records.empty()) {
                throw std::runtime_error("Direct synthetic window has no bootstrap record");
            }
            const TrajectoryStep bootstrap_record = records.back();
            records.pop_back();
            window.truncated = true;
            window.max_steps = static_cast<int>(records.size());
            window.steps = std::move(records);
            window.bootstrap_state = bootstrap_from_step(bootstrap_record);
        }

        for (size_t index = 0; index < window.steps.size(); ++index) {
            window.steps[index].step_index = static_cast<int>(index);
        }
    }

    KStepSample sample = build_k_step_sample(window, 0, config);
    sample.global_game_index = global_game_index;
    sample.start_step = start_step;
    sample.episode_terminal = cache.metadata.terminal;
    sample.episode_truncated = cache.metadata.truncated;
    return sample;
}

} // namespace

DirectSequenceReplayDataset::DirectSequenceReplayDataset(
    const TrainingReplayDataset& source
) : wave_mode_(source.wave_mode()) {
    if (!wave_mode_is_known(wave_mode_)) {
        throw std::runtime_error("Direct sequence replay requires explicit wave mode provenance");
    }
    if (!source.replay().transition_indexed()
        || source.replay().replay_format_version() != kReplayBinaryFormatVersion) {
        throw std::runtime_error("Direct sequence replay requires current transition shard v3 data");
    }

    const auto& paths = source.replay().shard_paths();
    readers_.reserve(paths.size());
    cumulative_games_.reserve(paths.size());
    size_t total_games = 0;
    for (const std::string& path : paths) {
        auto reader = std::make_unique<TransitionShardReader>(path);
        if (reader->observation_size() != kObservationSize
            || reader->policy_size() != kActionSpaceSize
            || reader->legal_mask_size() != kActionSpaceSize) {
            throw std::runtime_error("Direct sequence shard tensor dimensions are not current");
        }

        for (size_t game = 0; game < reader->history_count(); ++game) {
            const ReplayGameMetadata metadata = reader->game_metadata(game);
            if (metadata.wave_mode != wave_mode_) {
                throw std::runtime_error(
                    "Direct sequence shard wave mode disagrees with training provenance");
            }
            if (metadata.terminal == metadata.truncated) {
                throw std::runtime_error(
                    "Direct sequence game must be exactly one of terminal or truncated");
            }
            if (metadata.truncated && !metadata.has_bootstrap_state) {
                throw std::runtime_error(
                    "Direct sequence truncated game is missing a cutoff bootstrap root");
            }
            if (metadata.terminal && metadata.has_bootstrap_state) {
                throw std::runtime_error(
                    "Direct sequence terminal game must not contain a cutoff bootstrap root");
            }
        }

        total_games = checked_add(total_games, reader->history_count(), "Direct game index");
        cumulative_games_.push_back(total_games);
        readers_.push_back(std::move(reader));
    }
    if (total_games == 0) {
        throw std::runtime_error("Direct sequence replay contains zero games");
    }
}

DirectSequenceReplayDataset::~DirectSequenceReplayDataset() = default;
DirectSequenceReplayDataset::DirectSequenceReplayDataset(
    DirectSequenceReplayDataset&&) noexcept = default;
DirectSequenceReplayDataset& DirectSequenceReplayDataset::operator=(
    DirectSequenceReplayDataset&&) noexcept = default;

size_t DirectSequenceReplayDataset::shard_count() const {
    return readers_.size();
}

size_t DirectSequenceReplayDataset::game_count() const {
    return cumulative_games_.empty() ? 0u : cumulative_games_.back();
}

ReplaySampleRef DirectSequenceReplayDataset::locate_game(
    size_t global_game_index
) const {
    if (global_game_index >= game_count()) {
        throw std::out_of_range("Direct sequence game index is out of range");
    }
    const auto iterator = std::upper_bound(
        cumulative_games_.begin(), cumulative_games_.end(), global_game_index);
    const size_t shard_index = static_cast<size_t>(
        iterator - cumulative_games_.begin());
    const size_t shard_base = shard_index == 0
        ? 0u
        : cumulative_games_[shard_index - 1];

    ReplaySampleRef ref;
    ref.global_game_index = global_game_index;
    ref.shard_index = shard_index;
    ref.local_game_index = global_game_index - shard_base;
    return ref;
}

ReplayGameMetadata DirectSequenceReplayDataset::game_metadata(
    size_t global_game_index
) const {
    const ReplaySampleRef ref = locate_game(global_game_index);
    return readers_[ref.shard_index]->game_metadata(ref.local_game_index);
}

TrajectoryStep DirectSequenceReplayDataset::read_step(
    size_t global_game_index,
    size_t step_index
) {
    const ReplaySampleRef ref = locate_game(global_game_index);
    return readers_[ref.shard_index]->read_step(ref.local_game_index, step_index);
}

std::vector<TrajectoryStep> DirectSequenceReplayDataset::read_step_range(
    size_t global_game_index,
    size_t start_step,
    size_t count
) {
    const ReplayGameMetadata metadata = game_metadata(global_game_index);
    if (start_step > metadata.step_count
        || count > metadata.step_count - start_step) {
        throw std::out_of_range("Direct sequence step range is outside the game");
    }

    std::vector<TrajectoryStep> result;
    result.reserve(count);
    for (size_t offset = 0; offset < count; ++offset) {
        result.push_back(read_step(global_game_index, start_step + offset));
    }
    return result;
}

BootstrapState DirectSequenceReplayDataset::read_bootstrap_state(
    size_t global_game_index
) {
    const ReplaySampleRef ref = locate_game(global_game_index);
    return readers_[ref.shard_index]->read_bootstrap_state(ref.local_game_index);
}

ReplayIoStats DirectSequenceReplayDataset::io_stats() const {
    ReplayIoStats result;
    for (const auto& reader : readers_) {
        const ReplayIoStats current = reader->io_stats();
        result.physical_read_operations += current.physical_read_operations;
        result.physical_bytes_read += current.physical_bytes_read;
    }
    return result;
}

void DirectSequenceReplayDataset::reset_io_stats() {
    for (auto& reader : readers_) reader->reset_io_stats();
}

DirectPositionIndex::DirectPositionIndex(
    const DirectSequenceReplayDataset& dataset
) : dataset_(&dataset) {
    cumulative_positions_by_game_.reserve(dataset.game_count());
    for (size_t game = 0; game < dataset.game_count(); ++game) {
        const ReplayGameMetadata metadata = dataset.game_metadata(game);
        size_t positions = metadata.step_count;
        if (metadata.has_bootstrap_state) {
            positions = checked_add(positions, 1u, "Direct position count");
        }
        total_positions_ = checked_add(
            total_positions_, positions, "Direct position index");
        cumulative_positions_by_game_.push_back(total_positions_);
    }
    if (total_positions_ == 0) {
        throw std::runtime_error("Direct sequence replay contains zero persisted roots");
    }
}

ReplaySampleRef DirectPositionIndex::locate(
    size_t global_position_index
) const {
    if (global_position_index >= total_positions_) {
        throw std::out_of_range("Direct global position index is out of range");
    }
    const auto iterator = std::upper_bound(
        cumulative_positions_by_game_.begin(),
        cumulative_positions_by_game_.end(),
        global_position_index);
    const size_t global_game_index = static_cast<size_t>(
        iterator - cumulative_positions_by_game_.begin());
    const size_t game_base = global_game_index == 0
        ? 0u
        : cumulative_positions_by_game_[global_game_index - 1];

    ReplaySampleRef ref = dataset_->locate_game(global_game_index);
    ref.step_index = global_position_index - game_base;
    const ReplayGameMetadata metadata = dataset_->game_metadata(global_game_index);
    if (ref.step_index < metadata.step_count) {
        ref.physical_offset = 0u;
    } else if (ref.step_index != metadata.step_count
               || !metadata.has_bootstrap_state) {
        throw std::runtime_error("Direct position index disagrees with game metadata");
    }
    return ref;
}

ReplaySampleRef DirectPositionIndex::sample(uint64_t random_u64) const {
    return locate(static_cast<size_t>(random_u64 % total_positions_));
}

KStepBatch build_reference_k_step_batch(
    TrainingReplayDataset& dataset,
    const std::vector<ReplaySampleRef>& refs,
    const SequenceTargetConfig& config
) {
    validate_config(config);
    if (refs.empty()) {
        throw std::invalid_argument("Reference K-step batch requires at least one sample");
    }

    std::map<size_t, GameHistory> games;
    std::vector<KStepSample> samples;
    samples.reserve(refs.size());
    for (const ReplaySampleRef& ref : refs) {
        auto iterator = games.find(ref.global_game_index);
        if (iterator == games.end()) {
            iterator = games.emplace(
                ref.global_game_index,
                dataset.read_game(ref.global_game_index)).first;
        }
        const size_t state_count = iterator->second.steps.size()
            + (iterator->second.bootstrap_state ? 1u : 0u);
        if (ref.step_index >= state_count) {
            throw std::out_of_range("Reference K-step sample position is outside the game");
        }
        KStepSample sample = build_k_step_sample(
            iterator->second, ref.step_index, config);
        sample.global_game_index = ref.global_game_index;
        samples.push_back(std::move(sample));
    }
    return pack_k_step_samples(samples);
}

KStepBatch build_direct_k_step_batch(
    DirectSequenceReplayDataset& dataset,
    const std::vector<ReplaySampleRef>& refs,
    const SequenceTargetConfig& config,
    DirectSequenceBatchStats* stats
) {
    validate_config(config);
    if (refs.empty()) {
        throw std::invalid_argument("Direct K-step batch requires at least one sample");
    }

    DirectSequenceBatchStats local_stats;
    const ReplayIoStats io_before = dataset.io_stats();
    std::set<size_t> unique_games;
    std::set<size_t> unique_shards;
    std::map<size_t, CachedGameWindow> cache;
    std::map<size_t, std::vector<std::pair<size_t, size_t>>> intervals;
    std::set<size_t> bootstrap_games;

    for (const ReplaySampleRef& requested_ref : refs) {
        const ReplaySampleRef ref = dataset.locate_game(
            requested_ref.global_game_index);
        const ReplayGameMetadata metadata = dataset.game_metadata(
            ref.global_game_index);
        if (requested_ref.step_index > metadata.step_count
            || (requested_ref.step_index == metadata.step_count
                && !metadata.has_bootstrap_state)) {
            throw std::out_of_range("Direct K-step sample position is outside the game");
        }

        unique_games.insert(ref.global_game_index);
        unique_shards.insert(ref.shard_index);
        cache.try_emplace(ref.global_game_index).first->second.metadata = metadata;

        const size_t count = window_record_count(
            requested_ref.step_index, metadata.step_count, config);
        local_stats.requested_step_records = checked_add(
            static_cast<size_t>(local_stats.requested_step_records),
            count,
            "Requested direct step count");
        if (count != 0u) {
            intervals[ref.global_game_index].push_back({
                requested_ref.step_index,
                requested_ref.step_index + count});
        }
        if (metadata.has_bootstrap_state
            && requested_ref.step_index + count == metadata.step_count) {
            bootstrap_games.insert(ref.global_game_index);
        }
    }

    for (auto& game_intervals : intervals) {
        auto& ranges = game_intervals.second;
        std::sort(ranges.begin(), ranges.end());
        std::vector<std::pair<size_t, size_t>> merged;
        for (const auto& range : ranges) {
            if (merged.empty() || range.first > merged.back().second) {
                merged.push_back(range);
            } else {
                merged.back().second = std::max(merged.back().second, range.second);
            }
        }

        CachedGameWindow& game_cache = cache.at(game_intervals.first);
        for (const auto& range : merged) {
            const size_t count = range.second - range.first;
            std::vector<TrajectoryStep> steps = dataset.read_step_range(
                game_intervals.first, range.first, count);
            local_stats.unique_step_records_read += static_cast<uint64_t>(steps.size());
            for (size_t offset = 0; offset < steps.size(); ++offset) {
                game_cache.steps.emplace(range.first + offset, std::move(steps[offset]));
            }
        }
    }

    for (size_t game_index : bootstrap_games) {
        cache.at(game_index).bootstrap = dataset.read_bootstrap_state(game_index);
        ++local_stats.bootstrap_records_read;
    }

    std::vector<KStepSample> samples;
    samples.reserve(refs.size());
    for (const ReplaySampleRef& ref : refs) {
        samples.push_back(build_sample_from_cache(
            cache.at(ref.global_game_index),
            ref.global_game_index,
            ref.step_index,
            config));
    }

    const ReplayIoStats io_after = dataset.io_stats();
    local_stats.unique_games_touched = static_cast<int>(unique_games.size());
    local_stats.unique_shards_touched = static_cast<int>(unique_shards.size());
    local_stats.physical_read_operations =
        io_after.physical_read_operations - io_before.physical_read_operations;
    local_stats.physical_bytes_read =
        io_after.physical_bytes_read - io_before.physical_bytes_read;
    if (stats) *stats = local_stats;
    return pack_k_step_samples(samples);
}

DirectKStepReplaySampler::DirectKStepReplaySampler(
    DirectSequenceReplayDataset& dataset,
    SequenceTargetConfig config,
    uint64_t seed
) : dataset_(dataset),
    config_(config),
    positions_(dataset),
    rng_state_(seed ? seed : 0x9e3779b97f4a7c15ULL) {
    validate_config(config_);
}

uint64_t DirectKStepReplaySampler::next_u64() {
    uint64_t value = rng_state_;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    rng_state_ = value;
    return value;
}

std::vector<ReplaySampleRef> DirectKStepReplaySampler::sample_refs(
    int batch_size
) {
    if (batch_size <= 0) {
        throw std::invalid_argument("Direct K-step replay batch size must be positive");
    }
    std::vector<ReplaySampleRef> refs;
    refs.reserve(static_cast<size_t>(batch_size));
    for (int index = 0; index < batch_size; ++index) {
        refs.push_back(positions_.sample(next_u64()));
    }
    return refs;
}

KStepBatch DirectKStepReplaySampler::sample_batch(int batch_size) {
    const std::vector<ReplaySampleRef> refs = sample_refs(batch_size);
    return build_direct_k_step_batch(dataset_, refs, config_, &last_stats_);
}

} // namespace tdmz
