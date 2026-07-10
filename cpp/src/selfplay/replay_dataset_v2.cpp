#include "tdmz/selfplay/replay_dataset.hpp"

#include "tdmz/selfplay/trajectory_writer.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tdmz {

class ReplayShardReaderBase {
public:
    virtual ~ReplayShardReaderBase() = default;
    virtual uint32_t format_version() const = 0;
    virtual bool direct_transition_reads() const = 0;
    virtual bool io_stats_exact() const = 0;
    virtual size_t history_count() const = 0;
    virtual size_t step_count(size_t game_index) const = 0;
    virtual uint64_t step_physical_offset(size_t game_index, size_t step_index) const = 0;
    virtual int observation_size() const = 0;
    virtual int policy_size() const = 0;
    virtual int legal_mask_size() const = 0;
    virtual GameHistory read_game(size_t game_index) = 0;
    virtual TrajectoryStep read_step(size_t game_index, size_t step_index) = 0;
    virtual void read_step_into(
        size_t game_index,
        size_t step_index,
        const ReplayStepDestination& destination
    ) = 0;
    virtual ReplayIoStats io_stats() const = 0;
    virtual void reset_io_stats() = 0;
};

namespace {

struct PendingReplaySample {
    size_t global_game_index = 0;
    size_t output_index = 0;
    uint64_t random_value = 0;
    size_t shard_index = 0;
};

struct PendingDirectSample {
    ReplaySampleRef ref;
    size_t output_index = 0;
};

class LegacyV1ShardReader final : public ReplayShardReaderBase {
public:
    explicit LegacyV1ShardReader(const std::string& path) : reader_(path) {}

    uint32_t format_version() const override { return kLegacyReplayBinaryFormatVersion; }
    bool direct_transition_reads() const override { return false; }
    bool io_stats_exact() const override { return false; }
    size_t history_count() const override { return reader_.history_count(); }

    size_t step_count(size_t game_index) const override {
        BinaryShardReader reader(reader_.path());
        return reader.read_at(game_index).steps.size();
    }

    uint64_t step_physical_offset(size_t, size_t) const override {
        throw std::runtime_error("Replay v1 shards do not expose persisted transition offsets");
    }

    int observation_size() const override { return 0; }
    int policy_size() const override { return 0; }
    int legal_mask_size() const override { return 0; }

    GameHistory read_game(size_t game_index) override {
        ++stats_.physical_read_operations;
        return reader_.read_at(game_index);
    }

    TrajectoryStep read_step(size_t game_index, size_t step_index) override {
        GameHistory history = read_game(game_index);
        if (step_index >= history.steps.size()) {
            throw std::runtime_error("Replay v1 step index out of range");
        }
        return history.steps[step_index];
    }

    void read_step_into(size_t, size_t, const ReplayStepDestination&) override {
        throw std::runtime_error("Replay v1 does not support direct transition reads");
    }

    ReplayIoStats io_stats() const override { return stats_; }
    void reset_io_stats() override { stats_ = ReplayIoStats{}; }

private:
    BinaryShardReader reader_;
    ReplayIoStats stats_;
};

class IndexedV2ShardReader final : public ReplayShardReaderBase {
public:
    explicit IndexedV2ShardReader(const std::string& path) : reader_(path) {}

    uint32_t format_version() const override { return kTransitionShardFormatVersion; }
    bool direct_transition_reads() const override { return true; }
    bool io_stats_exact() const override { return true; }
    size_t history_count() const override { return reader_.history_count(); }
    size_t step_count(size_t game_index) const override { return reader_.step_count(game_index); }
    uint64_t step_physical_offset(size_t game_index, size_t step_index) const override {
        return reader_.step_physical_offset(game_index, step_index);
    }
    int observation_size() const override { return reader_.observation_size(); }
    int policy_size() const override { return reader_.policy_size(); }
    int legal_mask_size() const override { return reader_.legal_mask_size(); }
    GameHistory read_game(size_t game_index) override { return reader_.read_at(game_index); }
    TrajectoryStep read_step(size_t game_index, size_t step_index) override {
        return reader_.read_step(game_index, step_index);
    }
    void read_step_into(
        size_t game_index,
        size_t step_index,
        const ReplayStepDestination& destination
    ) override {
        reader_.read_step_into(game_index, step_index, destination);
    }
    ReplayIoStats io_stats() const override { return reader_.io_stats(); }
    void reset_io_stats() override { reader_.reset_io_stats(); }

private:
    TransitionShardReader reader_;
};

std::string read_text_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open replay index JSON: " + path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string unescape_json_string(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '\\' && i + 1 < value.size()) {
            const char next = value[++i];
            switch (next) {
                case '\\': out.push_back('\\'); break;
                case '"': out.push_back('"'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                default: out.push_back(next); break;
            }
        } else {
            out.push_back(value[i]);
        }
    }
    return out;
}

std::optional<std::string> find_string_field(const std::string& text, const char* field) {
    const std::regex pattern(
        std::string("\\\"") + field + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) return std::nullopt;
    return unescape_json_string(match[1].str());
}

std::optional<uint64_t> find_u64_field(const std::string& text, const char* field) {
    const std::regex pattern(
        std::string("\\\"") + field + "\\\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) return std::nullopt;
    try {
        return static_cast<uint64_t>(std::stoull(match[1].str()));
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Replay index invalid integer field '") + field + "'");
    }
}

uint32_t require_u32_field(const std::string& text, const char* field) {
    const auto value = find_u64_field(text, field);
    if (!value) throw std::runtime_error(std::string("Replay index missing field '") + field + "'");
    if (*value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string("Replay index field '") + field + "' exceeds uint32 range");
    }
    return static_cast<uint32_t>(*value);
}

int require_int_field(const std::string& text, const char* field) {
    const auto value = find_u64_field(text, field);
    if (!value) throw std::runtime_error(std::string("Replay index missing field '") + field + "'");
    if (*value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("Replay index field '") + field + "' exceeds int range");
    }
    return static_cast<int>(*value);
}

std::vector<std::string> parse_shard_paths(
    const std::string& text,
    const std::string& index_path
) {
    std::vector<std::string> paths;
    const std::filesystem::path parent = std::filesystem::path(index_path).parent_path();
    const std::regex path_pattern("\\\"path\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
    const auto end = std::sregex_iterator();
    for (auto it = std::sregex_iterator(text.begin(), text.end(), path_pattern); it != end; ++it) {
        std::filesystem::path path(unescape_json_string((*it)[1].str()));
        if (path.is_relative() && !std::filesystem::exists(path) && !parent.empty()) {
            const auto relative = parent / path;
            if (std::filesystem::exists(relative)) path = relative;
        }
        paths.push_back(path.lexically_normal().string());
    }
    if (paths.empty()) {
        throw std::runtime_error("No shard paths found in replay index JSON: " + index_path);
    }
    return paths;
}

ReplayIndexMetadata parse_index_metadata(
    const std::string& text,
    const std::string& index_path,
    LegacyReplayIndexPolicy legacy_policy
) {
    const auto format = find_string_field(text, "format");
    const auto version = find_u64_field(text, "version");

    if (!format || !version) {
        if (legacy_policy != LegacyReplayIndexPolicy::AllowV1) {
            throw std::runtime_error(
                "Replay index lacks explicit format/version metadata; pass AllowV1 only for a reviewed legacy index: "
                + index_path);
        }
        ReplayIndexMetadata metadata;
        metadata.format = format.value_or("");
        metadata.index_version = version ? static_cast<uint32_t>(*version) : 1u;
        metadata.legacy_v1 = true;
        metadata.compatibility.replay_format_version = kLegacyReplayBinaryFormatVersion;
        return metadata;
    }

    if (*format != kReplayIndexFormat) {
        throw std::runtime_error(
            "Replay index incompatible field 'format': actual=" + *format
            + " expected=" + kReplayIndexFormat);
    }
    if (*version > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Replay index field 'version' exceeds uint32 range");
    }

    ReplayIndexMetadata metadata;
    metadata.format = *format;
    metadata.index_version = static_cast<uint32_t>(*version);
    if (metadata.index_version == 1) {
        if (legacy_policy != LegacyReplayIndexPolicy::AllowV1) {
            throw std::runtime_error(
                "Replay index version 1 is legacy and has no complete compatibility metadata; "
                "pass AllowV1 only after reviewing its producer configuration: " + index_path);
        }
        metadata.legacy_v1 = true;
        metadata.compatibility.replay_format_version = kLegacyReplayBinaryFormatVersion;
        return metadata;
    }
    if (metadata.index_version != kReplayIndexVersion) {
        std::ostringstream out;
        out << "Replay index incompatible field 'version': actual=" << metadata.index_version
            << " expected=" << kReplayIndexVersion;
        throw std::runtime_error(out.str());
    }

    metadata.compatibility.replay_format_version = require_u32_field(text, "replay_format_version");
    if (metadata.compatibility.replay_format_version != kLegacyReplayBinaryFormatVersion
        && metadata.compatibility.replay_format_version != kReplayBinaryFormatVersion) {
        std::ostringstream out;
        out << "Replay index incompatible field 'replay_format_version': actual="
            << metadata.compatibility.replay_format_version << " supported="
            << kLegacyReplayBinaryFormatVersion << " or " << kReplayBinaryFormatVersion;
        throw std::runtime_error(out.str());
    }
    metadata.compatibility.environment_rule_version = require_u32_field(text, "environment_rule_version");
    metadata.compatibility.observation_schema_version = require_u32_field(text, "observation_schema_version");
    metadata.compatibility.action_space_version = require_u32_field(text, "action_space_version");
    metadata.compatibility.reward_transform_version = require_u32_field(text, "reward_transform_version");
    metadata.compatibility.network_architecture_version = require_u32_field(text, "network_architecture_version");
    metadata.compatibility.board_width = require_int_field(text, "board_width");
    metadata.compatibility.board_height = require_int_field(text, "board_height");
    metadata.compatibility.observation_channels = require_int_field(text, "observation_channels");
    metadata.compatibility.observation_size = require_int_field(text, "observation_size");
    metadata.compatibility.action_space_size = require_int_field(text, "action_space_size");
    metadata.compatibility.policy_size = require_int_field(text, "policy_size");
    metadata.compatibility.legal_mask_size = require_int_field(text, "legal_mask_size");

    CompatibilityMetadata expected = current_compatibility_metadata();
    expected.replay_format_version = metadata.compatibility.replay_format_version;
    validate_compatibility_metadata(metadata.compatibility, expected, "Replay index");
    return metadata;
}

int checked_tensor_size(size_t size, const char* label) {
    if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("Replay ") + label + " exceeds int range");
    }
    return static_cast<int>(size);
}

void ensure_step_tensor_sizes(
    const TrajectoryStep& step,
    bool& initialized,
    int& observation_size,
    int& policy_size,
    int& legal_size
) {
    const int current_observation = checked_tensor_size(step.observation.size(), "observation size");
    const int current_policy = checked_tensor_size(step.policy_target.size(), "policy size");
    const int current_legal = checked_tensor_size(step.legal_mask.size(), "legal-mask size");
    if (!initialized) {
        observation_size = current_observation;
        policy_size = current_policy;
        legal_size = current_legal;
        initialized = true;
    }
    if (current_observation != observation_size
        || current_policy != policy_size
        || current_legal != legal_size) {
        throw std::runtime_error("Replay batch contains inconsistent tensor sizes");
    }
}

void allocate_batch_storage(
    ReplayBatch& batch,
    int batch_size,
    int observation_size,
    int policy_size,
    int legal_size
) {
    if (observation_size < 0 || policy_size < 0 || legal_size < 0) {
        throw std::runtime_error("Replay batch tensor size is negative");
    }
    batch.batch_size = batch_size;
    batch.observation_size = observation_size;
    batch.policy_size = policy_size;
    batch.legal_mask_size = legal_size;
    batch.observations.resize(static_cast<size_t>(batch_size) * observation_size);
    batch.policy_targets.resize(static_cast<size_t>(batch_size) * policy_size);
    batch.legal_masks.resize(static_cast<size_t>(batch_size) * legal_size);
    batch.values.resize(batch_size);
    batch.rewards.resize(batch_size);
    batch.actions.resize(batch_size);
    batch.dones.resize(batch_size);
}

void copy_batch_row(
    ReplayBatch& batch,
    size_t source,
    size_t destination
) {
    if (source == destination) return;
    const size_t observation_size = static_cast<size_t>(batch.observation_size);
    const size_t policy_size = static_cast<size_t>(batch.policy_size);
    const size_t legal_size = static_cast<size_t>(batch.legal_mask_size);
    std::copy_n(
        batch.observations.begin() + source * observation_size,
        observation_size,
        batch.observations.begin() + destination * observation_size);
    std::copy_n(
        batch.policy_targets.begin() + source * policy_size,
        policy_size,
        batch.policy_targets.begin() + destination * policy_size);
    std::copy_n(
        batch.legal_masks.begin() + source * legal_size,
        legal_size,
        batch.legal_masks.begin() + destination * legal_size);
    batch.values[destination] = batch.values[source];
    batch.rewards[destination] = batch.rewards[source];
    batch.actions[destination] = batch.actions[source];
    batch.dones[destination] = batch.dones[source];
}

} // namespace

ReplayIndexInfo load_replay_index_json(
    const std::string& index_path,
    LegacyReplayIndexPolicy legacy_policy
) {
    const std::string text = read_text_file(index_path);
    ReplayIndexInfo result;
    result.index_path = index_path;
    result.metadata = parse_index_metadata(text, index_path, legacy_policy);
    result.shard_paths = parse_shard_paths(text, index_path);
    return result;
}

ReplayDataset::ReplayDataset(std::vector<std::string> shard_paths)
    : shard_paths_(std::move(shard_paths)) {
    open_shards();
}

ReplayDataset::ReplayDataset(
    std::vector<std::string> shard_paths,
    ReplayIndexMetadata index_metadata
) : shard_paths_(std::move(shard_paths)),
    index_metadata_(std::move(index_metadata)),
    has_index_metadata_(true) {
    open_shards();
}

ReplayDataset::~ReplayDataset() = default;
ReplayDataset::ReplayDataset(ReplayDataset&&) noexcept = default;
ReplayDataset& ReplayDataset::operator=(ReplayDataset&&) noexcept = default;

void ReplayDataset::open_shards() {
    if (shard_paths_.empty()) throw std::runtime_error("ReplayDataset requires at least one shard path");

    readers_.reserve(shard_paths_.size());
    cumulative_games_.reserve(shard_paths_.size());
    size_t total_games = 0;
    uint32_t detected_version = 0;

    for (const auto& path : shard_paths_) {
        std::unique_ptr<ReplayShardReaderBase> reader;
        if (is_transition_shard_v2(path)) {
            reader = std::make_unique<IndexedV2ShardReader>(path);
        } else {
            reader = std::make_unique<LegacyV1ShardReader>(path);
        }

        if (detected_version == 0) detected_version = reader->format_version();
        if (reader->format_version() != detected_version) {
            throw std::runtime_error("ReplayDataset does not allow mixed replay shard format versions");
        }

        if (reader->direct_transition_reads()) {
            if (readers_.empty()) {
                observation_size_ = reader->observation_size();
                policy_size_ = reader->policy_size();
                legal_mask_size_ = reader->legal_mask_size();
            } else if (reader->observation_size() != observation_size_
                       || reader->policy_size() != policy_size_
                       || reader->legal_mask_size() != legal_mask_size_) {
                throw std::runtime_error("Transition replay shards have inconsistent tensor sizes");
            }
        }

        total_games += reader->history_count();
        cumulative_games_.push_back(total_games);
        readers_.push_back(std::move(reader));
    }

    if (total_games == 0) throw std::runtime_error("ReplayDataset contains zero games");
    replay_format_version_ = detected_version;
    transition_indexed_ = detected_version == kTransitionShardFormatVersion;

    if (has_index_metadata_ && !index_metadata_.legacy_v1) {
        if (index_metadata_.compatibility.replay_format_version != replay_format_version_) {
            std::ostringstream out;
            out << "Replay index/shard format mismatch: index="
                << index_metadata_.compatibility.replay_format_version
                << " shard=" << replay_format_version_;
            throw std::runtime_error(out.str());
        }
        if (transition_indexed_) {
            if (index_metadata_.compatibility.observation_size != observation_size_
                || index_metadata_.compatibility.policy_size != policy_size_
                || index_metadata_.compatibility.legal_mask_size != legal_mask_size_) {
                throw std::runtime_error("Replay index tensor dimensions do not match transition shard header");
            }
        }
    }
}

ReplayDataset ReplayDataset::from_index_json(
    const std::string& index_path,
    LegacyReplayIndexPolicy legacy_policy
) {
    ReplayIndexInfo info = load_replay_index_json(index_path, legacy_policy);
    return ReplayDataset(std::move(info.shard_paths), std::move(info.metadata));
}

size_t ReplayDataset::shard_count() const {
    return readers_.size();
}

size_t ReplayDataset::game_count() const {
    return cumulative_games_.empty() ? 0 : cumulative_games_.back();
}

const std::vector<std::string>& ReplayDataset::shard_paths() const {
    return shard_paths_;
}

ReplaySampleRef ReplayDataset::locate_game(size_t global_game_index) const {
    if (global_game_index >= game_count()) {
        throw std::runtime_error("ReplayDataset game index out of range");
    }
    const auto it = std::upper_bound(
        cumulative_games_.begin(), cumulative_games_.end(), global_game_index);
    const size_t shard_index = static_cast<size_t>(it - cumulative_games_.begin());
    const size_t shard_base = shard_index == 0 ? 0 : cumulative_games_[shard_index - 1];
    ReplaySampleRef result;
    result.global_game_index = global_game_index;
    result.shard_index = shard_index;
    result.local_game_index = global_game_index - shard_base;
    return result;
}

GameHistory ReplayDataset::read_game(size_t global_game_index) {
    const ReplaySampleRef ref = locate_game(global_game_index);
    ++game_read_count_;
    return readers_[ref.shard_index]->read_game(ref.local_game_index);
}

size_t ReplayDataset::step_count(size_t global_game_index) const {
    const ReplaySampleRef ref = locate_game(global_game_index);
    return readers_[ref.shard_index]->step_count(ref.local_game_index);
}

ReplaySampleRef ReplayDataset::sample_ref(uint64_t random_u64) {
    ReplaySampleRef result = locate_game(static_cast<size_t>(random_u64 % game_count()));
    if (transition_indexed_) {
        const size_t count = readers_[result.shard_index]->step_count(result.local_game_index);
        if (count == 0) throw std::runtime_error("ReplayDataset sampled empty GameHistory");
        result.step_index = static_cast<size_t>((random_u64 >> 32) % count);
        result.physical_offset = readers_[result.shard_index]->step_physical_offset(
            result.local_game_index, result.step_index);
        return result;
    }

    GameHistory history = read_game(result.global_game_index);
    if (history.steps.empty()) throw std::runtime_error("ReplayDataset sampled empty GameHistory");
    result.step_index = static_cast<size_t>((random_u64 >> 32) % history.steps.size());
    return result;
}

TrajectoryStep ReplayDataset::sample_step(uint64_t random_u64) {
    ReplaySampleRef ref = sample_ref(random_u64);
    if (transition_indexed_) {
        return readers_[ref.shard_index]->read_step(ref.local_game_index, ref.step_index);
    }
    GameHistory history = read_game(ref.global_game_index);
    if (ref.step_index >= history.steps.size()) {
        throw std::runtime_error("ReplayDataset sampled step index out of range");
    }
    return history.steps[ref.step_index];
}

void ReplayDataset::read_step_into_batch(
    const ReplaySampleRef& ref,
    ReplayBatch& batch,
    size_t output_index
) {
    if (!transition_indexed_) {
        throw std::runtime_error("Direct replay step reads require transition-indexed v2 shards");
    }
    if (ref.shard_index >= readers_.size()) throw std::runtime_error("Replay shard index out of range");
    if (output_index >= static_cast<size_t>(batch.batch_size)) {
        throw std::runtime_error("Replay batch output index out of range");
    }
    if (batch.observation_size != observation_size_
        || batch.policy_size != policy_size_
        || batch.legal_mask_size != legal_mask_size_) {
        throw std::runtime_error("Replay batch dimensions do not match transition shard");
    }

    ReplayStepDestination destination;
    destination.observation = batch.observations.data()
        + output_index * static_cast<size_t>(observation_size_);
    destination.observation_size = static_cast<size_t>(observation_size_);
    destination.policy_target = batch.policy_targets.data()
        + output_index * static_cast<size_t>(policy_size_);
    destination.policy_size = static_cast<size_t>(policy_size_);
    destination.legal_mask = batch.legal_masks.data()
        + output_index * static_cast<size_t>(legal_mask_size_);
    destination.legal_mask_size = static_cast<size_t>(legal_mask_size_);
    destination.value = &batch.values[output_index];
    destination.reward = &batch.rewards[output_index];
    destination.action = &batch.actions[output_index];
    destination.done = &batch.dones[output_index];
    readers_[ref.shard_index]->read_step_into(
        ref.local_game_index, ref.step_index, destination);
}

ReplayIoStats ReplayDataset::io_stats() const {
    ReplayIoStats total;
    for (const auto& reader : readers_) {
        const ReplayIoStats current = reader->io_stats();
        total.physical_read_operations += current.physical_read_operations;
        total.physical_bytes_read += current.physical_bytes_read;
    }
    return total;
}

void ReplayDataset::reset_io_stats() {
    for (auto& reader : readers_) reader->reset_io_stats();
}

ReplayBatchSampler::ReplayBatchSampler(ReplayDataset& dataset, uint64_t seed)
    : dataset_(dataset), rng_state_(seed ? seed : 0x9e3779b97f4a7c15ULL) {}

uint64_t ReplayBatchSampler::next_u64() {
    uint64_t value = rng_state_;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    rng_state_ = value;
    return value;
}

ReplayBatch ReplayBatchSampler::sample_batch(int batch_size) {
    if (batch_size <= 0) throw std::runtime_error("Replay batch size must be positive");

    const ReplayIoStats io_before = dataset_.io_stats();

    if (dataset_.transition_indexed()) {
        ReplayBatch batch;
        allocate_batch_storage(
            batch,
            batch_size,
            dataset_.observation_size(),
            dataset_.policy_size(),
            dataset_.legal_mask_size());
        batch.direct_transition_reads = true;
        batch.io_stats_exact = true;

        std::vector<PendingDirectSample> pending;
        pending.reserve(batch_size);
        std::set<size_t> unique_games;
        std::set<size_t> unique_shards;
        for (int output = 0; output < batch_size; ++output) {
            const uint64_t random_value = next_u64();
            ReplaySampleRef ref = dataset_.sample_ref(random_value);
            unique_games.insert(ref.global_game_index);
            unique_shards.insert(ref.shard_index);
            pending.push_back({ref, static_cast<size_t>(output)});
        }
        std::sort(
            pending.begin(), pending.end(),
            [](const PendingDirectSample& left, const PendingDirectSample& right) {
                if (left.ref.shard_index != right.ref.shard_index) {
                    return left.ref.shard_index < right.ref.shard_index;
                }
                if (left.ref.physical_offset != right.ref.physical_offset) {
                    return left.ref.physical_offset < right.ref.physical_offset;
                }
                return left.output_index < right.output_index;
            });

        size_t begin = 0;
        while (begin < pending.size()) {
            size_t end = begin + 1;
            while (end < pending.size()
                   && pending[end].ref.shard_index == pending[begin].ref.shard_index
                   && pending[end].ref.physical_offset == pending[begin].ref.physical_offset) {
                ++end;
            }
            dataset_.read_step_into_batch(
                pending[begin].ref, batch, pending[begin].output_index);
            for (size_t i = begin + 1; i < end; ++i) {
                copy_batch_row(batch, pending[begin].output_index, pending[i].output_index);
            }
            begin = end;
        }

        batch.unique_games_touched = static_cast<int>(unique_games.size());
        batch.unique_shards_touched = static_cast<int>(unique_shards.size());
        const ReplayIoStats io_after = dataset_.io_stats();
        batch.physical_read_operations =
            io_after.physical_read_operations - io_before.physical_read_operations;
        batch.physical_bytes_read = io_after.physical_bytes_read - io_before.physical_bytes_read;
        return batch;
    }

    std::vector<PendingReplaySample> pending;
    pending.reserve(batch_size);
    std::set<size_t> unique_shards;
    for (int output = 0; output < batch_size; ++output) {
        const uint64_t random_value = next_u64();
        const size_t global_game = static_cast<size_t>(random_value % dataset_.game_count());
        const ReplaySampleRef location = dataset_.locate_game(global_game);
        unique_shards.insert(location.shard_index);
        pending.push_back({global_game, static_cast<size_t>(output), random_value, location.shard_index});
    }
    std::sort(
        pending.begin(), pending.end(),
        [](const PendingReplaySample& left, const PendingReplaySample& right) {
            if (left.global_game_index != right.global_game_index) {
                return left.global_game_index < right.global_game_index;
            }
            return left.output_index < right.output_index;
        });

    ReplayBatch batch;
    batch.batch_size = batch_size;
    batch.values.resize(batch_size);
    batch.rewards.resize(batch_size);
    batch.actions.resize(batch_size);
    batch.dones.resize(batch_size);
    batch.direct_transition_reads = false;
    batch.io_stats_exact = false;

    bool tensor_sizes_initialized = false;
    int observation_size = 0;
    int policy_size = 0;
    int legal_size = 0;
    int unique_games = 0;

    size_t begin = 0;
    while (begin < pending.size()) {
        size_t end = begin + 1;
        while (end < pending.size()
               && pending[end].global_game_index == pending[begin].global_game_index) {
            ++end;
        }
        ++unique_games;

        GameHistory history = dataset_.read_game(pending[begin].global_game_index);
        if (history.steps.empty()) throw std::runtime_error("ReplayDataset sampled empty GameHistory");

        for (size_t i = begin; i < end; ++i) {
            const size_t step_index = static_cast<size_t>(
                (pending[i].random_value >> 32) % history.steps.size());
            const TrajectoryStep& step = history.steps[step_index];
            ensure_step_tensor_sizes(
                step,
                tensor_sizes_initialized,
                observation_size,
                policy_size,
                legal_size);
            if (batch.observations.empty() && tensor_sizes_initialized) {
                allocate_batch_storage(
                    batch, batch_size, observation_size, policy_size, legal_size);
                batch.direct_transition_reads = false;
                batch.io_stats_exact = false;
            }

            const size_t output = pending[i].output_index;
            std::copy(
                step.observation.begin(), step.observation.end(),
                batch.observations.begin() + output * static_cast<size_t>(observation_size));
            std::copy(
                step.policy_target.begin(), step.policy_target.end(),
                batch.policy_targets.begin() + output * static_cast<size_t>(policy_size));
            std::copy(
                step.legal_mask.begin(), step.legal_mask.end(),
                batch.legal_masks.begin() + output * static_cast<size_t>(legal_size));
            batch.values[output] = step.root_value;
            batch.rewards[output] = step.reward;
            batch.actions[output] = step.action;
            batch.dones[output] = step.done ? 1u : 0u;
        }
        begin = end;
    }

    batch.unique_games_touched = unique_games;
    batch.unique_shards_touched = static_cast<int>(unique_shards.size());
    const ReplayIoStats io_after = dataset_.io_stats();
    batch.physical_read_operations =
        io_after.physical_read_operations - io_before.physical_read_operations;
    batch.physical_bytes_read = io_after.physical_bytes_read - io_before.physical_bytes_read;
    return batch;
}

uint64_t replay_batch_payload_bytes(const ReplayBatch& batch) {
    uint64_t bytes = 0;
    bytes += static_cast<uint64_t>(batch.observations.size()) * sizeof(float);
    bytes += static_cast<uint64_t>(batch.policy_targets.size()) * sizeof(float);
    bytes += static_cast<uint64_t>(batch.values.size()) * sizeof(float);
    bytes += static_cast<uint64_t>(batch.rewards.size()) * sizeof(float);
    bytes += static_cast<uint64_t>(batch.actions.size()) * sizeof(int);
    bytes += static_cast<uint64_t>(batch.legal_masks.size()) * sizeof(uint8_t);
    bytes += static_cast<uint64_t>(batch.dones.size()) * sizeof(uint8_t);
    return bytes;
}

double replay_batch_checksum(const ReplayBatch& batch) {
    double sum = batch.batch_size * 13.0
        + batch.observation_size * 0.01
        + batch.policy_size * 0.02;
    for (size_t i = 0; i < batch.observations.size(); i += 97) {
        sum += batch.observations[i] * static_cast<double>((i % 31) + 1);
    }
    for (size_t i = 0; i < batch.policy_targets.size(); i += 53) {
        sum += batch.policy_targets[i] * static_cast<double>((i % 17) + 1);
    }
    for (size_t i = 0; i < batch.legal_masks.size(); i += 59) {
        sum += batch.legal_masks[i] ? 0.25 : 0.0;
    }
    for (size_t i = 0; i < batch.values.size(); ++i) {
        sum += batch.values[i] * 5.0;
        sum += batch.rewards[i] * 3.0;
        sum += batch.actions[i] * 0.125;
        sum += batch.dones[i] ? 1.0 : 0.0;
    }
    return sum;
}

} // namespace tdmz
