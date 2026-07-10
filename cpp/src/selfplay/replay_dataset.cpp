#include "tdmz/selfplay/replay_dataset.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tdmz {

namespace {

struct PendingReplaySample {
    size_t global_game_index = 0;
    size_t output_index = 0;
    uint64_t random_value = 0;
};

std::string read_text_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open replay index JSON: " + path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string unescape_json_string(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char n = s[++i];
            switch (n) {
                case '\\': out.push_back('\\'); break;
                case '"': out.push_back('"'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                default: out.push_back(n); break;
            }
        } else {
            out.push_back(s[i]);
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
    if (!value) {
        throw std::runtime_error(std::string("Replay index missing field '") + field + "'");
    }
    if (*value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string("Replay index field '") + field + "' exceeds uint32 range");
    }
    return static_cast<uint32_t>(*value);
}

int require_int_field(const std::string& text, const char* field) {
    const auto value = find_u64_field(text, field);
    if (!value) {
        throw std::runtime_error(std::string("Replay index missing field '") + field + "'");
    }
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
    const std::filesystem::path index_parent = std::filesystem::path(index_path).parent_path();
    const std::regex path_re("\\\"path\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
    auto begin = std::sregex_iterator(text.begin(), text.end(), path_re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        std::filesystem::path shard_path(unescape_json_string((*it)[1].str()));
        if (shard_path.is_relative() && !std::filesystem::exists(shard_path) && !index_parent.empty()) {
            const std::filesystem::path relative_to_index = index_parent / shard_path;
            if (std::filesystem::exists(relative_to_index)) {
                shard_path = relative_to_index;
            }
        }
        paths.push_back(shard_path.lexically_normal().string());
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
                "Replay index lacks explicit format/version metadata; pass AllowV1 only for a reviewed legacy index: " +
                index_path);
        }
        ReplayIndexMetadata metadata;
        metadata.format = format.value_or("");
        metadata.index_version = version ? static_cast<uint32_t>(*version) : 1u;
        metadata.legacy_v1 = true;
        return metadata;
    }

    if (*format != kReplayIndexFormat) {
        throw std::runtime_error(
            "Replay index incompatible field 'format': actual=" + *format +
            " expected=" + kReplayIndexFormat);
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
        return metadata;
    }

    if (metadata.index_version != kReplayIndexVersion) {
        std::ostringstream out;
        out << "Replay index incompatible field 'version': actual=" << metadata.index_version
            << " expected=" << kReplayIndexVersion;
        throw std::runtime_error(out.str());
    }

    metadata.compatibility.replay_format_version = require_u32_field(text, "replay_format_version");
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

    validate_compatibility_metadata(
        metadata.compatibility,
        current_compatibility_metadata(),
        "Replay index");
    return metadata;
}

int checked_tensor_size(size_t size, const char* name) {
    if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("Replay ") + name + " exceeds int range");
    }
    return static_cast<int>(size);
}

void ensure_step_tensor_sizes(
    const TrajectoryStep& step,
    bool& initialized,
    int& obs_size,
    int& policy_size,
    int& mask_size
) {
    const int current_obs_size = checked_tensor_size(step.observation.size(), "observation size");
    const int current_policy_size = checked_tensor_size(step.policy_target.size(), "policy size");
    const int current_mask_size = checked_tensor_size(step.legal_mask.size(), "legal-mask size");

    if (!initialized) {
        obs_size = current_obs_size;
        policy_size = current_policy_size;
        mask_size = current_mask_size;
        initialized = true;
    }

    if (current_obs_size != obs_size ||
        current_policy_size != policy_size ||
        current_mask_size != mask_size) {
        throw std::runtime_error("Replay batch contains inconsistent tensor sizes");
    }
}

} // namespace

ReplayIndexInfo load_replay_index_json(
    const std::string& index_path,
    LegacyReplayIndexPolicy legacy_policy
) {
    const std::string text = read_text_file(index_path);
    ReplayIndexInfo info;
    info.index_path = index_path;
    info.metadata = parse_index_metadata(text, index_path, legacy_policy);
    info.shard_paths = parse_shard_paths(text, index_path);
    return info;
}

ReplayDataset::ReplayDataset(std::vector<std::string> shard_paths)
    : shard_paths_(std::move(shard_paths)) {
    open_shards();
}

ReplayDataset::ReplayDataset(
    std::vector<std::string> shard_paths,
    ReplayIndexMetadata index_metadata
)
    : shard_paths_(std::move(shard_paths)),
      index_metadata_(std::move(index_metadata)),
      has_index_metadata_(true) {
    open_shards();
}

void ReplayDataset::open_shards() {
    if (shard_paths_.empty()) {
        throw std::runtime_error("ReplayDataset requires at least one shard path");
    }

    readers_.reserve(shard_paths_.size());
    cumulative_games_.reserve(shard_paths_.size());
    size_t total = 0;
    for (const auto& path : shard_paths_) {
        auto reader = std::make_unique<BinaryShardReader>(path);
        total += reader->history_count();
        cumulative_games_.push_back(total);
        readers_.push_back(std::move(reader));
    }
    if (total == 0) throw std::runtime_error("ReplayDataset contains zero games");
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

GameHistory ReplayDataset::read_game(size_t global_game_index) {
    if (global_game_index >= game_count()) {
        throw std::runtime_error("ReplayDataset game index out of range");
    }
    auto it = std::upper_bound(cumulative_games_.begin(), cumulative_games_.end(), global_game_index);
    size_t shard_id = static_cast<size_t>(it - cumulative_games_.begin());
    size_t shard_base = shard_id == 0 ? 0 : cumulative_games_[shard_id - 1];
    size_t local_index = global_game_index - shard_base;
    ++game_read_count_;
    return readers_[shard_id]->read_at(local_index);
}

ReplaySampleRef ReplayDataset::sample_ref(uint64_t random_u64) {
    size_t global_game_index = static_cast<size_t>(random_u64 % game_count());
    auto it = std::upper_bound(cumulative_games_.begin(), cumulative_games_.end(), global_game_index);
    size_t shard_id = static_cast<size_t>(it - cumulative_games_.begin());
    size_t shard_base = shard_id == 0 ? 0 : cumulative_games_[shard_id - 1];
    size_t local_index = global_game_index - shard_base;
    GameHistory history = read_game(global_game_index);
    if (history.steps.empty()) throw std::runtime_error("ReplayDataset sampled empty GameHistory");
    size_t step_index = static_cast<size_t>((random_u64 >> 32) % history.steps.size());

    ReplaySampleRef ref;
    ref.shard_index = shard_id;
    ref.local_game_index = local_index;
    ref.step_index = step_index;
    return ref;
}

TrajectoryStep ReplayDataset::sample_step(uint64_t random_u64) {
    size_t global_game_index = static_cast<size_t>(random_u64 % game_count());
    GameHistory history = read_game(global_game_index);
    if (history.steps.empty()) throw std::runtime_error("ReplayDataset sampled empty GameHistory");
    size_t step_index = static_cast<size_t>((random_u64 >> 32) % history.steps.size());
    return history.steps[step_index];
}

ReplayBatchSampler::ReplayBatchSampler(ReplayDataset& dataset, uint64_t seed)
    : dataset_(dataset), rng_state_(seed ? seed : 0x9e3779b97f4a7c15ULL) {}

uint64_t ReplayBatchSampler::next_u64() {
    uint64_t x = rng_state_;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state_ = x;
    return x;
}

ReplayBatch ReplayBatchSampler::sample_batch(int batch_size) {
    if (batch_size <= 0) throw std::runtime_error("Replay batch size must be positive");

    std::vector<PendingReplaySample> pending;
    pending.reserve(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        const uint64_t random_value = next_u64();
        pending.push_back({
            static_cast<size_t>(random_value % dataset_.game_count()),
            static_cast<size_t>(i),
            random_value
        });
    }
    std::sort(
        pending.begin(), pending.end(),
        [](const PendingReplaySample& a, const PendingReplaySample& b) {
            if (a.global_game_index != b.global_game_index) {
                return a.global_game_index < b.global_game_index;
            }
            return a.output_index < b.output_index;
        });

    ReplayBatch batch;
    batch.batch_size = batch_size;
    batch.values.resize(batch_size);
    batch.rewards.resize(batch_size);
    batch.actions.resize(batch_size);
    batch.dones.resize(batch_size);

    bool tensor_sizes_initialized = false;
    int obs_size = 0;
    int policy_size = 0;
    int mask_size = 0;

    size_t begin = 0;
    while (begin < pending.size()) {
        size_t end = begin + 1;
        while (end < pending.size() &&
               pending[end].global_game_index == pending[begin].global_game_index) {
            ++end;
        }

        GameHistory history = dataset_.read_game(pending[begin].global_game_index);
        if (history.steps.empty()) throw std::runtime_error("ReplayDataset sampled empty GameHistory");

        for (size_t i = begin; i < end; ++i) {
            const size_t step_index = static_cast<size_t>(
                (pending[i].random_value >> 32) % history.steps.size());
            const TrajectoryStep& step = history.steps[step_index];
            ensure_step_tensor_sizes(
                step, tensor_sizes_initialized, obs_size, policy_size, mask_size);
            if (batch.observations.empty() && tensor_sizes_initialized) {
                batch.observation_size = obs_size;
                batch.policy_size = policy_size;
                batch.legal_mask_size = mask_size;
                batch.observations.resize(static_cast<size_t>(batch_size) * obs_size);
                batch.policy_targets.resize(static_cast<size_t>(batch_size) * policy_size);
                batch.legal_masks.resize(static_cast<size_t>(batch_size) * mask_size);
            }

            const size_t output_index = pending[i].output_index;
            std::copy(
                step.observation.begin(), step.observation.end(),
                batch.observations.begin() + output_index * static_cast<size_t>(obs_size));
            std::copy(
                step.policy_target.begin(), step.policy_target.end(),
                batch.policy_targets.begin() + output_index * static_cast<size_t>(policy_size));
            std::copy(
                step.legal_mask.begin(), step.legal_mask.end(),
                batch.legal_masks.begin() + output_index * static_cast<size_t>(mask_size));
            batch.values[output_index] = step.root_value;
            batch.rewards[output_index] = step.reward;
            batch.actions[output_index] = step.action;
            batch.dones[output_index] = step.done ? 1u : 0u;
        }

        begin = end;
    }

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
    double sum = batch.batch_size * 13.0 + batch.observation_size * 0.01 + batch.policy_size * 0.02;
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
