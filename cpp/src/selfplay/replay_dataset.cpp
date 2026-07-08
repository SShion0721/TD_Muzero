#include "tdmz/selfplay/replay_dataset.hpp"
#include <algorithm>
#include <fstream>
#include <iterator>
#include <regex>
#include <stdexcept>
#include <utility>

namespace tdmz {

namespace {

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

std::vector<std::string> parse_shard_paths_from_index_json(const std::string& index_path) {
    std::string text = read_text_file(index_path);
    std::vector<std::string> paths;
    std::regex path_re("\\\"path\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
    auto begin = std::sregex_iterator(text.begin(), text.end(), path_re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        paths.push_back(unescape_json_string((*it)[1].str()));
    }
    if (paths.empty()) throw std::runtime_error("No shard paths found in replay index JSON: " + index_path);
    return paths;
}

void ensure_step_tensor_sizes(const TrajectoryStep& step, int& obs_size, int& policy_size, int& mask_size) {
    if (obs_size == 0) obs_size = static_cast<int>(step.observation.size());
    if (policy_size == 0) policy_size = static_cast<int>(step.policy_target.size());
    if (mask_size == 0) mask_size = static_cast<int>(step.legal_mask.size());

    if (static_cast<int>(step.observation.size()) != obs_size ||
        static_cast<int>(step.policy_target.size()) != policy_size ||
        static_cast<int>(step.legal_mask.size()) != mask_size) {
        throw std::runtime_error("Replay batch contains inconsistent tensor sizes");
    }
}

} // namespace

ReplayDataset::ReplayDataset(std::vector<std::string> shard_paths)
    : shard_paths_(std::move(shard_paths)) {
    if (shard_paths_.empty()) throw std::runtime_error("ReplayDataset requires at least one shard path");

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

ReplayDataset ReplayDataset::from_index_json(const std::string& index_path) {
    return ReplayDataset(parse_shard_paths_from_index_json(index_path));
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
    return readers_[shard_id]->read_at(local_index);
}

ReplaySampleRef ReplayDataset::sample_ref(uint64_t random_u64) {
    size_t global_game_index = static_cast<size_t>(random_u64 % game_count());
    auto it = std::upper_bound(cumulative_games_.begin(), cumulative_games_.end(), global_game_index);
    size_t shard_id = static_cast<size_t>(it - cumulative_games_.begin());
    size_t shard_base = shard_id == 0 ? 0 : cumulative_games_[shard_id - 1];
    size_t local_index = global_game_index - shard_base;
    GameHistory history = readers_[shard_id]->read_at(local_index);
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

    ReplayBatch batch;
    batch.batch_size = batch_size;
    batch.values.reserve(batch_size);
    batch.rewards.reserve(batch_size);
    batch.actions.reserve(batch_size);
    batch.dones.reserve(batch_size);

    int obs_size = 0;
    int policy_size = 0;
    int mask_size = 0;

    for (int i = 0; i < batch_size; ++i) {
        TrajectoryStep step = dataset_.sample_step(next_u64());
        ensure_step_tensor_sizes(step, obs_size, policy_size, mask_size);

        if (i == 0) {
            batch.observation_size = obs_size;
            batch.policy_size = policy_size;
            batch.legal_mask_size = mask_size;
            batch.observations.reserve(static_cast<size_t>(batch_size) * obs_size);
            batch.policy_targets.reserve(static_cast<size_t>(batch_size) * policy_size);
            batch.legal_masks.reserve(static_cast<size_t>(batch_size) * mask_size);
        }

        batch.observations.insert(batch.observations.end(), step.observation.begin(), step.observation.end());
        batch.policy_targets.insert(batch.policy_targets.end(), step.policy_target.begin(), step.policy_target.end());
        batch.legal_masks.insert(batch.legal_masks.end(), step.legal_mask.begin(), step.legal_mask.end());
        batch.values.push_back(step.root_value);
        batch.rewards.push_back(step.reward);
        batch.actions.push_back(step.action);
        batch.dones.push_back(step.done ? 1u : 0u);
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
