#include "tdmz/selfplay/transition_shard.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace tdmz {
namespace {

constexpr char kTransitionMagic[8] = {'T', 'D', 'M', 'Z', 'S', 'H', 'D', '2'};

#pragma pack(push, 1)
struct TransitionShardHeader {
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

struct TransitionGameEntry {
    uint64_t seed;
    int32_t max_steps;
    uint32_t terminal;
    float total_reward;
    uint32_t step_count;
    uint64_t first_step_index;
};

struct TransitionStepEntry {
    uint64_t payload_offset;
    uint64_t payload_size;
};

struct TransitionStepHeader {
    int32_t step_index;
    int32_t action;
    float reward;
    float root_value;
    uint32_t done;
    int32_t money;
    int32_t base_hp;
    int32_t wave;
    float time;
};
#pragma pack(pop)

static_assert(sizeof(TransitionShardHeader) == 64, "Unexpected transition shard header size");
static_assert(sizeof(TransitionGameEntry) == 32, "Unexpected transition game entry size");
static_assert(sizeof(TransitionStepEntry) == 16, "Unexpected transition step entry size");
static_assert(sizeof(TransitionStepHeader) == 36, "Unexpected transition step header size");

namespace fs = std::filesystem;
std::atomic<uint64_t> g_transition_temp_counter{0};

template <typename T>
void write_raw(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("Failed while writing transition shard scalar");
}

template <typename T>
void read_raw(std::istream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("Failed while reading transition shard scalar");
}

uint64_t checked_add_u64(uint64_t a, uint64_t b, const char* label) {
    if (a > std::numeric_limits<uint64_t>::max() - b) {
        throw std::runtime_error(std::string(label) + " addition overflow");
    }
    return a + b;
}

uint64_t checked_mul_u64(uint64_t a, uint64_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::runtime_error(std::string(label) + " multiplication overflow");
    }
    return a * b;
}

uint32_t checked_u32(size_t value, const char* label) {
    if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string(label) + " exceeds uint32 range");
    }
    return static_cast<uint32_t>(value);
}

uint64_t stream_output_position(std::ostream& out) {
    const auto pos = out.tellp();
    if (pos < 0) throw std::runtime_error("Failed to query transition shard output position");
    return static_cast<uint64_t>(pos);
}

uint64_t stream_input_position(std::istream& in) {
    const auto pos = in.tellg();
    if (pos < 0) throw std::runtime_error("Failed to query transition shard input position");
    return static_cast<uint64_t>(pos);
}

uint64_t query_file_size(std::ifstream& in) {
    in.clear();
    in.seekg(0, std::ios::end);
    if (!in) throw std::runtime_error("Failed to seek transition shard end");
    const uint64_t size = stream_input_position(in);
    in.seekg(0, std::ios::beg);
    if (!in) throw std::runtime_error("Failed to seek transition shard start");
    return size;
}

std::string make_temp_path(const std::string& final_path) {
    const uint64_t nonce = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const uint64_t counter = g_transition_temp_counter.fetch_add(1, std::memory_order_relaxed);
    const fs::path final(final_path);
    const fs::path temp_name = final.filename().string()
        + ".tmp.transition." + std::to_string(nonce) + "." + std::to_string(counter);
    return (final.parent_path() / temp_name).string();
}

void remove_file_noexcept(const std::string& path) noexcept {
    std::error_code ec;
    fs::remove(fs::path(path), ec);
}

void publish_temp_file(const std::string& temp_path, const std::string& final_path) {
#ifdef _WIN32
    const fs::path temp(temp_path);
    const fs::path final(final_path);
    if (!MoveFileExW(temp.c_str(), final.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        throw std::system_error(
            static_cast<int>(GetLastError()),
            std::system_category(),
            "Failed to publish transition shard");
    }
#else
    std::error_code ec;
    fs::rename(fs::path(temp_path), fs::path(final_path), ec);
    if (ec) throw std::system_error(ec, "Failed to publish transition shard");
#endif
}

uint64_t step_payload_size(uint32_t observation_size, uint32_t policy_size, uint32_t legal_size) {
    uint64_t result = sizeof(TransitionStepHeader);
    result = checked_add_u64(
        result,
        checked_mul_u64(observation_size, sizeof(float), "observation payload"),
        "step payload");
    result = checked_add_u64(
        result,
        checked_mul_u64(policy_size, sizeof(float), "policy payload"),
        "step payload");
    result = checked_add_u64(result, legal_size, "step payload");
    return result;
}

void validate_step_scalars(const TransitionStepHeader& header) {
    if (header.done > 1u) throw std::runtime_error("Invalid transition done flag");
    if (!std::isfinite(header.reward)
        || !std::isfinite(header.root_value)
        || !std::isfinite(header.time)) {
        throw std::runtime_error("Non-finite transition step scalar");
    }
}

void validate_history_metadata(const GameHistory& history) {
    if (history.max_steps < 0) throw std::runtime_error("Transition history max_steps is negative");
    if (!std::isfinite(history.total_reward)) {
        throw std::runtime_error("Transition history total_reward is non-finite");
    }
}

void validate_destination(
    const ReplayStepDestination& destination,
    uint32_t observation_size,
    uint32_t policy_size,
    uint32_t legal_size
) {
    if (destination.observation_size != observation_size
        || destination.policy_size != policy_size
        || destination.legal_mask_size != legal_size) {
        throw std::runtime_error("Transition destination tensor size mismatch");
    }
    if ((observation_size > 0 && destination.observation == nullptr)
        || (policy_size > 0 && destination.policy_target == nullptr)
        || (legal_size > 0 && destination.legal_mask == nullptr)) {
        throw std::runtime_error("Transition destination tensor pointer is null");
    }
    if (!destination.value || !destination.reward || !destination.action || !destination.done) {
        throw std::runtime_error("Transition destination scalar pointer is null");
    }
}

template <typename T>
void write_vector(std::ostream& out, const std::vector<T>& values) {
    if (values.size() > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::runtime_error("Transition vector byte-size overflow");
    }
    const size_t bytes = values.size() * sizeof(T);
    if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Transition vector exceeds streamsize range");
    }
    if (bytes != 0) {
        out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(bytes));
        if (!out) throw std::runtime_error("Failed while writing transition vector");
    }
}

template <typename T>
void read_vector_into(std::istream& in, T* destination, size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::runtime_error("Transition vector byte-size overflow");
    }
    const size_t bytes = count * sizeof(T);
    if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Transition vector exceeds streamsize range");
    }
    if (bytes != 0) {
        in.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(bytes));
        if (!in) throw std::runtime_error("Failed while reading transition vector");
    }
}

} // namespace

struct TransitionShardWriter::Impl {
    Impl(const std::string& output_path, size_t expected_count)
        : path(output_path),
          temp_path(make_temp_path(output_path)),
          expected_history_count(expected_count),
          out(temp_path, std::ios::binary | std::ios::trunc) {
        if (!out) throw std::runtime_error("Failed to open temporary transition shard path");

        TransitionShardHeader placeholder{};
        std::memcpy(placeholder.magic, kTransitionMagic, sizeof(kTransitionMagic));
        placeholder.version = kTransitionShardFormatVersion;
        placeholder.history_count = checked_u32(expected_history_count, "history_count");
        write_raw(out, placeholder);
    }

    ~Impl() {
        if (!is_closed) {
            try {
                close();
            } catch (...) {
                out.close();
                remove_file_noexcept(temp_path);
                is_closed = true;
            }
        }
    }

    void establish_or_validate_shapes(const TrajectoryStep& step) {
        const uint32_t obs = checked_u32(step.observation.size(), "observation_size");
        const uint32_t policy = checked_u32(step.policy_target.size(), "policy_size");
        const uint32_t legal = checked_u32(step.legal_mask.size(), "legal_mask_size");
        if (!shapes_initialized) {
            observation_size = obs;
            policy_size = policy;
            legal_mask_size = legal;
            shapes_initialized = true;
        } else if (obs != observation_size || policy != policy_size || legal != legal_mask_size) {
            throw std::runtime_error("All transition shard steps must have identical tensor sizes");
        }
    }

    void write_history(const GameHistory& history) {
        if (is_closed) throw std::runtime_error("Cannot write to a closed transition shard");
        if (games.size() >= expected_history_count) {
            throw std::runtime_error("Transition shard writer received more histories than expected");
        }
        validate_history_metadata(history);

        TransitionGameEntry game{};
        game.seed = history.seed;
        game.max_steps = history.max_steps;
        game.terminal = history.terminal ? 1u : 0u;
        game.total_reward = history.total_reward;
        game.step_count = checked_u32(history.steps.size(), "step_count");
        game.first_step_index = static_cast<uint64_t>(steps.size());

        for (const auto& step : history.steps) {
            establish_or_validate_shapes(step);
            if (!std::isfinite(step.reward)
                || !std::isfinite(step.root_value)
                || !std::isfinite(step.time)) {
                throw std::runtime_error("Non-finite transition step scalar");
            }

            TransitionStepEntry entry{};
            entry.payload_offset = stream_output_position(out);

            TransitionStepHeader header{};
            header.step_index = step.step_index;
            header.action = step.action;
            header.reward = step.reward;
            header.root_value = step.root_value;
            header.done = step.done ? 1u : 0u;
            header.money = step.money;
            header.base_hp = step.base_hp;
            header.wave = step.wave;
            header.time = step.time;

            write_raw(out, header);
            write_vector(out, step.observation);
            write_vector(out, step.policy_target);
            write_vector(out, step.legal_mask);

            entry.payload_size = stream_output_position(out) - entry.payload_offset;
            const uint64_t expected_payload = step_payload_size(
                observation_size, policy_size, legal_mask_size);
            if (entry.payload_size != expected_payload) {
                throw std::runtime_error("Transition payload size mismatch while writing");
            }
            steps.push_back(entry);
        }

        games.push_back(game);
    }

    void close() {
        if (is_closed) return;
        if (games.size() != expected_history_count) {
            out.close();
            remove_file_noexcept(temp_path);
            is_closed = true;
            throw std::runtime_error("Transition shard writer closed before all expected histories were written");
        }

        try {
            const uint64_t game_table_offset = stream_output_position(out);
            for (const auto& game : games) write_raw(out, game);
            const uint64_t step_table_offset = stream_output_position(out);
            for (const auto& step : steps) write_raw(out, step);
            const uint64_t final_size = stream_output_position(out);

            TransitionShardHeader header{};
            std::memcpy(header.magic, kTransitionMagic, sizeof(kTransitionMagic));
            header.version = kTransitionShardFormatVersion;
            header.history_count = checked_u32(games.size(), "history_count");
            header.observation_size = observation_size;
            header.policy_size = policy_size;
            header.legal_mask_size = legal_mask_size;
            header.total_step_count = static_cast<uint64_t>(steps.size());
            header.game_table_offset = game_table_offset;
            header.step_table_offset = step_table_offset;
            header.file_size = final_size;

            out.seekp(0, std::ios::beg);
            if (!out) throw std::runtime_error("Failed to seek transition shard header");
            write_raw(out, header);
            out.flush();
            if (!out) throw std::runtime_error("Failed to flush transition shard");
            out.close();
            if (!out) throw std::runtime_error("Failed to close transition shard");
            publish_temp_file(temp_path, path);
            is_closed = true;
        } catch (...) {
            out.close();
            remove_file_noexcept(temp_path);
            is_closed = true;
            throw;
        }
    }

    std::string path;
    std::string temp_path;
    size_t expected_history_count = 0;
    bool is_closed = false;
    bool shapes_initialized = false;
    uint32_t observation_size = 0;
    uint32_t policy_size = 0;
    uint32_t legal_mask_size = 0;
    std::ofstream out;
    std::vector<TransitionGameEntry> games;
    std::vector<TransitionStepEntry> steps;
};

TransitionShardWriter::TransitionShardWriter(const std::string& path, size_t expected_history_count)
    : impl_(new Impl(path, expected_history_count)) {}

TransitionShardWriter::~TransitionShardWriter() = default;

void TransitionShardWriter::write_history(const GameHistory& history) {
    impl_->write_history(history);
}

void TransitionShardWriter::close() {
    impl_->close();
}

size_t TransitionShardWriter::expected_history_count() const {
    return impl_->expected_history_count;
}

size_t TransitionShardWriter::written_count() const {
    return impl_->games.size();
}

bool TransitionShardWriter::closed() const {
    return impl_->is_closed;
}

struct TransitionShardReader::Impl {
    explicit Impl(const std::string& input_path)
        : path(input_path), in(input_path, std::ios::binary) {
        if (!in) throw std::runtime_error("Failed to open transition shard path");
        const uint64_t actual_size = query_file_size(in);
        if (actual_size < sizeof(TransitionShardHeader)) {
            throw std::runtime_error("Truncated transition shard header");
        }

        read_raw(in, header);
        if (std::memcmp(header.magic, kTransitionMagic, sizeof(kTransitionMagic)) != 0) {
            throw std::runtime_error("Invalid transition shard magic");
        }
        if (header.version != kTransitionShardFormatVersion) {
            throw std::runtime_error("Unsupported transition shard version");
        }
        if (header.reserved != 0) throw std::runtime_error("Invalid transition shard reserved field");
        if (header.file_size != actual_size) {
            throw std::runtime_error("Transition shard file size does not match header");
        }

        const uint64_t game_table_bytes = checked_mul_u64(
            header.history_count, sizeof(TransitionGameEntry), "game table");
        const uint64_t step_table_bytes = checked_mul_u64(
            header.total_step_count, sizeof(TransitionStepEntry), "step table");
        if (header.game_table_offset < sizeof(TransitionShardHeader)) {
            throw std::runtime_error("Transition game table overlaps header");
        }
        if (header.step_table_offset != checked_add_u64(
                header.game_table_offset, game_table_bytes, "step table offset")) {
            throw std::runtime_error("Transition step table does not follow game table");
        }
        if (header.file_size != checked_add_u64(
                header.step_table_offset, step_table_bytes, "transition file size")) {
            throw std::runtime_error("Transition index tables do not end at file boundary");
        }

        seek(header.game_table_offset, "game table");
        games.resize(static_cast<size_t>(header.history_count));
        for (auto& game : games) read_raw(in, game);

        seek(header.step_table_offset, "step table");
        steps.resize(static_cast<size_t>(header.total_step_count));
        for (auto& step : steps) read_raw(in, step);

        uint64_t expected_first_step = 0;
        for (const auto& game : games) {
            if (game.terminal > 1u) throw std::runtime_error("Invalid transition game terminal flag");
            if (game.max_steps < 0) throw std::runtime_error("Invalid transition game max_steps");
            if (!std::isfinite(game.total_reward)) {
                throw std::runtime_error("Invalid transition game total_reward");
            }
            if (game.first_step_index != expected_first_step) {
                throw std::runtime_error("Transition game step ranges are not contiguous");
            }
            expected_first_step = checked_add_u64(
                expected_first_step, game.step_count, "game step range");
            if (expected_first_step > header.total_step_count) {
                throw std::runtime_error("Transition game step range exceeds step table");
            }
        }
        if (expected_first_step != header.total_step_count) {
            throw std::runtime_error("Transition game step ranges do not cover step table");
        }

        const uint64_t expected_payload_size = step_payload_size(
            header.observation_size, header.policy_size, header.legal_mask_size);
        uint64_t previous_end = sizeof(TransitionShardHeader);
        for (const auto& step : steps) {
            if (step.payload_size != expected_payload_size) {
                throw std::runtime_error("Transition step payload size is incompatible with shard tensors");
            }
            if (step.payload_offset != previous_end) {
                throw std::runtime_error("Transition step payloads are not contiguous");
            }
            previous_end = checked_add_u64(
                step.payload_offset, step.payload_size, "transition payload end");
            if (previous_end > header.game_table_offset) {
                throw std::runtime_error("Transition step payload overlaps index tables");
            }
        }
        if (previous_end != header.game_table_offset) {
            throw std::runtime_error("Transition payload region has gaps or trailing bytes");
        }
        if (steps.empty() && header.game_table_offset != sizeof(TransitionShardHeader)) {
            throw std::runtime_error("Empty transition shard has unexpected payload bytes");
        }
    }

    void seek(uint64_t offset, const char* label) {
        if (offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
            throw std::runtime_error(std::string("Transition ") + label + " offset exceeds streamoff range");
        }
        in.clear();
        in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!in) throw std::runtime_error(std::string("Failed to seek transition ") + label);
    }

    const TransitionGameEntry& game_at(size_t game_index) const {
        if (game_index >= games.size()) throw std::runtime_error("Transition game index out of range");
        return games[game_index];
    }

    const TransitionStepEntry& step_at(size_t game_index, size_t step_index) const {
        const auto& game = game_at(game_index);
        if (step_index >= game.step_count) throw std::runtime_error("Transition step index out of range");
        const uint64_t global_step = game.first_step_index + step_index;
        if (global_step >= steps.size()) throw std::runtime_error("Transition global step index out of range");
        return steps[static_cast<size_t>(global_step)];
    }

    void read_step_into(
        size_t game_index,
        size_t step_index,
        const ReplayStepDestination& destination
    ) {
        validate_destination(
            destination,
            header.observation_size,
            header.policy_size,
            header.legal_mask_size);
        const auto& entry = step_at(game_index, step_index);
        seek(entry.payload_offset, "step payload");

        TransitionStepHeader step_header{};
        read_raw(in, step_header);
        validate_step_scalars(step_header);
        read_vector_into(in, destination.observation, header.observation_size);
        read_vector_into(in, destination.policy_target, header.policy_size);
        read_vector_into(in, destination.legal_mask, header.legal_mask_size);

        const uint64_t consumed = stream_input_position(in) - entry.payload_offset;
        if (consumed != entry.payload_size) {
            throw std::runtime_error("Transition step read ended at an unexpected position");
        }

        *destination.value = step_header.root_value;
        *destination.reward = step_header.reward;
        *destination.action = step_header.action;
        *destination.done = step_header.done ? 1u : 0u;
        if (destination.step_index) *destination.step_index = step_header.step_index;
        if (destination.money) *destination.money = step_header.money;
        if (destination.base_hp) *destination.base_hp = step_header.base_hp;
        if (destination.wave) *destination.wave = step_header.wave;
        if (destination.time) *destination.time = step_header.time;

        ++stats.physical_read_operations;
        stats.physical_bytes_read = checked_add_u64(
            stats.physical_bytes_read, entry.payload_size, "physical bytes read");
    }

    std::string path;
    std::ifstream in;
    TransitionShardHeader header{};
    std::vector<TransitionGameEntry> games;
    std::vector<TransitionStepEntry> steps;
    ReplayIoStats stats;
};

TransitionShardReader::TransitionShardReader(const std::string& path)
    : impl_(new Impl(path)) {}

TransitionShardReader::~TransitionShardReader() = default;
TransitionShardReader::TransitionShardReader(TransitionShardReader&&) noexcept = default;
TransitionShardReader& TransitionShardReader::operator=(TransitionShardReader&&) noexcept = default;

size_t TransitionShardReader::history_count() const {
    return impl_->games.size();
}

size_t TransitionShardReader::step_count(size_t game_index) const {
    return impl_->game_at(game_index).step_count;
}

uint64_t TransitionShardReader::step_physical_offset(size_t game_index, size_t step_index) const {
    return impl_->step_at(game_index, step_index).payload_offset;
}

int TransitionShardReader::observation_size() const {
    if (impl_->header.observation_size > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Transition observation size exceeds int range");
    }
    return static_cast<int>(impl_->header.observation_size);
}

int TransitionShardReader::policy_size() const {
    if (impl_->header.policy_size > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Transition policy size exceeds int range");
    }
    return static_cast<int>(impl_->header.policy_size);
}

int TransitionShardReader::legal_mask_size() const {
    if (impl_->header.legal_mask_size > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Transition legal-mask size exceeds int range");
    }
    return static_cast<int>(impl_->header.legal_mask_size);
}

const std::string& TransitionShardReader::path() const {
    return impl_->path;
}

void TransitionShardReader::read_step_into(
    size_t game_index,
    size_t step_index,
    const ReplayStepDestination& destination
) {
    impl_->read_step_into(game_index, step_index, destination);
}

TrajectoryStep TransitionShardReader::read_step(size_t game_index, size_t step_index) {
    TrajectoryStep result;
    result.observation.resize(static_cast<size_t>(observation_size()));
    result.policy_target.resize(static_cast<size_t>(policy_size()));
    result.legal_mask.resize(static_cast<size_t>(legal_mask_size()));
    uint8_t done = 0;
    ReplayStepDestination destination;
    destination.observation = result.observation.data();
    destination.observation_size = result.observation.size();
    destination.policy_target = result.policy_target.data();
    destination.policy_size = result.policy_target.size();
    destination.legal_mask = result.legal_mask.data();
    destination.legal_mask_size = result.legal_mask.size();
    destination.value = &result.root_value;
    destination.reward = &result.reward;
    destination.action = &result.action;
    destination.done = &done;
    destination.step_index = &result.step_index;
    destination.money = &result.money;
    destination.base_hp = &result.base_hp;
    destination.wave = &result.wave;
    destination.time = &result.time;
    read_step_into(game_index, step_index, destination);
    result.done = done != 0;
    return result;
}

GameHistory TransitionShardReader::read_at(size_t game_index) {
    const auto& game = impl_->game_at(game_index);
    GameHistory history;
    history.seed = game.seed;
    history.max_steps = game.max_steps;
    history.terminal = game.terminal != 0;
    history.total_reward = game.total_reward;
    history.steps.reserve(game.step_count);
    for (size_t step = 0; step < game.step_count; ++step) {
        history.steps.push_back(read_step(game_index, step));
    }
    return history;
}

ReplayIoStats TransitionShardReader::io_stats() const {
    return impl_->stats;
}

void TransitionShardReader::reset_io_stats() {
    impl_->stats = ReplayIoStats{};
}

bool is_transition_shard_v2(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open replay shard path: " + path);
    char magic[8]{};
    in.read(magic, sizeof(magic));
    if (!in) throw std::runtime_error("Replay shard is too small to contain a magic value: " + path);
    return std::memcmp(magic, kTransitionMagic, sizeof(kTransitionMagic)) == 0;
}

struct AsyncTransitionShardWriter::Impl {
    Impl(const std::string& output_path, size_t expected_count, size_t queue_size)
        : path(output_path),
          expected_history_count(expected_count),
          max_queue_size(std::max<size_t>(1, queue_size)),
          writer(output_path, expected_count),
          worker(&Impl::writer_loop, this) {}

    ~Impl() {
        try {
            close();
        } catch (...) {
            if (worker.joinable()) worker.join();
        }
    }

    [[noreturn]] void rethrow_failure_locked() const {
        if (error) std::rethrow_exception(error);
        throw std::runtime_error("Async transition shard writer failed");
    }

    void write(GameHistory history) {
        std::unique_lock<std::mutex> lock(mutex);
        if (lifecycle == AsyncTransitionShardWriterState::Failed) rethrow_failure_locked();
        if (lifecycle != AsyncTransitionShardWriterState::Open) {
            throw std::runtime_error("Cannot write after async transition shard closing begins");
        }
        if (enqueued_count >= expected_history_count) {
            throw std::runtime_error("Async transition shard writer received more histories than expected");
        }
        not_full.wait(lock, [&] {
            return queue.size() < max_queue_size
                || lifecycle != AsyncTransitionShardWriterState::Open;
        });
        if (lifecycle == AsyncTransitionShardWriterState::Failed) rethrow_failure_locked();
        if (lifecycle != AsyncTransitionShardWriterState::Open) {
            throw std::runtime_error("Cannot write after async transition shard closing begins");
        }
        queue.push_back(std::move(history));
        ++enqueued_count;
        not_empty.notify_one();
    }

    void close() {
        bool join_worker = false;
        {
            std::unique_lock<std::mutex> lock(mutex);
            if (lifecycle == AsyncTransitionShardWriterState::Open) {
                lifecycle = AsyncTransitionShardWriterState::Closing;
                not_empty.notify_all();
                not_full.notify_all();
                state_changed.notify_all();
            }
            if (!join_claimed) {
                join_claimed = true;
                join_worker = true;
            } else {
                state_changed.wait(lock, [&] {
                    return lifecycle == AsyncTransitionShardWriterState::Closed
                        || lifecycle == AsyncTransitionShardWriterState::Failed;
                });
                if (lifecycle == AsyncTransitionShardWriterState::Failed) rethrow_failure_locked();
                return;
            }
        }

        if (join_worker && worker.joinable()) worker.join();

        std::unique_lock<std::mutex> lock(mutex);
        state_changed.wait(lock, [&] {
            return lifecycle == AsyncTransitionShardWriterState::Closed
                || lifecycle == AsyncTransitionShardWriterState::Failed;
        });
        if (lifecycle == AsyncTransitionShardWriterState::Failed) rethrow_failure_locked();
    }

    void writer_loop() {
        try {
            while (true) {
                GameHistory history;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    not_empty.wait(lock, [&] {
                        return !queue.empty()
                            || lifecycle != AsyncTransitionShardWriterState::Open;
                    });
                    if (queue.empty() && lifecycle != AsyncTransitionShardWriterState::Open) break;
                    history = std::move(queue.front());
                    queue.pop_front();
                    not_full.notify_one();
                }
                writer.write_history(history);
            }
            writer.close();
            std::lock_guard<std::mutex> lock(mutex);
            lifecycle = AsyncTransitionShardWriterState::Closed;
            state_changed.notify_all();
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            error = std::current_exception();
            lifecycle = AsyncTransitionShardWriterState::Failed;
            queue.clear();
            not_empty.notify_all();
            not_full.notify_all();
            state_changed.notify_all();
        }
    }

    std::string path;
    size_t expected_history_count = 0;
    size_t max_queue_size = 1;
    size_t enqueued_count = 0;
    AsyncTransitionShardWriterState lifecycle = AsyncTransitionShardWriterState::Open;
    bool join_claimed = false;
    std::exception_ptr error;
    TransitionShardWriter writer;
    mutable std::mutex mutex;
    std::condition_variable not_empty;
    std::condition_variable not_full;
    std::condition_variable state_changed;
    std::deque<GameHistory> queue;
    std::thread worker;
};

AsyncTransitionShardWriter::AsyncTransitionShardWriter(
    const std::string& path,
    size_t expected_history_count,
    size_t max_queue_size
) : impl_(new Impl(path, expected_history_count, max_queue_size)) {}

AsyncTransitionShardWriter::~AsyncTransitionShardWriter() = default;

void AsyncTransitionShardWriter::write_history(GameHistory history) {
    impl_->write(std::move(history));
}

void AsyncTransitionShardWriter::close() {
    impl_->close();
}

size_t AsyncTransitionShardWriter::expected_history_count() const {
    return impl_->expected_history_count;
}

size_t AsyncTransitionShardWriter::enqueued_count() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->enqueued_count;
}

size_t AsyncTransitionShardWriter::max_queue_size() const {
    return impl_->max_queue_size;
}

AsyncTransitionShardWriterState AsyncTransitionShardWriter::state() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->lifecycle;
}

bool AsyncTransitionShardWriter::closed() const {
    return state() == AsyncTransitionShardWriterState::Closed;
}

void write_histories_transition_shard(
    const std::vector<GameHistory>& histories,
    const std::string& path
) {
    TransitionShardWriter writer(path, histories.size());
    for (const auto& history : histories) writer.write_history(history);
    writer.close();
}

} // namespace tdmz
