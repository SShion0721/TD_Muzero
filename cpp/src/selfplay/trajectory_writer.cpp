#include "tdmz/selfplay/trajectory_writer.hpp"
#include "tdmz/core/action.hpp"
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
#include <iomanip>
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

constexpr char kBinaryMagic[8] = {'T', 'D', 'M', 'Z', 'S', 'P', 'B', '1'};
constexpr char kShardMagic[8] = {'T', 'D', 'M', 'Z', 'S', 'H', 'D', '1'};
constexpr uint32_t kBinaryVersion = 1;
constexpr uint32_t kShardVersion = 1;

#pragma pack(push, 1)
struct BinaryHistoryHeader {
    char magic[8];
    uint32_t version;
    uint32_t step_count;
    uint32_t observation_size;
    uint32_t policy_size;
    uint32_t legal_mask_size;
    uint64_t seed;
    int32_t max_steps;
    uint32_t terminal;
    float total_reward;
};

struct BinaryStepHeader {
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

struct BinaryShardHeader {
    char magic[8];
    uint32_t version;
    uint32_t history_count;
    uint64_t offset_table_bytes;
};
#pragma pack(pop)

static_assert(sizeof(BinaryHistoryHeader) == 48, "Unexpected binary history header size");
static_assert(sizeof(BinaryStepHeader) == 36, "Unexpected binary step header size");
static_assert(sizeof(BinaryShardHeader) == 24, "Unexpected binary shard header size");

namespace fs = std::filesystem;
std::atomic<uint64_t> g_temp_file_counter{0};

template <typename T>
void write_vector(std::ostream& out, const std::vector<T>& values) {
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ",";
        out << +values[i];
    }
    out << "]";
}

void write_step_json(std::ostream& out, const GameHistory& history, const TrajectoryStep& step) {
    out << std::setprecision(8);
    out << "{";
    out << "\"seed\":" << history.seed << ",";
    out << "\"step_index\":" << step.step_index << ",";
    out << "\"action\":" << step.action << ",";
    out << "\"reward\":" << step.reward << ",";
    out << "\"root_value\":" << step.root_value << ",";
    out << "\"done\":" << (step.done ? "true" : "false") << ",";
    out << "\"money\":" << step.money << ",";
    out << "\"base_hp\":" << step.base_hp << ",";
    out << "\"wave\":" << step.wave << ",";
    out << "\"time\":" << step.time << ",";
    out << "\"policy_target\":";
    write_vector(out, step.policy_target);
    out << ",\"legal_mask\":";
    write_vector(out, step.legal_mask);
    out << ",\"observation\":";
    write_vector(out, step.observation);
    out << "}";
}

template <typename T>
void write_raw(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("Failed while writing binary trajectory scalar");
}

template <typename T>
void read_raw(std::istream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("Failed while reading binary trajectory scalar");
}

uint64_t checked_add_u64(uint64_t a, uint64_t b, const char* name) {
    if (a > std::numeric_limits<uint64_t>::max() - b) {
        throw std::runtime_error(std::string(name) + " addition overflow");
    }
    return a + b;
}

uint64_t checked_mul_u64(uint64_t a, uint64_t b, const char* name) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::runtime_error(std::string(name) + " multiplication overflow");
    }
    return a * b;
}

uint32_t checked_u32(size_t value, const char* name) {
    if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string(name) + " exceeds uint32_t range");
    }
    return static_cast<uint32_t>(value);
}

uint64_t checked_u64_stream_pos(std::ostream& out) {
    const auto pos = out.tellp();
    if (pos < 0) throw std::runtime_error("Failed to query binary output stream position");
    return static_cast<uint64_t>(pos);
}

uint64_t checked_input_pos(std::istream& in) {
    const auto pos = in.tellg();
    if (pos < 0) throw std::runtime_error("Failed to query binary input stream position");
    return static_cast<uint64_t>(pos);
}

uint64_t query_file_size(std::ifstream& in) {
    in.clear();
    in.seekg(0, std::ios::end);
    if (!in) throw std::runtime_error("Failed to seek binary input end");
    const uint64_t size = checked_input_pos(in);
    in.seekg(0, std::ios::beg);
    if (!in) throw std::runtime_error("Failed to seek binary input start");
    return size;
}

std::string make_temp_path(const std::string& final_path) {
    const auto nonce = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const uint64_t counter = g_temp_file_counter.fetch_add(1, std::memory_order_relaxed);
    const fs::path final(final_path);
    const fs::path temp_name = final.filename().string()
        + ".tmp." + std::to_string(nonce) + "." + std::to_string(counter);
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
            "Failed to publish trajectory shard");
    }
#else
    std::error_code ec;
    fs::rename(fs::path(temp_path), fs::path(final_path), ec);
    if (ec) throw std::system_error(ec, "Failed to publish trajectory shard");
#endif
}

template <typename T>
void write_raw_vector(std::ostream& out, const std::vector<T>& values, size_t expected_size) {
    if (values.size() != expected_size) {
        throw std::runtime_error("Binary trajectory vector size mismatch");
    }
    if (values.size() > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::runtime_error("Binary trajectory vector byte-size overflow");
    }
    const size_t byte_count = values.size() * sizeof(T);
    if (byte_count > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Binary trajectory vector exceeds streamsize range");
    }
    if (!values.empty()) {
        out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(byte_count));
        if (!out) throw std::runtime_error("Failed while writing binary trajectory vector");
    }
}

template <typename T>
void read_raw_vector(std::istream& in, std::vector<T>& values, size_t expected_size) {
    if (expected_size > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::runtime_error("Binary trajectory vector byte-size overflow");
    }
    const size_t byte_count = expected_size * sizeof(T);
    if (byte_count > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Binary trajectory vector exceeds streamsize range");
    }
    values.resize(expected_size);
    if (!values.empty()) {
        in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(byte_count));
        if (!in) throw std::runtime_error("Failed while reading binary trajectory vector");
    }
}

void validate_binary_step_sizes(
    const GameHistory& history,
    uint32_t obs_size,
    uint32_t policy_size,
    uint32_t legal_size
) {
    for (const auto& step : history.steps) {
        if (step.observation.size() != obs_size
            || step.policy_target.size() != policy_size
            || step.legal_mask.size() != legal_size) {
            throw std::runtime_error("All binary trajectory steps must have identical tensor sizes");
        }
    }
}

void write_history_binary_stream(std::ostream& out, const GameHistory& history) {
    const uint32_t step_count = checked_u32(history.steps.size(), "step_count");
    uint32_t obs_size = 0;
    uint32_t policy_size = 0;
    uint32_t legal_size = 0;
    if (!history.steps.empty()) {
        obs_size = checked_u32(history.steps.front().observation.size(), "observation_size");
        policy_size = checked_u32(history.steps.front().policy_target.size(), "policy_size");
        legal_size = checked_u32(history.steps.front().legal_mask.size(), "legal_mask_size");
        validate_binary_step_sizes(history, obs_size, policy_size, legal_size);
    }

    BinaryHistoryHeader header{};
    std::memcpy(header.magic, kBinaryMagic, sizeof(kBinaryMagic));
    header.version = kBinaryVersion;
    header.step_count = step_count;
    header.observation_size = obs_size;
    header.policy_size = policy_size;
    header.legal_mask_size = legal_size;
    header.seed = history.seed;
    header.max_steps = history.max_steps;
    header.terminal = history.terminal ? 1u : 0u;
    header.total_reward = history.total_reward;
    write_raw(out, header);

    for (const auto& step : history.steps) {
        BinaryStepHeader sh{};
        sh.step_index = step.step_index;
        sh.action = step.action;
        sh.reward = step.reward;
        sh.root_value = step.root_value;
        sh.done = step.done ? 1u : 0u;
        sh.money = step.money;
        sh.base_hp = step.base_hp;
        sh.wave = step.wave;
        sh.time = step.time;
        write_raw(out, sh);
        write_raw_vector(out, step.observation, obs_size);
        write_raw_vector(out, step.policy_target, policy_size);
        write_raw_vector(out, step.legal_mask, legal_size);
    }
}

uint64_t checked_history_size(const BinaryHistoryHeader& header) {
    if (header.terminal > 1u) throw std::runtime_error("Invalid trajectory terminal flag");
    if (header.max_steps < 0) throw std::runtime_error("Invalid trajectory max_steps");
    if (!std::isfinite(header.total_reward)) throw std::runtime_error("Invalid trajectory total_reward");
    if (header.step_count == 0
        && (header.observation_size != 0
            || header.policy_size != 0
            || header.legal_mask_size != 0)) {
        throw std::runtime_error("Empty trajectory must have zero tensor sizes");
    }

    uint64_t per_step = sizeof(BinaryStepHeader);
    per_step = checked_add_u64(
        per_step,
        checked_mul_u64(header.observation_size, sizeof(float), "observation bytes"),
        "step bytes");
    per_step = checked_add_u64(
        per_step,
        checked_mul_u64(header.policy_size, sizeof(float), "policy bytes"),
        "step bytes");
    per_step = checked_add_u64(per_step, header.legal_mask_size, "step bytes");
    return checked_add_u64(
        sizeof(BinaryHistoryHeader),
        checked_mul_u64(header.step_count, per_step, "history payload"),
        "history bytes");
}

GameHistory read_history_binary_stream(std::istream& in, uint64_t available_bytes) {
    if (available_bytes < sizeof(BinaryHistoryHeader)) {
        throw std::runtime_error("Truncated trajectory binary header");
    }
    const uint64_t start = checked_input_pos(in);
    BinaryHistoryHeader header{};
    read_raw(in, header);
    if (std::memcmp(header.magic, kBinaryMagic, sizeof(kBinaryMagic)) != 0) {
        throw std::runtime_error("Invalid trajectory binary magic");
    }
    if (header.version != kBinaryVersion) {
        throw std::runtime_error("Unsupported trajectory binary version");
    }
    const uint64_t expected_bytes = checked_history_size(header);
    if (expected_bytes != available_bytes) {
        throw std::runtime_error("Trajectory history byte size does not match its container boundary");
    }

    GameHistory history;
    history.seed = header.seed;
    history.max_steps = header.max_steps;
    history.terminal = header.terminal != 0;
    history.total_reward = header.total_reward;
    history.steps.reserve(static_cast<size_t>(header.step_count));

    for (uint32_t i = 0; i < header.step_count; ++i) {
        BinaryStepHeader sh{};
        read_raw(in, sh);
        if (sh.done > 1u) throw std::runtime_error("Invalid trajectory done flag");
        if (!std::isfinite(sh.reward)
            || !std::isfinite(sh.root_value)
            || !std::isfinite(sh.time)) {
            throw std::runtime_error("Non-finite trajectory step scalar");
        }

        TrajectoryStep step;
        step.step_index = sh.step_index;
        step.action = sh.action;
        step.reward = sh.reward;
        step.root_value = sh.root_value;
        step.done = sh.done != 0;
        step.money = sh.money;
        step.base_hp = sh.base_hp;
        step.wave = sh.wave;
        step.time = sh.time;
        read_raw_vector(in, step.observation, static_cast<size_t>(header.observation_size));
        read_raw_vector(in, step.policy_target, static_cast<size_t>(header.policy_size));
        read_raw_vector(in, step.legal_mask, static_cast<size_t>(header.legal_mask_size));
        history.steps.push_back(std::move(step));
    }

    if (checked_input_pos(in) - start != expected_bytes) {
        throw std::runtime_error("Trajectory history ended at an unexpected stream position");
    }
    return history;
}

std::vector<uint64_t> read_shard_offsets(
    std::istream& in,
    BinaryShardHeader& header,
    uint64_t file_size
) {
    if (file_size < sizeof(BinaryShardHeader)) {
        throw std::runtime_error("Truncated trajectory shard header");
    }
    read_raw(in, header);
    if (std::memcmp(header.magic, kShardMagic, sizeof(kShardMagic)) != 0) {
        throw std::runtime_error("Invalid trajectory shard magic");
    }
    if (header.version != kShardVersion) {
        throw std::runtime_error("Unsupported trajectory shard version");
    }

    const uint64_t expected_table_bytes = checked_mul_u64(
        header.history_count,
        sizeof(uint64_t),
        "shard offset table");
    if (header.offset_table_bytes != expected_table_bytes) {
        throw std::runtime_error("Invalid trajectory shard offset table size");
    }
    const uint64_t data_start = checked_add_u64(
        sizeof(BinaryShardHeader),
        expected_table_bytes,
        "shard data start");
    if (data_start > file_size) {
        throw std::runtime_error("Trajectory shard offset table exceeds file size");
    }

    std::vector<uint64_t> offsets(static_cast<size_t>(header.history_count), 0);
    for (auto& offset : offsets) read_raw(in, offset);
    if (offsets.empty()) {
        if (data_start != file_size) {
            throw std::runtime_error("Empty trajectory shard has trailing payload bytes");
        }
        return offsets;
    }
    if (offsets.front() != data_start) {
        throw std::runtime_error("First trajectory shard offset does not follow the offset table");
    }
    uint64_t previous = data_start - 1;
    for (uint64_t offset : offsets) {
        if (offset <= previous || offset >= file_size) {
            throw std::runtime_error("Invalid or non-increasing trajectory shard offset");
        }
        previous = offset;
    }
    return offsets;
}

} // namespace

void write_history_jsonl(const GameHistory& history, const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to open trajectory JSONL path");
    for (const auto& step : history.steps) {
        write_step_json(out, history, step);
        out << "\n";
    }
}

void write_history_summary_json(const GameHistory& history, const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to open trajectory summary path");
    out << std::setprecision(8);
    out << "{";
    out << "\"seed\":" << history.seed << ",";
    out << "\"max_steps\":" << history.max_steps << ",";
    out << "\"num_steps\":" << history.steps.size() << ",";
    out << "\"terminal\":" << (history.terminal ? "true" : "false") << ",";
    out << "\"total_reward\":" << history.total_reward;
    out << "}\n";
}

void write_history_binary(const GameHistory& history, const std::string& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Failed to open trajectory binary path");
    write_history_binary_stream(out, history);
}

GameHistory read_history_binary(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open trajectory binary path");
    const uint64_t file_size = query_file_size(in);
    return read_history_binary_stream(in, file_size);
}

BinaryShardReader::BinaryShardReader(const std::string& path)
    : path_(path), in_(path, std::ios::binary) {
    if (!in_) throw std::runtime_error("Failed to open trajectory binary shard path");
    file_size_ = query_file_size(in_);
    BinaryShardHeader header{};
    offsets_ = read_shard_offsets(in_, header, file_size_);
}

BinaryShardReader::~BinaryShardReader() = default;

size_t BinaryShardReader::history_count() const {
    return offsets_.size();
}

const std::string& BinaryShardReader::path() const {
    return path_;
}

GameHistory BinaryShardReader::read_at(size_t index) {
    if (index >= offsets_.size()) {
        throw std::runtime_error("Trajectory shard index out of range");
    }
    const uint64_t begin = offsets_[index];
    const uint64_t end = index + 1 < offsets_.size() ? offsets_[index + 1] : file_size_;
    if (end <= begin) throw std::runtime_error("Invalid trajectory shard history boundary");
    if (begin > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("Trajectory shard offset exceeds streamoff range");
    }

    in_.clear();
    in_.seekg(static_cast<std::streamoff>(begin), std::ios::beg);
    if (!in_) throw std::runtime_error("Failed to seek trajectory shard history offset");
    return read_history_binary_stream(in_, end - begin);
}

std::vector<GameHistory> BinaryShardReader::read_all() {
    std::vector<GameHistory> histories;
    histories.reserve(offsets_.size());
    for (size_t i = 0; i < offsets_.size(); ++i) {
        histories.push_back(read_at(i));
    }
    return histories;
}

BinaryShardWriter::BinaryShardWriter(const std::string& path, size_t expected_history_count)
    : path_(path),
      temp_path_(make_temp_path(path)),
      expected_history_count_(expected_history_count),
      written_count_(0),
      closed_(false),
      out_(temp_path_, std::ios::binary | std::ios::trunc),
      offsets_(expected_history_count, 0) {
    if (!out_) throw std::runtime_error("Failed to open temporary trajectory binary shard path");

    BinaryShardHeader header{};
    std::memcpy(header.magic, kShardMagic, sizeof(kShardMagic));
    header.version = kShardVersion;
    header.history_count = checked_u32(expected_history_count, "history_count");
    header.offset_table_bytes = checked_mul_u64(
        expected_history_count,
        sizeof(uint64_t),
        "offset_table_bytes");
    write_raw(out_, header);

    for (size_t i = 0; i < expected_history_count; ++i) {
        const uint64_t zero = 0;
        write_raw(out_, zero);
    }
}

BinaryShardWriter::~BinaryShardWriter() {
    if (!closed_) {
        try {
            close();
        } catch (...) {
            out_.close();
            remove_file_noexcept(temp_path_);
            closed_ = true;
        }
    }
}

void BinaryShardWriter::write_history(const GameHistory& history) {
    if (closed_) throw std::runtime_error("Cannot write to a closed binary shard");
    if (written_count_ >= expected_history_count_) {
        throw std::runtime_error("Binary shard writer received more histories than expected");
    }
    offsets_[written_count_] = checked_u64_stream_pos(out_);
    write_history_binary_stream(out_, history);
    ++written_count_;
}

void BinaryShardWriter::close() {
    if (closed_) return;
    if (written_count_ != expected_history_count_) {
        out_.close();
        remove_file_noexcept(temp_path_);
        closed_ = true;
        throw std::runtime_error("Binary shard writer closed before all expected histories were written");
    }

    try {
        out_.seekp(static_cast<std::streamoff>(sizeof(BinaryShardHeader)), std::ios::beg);
        if (!out_) throw std::runtime_error("Failed to seek binary shard offset table");
        for (uint64_t offset : offsets_) write_raw(out_, offset);
        out_.flush();
        if (!out_) throw std::runtime_error("Failed to flush trajectory binary shard path");
        out_.close();
        if (!out_) throw std::runtime_error("Failed to close trajectory binary shard path");
        publish_temp_file(temp_path_, path_);
        closed_ = true;
    } catch (...) {
        out_.close();
        remove_file_noexcept(temp_path_);
        closed_ = true;
        throw;
    }
}

struct AsyncBinaryShardWriter::Impl {
    Impl(const std::string& path, size_t expected_history_count, size_t max_queue_size)
        : path(path),
          expected_history_count(expected_history_count),
          max_queue_size(std::max<size_t>(1, max_queue_size)),
          writer(path, expected_history_count),
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
        throw std::runtime_error("Async binary shard writer failed");
    }

    void write(GameHistory history) {
        std::unique_lock<std::mutex> lock(mutex);
        if (lifecycle == AsyncBinaryShardWriterState::Failed) rethrow_failure_locked();
        if (lifecycle != AsyncBinaryShardWriterState::Open) {
            throw std::runtime_error("Cannot write after async binary shard closing begins");
        }
        if (enqueued_count >= expected_history_count) {
            throw std::runtime_error("Async binary shard writer received more histories than expected");
        }

        not_full.wait(lock, [&] {
            return queue.size() < max_queue_size
                || lifecycle != AsyncBinaryShardWriterState::Open;
        });
        if (lifecycle == AsyncBinaryShardWriterState::Failed) rethrow_failure_locked();
        if (lifecycle != AsyncBinaryShardWriterState::Open) {
            throw std::runtime_error("Cannot write after async binary shard closing begins");
        }

        queue.push_back(std::move(history));
        ++enqueued_count;
        not_empty.notify_one();
    }

    void close() {
        bool join_worker = false;
        {
            std::unique_lock<std::mutex> lock(mutex);
            if (lifecycle == AsyncBinaryShardWriterState::Open) {
                lifecycle = AsyncBinaryShardWriterState::Closing;
                not_empty.notify_all();
                not_full.notify_all();
                state_changed.notify_all();
            }

            if (!join_claimed) {
                join_claimed = true;
                join_worker = true;
            } else {
                state_changed.wait(lock, [&] {
                    return lifecycle == AsyncBinaryShardWriterState::Closed
                        || lifecycle == AsyncBinaryShardWriterState::Failed;
                });
                if (lifecycle == AsyncBinaryShardWriterState::Failed) {
                    rethrow_failure_locked();
                }
                return;
            }
        }

        if (join_worker && worker.joinable()) worker.join();

        std::unique_lock<std::mutex> lock(mutex);
        state_changed.wait(lock, [&] {
            return lifecycle == AsyncBinaryShardWriterState::Closed
                || lifecycle == AsyncBinaryShardWriterState::Failed;
        });
        if (lifecycle == AsyncBinaryShardWriterState::Failed) {
            rethrow_failure_locked();
        }
    }

    void writer_loop() {
        try {
            while (true) {
                GameHistory history;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    not_empty.wait(lock, [&] {
                        return !queue.empty()
                            || lifecycle != AsyncBinaryShardWriterState::Open;
                    });
                    if (queue.empty()
                        && lifecycle != AsyncBinaryShardWriterState::Open) {
                        break;
                    }
                    history = std::move(queue.front());
                    queue.pop_front();
                    not_full.notify_one();
                }
                writer.write_history(history);
            }
            writer.close();
            std::lock_guard<std::mutex> lock(mutex);
            lifecycle = AsyncBinaryShardWriterState::Closed;
            state_changed.notify_all();
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            error = std::current_exception();
            lifecycle = AsyncBinaryShardWriterState::Failed;
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
    AsyncBinaryShardWriterState lifecycle = AsyncBinaryShardWriterState::Open;
    bool join_claimed = false;
    std::exception_ptr error;

    BinaryShardWriter writer;
    mutable std::mutex mutex;
    std::condition_variable not_empty;
    std::condition_variable not_full;
    std::condition_variable state_changed;
    std::deque<GameHistory> queue;
    std::thread worker;
};

AsyncBinaryShardWriter::AsyncBinaryShardWriter(
    const std::string& path,
    size_t expected_history_count,
    size_t max_queue_size
) : impl_(new Impl(path, expected_history_count, max_queue_size)) {}

AsyncBinaryShardWriter::~AsyncBinaryShardWriter() = default;

void AsyncBinaryShardWriter::write_history(GameHistory history) {
    impl_->write(std::move(history));
}

void AsyncBinaryShardWriter::close() {
    impl_->close();
}

size_t AsyncBinaryShardWriter::expected_history_count() const {
    return impl_->expected_history_count;
}

size_t AsyncBinaryShardWriter::enqueued_count() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->enqueued_count;
}

size_t AsyncBinaryShardWriter::max_queue_size() const {
    return impl_->max_queue_size;
}

AsyncBinaryShardWriterState AsyncBinaryShardWriter::state() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->lifecycle;
}

bool AsyncBinaryShardWriter::closed() const {
    return state() == AsyncBinaryShardWriterState::Closed;
}

void write_histories_binary_shard(
    const std::vector<GameHistory>& histories,
    const std::string& path
) {
    BinaryShardWriter writer(path, histories.size());
    for (const auto& history : histories) {
        writer.write_history(history);
    }
    writer.close();
}

std::vector<GameHistory> read_histories_binary_shard(const std::string& path) {
    BinaryShardReader reader(path);
    return reader.read_all();
}

GameHistory read_history_binary_shard_at(const std::string& path, size_t index) {
    BinaryShardReader reader(path);
    return reader.read_at(index);
}

} // namespace tdmz
