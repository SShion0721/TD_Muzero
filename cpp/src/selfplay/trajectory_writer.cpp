#include "tdmz/selfplay/trajectory_writer.hpp"
#include "tdmz/core/action.hpp"
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

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

template <typename T>
void write_raw_vector(std::ostream& out, const std::vector<T>& values, size_t expected_size) {
    if (values.size() != expected_size) {
        throw std::runtime_error("Binary trajectory vector size mismatch");
    }
    if (!values.empty()) {
        out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
        if (!out) throw std::runtime_error("Failed while writing binary trajectory vector");
    }
}

template <typename T>
void read_raw_vector(std::istream& in, std::vector<T>& values, size_t expected_size) {
    values.resize(expected_size);
    if (!values.empty()) {
        in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
        if (!in) throw std::runtime_error("Failed while reading binary trajectory vector");
    }
}

uint32_t checked_u32(size_t value, const char* name) {
    if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string(name) + " exceeds uint32_t range");
    }
    return static_cast<uint32_t>(value);
}

uint64_t checked_u64_stream_pos(std::ostream& out) {
    auto pos = out.tellp();
    if (pos < 0) throw std::runtime_error("Failed to query binary output stream position");
    return static_cast<uint64_t>(pos);
}

void validate_binary_step_sizes(const GameHistory& history, uint32_t obs_size, uint32_t policy_size, uint32_t legal_size) {
    for (const auto& step : history.steps) {
        if (step.observation.size() != obs_size ||
            step.policy_target.size() != policy_size ||
            step.legal_mask.size() != legal_size) {
            throw std::runtime_error("All binary trajectory steps must have identical tensor sizes");
        }
    }
}

void write_history_binary_stream(std::ostream& out, const GameHistory& history) {
    uint32_t step_count = checked_u32(history.steps.size(), "step_count");
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

GameHistory read_history_binary_stream(std::istream& in) {
    BinaryHistoryHeader header{};
    read_raw(in, header);
    if (std::memcmp(header.magic, kBinaryMagic, sizeof(kBinaryMagic)) != 0) {
        throw std::runtime_error("Invalid trajectory binary magic");
    }
    if (header.version != kBinaryVersion) {
        throw std::runtime_error("Unsupported trajectory binary version");
    }

    GameHistory history;
    history.seed = header.seed;
    history.max_steps = header.max_steps;
    history.terminal = header.terminal != 0;
    history.total_reward = header.total_reward;
    history.steps.reserve(header.step_count);

    for (uint32_t i = 0; i < header.step_count; ++i) {
        BinaryStepHeader sh{};
        read_raw(in, sh);
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
        read_raw_vector(in, step.observation, header.observation_size);
        read_raw_vector(in, step.policy_target, header.policy_size);
        read_raw_vector(in, step.legal_mask, header.legal_mask_size);
        history.steps.push_back(std::move(step));
    }

    return history;
}

std::vector<uint64_t> read_shard_offsets(std::istream& in, BinaryShardHeader& header) {
    read_raw(in, header);
    if (std::memcmp(header.magic, kShardMagic, sizeof(kShardMagic)) != 0) {
        throw std::runtime_error("Invalid trajectory shard magic");
    }
    if (header.version != kShardVersion) {
        throw std::runtime_error("Unsupported trajectory shard version");
    }
    if (header.offset_table_bytes != static_cast<uint64_t>(header.history_count) * sizeof(uint64_t)) {
        throw std::runtime_error("Invalid trajectory shard offset table size");
    }

    std::vector<uint64_t> offsets(header.history_count, 0);
    for (auto& offset : offsets) {
        read_raw(in, offset);
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
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open trajectory binary path");
    write_history_binary_stream(out, history);
}

GameHistory read_history_binary(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open trajectory binary path");
    return read_history_binary_stream(in);
}

BinaryShardWriter::BinaryShardWriter(const std::string& path, size_t expected_history_count)
    : path_(path),
      expected_history_count_(expected_history_count),
      written_count_(0),
      closed_(false),
      out_(path, std::ios::binary),
      offsets_(expected_history_count, 0) {
    if (!out_) throw std::runtime_error("Failed to open trajectory binary shard path");

    BinaryShardHeader header{};
    std::memcpy(header.magic, kShardMagic, sizeof(kShardMagic));
    header.version = kShardVersion;
    header.history_count = checked_u32(expected_history_count, "history_count");
    header.offset_table_bytes = static_cast<uint64_t>(expected_history_count) * sizeof(uint64_t);
    write_raw(out_, header);

    for (size_t i = 0; i < expected_history_count; ++i) {
        uint64_t zero = 0;
        write_raw(out_, zero);
    }
}

BinaryShardWriter::~BinaryShardWriter() {
    if (!closed_) {
        try {
            close();
        } catch (...) {
            // Destructors must not throw. Call close() explicitly in production paths
            // to surface write/seek/count errors.
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
        throw std::runtime_error("Binary shard writer closed before all expected histories were written");
    }

    out_.seekp(static_cast<std::streamoff>(sizeof(BinaryShardHeader)), std::ios::beg);
    if (!out_) throw std::runtime_error("Failed to seek binary shard offset table");
    for (uint64_t offset : offsets_) {
        write_raw(out_, offset);
    }
    out_.close();
    if (!out_) throw std::runtime_error("Failed to close trajectory binary shard path");
    closed_ = true;
}

struct AsyncBinaryShardWriter::Impl {
    Impl(const std::string& path, size_t expected_history_count, size_t max_queue_size)
        : path(path),
          expected_history_count(expected_history_count),
          max_queue_size(std::max<size_t>(1, max_queue_size)),
          writer(path, expected_history_count),
          worker(&Impl::writer_loop, this) {}

    ~Impl() {
        if (!closed) {
            try {
                close();
            } catch (...) {
                // Destructors must not throw. Explicit close() surfaces errors.
            }
        }
    }

    void write(GameHistory history) {
        std::unique_lock<std::mutex> lock(mutex);
        if (closed) throw std::runtime_error("Cannot write to a closed async binary shard");
        if (error) std::rethrow_exception(error);
        if (enqueued_count >= expected_history_count) {
            throw std::runtime_error("Async binary shard writer received more histories than expected");
        }

        not_full.wait(lock, [&] {
            return queue.size() < max_queue_size || error || closed;
        });
        if (closed) throw std::runtime_error("Cannot write to a closed async binary shard");
        if (error) std::rethrow_exception(error);

        queue.push_back(std::move(history));
        ++enqueued_count;
        not_empty.notify_one();
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (closed) return;
            producer_done = true;
            not_empty.notify_all();
            not_full.notify_all();
        }

        if (worker.joinable()) worker.join();

        std::lock_guard<std::mutex> lock(mutex);
        closed = true;
        if (error) std::rethrow_exception(error);
        if (enqueued_count != expected_history_count) {
            throw std::runtime_error("Async binary shard writer closed before all expected histories were enqueued");
        }
    }

    void writer_loop() {
        try {
            while (true) {
                GameHistory history;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    not_empty.wait(lock, [&] {
                        return !queue.empty() || producer_done;
                    });

                    if (queue.empty() && producer_done) break;
                    history = std::move(queue.front());
                    queue.pop_front();
                    not_full.notify_one();
                }

                writer.write_history(history);
            }
            writer.close();
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            error = std::current_exception();
            producer_done = true;
            not_empty.notify_all();
            not_full.notify_all();
        }
    }

    std::string path;
    size_t expected_history_count = 0;
    size_t max_queue_size = 1;
    size_t enqueued_count = 0;
    bool producer_done = false;
    bool closed = false;
    std::exception_ptr error;

    BinaryShardWriter writer;
    std::mutex mutex;
    std::condition_variable not_empty;
    std::condition_variable not_full;
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

bool AsyncBinaryShardWriter::closed() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->closed;
}

void write_histories_binary_shard(const std::vector<GameHistory>& histories, const std::string& path) {
    BinaryShardWriter writer(path, histories.size());
    for (const auto& history : histories) {
        writer.write_history(history);
    }
    writer.close();
}

std::vector<GameHistory> read_histories_binary_shard(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open trajectory binary shard path");

    BinaryShardHeader header{};
    std::vector<uint64_t> offsets = read_shard_offsets(in, header);
    std::vector<GameHistory> histories;
    histories.reserve(header.history_count);
    for (uint32_t i = 0; i < header.history_count; ++i) {
        in.clear();
        in.seekg(static_cast<std::streamoff>(offsets[i]), std::ios::beg);
        if (!in) throw std::runtime_error("Failed to seek trajectory shard history offset");
        histories.push_back(read_history_binary_stream(in));
    }
    return histories;
}

GameHistory read_history_binary_shard_at(const std::string& path, size_t index) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open trajectory binary shard path");

    BinaryShardHeader header{};
    std::vector<uint64_t> offsets = read_shard_offsets(in, header);
    if (index >= offsets.size()) {
        throw std::runtime_error("Trajectory shard index out of range");
    }
    in.clear();
    in.seekg(static_cast<std::streamoff>(offsets[index]), std::ios::beg);
    if (!in) throw std::runtime_error("Failed to seek trajectory shard history offset");
    return read_history_binary_stream(in);
}

} // namespace tdmz
