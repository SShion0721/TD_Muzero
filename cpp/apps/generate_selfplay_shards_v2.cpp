#include "tdmz/core/board_tables.hpp"
#include "tdmz/core/compatibility.hpp"
#include "tdmz/core/engine.hpp"
#include "tdmz/mcts/dummy_network.hpp"
#include "tdmz/selfplay/selfplay_config.hpp"
#include "tdmz/selfplay/selfplay_runner.hpp"
#include "tdmz/selfplay/transition_shard.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace tdmz;

namespace {

struct GeneratorConfig {
    int games = 64;
    int workers = 8;
    int max_steps = 64;
    int simulations = 32;
    int latent_top_k = 16;
    int max_nodes = 8192;
    int recurrent_batch_size = 1;
    int writer_queue_size = 8;
    uint64_t seed = 0;
    bool budgeted = false;
    bool async_write = true;
    std::string prefix = "data/selfplay/train";
};

struct WorkerPlan {
    int worker_id = 0;
    int games = 0;
    uint64_t seed_start = 0;
    std::string shard_path;
};

struct WorkerResult {
    int worker_id = 0;
    int games = 0;
    int steps = 0;
    int simulations = 0;
    int terminal_games = 0;
    int truncated_games = 0;
    int bootstrap_states = 0;
    uint64_t seed_start = 0;
    uint64_t bytes_written = 0;
    double seconds = 0.0;
    double total_reward = 0.0;
    double checksum = 0.0;
    std::string shard_path;
    std::string error;
};

uint64_t file_size_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return 0;
    return static_cast<uint64_t>(in.tellg());
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << ch; break;
        }
    }
    return out.str();
}

double checksum_history(const GameHistory& history) {
    double sum = static_cast<double>(history.steps.size()) * 17.0
        + static_cast<double>(history.total_reward);
    for (const auto& step : history.steps) {
        sum += step.action * 0.125;
        sum += step.reward * 3.0;
        sum += step.root_value * 5.0;
        sum += step.money * 0.01;
        sum += step.base_hp * 0.02;
        sum += step.wave * 0.5;
        if (!step.policy_target.empty()) {
            const int action = std::max(
                0,
                std::min(step.action, static_cast<int>(step.policy_target.size()) - 1));
            sum += step.policy_target[static_cast<size_t>(action)] * 11.0;
        }
        for (size_t i = 0; i < step.observation.size(); i += 97) {
            sum += step.observation[i] * static_cast<double>((i % 29) + 1);
        }
    }
    if (history.bootstrap_state) {
        sum += history.bootstrap_state->root_value * 7.0;
        for (size_t i = 0; i < history.bootstrap_state->observation.size(); i += 97) {
            sum += history.bootstrap_state->observation[i]
                * static_cast<double>((i % 31) + 1);
        }
    }
    return sum;
}

int parse_int_arg(const std::vector<std::string>& args, size_t& index, const char* name) {
    if (index + 1 >= args.size()) throw std::runtime_error(std::string("Missing value for ") + name);
    return std::stoi(args[++index]);
}

uint64_t parse_u64_arg(const std::vector<std::string>& args, size_t& index, const char* name) {
    if (index + 1 >= args.size()) throw std::runtime_error(std::string("Missing value for ") + name);
    return static_cast<uint64_t>(std::stoull(args[++index]));
}

std::string parse_string_arg(const std::vector<std::string>& args, size_t& index, const char* name) {
    if (index + 1 >= args.size()) throw std::runtime_error(std::string("Missing value for ") + name);
    return args[++index];
}

void print_help(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n\n"
        << "Options:\n"
        << "  --games N                  Total self-play games. Default: 64\n"
        << "  --workers N                Actor workers. Default: min(8, hardware concurrency)\n"
        << "  --sims N                   MCTS simulations per move. Default: 32\n"
        << "  --max-steps N              Maximum steps per game. Default: 64\n"
        << "  --latent-top-k N           Latent expansion top-k. Default: 16\n"
        << "  --max-nodes N              MCTS node capacity. Default: 8192\n"
        << "  --recurrent-batch-size N   Unique leaves per recurrent call. Default: 1\n"
        << "  --seed N                   First game seed. Default: 0\n"
        << "  --prefix PATH              Output prefix. Default: data/selfplay/train\n"
        << "  --budgeted                 Enable budgeted waves\n"
        << "  --fixed                    Use fixed waves\n"
        << "  --async-write              Use one bounded writer queue per actor. Default\n"
        << "  --sync-write               Serialize in the actor thread\n"
        << "  --writer-queue N           Async queue capacity. Default: 8\n"
        << "  --help                     Show this help\n";
}

GeneratorConfig parse_args(int argc, char** argv) {
    GeneratorConfig config;
    const unsigned int hardware = std::thread::hardware_concurrency();
    config.workers = std::max(1, std::min(8, static_cast<int>(hardware ? hardware : 8)));

    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--games") config.games = parse_int_arg(args, i, "--games");
        else if (arg == "--workers") config.workers = parse_int_arg(args, i, "--workers");
        else if (arg == "--sims") config.simulations = parse_int_arg(args, i, "--sims");
        else if (arg == "--max-steps") config.max_steps = parse_int_arg(args, i, "--max-steps");
        else if (arg == "--latent-top-k") config.latent_top_k = parse_int_arg(args, i, "--latent-top-k");
        else if (arg == "--max-nodes") config.max_nodes = parse_int_arg(args, i, "--max-nodes");
        else if (arg == "--recurrent-batch-size") config.recurrent_batch_size = parse_int_arg(args, i, "--recurrent-batch-size");
        else if (arg == "--writer-queue") config.writer_queue_size = parse_int_arg(args, i, "--writer-queue");
        else if (arg == "--seed") config.seed = parse_u64_arg(args, i, "--seed");
        else if (arg == "--prefix") config.prefix = parse_string_arg(args, i, "--prefix");
        else if (arg == "--budgeted") config.budgeted = true;
        else if (arg == "--fixed") config.budgeted = false;
        else if (arg == "--async-write") config.async_write = true;
        else if (arg == "--sync-write") config.async_write = false;
        else if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (config.games <= 0) throw std::runtime_error("--games must be positive");
    if (config.workers <= 0) throw std::runtime_error("--workers must be positive");
    if (config.max_steps <= 0) throw std::runtime_error("--max-steps must be positive");
    if (config.simulations <= 0) throw std::runtime_error("--sims must be positive");
    if (config.latent_top_k <= 0) throw std::runtime_error("--latent-top-k must be positive");
    if (config.max_nodes <= 0) throw std::runtime_error("--max-nodes must be positive");
    if (config.recurrent_batch_size <= 0) throw std::runtime_error("--recurrent-batch-size must be positive");
    if (config.writer_queue_size <= 0) throw std::runtime_error("--writer-queue must be positive");
    config.workers = std::min(config.workers, config.games);
    return config;
}

std::vector<WorkerPlan> make_worker_plans(const GeneratorConfig& config) {
    std::vector<WorkerPlan> plans;
    plans.reserve(config.workers);
    const int base = config.games / config.workers;
    const int remainder = config.games % config.workers;
    int seed_offset = 0;
    for (int worker = 0; worker < config.workers; ++worker) {
        WorkerPlan plan;
        plan.worker_id = worker;
        plan.games = base + (worker < remainder ? 1 : 0);
        plan.seed_start = config.seed + static_cast<uint64_t>(seed_offset);
        plan.shard_path = config.prefix + "_w" + std::to_string(worker) + ".tdmzshd";
        plans.push_back(plan);
        seed_offset += plan.games;
    }
    return plans;
}

template <typename Writer>
void generate_worker_games(
    const GeneratorConfig& config,
    const WorkerPlan& plan,
    Writer& writer,
    WorkerResult& result
) {
    DummyNetwork network;
    SelfPlayConfig selfplay;
    selfplay.max_steps = config.max_steps;
    selfplay.mcts.num_simulations = config.simulations;
    selfplay.mcts.latent_top_k = config.latent_top_k;
    selfplay.mcts.max_nodes = config.max_nodes;
    selfplay.mcts.recurrent_batch_size = config.recurrent_batch_size;
    selfplay.save_observations = true;
    selfplay.save_legal_mask = true;
    selfplay.save_bootstrap_state = true;

    for (int game = 0; game < plan.games; ++game) {
        selfplay.seed = plan.seed_start + static_cast<uint64_t>(game);
        TDEngine environment(11, 11, selfplay.seed, config.budgeted);
        SelfPlayRunner runner(selfplay);
        GameHistory history = runner.run(environment, network);

        const int steps = static_cast<int>(history.steps.size());
        const int bootstrap_searches = history.bootstrap_state ? 1 : 0;
        result.steps += steps;
        result.simulations += (steps + bootstrap_searches) * config.simulations;
        if (history.terminal) ++result.terminal_games;
        if (history.truncated) ++result.truncated_games;
        result.bootstrap_states += bootstrap_searches;
        result.total_reward += history.total_reward;
        result.checksum += checksum_history(history);
        writer.write_history(std::move(history));
    }
}

WorkerResult run_worker(const GeneratorConfig& config, const WorkerPlan& plan) {
    WorkerResult result;
    result.worker_id = plan.worker_id;
    result.games = plan.games;
    result.seed_start = plan.seed_start;
    result.shard_path = plan.shard_path;

    const auto start = std::chrono::high_resolution_clock::now();
    if (config.async_write) {
        AsyncTransitionShardWriter writer(
            plan.shard_path,
            static_cast<size_t>(plan.games),
            static_cast<size_t>(config.writer_queue_size));
        generate_worker_games(config, plan, writer, result);
        writer.close();
    } else {
        TransitionShardWriter writer(plan.shard_path, static_cast<size_t>(plan.games));
        generate_worker_games(config, plan, writer, result);
        writer.close();
    }
    result.bytes_written = file_size_bytes(plan.shard_path);
    result.seconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start).count();
    return result;
}

void write_index_json(
    const GeneratorConfig& config,
    const std::vector<WorkerResult>& results,
    const std::string& index_path,
    double total_seconds
) {
    int total_games = 0;
    int total_steps = 0;
    int total_simulations = 0;
    int terminal_games = 0;
    int truncated_games = 0;
    int bootstrap_states = 0;
    uint64_t bytes_written = 0;
    double total_reward = 0.0;
    double checksum = 0.0;
    for (const auto& result : results) {
        total_games += result.games;
        total_steps += result.steps;
        total_simulations += result.simulations;
        terminal_games += result.terminal_games;
        truncated_games += result.truncated_games;
        bootstrap_states += result.bootstrap_states;
        bytes_written += result.bytes_written;
        total_reward += result.total_reward;
        checksum += result.checksum;
    }

    const CompatibilityMetadata compatibility = current_compatibility_metadata();
    std::ofstream out(index_path);
    if (!out) throw std::runtime_error("Failed to open self-play index JSON");
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"format\": \"" << kReplayIndexFormat << "\",\n";
    out << "  \"version\": " << kReplayIndexVersion << ",\n";
    out << "  \"replay_format_version\": " << compatibility.replay_format_version << ",\n";
    out << "  \"environment_rule_version\": " << compatibility.environment_rule_version << ",\n";
    out << "  \"observation_schema_version\": " << compatibility.observation_schema_version << ",\n";
    out << "  \"action_space_version\": " << compatibility.action_space_version << ",\n";
    out << "  \"reward_transform_version\": " << compatibility.reward_transform_version << ",\n";
    out << "  \"network_architecture_version\": " << compatibility.network_architecture_version << ",\n";
    out << "  \"board_width\": " << compatibility.board_width << ",\n";
    out << "  \"board_height\": " << compatibility.board_height << ",\n";
    out << "  \"observation_channels\": " << compatibility.observation_channels << ",\n";
    out << "  \"observation_size\": " << compatibility.observation_size << ",\n";
    out << "  \"action_space_size\": " << compatibility.action_space_size << ",\n";
    out << "  \"policy_size\": " << compatibility.policy_size << ",\n";
    out << "  \"legal_mask_size\": " << compatibility.legal_mask_size << ",\n";
    out << "  \"sampling_distribution\": \"uniform_position\",\n";
    out << "  \"prefix\": \"" << json_escape(config.prefix) << "\",\n";
    out << "  \"wave_mode\": \"" << (config.budgeted ? "budgeted" : "fixed") << "\",\n";
    out << "  \"budgeted\": " << (config.budgeted ? "true" : "false") << ",\n";
    out << "  \"async_write\": " << (config.async_write ? "true" : "false") << ",\n";
    out << "  \"writer_queue_size\": " << config.writer_queue_size << ",\n";
    out << "  \"games\": " << total_games << ",\n";
    out << "  \"workers\": " << config.workers << ",\n";
    out << "  \"max_steps\": " << config.max_steps << ",\n";
    out << "  \"simulations_per_move\": " << config.simulations << ",\n";
    out << "  \"latent_top_k\": " << config.latent_top_k << ",\n";
    out << "  \"max_nodes\": " << config.max_nodes << ",\n";
    out << "  \"recurrent_batch_size\": " << config.recurrent_batch_size << ",\n";
    out << "  \"seed_start\": " << config.seed << ",\n";
    out << "  \"total_steps\": " << total_steps << ",\n";
    out << "  \"total_simulations\": " << total_simulations << ",\n";
    out << "  \"terminal_games\": " << terminal_games << ",\n";
    out << "  \"truncated_games\": " << truncated_games << ",\n";
    out << "  \"bootstrap_states\": " << bootstrap_states << ",\n";
    out << "  \"total_reward\": " << total_reward << ",\n";
    out << "  \"total_seconds\": " << total_seconds << ",\n";
    out << "  \"bytes_written\": " << bytes_written << ",\n";
    out << "  \"checksum\": " << checksum << ",\n";
    out << "  \"shards\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        out << "    {\"worker\": " << result.worker_id
            << ", \"path\": \"" << json_escape(result.shard_path) << "\""
            << ", \"games\": " << result.games
            << ", \"seed_start\": " << result.seed_start
            << ", \"steps\": " << result.steps
            << ", \"simulations\": " << result.simulations
            << ", \"terminal_games\": " << result.terminal_games
            << ", \"truncated_games\": " << result.truncated_games
            << ", \"bootstrap_states\": " << result.bootstrap_states
            << ", \"total_reward\": " << result.total_reward
            << ", \"seconds\": " << result.seconds
            << ", \"bytes\": " << result.bytes_written
            << ", \"checksum\": " << result.checksum
            << "}" << (i + 1 == results.size() ? "" : ",") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

void print_summary(
    const GeneratorConfig& config,
    const std::vector<WorkerResult>& results,
    const std::string& index_path,
    double seconds
) {
    int games = 0;
    int steps = 0;
    int simulations = 0;
    int bootstrap_states = 0;
    uint64_t bytes = 0;
    double checksum = 0.0;
    for (const auto& result : results) {
        games += result.games;
        steps += result.steps;
        simulations += result.simulations;
        bootstrap_states += result.bootstrap_states;
        bytes += result.bytes_written;
        checksum += result.checksum;
    }
    const double megabytes = static_cast<double>(bytes) / (1024.0 * 1024.0);
    std::cout << std::fixed << std::setprecision(6)
              << "{\"index\":\"" << json_escape(index_path) << "\""
              << ",\"replay_format_version\":" << kReplayBinaryFormatVersion
              << ",\"recurrent_batch_size\":" << config.recurrent_batch_size
              << ",\"budgeted\":" << (config.budgeted ? "true" : "false")
              << ",\"async_write\":" << (config.async_write ? "true" : "false")
              << ",\"workers\":" << config.workers
              << ",\"games\":" << games
              << ",\"steps\":" << steps
              << ",\"bootstrap_states\":" << bootstrap_states
              << ",\"simulations\":" << simulations
              << ",\"seconds\":" << seconds
              << ",\"games_per_second\":" << (games / seconds)
              << ",\"steps_per_second\":" << (steps / seconds)
              << ",\"simulations_per_second\":" << (simulations / seconds)
              << ",\"mb_written\":" << megabytes
              << ",\"mb_per_second\":" << (megabytes / seconds)
              << ",\"checksum\":" << checksum
              << "}" << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const GeneratorConfig config = parse_args(argc, argv);
        const std::filesystem::path prefix_path(config.prefix);
        if (!prefix_path.parent_path().empty()) {
            std::filesystem::create_directories(prefix_path.parent_path());
        }

        (void)board_tables();
        {
            TDEngine warmup(11, 11, 999);
            (void)warmup.legal_actions();
        }

        const std::vector<WorkerPlan> plans = make_worker_plans(config);
        std::vector<WorkerResult> results(plans.size());
        std::vector<std::thread> workers;
        workers.reserve(plans.size());

        const auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < plans.size(); ++i) {
            workers.emplace_back([&, i] {
                try {
                    results[i] = run_worker(config, plans[i]);
                } catch (const std::exception& error) {
                    results[i].worker_id = plans[i].worker_id;
                    results[i].games = plans[i].games;
                    results[i].seed_start = plans[i].seed_start;
                    results[i].shard_path = plans[i].shard_path;
                    results[i].error = error.what();
                }
            });
        }
        for (auto& worker : workers) worker.join();
        const double seconds = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();

        for (const auto& result : results) {
            if (!result.error.empty()) {
                std::cerr << "worker " << result.worker_id << " failed: " << result.error << std::endl;
                return 1;
            }
        }

        const std::string index_path = config.prefix + ".index.json";
        write_index_json(config, results, index_path, seconds);
        print_summary(config, results, index_path, seconds);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
