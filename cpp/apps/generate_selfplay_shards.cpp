#include "tdmz/core/board_tables.hpp"
#include "tdmz/core/engine.hpp"
#include "tdmz/mcts/dummy_network.hpp"
#include "tdmz/selfplay/selfplay_config.hpp"
#include "tdmz/selfplay/selfplay_runner.hpp"
#include "tdmz/selfplay/trajectory_writer.hpp"
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
    int workers = 4;
    int max_steps = 64;
    int simulations = 32;
    int latent_top_k = 16;
    int max_nodes = 8192;
    int writer_queue_size = 4;
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

std::string json_escape(const std::string& s) {
    std::ostringstream out;
    for (char ch : s) {
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
    double sum = static_cast<double>(history.steps.size()) * 17.0 + static_cast<double>(history.total_reward);
    for (const auto& step : history.steps) {
        sum += step.action * 0.125;
        sum += step.reward * 3.0;
        sum += step.root_value * 5.0;
        sum += step.money * 0.01;
        sum += step.base_hp * 0.02;
        sum += step.wave * 0.5;
        if (!step.policy_target.empty()) sum += step.policy_target[step.action >= 0 ? step.action : 0] * 11.0;
        for (size_t i = 0; i < step.observation.size(); i += 97) {
            sum += step.observation[i] * static_cast<double>((i % 29) + 1);
        }
    }
    return sum;
}

int parse_int_arg(const std::vector<std::string>& args, size_t& i, const char* name) {
    if (i + 1 >= args.size()) throw std::runtime_error(std::string("Missing value for ") + name);
    ++i;
    int value = std::stoi(args[i]);
    return value;
}

uint64_t parse_u64_arg(const std::vector<std::string>& args, size_t& i, const char* name) {
    if (i + 1 >= args.size()) throw std::runtime_error(std::string("Missing value for ") + name);
    ++i;
    return static_cast<uint64_t>(std::stoull(args[i]));
}

std::string parse_string_arg(const std::vector<std::string>& args, size_t& i, const char* name) {
    if (i + 1 >= args.size()) throw std::runtime_error(std::string("Missing value for ") + name);
    ++i;
    return args[i];
}

void print_help(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "\n"
        << "Options:\n"
        << "  --games N          Total self-play games to generate. Default: 64\n"
        << "  --workers N        Number of actor workers. Default: min(4, hardware_concurrency)\n"
        << "  --sims N           MCTS simulations per move. Default: 32\n"
        << "  --max-steps N      Max steps per game. Default: 64\n"
        << "  --latent-top-k N   MCTS latent expansion top-k. Default: 16\n"
        << "  --max-nodes N      MCTS node pool limit. Default: 8192\n"
        << "  --seed N           First game seed. Default: 0\n"
        << "  --prefix PATH      Output prefix. Default: data/selfplay/train\n"
        << "  --budgeted         Enable budgeted wave mode. Default: fixed waves\n"
        << "  --fixed            Force fixed wave mode\n"
        << "  --async-write      Use bounded memory queue + writer thread per actor. Default\n"
        << "  --sync-write       Write shard synchronously in each actor\n"
        << "  --writer-queue N   Async writer queue size per actor. Default: 4\n"
        << "  --help             Show this help\n";
}

GeneratorConfig parse_args(int argc, char** argv) {
    GeneratorConfig cfg;
    unsigned int hc = std::thread::hardware_concurrency();
    cfg.workers = std::max(1, std::min(4, static_cast<int>(hc ? hc : 4)));

    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);

    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a == "--games") cfg.games = parse_int_arg(args, i, "--games");
        else if (a == "--workers") cfg.workers = parse_int_arg(args, i, "--workers");
        else if (a == "--sims") cfg.simulations = parse_int_arg(args, i, "--sims");
        else if (a == "--max-steps") cfg.max_steps = parse_int_arg(args, i, "--max-steps");
        else if (a == "--latent-top-k") cfg.latent_top_k = parse_int_arg(args, i, "--latent-top-k");
        else if (a == "--max-nodes") cfg.max_nodes = parse_int_arg(args, i, "--max-nodes");
        else if (a == "--writer-queue") cfg.writer_queue_size = parse_int_arg(args, i, "--writer-queue");
        else if (a == "--seed") cfg.seed = parse_u64_arg(args, i, "--seed");
        else if (a == "--prefix") cfg.prefix = parse_string_arg(args, i, "--prefix");
        else if (a == "--budgeted") cfg.budgeted = true;
        else if (a == "--fixed") cfg.budgeted = false;
        else if (a == "--async-write") cfg.async_write = true;
        else if (a == "--sync-write") cfg.async_write = false;
        else if (a == "--help" || a == "-h") {
            print_help(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + a);
        }
    }

    if (cfg.games <= 0) throw std::runtime_error("--games must be positive");
    if (cfg.workers <= 0) throw std::runtime_error("--workers must be positive");
    if (cfg.max_steps <= 0) throw std::runtime_error("--max-steps must be positive");
    if (cfg.simulations <= 0) throw std::runtime_error("--sims must be positive");
    if (cfg.latent_top_k <= 0) throw std::runtime_error("--latent-top-k must be positive");
    if (cfg.max_nodes <= 0) throw std::runtime_error("--max-nodes must be positive");
    if (cfg.writer_queue_size <= 0) throw std::runtime_error("--writer-queue must be positive");
    cfg.workers = std::min(cfg.workers, cfg.games);
    return cfg;
}

std::vector<WorkerPlan> make_worker_plans(const GeneratorConfig& cfg) {
    std::vector<WorkerPlan> plans;
    plans.reserve(cfg.workers);
    int base = cfg.games / cfg.workers;
    int rem = cfg.games % cfg.workers;
    int seed_offset = 0;
    for (int w = 0; w < cfg.workers; ++w) {
        int count = base + (w < rem ? 1 : 0);
        WorkerPlan plan;
        plan.worker_id = w;
        plan.games = count;
        plan.seed_start = cfg.seed + static_cast<uint64_t>(seed_offset);
        plan.shard_path = cfg.prefix + "_w" + std::to_string(w) + ".tdmzshd";
        plans.push_back(plan);
        seed_offset += count;
    }
    return plans;
}

template <typename Writer>
void run_generation_loop(const GeneratorConfig& cfg, const WorkerPlan& plan, Writer& writer, WorkerResult& result) {
    DummyNetwork net;
    SelfPlayConfig sp;
    sp.max_steps = cfg.max_steps;
    sp.mcts.num_simulations = cfg.simulations;
    sp.mcts.latent_top_k = cfg.latent_top_k;
    sp.mcts.max_nodes = cfg.max_nodes;
    sp.save_observations = true;
    sp.save_legal_mask = true;

    for (int g = 0; g < plan.games; ++g) {
        sp.seed = plan.seed_start + static_cast<uint64_t>(g);
        TDEngine env(11, 11, sp.seed, cfg.budgeted);
        SelfPlayRunner runner(sp);
        GameHistory history = runner.run(env, net);

        int steps = static_cast<int>(history.steps.size());
        result.steps += steps;
        result.simulations += steps * cfg.simulations;
        if (history.terminal) ++result.terminal_games;
        result.total_reward += history.total_reward;
        result.checksum += checksum_history(history);

        writer.write_history(std::move(history));
    }
}

WorkerResult run_worker(const GeneratorConfig& cfg, const WorkerPlan& plan) {
    WorkerResult result;
    result.worker_id = plan.worker_id;
    result.games = plan.games;
    result.seed_start = plan.seed_start;
    result.shard_path = plan.shard_path;

    auto start = std::chrono::high_resolution_clock::now();
    if (cfg.async_write) {
        AsyncBinaryShardWriter writer(
            plan.shard_path,
            static_cast<size_t>(plan.games),
            static_cast<size_t>(cfg.writer_queue_size)
        );
        run_generation_loop(cfg, plan, writer, result);
        writer.close();
    } else {
        BinaryShardWriter writer(plan.shard_path, static_cast<size_t>(plan.games));
        run_generation_loop(cfg, plan, writer, result);
        writer.close();
    }
    result.bytes_written = file_size_bytes(plan.shard_path);
    auto end = std::chrono::high_resolution_clock::now();
    result.seconds = std::chrono::duration<double>(end - start).count();
    return result;
}

void write_index_json(const GeneratorConfig& cfg, const std::vector<WorkerResult>& results, const std::string& index_path, double total_seconds) {
    int total_games = 0;
    int total_steps = 0;
    int total_sims = 0;
    int total_terminals = 0;
    uint64_t total_bytes = 0;
    double total_reward = 0.0;
    double checksum = 0.0;
    for (const auto& r : results) {
        total_games += r.games;
        total_steps += r.steps;
        total_sims += r.simulations;
        total_terminals += r.terminal_games;
        total_bytes += r.bytes_written;
        total_reward += r.total_reward;
        checksum += r.checksum;
    }

    std::ofstream out(index_path);
    if (!out) throw std::runtime_error("Failed to open self-play index JSON");
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"format\": \"tdmz_selfplay_shard_index\",\n";
    out << "  \"version\": 1,\n";
    out << "  \"prefix\": \"" << json_escape(cfg.prefix) << "\",\n";
    out << "  \"budgeted\": " << (cfg.budgeted ? "true" : "false") << ",\n";
    out << "  \"async_write\": " << (cfg.async_write ? "true" : "false") << ",\n";
    out << "  \"writer_queue_size\": " << cfg.writer_queue_size << ",\n";
    out << "  \"games\": " << total_games << ",\n";
    out << "  \"workers\": " << cfg.workers << ",\n";
    out << "  \"max_steps\": " << cfg.max_steps << ",\n";
    out << "  \"simulations_per_move\": " << cfg.simulations << ",\n";
    out << "  \"latent_top_k\": " << cfg.latent_top_k << ",\n";
    out << "  \"max_nodes\": " << cfg.max_nodes << ",\n";
    out << "  \"seed_start\": " << cfg.seed << ",\n";
    out << "  \"total_steps\": " << total_steps << ",\n";
    out << "  \"total_simulations\": " << total_sims << ",\n";
    out << "  \"terminal_games\": " << total_terminals << ",\n";
    out << "  \"total_reward\": " << total_reward << ",\n";
    out << "  \"total_seconds\": " << total_seconds << ",\n";
    out << "  \"games_per_second\": " << (total_games / total_seconds) << ",\n";
    out << "  \"steps_per_second\": " << (total_steps / total_seconds) << ",\n";
    out << "  \"simulations_per_second\": " << (total_sims / total_seconds) << ",\n";
    out << "  \"bytes_written\": " << total_bytes << ",\n";
    out << "  \"mb_written\": " << (static_cast<double>(total_bytes) / (1024.0 * 1024.0)) << ",\n";
    out << "  \"mb_per_second\": " << ((static_cast<double>(total_bytes) / (1024.0 * 1024.0)) / total_seconds) << ",\n";
    out << "  \"checksum\": " << checksum << ",\n";
    out << "  \"shards\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out << "    {\"worker\": " << r.worker_id
            << ", \"path\": \"" << json_escape(r.shard_path) << "\""
            << ", \"games\": " << r.games
            << ", \"seed_start\": " << r.seed_start
            << ", \"steps\": " << r.steps
            << ", \"simulations\": " << r.simulations
            << ", \"terminal_games\": " << r.terminal_games
            << ", \"total_reward\": " << r.total_reward
            << ", \"seconds\": " << r.seconds
            << ", \"bytes\": " << r.bytes_written
            << ", \"checksum\": " << r.checksum
            << "}" << (i + 1 == results.size() ? "" : ",") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

void print_summary_json(const GeneratorConfig& cfg, const std::vector<WorkerResult>& results, const std::string& index_path, double seconds) {
    int total_games = 0;
    int total_steps = 0;
    int total_sims = 0;
    uint64_t total_bytes = 0;
    double checksum = 0.0;
    for (const auto& r : results) {
        total_games += r.games;
        total_steps += r.steps;
        total_sims += r.simulations;
        total_bytes += r.bytes_written;
        checksum += r.checksum;
    }
    double mb = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
    std::cout << std::fixed << std::setprecision(6)
              << "{\"index\":\"" << json_escape(index_path) << "\""
              << ",\"budgeted\":" << (cfg.budgeted ? "true" : "false")
              << ",\"async_write\":" << (cfg.async_write ? "true" : "false")
              << ",\"writer_queue_size\":" << cfg.writer_queue_size
              << ",\"workers\":" << cfg.workers
              << ",\"games\":" << total_games
              << ",\"steps\":" << total_steps
              << ",\"simulations\":" << total_sims
              << ",\"seconds\":" << seconds
              << ",\"games_per_second\":" << (total_games / seconds)
              << ",\"steps_per_second\":" << (total_steps / seconds)
              << ",\"simulations_per_second\":" << (total_sims / seconds)
              << ",\"mb_written\":" << mb
              << ",\"mb_per_second\":" << (mb / seconds)
              << ",\"checksum\":" << checksum
              << "}" << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    try {
        GeneratorConfig cfg = parse_args(argc, argv);

        std::filesystem::path prefix_path(cfg.prefix);
        if (!prefix_path.parent_path().empty()) {
            std::filesystem::create_directories(prefix_path.parent_path());
        }

        (void)board_tables();
        {
            TDEngine warmup(11, 11, 999);
            (void)warmup.legal_actions();
        }

        std::vector<WorkerPlan> plans = make_worker_plans(cfg);
        std::vector<WorkerResult> results(plans.size());
        std::vector<std::thread> threads;
        threads.reserve(plans.size());

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < plans.size(); ++i) {
            threads.emplace_back([&, i]() {
                try {
                    results[i] = run_worker(cfg, plans[i]);
                } catch (const std::exception& e) {
                    results[i].worker_id = plans[i].worker_id;
                    results[i].games = plans[i].games;
                    results[i].seed_start = plans[i].seed_start;
                    results[i].shard_path = plans[i].shard_path;
                    results[i].error = e.what();
                }
            });
        }
        for (auto& t : threads) t.join();
        auto end = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration<double>(end - start).count();

        for (const auto& r : results) {
            if (!r.error.empty()) {
                std::cerr << "worker " << r.worker_id << " failed: " << r.error << std::endl;
                return 1;
            }
        }

        std::string index_path = cfg.prefix + ".index.json";
        write_index_json(cfg, results, index_path, seconds);
        print_summary_json(cfg, results, index_path, seconds);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
