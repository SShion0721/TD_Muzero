#include "tdmz/core/golden_trace.hpp"
#include "tdmz/core/board_tables.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace tdmz {

namespace {

uint64_t fnv1a_update(uint64_t h, uint64_t v) {
    constexpr uint64_t kPrime = 1099511628211ULL;
    for (int i = 0; i < 8; ++i) {
        h ^= (v >> (i * 8)) & 0xffULL;
        h *= kPrime;
    }
    return h;
}

int scaled(float v) {
    return static_cast<int>(std::lround(v * 1000.0f));
}

int count_placeable(const std::array<std::array<bool, kBoardW>, kBoardH>& mask) {
    int count = 0;
    for (const auto& row : mask) {
        for (bool v : row) {
            if (v) ++count;
        }
    }
    return count;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char ch : s) {
        if (ch == '"') out += "\\\"";
        else if (ch == '\\') out += "\\\\";
        else out += ch;
    }
    return out;
}

} // namespace

std::vector<GoldenTraceCase> default_golden_trace_cases() {
    const auto& tables = board_tables();
    const int wait = tables.wait_action;

    GoldenTraceCase wait_only;
    wait_only.name = "wait_only_seed0";
    wait_only.seed = 0;
    wait_only.actions.assign(16, wait);

    GoldenTraceCase mixed_build;
    mixed_build.name = "mixed_build_seed1";
    mixed_build.seed = 1;
    mixed_build.actions = {
        encode_action(Action{ActionType::BuildBasic, 1, 1, 1}),
        wait,
        encode_action(Action{ActionType::BuildSniper, 5, 4, 1}),
        wait,
        wait,
        encode_action(Action{ActionType::Upgrade, 1, 1, 1}),
        wait,
        encode_action(Action{ActionType::Sell, 1, 1, 1}),
        wait,
        wait
    };

    GoldenTraceCase invalid_and_slow;
    invalid_and_slow.name = "invalid_and_slow_seed42";
    invalid_and_slow.seed = 42;
    invalid_and_slow.actions = {
        encode_action(Action{ActionType::Upgrade, 0, 0, 1}),
        encode_action(Action{ActionType::BuildSlow, 1, 4, 1}),
        wait,
        wait,
        encode_action(Action{ActionType::BuildBasic, 3, 4, 1}),
        wait,
        wait,
        wait,
        wait,
        wait
    };

    return {wait_only, mixed_build, invalid_and_slow};
}

std::vector<GoldenTraceStep> run_golden_trace_case(const GoldenTraceCase& tc) {
    TDEngine env(11, 11, tc.seed);
    std::vector<GoldenTraceStep> out;
    out.reserve(tc.actions.size());

    for (int i = 0; i < static_cast<int>(tc.actions.size()); ++i) {
        env.reset_perf_counters();
        StepResult result = env.step_action(tc.actions[i]);
        auto legal = env.legal_actions();
        auto placeable = env.compute_placeable_mask();

        GoldenTraceStep s;
        s.case_name = tc.name;
        s.seed = tc.seed;
        s.step = i;
        s.action = tc.actions[i];
        s.reward = result.reward;
        s.done = result.done;
        s.money = env.money();
        s.base_hp = env.base_hp();
        s.wave = env.wave();
        s.time = env.time();
        s.tower_count = static_cast<int>(env.towers().size());
        s.enemy_count = static_cast<int>(env.enemies().size());
        s.to_spawn_count = env.enemies_to_spawn_count();
        s.legal_action_count = static_cast<int>(legal.size());
        s.placeable_count = count_placeable(placeable);
        s.perf = env.perf_counters();
        out.push_back(s);

        if (result.done) break;
    }

    return out;
}

uint64_t hash_golden_trace(const std::vector<GoldenTraceStep>& steps) {
    uint64_t h = 1469598103934665603ULL;
    for (const auto& s : steps) {
        for (unsigned char ch : s.case_name) h = fnv1a_update(h, ch);
        h = fnv1a_update(h, s.seed);
        h = fnv1a_update(h, static_cast<uint64_t>(s.step));
        h = fnv1a_update(h, static_cast<uint64_t>(s.action));
        h = fnv1a_update(h, static_cast<uint64_t>(scaled(s.reward)));
        h = fnv1a_update(h, static_cast<uint64_t>(s.done ? 1 : 0));
        h = fnv1a_update(h, static_cast<uint64_t>(s.money));
        h = fnv1a_update(h, static_cast<uint64_t>(s.base_hp));
        h = fnv1a_update(h, static_cast<uint64_t>(s.wave));
        h = fnv1a_update(h, static_cast<uint64_t>(scaled(s.time)));
        h = fnv1a_update(h, static_cast<uint64_t>(s.tower_count));
        h = fnv1a_update(h, static_cast<uint64_t>(s.enemy_count));
        h = fnv1a_update(h, static_cast<uint64_t>(s.to_spawn_count));
        h = fnv1a_update(h, static_cast<uint64_t>(s.legal_action_count));
        h = fnv1a_update(h, static_cast<uint64_t>(s.placeable_count));
    }
    return h;
}

uint64_t hash_golden_trace_cases(const std::vector<GoldenTraceCase>& cases) {
    uint64_t h = 1469598103934665603ULL;
    for (const auto& tc : cases) {
        auto steps = run_golden_trace_case(tc);
        h = fnv1a_update(h, hash_golden_trace(steps));
    }
    return h;
}

std::string golden_trace_step_to_jsonl(const GoldenTraceStep& s) {
    std::ostringstream os;
    os << "{"
       << "\"case\":\"" << json_escape(s.case_name) << "\"," 
       << "\"seed\":" << s.seed << ","
       << "\"step\":" << s.step << ","
       << "\"action\":" << s.action << ","
       << "\"reward\":" << std::fixed << std::setprecision(3) << s.reward << ","
       << "\"done\":" << (s.done ? "true" : "false") << ","
       << "\"money\":" << s.money << ","
       << "\"base_hp\":" << s.base_hp << ","
       << "\"wave\":" << s.wave << ","
       << "\"time\":" << std::fixed << std::setprecision(3) << s.time << ","
       << "\"tower_count\":" << s.tower_count << ","
       << "\"enemy_count\":" << s.enemy_count << ","
       << "\"to_spawn_count\":" << s.to_spawn_count << ","
       << "\"legal_action_count\":" << s.legal_action_count << ","
       << "\"placeable_count\":" << s.placeable_count << ","
       << "\"pathfind_calls\":" << s.perf.pathfind_calls << ","
       << "\"placeable_recompute\":" << s.perf.placeable_recompute << ","
       << "\"legal_recompute\":" << s.perf.legal_recompute << ","
       << "\"base_distance_recompute\":" << s.perf.base_distance_recompute
       << "}";
    return os.str();
}

std::string golden_trace_summary_to_json(const GoldenTraceCase& tc, const std::vector<GoldenTraceStep>& steps) {
    std::ostringstream os;
    os << "{"
       << "\"case\":\"" << json_escape(tc.name) << "\"," 
       << "\"seed\":" << tc.seed << ","
       << "\"steps\":" << steps.size() << ","
       << "\"hash\":\"0x" << std::hex << hash_golden_trace(steps) << std::dec << "\"";
    if (!steps.empty()) {
        const auto& last = steps.back();
        os << ",\"final_money\":" << last.money
           << ",\"final_base_hp\":" << last.base_hp
           << ",\"final_wave\":" << last.wave
           << ",\"final_time\":" << std::fixed << std::setprecision(3) << last.time
           << ",\"final_tower_count\":" << last.tower_count
           << ",\"final_enemy_count\":" << last.enemy_count;
    }
    os << "}";
    return os.str();
}

} // namespace tdmz
