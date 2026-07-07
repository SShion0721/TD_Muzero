#pragma once
#include "tdmz/core/action.hpp"
#include "tdmz/core/engine.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace tdmz {

struct GoldenTraceCase {
    std::string name;
    uint64_t seed = 0;
    std::vector<int> actions;
};

struct GoldenTraceStep {
    std::string case_name;
    uint64_t seed = 0;
    int step = 0;
    int action = -1;
    float reward = 0.0f;
    bool done = false;

    int money = 0;
    int base_hp = 0;
    int wave = 0;
    float time = 0.0f;
    int tower_count = 0;
    int enemy_count = 0;
    int to_spawn_count = 0;
    int legal_action_count = 0;
    int placeable_count = 0;

    EnginePerfCounters perf;
};

std::vector<GoldenTraceCase> default_golden_trace_cases();
std::vector<GoldenTraceStep> run_golden_trace_case(const GoldenTraceCase& tc);
uint64_t hash_golden_trace(const std::vector<GoldenTraceStep>& steps);
uint64_t hash_golden_trace_cases(const std::vector<GoldenTraceCase>& cases);
std::string golden_trace_step_to_jsonl(const GoldenTraceStep& step);
std::string golden_trace_summary_to_json(const GoldenTraceCase& tc, const std::vector<GoldenTraceStep>& steps);

} // namespace tdmz
