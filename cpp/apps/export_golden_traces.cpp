#include "tdmz/core/golden_trace.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace tdmz;

int main(int argc, char** argv) {
    std::string out_path = "golden_trace_current.jsonl";
    if (argc >= 2) {
        out_path = argv[1];
    }

    std::ofstream out(out_path);
    if (!out) {
        throw std::runtime_error("failed to open golden trace output file");
    }

    auto cases = default_golden_trace_cases();
    uint64_t combined = hash_golden_trace_cases(cases);

    std::cout << "golden_trace_combined_hash=0x" << std::hex << combined << std::dec << std::endl;
    for (const auto& tc : cases) {
        auto steps = run_golden_trace_case(tc);
        uint64_t h = hash_golden_trace(steps);
        std::cout << tc.name << " hash=0x" << std::hex << h << std::dec
                  << " steps=" << steps.size() << std::endl;
        out << golden_trace_summary_to_json(tc, steps) << "\n";
        for (const auto& step : steps) {
            out << golden_trace_step_to_jsonl(step) << "\n";
        }
    }

    std::cout << "wrote " << out_path << std::endl;
    return 0;
}
