#include "tdmz/core/golden_trace.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

using namespace tdmz;

static void check_true(bool ok, const char* expr, int line) {
    if (!ok) throw std::runtime_error(std::string("check failed at line ") + std::to_string(line) + ": " + expr);
}
#define CHECK_TRUE(x) check_true(static_cast<bool>(x), #x, __LINE__)

int main() {
    auto cases = default_golden_trace_cases();
    CHECK_TRUE(cases.size() == 3);

    uint64_t combined_a = hash_golden_trace_cases(cases);
    uint64_t combined_b = hash_golden_trace_cases(cases);
    CHECK_TRUE(combined_a == combined_b);
    CHECK_TRUE(combined_a != 0);

    for (const auto& tc : cases) {
        auto a = run_golden_trace_case(tc);
        auto b = run_golden_trace_case(tc);
        CHECK_TRUE(!a.empty());
        CHECK_TRUE(a.size() == b.size());
        CHECK_TRUE(hash_golden_trace(a) == hash_golden_trace(b));
        CHECK_TRUE(a.front().step == 0);
        CHECK_TRUE(a.front().seed == tc.seed);
        CHECK_TRUE(a.back().time >= a.front().time);
        CHECK_TRUE(a.back().legal_action_count > 0);
        CHECK_TRUE(a.back().placeable_count >= 0);
    }

    std::cout << "Golden trace determinism tests passed. combined_hash=0x" << std::hex << combined_a << std::dec << std::endl;
    return 0;
}
