#pragma once
#include <vector>

namespace tdmz {

struct RootSearchOutput {
    int action = -1;
    float root_value = 0.0f;

    std::vector<int> root_actions;
    std::vector<int> visit_counts;
    std::vector<float> policy_full;   // size = kActionSpaceSize
};

struct BatchSearchOutput {
    std::vector<RootSearchOutput> roots;
};

} // namespace tdmz
