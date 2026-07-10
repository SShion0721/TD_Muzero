#pragma once
#include <vector>

namespace tdmz {

struct SearchDebugStats {
    int total_nodes = 0;
    int max_root_branching = 0;
    int max_latent_branching = 0;
    int max_search_depth = 0;

    int node_objects_created = 0;
    int node_objects_reused = 0;
    int node_buffer_growth_events = 0;
    int scratch_capacity_growth_events = 0;
};

struct RootSearchOutput {
    int action = -1;
    float root_value = 0.0f;

    std::vector<int> root_actions;
    std::vector<float> root_priors;
    std::vector<int> visit_counts;
    std::vector<float> policy_full;   // size = kActionSpaceSize

    SearchDebugStats debug;
};

struct BatchSearchOutput {
    std::vector<RootSearchOutput> roots;
};

} // namespace tdmz
