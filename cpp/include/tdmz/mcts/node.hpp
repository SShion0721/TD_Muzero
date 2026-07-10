#pragma once
#include <vector>

namespace tdmz {

struct Node {
    int parent = -1;
    int action_from_parent = -1;

    int visit_count = 0;
    float prior = 0.0f;
    float value_sum = 0.0f;
    float reward = 0.0f;

    int latent_index = -1;
    int batch_index = -1;

    std::vector<int> actions;
    std::vector<int> children;

    void reset_for_reuse() {
        parent = -1;
        action_from_parent = -1;
        visit_count = 0;
        prior = 0.0f;
        value_sum = 0.0f;
        reward = 0.0f;
        latent_index = -1;
        batch_index = -1;
        actions.clear();
        children.clear();
    }

    bool expanded() const {
        return !actions.empty();
    }

    float value() const {
        if (visit_count == 0) return 0.0f;
        return value_sum / static_cast<float>(visit_count);
    }

    int child_index_for_action(int action) const {
        for (int i = 0; i < static_cast<int>(actions.size()); ++i) {
            if (actions[i] == action) return i;
        }
        return -1;
    }
};

} // namespace tdmz
