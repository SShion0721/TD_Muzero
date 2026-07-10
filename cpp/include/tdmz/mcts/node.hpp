#pragma once

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

    int first_edge = -1;
    int edge_count = 0;

    void reset_for_reuse() {
        parent = -1;
        action_from_parent = -1;
        visit_count = 0;
        prior = 0.0f;
        value_sum = 0.0f;
        reward = 0.0f;
        latent_index = -1;
        batch_index = -1;
        first_edge = -1;
        edge_count = 0;
    }

    bool expanded() const {
        return edge_count > 0;
    }

    float value() const {
        if (visit_count == 0) return 0.0f;
        return value_sum / static_cast<float>(visit_count);
    }
};

} // namespace tdmz
