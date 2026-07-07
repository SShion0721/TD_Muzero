#pragma once
#include <vector>

namespace tdmz {

struct SearchResults {
    int batch_size = 0;

    std::vector<int> leaf_node_ids;
    std::vector<std::vector<int>> search_paths;

    std::vector<int> latent_indices;
    std::vector<int> latent_batch_indices;
    std::vector<int> last_actions;
    std::vector<int> search_lens;

    explicit SearchResults(int n) : batch_size(n) {
        leaf_node_ids.resize(n);
        search_paths.resize(n);
        latent_indices.resize(n);
        latent_batch_indices.resize(n);
        last_actions.resize(n);
        search_lens.resize(n);
    }
};

} // namespace tdmz
