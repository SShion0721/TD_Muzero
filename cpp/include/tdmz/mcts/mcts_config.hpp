#pragma once

namespace tdmz {

struct MCTSConfig {
    int num_simulations = 32;

    float discount = 0.99f;
    float pb_c_base = 19652.0f;
    float pb_c_init = 1.25f;

    float root_dirichlet_alpha = 0.3f;
    float root_noise_weight = 0.25f;
    bool add_root_noise = false;

    int latent_top_k = 32;
    int max_nodes = 4096;

    float value_delta_max = 0.01f;

    bool single_player = true;
};

} // namespace tdmz
