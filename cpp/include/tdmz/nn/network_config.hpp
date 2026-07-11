#pragma once
#include "tdmz/core/action.hpp"
#include "tdmz/core/observation.hpp"

namespace tdmz {

struct NetworkConfig {
    int observation_channels = OBS_CHANNELS;
    int board_h = kBoardH;
    int board_w = kBoardW;
    int action_space_size = kActionSpaceSize;

    int latent_channels = 32;
    int hidden_channels = 64;
    int action_planes = kSpatialActionPlanes;
    int policy_planes = kSpatialPolicyPlanes;

    int value_dim = 1;
    int reward_dim = 1;
};

} // namespace tdmz
