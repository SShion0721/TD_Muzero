#pragma once

#include "tdmz/core/compatibility.hpp"
#include "tdmz/nn/network_config.hpp"

#include <cstdint>
#include <string>

namespace tdmz {

inline constexpr const char* kCheckpointManifestFormat = "tdmz_checkpoint_manifest";

struct CheckpointManifest {
    uint32_t manifest_version = kCheckpointManifestVersion;
    CompatibilityMetadata compatibility = current_compatibility_metadata();

    uint64_t training_step = 0;
    uint64_t seed = 0;
    bool optimizer_state_present = false;

    int latent_channels = 0;
    int hidden_channels = 0;
    int action_planes = 0;
    int policy_planes = 0;
    int value_dim = 0;
    int reward_dim = 0;
};

CheckpointManifest make_checkpoint_manifest(
    const NetworkConfig& config,
    uint64_t training_step,
    uint64_t seed,
    bool optimizer_state_present
);

void write_checkpoint_manifest_json(
    const CheckpointManifest& manifest,
    const std::string& path
);

CheckpointManifest read_checkpoint_manifest_json(const std::string& path);

void validate_checkpoint_manifest(
    const CheckpointManifest& manifest,
    const NetworkConfig& expected_config,
    bool require_optimizer_state = false
);

} // namespace tdmz
