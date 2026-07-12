#pragma once

#include "tdmz/core/action.hpp"
#include "tdmz/core/observation.hpp"

#include <cstdint>
#include <string>

namespace tdmz {

inline constexpr const char* kReplayIndexFormat = "tdmz_selfplay_shard_index";
inline constexpr uint32_t kReplayIndexVersion = 3;
inline constexpr uint32_t kReplayBinaryFormatVersion = 2;
inline constexpr uint32_t kLegacyReplayBinaryFormatVersion = 1;

// Rule v5 includes the v4 absorbing terminal semantics and makes wave mode
// immutable after the first environment mutation or time step.
inline constexpr uint32_t kEnvironmentRuleVersion = 5;

// Observation v2 adds Markov-critical pending-wave data, absolute enemy state,
// continuous-position splatting, route direction/distance, and bounded scaling.
inline constexpr uint32_t kObservationSchemaVersion = 2;
inline constexpr uint32_t kActionSpaceVersion = 1;
inline constexpr uint32_t kRewardTransformVersion = 1;

// Architecture v3 uses seven local action planes in Dynamics and six spatial
// policy/legality planes plus one wait scalar in Prediction.
inline constexpr uint32_t kNetworkArchitectureVersion = 3;
inline constexpr uint32_t kCheckpointManifestVersion = 2;
inline constexpr int kObservationSize = OBS_CHANNELS * kBoardH * kBoardW;

struct CompatibilityMetadata {
    uint32_t replay_format_version = kReplayBinaryFormatVersion;
    uint32_t environment_rule_version = kEnvironmentRuleVersion;
    uint32_t observation_schema_version = kObservationSchemaVersion;
    uint32_t action_space_version = kActionSpaceVersion;
    uint32_t reward_transform_version = kRewardTransformVersion;
    uint32_t network_architecture_version = kNetworkArchitectureVersion;

    int board_width = kBoardW;
    int board_height = kBoardH;
    int observation_channels = OBS_CHANNELS;
    int observation_size = kObservationSize;
    int action_space_size = kActionSpaceSize;
    int policy_size = kActionSpaceSize;
    int legal_mask_size = kActionSpaceSize;
};

constexpr CompatibilityMetadata current_compatibility_metadata() {
    return CompatibilityMetadata{};
}

void validate_compatibility_metadata(
    const CompatibilityMetadata& actual,
    const CompatibilityMetadata& expected = current_compatibility_metadata(),
    const std::string& context = "Compatibility metadata"
);

} // namespace tdmz
