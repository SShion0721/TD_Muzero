#include "tdmz/core/compatibility.hpp"

#include <sstream>
#include <stdexcept>

namespace tdmz {

namespace {

template <typename T>
void require_equal(
    const char* field,
    const T& actual,
    const T& expected,
    const std::string& context
) {
    if (actual == expected) return;

    std::ostringstream out;
    out << context << " incompatible field '" << field
        << "': actual=" << actual << " expected=" << expected;
    throw std::runtime_error(out.str());
}

} // namespace

void validate_compatibility_metadata(
    const CompatibilityMetadata& actual,
    const CompatibilityMetadata& expected,
    const std::string& context
) {
    require_equal("replay_format_version", actual.replay_format_version, expected.replay_format_version, context);
    require_equal("environment_rule_version", actual.environment_rule_version, expected.environment_rule_version, context);
    require_equal("observation_schema_version", actual.observation_schema_version, expected.observation_schema_version, context);
    require_equal("action_space_version", actual.action_space_version, expected.action_space_version, context);
    require_equal("reward_transform_version", actual.reward_transform_version, expected.reward_transform_version, context);
    require_equal("network_architecture_version", actual.network_architecture_version, expected.network_architecture_version, context);
    require_equal("board_width", actual.board_width, expected.board_width, context);
    require_equal("board_height", actual.board_height, expected.board_height, context);
    require_equal("observation_channels", actual.observation_channels, expected.observation_channels, context);
    require_equal("observation_size", actual.observation_size, expected.observation_size, context);
    require_equal("action_space_size", actual.action_space_size, expected.action_space_size, context);
    require_equal("policy_size", actual.policy_size, expected.policy_size, context);
    require_equal("legal_mask_size", actual.legal_mask_size, expected.legal_mask_size, context);
}

} // namespace tdmz
