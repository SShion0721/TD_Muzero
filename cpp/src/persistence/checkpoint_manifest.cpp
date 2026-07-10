#include "tdmz/persistence/checkpoint_manifest.hpp"

#include <fstream>
#include <iterator>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace tdmz {

namespace {

std::string read_text_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open checkpoint manifest: " + path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string require_string_field(const std::string& text, const char* field) {
    const std::regex pattern(
        std::string("\\\"") + field + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error(std::string("Checkpoint manifest missing field '") + field + "'");
    }
    return match[1].str();
}

uint64_t require_u64_field(const std::string& text, const char* field) {
    const std::regex pattern(
        std::string("\\\"") + field + "\\\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error(std::string("Checkpoint manifest missing field '") + field + "'");
    }
    try {
        return static_cast<uint64_t>(std::stoull(match[1].str()));
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Checkpoint manifest invalid integer field '") + field + "'");
    }
}

uint32_t require_u32_field(const std::string& text, const char* field) {
    const uint64_t value = require_u64_field(text, field);
    if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string("Checkpoint manifest field '") + field + "' exceeds uint32 range");
    }
    return static_cast<uint32_t>(value);
}

int require_int_field(const std::string& text, const char* field) {
    const std::regex pattern(
        std::string("\\\"") + field + "\\\"\\s*:\\s*(-?[0-9]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error(std::string("Checkpoint manifest missing field '") + field + "'");
    }
    try {
        const long long value = std::stoll(match[1].str());
        if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
            throw std::out_of_range("int range");
        }
        return static_cast<int>(value);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Checkpoint manifest invalid integer field '") + field + "'");
    }
}

bool require_bool_field(const std::string& text, const char* field) {
    const std::regex pattern(
        std::string("\\\"") + field + "\\\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error(std::string("Checkpoint manifest missing field '") + field + "'");
    }
    return match[1].str() == "true";
}

template <typename T>
void require_equal(const char* field, const T& actual, const T& expected) {
    if (actual == expected) return;
    std::ostringstream out;
    out << "Checkpoint manifest incompatible field '" << field
        << "': actual=" << actual << " expected=" << expected;
    throw std::runtime_error(out.str());
}

CompatibilityMetadata expected_metadata_for_config(const NetworkConfig& config) {
    CompatibilityMetadata metadata = current_compatibility_metadata();
    metadata.board_width = config.board_w;
    metadata.board_height = config.board_h;
    metadata.observation_channels = config.observation_channels;
    metadata.observation_size = config.observation_channels * config.board_h * config.board_w;
    metadata.action_space_size = config.action_space_size;
    metadata.policy_size = config.action_space_size;
    metadata.legal_mask_size = config.action_space_size;
    return metadata;
}

} // namespace

CheckpointManifest make_checkpoint_manifest(
    const NetworkConfig& config,
    uint64_t training_step,
    uint64_t seed,
    bool optimizer_state_present
) {
    CheckpointManifest manifest;
    manifest.compatibility = expected_metadata_for_config(config);
    manifest.training_step = training_step;
    manifest.seed = seed;
    manifest.optimizer_state_present = optimizer_state_present;
    manifest.latent_channels = config.latent_channels;
    manifest.hidden_channels = config.hidden_channels;
    manifest.action_embedding_dim = config.action_embedding_dim;
    manifest.value_dim = config.value_dim;
    manifest.reward_dim = config.reward_dim;
    return manifest;
}

void write_checkpoint_manifest_json(
    const CheckpointManifest& manifest,
    const std::string& path
) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to create checkpoint manifest: " + path);

    const CompatibilityMetadata& c = manifest.compatibility;
    out << "{\n";
    out << "  \"format\": \"" << kCheckpointManifestFormat << "\",\n";
    out << "  \"manifest_version\": " << manifest.manifest_version << ",\n";
    out << "  \"replay_format_version\": " << c.replay_format_version << ",\n";
    out << "  \"environment_rule_version\": " << c.environment_rule_version << ",\n";
    out << "  \"observation_schema_version\": " << c.observation_schema_version << ",\n";
    out << "  \"action_space_version\": " << c.action_space_version << ",\n";
    out << "  \"reward_transform_version\": " << c.reward_transform_version << ",\n";
    out << "  \"network_architecture_version\": " << c.network_architecture_version << ",\n";
    out << "  \"board_width\": " << c.board_width << ",\n";
    out << "  \"board_height\": " << c.board_height << ",\n";
    out << "  \"observation_channels\": " << c.observation_channels << ",\n";
    out << "  \"observation_size\": " << c.observation_size << ",\n";
    out << "  \"action_space_size\": " << c.action_space_size << ",\n";
    out << "  \"policy_size\": " << c.policy_size << ",\n";
    out << "  \"legal_mask_size\": " << c.legal_mask_size << ",\n";
    out << "  \"training_step\": " << manifest.training_step << ",\n";
    out << "  \"seed\": " << manifest.seed << ",\n";
    out << "  \"optimizer_state_present\": "
        << (manifest.optimizer_state_present ? "true" : "false") << ",\n";
    out << "  \"latent_channels\": " << manifest.latent_channels << ",\n";
    out << "  \"hidden_channels\": " << manifest.hidden_channels << ",\n";
    out << "  \"action_embedding_dim\": " << manifest.action_embedding_dim << ",\n";
    out << "  \"value_dim\": " << manifest.value_dim << ",\n";
    out << "  \"reward_dim\": " << manifest.reward_dim << "\n";
    out << "}\n";
    if (!out) throw std::runtime_error("Failed while writing checkpoint manifest: " + path);
}

CheckpointManifest read_checkpoint_manifest_json(const std::string& path) {
    const std::string text = read_text_file(path);
    const std::string format = require_string_field(text, "format");
    if (format != kCheckpointManifestFormat) {
        throw std::runtime_error(
            "Checkpoint manifest incompatible field 'format': actual=" + format +
            " expected=" + kCheckpointManifestFormat);
    }

    CheckpointManifest manifest;
    manifest.manifest_version = require_u32_field(text, "manifest_version");
    manifest.compatibility.replay_format_version = require_u32_field(text, "replay_format_version");
    manifest.compatibility.environment_rule_version = require_u32_field(text, "environment_rule_version");
    manifest.compatibility.observation_schema_version = require_u32_field(text, "observation_schema_version");
    manifest.compatibility.action_space_version = require_u32_field(text, "action_space_version");
    manifest.compatibility.reward_transform_version = require_u32_field(text, "reward_transform_version");
    manifest.compatibility.network_architecture_version = require_u32_field(text, "network_architecture_version");
    manifest.compatibility.board_width = require_int_field(text, "board_width");
    manifest.compatibility.board_height = require_int_field(text, "board_height");
    manifest.compatibility.observation_channels = require_int_field(text, "observation_channels");
    manifest.compatibility.observation_size = require_int_field(text, "observation_size");
    manifest.compatibility.action_space_size = require_int_field(text, "action_space_size");
    manifest.compatibility.policy_size = require_int_field(text, "policy_size");
    manifest.compatibility.legal_mask_size = require_int_field(text, "legal_mask_size");
    manifest.training_step = require_u64_field(text, "training_step");
    manifest.seed = require_u64_field(text, "seed");
    manifest.optimizer_state_present = require_bool_field(text, "optimizer_state_present");
    manifest.latent_channels = require_int_field(text, "latent_channels");
    manifest.hidden_channels = require_int_field(text, "hidden_channels");
    manifest.action_embedding_dim = require_int_field(text, "action_embedding_dim");
    manifest.value_dim = require_int_field(text, "value_dim");
    manifest.reward_dim = require_int_field(text, "reward_dim");
    return manifest;
}

void validate_checkpoint_manifest(
    const CheckpointManifest& manifest,
    const NetworkConfig& expected_config,
    bool require_optimizer_state
) {
    require_equal("manifest_version", manifest.manifest_version, kCheckpointManifestVersion);
    validate_compatibility_metadata(
        manifest.compatibility,
        expected_metadata_for_config(expected_config),
        "Checkpoint manifest");
    require_equal("latent_channels", manifest.latent_channels, expected_config.latent_channels);
    require_equal("hidden_channels", manifest.hidden_channels, expected_config.hidden_channels);
    require_equal("action_embedding_dim", manifest.action_embedding_dim, expected_config.action_embedding_dim);
    require_equal("value_dim", manifest.value_dim, expected_config.value_dim);
    require_equal("reward_dim", manifest.reward_dim, expected_config.reward_dim);
    if (require_optimizer_state && !manifest.optimizer_state_present) {
        throw std::runtime_error(
            "Checkpoint manifest incompatible field 'optimizer_state_present': actual=0 expected=1");
    }
}

} // namespace tdmz
