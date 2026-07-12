#include "tdmz/persistence/checkpoint_manifest.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace tdmz;

namespace {

void check(bool condition, const char* expr, int line) {
    if (!condition) {
        throw std::runtime_error(
            std::string("check failed at line ") + std::to_string(line) + ": " + expr);
    }
}

#define CHECK(expr) check(static_cast<bool>(expr), #expr, __LINE__)

template <typename Fn>
void check_throws_contains(Fn&& fn, const std::string& needle) {
    bool threw = false;
    try {
        fn();
    } catch (const std::runtime_error& e) {
        threw = true;
        CHECK(std::string(e.what()).find(needle) != std::string::npos);
    }
    CHECK(threw);
}

void test_current_compatibility_versions() {
    const CompatibilityMetadata metadata = current_compatibility_metadata();
    CHECK(metadata.environment_rule_version == 5);
    CHECK(metadata.observation_schema_version == 2);
    CHECK(metadata.network_architecture_version == 3);
    CHECK(kCheckpointManifestVersion == 2);
    CHECK(metadata.observation_channels == 40);
    CHECK(metadata.observation_size == 40 * kBoardH * kBoardW);
}

void test_manifest_roundtrip() {
    const std::string path = "test_checkpoint_manifest.json";
    NetworkConfig config;
    config.latent_channels = 48;
    config.hidden_channels = 96;
    config.action_planes = kSpatialActionPlanes;
    config.policy_planes = kSpatialPolicyPlanes;

    const CheckpointManifest original = make_checkpoint_manifest(config, 1234, 77, true);
    write_checkpoint_manifest_json(original, path);
    const CheckpointManifest loaded = read_checkpoint_manifest_json(path);

    CHECK(loaded.manifest_version == kCheckpointManifestVersion);
    CHECK(loaded.training_step == 1234);
    CHECK(loaded.seed == 77);
    CHECK(loaded.optimizer_state_present);
    CHECK(loaded.latent_channels == 48);
    CHECK(loaded.hidden_channels == 96);
    CHECK(loaded.action_planes == kSpatialActionPlanes);
    CHECK(loaded.policy_planes == kSpatialPolicyPlanes);
    CHECK(loaded.compatibility.environment_rule_version == 5);
    CHECK(loaded.compatibility.observation_schema_version == 2);
    CHECK(loaded.compatibility.observation_channels == OBS_CHANNELS);
    CHECK(loaded.compatibility.network_architecture_version == 3);
    validate_checkpoint_manifest(loaded, config, true);

    std::filesystem::remove(path);
}

void test_manifest_rejects_architecture_mismatch() {
    NetworkConfig config;
    CheckpointManifest manifest = make_checkpoint_manifest(config, 0, 0, false);
    ++manifest.compatibility.network_architecture_version;

    check_throws_contains(
        [&] { validate_checkpoint_manifest(manifest, config); },
        "network_architecture_version");
}

void test_manifest_rejects_spatial_dimension_mismatch() {
    NetworkConfig config;
    CheckpointManifest manifest = make_checkpoint_manifest(config, 0, 0, false);
    ++manifest.action_planes;

    check_throws_contains(
        [&] { validate_checkpoint_manifest(manifest, config); },
        "action_planes");
}

void test_manifest_rejects_observation_schema_mismatch() {
    NetworkConfig config;
    CheckpointManifest manifest = make_checkpoint_manifest(config, 0, 0, false);
    manifest.compatibility.observation_schema_version = 1;
    manifest.compatibility.observation_channels = 20;
    manifest.compatibility.observation_size = 20 * kBoardH * kBoardW;

    check_throws_contains(
        [&] { validate_checkpoint_manifest(manifest, config); },
        "observation_schema_version");
}

void test_manifest_requires_optimizer_for_resume() {
    NetworkConfig config;
    const CheckpointManifest manifest = make_checkpoint_manifest(config, 0, 0, false);

    validate_checkpoint_manifest(manifest, config, false);
    check_throws_contains(
        [&] { validate_checkpoint_manifest(manifest, config, true); },
        "optimizer_state_present");
}

void test_manifest_rejects_missing_fields() {
    const std::string path = "test_checkpoint_manifest_missing.json";
    {
        std::ofstream out(path);
        out << "{\"format\":\"tdmz_checkpoint_manifest\",\"manifest_version\":2}\n";
    }

    check_throws_contains(
        [&] { (void)read_checkpoint_manifest_json(path); },
        "replay_format_version");
    std::filesystem::remove(path);
}

} // namespace

int main() {
    try {
        test_current_compatibility_versions();
        test_manifest_roundtrip();
        test_manifest_rejects_architecture_mismatch();
        test_manifest_rejects_spatial_dimension_mismatch();
        test_manifest_rejects_observation_schema_mismatch();
        test_manifest_requires_optimizer_for_resume();
        test_manifest_rejects_missing_fields();
        std::cout << "Checkpoint manifest tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
