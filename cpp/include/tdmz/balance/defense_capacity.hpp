#pragma once
#include "tdmz/core/board_tables.hpp"
#include "tdmz/core/engine.hpp"
#include "tdmz/core/tower.hpp"
#include <array>
#include <vector>

namespace tdmz {

struct DefenseCapacityConfig {
    float wave_window_seconds = 15.0f;
    float safety_factor = 0.90f;
    float aoe_multiplier = 1.50f;
    float slow_multiplier = 1.10f;
    int virtual_base_hp = 100;
    bool include_spendable_money = true;

    bool path_aware = true;
    float full_coverage_fraction = 0.25f;
    int min_full_coverage_cells = 3;
    bool include_upgrade_candidates = true;
};

struct PathCoverageInfo {
    Bitboard128 path_bb{};
    int path_len = 0;
    std::array<int, kCells> path_order{};
};

struct DefenseCandidate {
    enum class Kind { Build, Upgrade };
    Kind kind = Kind::Build;
    TowerType type = TowerType::Basic;
    int cell = -1;
    int tower_index = -1;
    int cost = 0;
    float cap_gain = 0.0f;
    float value_per_cost = 0.0f;
};

struct DefenseCapacityResult {
    float current_tower_damage_cap = 0.0f;
    float spendable_money_damage_cap = 0.0f;
    float leak_hp_cap = 0.0f;
    float raw_hp_cap = 0.0f;
    float allowed_attack_hp = 0.0f;

    float static_current_tower_damage_cap = 0.0f;
    float path_coverage_factor_sum = 0.0f;
    float selected_build_cap = 0.0f;
    float selected_upgrade_cap = 0.0f;
    float best_build_value_per_cost = 0.0f;
    float best_upgrade_value_per_cost = 0.0f;

    int path_len = 0;
    int placeable_count = 0;
    int build_candidate_count = 0;
    int upgrade_candidate_count = 0;
    int selected_new_tower_count = 0;
    int selected_upgrade_count = 0;
    TowerType selected_new_tower_type = TowerType::Basic;
};

PathCoverageInfo make_main_path_coverage_info(const TDEngine& env);
float estimate_path_coverage_factor(Bitboard128 covered_path, int path_len, const DefenseCapacityConfig& cfg);

float estimate_tower_damage_capacity(TowerType type, int level, float window_seconds, float aoe_multiplier, float slow_multiplier);
float estimate_existing_tower_damage_capacity(const Tower& tower, float window_seconds, float aoe_multiplier, float slow_multiplier);
float estimate_tower_path_damage_capacity(TowerType type, int level, int cell, const PathCoverageInfo& path, const DefenseCapacityConfig& cfg);
float estimate_existing_tower_path_damage_capacity(const Tower& tower, const PathCoverageInfo& path, const DefenseCapacityConfig& cfg);
std::vector<DefenseCandidate> enumerate_defense_candidates(const TDEngine& env, const DefenseCapacityConfig& cfg, const PathCoverageInfo& path, float current_tower_cap);
DefenseCapacityResult estimate_defense_capacity(const TDEngine& env, const DefenseCapacityConfig& cfg = DefenseCapacityConfig{});

} // namespace tdmz
