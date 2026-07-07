#include "tdmz/balance/defense_capacity.hpp"
#include <algorithm>
#include <cmath>

namespace tdmz {

namespace {

float sanitize_nonnegative(float v) {
    if (!std::isfinite(v) || v < 0.0f) return 0.0f;
    return v;
}

int count_placeable_cells(const std::array<std::array<bool, kBoardW>, kBoardH>& mask) {
    int count = 0;
    for (const auto& row : mask) {
        for (bool v : row) {
            if (v) ++count;
        }
    }
    return count;
}

TowerStats stats_after_level(TowerType type, int level) {
    TowerStats s = tower_stats(type);
    int target_level = std::max(1, std::min(level, kTowerMaxLevel));
    for (int lv = 1; lv < target_level; ++lv) {
        s.damage *= 1.5f;
        s.range *= 1.1f;
        s.cooldown = std::max(0.1f, s.cooldown * 0.9f);
    }
    return s;
}

float estimate_stats_damage_capacity(const TowerStats& s, TowerType type, float window_seconds, float aoe_multiplier, float slow_multiplier) {
    float window = sanitize_nonnegative(window_seconds);
    float cooldown = std::max(0.1f, s.cooldown);
    int shots = static_cast<int>(std::floor(window / cooldown));
    if (window > 0.0f && shots < 1) shots = 1;

    float cap = static_cast<float>(shots) * std::max(0.0f, s.damage);
    if (type == TowerType::AOE) {
        cap *= std::max(1.0f, aoe_multiplier);
    }
    if (type == TowerType::Slow) {
        cap *= std::max(1.0f, slow_multiplier);
    }
    return cap;
}

float clamp01(float v) {
    return std::max(0.0f, std::min(1.0f, v));
}

Bitboard128 range_mask_for(TowerType type, int level, int cell) {
    if (!valid_cell(cell)) return Bitboard128::zero();
    int t = static_cast<int>(type);
    int lv = std::max(1, std::min(level, kTowerMaxLevel));
    return board_tables().range_mask[t][lv][cell];
}

} // namespace

PathCoverageInfo make_main_path_coverage_info(const TDEngine& env) {
    PathCoverageInfo info;
    info.path_bb = Bitboard128::zero();
    info.path_len = 0;
    info.path_order.fill(-1);

    auto path = env.find_path(env.spawn_x(), env.spawn_y(), env.base_x(), env.base_y());
    for (const auto& p : path) {
        int x = p.first;
        int y = p.second;
        if (!valid_cell_xy(x, y)) continue;
        int cell = cell_id(x, y);
        if (info.path_order[cell] >= 0) continue;
        info.path_order[cell] = info.path_len++;
        info.path_bb.set(cell);
    }
    return info;
}

float estimate_path_coverage_factor(Bitboard128 covered_path, int path_len, const DefenseCapacityConfig& cfg) {
    if (path_len <= 0) return 0.0f;
    int covered = covered_path.popcount();
    if (covered <= 0) return 0.0f;

    float fraction = clamp01(cfg.full_coverage_fraction);
    int full_cells = static_cast<int>(std::ceil(static_cast<float>(path_len) * fraction));
    full_cells = std::max(1, std::max(cfg.min_full_coverage_cells, full_cells));
    full_cells = std::min(path_len, full_cells);
    return clamp01(static_cast<float>(covered) / static_cast<float>(full_cells));
}

float estimate_tower_damage_capacity(TowerType type, int level, float window_seconds, float aoe_multiplier, float slow_multiplier) {
    TowerStats s = stats_after_level(type, level);
    return estimate_stats_damage_capacity(s, type, window_seconds, aoe_multiplier, slow_multiplier);
}

float estimate_existing_tower_damage_capacity(const Tower& tower, float window_seconds, float aoe_multiplier, float slow_multiplier) {
    float window = sanitize_nonnegative(window_seconds);
    float cooldown = std::max(0.1f, tower.cooldown_max);
    float first_shot_delay = std::max(0.0f, tower.cooldown);
    if (window <= 0.0f || first_shot_delay > window) {
        return 0.0f;
    }

    int shots = 1 + static_cast<int>(std::floor((window - first_shot_delay) / cooldown));
    float cap = static_cast<float>(shots) * std::max(0.0f, tower.damage);
    if (tower.type == TowerType::AOE) {
        cap *= std::max(1.0f, aoe_multiplier);
    }
    if (tower.type == TowerType::Slow) {
        cap *= std::max(1.0f, slow_multiplier);
    }
    return cap;
}

float estimate_tower_path_damage_capacity(TowerType type, int level, int cell, const PathCoverageInfo& path, const DefenseCapacityConfig& cfg) {
    float full = estimate_tower_damage_capacity(type, level, cfg.wave_window_seconds, cfg.aoe_multiplier, cfg.slow_multiplier);
    if (!cfg.path_aware) return full;
    Bitboard128 covered = range_mask_for(type, level, cell) & path.path_bb;
    float factor = estimate_path_coverage_factor(covered, path.path_len, cfg);
    return full * factor;
}

float estimate_existing_tower_path_damage_capacity(const Tower& tower, const PathCoverageInfo& path, const DefenseCapacityConfig& cfg) {
    float full = estimate_existing_tower_damage_capacity(tower, cfg.wave_window_seconds, cfg.aoe_multiplier, cfg.slow_multiplier);
    if (!cfg.path_aware) return full;
    int cell = cell_id(tower.x, tower.y);
    Bitboard128 covered = range_mask_for(tower.type, tower.level, cell) & path.path_bb;
    float factor = estimate_path_coverage_factor(covered, path.path_len, cfg);
    return full * factor;
}

std::vector<DefenseCandidate> enumerate_defense_candidates(const TDEngine& env, const DefenseCapacityConfig& cfg, const PathCoverageInfo& path, float /*current_tower_cap*/) {
    std::vector<DefenseCandidate> candidates;
    auto placeable = env.compute_placeable_mask();
    const auto& tables = board_tables();

    for (int c = 0; c < kCells; ++c) {
        int x = tables.x[c];
        int y = tables.y[c];
        if (!placeable[y][x]) continue;
        for (int t = 0; t < kBuildTypes; ++t) {
            TowerType type = static_cast<TowerType>(t);
            TowerStats s = tower_stats(type);
            if (s.cost <= 0) continue;
            float cap = estimate_tower_path_damage_capacity(type, 1, c, path, cfg);
            if (cap <= 0.0f) continue;
            DefenseCandidate cand;
            cand.kind = DefenseCandidate::Kind::Build;
            cand.type = type;
            cand.cell = c;
            cand.cost = s.cost;
            cand.cap_gain = cap;
            cand.value_per_cost = cap / static_cast<float>(s.cost);
            candidates.push_back(cand);
        }
    }

    if (cfg.include_upgrade_candidates) {
        const auto& towers = env.towers();
        for (int i = 0; i < static_cast<int>(towers.size()); ++i) {
            const auto& tower = towers[i];
            if (!tower.can_upgrade() || tower.upgrade_cost <= 0) continue;
            Tower upgraded = tower;
            upgraded.upgrade();
            float before = estimate_existing_tower_path_damage_capacity(tower, path, cfg);
            float after = estimate_existing_tower_path_damage_capacity(upgraded, path, cfg);
            float gain = after - before;
            if (gain <= 0.0f) continue;
            DefenseCandidate cand;
            cand.kind = DefenseCandidate::Kind::Upgrade;
            cand.type = tower.type;
            cand.cell = cell_id(tower.x, tower.y);
            cand.tower_index = i;
            cand.cost = tower.upgrade_cost;
            cand.cap_gain = gain;
            cand.value_per_cost = gain / static_cast<float>(tower.upgrade_cost);
            candidates.push_back(cand);
        }
    }

    return candidates;
}

DefenseCapacityResult estimate_defense_capacity(const TDEngine& env, const DefenseCapacityConfig& cfg) {
    DefenseCapacityResult result;

    const float safety = std::max(0.0f, cfg.safety_factor);
    const float aoe_mult = std::max(1.0f, cfg.aoe_multiplier);
    const float slow_mult = std::max(1.0f, cfg.slow_multiplier);
    PathCoverageInfo path = make_main_path_coverage_info(env);
    result.path_len = path.path_len;

    for (const auto& tower : env.towers()) {
        float static_cap = estimate_existing_tower_damage_capacity(tower, cfg.wave_window_seconds, aoe_mult, slow_mult);
        float path_cap = estimate_existing_tower_path_damage_capacity(tower, path, cfg);
        result.static_current_tower_damage_cap += static_cap;
        result.current_tower_damage_cap += path_cap;
        if (static_cap > 0.0f) {
            result.path_coverage_factor_sum += path_cap / static_cap;
        }
    }

    auto placeable = env.compute_placeable_mask();
    result.placeable_count = count_placeable_cells(placeable);

    if (cfg.include_spendable_money && result.placeable_count > 0 && env.money() > 0) {
        auto candidates = enumerate_defense_candidates(env, cfg, path, result.current_tower_damage_cap);
        for (const auto& c : candidates) {
            if (c.kind == DefenseCandidate::Kind::Build) {
                ++result.build_candidate_count;
                result.best_build_value_per_cost = std::max(result.best_build_value_per_cost, c.value_per_cost);
            } else {
                ++result.upgrade_candidate_count;
                result.best_upgrade_value_per_cost = std::max(result.best_upgrade_value_per_cost, c.value_per_cost);
            }
        }

        std::sort(candidates.begin(), candidates.end(), [](const DefenseCandidate& a, const DefenseCandidate& b) {
            if (a.value_per_cost != b.value_per_cost) return a.value_per_cost > b.value_per_cost;
            if (a.cap_gain != b.cap_gain) return a.cap_gain > b.cap_gain;
            return a.cost < b.cost;
        });

        int money_left = env.money();
        Bitboard128 used_build_cells = Bitboard128::zero();
        std::vector<uint8_t> used_upgrade_tower(env.towers().size(), 0);

        for (const auto& cand : candidates) {
            if (cand.cost <= 0 || cand.cost > money_left) continue;
            if (cand.kind == DefenseCandidate::Kind::Build) {
                if (!valid_cell(cand.cell) || used_build_cells.test(cand.cell)) continue;
                used_build_cells.set(cand.cell);
                money_left -= cand.cost;
                result.spendable_money_damage_cap += cand.cap_gain;
                result.selected_build_cap += cand.cap_gain;
                if (result.selected_new_tower_count == 0) {
                    result.selected_new_tower_type = cand.type;
                }
                ++result.selected_new_tower_count;
            } else {
                if (cand.tower_index < 0 || cand.tower_index >= static_cast<int>(used_upgrade_tower.size())) continue;
                if (used_upgrade_tower[cand.tower_index]) continue;
                used_upgrade_tower[cand.tower_index] = 1;
                money_left -= cand.cost;
                result.spendable_money_damage_cap += cand.cap_gain;
                result.selected_upgrade_cap += cand.cap_gain;
                ++result.selected_upgrade_count;
            }
        }
    }

    int virtual_hp = std::max(0, cfg.virtual_base_hp);
    result.leak_hp_cap = static_cast<float>(std::max(0, virtual_hp - 1) * 10);

    result.raw_hp_cap = result.current_tower_damage_cap + result.spendable_money_damage_cap + result.leak_hp_cap;
    result.allowed_attack_hp = safety * result.raw_hp_cap;
    return result;
}

} // namespace tdmz
