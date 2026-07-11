#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include "tdmz/core/action.hpp"
#include "tdmz/core/board_tables.hpp"
#include "tdmz/core/tower.hpp"
#include "tdmz/core/enemy.hpp"
#include "tdmz/core/pending_spawn_queue.hpp"
#include "tdmz/core/rng.hpp"

namespace tdmz {

struct StepResult {
    float reward = 0.0f;
    bool done = false;
};

struct EnginePerfCounters {
    uint64_t pathfind_calls = 0;
    uint64_t placeable_recompute = 0;
    uint64_t legal_recompute = 0;
    uint64_t base_distance_recompute = 0;
    uint64_t enemy_occupancy_recompute = 0;
    uint64_t enemy_bucket_recompute = 0;
    uint64_t tower_candidate_cells = 0;
    uint64_t tower_exact_distance_checks = 0;
    uint64_t tower_aoe_distance_checks = 0;
};

class TDEngine {
public:
    TDEngine(int width = 11, int height = 11, uint64_t seed = 0);
    TDEngine(int width, int height, uint64_t seed, bool use_budgeted_waves);

    void reset(uint64_t seed);
    StepResult step_action(int flat_action);
    StepResult step_time(float dt);
    StepResult step_wait(int wait_steps);

    bool can_place_tower(int x, int y, TowerType type) const;
    bool place_tower(int x, int y, TowerType type);
    bool upgrade_tower(int x, int y);
    bool sell_tower(int x, int y);

    std::vector<int> legal_actions() const;
    std::vector<uint8_t> legal_action_mask() const;

    std::array<std::array<bool, kBoardW>, kBoardH> compute_placeable_mask() const;
    std::vector<std::pair<int,int>> find_path(int sx, int sy, int ex, int ey) const;
    int distance_to_base_cell(int cell) const;

    const std::vector<Tower>& towers() const { return towers_; }
    const std::vector<Enemy>& enemies() const { return enemies_; }
    const PendingSpawnQueue& pending_spawns() const { return enemies_to_spawn_; }

    int money() const { return money_; }
    int base_hp() const { return base_hp_; }
    int wave() const { return wave_; }
    float spawn_timer() const { return spawn_timer_; }
    int enemies_to_spawn_count() const { return static_cast<int>(enemies_to_spawn_.size()); }
    float pending_spawn_total_hp() const;
    bool game_over() const { return game_over_; }
    float time() const { return time_; }
    uint64_t seed() const { return seed_; }

    int spawn_x() const { return spawn_x_; }
    int spawn_y() const { return spawn_y_; }
    int base_x() const { return base_x_; }
    int base_y() const { return base_y_; }
    int width() const { return width_; }
    int height() const { return height_; }

    uint64_t grid_version() const { return grid_version_; }
    uint64_t tower_version() const { return tower_version_; }
    uint64_t money_version() const { return money_version_; }
    uint64_t enemy_version() const { return enemy_version_; }
    uint64_t wave_version() const { return wave_version_; }

    bool use_budgeted_waves() const { return use_budgeted_waves_; }
    void set_use_budgeted_waves(bool enabled, bool regenerate_pending_wave = false);

    bool in_bounds(int x, int y) const;

    const std::array<std::array<int, kBoardW>, kBoardH>& grid() const { return grid_; }
    Bitboard128 blocked_bitboard() const { return bb_blocked_; }

    const EnginePerfCounters& perf_counters() const { return perf_; }
    void reset_perf_counters() const { perf_ = EnginePerfCounters{}; }

private:
    int width_;
    int height_;
    int spawn_x_;
    int spawn_y_;
    int base_x_;
    int base_y_;

    int money_;
    int base_hp_;
    int wave_;
    float spawn_timer_;
    float time_;
    bool game_over_;
    bool use_budgeted_waves_ = false;

    std::array<std::array<int, kBoardW>, kBoardH> grid_;
    Bitboard128 bb_blocked_;
    std::vector<Tower> towers_;
    std::vector<Enemy> enemies_;
    PendingSpawnQueue enemies_to_spawn_;
    std::vector<int> enemy_bucket_next_scratch_;

    uint64_t seed_;
    PythonRNG rng_;
    int enemy_id_counter_;

    uint64_t grid_version_ = 0;
    uint64_t tower_version_ = 0;
    uint64_t money_version_ = 0;
    uint64_t enemy_version_ = 0;
    uint64_t wave_version_ = 0;

    mutable EnginePerfCounters perf_;

    mutable bool placeable_cache_valid_ = false;
    mutable uint64_t placeable_grid_version_ = 0;
    mutable std::array<std::array<bool, kBoardW>, kBoardH> cached_placeable_mask_{};

    mutable bool legal_cache_valid_ = false;
    mutable uint64_t legal_grid_version_ = 0;
    mutable uint64_t legal_tower_version_ = 0;
    mutable uint64_t legal_money_version_ = 0;
    mutable uint64_t legal_enemy_version_ = 0;
    mutable std::vector<int> cached_legal_actions_;

    mutable bool enemy_occupancy_cache_valid_ = false;
    mutable uint64_t enemy_occupancy_version_ = 0;
    mutable Bitboard128 cached_enemy_occupancy_{};

    mutable bool base_distance_cache_valid_ = false;
    mutable uint64_t base_distance_grid_version_ = 0;
    mutable std::array<int, kCells> cached_base_distance_{};
    mutable std::array<int, kCells> cached_next_to_base_{};

    void invalidate_all_caches();
    void mark_grid_changed();
    void mark_towers_changed();
    void mark_money_changed();
    void mark_enemies_changed();
    void mark_wave_changed();

    StepResult step_one_tick();

    Bitboard128 enemy_occupancy_bitboard() const;
    void recompute_base_distance_cache() const;
    std::vector<std::pair<int,int>> path_to_base_from_cell(int start_cell) const;

    std::vector<EnemySpec> get_fixed_wave_enemies();
    std::vector<EnemySpec> get_budgeted_wave_enemies();
    std::vector<EnemySpec> get_wave_enemies();
};

} // namespace tdmz
