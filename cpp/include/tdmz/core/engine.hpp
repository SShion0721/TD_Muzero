#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include "tdmz/core/action.hpp"
#include "tdmz/core/tower.hpp"
#include "tdmz/core/enemy.hpp"
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
};

class TDEngine {
public:
    TDEngine(int width = 11, int height = 11, uint64_t seed = 0);

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

    const std::vector<Tower>& towers() const { return towers_; }
    const std::vector<Enemy>& enemies() const { return enemies_; }

    int money() const { return money_; }
    int base_hp() const { return base_hp_; }
    int wave() const { return wave_; }
    float spawn_timer() const { return spawn_timer_; }
    int enemies_to_spawn_count() const { return static_cast<int>(enemies_to_spawn_.size()); }
    bool game_over() const { return game_over_; }
    float time() const { return time_; }

    int spawn_x() const { return spawn_x_; }
    int spawn_y() const { return spawn_y_; }
    int base_x() const { return base_x_; }
    int base_y() const { return base_y_; }
    int width() const { return width_; }
    int height() const { return height_; }

    bool in_bounds(int x, int y) const;

    const std::array<std::array<int, kBoardW>, kBoardH>& grid() const { return grid_; }

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

    std::array<std::array<int, kBoardW>, kBoardH> grid_;
    std::vector<Tower> towers_;
    std::vector<Enemy> enemies_;
    std::vector<EnemySpec> enemies_to_spawn_;

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
    mutable std::vector<int> cached_legal_actions_;

    void invalidate_all_caches();
    void mark_grid_changed();
    void mark_towers_changed();
    void mark_money_changed();
    void mark_enemies_changed();
    void mark_wave_changed();

    std::vector<EnemySpec> get_wave_enemies();
};

} // namespace tdmz
