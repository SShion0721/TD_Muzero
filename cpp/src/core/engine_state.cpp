#include "tdmz/core/engine.hpp"
#include "tdmz/balance/attack_budget.hpp"
#include "tdmz/balance/budgeted_wave_generator.hpp"
#include <stdexcept>

namespace tdmz {

TDEngine::TDEngine(int width, int height, uint64_t seed)
    : TDEngine(width, height, seed, false) {}

TDEngine::TDEngine(int width, int height, uint64_t seed, bool use_budgeted_waves)
    : width_(width), height_(height),
      spawn_x_(0), spawn_y_(height / 2),
      base_x_(width - 1), base_y_(height / 2),
      use_budgeted_waves_(use_budgeted_waves),
      seed_(seed), rng_(seed) {
    if (width != kBoardW || height != kBoardH) {
        throw std::invalid_argument("TDEngine currently supports only 11x11 boards");
    }
    reset(seed);
}

void TDEngine::invalidate_all_caches() {
    placeable_cache_valid_ = false;
    legal_cache_valid_ = false;
    enemy_occupancy_cache_valid_ = false;
    base_distance_cache_valid_ = false;
}

void TDEngine::mark_grid_changed() {
    ++grid_version_;
    placeable_cache_valid_ = false;
    legal_cache_valid_ = false;
    base_distance_cache_valid_ = false;
}

void TDEngine::mark_towers_changed() {
    ++tower_version_;
    legal_cache_valid_ = false;
}

void TDEngine::mark_money_changed() {
    ++money_version_;
    legal_cache_valid_ = false;
}

void TDEngine::mark_enemies_changed() {
    ++enemy_version_;
    legal_cache_valid_ = false;
    enemy_occupancy_cache_valid_ = false;
}

void TDEngine::mark_wave_changed() {
    ++wave_version_;
}

void TDEngine::reset(uint64_t seed) {
    seed_ = seed;
    rng_.seed(seed_);

    money_ = 200;
    base_hp_ = 100;

    for (auto& row : grid_) {
        row.fill(0);
    }
    bb_blocked_ = Bitboard128::zero();

    towers_.clear();
    enemies_.clear();
    enemies_.reserve(128);
    enemies_to_spawn_.reserve(128);
    enemy_bucket_next_scratch_.reserve(128);
    enemy_id_counter_ = 0;

    wave_ = 1;
    spawn_timer_ = 0.0f;
    time_ = 0.0f;
    game_over_ = false;

    ++grid_version_;
    ++tower_version_;
    ++money_version_;
    ++enemy_version_;
    ++wave_version_;
    invalidate_all_caches();

    enemies_to_spawn_ = get_wave_enemies();
}

std::vector<EnemySpec> TDEngine::get_fixed_wave_enemies() {
    std::vector<EnemySpec> format;
    float base_hp = 20.0f + wave_ * 15.0f + (wave_ * wave_) * 2.0f;

    int swarm_count = wave_ * 2;
    for (int i = 0; i < swarm_count; ++i) {
        format.push_back({base_hp * 0.3f, 2.8f, 5});
    }

    int reg_count = 5 + wave_ * 2;
    for (int i = 0; i < reg_count; ++i) {
        format.push_back({base_hp, 1.5f, 10});
    }

    if (wave_ >= 3) {
        int tank_count = wave_;
        for (int i = 0; i < tank_count; ++i) {
            format.push_back({base_hp * 3.5f, 0.8f, 30});
        }
    }

    if (wave_ >= 5 && wave_ % 2 == 1) {
        int boss_count = 1 + (wave_ / 10);
        for (int i = 0; i < boss_count; ++i) {
            format.push_back({base_hp * 10.0f, 0.6f, 100});
        }
    }

    rng_.shuffle(format);
    return format;
}

std::vector<EnemySpec> TDEngine::get_budgeted_wave_enemies() {
    AttackBudgetConfig attack_cfg;
    BudgetedWaveConfig wave_cfg;
    auto budget = estimate_attack_budget(*this, attack_cfg);
    auto wave = generate_budgeted_wave(budget, wave_cfg);
    rng_.shuffle(wave.enemies);
    return wave.enemies;
}

std::vector<EnemySpec> TDEngine::get_wave_enemies() {
    if (use_budgeted_waves_) {
        return get_budgeted_wave_enemies();
    }
    return get_fixed_wave_enemies();
}

float TDEngine::pending_spawn_total_hp() const {
    float total = 0.0f;
    for (const auto& enemy : enemies_to_spawn_) {
        total += enemy.hp;
    }
    return total;
}

void TDEngine::set_use_budgeted_waves(bool enabled, bool regenerate_pending_wave) {
    if (game_over_) return;
    if (use_budgeted_waves_ == enabled && !regenerate_pending_wave) {
        return;
    }

    use_budgeted_waves_ = enabled;
    if (regenerate_pending_wave) {
        enemies_to_spawn_ = get_wave_enemies();
        mark_wave_changed();
    }
}

} // namespace tdmz
