#include "tdmz/core/engine.hpp"
#include "tdmz/balance/attack_budget.hpp"
#include "tdmz/balance/budgeted_wave_generator.hpp"
#include "tdmz/core/pathfinding.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace tdmz {

namespace {
constexpr int kInf = 1000000000;

int enemy_rounded_cell(const Enemy& enemy) {
    int ex = static_cast<int>(std::round(enemy.x));
    int ey = static_cast<int>(std::round(enemy.y));
    if (!valid_cell_xy(ex, ey)) return -1;
    return cell_id(ex, ey);
}

} // namespace

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
    for (const auto& e : enemies_to_spawn_) {
        total += e.hp;
    }
    return total;
}

void TDEngine::set_use_budgeted_waves(bool enabled, bool regenerate_pending_wave) {
    if (use_budgeted_waves_ == enabled && !regenerate_pending_wave) {
        return;
    }
    use_budgeted_waves_ = enabled;
    if (regenerate_pending_wave) {
        enemies_to_spawn_ = get_wave_enemies();
        mark_wave_changed();
    }
}

void TDEngine::recompute_base_distance_cache() const {
    if (base_distance_cache_valid_ && base_distance_grid_version_ == grid_version_) {
        return;
    }

    ++perf_.base_distance_recompute;
    const auto& tables = board_tables();
    cached_base_distance_.fill(kInf);
    cached_next_to_base_.fill(-1);

    std::array<int, kCells> queue;
    int head = 0;
    int tail = 0;

    const int base = cell_id(base_x_, base_y_);
    cached_base_distance_[base] = 0;
    queue[tail++] = base;

    while (head < tail) {
        int cur = queue[head++];
        int cnt = tables.neighbor4_count[cur];
        for (int i = 0; i < cnt; ++i) {
            int next = tables.neighbors4[cur][i];
            int nx = tables.x[next];
            int ny = tables.y[next];
            if (grid_[ny][nx] != 0) {
                continue;
            }
            if (cached_base_distance_[next] != kInf) {
                continue;
            }
            cached_base_distance_[next] = cached_base_distance_[cur] + 1;
            cached_next_to_base_[next] = cur;
            queue[tail++] = next;
        }
    }

    base_distance_grid_version_ = grid_version_;
    base_distance_cache_valid_ = true;
}

std::vector<std::pair<int,int>> TDEngine::path_to_base_from_cell(int start_cell) const {
    if (!valid_cell(start_cell)) return {};
    const auto& tables = board_tables();
    const int base = cell_id(base_x_, base_y_);
    if (start_cell == base) return {};

    recompute_base_distance_cache();

    int cur = start_cell;
    if (cached_base_distance_[cur] == kInf) {
        int best = -1;
        int best_dist = kInf;
        int cnt = tables.neighbor4_count[cur];
        for (int i = 0; i < cnt; ++i) {
            int next = tables.neighbors4[cur][i];
            if (cached_base_distance_[next] < best_dist) {
                best_dist = cached_base_distance_[next];
                best = next;
            }
        }
        if (best < 0 || best_dist == kInf) {
            return {};
        }
        cur = best;
    } else {
        cur = cached_next_to_base_[cur];
    }

    std::vector<std::pair<int,int>> path;
    while (cur != base && cur >= 0) {
        path.push_back({tables.x[cur], tables.y[cur]});
        cur = cached_next_to_base_[cur];
    }
    if (cur == base) {
        path.push_back({base_x_, base_y_});
    }
    return path;
}

int TDEngine::distance_to_base_cell(int cell) const {
    if (!valid_cell(cell)) return kInf;
    recompute_base_distance_cache();
    return cached_base_distance_[cell];
}

std::vector<std::pair<int,int>> TDEngine::find_path(int sx, int sy, int ex, int ey) const {
    ++perf_.pathfind_calls;
    if (!valid_cell_xy(sx, sy) || !valid_cell_xy(ex, ey)) {
        return {};
    }
    if (ex == base_x_ && ey == base_y_) {
        return path_to_base_from_cell(cell_id(sx, sy));
    }
    return find_shortest_path(sx, sy, ex, ey, grid_);
}

bool TDEngine::in_bounds(int x, int y) const {
    return x >= 0 && x < width_ && y >= 0 && y < height_;
}

bool TDEngine::can_place_tower(int x, int y, TowerType type) const {
    if (!in_bounds(x, y)) return false;
    if (grid_[y][x] != 0) return false;
    if (x == spawn_x_ && y == spawn_y_) return false;
    if (x == base_x_ && y == base_y_) return false;

    auto stats = tower_stats(type);
    if (money_ < stats.cost) return false;

    auto placeable = compute_placeable_mask();
    return placeable[y][x];
}

bool TDEngine::place_tower(int x, int y, TowerType type) {
    if (can_place_tower(x, y, type)) {
        auto stats = tower_stats(type);
        money_ -= stats.cost;
        grid_[y][x] = 1;
        bb_blocked_.set(cell_id(x, y));
        towers_.emplace_back(x, y, type);

        mark_money_changed();
        mark_grid_changed();
        mark_towers_changed();

        for (auto& enemy : enemies_) {
            int curr_grid_x = static_cast<int>(std::round(enemy.x));
            int curr_grid_y = static_cast<int>(std::round(enemy.y));
            auto path = find_path(curr_grid_x, curr_grid_y, base_x_, base_y_);
            if (!path.empty()) {
                enemy.set_path(path);
            }
        }
        if (!enemies_.empty()) mark_enemies_changed();
        return true;
    }
    return false;
}

bool TDEngine::upgrade_tower(int x, int y) {
    for (auto& tower : towers_) {
        if (tower.x == x && tower.y == y) {
            if (!tower.can_upgrade()) return false;
            if (money_ >= tower.upgrade_cost) {
                money_ -= tower.upgrade_cost;
                tower.upgrade();
                mark_money_changed();
                mark_towers_changed();
                return true;
            }
            break;
        }
    }
    return false;
}

bool TDEngine::sell_tower(int x, int y) {
    for (auto it = towers_.begin(); it != towers_.end(); ++it) {
        if (it->x == x && it->y == y) {
            int refund = static_cast<int>(it->total_spent * 0.8f);
            money_ += refund;
            towers_.erase(it);
            grid_[y][x] = 0;
            bb_blocked_.clear(cell_id(x, y));

            mark_money_changed();
            mark_grid_changed();
            mark_towers_changed();

            for (auto& enemy : enemies_) {
                int curr_grid_x = static_cast<int>(std::round(enemy.x));
                int curr_grid_y = static_cast<int>(std::round(enemy.y));
                auto path = find_path(curr_grid_x, curr_grid_y, base_x_, base_y_);
                if (!path.empty()) {
                    enemy.set_path(path);
                }
            }
            if (!enemies_.empty()) mark_enemies_changed();
            return true;
        }
    }
    return false;
}

std::array<std::array<bool, kBoardW>, kBoardH> TDEngine::compute_placeable_mask() const {
    if (placeable_cache_valid_ && placeable_grid_version_ == grid_version_) {
        return cached_placeable_mask_;
    }

    ++perf_.placeable_recompute;

    std::array<std::array<bool, kBoardW>, kBoardH> mask;
    for (auto& row : mask) row.fill(false);

    const auto& tables = board_tables();
    const int spawn = cell_id(spawn_x_, spawn_y_);
    const int base = cell_id(base_x_, base_y_);

    std::array<int, kCells> tin;
    std::array<int, kCells> low;
    std::array<int, kCells> parent;
    std::array<bool, kCells> contains_base;
    std::array<bool, kCells> st_cut;
    tin.fill(-1);
    low.fill(0);
    parent.fill(-1);
    contains_base.fill(false);
    st_cut.fill(false);

    int timer = 0;
    auto is_open = [&](int c) {
        return grid_[tables.y[c]][tables.x[c]] == 0;
    };

    std::function<void(int)> dfs = [&](int u) {
        tin[u] = low[u] = timer++;
        contains_base[u] = (u == base);
        int cnt = tables.neighbor4_count[u];
        for (int i = 0; i < cnt; ++i) {
            int v = tables.neighbors4[u][i];
            if (!is_open(v)) continue;
            if (tin[v] == -1) {
                parent[v] = u;
                dfs(v);
                contains_base[u] = contains_base[u] || contains_base[v];
                low[u] = std::min(low[u], low[v]);
                if (u != spawn && u != base && contains_base[v] && low[v] >= tin[u]) {
                    st_cut[u] = true;
                }
            } else if (v != parent[u]) {
                low[u] = std::min(low[u], tin[v]);
            }
        }
    };

    if (is_open(spawn)) {
        dfs(spawn);
    }

    if (tin[base] != -1) {
        for (int c = 0; c < kCells; ++c) {
            int x = tables.x[c];
            int y = tables.y[c];
            if (grid_[y][x] == 0 && c != spawn && c != base && !st_cut[c]) {
                mask[y][x] = true;
            }
        }
    }

    cached_placeable_mask_ = mask;
    placeable_grid_version_ = grid_version_;
    placeable_cache_valid_ = true;
    return cached_placeable_mask_;
}

std::vector<int> TDEngine::legal_actions() const {
    if (legal_cache_valid_ &&
        legal_grid_version_ == grid_version_ &&
        legal_tower_version_ == tower_version_ &&
        legal_money_version_ == money_version_) {
        return cached_legal_actions_;
    }

    ++perf_.legal_recompute;

    const auto& tables = board_tables();
    std::vector<int> actions;
    auto placeable = compute_placeable_mask();

    for (int t = 0; t < kBuildTypes; ++t) {
        auto type = static_cast<TowerType>(t);
        if (money_ >= tower_stats(type).cost) {
            for (int c = 0; c < kCells; ++c) {
                int x = tables.x[c];
                int y = tables.y[c];
                if (placeable[y][x]) {
                    actions.push_back(tables.build_action[t][c]);
                }
            }
        }
    }

    for (const auto& tower : towers_) {
        int c = cell_id(tower.x, tower.y);
        if (tower.can_upgrade() && money_ >= tower.upgrade_cost) {
            actions.push_back(tables.upgrade_action[c]);
        }
        actions.push_back(tables.sell_action[c]);
    }

    actions.push_back(tables.wait_action);

    cached_legal_actions_ = actions;
    legal_grid_version_ = grid_version_;
    legal_tower_version_ = tower_version_;
    legal_money_version_ = money_version_;
    legal_cache_valid_ = true;
    return cached_legal_actions_;
}

std::vector<uint8_t> TDEngine::legal_action_mask() const {
    std::vector<uint8_t> mask(kActionSpaceSize, 0);
    for (int a : legal_actions()) {
        mask[a] = 1;
    }
    return mask;
}

StepResult TDEngine::step_wait(int wait_steps) {
    StepResult r;
    for (int i = 0; i < wait_steps; ++i) {
        auto step_r = step_time(1.0f);
        r.reward += step_r.reward;
        if (step_r.done) {
            r.done = true;
            break;
        }
    }
    return r;
}

StepResult TDEngine::step_action(int flat_action) {
    Action a = decode_action(flat_action);
    float reward = 0.0f;

    switch (a.type) {
        case ActionType::BuildBasic:
        case ActionType::BuildSniper:
        case ActionType::BuildAOE:
        case ActionType::BuildSlow:
            if (!place_tower(a.x, a.y, static_cast<TowerType>(static_cast<int>(a.type)))) {
                reward -= 5.0f;
            }
            break;

        case ActionType::Upgrade:
            if (!upgrade_tower(a.x, a.y)) reward -= 5.0f;
            break;

        case ActionType::Sell:
            if (!sell_tower(a.x, a.y)) reward -= 5.0f;
            break;

        case ActionType::Wait1:
            break;
    }

    StepResult r = step_wait(a.wait_steps);
    r.reward += reward;
    return r;
}

StepResult TDEngine::step_time(float dt) {
    if (game_over_) return {0.0f, true};

    float step_reward = 0.0f;
    bool enemies_changed = false;

    spawn_timer_ -= dt;
    if (spawn_timer_ <= 0.0f && !enemies_to_spawn_.empty()) {
        auto e_data = enemies_to_spawn_.front();
        enemies_to_spawn_.erase(enemies_to_spawn_.begin());

        enemies_.emplace_back(enemy_id_counter_++, static_cast<float>(spawn_x_), static_cast<float>(spawn_y_),
                              e_data.hp, e_data.speed, e_data.reward);

        auto path = find_path(spawn_x_, spawn_y_, base_x_, base_y_);
        enemies_.back().set_path(path);

        spawn_timer_ = 1.0f;
        enemies_changed = true;
    }

    std::vector<int> reached_base_ids;
    for (auto& enemy : enemies_) {
        enemy.step(dt);
        enemies_changed = true;
        if (std::hypot(enemy.x - static_cast<float>(base_x_), enemy.y - static_cast<float>(base_y_)) < 0.2f) {
            reached_base_ids.push_back(enemy.id);
        }
    }

    for (int id : reached_base_ids) {
        auto it = std::find_if(enemies_.begin(), enemies_.end(), [id](const Enemy& e) { return e.id == id; });
        if (it != enemies_.end()) {
            base_hp_ -= static_cast<int>(it->max_hp) / 10;
            enemies_.erase(it);
            step_reward -= 50.0f;
            enemies_changed = true;
        }
    }

    if (base_hp_ <= 0) {
        game_over_ = true;
        if (enemies_changed) mark_enemies_changed();
        step_reward -= 1000.0f;
        return {step_reward, true};
    }

    const auto& tables = board_tables();
    std::array<int, kCells> bucket_head;
    bucket_head.fill(-1);
    std::vector<int> bucket_next(enemies_.size(), -1);
    Bitboard128 enemy_occupancy = Bitboard128::zero();
    bool has_offboard_enemy = false;
    if (!enemies_.empty()) {
        ++perf_.enemy_bucket_recompute;
        for (int i = 0; i < static_cast<int>(enemies_.size()); ++i) {
            int c = enemy_rounded_cell(enemies_[i]);
            if (c >= 0) {
                bucket_next[i] = bucket_head[c];
                bucket_head[c] = i;
                enemy_occupancy |= tables.cell_bb[c];
            } else {
                has_offboard_enemy = true;
            }
        }
    }

    for (auto& tower : towers_) {
        tower.step(dt);
        if (tower.can_shoot()) {
            int best_index = -1;
            float best_dist2 = 1e30f;
            const float range2 = tower.range * tower.range;

            if (!enemies_.empty()) {
                int tower_cell = cell_id(tower.x, tower.y);
                int tower_type = static_cast<int>(tower.type);
                int tower_level = std::max(1, std::min(kTowerMaxLevel, tower.level));
                Bitboard128 candidate_cells = tables.range_mask_expanded8[tower_type][tower_level][tower_cell] & enemy_occupancy;
                perf_.tower_candidate_cells += static_cast<uint64_t>(candidate_cells.popcount());

                while (candidate_cells) {
                    int c = candidate_cells.pop_lsb();
                    for (int enemy_index = bucket_head[c]; enemy_index >= 0; enemy_index = bucket_next[enemy_index]) {
                        Enemy& enemy = enemies_[enemy_index];
                        float dx = enemy.x - static_cast<float>(tower.x);
                        float dy = enemy.y - static_cast<float>(tower.y);
                        float dist2 = dx * dx + dy * dy;
                        ++perf_.tower_exact_distance_checks;
                        if (dist2 <= range2 && (dist2 < best_dist2 || (dist2 == best_dist2 && (best_index < 0 || enemy_index < best_index)))) {
                            best_dist2 = dist2;
                            best_index = enemy_index;
                        }
                    }
                }

                // Extremely defensive fallback for any future off-board/interpolated enemy states.
                if (best_index < 0 && has_offboard_enemy) {
                    for (int enemy_index = 0; enemy_index < static_cast<int>(enemies_.size()); ++enemy_index) {
                        if (enemy_rounded_cell(enemies_[enemy_index]) >= 0) continue;
                        Enemy& enemy = enemies_[enemy_index];
                        float dx = enemy.x - static_cast<float>(tower.x);
                        float dy = enemy.y - static_cast<float>(tower.y);
                        float dist2 = dx * dx + dy * dy;
                        ++perf_.tower_exact_distance_checks;
                        if (dist2 <= range2 && dist2 < best_dist2) {
                            best_dist2 = dist2;
                            best_index = enemy_index;
                        }
                    }
                }
            }

            if (best_index >= 0) {
                Enemy* best_target = &enemies_[best_index];
                if (tower.aoe_radius > 0.0f) {
                    const float aoe2 = tower.aoe_radius * tower.aoe_radius;
                    for (auto& e : enemies_) {
                        float dx = e.x - best_target->x;
                        float dy = e.y - best_target->y;
                        ++perf_.tower_aoe_distance_checks;
                        if (dx * dx + dy * dy <= aoe2) {
                            e.hp -= tower.damage;
                            e.apply_slow(tower.slow_factor, tower.slow_duration);
                            enemies_changed = true;
                        }
                    }
                } else {
                    best_target->hp -= tower.damage;
                    best_target->apply_slow(tower.slow_factor, tower.slow_duration);
                    enemies_changed = true;
                }
                tower.shoot();
            }
        }
    }

    std::vector<int> dead_ids;
    for (const auto& e : enemies_) {
        if (e.hp <= 0.0f) dead_ids.push_back(e.id);
    }

    for (int id : dead_ids) {
        auto it = std::find_if(enemies_.begin(), enemies_.end(), [id](const Enemy& e) { return e.id == id; });
        if (it != enemies_.end()) {
            money_ += it->reward;
            step_reward += it->reward;
            enemies_.erase(it);
            mark_money_changed();
            enemies_changed = true;
        }
    }

    if (enemies_.empty() && enemies_to_spawn_.empty()) {
        wave_ += 1;
        enemies_to_spawn_ = get_wave_enemies();
        spawn_timer_ = 3.0f;
        step_reward += 100.0f;
        mark_wave_changed();
    }

    if (enemies_changed) mark_enemies_changed();
    time_ += dt;
    return {step_reward, false};
}

} // namespace tdmz
