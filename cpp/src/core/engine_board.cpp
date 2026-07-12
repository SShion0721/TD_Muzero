#include "tdmz/core/engine.hpp"
#include "tdmz/core/pathfinding.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>

namespace tdmz {

namespace {
constexpr int kInf = 1000000000;

int enemy_rounded_cell(const Enemy& enemy) {
    const int x = static_cast<int>(std::round(enemy.x));
    const int y = static_cast<int>(std::round(enemy.y));
    if (!valid_cell_xy(x, y)) return -1;
    return cell_id(x, y);
}

} // namespace

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
        const int current = queue[head++];
        const int neighbor_count = tables.neighbor4_count[current];
        for (int index = 0; index < neighbor_count; ++index) {
            const int next = tables.neighbors4[current][index];
            const int x = tables.x[next];
            const int y = tables.y[next];
            if (grid_[y][x] != 0 || cached_base_distance_[next] != kInf) {
                continue;
            }
            cached_base_distance_[next] = cached_base_distance_[current] + 1;
            cached_next_to_base_[next] = current;
            queue[tail++] = next;
        }
    }

    base_distance_grid_version_ = grid_version_;
    base_distance_cache_valid_ = true;
}

std::vector<std::pair<int, int>> TDEngine::path_to_base_from_cell(int start_cell) const {
    if (!valid_cell(start_cell)) return {};

    const auto& tables = board_tables();
    const int base = cell_id(base_x_, base_y_);
    if (start_cell == base) return {};

    recompute_base_distance_cache();

    int current = start_cell;
    if (cached_base_distance_[current] == kInf) {
        int best = -1;
        int best_distance = kInf;
        const int neighbor_count = tables.neighbor4_count[current];
        for (int index = 0; index < neighbor_count; ++index) {
            const int next = tables.neighbors4[current][index];
            if (cached_base_distance_[next] < best_distance) {
                best_distance = cached_base_distance_[next];
                best = next;
            }
        }
        if (best < 0 || best_distance == kInf) return {};
        current = best;
    } else {
        current = cached_next_to_base_[current];
    }

    std::vector<std::pair<int, int>> path;
    while (current != base && current >= 0) {
        path.push_back({tables.x[current], tables.y[current]});
        current = cached_next_to_base_[current];
    }
    if (current == base) {
        path.push_back({base_x_, base_y_});
    }
    return path;
}

int TDEngine::distance_to_base_cell(int cell) const {
    if (!valid_cell(cell)) return kInf;
    recompute_base_distance_cache();
    return cached_base_distance_[cell];
}

std::vector<std::pair<int, int>> TDEngine::find_path(int sx, int sy, int ex, int ey) const {
    ++perf_.pathfind_calls;
    if (!valid_cell_xy(sx, sy) || !valid_cell_xy(ex, ey)) return {};
    if (ex == base_x_ && ey == base_y_) {
        return path_to_base_from_cell(cell_id(sx, sy));
    }
    return find_shortest_path(sx, sy, ex, ey, grid_);
}

Bitboard128 TDEngine::enemy_occupancy_bitboard() const {
    if (enemy_occupancy_cache_valid_ && enemy_occupancy_version_ == enemy_version_) {
        return cached_enemy_occupancy_;
    }

    ++perf_.enemy_occupancy_recompute;
    Bitboard128 occupancy = Bitboard128::zero();
    for (const auto& enemy : enemies_) {
        if (enemy.hp <= 0.0f) continue;
        const int cell = enemy_rounded_cell(enemy);
        if (cell >= 0) occupancy.set(cell);
    }

    cached_enemy_occupancy_ = occupancy;
    enemy_occupancy_version_ = enemy_version_;
    enemy_occupancy_cache_valid_ = true;
    return cached_enemy_occupancy_;
}

bool TDEngine::in_bounds(int x, int y) const {
    return x >= 0 && x < width_ && y >= 0 && y < height_;
}

bool TDEngine::can_place_tower(int x, int y, TowerType type) const {
    if (game_over_) return false;
    if (!in_bounds(x, y)) return false;
    if (grid_[y][x] != 0) return false;
    if (x == spawn_x_ && y == spawn_y_) return false;
    if (x == base_x_ && y == base_y_) return false;
    if (enemy_occupancy_bitboard().test(cell_id(x, y))) return false;

    const auto stats = tower_stats(type);
    if (money_ < stats.cost) return false;

    const auto placeable = compute_placeable_mask();
    return placeable[y][x];
}

bool TDEngine::place_tower(int x, int y, TowerType type) {
    if (game_over_ || !can_place_tower(x, y, type)) return false;

    episode_started_ = true;
    const auto stats = tower_stats(type);
    money_ -= stats.cost;
    grid_[y][x] = 1;
    bb_blocked_.set(cell_id(x, y));
    towers_.emplace_back(x, y, type);

    mark_money_changed();
    mark_grid_changed();
    mark_towers_changed();

    for (auto& enemy : enemies_) {
        const int current_x = static_cast<int>(std::round(enemy.x));
        const int current_y = static_cast<int>(std::round(enemy.y));
        const auto path = find_path(current_x, current_y, base_x_, base_y_);
        if (!path.empty()) enemy.set_path(path);
    }
    if (!enemies_.empty()) mark_enemies_changed();
    return true;
}

bool TDEngine::upgrade_tower(int x, int y) {
    if (game_over_) return false;

    for (auto& tower : towers_) {
        if (tower.x != x || tower.y != y) continue;
        if (!tower.can_upgrade() || money_ < tower.upgrade_cost) return false;

        episode_started_ = true;
        money_ -= tower.upgrade_cost;
        tower.upgrade();
        mark_money_changed();
        mark_towers_changed();
        return true;
    }
    return false;
}

bool TDEngine::sell_tower(int x, int y) {
    if (game_over_) return false;

    for (auto iterator = towers_.begin(); iterator != towers_.end(); ++iterator) {
        if (iterator->x != x || iterator->y != y) continue;

        episode_started_ = true;
        const int refund = static_cast<int>(iterator->total_spent * 0.8f);
        money_ += refund;
        towers_.erase(iterator);
        grid_[y][x] = 0;
        bb_blocked_.clear(cell_id(x, y));

        mark_money_changed();
        mark_grid_changed();
        mark_towers_changed();

        for (auto& enemy : enemies_) {
            const int current_x = static_cast<int>(std::round(enemy.x));
            const int current_y = static_cast<int>(std::round(enemy.y));
            const auto path = find_path(current_x, current_y, base_x_, base_y_);
            if (!path.empty()) enemy.set_path(path);
        }
        if (!enemies_.empty()) mark_enemies_changed();
        return true;
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

    std::array<int, kCells> discovery;
    std::array<int, kCells> low;
    std::array<int, kCells> parent;
    std::array<bool, kCells> contains_base;
    std::array<bool, kCells> separates_spawn_and_base;
    discovery.fill(-1);
    low.fill(0);
    parent.fill(-1);
    contains_base.fill(false);
    separates_spawn_and_base.fill(false);

    int timer = 0;
    const auto is_open = [&](int cell) {
        return grid_[tables.y[cell]][tables.x[cell]] == 0;
    };

    std::function<void(int)> dfs = [&](int current) {
        discovery[current] = low[current] = timer++;
        contains_base[current] = current == base;

        const int neighbor_count = tables.neighbor4_count[current];
        for (int index = 0; index < neighbor_count; ++index) {
            const int next = tables.neighbors4[current][index];
            if (!is_open(next)) continue;

            if (discovery[next] == -1) {
                parent[next] = current;
                dfs(next);
                contains_base[current] = contains_base[current] || contains_base[next];
                low[current] = std::min(low[current], low[next]);
                if (current != spawn && current != base &&
                    contains_base[next] && low[next] >= discovery[current]) {
                    separates_spawn_and_base[current] = true;
                }
            } else if (next != parent[current]) {
                low[current] = std::min(low[current], discovery[next]);
            }
        }
    };

    if (is_open(spawn)) dfs(spawn);

    if (discovery[base] != -1) {
        for (int cell = 0; cell < kCells; ++cell) {
            const int x = tables.x[cell];
            const int y = tables.y[cell];
            if (grid_[y][x] == 0 && cell != spawn && cell != base &&
                !separates_spawn_and_base[cell]) {
                mask[y][x] = true;
            }
        }
    }

    cached_placeable_mask_ = mask;
    placeable_grid_version_ = grid_version_;
    placeable_cache_valid_ = true;
    return cached_placeable_mask_;
}

} // namespace tdmz
