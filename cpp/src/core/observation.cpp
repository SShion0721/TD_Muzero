#include "tdmz/core/observation.hpp"
#include "tdmz/core/board_tables.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace tdmz {

namespace {

constexpr int kObsPlane = kBoardW * kBoardH;
constexpr float kEpsilon = 1e-6f;
constexpr float kDensityReference = 32.0f;
constexpr float kCellHpReference = 100000.0f;
constexpr float kCellRewardReference = 10000.0f;
constexpr float kCellLeakReference = 10000.0f;
constexpr float kMoneyReference = 100000.0f;
constexpr float kWaveReference = 100.0f;
constexpr float kPendingCountReference = 512.0f;
constexpr float kPendingHpReference = 10000000.0f;
constexpr float kPendingRewardReference = 1000000.0f;
constexpr float kPendingLeakReference = 1000000.0f;
constexpr float kPathDistanceReference = static_cast<float>(kCells - 1);
constexpr float kEnemySpeedReference = 3.0f;
constexpr float kSlowTimerReference = 3.0f;
constexpr float kSpawnTimerReference = 3.0f;

inline int obs_index(int channel, int y, int x) {
    return (channel * kBoardH + y) * kBoardW + x;
}

inline float clamp01(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

float log_scale(float value, float reference) {
    if (!(value > 0.0f) || !(reference > 0.0f)) return 0.0f;
    return clamp01(std::log1p(value) / std::log1p(reference));
}

int leak_damage_from_hp(float max_hp) {
    return std::max(0, static_cast<int>(std::max(0.0f, max_hp)) / 10);
}

uint16_t pack_static_tower(const Tower& tower) {
    const int cell = cell_id(tower.x, tower.y);
    const int type = static_cast<int>(tower.type);
    return static_cast<uint16_t>(
        (cell & 0x7f) |
        ((type & 0x03) << 7) |
        ((tower.level & 0x07) << 9)
    );
}

struct ObservationStaticCache {
    bool valid = false;
    Bitboard128 blocked{};
    std::vector<uint16_t> towers;
    Observation static_obs;
};

thread_local ObservationStaticCache g_static_cache;

bool static_key_matches(const TDEngine& env, Bitboard128 blocked) {
    if (!g_static_cache.valid || g_static_cache.blocked != blocked) return false;
    if (g_static_cache.towers.size() != env.towers().size()) return false;

    for (std::size_t i = 0; i < env.towers().size(); ++i) {
        if (g_static_cache.towers[i] != pack_static_tower(env.towers()[i])) {
            return false;
        }
    }
    return true;
}

const Observation& get_static_observation_channels(const TDEngine& env) {
    const Bitboard128 blocked = env.blocked_bitboard();
    if (static_key_matches(env, blocked) &&
        g_static_cache.static_obs.size() == OBS_CHANNELS * kObsPlane) {
        return g_static_cache.static_obs;
    }

    g_static_cache.valid = true;
    g_static_cache.blocked = blocked;
    g_static_cache.towers.resize(env.towers().size());
    for (std::size_t i = 0; i < env.towers().size(); ++i) {
        g_static_cache.towers[i] = pack_static_tower(env.towers()[i]);
    }
    g_static_cache.static_obs.assign(OBS_CHANNELS * kObsPlane, 0.0f);

    auto at = [&](int channel, int y, int x) -> float& {
        return g_static_cache.static_obs[obs_index(channel, y, x)];
    };

    for (int y = 0; y < kBoardH; ++y) {
        for (int x = 0; x < kBoardW; ++x) {
            if (env.grid()[y][x] == 1) at(CH_BLOCKED, y, x) = 1.0f;
        }
    }

    at(CH_SPAWN, env.spawn_y(), env.spawn_x()) = 1.0f;
    at(CH_BASE, env.base_y(), env.base_x()) = 1.0f;

    for (const auto& tower : env.towers()) {
        const int x = tower.x;
        const int y = tower.y;
        if (tower.type == TowerType::Basic) at(CH_TOWER_BASIC, y, x) = 1.0f;
        else if (tower.type == TowerType::Sniper) at(CH_TOWER_SNIPER, y, x) = 1.0f;
        else if (tower.type == TowerType::AOE) at(CH_TOWER_AOE, y, x) = 1.0f;
        else if (tower.type == TowerType::Slow) at(CH_TOWER_SLOW, y, x) = 1.0f;

        at(CH_TOWER_LEVEL, y, x) = clamp01(
            static_cast<float>(tower.level) / static_cast<float>(kTowerMaxLevel));
    }

    const auto placeable_mask = env.compute_placeable_mask();
    for (int y = 0; y < kBoardH; ++y) {
        for (int x = 0; x < kBoardW; ++x) {
            const int cell = cell_id(x, y);
            if (placeable_mask[y][x]) {
                at(CH_STRUCTURAL_PLACEABLE, y, x) = 1.0f;
            }
            const int distance = env.distance_to_base_cell(cell);
            if (distance < 1000000000) {
                at(CH_DISTANCE_TO_BASE, y, x) = clamp01(
                    static_cast<float>(distance) / kPathDistanceReference);
            }
        }
    }

    return g_static_cache.static_obs;
}

template <typename Fn>
void for_each_bilinear_cell(float x, float y, Fn&& fn) {
    const float clamped_x = std::clamp(x, 0.0f, static_cast<float>(kBoardW - 1));
    const float clamped_y = std::clamp(y, 0.0f, static_cast<float>(kBoardH - 1));
    const int x0 = static_cast<int>(std::floor(clamped_x));
    const int y0 = static_cast<int>(std::floor(clamped_y));
    const int x1 = std::min(x0 + 1, kBoardW - 1);
    const int y1 = std::min(y0 + 1, kBoardH - 1);
    const float fx = clamped_x - static_cast<float>(x0);
    const float fy = clamped_y - static_cast<float>(y0);

    const std::array<int, 4> xs{x0, x1, x0, x1};
    const std::array<int, 4> ys{y0, y0, y1, y1};
    const std::array<float, 4> weights{
        (1.0f - fx) * (1.0f - fy),
        fx * (1.0f - fy),
        (1.0f - fx) * fy,
        fx * fy,
    };

    for (int i = 0; i < 4; ++i) {
        if (weights[static_cast<std::size_t>(i)] > 0.0f) {
            fn(cell_id(xs[static_cast<std::size_t>(i)], ys[static_cast<std::size_t>(i)]),
               weights[static_cast<std::size_t>(i)]);
        }
    }
}

float enemy_remaining_path_distance(const Enemy& enemy) {
    if (enemy.path_cursor >= enemy.path.size()) return 0.0f;
    const auto& next = enemy.path[enemy.path_cursor];
    const float first_segment = std::hypot(
        static_cast<float>(next.first) - enemy.x,
        static_cast<float>(next.second) - enemy.y);
    const std::size_t full_segments = enemy.path.size() - enemy.path_cursor - 1u;
    return first_segment + static_cast<float>(full_segments);
}

void fill_plane(Observation& output, int channel, float value) {
    const auto begin = output.begin() + static_cast<std::ptrdiff_t>(channel * kObsPlane);
    std::fill(begin, begin + kObsPlane, value);
}

} // namespace

Observation make_observation_python_parity(const TDEngine& env) {
    const int w = env.width();
    const int h = env.height();
    Observation obs(5 * h * w, 0.0f);

    auto at = [&](int channel, int y, int x) -> float& {
        return obs[(channel * h + y) * w + x];
    };

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (env.grid()[y][x] == 1) at(0, y, x) = 1.0f;
        }
    }

    at(1, env.spawn_y(), env.spawn_x()) = 1.0f;
    at(1, env.base_y(), env.base_x()) = 1.0f;

    for (const auto& enemy : env.enemies()) {
        const int grid_x = static_cast<int>(enemy.x);
        const int grid_y = static_cast<int>(enemy.y);
        if (grid_x >= 0 && grid_x < w && grid_y >= 0 && grid_y < h) {
            const float added_hp = enemy.hp / std::max(1.0f, enemy.max_hp);
            at(2, grid_y, grid_x) = std::min(1.0f, at(2, grid_y, grid_x) + added_hp);
        }
    }

    fill_plane(obs, 3, static_cast<float>(env.base_hp()) / 100.0f);
    fill_plane(obs, 4, std::min(1.0f, static_cast<float>(env.money()) / 1000.0f));
    return obs;
}

std::array<int, 3> observation_shape_python_parity() {
    return {5, kBoardH, kBoardW};
}

void make_observation_v2_into(const TDEngine& env, Observation& output) {
    const Observation& static_obs = get_static_observation_channels(env);
    output.assign(static_obs.begin(), static_obs.end());

    auto at_cell = [&](int channel, int cell) -> float& {
        return output[static_cast<std::size_t>(channel * kObsPlane + cell)];
    };
    auto at = [&](int channel, int y, int x) -> float& {
        return output[obs_index(channel, y, x)];
    };

    for (const auto& tower : env.towers()) {
        const float cooldown_ratio = tower.cooldown_max > 0.0f
            ? tower.cooldown / tower.cooldown_max
            : 0.0f;
        at(CH_TOWER_COOLDOWN, tower.y, tower.x) = clamp01(cooldown_ratio);
    }

    std::array<float, kCells> density{};
    std::array<float, kCells> hp_ratio_mass{};
    std::array<float, kCells> current_hp_mass{};
    std::array<float, kCells> max_hp_mass{};
    std::array<float, kCells> speed_mass{};
    std::array<float, kCells> slow_timer_mass{};
    std::array<float, kCells> reward_mass{};
    std::array<float, kCells> leak_mass{};
    std::array<float, kCells> remaining_distance_mass{};
    std::array<float, kCells> direction_right_mass{};
    std::array<float, kCells> direction_left_mass{};
    std::array<float, kCells> direction_down_mass{};
    std::array<float, kCells> direction_up_mass{};

    for (const auto& enemy : env.enemies()) {
        if (enemy.hp <= 0.0f) continue;

        const float current_hp = std::max(0.0f, enemy.hp);
        const float max_hp = std::max(1.0f, enemy.max_hp);
        const float hp_ratio = clamp01(current_hp / max_hp);
        const float speed = std::max(0.0f, enemy.speed);
        const float slow_timer = std::max(0.0f, enemy.slow_timer);
        const float reward = static_cast<float>(std::max(0, enemy.reward));
        const float leak_damage = static_cast<float>(leak_damage_from_hp(enemy.max_hp));
        const float remaining_distance = enemy_remaining_path_distance(enemy);

        float dir_x = 0.0f;
        float dir_y = 0.0f;
        const float target_dx = enemy.target_x - enemy.x;
        const float target_dy = enemy.target_y - enemy.y;
        const float target_distance = std::hypot(target_dx, target_dy);
        if (target_distance > kEpsilon) {
            dir_x = target_dx / target_distance;
            dir_y = target_dy / target_distance;
        }

        for_each_bilinear_cell(enemy.x, enemy.y, [&](int cell, float weight) {
            density[static_cast<std::size_t>(cell)] += weight;
            hp_ratio_mass[static_cast<std::size_t>(cell)] += weight * hp_ratio;
            current_hp_mass[static_cast<std::size_t>(cell)] += weight * current_hp;
            max_hp_mass[static_cast<std::size_t>(cell)] += weight * max_hp;
            speed_mass[static_cast<std::size_t>(cell)] += weight * speed;
            slow_timer_mass[static_cast<std::size_t>(cell)] += weight * slow_timer;
            reward_mass[static_cast<std::size_t>(cell)] += weight * reward;
            leak_mass[static_cast<std::size_t>(cell)] += weight * leak_damage;
            remaining_distance_mass[static_cast<std::size_t>(cell)] += weight * remaining_distance;
            direction_right_mass[static_cast<std::size_t>(cell)] += weight * std::max(0.0f, dir_x);
            direction_left_mass[static_cast<std::size_t>(cell)] += weight * std::max(0.0f, -dir_x);
            direction_down_mass[static_cast<std::size_t>(cell)] += weight * std::max(0.0f, dir_y);
            direction_up_mass[static_cast<std::size_t>(cell)] += weight * std::max(0.0f, -dir_y);
        });

        const int occupied_x = static_cast<int>(std::round(enemy.x));
        const int occupied_y = static_cast<int>(std::round(enemy.y));
        if (valid_cell_xy(occupied_x, occupied_y)) {
            at(CH_ENEMY_OCCUPIED, occupied_y, occupied_x) = 1.0f;
        }
    }

    for (int cell = 0; cell < kCells; ++cell) {
        const float count = density[static_cast<std::size_t>(cell)];
        if (count <= kEpsilon) continue;

        const float inv_count = 1.0f / count;
        at_cell(CH_ENEMY_DENSITY, cell) = log_scale(count, kDensityReference);
        at_cell(CH_ENEMY_HP_RATIO, cell) = clamp01(
            hp_ratio_mass[static_cast<std::size_t>(cell)] * inv_count);
        at_cell(CH_ENEMY_CURRENT_HP, cell) = log_scale(
            current_hp_mass[static_cast<std::size_t>(cell)], kCellHpReference);
        at_cell(CH_ENEMY_MAX_HP, cell) = log_scale(
            max_hp_mass[static_cast<std::size_t>(cell)], kCellHpReference);
        at_cell(CH_ENEMY_SPEED, cell) = clamp01(
            speed_mass[static_cast<std::size_t>(cell)] * inv_count / kEnemySpeedReference);
        at_cell(CH_ENEMY_SLOW_TIMER, cell) = clamp01(
            slow_timer_mass[static_cast<std::size_t>(cell)] * inv_count / kSlowTimerReference);
        at_cell(CH_ENEMY_REWARD, cell) = log_scale(
            reward_mass[static_cast<std::size_t>(cell)], kCellRewardReference);
        at_cell(CH_ENEMY_LEAK_DAMAGE, cell) = log_scale(
            leak_mass[static_cast<std::size_t>(cell)], kCellLeakReference);
        at_cell(CH_ENEMY_REMAINING_DISTANCE, cell) = clamp01(
            remaining_distance_mass[static_cast<std::size_t>(cell)] * inv_count /
            kPathDistanceReference);
        at_cell(CH_ENEMY_DIR_RIGHT, cell) = clamp01(
            direction_right_mass[static_cast<std::size_t>(cell)] * inv_count);
        at_cell(CH_ENEMY_DIR_LEFT, cell) = clamp01(
            direction_left_mass[static_cast<std::size_t>(cell)] * inv_count);
        at_cell(CH_ENEMY_DIR_DOWN, cell) = clamp01(
            direction_down_mass[static_cast<std::size_t>(cell)] * inv_count);
        at_cell(CH_ENEMY_DIR_UP, cell) = clamp01(
            direction_up_mass[static_cast<std::size_t>(cell)] * inv_count);
    }

    const auto& pending = env.pending_spawns();
    float pending_total_hp = 0.0f;
    float pending_total_reward = 0.0f;
    float pending_total_leak = 0.0f;
    float pending_total_speed = 0.0f;
    float pending_max_speed = 0.0f;
    for (const auto& spec : pending) {
        pending_total_hp += std::max(0.0f, spec.hp);
        pending_total_reward += static_cast<float>(std::max(0, spec.reward));
        pending_total_leak += static_cast<float>(leak_damage_from_hp(spec.hp));
        pending_total_speed += std::max(0.0f, spec.speed);
        pending_max_speed = std::max(pending_max_speed, std::max(0.0f, spec.speed));
    }

    const float pending_count = static_cast<float>(pending.size());
    const float base_hp_value = clamp01(static_cast<float>(env.base_hp()) / 100.0f);
    const float money_value = log_scale(static_cast<float>(std::max(0, env.money())), kMoneyReference);
    const float wave_value = log_scale(static_cast<float>(std::max(0, env.wave())), kWaveReference);
    const float spawn_timer_value = pending.empty()
        ? 0.0f
        : clamp01(env.spawn_timer() / kSpawnTimerReference);
    const float pending_count_value = log_scale(pending_count, kPendingCountReference);
    const float pending_mean_speed_value = pending.empty()
        ? 0.0f
        : clamp01((pending_total_speed / pending_count) / kEnemySpeedReference);

    fill_plane(output, CH_BASE_HP, base_hp_value);
    fill_plane(output, CH_MONEY, money_value);
    fill_plane(output, CH_WAVE, wave_value);
    fill_plane(output, CH_SPAWN_TIMER, spawn_timer_value);
    fill_plane(output, CH_PENDING_COUNT, pending_count_value);
    fill_plane(output, CH_PENDING_TOTAL_HP, log_scale(pending_total_hp, kPendingHpReference));
    fill_plane(output, CH_PENDING_TOTAL_REWARD, log_scale(pending_total_reward, kPendingRewardReference));
    fill_plane(output, CH_PENDING_TOTAL_LEAK_DAMAGE, log_scale(pending_total_leak, kPendingLeakReference));
    fill_plane(output, CH_PENDING_MEAN_SPEED, pending_mean_speed_value);
    fill_plane(output, CH_PENDING_MAX_SPEED, clamp01(pending_max_speed / kEnemySpeedReference));

    if (!pending.empty()) {
        const EnemySpec& next = pending.front();
        fill_plane(output, CH_NEXT_ENEMY_PRESENT, 1.0f);
        fill_plane(output, CH_NEXT_ENEMY_HP, log_scale(std::max(0.0f, next.hp), kCellHpReference));
        fill_plane(output, CH_NEXT_ENEMY_SPEED, clamp01(std::max(0.0f, next.speed) / kEnemySpeedReference));
        fill_plane(output, CH_NEXT_ENEMY_REWARD, log_scale(
            static_cast<float>(std::max(0, next.reward)), kCellRewardReference));
        fill_plane(output, CH_NEXT_ENEMY_LEAK_DAMAGE, log_scale(
            static_cast<float>(leak_damage_from_hp(next.hp)), kCellLeakReference));
    }
}

Observation make_observation_v2(const TDEngine& env) {
    Observation output;
    make_observation_v2_into(env, output);
    return output;
}

std::array<int, 3> observation_shape_v2() {
    return {OBS_CHANNELS, kBoardH, kBoardW};
}

} // namespace tdmz
