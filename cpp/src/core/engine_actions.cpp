#include "tdmz/core/engine.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace tdmz {

std::vector<int> TDEngine::legal_actions() const {
    const auto& tables = board_tables();
    if (game_over_) {
        return {tables.wait_action};
    }

    if (legal_cache_valid_ &&
        legal_grid_version_ == grid_version_ &&
        legal_tower_version_ == tower_version_ &&
        legal_money_version_ == money_version_ &&
        legal_enemy_version_ == enemy_version_) {
        return cached_legal_actions_;
    }

    ++perf_.legal_recompute;

    std::vector<int> actions;
    const auto placeable = compute_placeable_mask();
    const Bitboard128 enemy_occupancy = enemy_occupancy_bitboard();

    for (int tower_index = 0; tower_index < kBuildTypes; ++tower_index) {
        const auto type = static_cast<TowerType>(tower_index);
        if (money_ < tower_stats(type).cost) continue;

        for (int cell = 0; cell < kCells; ++cell) {
            const int x = tables.x[cell];
            const int y = tables.y[cell];
            if (placeable[y][x] && !enemy_occupancy.test(cell)) {
                actions.push_back(tables.build_action[tower_index][cell]);
            }
        }
    }

    for (const auto& tower : towers_) {
        const int cell = cell_id(tower.x, tower.y);
        if (tower.can_upgrade() && money_ >= tower.upgrade_cost) {
            actions.push_back(tables.upgrade_action[cell]);
        }
        actions.push_back(tables.sell_action[cell]);
    }

    actions.push_back(tables.wait_action);

    cached_legal_actions_ = actions;
    legal_grid_version_ = grid_version_;
    legal_tower_version_ = tower_version_;
    legal_money_version_ = money_version_;
    legal_enemy_version_ = enemy_version_;
    legal_cache_valid_ = true;
    return cached_legal_actions_;
}

std::vector<uint8_t> TDEngine::legal_action_mask() const {
    std::vector<uint8_t> mask(kActionSpaceSize, 0);
    for (int action : legal_actions()) {
        mask[static_cast<std::size_t>(action)] = 1;
    }
    return mask;
}

StepResult TDEngine::step_wait(int wait_steps) {
    if (wait_steps < 0) {
        throw std::invalid_argument("wait_steps must be non-negative");
    }

    StepResult result{0.0f, game_over_};
    for (int step = 0; step < wait_steps && !result.done; ++step) {
        const StepResult tick = step_one_tick();
        result.reward += tick.reward;
        result.done = tick.done;
    }
    return result;
}

StepResult TDEngine::step_action(int flat_action) {
    // Terminal states are absorbing. The guard intentionally precedes action
    // decoding so even an arbitrary stale action ID cannot mutate or advance a
    // completed episode.
    if (game_over_) return {0.0f, true};

    const Action action = decode_action(flat_action);
    float reward = 0.0f;

    switch (action.type) {
        case ActionType::BuildBasic:
        case ActionType::BuildSniper:
        case ActionType::BuildAOE:
        case ActionType::BuildSlow:
            if (!place_tower(
                    action.x,
                    action.y,
                    static_cast<TowerType>(static_cast<int>(action.type)))) {
                reward -= 5.0f;
            }
            break;

        case ActionType::Upgrade:
            if (!upgrade_tower(action.x, action.y)) reward -= 5.0f;
            break;

        case ActionType::Sell:
            if (!sell_tower(action.x, action.y)) reward -= 5.0f;
            break;

        case ActionType::Wait1:
            break;
    }

    StepResult result = step_wait(action.wait_steps);
    result.reward += reward;
    return result;
}

StepResult TDEngine::step_time(float dt) {
    const double seconds = static_cast<double>(dt);
    if (!std::isfinite(seconds) || seconds < 0.0 || std::floor(seconds) != seconds ||
        seconds > static_cast<double>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "step_time requires a finite non-negative integer number of seconds");
    }
    return step_wait(static_cast<int>(seconds));
}

} // namespace tdmz
