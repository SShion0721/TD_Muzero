#include "tdmz/core/board_tables.hpp"
#include <cmath>

namespace tdmz {

namespace {

Bitboard128 expand_mask_8_neighbors(Bitboard128 mask, const std::array<int, kCells>& xs, const std::array<int, kCells>& ys, const std::array<Bitboard128, kCells>& cell_bb) {
    Bitboard128 expanded = mask;
    Bitboard128 work = mask;
    while (work) {
        int c = work.pop_lsb();
        int cx = xs[c];
        int cy = ys[c];
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = cx + dx;
                int ny = cy + dy;
                if (valid_cell_xy(nx, ny)) {
                    expanded |= cell_bb[cell_id(nx, ny)];
                }
            }
        }
    }
    return expanded;
}

} // namespace

BoardTables::BoardTables() {
    board_mask = Bitboard128::zero();

    for (int c = 0; c < kCells; ++c) {
        x[c] = cell_x(c);
        y[c] = cell_y(c);
        cell_bb[c] = Bitboard128::from_cell(c);
        board_mask |= cell_bb[c];
        neighbor4_count[c] = 0;
        neighbor4_bb[c] = Bitboard128::zero();
        for (int i = 0; i < 4; ++i) neighbors4[c][i] = -1;
    }

    const int dx[4] = {0, 1, 0, -1};
    const int dy[4] = {1, 0, -1, 0};
    for (int c = 0; c < kCells; ++c) {
        int n = 0;
        for (int i = 0; i < 4; ++i) {
            int nx = x[c] + dx[i];
            int ny = y[c] + dy[i];
            if (valid_cell_xy(nx, ny)) {
                int nc = cell_id(nx, ny);
                neighbors4[c][n++] = nc;
                neighbor4_bb[c] |= cell_bb[nc];
            }
        }
        neighbor4_count[c] = n;
    }

    for (int a = 0; a < kCells; ++a) {
        for (int b = 0; b < kCells; ++b) {
            int dx_abs = x[a] > x[b] ? x[a] - x[b] : x[b] - x[a];
            int dy_abs = y[a] > y[b] ? y[a] - y[b] : y[b] - y[a];
            manhattan[a][b] = dx_abs + dy_abs;
            float fx = static_cast<float>(x[a] - x[b]);
            float fy = static_cast<float>(y[a] - y[b]);
            dist2[a][b] = fx * fx + fy * fy;
        }
    }

    for (int t = 0; t < kBuildTypes; ++t) {
        TowerStats stats = tower_stats(static_cast<TowerType>(t));
        float range = stats.range;
        for (int lvl = 0; lvl <= kTowerMaxLevel; ++lvl) {
            for (int c = 0; c < kCells; ++c) {
                range_mask[t][lvl][c] = Bitboard128::zero();
                range_mask_expanded8[t][lvl][c] = Bitboard128::zero();
            }
        }
        for (int lvl = 1; lvl <= kTowerMaxLevel; ++lvl) {
            float r2 = range * range;
            for (int c = 0; c < kCells; ++c) {
                Bitboard128 mask = Bitboard128::zero();
                for (int target = 0; target < kCells; ++target) {
                    if (dist2[c][target] <= r2 + 1e-6f) {
                        mask |= cell_bb[target];
                    }
                }
                range_mask[t][lvl][c] = mask;
                range_mask_expanded8[t][lvl][c] = expand_mask_8_neighbors(mask, x, y, cell_bb);
            }
            range *= 1.1f;
        }
    }

    for (int id = 0; id < kActionSpaceSize; ++id) {
        action_type[id] = ActionType::Wait1;
        action_x[id] = -1;
        action_y[id] = -1;
        action_wait[id] = 1;
    }

    for (int t = 0; t < kBuildTypes; ++t) {
        for (int c = 0; c < kCells; ++c) {
            int id = kFlatBuildOffset + t * kCells + c;
            build_action[t][c] = id;
            action_type[id] = static_cast<ActionType>(t);
            action_x[id] = x[c];
            action_y[id] = y[c];
        }
    }

    for (int c = 0; c < kCells; ++c) {
        int up = kFlatUpgradeOffset + c;
        int sell = kFlatSellOffset + c;
        upgrade_action[c] = up;
        sell_action[c] = sell;

        action_type[up] = ActionType::Upgrade;
        action_x[up] = x[c];
        action_y[up] = y[c];

        action_type[sell] = ActionType::Sell;
        action_x[sell] = x[c];
        action_y[sell] = y[c];
    }

    wait_action = kFlatWaitOffset;
    action_type[wait_action] = ActionType::Wait1;
    action_x[wait_action] = -1;
    action_y[wait_action] = -1;
    action_wait[wait_action] = 1;
}

const BoardTables& board_tables() {
    static const BoardTables tables;
    return tables;
}

} // namespace tdmz
