#include "tdmz/core/action.hpp"
#include "tdmz/core/board_tables.hpp"
#include "tdmz/core/pathfinding.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace tdmz;

static void check_true(bool ok, const char* expr, int line) {
    if (!ok) throw std::runtime_error(std::string("check failed at line ") + std::to_string(line) + ": " + expr);
}
#define CHECK_TRUE(x) check_true(static_cast<bool>(x), #x, __LINE__)

void test_bitboard_basics() {
    Bitboard128 bb = Bitboard128::zero();
    bb.set(0);
    bb.set(64);
    bb.set(120);
    CHECK_TRUE(bb.test(0));
    CHECK_TRUE(bb.test(64));
    CHECK_TRUE(bb.test(120));
    CHECK_TRUE(bb.popcount() == 3);
    CHECK_TRUE(bb.pop_lsb() == 0);
    bb.clear(64);
    CHECK_TRUE(!bb.test(64));
    CHECK_TRUE(bb.popcount() == 1);
}

void test_board_geometry_tables() {
    const auto& t = board_tables();
    CHECK_TRUE(t.board_mask.popcount() == kCells);
    CHECK_TRUE(t.x[0] == 0 && t.y[0] == 0);
    CHECK_TRUE(t.x[120] == 10 && t.y[120] == 10);
    CHECK_TRUE(t.neighbor4_count[cell_id(0, 0)] == 2);
    CHECK_TRUE(t.neighbors4[cell_id(0, 0)][0] == cell_id(0, 1));
    CHECK_TRUE(t.neighbors4[cell_id(0, 0)][1] == cell_id(1, 0));
    CHECK_TRUE(t.neighbor4_count[cell_id(5, 5)] == 4);
    CHECK_TRUE(t.manhattan[cell_id(0, 5)][cell_id(10, 5)] == 10);
    CHECK_TRUE(t.dist2[cell_id(0, 0)][cell_id(3, 4)] == 25.0f);
}

void test_action_tables_roundtrip() {
    for (int id = 0; id < kActionSpaceSize; ++id) {
        Action a = decode_action(id);
        CHECK_TRUE(encode_action(a) == id);
    }
    const auto& t = board_tables();
    CHECK_TRUE(t.wait_action == kFlatWaitOffset);
    CHECK_TRUE(t.build_action[0][cell_id(3, 7)] == encode_action(Action{ActionType::BuildBasic, 3, 7, 1}));
    CHECK_TRUE(t.upgrade_action[cell_id(2, 4)] == encode_action(Action{ActionType::Upgrade, 2, 4, 1}));
    CHECK_TRUE(t.sell_action[cell_id(9, 8)] == encode_action(Action{ActionType::Sell, 9, 8, 1}));
}

void test_range_masks() {
    const auto& t = board_tables();
    int center = cell_id(5, 5);
    auto basic_l1 = t.range_mask[0][1][center];
    auto sniper_l1 = t.range_mask[1][1][center];
    auto basic_l5 = t.range_mask[0][kTowerMaxLevel][center];
    CHECK_TRUE(basic_l1.test(center));
    CHECK_TRUE(sniper_l1.popcount() > basic_l1.popcount());
    CHECK_TRUE(basic_l5.popcount() >= basic_l1.popcount());
}

void test_fixed_bfs_pathfinding() {
    std::array<std::array<int, kBoardW>, kBoardH> grid;
    for (auto& row : grid) row.fill(0);
    auto direct = find_shortest_path(0, 5, 10, 5, grid);
    CHECK_TRUE(direct.size() == 10);
    CHECK_TRUE(direct.front().first == 1 && direct.front().second == 5);
    CHECK_TRUE(direct.back().first == 10 && direct.back().second == 5);
    for (int y = 0; y < kBoardH; ++y) grid[y][5] = 1;
    auto no_path = find_shortest_path(0, 5, 10, 5, grid);
    CHECK_TRUE(no_path.empty());
}

int main() {
    test_bitboard_basics();
    test_board_geometry_tables();
    test_action_tables_roundtrip();
    test_range_masks();
    test_fixed_bfs_pathfinding();
    std::cout << "Board table tests passed!" << std::endl;
    return 0;
}
