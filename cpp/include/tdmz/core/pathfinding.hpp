#pragma once
#include <vector>
#include <utility>
#include <array>

#include "tdmz/core/action.hpp"

namespace tdmz {

inline int manhattan(int x1, int y1, int x2, int y2) {
    return std::abs(x1 - x2) + std::abs(y1 - y2);
}

std::vector<std::pair<int, int>> find_shortest_path(
    int start_x, int start_y, int end_x, int end_y,
    const std::array<std::array<int, kBoardW>, kBoardH>& grid
);

} // namespace tdmz
