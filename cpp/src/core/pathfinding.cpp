#include "tdmz/core/pathfinding.hpp"
#include "tdmz/core/board_tables.hpp"
#include <algorithm>
#include <array>

namespace tdmz {

std::vector<std::pair<int, int>> find_shortest_path(
    int start_x, int start_y, int end_x, int end_y,
    const std::array<std::array<int, kBoardW>, kBoardH>& grid
) {
    if (!valid_cell_xy(start_x, start_y) || !valid_cell_xy(end_x, end_y)) {
        return {};
    }

    const auto& tables = board_tables();
    const int start = cell_id(start_x, start_y);
    const int goal = cell_id(end_x, end_y);

    if (start == goal) {
        return {};
    }

    constexpr int INF = 1000000000;
    std::array<int, kCells> dist;
    std::array<int, kCells> parent;
    std::array<int, kCells> queue;
    dist.fill(INF);
    parent.fill(-1);

    int head = 0;
    int tail = 0;
    queue[tail++] = start;
    dist[start] = 0;

    while (head < tail) {
        int cur = queue[head++];
        if (cur == goal) break;

        int cnt = tables.neighbor4_count[cur];
        for (int i = 0; i < cnt; ++i) {
            int next = tables.neighbors4[cur][i];
            int nx = tables.x[next];
            int ny = tables.y[next];

            if (grid[ny][nx] != 0) {
                continue;
            }
            if (dist[next] != INF) {
                continue;
            }
            dist[next] = dist[cur] + 1;
            parent[next] = cur;
            queue[tail++] = next;
        }
    }

    if (dist[goal] == INF) {
        return {};
    }

    std::vector<std::pair<int, int>> path;
    int cur = goal;
    while (cur != start && cur >= 0) {
        path.push_back({tables.x[cur], tables.y[cur]});
        cur = parent[cur];
    }
    std::reverse(path.begin(), path.end());
    return path;
}

} // namespace tdmz
