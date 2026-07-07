#include "tdmz/core/pathfinding.hpp"
#include <queue>
#include <tuple>
#include <map>
#include <algorithm>

namespace tdmz {

std::vector<std::pair<int, int>> find_shortest_path(
    int start_x, int start_y, int end_x, int end_y,
    const std::array<std::array<int, kBoardW>, kBoardH>& grid
) {
    using Node = std::pair<int, int>;
    using PQElement = std::pair<int, Node>;

    std::priority_queue<PQElement, std::vector<PQElement>, std::greater<PQElement>> open_set;
    std::map<Node, Node> came_from;
    
    std::array<std::array<int, kBoardW>, kBoardH> g_score;
    for (auto& row : g_score) row.fill(1000000000);
    
    open_set.push({manhattan(start_x, start_y, end_x, end_y), {start_x, start_y}});
    g_score[start_y][start_x] = 0;

    const int dx[] = {0, 1, 0, -1};
    const int dy[] = {1, 0, -1, 0};

    while (!open_set.empty()) {
        auto [current_f, current] = open_set.top();
        open_set.pop();

        if (current.first == end_x && current.second == end_y) {
            std::vector<Node> path;
            auto curr = current;
            while (came_from.count(curr)) {
                path.push_back(curr);
                curr = came_from[curr];
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        int cx = current.first;
        int cy = current.second;

        for (int i = 0; i < 4; ++i) {
            int nx = cx + dx[i];
            int ny = cy + dy[i];

            if (nx >= 0 && nx < kBoardW && ny >= 0 && ny < kBoardH && grid[ny][nx] == 0) {
                int tentative_g = g_score[cy][cx] + 1;
                if (tentative_g < g_score[ny][nx]) {
                    came_from[{nx, ny}] = current;
                    g_score[ny][nx] = tentative_g;
                    int f = tentative_g + manhattan(nx, ny, end_x, end_y);
                    open_set.push({f, {nx, ny}});
                }
            }
        }
    }

    return {}; // Empty means no path
}

} // namespace tdmz
