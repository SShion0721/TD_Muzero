#pragma once
#include <cstddef>
#include <utility>
#include <vector>

namespace tdmz {

struct EnemySpec {
    float hp;
    float speed;
    int reward;
};

struct Enemy {
    int id;
    float x;
    float y;
    float target_x;
    float target_y;

    float hp;
    float max_hp;
    float speed;
    float base_speed;
    float slow_timer;
    int reward;

    // The path storage remains stable while path_cursor advances. Avoiding
    // erase(path.begin()) removes an O(path_length) shift at every waypoint.
    std::vector<std::pair<int, int>> path;
    std::size_t path_cursor = 0;

    Enemy(int id, float x, float y, float hp, float speed, int reward);

    void apply_slow(float factor, float duration);
    void set_path(const std::vector<std::pair<int, int>>& new_path);
    void step(float dt);
};

} // namespace tdmz
