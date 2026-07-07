#include "tdmz/core/enemy.hpp"
#include <cmath>
#include <algorithm>

namespace tdmz {

Enemy::Enemy(int id, float x, float y, float hp, float speed, int reward)
    : id(id), x(x), y(y), target_x(x), target_y(y),
      hp(hp), max_hp(hp), speed(speed), base_speed(speed),
      slow_timer(0.0f), reward(reward) {
}

void Enemy::apply_slow(float factor, float duration) {
    if (factor < 1.0f) {
        speed = base_speed * factor;
        slow_timer = std::max(slow_timer, duration);
    }
}

void Enemy::set_path(const std::vector<std::pair<int, int>>& new_path) {
    path = new_path;
    if (!path.empty()) {
        target_x = static_cast<float>(path.front().first);
        target_y = static_cast<float>(path.front().second);
    }
}

void Enemy::step(float dt) {
    if (slow_timer > 0.0f) {
        slow_timer -= dt;
        if (slow_timer <= 0.0f) {
            speed = base_speed;
        }
    }
    
    if (path.empty()) {
        return;
    }
    
    float dx = target_x - x;
    float dy = target_y - y;
    float dist = std::hypot(dx, dy);
    float move_dist = speed * dt;

    if (dist <= move_dist) {
        x = target_x;
        y = target_y;
        path.erase(path.begin());
        if (!path.empty()) {
            target_x = static_cast<float>(path.front().first);
            target_y = static_cast<float>(path.front().second);
        }
    } else {
        x += (dx / dist) * move_dist;
        y += (dy / dist) * move_dist;
    }
}

} // namespace tdmz
