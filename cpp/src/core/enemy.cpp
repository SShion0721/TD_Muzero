#include "tdmz/core/enemy.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace tdmz {

Enemy::Enemy(int id, float x, float y, float hp, float speed, int reward)
    : id(id), x(x), y(y), target_x(x), target_y(y),
      hp(hp), max_hp(hp), speed(speed), base_speed(speed),
      slow_timer(0.0f), reward(reward) {
}

void Enemy::apply_slow(float factor, float duration) {
    if (factor < 1.0f && duration > 0.0f) {
        const float slowed_speed = base_speed * std::max(0.0f, factor);
        speed = slow_timer > 0.0f ? std::min(speed, slowed_speed) : slowed_speed;
        slow_timer = std::max(slow_timer, duration);
    }
}

void Enemy::set_path(const std::vector<std::pair<int, int>>& new_path) {
    path = new_path;
    if (!path.empty()) {
        target_x = static_cast<float>(path.front().first);
        target_y = static_cast<float>(path.front().second);
    } else {
        target_x = x;
        target_y = y;
    }
}

void Enemy::step(float dt) {
    if (dt <= 0.0f) {
        return;
    }

    constexpr float kMoveEpsilon = 1e-6f;

    auto advance_distance = [&](float move_dist) {
        std::size_t consumed = 0;
        while (move_dist > kMoveEpsilon && consumed < path.size()) {
            const float next_x = static_cast<float>(path[consumed].first);
            const float next_y = static_cast<float>(path[consumed].second);
            const float dx = next_x - x;
            const float dy = next_y - y;
            const float dist = std::hypot(dx, dy);

            if (dist <= kMoveEpsilon) {
                x = next_x;
                y = next_y;
                ++consumed;
                continue;
            }

            if (dist <= move_dist + kMoveEpsilon) {
                x = next_x;
                y = next_y;
                move_dist = std::max(0.0f, move_dist - dist);
                ++consumed;
            } else {
                x += (dx / dist) * move_dist;
                y += (dy / dist) * move_dist;
                move_dist = 0.0f;
            }
        }

        if (consumed > 0) {
            path.erase(path.begin(), path.begin() + static_cast<std::ptrdiff_t>(consumed));
        }
        if (!path.empty()) {
            target_x = static_cast<float>(path.front().first);
            target_y = static_cast<float>(path.front().second);
        } else {
            target_x = x;
            target_y = y;
        }
    };

    float remaining_time = dt;
    if (slow_timer > 0.0f) {
        const float slowed_time = std::min(remaining_time, slow_timer);
        advance_distance(std::max(0.0f, speed) * slowed_time);
        slow_timer -= slowed_time;
        remaining_time -= slowed_time;

        if (slow_timer <= kMoveEpsilon) {
            slow_timer = 0.0f;
            speed = base_speed;
        }
    }

    if (remaining_time > kMoveEpsilon) {
        advance_distance(std::max(0.0f, speed) * remaining_time);
    }
}

} // namespace tdmz
