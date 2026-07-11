#include "tdmz/core/tower.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tdmz {

TowerStats tower_stats(TowerType type) {
    switch (type) {
        case TowerType::Basic:
            return {50, 10.0f, 2.5f, 1.0f, 0.0f, 1.0f, 0.0f};
        case TowerType::Sniper:
            return {100, 50.0f, 5.5f, 3.0f, 0.0f, 1.0f, 0.0f};
        case TowerType::AOE:
            return {150, 15.0f, 2.0f, 1.5f, 2.0f, 1.0f, 0.0f};
        case TowerType::Slow:
            return {75, 2.0f, 3.0f, 1.0f, 0.0f, 0.4f, 2.0f};
    }
    throw std::invalid_argument("Invalid tower type");
}

Tower::Tower(int x, int y, TowerType type) : x(x), y(y), type(type) {
    auto stats = tower_stats(type);
    cost = stats.cost;
    damage = stats.damage;
    range = stats.range;
    cooldown_max = stats.cooldown;
    aoe_radius = stats.aoe_radius;
    slow_factor = stats.slow_factor;
    slow_duration = stats.slow_duration;

    cooldown = 0.0f;
    level = 1;
    total_spent = cost;
    upgrade_cost = static_cast<int>(cost * 1.5f);
}

bool Tower::can_upgrade() const {
    return level < kTowerMaxLevel;
}

void Tower::upgrade() {
    if (level >= kTowerMaxLevel) return;
    level += 1;
    total_spent += upgrade_cost;
    damage *= 1.5f;
    range *= 1.1f;
    cooldown_max = std::max(0.1f, cooldown_max * 0.9f);
    upgrade_cost = static_cast<int>(upgrade_cost * 1.5f);
}

void Tower::step(float dt) {
    if (!std::isfinite(dt) || dt < 0.0f) {
        throw std::invalid_argument("Tower step dt must be finite and non-negative");
    }
    if (cooldown > 0.0f) {
        cooldown = std::max(0.0f, cooldown - dt);
    }
}

bool Tower::can_shoot() const {
    return cooldown <= 0.0f;
}

void Tower::shoot() {
    cooldown = cooldown_max;
}

bool Tower::advance_until_ready(float& remaining_dt) {
    if (!std::isfinite(remaining_dt) || remaining_dt < 0.0f) {
        throw std::invalid_argument("Tower remaining dt must be finite and non-negative");
    }
    if (remaining_dt <= 0.0f) return false;

    if (cooldown <= 0.0f) {
        cooldown = 0.0f;
        return true;
    }

    // The tick interval is half-open. Expiring exactly at its end leaves the
    // tower ready for the next tick but does not create an extra boundary shot.
    if (cooldown >= remaining_dt) {
        cooldown = std::max(0.0f, cooldown - remaining_dt);
        remaining_dt = 0.0f;
        return false;
    }

    remaining_dt -= cooldown;
    cooldown = 0.0f;
    return true;
}

} // namespace tdmz
