#pragma once

namespace tdmz {

constexpr int kTowerMaxLevel = 5;

enum class TowerType : int {
    Basic = 0,
    Sniper = 1,
    AOE = 2,
    Slow = 3
};

struct TowerStats {
    int cost;
    float damage;
    float range;
    float cooldown;
    float aoe_radius;
    float slow_factor;
    float slow_duration;
};

TowerStats tower_stats(TowerType type);

struct Tower {
    int x;
    int y;
    TowerType type;

    int cost;
    float damage;
    float range;
    float cooldown_max;
    float aoe_radius;
    float slow_factor;
    float slow_duration;

    float cooldown;
    int level;
    int total_spent;
    int upgrade_cost;

    Tower(int x, int y, TowerType type);

    bool can_upgrade() const;
    void upgrade();
    void step(float dt);
    bool can_shoot() const;
    void shoot();
};

} // namespace tdmz
