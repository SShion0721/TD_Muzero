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

    // Advances the cooldown clock inside a half-open time window [0, dt).
    // remaining_dt is reduced to the instant at which the tower becomes ready.
    // A cooldown expiring exactly at the end of the window is carried as zero
    // into the next tick and does not grant an extra shot in the current tick.
    bool advance_until_ready(float& remaining_dt);
};

} // namespace tdmz
