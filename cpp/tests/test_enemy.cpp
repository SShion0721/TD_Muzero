#include "tdmz/core/enemy.hpp"
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tdmz;

namespace {

void check_true(bool ok, const char* expr, int line) {
    if (!ok) {
        throw std::runtime_error(
            std::string("check failed at line ") + std::to_string(line) + ": " + expr
        );
    }
}

void check_close(float actual, float expected, float eps, const char* expr, int line) {
    if (std::fabs(actual - expected) > eps) {
        throw std::runtime_error(
            std::string("check failed at line ") + std::to_string(line) + ": " + expr +
            " actual=" + std::to_string(actual) + " expected=" + std::to_string(expected)
        );
    }
}

#define CHECK_TRUE(x) check_true(static_cast<bool>(x), #x, __LINE__)
#define CHECK_CLOSE(a, b, eps) check_close((a), (b), (eps), #a " ~= " #b, __LINE__)

std::vector<std::pair<int, int>> straight_path(int last_x) {
    std::vector<std::pair<int, int>> path;
    for (int x = 1; x <= last_x; ++x) path.push_back({x, 0});
    return path;
}

void test_fast_enemy_consumes_multiple_waypoints() {
    Enemy enemy(1, 0.0f, 0.0f, 100.0f, 2.8f, 1);
    enemy.set_path(straight_path(5));
    const std::size_t original_size = enemy.path.size();

    enemy.step(1.0f);

    CHECK_CLOSE(enemy.x, 2.8f, 1e-5f);
    CHECK_CLOSE(enemy.y, 0.0f, 1e-5f);
    CHECK_TRUE(enemy.path_cursor == 2);
    CHECK_TRUE(enemy.path.size() == original_size);
    CHECK_CLOSE(enemy.target_x, 3.0f, 1e-5f);
}

void test_regular_enemy_moves_fractional_distance() {
    Enemy enemy(2, 0.0f, 0.0f, 100.0f, 1.5f, 1);
    enemy.set_path(straight_path(5));

    enemy.step(1.0f);

    CHECK_CLOSE(enemy.x, 1.5f, 1e-5f);
    CHECK_TRUE(enemy.path_cursor == 1);
    CHECK_CLOSE(enemy.target_x, 2.0f, 1e-5f);
}

void test_slow_expiry_splits_one_tick() {
    Enemy enemy(3, 0.0f, 0.0f, 100.0f, 2.0f, 1);
    enemy.set_path(straight_path(5));
    enemy.apply_slow(0.5f, 0.25f);

    enemy.step(1.0f);

    // 0.25 seconds at speed 1.0, then 0.75 seconds at base speed 2.0.
    CHECK_CLOSE(enemy.x, 1.75f, 1e-5f);
    CHECK_CLOSE(enemy.speed, 2.0f, 1e-5f);
    CHECK_CLOSE(enemy.slow_timer, 0.0f, 1e-5f);
}

void test_set_path_resets_cursor() {
    Enemy enemy(4, 0.0f, 0.0f, 100.0f, 3.0f, 1);
    enemy.set_path(straight_path(5));
    enemy.step(1.0f);
    CHECK_TRUE(enemy.path_cursor == 3);

    enemy.set_path({{4, 0}, {5, 0}});

    CHECK_TRUE(enemy.path_cursor == 0);
    CHECK_CLOSE(enemy.target_x, 4.0f, 1e-5f);
}

void test_enemy_stops_at_path_end() {
    Enemy enemy(5, 0.0f, 0.0f, 100.0f, 10.0f, 1);
    enemy.set_path({{1, 0}, {2, 0}});

    enemy.step(1.0f);

    CHECK_CLOSE(enemy.x, 2.0f, 1e-5f);
    CHECK_CLOSE(enemy.y, 0.0f, 1e-5f);
    CHECK_TRUE(enemy.path_cursor == enemy.path.size());
    CHECK_CLOSE(enemy.target_x, enemy.x, 1e-5f);
    CHECK_CLOSE(enemy.target_y, enemy.y, 1e-5f);
}

} // namespace

int main() {
    try {
        test_fast_enemy_consumes_multiple_waypoints();
        test_regular_enemy_moves_fractional_distance();
        test_slow_expiry_splits_one_tick();
        test_set_path_resets_cursor();
        test_enemy_stops_at_path_end();
        std::cout << "Enemy movement tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
