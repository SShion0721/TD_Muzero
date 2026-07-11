#include "tdmz/core/engine.hpp"
#include "tdmz/core/pending_spawn_queue.hpp"
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

#define CHECK_TRUE(x) check_true(static_cast<bool>(x), #x, __LINE__)

bool near_float(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

float pending_hp(const PendingSpawnQueue& queue) {
    float total = 0.0f;
    for (const auto& enemy : queue) total += enemy.hp;
    return total;
}

void test_front_consumption_advances_cursor_without_shifting_storage() {
    PendingSpawnQueue queue;
    queue.reserve(16);
    queue = std::vector<EnemySpec>{{10.0f, 1.0f, 1}, {20.0f, 2.0f, 2}, {30.0f, 3.0f, 3}};

    CHECK_TRUE(queue.size() == 3);
    CHECK_TRUE(queue.storage_size() == 3);
    CHECK_TRUE(queue.consumed_count() == 0);
    CHECK_TRUE(near_float(pending_hp(queue), 60.0f));

    const EnemySpec* second_address = &*(queue.begin() + 1);
    queue.erase(queue.begin());

    CHECK_TRUE(queue.size() == 2);
    CHECK_TRUE(queue.storage_size() == 3);
    CHECK_TRUE(queue.consumed_count() == 1);
    CHECK_TRUE(&queue.front() == second_address);
    CHECK_TRUE(near_float(queue.front().hp, 20.0f));
    CHECK_TRUE(near_float(pending_hp(queue), 50.0f));

    queue.erase(queue.begin());
    queue.erase(queue.begin());
    CHECK_TRUE(queue.empty());
    CHECK_TRUE(queue.size() == 0);
    CHECK_TRUE(queue.storage_size() == 3);
    CHECK_TRUE(queue.consumed_count() == 3);
}

void test_assignment_starts_a_new_wave_and_resets_cursor() {
    PendingSpawnQueue queue;
    queue = std::vector<EnemySpec>{{1.0f, 1.0f, 1}, {2.0f, 1.0f, 1}};
    queue.erase(queue.begin());
    CHECK_TRUE(queue.consumed_count() == 1);

    queue = std::vector<EnemySpec>{{7.0f, 1.0f, 1}, {8.0f, 1.0f, 1}, {9.0f, 1.0f, 1}};
    CHECK_TRUE(queue.consumed_count() == 0);
    CHECK_TRUE(queue.size() == 3);
    CHECK_TRUE(near_float(queue.front().hp, 7.0f));
    CHECK_TRUE(near_float(pending_hp(queue), 24.0f));
}

void test_non_front_erase_is_rejected() {
    PendingSpawnQueue queue;
    queue = std::vector<EnemySpec>{{1.0f, 1.0f, 1}, {2.0f, 1.0f, 1}};

    bool threw = false;
    try {
        queue.erase(queue.begin() + 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK_TRUE(threw);
    CHECK_TRUE(queue.size() == 2);
    CHECK_TRUE(queue.consumed_count() == 0);
}

void test_engine_pending_counts_preserve_previous_semantics() {
    TDEngine env(11, 11, 0);
    const int initial_count = env.enemies_to_spawn_count();
    const float initial_hp = env.pending_spawn_total_hp();

    CHECK_TRUE(initial_count == 9);
    CHECK_TRUE(initial_hp > 0.0f);

    const StepResult first_tick = env.step_wait(1);
    CHECK_TRUE(!first_tick.done);
    CHECK_TRUE(env.enemies_to_spawn_count() == initial_count - 1);
    CHECK_TRUE(env.pending_spawn_total_hp() < initial_hp);

    env.reset(0);
    CHECK_TRUE(env.enemies_to_spawn_count() == initial_count);
    CHECK_TRUE(near_float(env.pending_spawn_total_hp(), initial_hp));
}

} // namespace

int main() {
    try {
        test_front_consumption_advances_cursor_without_shifting_storage();
        test_assignment_starts_a_new_wave_and_resets_cursor();
        test_non_front_erase_is_rejected();
        test_engine_pending_counts_preserve_previous_semantics();
        std::cout << "Pending spawn queue tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
