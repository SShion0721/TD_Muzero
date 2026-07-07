#include "tdmz/core/engine.hpp"
#include "tdmz/core/action.hpp"
#include "tdmz/core/observation.hpp"
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

using namespace tdmz;

namespace {

void check(bool condition, const char* expr, const char* file, int line) {
    if (!condition) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                                 " CHECK failed: " + expr);
    }
}

#define CHECK(expr) check(static_cast<bool>(expr), #expr, __FILE__, __LINE__)

template <typename Fn>
void check_throws_invalid_argument(Fn&& fn) {
    bool threw = false;
    try {
        fn();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

} // namespace

void test_action_encode_decode() {
    Action a{ActionType::BuildAOE, 5, 5, 1};
    int id = encode_action(a);
    Action a2 = decode_action(id);
    CHECK(a == a2);
    CHECK(a2.flat_id == id);

    Action w{ActionType::Wait1, -1, -1, 1};
    Action w2 = decode_action(encode_action(w));
    CHECK(w == w2);
    CHECK(w2.flat_id == kFlatWaitOffset);
}

void test_action_invalid_inputs() {
    check_throws_invalid_argument([] { decode_action(-1); });
    check_throws_invalid_argument([] { decode_action(kActionSpaceSize); });
    check_throws_invalid_argument([] { encode_action(Action{ActionType::BuildBasic, -1, 0, 1}); });
    check_throws_invalid_argument([] { encode_action(Action{ActionType::Upgrade, kBoardW, 0, 1}); });
    check_throws_invalid_argument([] { encode_action(Action{ActionType::Sell, 0, kBoardH, 1}); });
}

void test_engine_initialization() {
    TDEngine env(11, 11, 0);
    CHECK(env.money() == 200);
    CHECK(env.base_hp() == 100);
    CHECK(env.wave() == 1);
}

void test_engine_step() {
    TDEngine env(11, 11, 0);
    int flat_action = encode_action(Action{ActionType::BuildBasic, 1, 1, 1});
    auto result = env.step_action(flat_action);
    CHECK(!result.done);
    CHECK(env.money() == 150);
    CHECK(env.towers().size() == 1);
}

void test_observation_v1_not_empty() {
    TDEngine env(11, 11, 0);
    auto obs = make_observation_v1(env);
    CHECK(obs.size() == OBS_CHANNELS * kBoardW * kBoardH);

    float sum = 0.0f;
    for (float v : obs) sum += std::abs(v);
    CHECK(sum > 0.0f);
}

void test_observation_distance_channel() {
    TDEngine env(11, 11, 0);
    auto obs = make_observation_v1(env);
    auto at = [&](int c, int y, int x) -> float {
        return obs[(c * kBoardH + y) * kBoardW + x];
    };

    CHECK(at(CH_DISTANCE_TO_BASE, env.base_y(), env.base_x()) == 0.0f);
    CHECK(at(CH_DISTANCE_TO_BASE, env.spawn_y(), env.spawn_x()) > 0.0f);
}

void test_legal_action_mask_size() {
    TDEngine env(11, 11, 0);
    auto legal = env.legal_actions();
    auto mask = env.legal_action_mask();

    CHECK(mask.size() == kActionSpaceSize);

    int enabled = 0;
    for (int value : mask) enabled += value;
    CHECK(enabled == static_cast<int>(legal.size()));

    for (int a : legal) {
        CHECK(a >= 0);
        CHECK(a < kActionSpaceSize);
        CHECK(mask[a] == 1);
    }
}

int main() {
    test_action_encode_decode();
    test_action_invalid_inputs();
    test_engine_initialization();
    test_engine_step();
    test_observation_v1_not_empty();
    test_observation_distance_channel();
    test_legal_action_mask_size();
    std::cout << "All engine tests passed!" << std::endl;
    return 0;
}
