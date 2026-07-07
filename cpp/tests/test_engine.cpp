#include "tdmz/core/engine.hpp"
#include "tdmz/core/action.hpp"
#include "tdmz/core/observation.hpp"
#include <iostream>
#include <cassert>

using namespace tdmz;

void test_action_encode_decode() {
    Action a{ActionType::BuildAOE, 5, 5, 1};
    int id = encode_action(a);
    Action a2 = decode_action(id);
    assert(a == a2);
    
    Action w{ActionType::Wait1, -1, -1, 1};
    assert(decode_action(encode_action(w)) == w);
}

void test_engine_initialization() {
    TDEngine env(11, 11, 0);
    assert(env.money() == 200);
    assert(env.base_hp() == 100);
    assert(env.wave() == 1);
}

void test_engine_step() {
    TDEngine env(11, 11, 0);
    int flat_action = encode_action(Action{ActionType::BuildBasic, 1, 1, 1});
    auto result = env.step_action(flat_action);
    assert(env.money() == 100);
    assert(env.towers().size() == 1);
}

void test_observation_v1_not_empty() {
    TDEngine env(11, 11, 0);
    auto obs = make_observation_v1(env);
    assert(obs.size() == OBS_CHANNELS * kBoardW * kBoardH);

    float sum = 0.0f;
    for (float v : obs) sum += std::abs(v);
    assert(sum > 0.0f);
}

void test_legal_action_mask_size() {
    TDEngine env(11, 11, 0);
    auto legal = env.legal_actions();
    auto mask = env.legal_action_mask();

    assert(mask.size() == kActionSpaceSize);
    for (int a : legal) {
        assert(mask[a] == 1);
    }
}

int main() {
    test_action_encode_decode();
    test_engine_initialization();
    test_engine_step();
    test_observation_v1_not_empty();
    test_legal_action_mask_size();
    std::cout << "All engine tests passed!" << std::endl;
    return 0;
}
