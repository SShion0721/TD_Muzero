#include "tdmz/core/engine.hpp"
#include <iostream>
using namespace tdmz;
int main() {
    TDEngine env(11, 11, 0);
    env.place_tower(9, 3, TowerType::AOE);
    auto p = env.find_path(env.spawn_x(), env.spawn_y(), env.base_x(), env.base_y());
    std::cout << "C++ Path: ";
    for(auto t : p) std::cout << "(" << t.first << ", " << t.second << ") ";
    std::cout << std::endl;
}
