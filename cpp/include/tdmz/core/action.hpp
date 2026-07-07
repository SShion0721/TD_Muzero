#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace tdmz {

constexpr int kBoardW = 11;
constexpr int kBoardH = 11;
constexpr int kCells = kBoardW * kBoardH;

enum class ActionType : int {
    BuildBasic = 0,
    BuildSniper = 1,
    BuildAOE = 2,
    BuildSlow = 3,
    Upgrade = 4,
    Sell = 5,
    Wait1 = 6,
    Wait2 = 7,
    Wait4 = 8,
    Wait8 = 9,
};

constexpr int kBuildTypes = 4;
constexpr int kActionTypes = 10;
constexpr int kFlatBuildOffset = 0;
constexpr int kFlatUpgradeOffset = kCells * 4;
constexpr int kFlatSellOffset = kCells * 5;
constexpr int kFlatWaitOffset = kCells * 6;
constexpr int kWaitActions = 4;
constexpr int kActionSpaceSize = kCells * 6 + kWaitActions; // 121 * 6 + 4 = 726 + 4 = 730 actions

struct Action {
    ActionType type;
    int x = -1;
    int y = -1;
    int wait_steps = 1;
    int flat_id = -1;

    bool operator==(const Action& other) const {
        return type == other.type && x == other.x && y == other.y && wait_steps == other.wait_steps && flat_id == other.flat_id;
    }
};

Action decode_action(int flat_id);
int encode_action(const Action& a);
bool is_build(ActionType t);
bool is_wait(ActionType t);
std::string action_to_string(const Action& a);

} // namespace tdmz
