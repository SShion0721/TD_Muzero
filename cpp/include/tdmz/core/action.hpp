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
};

constexpr int kBuildTypes = 4;
constexpr int kActionTypes = 7;
constexpr int kSpatialPolicyPlanes = 6;
constexpr int kSpatialActionPlanes = kActionTypes;
constexpr int kFlatBuildOffset = 0;
constexpr int kFlatUpgradeOffset = kCells * 4;
constexpr int kFlatSellOffset = kCells * 5;
constexpr int kFlatWaitOffset = kCells * kSpatialPolicyPlanes;
constexpr int kWaitActions = 1;
constexpr int kActionSpaceSize = kCells * kSpatialPolicyPlanes + kWaitActions;

static_assert(kSpatialActionPlanes == 7, "Spatial action plane count changed");
static_assert(kSpatialPolicyPlanes == 6, "Spatial policy plane count changed");
static_assert(kFlatWaitOffset == 726, "Action-space ABI changed");
static_assert(kActionSpaceSize == 727, "Action-space ABI changed");

struct Action {
    ActionType type = ActionType::Wait1;
    int x = -1;
    int y = -1;
    int wait_steps = 1;
    int flat_id = -1;

    // Semantic equality only. flat_id is a cached encoding/debug field and is
    // intentionally ignored so manually constructed actions compare equal to
    // decoded actions with the same meaning.
    bool operator==(const Action& other) const {
        return type == other.type &&
               x == other.x &&
               y == other.y &&
               wait_steps == other.wait_steps;
    }
};

Action decode_action(int flat_id);
int encode_action(const Action& a);
bool is_build(ActionType t);
bool is_wait(ActionType t);
std::string action_to_string(const Action& a);

} // namespace tdmz
