#include "tdmz/core/action.hpp"
#include <stdexcept>
#include <sstream>

namespace tdmz {

Action decode_action(int id) {
    Action a;
    a.flat_id = id;

    if (id < kFlatUpgradeOffset) {
        int tower_type = id / kCells;
        int rem = id % kCells;
        a.type = static_cast<ActionType>(tower_type);
        a.y = rem / kBoardW;
        a.x = rem % kBoardW;
        a.wait_steps = 1;
        return a;
    }

    if (id < kFlatSellOffset) {
        int rem = id - kFlatUpgradeOffset;
        a.type = ActionType::Upgrade;
        a.y = rem / kBoardW;
        a.x = rem % kBoardW;
        a.wait_steps = 1;
        return a;
    }

    if (id < kFlatWaitOffset) {
        int rem = id - kFlatSellOffset;
        a.type = ActionType::Sell;
        a.y = rem / kBoardW;
        a.x = rem % kBoardW;
        a.wait_steps = 1;
        return a;
    }

    int wait_id = id - kFlatWaitOffset;
    if (wait_id != 0) {
        throw std::invalid_argument("Invalid flat action id");
    }
    
    a.type = ActionType::Wait1;
    a.wait_steps = 1;
    a.x = -1;
    a.y = -1;
    return a;
}

int encode_action(const Action& a) {
    if (is_build(a.type)) {
        return static_cast<int>(a.type) * kCells + a.y * kBoardW + a.x;
    }
    if (a.type == ActionType::Upgrade) {
        return kFlatUpgradeOffset + a.y * kBoardW + a.x;
    }
    if (a.type == ActionType::Sell) {
        return kFlatSellOffset + a.y * kBoardW + a.x;
    }
    
    // Wait actions
    if (a.type == ActionType::Wait1) return kFlatWaitOffset + 0;
    
    throw std::invalid_argument("Invalid action type for encoding");
}

bool is_build(ActionType t) {
    return t == ActionType::BuildBasic || 
           t == ActionType::BuildSniper || 
           t == ActionType::BuildAOE || 
           t == ActionType::BuildSlow;
}

bool is_wait(ActionType t) {
    return t == ActionType::Wait1;
}

std::string action_to_string(const Action& a) {
    std::stringstream ss;
    if (is_build(a.type)) {
        std::string t_str = "";
        if (a.type == ActionType::BuildBasic) t_str = "BASIC";
        else if (a.type == ActionType::BuildSniper) t_str = "SNIPER";
        else if (a.type == ActionType::BuildAOE) t_str = "AOE";
        else if (a.type == ActionType::BuildSlow) t_str = "SLOW";
        ss << "Build " << t_str << " at (" << a.x << ", " << a.y << ")";
    } else if (a.type == ActionType::Upgrade) {
        ss << "Upgrade Tower at (" << a.x << ", " << a.y << ")";
    } else if (a.type == ActionType::Sell) {
        ss << "Sell Tower at (" << a.x << ", " << a.y << ")";
    } else if (is_wait(a.type)) {
        ss << "Wait " << a.wait_steps << " steps";
    } else {
        ss << "UNKNOWN";
    }
    return ss.str();
}

} // namespace tdmz
