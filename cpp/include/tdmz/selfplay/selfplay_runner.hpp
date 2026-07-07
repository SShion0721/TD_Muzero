#pragma once
#include "tdmz/core/engine.hpp"
#include "tdmz/mcts/network_evaluator.hpp"
#include "tdmz/selfplay/game_history.hpp"
#include "tdmz/selfplay/selfplay_config.hpp"

namespace tdmz {

class SelfPlayRunner {
public:
    explicit SelfPlayRunner(SelfPlayConfig cfg = SelfPlayConfig{});

    GameHistory run(INetworkEvaluator& evaluator) const;
    GameHistory run(TDEngine& env, INetworkEvaluator& evaluator) const;

private:
    SelfPlayConfig cfg_;
};

} // namespace tdmz
