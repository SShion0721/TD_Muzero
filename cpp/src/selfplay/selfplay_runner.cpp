#include "tdmz/selfplay/selfplay_runner.hpp"
#include "tdmz/core/observation.hpp"
#include "tdmz/mcts/mcts.hpp"
#include <stdexcept>
#include <utility>

namespace tdmz {

SelfPlayRunner::SelfPlayRunner(SelfPlayConfig cfg) : cfg_(cfg) {}

GameHistory SelfPlayRunner::run(INetworkEvaluator& evaluator) const {
    TDEngine env(11, 11, cfg_.seed);
    return run(env, evaluator);
}

GameHistory SelfPlayRunner::run(TDEngine& env, INetworkEvaluator& evaluator) const {
    GameHistory history;
    history.seed = cfg_.seed;
    history.max_steps = cfg_.max_steps;

    MCTSConfig mcts_cfg = cfg_.mcts;
    mcts_cfg.random_seed = cfg_.seed;
    MCTS mcts(mcts_cfg);

    for (int step = 0; step < cfg_.max_steps; ++step) {
        if (cfg_.stop_on_terminal && env.game_over()) {
            history.terminal = true;
            break;
        }

        auto observation = make_observation_v1(env);
        auto legal = env.legal_actions();
        if (legal.empty()) {
            throw std::runtime_error("Self-play encountered a state with no legal actions");
        }
        auto legal_mask = env.legal_action_mask();

        TrajectoryStep record;
        record.step_index = step;
        record.money = env.money();
        record.base_hp = env.base_hp();
        record.wave = env.wave();
        record.time = env.time();

        auto search = mcts.search_single(evaluator, observation, legal);
        if (search.action < 0) {
            throw std::runtime_error("MCTS returned an invalid action");
        }

        StepResult result = env.step_action(search.action);

        record.action = search.action;
        record.reward = result.reward;
        record.root_value = search.root_value;
        record.done = result.done;
        record.policy_target = search.policy_full;
        if (cfg_.save_observations) record.observation = std::move(observation);
        if (cfg_.save_legal_mask) record.legal_mask = std::move(legal_mask);

        history.total_reward += result.reward;
        history.steps.push_back(std::move(record));

        if (result.done) {
            history.terminal = true;
            break;
        }
    }

    return history;
}

} // namespace tdmz
