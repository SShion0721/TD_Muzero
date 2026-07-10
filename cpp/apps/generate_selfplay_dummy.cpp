#include "tdmz/mcts/dummy_network.hpp"
#include "tdmz/selfplay/selfplay_runner.hpp"
#include "tdmz/selfplay/trajectory_writer.hpp"
#include <iostream>

using namespace tdmz;

int main() {
    SelfPlayConfig cfg;
    cfg.seed = 0;
    cfg.max_steps = 64;
    cfg.mcts.num_simulations = 32;
    cfg.mcts.latent_top_k = 16;
    cfg.mcts.max_nodes = 8192;

    DummyNetwork net;
    SelfPlayRunner runner(cfg);
    auto history = runner.run(net);

    write_history_jsonl(history, "selfplay_dummy.jsonl");
    write_history_binary(history, "selfplay_dummy.tdmzspb");
    write_history_summary_json(history, "selfplay_dummy.summary.json");

    std::cout << "steps=" << history.steps.size() << " total_reward=" << history.total_reward << std::endl;
    return 0;
}
