#include "tdmz/mcts/dummy_network.hpp"
#include "tdmz/selfplay/selfplay_runner.hpp"
#include "tdmz/selfplay/trajectory_writer.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace tdmz;

static void check_true(bool ok, const char* expr, int line) {
    if (!ok) throw std::runtime_error(std::string("check failed at line ") + std::to_string(line) + ": " + expr);
}
#define CHECK_TRUE(x) check_true(static_cast<bool>(x), #x, __LINE__)

void test_selfplay_dummy_history() {
    SelfPlayConfig cfg;
    cfg.seed = 7;
    cfg.max_steps = 4;
    cfg.mcts.num_simulations = 8;
    cfg.mcts.latent_top_k = 8;
    cfg.mcts.max_nodes = 4096;

    DummyNetwork net;
    SelfPlayRunner runner(cfg);
    auto history = runner.run(net);

    CHECK_TRUE(history.seed == cfg.seed);
    CHECK_TRUE(history.wave_mode == WaveMode::Fixed);
    CHECK_TRUE(!history.steps.empty());
    CHECK_TRUE(history.steps.size() <= static_cast<size_t>(cfg.max_steps));
    CHECK_TRUE(history.completed());
    CHECK_TRUE(history.terminal != history.truncated);
    CHECK_TRUE(history.truncated);
    CHECK_TRUE(!history.terminal);
    CHECK_TRUE(!history.bootstrap_state.has_value());

    for (const auto& step : history.steps) {
        CHECK_TRUE(step.action >= 0);
        CHECK_TRUE(step.action < kActionSpaceSize);
        CHECK_TRUE(step.policy_target.size() == kActionSpaceSize);
        CHECK_TRUE(step.legal_mask.size() == kActionSpaceSize);
        CHECK_TRUE(step.observation.size() == OBS_CHANNELS * kBoardH * kBoardW);
        float sum = 0.0f;
        for (float p : step.policy_target) sum += p;
        CHECK_TRUE(std::abs(sum - 1.0f) < 1e-4f);
        CHECK_TRUE(step.legal_mask[step.action] == 1);
        CHECK_TRUE(!step.done);
    }
}

void test_truncated_bootstrap_capture() {
    SelfPlayConfig cfg;
    cfg.seed = 17;
    cfg.max_steps = 2;
    cfg.save_bootstrap_state = true;
    cfg.mcts.num_simulations = 4;
    cfg.mcts.latent_top_k = 4;
    cfg.mcts.max_nodes = 2048;

    DummyNetwork net;
    SelfPlayRunner runner(cfg);
    const GameHistory history = runner.run(net);

    CHECK_TRUE(!history.terminal);
    CHECK_TRUE(history.truncated);
    CHECK_TRUE(history.steps.size() == 2u);
    CHECK_TRUE(history.bootstrap_state.has_value());
    const BootstrapState& bootstrap = *history.bootstrap_state;
    CHECK_TRUE(std::isfinite(bootstrap.root_value));
    CHECK_TRUE(bootstrap.observation.size() == OBS_CHANNELS * kBoardH * kBoardW);
    CHECK_TRUE(bootstrap.policy_target.size() == kActionSpaceSize);
    CHECK_TRUE(bootstrap.legal_mask.size() == kActionSpaceSize);
    CHECK_TRUE(bootstrap.legal_mask[kFlatWaitOffset] == 1u);
    float sum = 0.0f;
    for (float probability : bootstrap.policy_target) sum += probability;
    CHECK_TRUE(std::abs(sum - 1.0f) < 1e-4f);
}

void test_external_environment_provenance_is_authoritative() {
    SelfPlayConfig cfg;
    cfg.seed = 7;
    cfg.max_steps = 1;
    cfg.mcts.num_simulations = 2;
    cfg.mcts.latent_top_k = 2;
    cfg.mcts.max_nodes = 2048;

    TDEngine env(11, 11, 99, true);
    DummyNetwork net;
    SelfPlayRunner runner(cfg);
    const auto history = runner.run(env, net);
    CHECK_TRUE(history.seed == 99);
    CHECK_TRUE(history.wave_mode == WaveMode::Budgeted);
    CHECK_TRUE(history.truncated);
    CHECK_TRUE(!history.terminal);
}

void test_selfplay_jsonl_writer() {
    SelfPlayConfig cfg;
    cfg.seed = 3;
    cfg.max_steps = 2;
    cfg.mcts.num_simulations = 4;
    cfg.mcts.latent_top_k = 4;
    cfg.mcts.max_nodes = 2048;

    DummyNetwork net;
    SelfPlayRunner runner(cfg);
    auto history = runner.run(net);

    const std::string path = "selfplay_test_tmp.jsonl";
    const std::string summary = "selfplay_test_summary_tmp.json";
    write_history_jsonl(history, path);
    write_history_summary_json(history, summary);

    std::ifstream in(path);
    CHECK_TRUE(in.good());
    std::string line;
    std::getline(in, line);
    CHECK_TRUE(line.find("\"policy_target\"") != std::string::npos);
    CHECK_TRUE(line.find("\"observation\"") != std::string::npos);

    std::ifstream summary_in(summary);
    CHECK_TRUE(summary_in.good());
}

int main() {
    test_selfplay_dummy_history();
    test_truncated_bootstrap_capture();
    test_external_environment_provenance_is_authoritative();
    test_selfplay_jsonl_writer();
    std::cout << "Self-play tests passed!" << std::endl;
    return 0;
}
