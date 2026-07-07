#include "tdmz/core/engine.hpp"
#include "tdmz/core/observation.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <string>

using namespace tdmz;

int main(int argc, char** argv) {
    uint64_t seed = 0;
    std::string out_file = "cpp_trace.json";
    
    if (argc > 1) {
        seed = std::stoull(argv[1]);
    }
    if (argc > 2) {
        out_file = argv[2];
    }
    
    int num_steps = 1000;
    TDEngine env(11, 11, seed);
    PythonRNG rng(seed);

    std::ofstream out(out_file);
    if (!out) {
        std::cerr << "Failed to open " << out_file << "\n";
        return 1;
    }

    try {

    auto print_step = [&](bool is_first, float reward, bool done) {
        if (!is_first) out << ",";
        out << "{";
        out << "\"reward\":" << reward << ",";
        out << "\"done\":" << (done ? "true" : "false") << ",";
        out << "\"money\":" << env.money() << ",";
        out << "\"base_hp\":" << env.base_hp() << ",";
        out << "\"wave\":" << env.wave() << ",";
        
        auto legal = env.legal_actions();
        std::sort(legal.begin(), legal.end());
        out << "\"legal_actions\":[";
        for (size_t i = 0; i < legal.size(); ++i) {
            out << legal[i] << (i + 1 == legal.size() ? "" : ",");
        }
        out << "],";
        
        auto obs = make_observation_python_parity(env);
        out << "\"obs_bytes\":\"";
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(obs.data());
        size_t num_bytes = obs.size() * sizeof(float);
        out << std::hex << std::setfill('0');
        for (size_t i = 0; i < num_bytes; ++i) {
            out << std::setw(2) << static_cast<int>(bytes[i]);
        }
        out << std::dec << "\"";
        out << "}";
    };

    out << "{\"seed\":" << seed << ",\"actions\":[";
    
    std::vector<int> actions;
    TDEngine sim_env(11, 11, seed);
    for (int i = 0; i < num_steps; ++i) {
        auto legal = sim_env.legal_actions();
        std::sort(legal.begin(), legal.end());
        int action = rng.choice(legal);
        actions.push_back(action);
        if (sim_env.step_action(action).done) break;
    }
    
    for (size_t i = 0; i < actions.size(); ++i) {
        out << actions[i] << (i + 1 == actions.size() ? "" : ",");
    }
    out << "],\"steps\":[";
    
    print_step(true, 0.0f, false);
    
    for (int action : actions) {
        auto res = env.step_action(action);
        print_step(false, res.reward, res.done);
    }
    
    out << "]}" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
