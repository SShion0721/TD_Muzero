#include "tdmz/selfplay/trajectory_writer.hpp"
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace tdmz {

namespace {

template <typename T>
void write_vector(std::ostream& out, const std::vector<T>& values) {
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ",";
        out << values[i];
    }
    out << "]";
}

void write_step_json(std::ostream& out, const GameHistory& history, const TrajectoryStep& step) {
    out << std::setprecision(8);
    out << "{";
    out << "\"seed\":" << history.seed << ",";
    out << "\"step_index\":" << step.step_index << ",";
    out << "\"action\":" << step.action << ",";
    out << "\"reward\":" << step.reward << ",";
    out << "\"root_value\":" << step.root_value << ",";
    out << "\"done\":" << (step.done ? "true" : "false") << ",";
    out << "\"money\":" << step.money << ",";
    out << "\"base_hp\":" << step.base_hp << ",";
    out << "\"wave\":" << step.wave << ",";
    out << "\"time\":" << step.time << ",";
    out << "\"policy_target\":";
    write_vector(out, step.policy_target);
    out << ",\"legal_mask\":";
    write_vector(out, step.legal_mask);
    out << ",\"observation\":";
    write_vector(out, step.observation);
    out << "}";
}

} // namespace

void write_history_jsonl(const GameHistory& history, const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to open trajectory JSONL path");
    for (const auto& step : history.steps) {
        write_step_json(out, history, step);
        out << "\n";
    }
}

void write_history_summary_json(const GameHistory& history, const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to open trajectory summary path");
    out << std::setprecision(8);
    out << "{";
    out << "\"seed\":" << history.seed << ",";
    out << "\"max_steps\":" << history.max_steps << ",";
    out << "\"num_steps\":" << history.steps.size() << ",";
    out << "\"terminal\":" << (history.terminal ? "true" : "false") << ",";
    out << "\"total_reward\":" << history.total_reward;
    out << "}\n";
}

} // namespace tdmz
