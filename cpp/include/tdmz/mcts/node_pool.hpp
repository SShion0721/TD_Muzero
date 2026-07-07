#pragma once
#include <vector>
#include <stdexcept>
#include "tdmz/mcts/node.hpp"

namespace tdmz {

class NodePool {
public:
    explicit NodePool(int max_nodes = 4096) : max_nodes_(max_nodes) {
        nodes_.reserve(max_nodes);
    }

    int allocate() {
        if (static_cast<int>(nodes_.size()) >= max_nodes_) {
            throw std::runtime_error("NodePool exhausted");
        }
        nodes_.push_back(Node{});
        return static_cast<int>(nodes_.size()) - 1;
    }

    Node& get(int id) {
        return nodes_.at(id);
    }

    const Node& get(int id) const {
        return nodes_.at(id);
    }

    int size() const {
        return static_cast<int>(nodes_.size());
    }

    void clear() {
        nodes_.clear();
    }

private:
    int max_nodes_;
    std::vector<Node> nodes_;
};

} // namespace tdmz
