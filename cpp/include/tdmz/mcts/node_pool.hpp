#pragma once
#include <stdexcept>
#include <vector>
#include "tdmz/mcts/node.hpp"

namespace tdmz {

class NodePool {
public:
    explicit NodePool(int max_nodes = 4096) : max_nodes_(max_nodes) {
        if (max_nodes_ <= 0) {
            throw std::invalid_argument("NodePool max_nodes must be positive");
        }
        nodes_.reserve(static_cast<size_t>(max_nodes_));
    }

    int allocate() {
        if (static_cast<int>(nodes_.size()) >= max_nodes_) {
            throw std::runtime_error("NodePool exhausted");
        }
        nodes_.push_back(Node{});
        return static_cast<int>(nodes_.size()) - 1;
    }

    Node& get(int id) {
        return nodes_.at(static_cast<size_t>(id));
    }

    const Node& get(int id) const {
        return nodes_.at(static_cast<size_t>(id));
    }

    int size() const {
        return static_cast<int>(nodes_.size());
    }

    int capacity() const {
        return max_nodes_;
    }

    int remaining() const {
        return max_nodes_ - size();
    }

    void clear() {
        nodes_.clear();
    }

private:
    int max_nodes_;
    std::vector<Node> nodes_;
};

} // namespace tdmz
