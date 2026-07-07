#pragma once
#include <vector>
#include <stdexcept>
#include "tdmz/mcts/node.hpp"

namespace tdmz {

class NodePool {
public:
    explicit NodePool(int reserve_nodes = 4096) {
        nodes_.reserve(reserve_nodes);
    }

    int allocate() {
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
    std::vector<Node> nodes_;
};

} // namespace tdmz
