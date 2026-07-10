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
        if (active_size_ >= max_nodes_) {
            throw std::runtime_error("NodePool exhausted");
        }

        const int id = active_size_++;
        if (id < static_cast<int>(nodes_.size())) {
            nodes_[static_cast<size_t>(id)].reset_for_reuse();
            ++reused_nodes_this_search_;
        } else {
            nodes_.push_back(Node{});
            ++new_nodes_this_search_;
        }
        return id;
    }

    Node& get(int id) {
        validate_active_id(id);
        return nodes_[static_cast<size_t>(id)];
    }

    const Node& get(int id) const {
        validate_active_id(id);
        return nodes_[static_cast<size_t>(id)];
    }

    int size() const {
        return active_size_;
    }

    int capacity() const {
        return max_nodes_;
    }

    int remaining() const {
        return max_nodes_ - active_size_;
    }

    int retained_node_count() const {
        return static_cast<int>(nodes_.size());
    }

    int new_nodes_this_search() const {
        return new_nodes_this_search_;
    }

    int reused_nodes_this_search() const {
        return reused_nodes_this_search_;
    }

    void clear() {
        active_size_ = 0;
        new_nodes_this_search_ = 0;
        reused_nodes_this_search_ = 0;
    }

private:
    void validate_active_id(int id) const {
        if (id < 0 || id >= active_size_) {
            throw std::out_of_range("NodePool node id is outside the active search");
        }
    }

    int max_nodes_;
    int active_size_ = 0;
    int new_nodes_this_search_ = 0;
    int reused_nodes_this_search_ = 0;
    std::vector<Node> nodes_;
};

} // namespace tdmz
