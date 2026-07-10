#pragma once

#include <stdexcept>
#include <vector>

namespace tdmz {

struct Edge {
    int action = -1;
    int child = -1;
};

class EdgePool {
public:
    explicit EdgePool(int max_edges) : max_edges_(max_edges) {
        if (max_edges_ <= 0) {
            throw std::invalid_argument("EdgePool max_edges must be positive");
        }
        edges_.reserve(static_cast<size_t>(max_edges_));
    }

    int allocate_range(int count) {
        if (count < 0 || count > remaining()) {
            throw std::runtime_error("EdgePool capacity is insufficient for edge range");
        }
        const int first = active_size_;
        const int required = active_size_ + count;
        if (required > static_cast<int>(edges_.size())) {
            const int created = required - static_cast<int>(edges_.size());
            edges_.resize(static_cast<size_t>(required));
            created_edges_this_search_ += created;
            reused_edges_this_search_ += count - created;
        } else {
            reused_edges_this_search_ += count;
        }
        active_size_ = required;
        return first;
    }

    Edge& get(int id) {
        validate_active_id(id);
        return edges_[static_cast<size_t>(id)];
    }

    const Edge& get(int id) const {
        validate_active_id(id);
        return edges_[static_cast<size_t>(id)];
    }

    int size() const { return active_size_; }
    int capacity() const { return max_edges_; }
    int remaining() const { return max_edges_ - active_size_; }
    int retained_edge_count() const { return static_cast<int>(edges_.size()); }
    int created_edges_this_search() const { return created_edges_this_search_; }
    int reused_edges_this_search() const { return reused_edges_this_search_; }

    void clear() {
        active_size_ = 0;
        created_edges_this_search_ = 0;
        reused_edges_this_search_ = 0;
    }

private:
    void validate_active_id(int id) const {
        if (id < 0 || id >= active_size_) {
            throw std::out_of_range("EdgePool edge id is outside the active search");
        }
    }

    int max_edges_ = 0;
    int active_size_ = 0;
    int created_edges_this_search_ = 0;
    int reused_edges_this_search_ = 0;
    std::vector<Edge> edges_;
};

} // namespace tdmz
