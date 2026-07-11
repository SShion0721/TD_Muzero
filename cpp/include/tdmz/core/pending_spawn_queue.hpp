#pragma once

#include "tdmz/core/enemy.hpp"
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tdmz {

// A contiguous pending-wave queue that consumes entries by advancing a cursor
// instead of erasing from the front of a std::vector. Assignment starts a new
// wave and resets the cursor while preserving vector capacity where possible.
class PendingSpawnQueue {
public:
    using container_type = std::vector<EnemySpec>;
    using iterator = container_type::iterator;
    using const_iterator = container_type::const_iterator;

    PendingSpawnQueue() = default;

    PendingSpawnQueue& operator=(container_type values) {
        values_ = std::move(values);
        cursor_ = 0;
        return *this;
    }

    void reserve(std::size_t capacity) {
        values_.reserve(capacity);
    }

    bool empty() const {
        return cursor_ >= values_.size();
    }

    std::size_t size() const {
        return empty() ? 0u : values_.size() - cursor_;
    }

    EnemySpec& front() {
        if (empty()) throw std::out_of_range("Pending spawn queue is empty");
        return values_[cursor_];
    }

    const EnemySpec& front() const {
        if (empty()) throw std::out_of_range("Pending spawn queue is empty");
        return values_[cursor_];
    }

    iterator begin() {
        return values_.begin() + static_cast<container_type::difference_type>(cursor_);
    }

    const_iterator begin() const {
        return values_.begin() + static_cast<container_type::difference_type>(cursor_);
    }

    const_iterator cbegin() const {
        return begin();
    }

    iterator end() {
        return values_.end();
    }

    const_iterator end() const {
        return values_.end();
    }

    const_iterator cend() const {
        return end();
    }

    // Compatibility with the previous vector-based call site. Only erasing the
    // logical front is supported; no elements are moved and iterators after the
    // consumed entry remain valid until the next assignment/reallocation.
    iterator erase(iterator position) {
        if (empty() || position != begin()) {
            throw std::invalid_argument("Pending spawn queue can erase only its logical front");
        }
        ++cursor_;
        return begin();
    }

    std::size_t consumed_count() const {
        return cursor_;
    }

    std::size_t storage_size() const {
        return values_.size();
    }

private:
    container_type values_;
    std::size_t cursor_ = 0;
};

} // namespace tdmz
