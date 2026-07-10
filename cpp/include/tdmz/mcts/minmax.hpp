#pragma once
#include <limits>
#include <algorithm>

namespace tdmz {

class MinMaxStats {
public:
    explicit MinMaxStats(float value_delta_max = 0.01f)
        : value_delta_max_(value_delta_max) {}

    void clear() {
        maximum_ = -std::numeric_limits<float>::infinity();
        minimum_ = std::numeric_limits<float>::infinity();
    }

    void update(float value) {
        maximum_ = std::max(maximum_, value);
        minimum_ = std::min(minimum_, value);
    }

    float normalize(float value) const {
        if (maximum_ > minimum_) {
            return (value - minimum_) / (maximum_ - minimum_ + value_delta_max_);
        }
        return value;
    }

private:
    float maximum_ = -std::numeric_limits<float>::infinity();
    float minimum_ = std::numeric_limits<float>::infinity();
    float value_delta_max_;
};

} // namespace tdmz
