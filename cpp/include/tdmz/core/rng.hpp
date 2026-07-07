#pragma once
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace tdmz {

class PythonRNG {
public:
    explicit PythonRNG(uint64_t seed);
    void seed(uint64_t seed);

    uint32_t getrandbits(int k);
    double random();

    // Returns a random integer N such that 0 <= N < stop
    int randrange(int stop);

    template<typename T>
    void shuffle(std::vector<T>& x) {
        for (int i = static_cast<int>(x.size()) - 1; i > 0; --i) {
            int j = randrange(i + 1);
            std::swap(x[i], x[j]);
        }
    }

    template<typename T>
    T choice(const std::vector<T>& x) {
        if (x.empty()) throw std::runtime_error("Cannot choose from an empty sequence");
        return x[randrange(static_cast<int>(x.size()))];
    }

private:
    std::mt19937 rng_;
};

} // namespace tdmz
