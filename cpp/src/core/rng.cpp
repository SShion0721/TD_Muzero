#include "tdmz/core/rng.hpp"
#include <stdexcept>

namespace tdmz {

PythonRNG::PythonRNG(uint64_t seed_val) : rng_(static_cast<uint32_t>(seed_val)) {}

void PythonRNG::seed(uint64_t seed_val) {
    rng_.seed(static_cast<uint32_t>(seed_val));
}

uint32_t PythonRNG::getrandbits(int k) {
    if (k <= 0 || k > 32) {
        throw std::invalid_argument("getrandbits only supports 1 to 32 bits");
    }
    uint32_t r = rng_();
    if (k < 32) {
        return r >> (32 - k);
    }
    return r;
}

double PythonRNG::random() {
    uint32_t a = rng_() >> 5;
    uint32_t b = rng_() >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

int PythonRNG::randrange(int stop) {
    if (stop <= 0) {
        throw std::invalid_argument("empty range for randrange()");
    }
    
    // Python's _randbelow_with_getrandbits
    int k = 0;
    int n = stop - 1;
    while (n > 0) {
        n >>= 1;
        k += 1;
    }
    
    if (k == 0) return 0;
    
    while (true) {
        uint32_t r = getrandbits(k);
        if (static_cast<int>(r) < stop) {
            return static_cast<int>(r);
        }
    }
}

} // namespace tdmz
