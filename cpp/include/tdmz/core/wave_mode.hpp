#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace tdmz {

enum class WaveMode : uint32_t {
    Unknown = 0,
    Fixed = 1,
    Budgeted = 2,
};

inline constexpr WaveMode wave_mode_from_budgeted(bool budgeted) {
    return budgeted ? WaveMode::Budgeted : WaveMode::Fixed;
}

inline constexpr bool wave_mode_is_known(WaveMode mode) {
    return mode == WaveMode::Fixed || mode == WaveMode::Budgeted;
}

inline constexpr const char* wave_mode_name(WaveMode mode) {
    switch (mode) {
        case WaveMode::Fixed: return "fixed";
        case WaveMode::Budgeted: return "budgeted";
        case WaveMode::Unknown: return "unknown";
    }
    return "unknown";
}

inline WaveMode parse_wave_mode(const std::string& value) {
    if (value == "fixed") return WaveMode::Fixed;
    if (value == "budgeted") return WaveMode::Budgeted;
    if (value == "unknown") return WaveMode::Unknown;
    throw std::runtime_error("Invalid wave mode: " + value);
}

inline WaveMode wave_mode_from_u32(uint32_t value) {
    switch (value) {
        case 0u: return WaveMode::Unknown;
        case 1u: return WaveMode::Fixed;
        case 2u: return WaveMode::Budgeted;
        default: throw std::runtime_error("Invalid serialized wave-mode value");
    }
}

} // namespace tdmz
