#pragma once
#include "tdmz/core/action.hpp"
#include "tdmz/core/tower.hpp"
#include <array>
#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace tdmz {

struct Bitboard128 {
    uint64_t lo = 0;
    uint64_t hi = 0;

    static constexpr Bitboard128 zero() { return {0ULL, 0ULL}; }

    static constexpr Bitboard128 from_cell(int cell) {
        return cell < 64 ? Bitboard128{1ULL << cell, 0ULL}
                         : Bitboard128{0ULL, 1ULL << (cell - 64)};
    }

    void set(int cell) {
        if (cell < 64) lo |= 1ULL << cell;
        else hi |= 1ULL << (cell - 64);
    }

    void clear(int cell) {
        if (cell < 64) lo &= ~(1ULL << cell);
        else hi &= ~(1ULL << (cell - 64));
    }

    bool test(int cell) const {
        if (cell < 64) return ((lo >> cell) & 1ULL) != 0;
        return ((hi >> (cell - 64)) & 1ULL) != 0;
    }

    Bitboard128 operator&(Bitboard128 b) const { return {lo & b.lo, hi & b.hi}; }
    Bitboard128 operator|(Bitboard128 b) const { return {lo | b.lo, hi | b.hi}; }
    Bitboard128 operator^(Bitboard128 b) const { return {lo ^ b.lo, hi ^ b.hi}; }
    Bitboard128 operator~() const { return {~lo, ~hi}; }
    Bitboard128& operator&=(Bitboard128 b) { lo &= b.lo; hi &= b.hi; return *this; }
    Bitboard128& operator|=(Bitboard128 b) { lo |= b.lo; hi |= b.hi; return *this; }
    Bitboard128& operator^=(Bitboard128 b) { lo ^= b.lo; hi ^= b.hi; return *this; }
    bool operator==(Bitboard128 b) const { return lo == b.lo && hi == b.hi; }
    bool operator!=(Bitboard128 b) const { return !(*this == b); }
    explicit operator bool() const { return lo != 0 || hi != 0; }

    int popcount() const {
#if defined(_MSC_VER)
        return static_cast<int>(__popcnt64(lo)) + static_cast<int>(__popcnt64(hi));
#else
        return __builtin_popcountll(lo) + __builtin_popcountll(hi);
#endif
    }

    int lsb() const {
        if (lo) {
#if defined(_MSC_VER)
            unsigned long idx = 0;
            _BitScanForward64(&idx, lo);
            return static_cast<int>(idx);
#else
            return __builtin_ctzll(lo);
#endif
        }
        if (hi) {
#if defined(_MSC_VER)
            unsigned long idx = 0;
            _BitScanForward64(&idx, hi);
            return 64 + static_cast<int>(idx);
#else
            return 64 + __builtin_ctzll(hi);
#endif
        }
        return -1;
    }

    int pop_lsb() {
        if (lo) {
#if defined(_MSC_VER)
            unsigned long idx = 0;
            _BitScanForward64(&idx, lo);
            lo &= lo - 1;
            return static_cast<int>(idx);
#else
            int idx = __builtin_ctzll(lo);
            lo &= lo - 1;
            return idx;
#endif
        }
        if (hi) {
#if defined(_MSC_VER)
            unsigned long idx = 0;
            _BitScanForward64(&idx, hi);
            hi &= hi - 1;
            return 64 + static_cast<int>(idx);
#else
            int idx = __builtin_ctzll(hi);
            hi &= hi - 1;
            return 64 + idx;
#endif
        }
        return -1;
    }
};

inline int cell_id(int x, int y) { return y * kBoardW + x; }
inline int cell_x(int cell) { return cell % kBoardW; }
inline int cell_y(int cell) { return cell / kBoardW; }
inline bool valid_cell_xy(int x, int y) { return x >= 0 && x < kBoardW && y >= 0 && y < kBoardH; }
inline bool valid_cell(int cell) { return cell >= 0 && cell < kCells; }

struct BoardTables {
    std::array<int, kCells> x{};
    std::array<int, kCells> y{};
    std::array<Bitboard128, kCells> cell_bb{};
    Bitboard128 board_mask{};

    std::array<std::array<int, 4>, kCells> neighbors4{};
    std::array<int, kCells> neighbor4_count{};
    std::array<Bitboard128, kCells> neighbor4_bb{};

    std::array<std::array<int, kCells>, kCells> manhattan{};
    std::array<std::array<float, kCells>, kCells> dist2{};

    // [tower_type][level][cell]
    std::array<std::array<std::array<Bitboard128, kCells>, kTowerMaxLevel + 1>, kBuildTypes> range_mask{};
    std::array<std::array<std::array<Bitboard128, kCells>, kTowerMaxLevel + 1>, kBuildTypes> range_mask_expanded8{};

    std::array<ActionType, kActionSpaceSize> action_type{};
    std::array<int, kActionSpaceSize> action_x{};
    std::array<int, kActionSpaceSize> action_y{};
    std::array<int, kActionSpaceSize> action_wait{};

    std::array<std::array<int, kCells>, kBuildTypes> build_action{};
    std::array<int, kCells> upgrade_action{};
    std::array<int, kCells> sell_action{};
    int wait_action = kFlatWaitOffset;

    BoardTables();
};

const BoardTables& board_tables();

} // namespace tdmz
