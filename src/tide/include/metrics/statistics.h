#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ann {

// Nearest-rank percentile index, ceil(p / 100 * n) - 1, of sorted samples.
inline size_t nearest_rank_percentile_index(size_t sample_count,
                                            unsigned percentile) {
    if (sample_count == 0) {
        throw std::invalid_argument("nearest-rank percentile requires samples");
    }
    if (percentile == 0 || percentile > 100) {
        throw std::invalid_argument("nearest-rank percentile must be in [1, 100]");
    }

    // split to avoid overflowing sample_count * percentile
    const size_t whole_hundreds = sample_count / 100;
    const size_t remainder = sample_count % 100;
    const size_t one_based_rank = whole_hundreds * percentile +
        (remainder * percentile + 99) / 100;
    return one_based_rank - 1;
}

inline void update_interval_max(std::atomic<uint64_t>& interval_max,
                                uint64_t sample) {
    uint64_t observed = interval_max.load(std::memory_order_relaxed);
    while (observed < sample &&
           !interval_max.compare_exchange_weak(
               observed, sample, std::memory_order_relaxed,
               std::memory_order_relaxed)) {}
}

inline uint64_t take_interval_max(std::atomic<uint64_t>& interval_max) {
    return interval_max.exchange(0, std::memory_order_relaxed);
}

inline uint64_t snapshot_interval_max(std::atomic<uint64_t>& interval_max,
                                      bool consume_interval_max) {
    if (consume_interval_max) {
        return take_interval_max(interval_max);
    }
    return interval_max.load(std::memory_order_relaxed);
}

}  // namespace ann
