#pragma once

#include <cstdint>

namespace tide {

// A timer may throttle genuine progress; it must never manufacture progress.
inline bool workload_progress_due(uint64_t completed, uint64_t reported,
                                  uint64_t total, uint64_t count_interval,
                                  uint64_t elapsed_since_report_ms,
                                  uint64_t time_interval_ms = 2000) {
    return completed > reported && completed <= total &&
        (completed == total || completed - reported >= count_interval ||
         elapsed_since_report_ms >= time_interval_ms);
}

}  // namespace tide
