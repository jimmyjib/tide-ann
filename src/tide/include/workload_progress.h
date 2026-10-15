#pragma once

#include <cstdint>

namespace tide {

// true when there is new progress and the run finished or a count or time interval passed
inline bool workload_progress_due(uint64_t completed, uint64_t reported,
                                  uint64_t total, uint64_t count_interval,
                                  uint64_t elapsed_since_report_ms,
                                  uint64_t time_interval_ms = 2000) {
    return completed > reported && completed <= total &&
        (completed == total || completed - reported >= count_interval ||
         elapsed_since_report_ms >= time_interval_ms);
}

}  // namespace tide
