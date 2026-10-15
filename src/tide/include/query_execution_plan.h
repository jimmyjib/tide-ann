#pragma once

#include <algorithm>
#include <cstdint>

namespace tide {

struct AlternateQueryExecutionPlan {
    uint32_t loaded_rows = 0;
    uint32_t execution_count = 0;
};

// Load no more unique query rows than the execution can use.  The execution
// count is deliberately not clamped here: a workload may cycle the loaded query
// rows through workload_generator's modulo assignment.
inline AlternateQueryExecutionPlan make_alternate_query_execution_plan(
        uint32_t execution_query_count, uint32_t available_rows) {
    return {std::min(execution_query_count, available_rows),
            execution_query_count};
}

}  // namespace tide
