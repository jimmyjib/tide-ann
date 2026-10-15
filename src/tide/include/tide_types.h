#pragma once
#include "types/pending_insert.h"
#include "types/timing_types.h"
#include "types/search_context.h"
#include "types/workload_types.h"

namespace tide {

void generate_mixed_workloads(UnifiedContext& ctx);
void send_workload(UnifiedContext& ctx, uint32_t workload_idx);
void run_unified_workload(UnifiedContext& ctx);

}  // namespace tide
