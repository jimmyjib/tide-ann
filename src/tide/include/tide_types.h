/**
 * Coordinator types
 *
 * Main include file that pulls in all type definitions.
 * Backward-compatible: existing code can still include "tide_types.h"
 */

#ifndef TIDE_TYPES_H
#define TIDE_TYPES_H

// Include all type sub-headers
#include "types/pending_insert.h"
#include "types/timing_types.h"
#include "types/search_context.h"
#include "types/workload_types.h"

namespace tide {

// ============================================================================
// Function declarations for unified pipeline
// ============================================================================

void generate_mixed_workloads(UnifiedContext& ctx);
void send_workload(UnifiedContext& ctx, uint32_t workload_idx);
void process_response(UnifiedContext& ctx, uint32_t server_idx, uint32_t& completed);
void run_unified_workload(UnifiedContext& ctx);

// Debug functions for partition distribution analysis: pipeline_types.h.

}  // namespace tide

#endif  // TIDE_TYPES_H
