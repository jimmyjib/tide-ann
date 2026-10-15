/**
 * Metric thread of the unified pipeline.
 *
 * This thread handles post-processing operations kept off the critical path
 * of process_response():
 * - log_search_result() - File I/O
 * - log_hop_state() - File I/O
 * - merge_into_results() - CPU computation
 * - beam tracking - CPU computation
 *
 * By moving these operations off the search thread, we achieve:
 * 1. More accurate network timing (poll_to_recv does not include file I/O)
 * 2. Better throughput (search thread doesn't block on logging)
 * 3. Cleaner separation of concerns
 */

#include "pipeline_types.h"
#include "types/workload_types.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <iostream>

namespace tide {

void process_metric_event(UnifiedContext& ctx, const MetricEvent& evt) {
    auto& item = ctx.workloads[evt.workload_idx];

    switch (evt.type) {
        case MetricEventType::SEARCH_DONE: {
            // 1. Debug logging (if enabled)
            if (g_debug.enabled && !item.result_neighbors.empty()) {
                // The owners come with the event (resolved on the pipeline thread).
                uint32_t result_partition = evt.labels.empty() ? 0 : evt.labels[0];

                // Atomic counters for debug stats
                if (result_partition < g_debug.query_result_partition.size()) {
                    g_debug.query_result_partition[result_partition]++;
                }
                g_debug.query_count++;
                g_debug.query_total_hops += item.hops;

                // Log to CSV file
                log_search_result("QUERY", item.id, item.entry_partition, result_partition,
                                  item.hops, item.result_neighbors, evt.labels, item.result_distances, ctx.K,
                                  evt.gt_row, evt.gt_cols);
            }

            // 2. Merge pending inserts with beam, extract top-K
            if (ctx.pending_mgr && ctx.pending_mgr->size() > 0 && !ctx.no_merge) {
                // For inserted queries: exclude self and later inserts (>= threshold)
                // For baseline: exclude all inserted nodes (>= insert_threshold_id)
                uint32_t exclude_threshold = UINT32_MAX;
                if (ctx.insert_threshold_id > 0) {
                    exclude_threshold = ctx.insert_threshold_id;
                    if (item.is_inserted_query) {
                        exclude_threshold = ctx.insert_threshold_id + item.insert_gt_index;
                    }
                }

                auto merge_start = std::chrono::steady_clock::now();
                auto merge_stats = ctx.pending_mgr->merge_with_beam(
                    item.query.data(), ctx.dim,
                    evt.beam_snapshot,
                    item.result_neighbors, item.result_distances, ctx.K,
                    exclude_threshold);
                auto merge_end = std::chrono::steady_clock::now();
                ctx.merge_search_wall_us +=
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        merge_end - merge_start).count();
                ctx.merge_search_calls++;
                ctx.merge_search_pending_total += merge_stats.pending_total;
                ctx.merge_search_pending_eligible += merge_stats.pending_eligible;
                ctx.merge_search_pending_topk += merge_stats.pending_topk;
                if (merge_stats.result_changed) {
                    ctx.merge_search_result_changed++;
                }
            }

            // 3. Beam tracking: count inserted nodes in final results
            if (ctx.insert_threshold_id > 0) {
                for (auto id : item.result_neighbors) {
                    ctx.result_total_entries++;
                    if (id >= ctx.insert_threshold_id) {
                        ctx.result_inserted_entries++;
                    }
                }
                // Also track final beam state (from beam_snapshot)
                for (const auto& entry : evt.beam_snapshot) {
                    ctx.beam_total_entries++;
                    if (entry.node_id >= ctx.insert_threshold_id) {
                        ctx.beam_inserted_entries++;
                    }
                }
            }
            break;
        }

        case MetricEventType::SEARCH_HOP: {
            // 1. Debug logging
            if (g_debug.enabled) {
                log_hop_state("QUERY", item.id, evt.hop_num,
                              evt.from_partition, evt.to_partition, evt.beam_snapshot, evt.labels);
            }

            // 2. Beam tracking (intermediate beams)
            if (ctx.insert_threshold_id > 0) {
                for (const auto& entry : evt.beam_snapshot) {
                    ctx.beam_total_entries++;
                    if (entry.node_id >= ctx.insert_threshold_id) {
                        ctx.beam_inserted_entries++;
                    }
                }
            }
            break;
        }

        case MetricEventType::INSERT_SEARCH_DONE: {
            // Insert search phase logging. The insert counters (final partition,
            // insert count and hops) are the pipeline thread's, which counts them
            // when it pushes this event; this thread only writes the log line.
            if (g_debug.enabled && !item.search_neighbors.empty()) {
                uint32_t result_partition = evt.labels.empty() ? 0 : evt.labels[0];
                log_search_result("INSERT", item.id, evt.from_partition, result_partition,
                                  item.hops, item.search_neighbors, evt.labels, item.search_distances, ctx.K,
                                  nullptr, 0);
            }
            break;
        }

        case MetricEventType::INSERT_HOP: {
            if (g_debug.enabled) {
                log_hop_state("INSERT", item.id, evt.hop_num,
                              evt.from_partition, evt.to_partition, evt.beam_snapshot, evt.labels);
            }
            break;
        }

        default:
            break;
    }
}

void run_metric_thread(UnifiedContext& ctx) {
    while (ctx.metric_thread_running.load(std::memory_order_relaxed)) {
        MetricEvent evt;

        // Wait with timeout for graceful shutdown check
        if (!ctx.metric_queue->wait_pop(evt, 100)) {
            continue;  // Timeout - check if still running
        }

        if (evt.type == MetricEventType::SHUTDOWN) {
            // Drain remaining events before exit
            while (ctx.metric_queue->try_pop(evt)) {
                if (evt.type != MetricEventType::SHUTDOWN) {
                    process_metric_event(ctx, evt);
                }
            }
            break;
        }

        process_metric_event(ctx, evt);
    }

    ctx.metric_thread_running.store(false, std::memory_order_relaxed);
}

}  // namespace tide
