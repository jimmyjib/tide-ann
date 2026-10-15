// Metric thread: CSV logs, pending-insert merge and beam tracking, off the response path.

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
            if (g_debug.enabled && !item.result_neighbors.empty()) {
                // owners were resolved on the pipeline thread
                uint32_t result_partition = evt.labels.empty() ? 0 : evt.labels[0];

                if (result_partition < g_debug.query_result_partition.size()) {
                    g_debug.query_result_partition[result_partition]++;
                }
                g_debug.query_count++;
                g_debug.query_total_hops += item.hops;

                log_search_result("QUERY", item.id, item.entry_partition, result_partition,
                                  item.hops, item.result_neighbors, evt.labels, item.result_distances, ctx.K,
                                  evt.gt_row, evt.gt_cols);
            }

            // merge pending inserts into the result
            if (ctx.pending_mgr && ctx.pending_mgr->size() > 0 && !ctx.no_merge) {
                // Baseline queries exclude every insert. An inserted query excludes itself and later inserts.
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

            // count inserted nodes in results and beams
            if (ctx.insert_threshold_id > 0) {
                for (auto id : item.result_neighbors) {
                    ctx.result_total_entries++;
                    if (id >= ctx.insert_threshold_id) {
                        ctx.result_inserted_entries++;
                    }
                }
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
            // only the log line is written here (the pipeline thread already counted the insert)
            if (g_debug.enabled && !item.search_neighbors.empty()) {
                uint32_t result_partition = evt.labels.empty() ? 0 : evt.labels[0];
                log_search_result("INSERT", item.id, evt.from_partition, result_partition,
                                  item.hops, item.search_neighbors, evt.labels, item.search_distances, ctx.K,
                                  nullptr, 0);
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

        if (!ctx.metric_queue->wait_pop(evt, 100)) {
            continue;
        }

        if (evt.type == MetricEventType::SHUTDOWN) {
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
