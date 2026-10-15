/**
 * State-passing index server (in memory).
 *
 * Holds one partition of the graph in RAM (hybrid PQ own-only storage by default,
 * full replication with --full-vectors) and passes search state between partitions.
 *
 * Usage:
 *   tide-server --data-dir /data/runtime --rank 0 --num-servers 4 --port 28800 \
 *       --peers tcp://server0:28800,tcp://server1:28800,tcp://server2:28800,tcp://server3:28800
 */

#include <zmq.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cstring>
#include <getopt.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <unordered_map>
#include <memory>
#include <filesystem>
#include <stdexcept>
#include <unistd.h>
#include <fcntl.h>

#include "server_types.h"
#include "stats_window.h"
#include "metrics/statistics.h"
#include "state_protocol.h"
#include "distance.h"
#include "protocol/broadcast_messages.h"
#include "coordinator_utils.h"
#include "npy_reader.h"
#include "npy_writer.h"
#include "thread_pool.h"
#include "observability.h"
#include "membership_migration.h"

using namespace ann;
using namespace std::chrono;
using namespace tide;

// =============================================================================
// Global variable definitions (declared extern in server_types.h)
// =============================================================================
namespace tide {
std::shared_mutex g_resize_mutex;
std::mutex g_alloc_mutex;
LockMetrics g_lock_metrics;
GraphMetrics g_graph_metrics;
std::string g_data_dir;
int g_rank = 0;
uint32_t g_start_node = 0;

// Start of the current measurement interval: updated on RESET_REQUEST. The queue-wait
// accumulators skip items enqueued before it (work queued before that RESET).
// Stored as nanoseconds since steady_clock epoch (atomic-safe plain int64).
std::atomic<int64_t> g_interval_start_ns{0};
uint64_t g_ownership_epoch = 0;
MembershipMigrationPlan g_membership_migration_plan;
// Epochs of the accepted migrations, oldest first (the plans are dropped).
std::vector<uint64_t> g_accepted_migration_epochs;

// L2 is the only metric: cosine data is normalised offline (tools/prepare/normalize_vectors.py),
// where squared L2 / 2 = 1 - cosine.
constexpr const char* kMetricName = "l2";

// RESET-vs-worker sync. The search and maintenance hot paths take no shared_lock on
// g_resize_mutex; the RESET path's ChunkedArray.clear() is NOT thread-safe
// with concurrent reads. Dedicated rarely-contested mutex separate from
// g_resize_mutex: workers take shared_lock (cheap, ~10ns), RESET takes
// unique_lock.
std::shared_mutex g_admin_mutex;
}

// Resident set size of this process (Linux /proc/self/statm), 0 when unavailable.
static uint64_t process_rss_bytes() {
    std::ifstream statm("/proc/self/statm");
    uint64_t total_pages = 0, resident_pages = 0;
    if (!(statm >> total_pages >> resident_pages)) return 0;
    const long page = sysconf(_SC_PAGESIZE);
    return page > 0 ? resident_pages * static_cast<uint64_t>(page) : 0;
}

static ann::LockRoleStatsHeader snapshot_lock_role(const ann::LockRoleMetrics& src) {
    ann::LockRoleStatsHeader dst{};
    dst.acquisitions = src.acquisitions.load(std::memory_order_relaxed);
    dst.wait_us = src.wait_us.load(std::memory_order_relaxed);
    dst.hold_us = src.hold_us.load(std::memory_order_relaxed);
    dst.max_wait_us = src.max_wait_us.load(std::memory_order_relaxed);
    dst.max_hold_us = src.max_hold_us.load(std::memory_order_relaxed);
    return dst;
}

// =============================================================================
// worker_thread - Worker thread function with batch processing
// =============================================================================
void worker_thread(
    int worker_id,
    ServerData& data,
    ThreadSafeQueue<WorkItem>& work_queue,
    ThreadSafeQueue<WorkItem>& backlink_queue,
    ThreadSafeQueue<WorkItem>& broadcast_queue,
    ThreadSafeQueue<WorkItem>& insert_queue,
    ThreadSafeQueue<ResponseItem>& response_queue,
    std::atomic<bool>& running,
    std::atomic<uint64_t>& processed_count,
    TimingMetrics& metrics,
    IOTimingState& io_timing,
    std::atomic<uint32_t>& maintenance_in_flight,
    std::atomic<uint32_t>& search_in_flight,
    uint32_t insert_batch_size,
    uint32_t backlink_batch_size,
    uint32_t broadcast_batch_size
) {
    // Thread-local: previous serialize_us for this worker
    thread_local uint32_t last_serialize_us = 0;

    // Each worker has its own seen vector and search worker
    std::vector<bool> seen(n_base_load(data), false);
    std::vector<uint32_t> seen_nodes;
    seen_nodes.reserve(10000);

    bool has_deferred_work = false;
    WorkItem deferred_work;
    ann::SearchInFlightOwnership search_ownership(search_in_flight);
    uint32_t next_maintenance_kind = 0;

    // Strict batching only: a queue shorter than batch_size is not processed.
    auto process_ready_maintenance_batch = [&]() -> bool {
        auto process_insert_batch = [&]() -> bool {
            uint32_t batch_size = std::max<uint32_t>(insert_batch_size, 1);
            std::shared_lock<std::shared_mutex> admin_lock(
                tide::g_admin_mutex);
            ann::InFlightOwnership ownership(maintenance_in_flight);
            ownership.acquire();
            size_t queued = insert_queue.size();
            if (queued < batch_size) return false;
            auto batch = insert_queue.try_pop_n(batch_size);
            if (batch.empty()) return false;
            auto responses = batch_process_inserts(data, batch, metrics);
            auto now = steady_clock::now();
            for (size_t i = 0; i < responses.size(); i++) {
                responses[i].enqueue_time = now;
                response_queue.push(std::move(responses[i]));
            }
            processed_count += batch.size();
            return true;
        };

        auto process_backlink_batch = [&]() -> bool {
            uint32_t batch_size = std::max<uint32_t>(backlink_batch_size, 1);
            std::shared_lock<std::shared_mutex> admin_lock(
                tide::g_admin_mutex);
            ann::InFlightOwnership ownership(maintenance_in_flight);
            ownership.acquire();
            size_t queued = backlink_queue.size();
            if (queued < batch_size) return false;
            auto batch = backlink_queue.try_pop_n(batch_size);
            if (batch.empty()) return false;
            auto responses = batch_process_backlinks(data, batch, metrics);
            auto now = steady_clock::now();
            for (size_t i = 0; i < responses.size(); i++) {
                responses[i].enqueue_time = now;
                response_queue.push(std::move(responses[i]));
            }
            processed_count += batch.size();
            return true;
        };

        auto process_broadcast_batch = [&]() -> bool {
            uint32_t batch_size = std::max<uint32_t>(broadcast_batch_size, 1);
            std::shared_lock<std::shared_mutex> admin_lock(
                tide::g_admin_mutex);
            ann::InFlightOwnership ownership(maintenance_in_flight);
            ownership.acquire();
            size_t queued = broadcast_queue.size();
            if (queued < batch_size) return false;
            auto batch = broadcast_queue.try_pop_n(batch_size);
            if (batch.empty()) return false;
            auto responses = batch_process_broadcasts(data, batch, metrics);
            auto now = steady_clock::now();
            for (size_t i = 0; i < responses.size(); i++) {
                if (responses[i].response.empty()) continue;
                responses[i].enqueue_time = now;
                response_queue.push(std::move(responses[i]));
            }
            processed_count += batch.size();
            return true;
        };

        for (uint32_t offset = 0; offset < 3; ++offset) {
            uint32_t kind = (next_maintenance_kind + offset) % 3;
            bool processed = false;
            if (kind == 0) {
                processed = process_insert_batch();
            } else if (kind == 1) {
                processed = process_backlink_batch();
            } else {
                processed = process_broadcast_batch();
            }
            if (processed) {
                next_maintenance_kind = (kind + 1) % 3;
                return true;
            }
        }

        return false;
    };

    while (running) {
        WorkItem work;
        bool from_deferred = false;
        if (has_deferred_work) {
            work = std::move(deferred_work);
            has_deferred_work = false;
            from_deferred = true;
        } else {
            bool got = false;
            {
                // Shutdown takes the unique side of this lock. A search is
                // counted before the queue pop becomes invisible to that
                // quiescence check, closing the dequeue→handler-lock gap.
                std::shared_lock<std::shared_mutex> dequeue_lock(
                    tide::g_admin_mutex);
                got = work_queue.try_pop(work);
                if (got) {
                    search_ownership.acquire_if_search(work.msg_type);
                }
            }
            if (!got) {
                if (process_ready_maintenance_batch()) {
                    continue;
                }
                // Wait on the queue CV without holding the shared admin lock
                // and without dequeuing. Eight idle readers otherwise starve
                // the terminal shutdown writer. The locked try_pop above
                // remains the only dequeue point, preserving the
                // dequeue→in-flight accounting contract.
                work_queue.wait_until_nonempty(50);
                continue;
            }
        }

        bool is_control_work = (work.msg_type == MessageType::DRAIN_REQUEST);
        if (!from_deferred && !is_control_work && process_ready_maintenance_batch()) {
            deferred_work = std::move(work);
            has_deferred_work = true;
            continue;
        }

        // Queue wait time
        auto pickup_time = steady_clock::now();
        uint64_t queue_wait = duration_cast<microseconds>(pickup_time - work.enqueue_time).count();
        // Gate: skip stale items enqueued before the last RESET. Their
        // measurement is correct but belongs to the time before that RESET.
        int64_t interval_start_ns = tide::g_interval_start_ns.load(std::memory_order_acquire);
        bool is_stale = work.enqueue_time.time_since_epoch().count() < interval_start_ns;
        if (!is_stale) {
            metrics.queue_wait_us.fetch_add(queue_wait, std::memory_order_relaxed);
            metrics.queue_wait_count.fetch_add(1, std::memory_order_relaxed);
            ann::update_interval_max(metrics.queue_wait_interval_max_us, queue_wait);
        }

        // RESET-vs-worker: shared_lock on g_admin_mutex protects every
        // handler dispatched below from RESET's unique_lock + ChunkedArray.clear().
        // Uncontested cost is small; RESET is admin-only.
        std::shared_lock<std::shared_mutex> admin_lock(tide::g_admin_mutex);

        // ============================================================
        // Unified Pipeline: Dispatch based on message type
        // ============================================================
        if (work.msg_type == MessageType::DRAIN_REQUEST) {
            // Flush residual backlink/broadcast queues.
            // Workers skip queues with < batch_size items, so residuals
            // accumulate. We process them here before the drain ACK.
            // handle_* functions acquire locks internally.
            size_t ins_flushed = 0, bl_flushed = 0, bc_flushed = 0;
            while (true) {
                auto batch = insert_queue.try_pop_n(1000);
                if (batch.empty()) break;
                auto responses = batch_process_inserts(data, batch, metrics);
                auto now = steady_clock::now();
                for (auto& r : responses) {
                    r.enqueue_time = now;
                    response_queue.push(std::move(r));
                }
                ins_flushed += batch.size();
            }
            while (true) {
                auto batch = backlink_queue.try_pop_n(1000);
                if (batch.empty()) break;
                auto responses = batch_process_backlinks(data, batch, metrics);
                auto now = steady_clock::now();
                for (auto& r : responses) {
                    r.enqueue_time = now;
                    response_queue.push(std::move(r));
                }
                bl_flushed += batch.size();
            }
            while (true) {
                auto batch = broadcast_queue.try_pop_n(1000);
                if (batch.empty()) break;
                auto responses = batch_process_broadcasts(data, batch, metrics);
                auto now = steady_clock::now();
                for (auto& r : responses) {
                    if (r.response.empty()) continue;
                    r.enqueue_time = now;
                    response_queue.push(std::move(r));
                }
                bc_flushed += batch.size();
            }
            if (ins_flushed > 0 || bl_flushed > 0 || bc_flushed > 0) {
                std::cout << "[DRAIN] Flushed " << ins_flushed << " inserts, "
                          << bl_flushed << " backlinks, "
                          << bc_flushed << " broadcasts" << std::endl;
            }

            // DRAIN: flush + ACK
            ResponseItem resp;
            resp.identity = std::move(work.identity);
            resp.coordinator_id = std::move(work.coordinator_id);
            resp.needs_forward = false;
            uint8_t ack = static_cast<uint8_t>(MessageType::DRAIN_ACK);
            resp.response.assign(&ack, &ack + 1);
            resp.enqueue_time = steady_clock::now();
            response_queue.push(std::move(resp));
            continue;
        }

        // INSERT_REQUEST, BATCH_BACKLINK_REQUEST and VECTOR_BROADCAST never
        // reach this queue: the I/O thread routes them to the insert, backlink
        // and broadcast batch queues handled by process_ready_maintenance_batch.

        // ============================================================
        // SEARCH_REQUEST
        // ============================================================
        // Deserialize
        auto t_deser_start = steady_clock::now();
        // Measure dispatch gap: time from pickup to deserialization start
        uint32_t dispatch_gap_us = static_cast<uint32_t>(
            duration_cast<microseconds>(t_deser_start - pickup_time).count()
        );
        StateHeader header;
        std::vector<float> query;
        std::vector<BeamEntry> beam;

        // Skip first byte (MessageType) for SEARCH_REQUEST
        std::vector<uint32_t> incoming_seen_nodes;
        std::vector<RouteHop> incoming_route;
        if (!deserialize_state(work.request.data() + 1, work.request.size() - 1,
                               header, query, beam, incoming_seen_nodes, incoming_route)) {
            std::cerr << "Worker " << worker_id << ": Failed to deserialize\n";
            search_ownership.release();
            continue;
        }
        if (header.dim != data.dim || query.size() != data.dim) {
            // The scorers read data.dim floats of the query (exact, ADC table,
            // re-rank): a query of another dimension is refused, never read past.
            // It is answered with an ERROR reply (no beam, no result), so the
            // coordinator ends the command instead of waiting for a reply.
            std::cerr << "Worker " << worker_id << ": refused a query of dimension " << header.dim
                      << " (server dimension " << data.dim << ")\n";
            ResponseItem refused;
            refused.identity = std::move(work.identity);
            refused.coordinator_id = std::move(work.coordinator_id);
            refused.needs_forward = false;
            refused.target_partition = 0;
            refused.response = serialize_search_error(header.query_id, header.hop_count);
            refused.enqueue_time = steady_clock::now();
            response_queue.push(std::move(refused));
            search_ownership.release();
            continue;
        }
        auto t_deser_end = steady_clock::now();
        metrics.deserialize_us += duration_cast<microseconds>(t_deser_end - t_deser_start).count();

        // Process with timing
        auto t_search_start = steady_clock::now();
        StateStatus status;
        uint32_t next_partition = 0;
        std::vector<uint32_t> result_neighbors;
        std::vector<float> result_distances;
        ProcessMetrics pmetrics;
        uint64_t this_request_lock_wait_us = 0;  // Per-request lock wait

        // Always clear the stale state of the previous query
        for (uint32_t node : seen_nodes) {
            if (node < seen.size()) seen[node] = false;
        }
        seen_nodes.clear();

        if (!incoming_seen_nodes.empty()) {
            for (uint32_t node : incoming_seen_nodes) {
                if (node >= seen.size()) seen.resize(node + 1, false);
                if (!seen[node]) {
                    seen[node] = true;
                    seen_nodes.push_back(node);
                }
            }
        }

        {
            // Lock-free SEARCH. All read access (adj/g2l/l2g/
            // full_base/membership/node_locks) is to reallocation-free storage
            // (ChunkedArray + deque). RESET protection done by dispatch-top
            // shared_lock(g_admin_mutex) above.
            auto t_lock_start = steady_clock::now();
            auto t_lock_acquired = t_lock_start;
            uint64_t lock_wait_us = 0;
            this_request_lock_wait_us = 0;
            g_lock_metrics.shared_lock_count++;

            process_state(
                data,
                query.data(), header.dim,
                header.L, header.K, header.W,
                beam,
                status, next_partition,
                result_neighbors, result_distances,
                pmetrics,
                seen, seen_nodes,
                header.query_id, header.hop_count
            );
            auto t_lock_release = steady_clock::now();
            g_lock_metrics.search_shared.record(
                lock_wait_us,
                duration_cast<microseconds>(t_lock_release - t_lock_acquired).count());
        }
        auto t_search_end = steady_clock::now();

        // Calculate server processing time in microseconds
        uint32_t server_processing_us = static_cast<uint32_t>(
            duration_cast<microseconds>(t_search_end - t_search_start).count()
        );

        metrics.search_total_us += server_processing_us;
        metrics.beam_sort_us += pmetrics.beam_sort_us / 1000;
        metrics.full_distance_count += pmetrics.exact_distances + pmetrics.rerank_distances;
        metrics.pq_distance_count += pmetrics.adc_distances;

        // cumulative process_state sub-stage counters.
        g_lock_metrics.search_distance_us.fetch_add(
            pmetrics.distance_us / 1000, std::memory_order_relaxed);
        g_lock_metrics.search_beam_sort_us.fetch_add(
            pmetrics.beam_sort_us / 1000, std::memory_order_relaxed);
        g_lock_metrics.search_distance_count.fetch_add(
            pmetrics.distance_computations, std::memory_order_relaxed);
        g_lock_metrics.search_neighbors_evaluated.fetch_add(
            pmetrics.neighbors_evaluated, std::memory_order_relaxed);
        g_lock_metrics.search_exact_count.fetch_add(pmetrics.exact_distances, std::memory_order_relaxed);
        g_lock_metrics.search_adc_count.fetch_add(pmetrics.adc_distances, std::memory_order_relaxed);
        g_lock_metrics.search_rerank_count.fetch_add(pmetrics.rerank_distances, std::memory_order_relaxed);
        g_lock_metrics.search_missing_code_count.fetch_add(pmetrics.missing_codes, std::memory_order_relaxed);
        g_lock_metrics.search_adc_table_count.fetch_add(pmetrics.adc_tables, std::memory_order_relaxed);
        g_lock_metrics.search_adc_table_ns.fetch_add(pmetrics.adc_table_ns, std::memory_order_relaxed);
        g_lock_metrics.search_request_count.fetch_add(1, std::memory_order_relaxed);

        // Serialize response
        auto t_ser_start = steady_clock::now();
        // Measure pre-serialize gap: time from search end to serialize start
        uint32_t pre_serialize_gap_us = static_cast<uint32_t>(
            duration_cast<microseconds>(t_ser_start - t_search_end).count()
        );

        ResponseItem resp;
        resp.identity = std::move(work.identity);
        resp.coordinator_id = std::move(work.coordinator_id);
        resp.needs_forward = false;
        resp.target_partition = 0;

        // Check if P2P mode and need to forward
        bool is_p2p = (header.flags & STATE_FLAG_P2P) != 0;
        bool trace_route = (header.flags & STATE_FLAG_TRACE_ROUTE) != 0;

        // Build ResponseMetrics for observability
        ResponseMetrics resp_metrics;
        // Per hop: ADC evaluations, and full-precision ones (exact neighbour
        // scores plus own-only re-ranks). Under full replication pq is 0 and
        // full = distance_computations, as before.
        resp_metrics.pq_distance_count = pmetrics.adc_distances;
        resp_metrics.full_distance_count = pmetrics.exact_distances + pmetrics.rerank_distances;
        resp_metrics.beam_size_in = pmetrics.beam_size_in;
        resp_metrics.local_nodes_expanded = pmetrics.local_nodes_expanded;
        resp_metrics.top_W_count = pmetrics.top_W_count;
        resp_metrics.neighbors_evaluated = pmetrics.neighbors_evaluated;
        resp_metrics.queue_wait_us = static_cast<uint32_t>(queue_wait);

        uint32_t deser_us = static_cast<uint32_t>(
            duration_cast<microseconds>(t_deser_end - t_deser_start).count()
        );

        resp_metrics.zmq_recv_us = work.zmq_recv_us;
        resp_metrics.deserialize_us = deser_us;
        resp_metrics.queue_depth = work.queue_depth;
        resp_metrics.serialize_us = last_serialize_us;
        resp_metrics.zmq_send_us = io_timing.last_zmq_send_us.load(std::memory_order_relaxed);
        resp_metrics.response_queue_wait_us = io_timing.last_resp_queue_wait_us.load(std::memory_order_relaxed);
        resp_metrics.io_processing_us = work.io_processing_us;
        resp_metrics.send_prep_us = io_timing.last_send_prep_us.load(std::memory_order_relaxed);

        uint32_t total_processing_us = deser_us + server_processing_us;

        uint32_t new_cumulative_queue = header.cumulative_queue_us + static_cast<uint32_t>(queue_wait);
        uint32_t new_cumulative_processing = header.cumulative_processing_us + total_processing_us;
        const uint32_t new_cumulative_distance =
            header.cumulative_distance_computations +
            pmetrics.distance_computations;
        const uint32_t total_dist =
            is_p2p ? new_cumulative_distance : pmetrics.distance_computations;

        resp_metrics.server_processing_us = total_processing_us;
        resp_metrics.cumulative_queue_us = new_cumulative_queue;
        resp_metrics.cumulative_processing_us = new_cumulative_processing;

        // Per-request lock wait (not global cumulative)
        resp_metrics.shared_lock_count = 1;  // This request acquired 1 shared lock
        resp_metrics.unique_lock_count = 0;  // Search requests don't use unique locks
        resp_metrics.shared_lock_wait_us = static_cast<uint32_t>(this_request_lock_wait_us);
        resp_metrics.unique_lock_wait_us = 0;

        // Worker overhead: unmeasured gaps in worker thread processing
        // dispatch_gap: pickup → deser_start (type checking, dispatch code)
        // pre_serialize_gap: search_end → ser_start (metric calculations)
        resp_metrics.worker_overhead_us = dispatch_gap_us + pre_serialize_gap_us;

        // Append current hop to route with timing
        if (trace_route) {
            RouteHop this_hop;
            this_hop.partition = header.current_partition;
            this_hop.arrive_us = static_cast<uint64_t>(
                duration_cast<microseconds>(pickup_time.time_since_epoch()).count());
            this_hop.depart_us = static_cast<uint64_t>(
                duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count());
            incoming_route.push_back(this_hop);
        }

        if (is_p2p && status == StateStatus::NEED_HOP) {
            // P2P mode: forward state to next server
            resp.needs_forward = true;
            resp.target_partition = next_partition;

            // Same values as new_cumulative_* above (test_search_state_wire).
            resp.forward_state = serialize_forward_state(
                header,
                next_partition,
                query.data(),
                beam,
                static_cast<uint32_t>(queue_wait),
                total_processing_us,
                pmetrics.distance_computations,
                seen_nodes,
                incoming_route
            );

            resp.response = serialize_response(
                header.query_id,
                status,
                next_partition,
                beam,
                result_neighbors,
                result_distances,
                total_dist,
                header.hop_count + 1,
                resp_metrics,
                seen_nodes,  // NEED_HOP: include seen for next server
                incoming_route
            );
        } else if (status == StateStatus::NEED_HOP) {
            // Non-P2P NEED_HOP: the coordinator forwards the state to the
            // next server, so the response must carry seen_nodes forward.
            // Without them the next server would re-expand nodes already
            // visited, and a query could hop between servers without end.
            resp.response = serialize_response(
                header.query_id,
                status,
                next_partition,
                beam,
                result_neighbors,
                result_distances,
                total_dist,
                header.hop_count + 1,
                resp_metrics,
                seen_nodes,  // the visited set, for the next server
                incoming_route
            );
        } else {
            // DONE: seen_nodes not needed (final response).
            resp.response = serialize_response(
                header.query_id,
                status,
                next_partition,
                beam,
                result_neighbors,
                result_distances,
                total_dist,
                header.hop_count + 1,
                resp_metrics,
                {},  // no seen_nodes
                incoming_route
            );
        }

        auto t_ser_end = steady_clock::now();
        uint32_t this_serialize_us = static_cast<uint32_t>(
            duration_cast<microseconds>(t_ser_end - t_ser_start).count()
        );
        metrics.serialize_us += this_serialize_us;
        last_serialize_us = this_serialize_us;

        // Patch serialize_us into the wire buffer with this request's own
        // measured value (symmetric to HeaderPatch for response_queue_wait_us on
        // I/O thread side, but done by the worker before push because serialize
        // time is only known post-serialize).
        {
            MessageType resp_type = get_message_type(
                resp.response.data(), resp.response.size());
            if (resp_type == MessageType::SEARCH_RESPONSE &&
                resp.response.size() >= 1 + sizeof(StateResponseHeader)) {
                StateResponseHeader* hdr = reinterpret_cast<StateResponseHeader*>(
                    resp.response.data() + 1);
                hdr->serialize_us = this_serialize_us;
            }
        }

        // Enqueue response
        auto t_resp_start = steady_clock::now();
        resp.enqueue_time = t_resp_start;
        response_queue.push(std::move(resp));
        auto t_resp_end = steady_clock::now();
        metrics.response_queue_us += duration_cast<microseconds>(t_resp_end - t_resp_start).count();

        metrics.request_count++;
        processed_count++;
        search_ownership.release();
    }
}

// =============================================================================
// print_usage
// =============================================================================
void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --data-dir PATH     Data directory: membership.npy, partition_<rank>/{adj,local_to_global.npy}\n"
              << "                      and, by storage mode, pq_pivots.bin + pq_codes.bin +\n"
              << "                      partition_<rank>/vectors.npy (--pq) or base.npy (--full-vectors)\n"
              << "  --rank N            Server rank (0..K-1)\n"
              << "  --port PORT         ZMQ port (default: 29600)\n"
              << "  --threads N         Number of worker threads (default: 8)\n"
              << "  --num-servers K     Number of servers (required; every membership label must be below K)\n"
              << "  --insert-batch-size N     Batch size for insert (default: 16)\n"
              << "  --backlink-batch-size N   Batch size for backlink (default: 16)\n"
              << "  --broadcast-batch-size N  Batch size for broadcast (default: 16)\n"
              << "  --edge-overload N         Deferred prune slack: prune triggers at size > R + N (default: 16)\n"
              << "  --io-recv-budget N  Messages received per I/O loop iteration (default: 1)\n"
              << "  --io-send-budget N  Responses sent per I/O loop iteration, 0 = all queued (default: 0)\n"
              << "  --pq                Hybrid PQ own-only storage (the default): original vectors of the owned\n"
              << "                      nodes, a PQ code of every node; owned neighbours scored exactly, others\n"
              << "                      with ADC, every expanded node re-scored exactly (exact final top-K)\n"
              << "  --full-vectors      Full replication: every vector from base.npy, every distance exact\n"
              << "  --print-wire-contract  Print the compiled metric and wire contract as JSON and exit\n"
              << "  --peers LIST        P2P mode: comma-separated addresses of all K servers in rank order,\n"
              << "                      this server's own included (e.g., tcp://node0:28800,tcp://node1:28801,...)\n"
              << "  --help              Show this help\n"
              << "\n"
              << "The search width per server visit is the coordinator's --W (--insert-W for insert searches),\n"
              << "sent with every search state.\n"
              << "P2P mode: When --peers is specified, servers forward states directly to each other.\n"
              << "Note: loads only this partition's graph rows (partition_<rank>/adj).\n";
}

// =============================================================================
// main
// =============================================================================
int main(int argc, char* argv[]) {
    std::string data_dir;
    std::string peers_str;
    int rank = -1;
    int port = 29600;
    int num_threads = 8;
    int num_servers = -1;  // required: no default (a wrong K folds owner labels onto other servers)
    int insert_batch_size = 16;
    int backlink_batch_size = 16;
    int broadcast_batch_size = 16;
    int edge_overload = 16;  // deferred prune slack (size > R + edge_overload triggers prune)
    int io_recv_budget = 1;
    int io_send_budget = 0;
    bool print_wire_contract = false;
    int storage_flag = 0;  // 1 = --pq, 2 = --full-vectors; 0 = default (own-only PQ)
    bool storage_conflict = false;

    static struct option long_options[] = {
        {"data-dir", required_argument, 0, 'd'},
        {"rank", required_argument, 0, 'r'},
        {"port", required_argument, 0, 'p'},
        {"threads", required_argument, 0, 't'},
        {"num-servers", required_argument, 0, 'n'},
        {"insert-batch-size", required_argument, 0, 1003},
        {"backlink-batch-size", required_argument, 0, 1001},
        {"broadcast-batch-size", required_argument, 0, 1002},
        {"edge-overload", required_argument, 0, 1008},
        {"io-recv-budget", required_argument, 0, 1004},
        {"io-send-budget", required_argument, 0, 1005},
        {"print-wire-contract", no_argument, 0, 1013},
        {"pq", no_argument, 0, 1014},
        {"full-vectors", no_argument, 0, 1015},
        {"peers", required_argument, 0, 'P'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:r:p:t:n:P:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'd': data_dir = optarg; break;
            case 'r': rank = std::atoi(optarg); break;
            case 'p': port = std::atoi(optarg); break;
            case 't': num_threads = std::atoi(optarg); break;
            case 'n': num_servers = std::atoi(optarg); break;
            case 1001: backlink_batch_size = std::atoi(optarg); break;
            case 1002: broadcast_batch_size = std::atoi(optarg); break;
            case 1003: insert_batch_size = std::atoi(optarg); break;
            case 1004: io_recv_budget = std::max(1, std::atoi(optarg)); break;
            case 1005: io_send_budget = std::max(0, std::atoi(optarg)); break;
            case 1008: edge_overload = std::max(0, std::atoi(optarg)); break;
            case 1013: print_wire_contract = true; break;
            case 1014: storage_conflict |= storage_flag == 2; storage_flag = 1; break;
            case 1015: storage_conflict |= storage_flag == 1; storage_flag = 2; break;
            case 'P': peers_str = optarg; break;
            case 'h':
            default:
                print_usage(argv[0]);
                return opt == 'h' ? 0 : 1;
        }
    }

    if (print_wire_contract) {
        std::cout << "{\"metric\":\"" << kMetricName
                  << "\",\"wire_contract\":\""
                  << ann::state_wire_contract_token() << "\"}\n";
        return 0;
    }
    if (storage_conflict) {
        std::cerr << "Error: --pq and --full-vectors are exclusive\n";
        return 1;
    }
    if (data_dir.empty() || rank < 0 || num_servers < 1) {
        std::cerr << "Error: --data-dir, --rank and --num-servers are required\n";
        print_usage(argv[0]);
        return 1;
    }
    if (rank >= num_servers) {
        std::cerr << "Error: --rank " << rank << " is not below --num-servers " << num_servers << "\n";
        return 1;
    }
    if (num_threads <= 0) {
        std::cerr << "Error: --threads must be positive\n";
        return 1;
    }

    // Parse peer addresses for P2P mode
    std::vector<std::string> peers;
    bool p2p_mode = !peers_str.empty();
    if (p2p_mode) {
        peers = parse_servers(peers_str);
        if (peers.size() != static_cast<size_t>(num_servers)) {
            std::cerr << "Error: --peers names " << peers.size() << " servers, --num-servers " << num_servers << "\n";
            return 1;
        }
        std::cout << "P2P Mode enabled with " << peers.size() << " peers" << std::endl;
    }

    std::cout << "=== State Passing Server (In-Memory) ===" << std::endl;
    std::cout << "Rank: " << rank << ", Port: " << port << ", Workers: " << num_threads << std::endl;
    std::cout << "Insert Batch Size: " << insert_batch_size << std::endl;
    std::cout << "Backlink Batch Size: " << backlink_batch_size << std::endl;
    std::cout << "Broadcast Batch Size: " << broadcast_batch_size << std::endl;
    std::cout << "Edge Overload: " << edge_overload << " (prune triggers at size > R + overload)" << std::endl;
    std::cout << "I/O Recv Budget: " << io_recv_budget << std::endl;
    std::cout << "I/O Send Budget: " << io_send_budget << " (0=all)" << std::endl;
    std::cout << "Storage: " << (storage_flag == 2 ? "full replication (--full-vectors)"
                                                   : "hybrid PQ own-only (--pq)") << std::endl;

    // Set globals for the reset handler
    g_data_dir = data_dir;
    g_rank = rank;

    // Load data using shared function
    std::cout << "\nLoading data..." << std::endl;
    ServerData data;
    data.my_partition = static_cast<uint32_t>(rank);
    data.n_servers = static_cast<uint32_t>(num_servers);
    data.edge_overload = static_cast<uint32_t>(edge_overload);
    data.own_only = storage_flag != 2;

    uint32_t start_node = 0;
    try {
        start_node = load_server_data(data, data_dir, rank);
    } catch (const std::exception& e) {
        std::cerr << "Error: cannot load " << data_dir << ": " << e.what() << "\n";
        return 1;
    }
    g_start_node = start_node;

    uint32_t init_n_base = n_base_load(data);
    const StorageBytes init_bytes = storage_bytes(data);
    if (data.own_only) {
        std::cout << "  Vectors: " << init_n_base << " x " << data.dim << " (own-only: " << n_local_load(data)
                  << " own vectors, " << (init_bytes.vector_bytes / 1024.0 / 1024.0) << " MB allocated; PQ M="
                  << data.pq_table.M << " codes " << (init_bytes.code_bytes / 1024.0 / 1024.0) << " MB, tables "
                  << (init_bytes.pq_table_bytes / 1024.0 / 1024.0) << " MB)" << std::endl;
    } else {
        std::cout << "  Vectors: " << init_n_base << " x " << data.dim
                  << " (" << (init_bytes.vector_bytes / 1024.0 / 1024.0) << " MB, full replication)" << std::endl;
    }
    std::cout << "  Membership: " << data.membership.size() << " nodes" << std::endl;
    uint32_t init_n_local = n_local_load(data);
    std::cout << "  Partition graph: " << init_n_local << " local nodes (vs " << init_n_base << " total)" << std::endl;

    uint32_t local_count = 0;
    for (size_t i = 0; i < data.membership.size(); i++) {
        if (data.membership[i] == data.my_partition) local_count++;
    }
    std::cout << "  My partition (" << data.my_partition << "): " << local_count << " nodes" << std::endl;

    // Setup queues
    ThreadSafeQueue<WorkItem> work_queue;          // Search requests
    ThreadSafeQueue<WorkItem> insert_queue;        // Insert requests (batched)
    ThreadSafeQueue<WorkItem> backlink_queue;      // Backlink requests (batched)
    ThreadSafeQueue<WorkItem> broadcast_queue;     // Broadcast requests (batched)
    ThreadSafeQueue<ResponseItem> response_queue;
    std::atomic<uint32_t> maintenance_in_flight{0};
    std::atomic<uint32_t> search_in_flight{0};

    // Signal pipe: workers wake up I/O thread via response_queue.push() → pipe write
    int signal_pipe[2];
    if (pipe(signal_pipe) != 0) {
        perror("pipe");
        return 1;
    }
    fcntl(signal_pipe[0], F_SETFL, O_NONBLOCK);
    fcntl(signal_pipe[1], F_SETFL, O_NONBLOCK);
    response_queue.set_signal_fd(signal_pipe[1]);

    std::atomic<bool> running{true};
    std::atomic<uint64_t> processed_count{0};
    TimingMetrics metrics;
    IOTimingState io_timing;

    // Setup ZMQ
    zmq::context_t context(1);
    zmq::socket_t socket = create_server_socket(context, port);
    std::cout << "Listening on tcp://*:" << port << " (in-memory mode)" << std::endl;


    // Start workers
    std::cout << "\nStarting " << num_threads << " worker threads..." << std::endl;
    std::vector<std::thread> workers;
    for (int i = 0; i < num_threads; i++) {
        workers.emplace_back(worker_thread, i, std::ref(data),
                             std::ref(work_queue), std::ref(backlink_queue),
                             std::ref(broadcast_queue), std::ref(insert_queue),
                             std::ref(response_queue),
                             std::ref(running), std::ref(processed_count),
                             std::ref(metrics), std::ref(io_timing),
                             std::ref(maintenance_in_flight),
                             std::ref(search_in_flight),
                             static_cast<uint32_t>(insert_batch_size),
                             static_cast<uint32_t>(backlink_batch_size),
                             static_cast<uint32_t>(broadcast_batch_size));
    }

    // P2P: Setup peer connections
    std::vector<std::unique_ptr<zmq::socket_t>> peer_sockets(static_cast<size_t>(num_servers));
    if (p2p_mode) {
        std::cout << "Connecting to peers..." << std::endl;
        for (size_t i = 0; i < peers.size(); i++) {
            auto sock = std::make_unique<zmq::socket_t>(context, zmq::socket_type::dealer);
            sock->set(zmq::sockopt::linger, 0);
            sock->set(zmq::sockopt::sndtimeo, 100);
            sock->set(zmq::sockopt::sndhwm, 10000);
            sock->set(zmq::sockopt::rcvhwm, 10000);
            std::string identity = "peer-" + std::to_string(rank) + "-to-" + std::to_string(i);
            sock->set(zmq::sockopt::routing_id, identity);
            sock->connect(peers[i]);
            // --peers names exactly K servers (checked above): peer i is rank i.
            const size_t partition = i;
            std::cout << "  Connected to peer " << i
                      << " (partition " << partition << "): "
                      << peers[i] << std::endl;
            peer_sockets[partition] = std::move(sock);
        }
    }

    // Timing
    auto start_time = steady_clock::now();
    uint64_t recv_count = 0;
    uint64_t send_count = 0;
    uint64_t p2p_forward_count = 0;
    uint64_t last_stats_count = 0;
    const uint32_t recv_budget = static_cast<uint32_t>(io_recv_budget);
    const uint32_t send_budget = static_cast<uint32_t>(io_send_budget);
    tide::StatsWindowSnapshotTask snapshot_task;
    // This is the only completion consumer and owns every socket operation.
    // wait=true is reserved for ordered control/shutdown, never query traffic.
    auto finish_snapshot = [&](bool wait) {
        if (!snapshot_task.pending() || (!wait && !snapshot_task.ready())) return;
        tide::StatsWindowSnapshotReply reply;
        if (!snapshot_task.finish(reply, wait)) return;
        zmq::message_t id(reply.identity.data(), reply.identity.size());
        zmq::message_t empty(0);
        zmq::message_t body(reply.response.data(), reply.response.size());
        const auto a = socket.send(id, zmq::send_flags::sndmore);
        const auto b = a ? socket.send(empty, zmq::send_flags::sndmore) : zmq::send_result_t{};
        const auto c = b ? socket.send(body, zmq::send_flags::none) : zmq::send_result_t{};
        if (!(c && *c == reply.response.size()))
            throw std::runtime_error("async stats-window response send failed");
    };

    // P2P: Track query origins for response routing
    std::unordered_map<uint32_t, std::string> query_origins;
    std::mutex query_origins_mutex;

    // Main I/O loop with zmq::poll + signal pipe (event-driven wake-up)
    zmq::pollitem_t poll_items[] = {
        {socket, 0, ZMQ_POLLIN, 0},
        {nullptr, signal_pipe[0], ZMQ_POLLIN, 0}
    };

    bool graceful_shutdown = false;
    while (!graceful_shutdown) {
        finish_snapshot(false);
        zmq::poll(poll_items, 2, std::chrono::milliseconds(-1));  // block until event

        // Drain signal pipe
        if (poll_items[1].revents & ZMQ_POLLIN) {
            char buf[256];
            while (::read(signal_pipe[0], buf, sizeof(buf)) > 0) {}
        }
        finish_snapshot(false);

        // Receive if data available
        if (poll_items[0].revents & ZMQ_POLLIN) {
            for (uint32_t recv_attempt = 0;
                 recv_attempt < recv_budget;
                 recv_attempt++) {
                zmq::message_t identity;
                zmq::message_t empty_frame;
                zmq::message_t request;

                try {
                    auto recv_start = steady_clock::now();

                    auto recv_id = socket.recv(identity, zmq::recv_flags::dontwait);
                    if (!recv_id) break;

                    socket.recv(empty_frame, zmq::recv_flags::none);

                    WorkItem work;
                    work.identity = std::string(static_cast<char*>(identity.data()), identity.size());
                    work.from_peer = (work.identity.substr(0, 5) == "peer-");

                    if (work.from_peer && p2p_mode) {
                        zmq::message_t coord_id_frame;
                        socket.recv(coord_id_frame, zmq::recv_flags::none);
                        socket.recv(request, zmq::recv_flags::none);
                        work.coordinator_id = std::string(static_cast<char*>(coord_id_frame.data()), coord_id_frame.size());
                    } else {
                        socket.recv(request, zmq::recv_flags::none);
                        work.coordinator_id = work.identity;
                    }

                    auto recv_end = steady_clock::now();
                    work.zmq_recv_us = static_cast<uint32_t>(
                        duration_cast<microseconds>(recv_end - recv_start).count()
                    );

                    work.request.resize(request.size());
                    memcpy(work.request.data(), request.data(), request.size());

                    MessageType msg_type = get_message_type(
                        static_cast<const char*>(request.data()), request.size());
                    work.msg_type = msg_type;
                    // Every queue empty and no search or maintenance work in
                    // flight. Migration REFRESH/PREPARE/COMMIT/ROLLBACK/ACCEPT
                    // require it; SHUTDOWN checks the same condition.
                    auto server_quiescent = [&]() {
                        return ann::shutdown_state_is_quiescent(
                            work_queue.size(), insert_queue.size(),
                            backlink_queue.size(), broadcast_queue.size(),
                            response_queue.size(),
                            maintenance_in_flight.load(std::memory_order_acquire),
                            search_in_flight.load(std::memory_order_acquire));
                    };

                    auto send_migration_ack = [&](MessageType response_type,
                                                   uint8_t status,
                                                   uint64_t epoch_from,
                                                   uint64_t epoch_to,
                                                   uint64_t move_count,
                                                   uint16_t request_id) {
                        ann::MembershipMigrationAck ack{};
                        ack.msg_type = response_type;
                        ack.status = status;
                        ack.request_id = request_id;
                        ack.server_rank = static_cast<uint32_t>(g_rank);
                        ack.epoch_from = epoch_from;
                        ack.epoch_to = epoch_to;
                        ack.move_count = move_count;
                        zmq::message_t ack_identity(
                            work.identity.data(), work.identity.size());
                        zmq::message_t ack_empty(0);
                        zmq::message_t ack_data(&ack, sizeof(ack));
                        socket.send(ack_identity, zmq::send_flags::sndmore);
                        socket.send(ack_empty, zmq::send_flags::sndmore);
                        socket.send(ack_data, zmq::send_flags::none);
                    };

                    auto send_migration_refresh = [&] (
                        MessageType response_type, uint8_t status,
                        uint64_t epoch_from,
                        uint64_t epoch_to, uint64_t move_count,
                        const std::vector<char>& payload) {
                        ann::MembershipMigrationRefreshHeader header{};
                        header.msg_type = response_type;
                        header.status = status;
                        header.server_rank = static_cast<uint32_t>(g_rank);
                        header.epoch_from = epoch_from;
                        header.epoch_to = epoch_to;
                        header.move_count = move_count;
                        header.payload_bytes = payload.size();
                        std::vector<char> response(sizeof(header) + payload.size());
                        std::memcpy(response.data(), &header, sizeof(header));
                        if (!payload.empty()) {
                            std::memcpy(response.data() + sizeof(header),
                                        payload.data(), payload.size());
                        }
                        zmq::message_t ack_identity(
                            work.identity.data(), work.identity.size());
                        zmq::message_t ack_empty(0);
                        zmq::message_t ack_data(
                            response.data(), response.size());
                        socket.send(ack_identity, zmq::send_flags::sndmore);
                        socket.send(ack_empty, zmq::send_flags::sndmore);
                        socket.send(ack_data, zmq::send_flags::none);
                    };

                    if (msg_type == MessageType::STATS_WINDOW_BEGIN_REQUEST ||
                        msg_type == MessageType::STATS_WINDOW_SNAPSHOT_REQUEST ||
                        msg_type == MessageType::STATS_WINDOW_STOP_REQUEST) {
                        // Serialize window controls, including their replies.
                        // The normal coordinator awaits SNAPSHOT before BEGIN;
                        // an overlapping caller cannot overtake its epoch.
                        finish_snapshot(true);
                        // Runtime window statistics. Unlike the other
                        // control handlers these run WITHOUT quiescence:
                        // windows measure live serving. Counters are relaxed
                        // atomics, so zeroing/reading races only smear the
                        // window boundary by in-flight increments.
                        ann::StatsWindowAck ack{};
                        ack.msg_type =
                            msg_type == MessageType::STATS_WINDOW_BEGIN_REQUEST
                                ? MessageType::STATS_WINDOW_BEGIN_ACK
                                : MessageType::STATS_WINDOW_HARVEST_ACK;
                        ack.status = 1;
                        ack.server_rank = static_cast<uint32_t>(g_rank);
                        // Cumulative-at-ack search counters; the coordinator
                        // differences begin vs harvest acks for the window's
                        // measured coefficients.
                        ack.search_request_count =
                            g_lock_metrics.search_request_count.load();
                        ack.search_distance_count =
                            g_lock_metrics.search_distance_count.load();
                        ack.search_distance_us =
                            g_lock_metrics.search_distance_us.load();
                        ack.search_total_us = metrics.search_total_us.load();
                        ack.search_exact_count = g_lock_metrics.search_exact_count.load();
                        ack.search_adc_count = g_lock_metrics.search_adc_count.load();
                        ack.search_rerank_count = g_lock_metrics.search_rerank_count.load();
                        ack.search_adc_table_count = g_lock_metrics.search_adc_table_count.load();
                        std::vector<char> response;
                        if (msg_type == MessageType::STATS_WINDOW_STOP_REQUEST) {
                            tide::WindowStopRequest stop{};
                            if (request.size() == sizeof(stop)) std::memcpy(&stop, request.data(), sizeof(stop));
                            else stop.version = 0;
                            const auto stopped = tide::stats_window_stop_local(data, stop, g_rank);
                            response.resize(sizeof(stopped));
                            std::memcpy(response.data(), &stopped, sizeof(stopped));
                        } else if (msg_type == MessageType::STATS_WINDOW_SNAPSHOT_REQUEST) {
                            tide::WindowSnapshotRequest snapshot{};
                            if (request.size() == sizeof(snapshot)) std::memcpy(&snapshot, request.data(), sizeof(snapshot));
                            else snapshot.version = 0;
                            auto captured = tide::stats_window_capture_snapshot(data, snapshot, ack);
                            if (!snapshot_task.start(work.identity, std::move(captured), signal_pipe[1]))
                                throw std::logic_error("snapshot slot not drained before capture");
                            break;
                        } else if (msg_type == MessageType::STATS_WINDOW_BEGIN_REQUEST
                            && request.size() ==
                                   sizeof(ann::StatsWindowBeginRequest)) {
                            ann::StatsWindowBeginRequest begin{};
                            std::memcpy(&begin, request.data(), sizeof(begin));
                            if (tide::stats_window_begin_local(data, begin)) {
                                ack.status = 0;
                                ack.window_epoch = begin.window_epoch;
                            }
                        }
                        if (response.empty()) {
                            response.resize(sizeof(ack));
                            std::memcpy(response.data(), &ack, sizeof(ack));
                        }
                        zmq::message_t ack_identity(
                            work.identity.data(), work.identity.size());
                        zmq::message_t ack_empty(0);
                        zmq::message_t ack_data(
                            response.data(), response.size());
                        const auto sent_identity = socket.send(ack_identity, zmq::send_flags::sndmore);
                        const auto sent_empty = sent_identity ? socket.send(ack_empty, zmq::send_flags::sndmore) : zmq::send_result_t{};
                        const auto sent_body = sent_empty ? socket.send(ack_data, zmq::send_flags::none) : zmq::send_result_t{};
                        if (!(sent_body && *sent_body == response.size()))
                            throw std::runtime_error("stats-window response send failed");
                        break;
                    }
                    // The coordinator sends every plan inline, in the request.
                    if (msg_type == MessageType::MIGRATION_REFRESH_INLINE_REQUEST) {
                        uint8_t status = 1;
                        uint64_t epoch_from = g_ownership_epoch;
                        uint64_t epoch_to = g_ownership_epoch;
                        uint64_t move_count = 0;
                        std::vector<char> payload;
                        std::string migration_error =
                            "malformed migration REFRESH";
                        const char* request_bytes =
                            static_cast<const char*>(request.data());
                        if (request.size() > 1) {
                            std::unique_lock<std::shared_mutex> admin_lock(
                                tide::g_admin_mutex);
                            MembershipMigrationPlan move_template;
                            MembershipMigrationPlan source_fragment;
                            if (!server_quiescent()) {
                                migration_error =
                                    "migration REFRESH requires quiescence";
                            } else if (!load_membership_migration_plan_bytes(
                                           request_bytes + 1, request.size() - 1, n_base_load(data),
                                           data.n_servers, move_template, migration_error)) {
                                // Error already populated.
                            } else if (move_template.epoch_from !=
                                       g_ownership_epoch) {
                                migration_error =
                                    "migration REFRESH ownership epoch mismatch";
                            } else if (!snapshot_membership_migration_source_rows(
                                           data, move_template,
                                           source_fragment,
                                           migration_error)) {
                                // Error already populated.
                            } else {
                                epoch_from = move_template.epoch_from;
                                epoch_to = move_template.epoch_to;
                                move_count = source_fragment.moves.size();
                                if (move_count == 0 ||
                                    serialize_membership_migration_plan(
                                        source_fragment, payload,
                                        migration_error)) {
                                    status = 0;
                                }
                            }
                        }
                        if (status != 0) {
                            std::cerr << "[MIGRATION] REFRESH rejected: "
                                      << migration_error << std::endl;
                        }
                        send_migration_refresh(
                            MessageType::MIGRATION_REFRESH_RESPONSE,
                            status, epoch_from, epoch_to, move_count, payload);
                        break;
                    }

                    if (msg_type == MessageType::MIGRATION_PREPARE_INLINE_REQUEST) {
                        uint8_t status = 1;
                        uint64_t epoch_from = g_ownership_epoch;
                        uint64_t epoch_to = g_ownership_epoch;
                        uint64_t move_count = 0;
                        std::string migration_error =
                            "malformed migration PREPARE";
                        if (request.size() > 1) {
                            std::unique_lock<std::shared_mutex> admin_lock(
                                tide::g_admin_mutex);
                            if (!server_quiescent()) {
                                migration_error =
                                    "migration PREPARE requires quiescence";
                            } else {
                                MembershipMigrationPlan plan;
                                MembershipMigrationJournal journal;
                                const bool loaded = load_membership_migration_plan_bytes(
                                    static_cast<const char*>(request.data()) + 1,
                                    request.size() - 1, n_base_load(data),
                                    data.n_servers, plan, migration_error);
                                if (!loaded) {
                                    // Error already populated.
                                } else if (plan.epoch_from !=
                                           g_ownership_epoch) {
                                    migration_error =
                                        "migration PREPARE ownership epoch mismatch";
                                } else if (plan.vector_target != kMigrationNoVectorTarget &&
                                           plan.vector_target != static_cast<uint32_t>(g_rank)) {
                                    // A per-target fragment carries another
                                    // server's vectors: never install them here.
                                    migration_error =
                                        "migration PREPARE fragment is addressed to server " +
                                        std::to_string(plan.vector_target);
                                } else if (prepare_membership_migration(
                                               data, plan.moves,
                                               plan.epoch_from, plan.epoch_to,
                                               journal, migration_error,
                                               plan.vector_dim > 0)) {
                                    epoch_from = plan.epoch_from;
                                    epoch_to = plan.epoch_to;
                                    move_count = plan.moves.size();
                                    // The rows and vectors are installed; COMMIT,
                                    // ROLLBACK and ACCEPT need only the digests.
                                    strip_membership_migration_payload(plan);
                                    g_membership_migration_plan =
                                        std::move(plan);
                                    g_membership_migration_journal =
                                        std::move(journal);
                                    status = 0;
                                }
                            }
                        }
                        if (status != 0) {
                            std::cerr << "[MIGRATION] PREPARE rejected: "
                                      << migration_error << std::endl;
                        }
                        send_migration_ack(
                            MessageType::MIGRATION_PREPARE_ACK, status,
                            epoch_from, epoch_to, move_count, 0);
                        break;
                    }

                    if (msg_type == MessageType::MIGRATION_COMMIT_REQUEST) {
                        uint8_t status = 1;
                        std::string migration_error =
                            "malformed migration COMMIT";
                        uint64_t epoch_from = g_ownership_epoch;
                        uint64_t epoch_to = g_ownership_epoch;
                        uint64_t move_count =
                            g_membership_migration_plan.moves.size();
                        if (request.size() == 1) {
                            std::unique_lock<std::shared_mutex> admin_lock(
                                tide::g_admin_mutex);
                            if (!server_quiescent()) {
                                migration_error =
                                    "migration COMMIT requires quiescence";
                            } else if (g_membership_migration_plan.moves.empty()) {
                                migration_error =
                                    "migration COMMIT has no prepared plan";
                            } else {
                                epoch_from =
                                    g_membership_migration_plan.epoch_from;
                                epoch_to = g_membership_migration_plan.epoch_to;
                                if (commit_membership_migration(
                                        data,
                                        g_membership_migration_plan.moves,
                                        g_membership_migration_journal,
                                        migration_error)) {
                                    g_ownership_epoch = epoch_to;
                                    status = 0;
                                }
                            }
                        }
                        if (status != 0) {
                            std::cerr << "[MIGRATION] COMMIT rejected: "
                                      << migration_error << std::endl;
                        }
                        send_migration_ack(
                            MessageType::MIGRATION_COMMIT_ACK, status,
                            epoch_from, epoch_to, move_count, 0);
                        break;
                    }

                    if (msg_type == MessageType::MIGRATION_ROLLBACK_REQUEST) {
                        uint8_t status = 1;
                        std::string migration_error =
                            "malformed migration ROLLBACK";
                        uint64_t epoch_from = g_ownership_epoch;
                        uint64_t epoch_to = g_ownership_epoch;
                        uint64_t move_count =
                            g_membership_migration_plan.moves.size();
                        // The request names the plan: type, epoch_from, epoch_to.
                        if (request.size() == 1 + 2 * sizeof(uint64_t)) {
                            uint64_t requested_from = 0, requested_to = 0;
                            std::memcpy(&requested_from,
                                        static_cast<const char*>(request.data()) + 1,
                                        sizeof(uint64_t));
                            std::memcpy(&requested_to,
                                        static_cast<const char*>(request.data()) + 1 +
                                            sizeof(uint64_t),
                                        sizeof(uint64_t));
                            std::unique_lock<std::shared_mutex> admin_lock(
                                tide::g_admin_mutex);
                            const auto disposition =
                                membership_migration_rollback_disposition(
                                    g_membership_migration_plan,
                                    g_accepted_migration_epochs,
                                    g_ownership_epoch, requested_from,
                                    requested_to);
                            if (!server_quiescent()) {
                                migration_error =
                                    "migration ROLLBACK requires quiescence";
                            } else if (disposition ==
                                       MigrationRollbackDisposition::REFUSE_ACCEPTED) {
                                migration_error =
                                    "migration ROLLBACK of an accepted epoch refused";
                            } else if (disposition ==
                                       MigrationRollbackDisposition::REFUSE_MISMATCH) {
                                migration_error =
                                    "migration ROLLBACK names another plan or epoch";
                            } else if (disposition ==
                                       MigrationRollbackDisposition::NO_OP) {
                                // Idempotent cleanup after a partial distributed
                                // PREPARE failure: this server never prepared it.
                                status = 0;
                            } else {
                                epoch_from =
                                    g_membership_migration_plan.epoch_from;
                                epoch_to = g_membership_migration_plan.epoch_to;
                                if (rollback_membership_migration(
                                        data,
                                        g_membership_migration_plan.moves,
                                        g_membership_migration_journal,
                                        migration_error)) {
                                    g_ownership_epoch = epoch_from;
                                    g_membership_migration_plan =
                                        MembershipMigrationPlan{};
                                    g_membership_migration_journal =
                                        MembershipMigrationJournal{};
                                    status = 0;
                                }
                            }
                        }
                        if (status != 0) {
                            std::cerr << "[MIGRATION] ROLLBACK rejected: "
                                      << migration_error << std::endl;
                        }
                        send_migration_ack(
                            MessageType::MIGRATION_ROLLBACK_ACK, status,
                            epoch_from, epoch_to, move_count, 0);
                        break;
                    }

                    if (msg_type == MessageType::MIGRATION_ACCEPT_REQUEST) {
                        uint8_t status = 1;
                        std::string migration_error =
                            "malformed migration ACCEPT";
                        uint64_t epoch_from = g_ownership_epoch;
                        uint64_t epoch_to = g_ownership_epoch;
                        uint64_t move_count =
                            g_membership_migration_plan.moves.size();
                        // The request carries the coordinator's attempt id
                        // (type, u16); the ACK echoes it, so a reply that
                        // arrives after its attempt timed out is never
                        // counted for a later attempt.
                        uint16_t request_id = 0;
                        if (request.size() == 1 + sizeof(request_id)) {
                            std::memcpy(&request_id,
                                        static_cast<const char*>(request.data()) + 1,
                                        sizeof(request_id));
                            std::unique_lock<std::shared_mutex> admin_lock(
                                tide::g_admin_mutex);
                            if (!server_quiescent()) {
                                migration_error =
                                    "migration ACCEPT requires quiescence";
                            } else if (membership_migration_accept_is_repeat(
                                           g_membership_migration_plan,
                                           g_accepted_migration_epochs,
                                           g_ownership_epoch)) {
                                // A retried ACCEPT of the epoch this server
                                // already accepted (the coordinator completes a
                                // partial ACCEPT): success, move_count 0.
                                epoch_from = g_ownership_epoch - 1;
                                epoch_to = g_ownership_epoch;
                                move_count = 0;
                                status = 0;
                            } else if (g_membership_migration_plan.moves.empty()) {
                                migration_error =
                                    "migration ACCEPT has no committed plan";
                            } else {
                                epoch_from =
                                    g_membership_migration_plan.epoch_from;
                                epoch_to =
                                    g_membership_migration_plan.epoch_to;
                                if (accept_membership_migration_history(
                                        data, g_membership_migration_plan,
                                        g_membership_migration_journal,
                                        g_accepted_migration_epochs,
                                        migration_error)) {
                                    status = 0;
                                }
                            }
                        }
                        if (status != 0) {
                            std::cerr << "[MIGRATION] ACCEPT rejected: "
                                      << migration_error << std::endl;
                        }
                        send_migration_ack(
                            MessageType::MIGRATION_ACCEPT_ACK, status,
                            epoch_from, epoch_to, move_count, request_id);
                        break;
                    }

                    if (msg_type == MessageType::MEMBERSHIP_CHECK_REQUEST) {
                        // membership-check: this server's state against the
                        // coordinator's owner of every id.
                        ann::MembershipCheckResponse response{};
                        response.msg_type = MessageType::MEMBERSHIP_CHECK_RESPONSE;
                        response.status = 2;
                        response.server_rank = static_cast<uint32_t>(g_rank);
                        response.own_only = data.own_only ? 1 : 0;
                        response.first_mismatch = UINT32_MAX;
                        uint32_t ids = 0;
                        if (request.size() >= 1 + sizeof(ids)) {
                            std::memcpy(&ids, static_cast<const char*>(request.data()) + 1, sizeof(ids));
                        }
                        if (request.size() >= 1 + sizeof(ids) &&
                            request.size() == 1 + sizeof(ids) + uint64_t{ids} * sizeof(uint32_t)) {
                            std::unique_lock<std::shared_mutex> admin_lock(tide::g_admin_mutex);
                            if (!server_quiescent()) {
                                response.status = 1;
                            } else {
                                std::vector<uint32_t> owners(ids);
                                if (ids) {
                                    std::memcpy(owners.data(),
                                                static_cast<const char*>(request.data()) + 1 + sizeof(ids),
                                                uint64_t{ids} * sizeof(uint32_t));
                                }
                                const auto counts = tide::membership_check(data, owners.data(), ids);
                                response.status = 0;
                                response.pool_ok = counts.pool_ok ? 1 : 0;
                                response.ids = counts.ids;
                                response.server_ids = counts.server_ids;
                                response.n_local = counts.n_local;
                                response.free_slots = counts.free_slots;
                                response.first_mismatch = counts.first_mismatch;
                                response.ownership_epoch = g_ownership_epoch;
                                response.owned = counts.owned;
                                response.mapped = counts.mapped;
                                response.vectors = counts.vectors;
                                response.membership_mismatches = counts.membership_mismatches;
                                response.mapping_mismatches = counts.mapping_mismatches;
                                response.vector_mismatches = counts.vector_mismatches;
                                response.missing_codes = counts.missing_codes;
                                response.vector_digest = counts.vector_digest;
                            }
                        }
                        zmq::message_t check_identity(work.identity.data(), work.identity.size());
                        zmq::message_t check_empty(0);
                        zmq::message_t check_data(&response, sizeof(response));
                        socket.send(check_identity, zmq::send_flags::sndmore);
                        socket.send(check_empty, zmq::send_flags::sndmore);
                        socket.send(check_data, zmq::send_flags::none);
                        break;
                    }

                    // ============================================================
                    // Admin commands: RESET (handled in I/O thread)
                    // ============================================================
                    if (msg_type == MessageType::SHUTDOWN_REQUEST) {
                        finish_snapshot(true);
                        std::unique_lock<std::shared_mutex> admin_lock(
                            tide::g_admin_mutex);
                        const bool quiescent =
                            ann::shutdown_state_is_quiescent(
                                work_queue.size(), insert_queue.size(),
                                backlink_queue.size(),
                                broadcast_queue.size(),
                                response_queue.size(),
                                maintenance_in_flight.load(
                                    std::memory_order_acquire),
                                search_in_flight.load(
                                    std::memory_order_acquire));
                        if (!quiescent) {
                            std::cerr
                                << "[SHUTDOWN] Refusing ACK: server is not "
                                   "quiescent"
                                << " work_q=" << work_queue.size()
                                << " ins_q=" << insert_queue.size()
                                << " bl_q=" << backlink_queue.size()
                                << " bc_q=" << broadcast_queue.size()
                                << " resp_q=" << response_queue.size()
                                << " maintenance_in_flight="
                                << maintenance_in_flight.load(
                                       std::memory_order_acquire)
                                << " search_in_flight="
                                << search_in_flight.load(
                                       std::memory_order_acquire)
                                << std::endl;
                            break;
                        }

                        const uint8_t ack = static_cast<uint8_t>(
                            MessageType::SHUTDOWN_ACK);
                        zmq::message_t ack_identity(
                            work.identity.data(), work.identity.size());
                        zmq::message_t ack_empty(0);
                        zmq::message_t ack_data(&ack, sizeof(ack));
                        socket.send(
                            ack_identity, zmq::send_flags::sndmore);
                        socket.send(
                            ack_empty, zmq::send_flags::sndmore);
                        socket.send(ack_data, zmq::send_flags::none);
                        std::cout
                            << "[SHUTDOWN] Quiescent ACK sent"
                            << std::endl;
                        running.store(false, std::memory_order_release);
                        graceful_shutdown = true;
                        break;
                    }

                    if (msg_type == MessageType::RESET_REQUEST) {
                        finish_snapshot(true);
                        std::cout << "[RESET] my_p=" << g_rank << " starting reload..." << std::endl;

                        // RESET-vs-worker: unique_lock blocks until all worker
                        // shared_lock holders release. Held for the duration of
                        // clear + reload below.
                        std::unique_lock<std::shared_mutex> admin_lock(tide::g_admin_mutex);

                        // Update the measurement start BEFORE reload. Any item
                        // with enqueue_time < this ts was queued before RESET
                        // and is excluded from the accumulators from now on.
                        tide::g_interval_start_ns.store(
                            steady_clock::now().time_since_epoch().count(),
                            std::memory_order_release);

                        // Flush remaining batch queues before reset
                        size_t backlink_discarded = 0;
                        size_t broadcast_discarded = 0;
                        while (true) {
                            auto batch = backlink_queue.try_pop_n(1000);
                            if (batch.empty()) break;
                            backlink_discarded += batch.size();
                        }
                        while (true) {
                            auto batch = broadcast_queue.try_pop_n(1000);
                            if (batch.empty()) break;
                            broadcast_discarded += batch.size();
                        }
                        if (backlink_discarded > 0 || broadcast_discarded > 0) {
                            std::cout << "[RESET] Discarded " << backlink_discarded
                                      << " backlinks, " << broadcast_discarded
                                      << " broadcasts from queues" << std::endl;
                        }

                        {
                            std::unique_lock<std::shared_mutex> lock(g_resize_mutex);
                            // Every store cleared and reloaded from disk in the
                            // start-up mode (graph_ops.cpp, unit-tested under
                            // both storage modes).
                            start_node = reset_server_data(data, g_data_dir, g_rank);
                            g_ownership_epoch = 0;
                            g_membership_migration_plan =
                                MembershipMigrationPlan{};
                            g_membership_migration_journal =
                                MembershipMigrationJournal{};
                            g_accepted_migration_epochs.clear();
                            g_lock_metrics.reset();
                            // RESET starts new cumulative queue sums/counts and
                            // clears the current max intervals. Consuming
                            // GET_STATS requests exchange-reset only the maxima.
                            // Stale pre-reset items are already excluded by
                            // g_interval_start_ns in both queue paths.
                            metrics.reset();
                        }
                        // admin_lock released here on scope exit; worker shared_locks
                        // can now proceed against the reloaded state.

                        std::cout << "[RESET] my_p=" << g_rank << " complete: "
                                  << "membership_size=" << data.membership.size()
                                  << " full_base_size=" << data.full_base.size()
                                  << " own_vectors=" << (data.own_only ? n_local_load(data) : 0)
                                  << " adj_size=" << n_local_load(data) << std::endl;

                        std::vector<uint8_t> ack_msg(1);
                        ack_msg[0] = static_cast<uint8_t>(MessageType::RESET_ACK);

                        zmq::message_t ack_identity(work.identity.data(), work.identity.size());
                        zmq::message_t ack_empty(0);
                        zmq::message_t ack_data(ack_msg.data(), ack_msg.size());

                        socket.send(ack_identity, zmq::send_flags::sndmore);
                        socket.send(ack_empty, zmq::send_flags::sndmore);
                        socket.send(ack_data, zmq::send_flags::none);
                        break;
                    }

                    if (msg_type == MessageType::DRAIN_REQUEST) {
                        // DRAIN is a control barrier. Handle it on the I/O
                        // thread so worker scheduling cannot delay the ACK
                        // behind maintenance work.
                        size_t ins_flushed = 0, bl_flushed = 0, bc_flushed = 0;
                        size_t resp_forwarded = 0;
                        auto enqueue_drain_responses = [&](std::vector<ResponseItem>& responses) {
                            auto now = steady_clock::now();
                            for (auto& r : responses) {
                                if (r.response.empty()) continue;
                                r.enqueue_time = now;
                                response_queue.push(std::move(r));
                                resp_forwarded++;
                            }
                        };

                        while (true) {
                            bool did_work = false;
                            while (true) {
                                auto batch = insert_queue.try_pop_n(1000);
                                if (batch.empty()) break;
                                auto responses = batch_process_inserts(data, batch, metrics);
                                enqueue_drain_responses(responses);
                                ins_flushed += batch.size();
                                did_work = true;
                            }
                            while (true) {
                                auto batch = backlink_queue.try_pop_n(1000);
                                if (batch.empty()) break;
                                auto responses = batch_process_backlinks(data, batch, metrics);
                                enqueue_drain_responses(responses);
                                bl_flushed += batch.size();
                                did_work = true;
                            }
                            while (true) {
                                auto batch = broadcast_queue.try_pop_n(1000);
                                if (batch.empty()) break;
                                auto responses = batch_process_broadcasts(data, batch, metrics);
                                enqueue_drain_responses(responses);
                                bc_flushed += batch.size();
                                did_work = true;
                            }

                            bool queues_empty = insert_queue.size() == 0 &&
                                                backlink_queue.size() == 0 &&
                                                broadcast_queue.size() == 0;
                            if (queues_empty &&
                                maintenance_in_flight.load(std::memory_order_acquire) == 0) {
                                if (insert_queue.size() == 0 &&
                                    backlink_queue.size() == 0 &&
                                    broadcast_queue.size() == 0) {
                                    break;
                                }
                            }

                            if (!did_work) {
                                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            }
                        }

                        if (ins_flushed > 0 || bl_flushed > 0 ||
                            bc_flushed > 0 || resp_forwarded > 0) {
                            std::cout << "[DRAIN] Applied " << ins_flushed
                                      << " inserts, " << bl_flushed
                                      << " backlinks, " << bc_flushed
                                      << " broadcasts; forwarded "
                                      << resp_forwarded << " queued responses"
                                      << std::endl;
                        }

                        ResponseItem resp;
                        resp.identity = std::move(work.identity);
                        resp.coordinator_id = std::move(work.coordinator_id);
                        resp.needs_forward = false;
                        uint8_t ack = static_cast<uint8_t>(MessageType::DRAIN_ACK);
                        resp.response.assign(&ack, &ack + 1);
                        resp.enqueue_time = steady_clock::now();
                        response_queue.push(std::move(resp));
                        break;
                    }

                    if (msg_type == MessageType::GET_STATS_REQUEST) {
                        // One-byte requests and ordinary internal reads are
                        // observational. Only an explicit wire flag (the admin
                        // stats command) takes and resets the interval maxima.
                        bool consume_interval_max = false;
                        uint64_t stats_request_id = 0;
                        if (request.size() >= sizeof(StatsRequest)) {
                            const auto* stats_request =
                                reinterpret_cast<const StatsRequest*>(request.data());
                            consume_interval_max =
                                stats_request->consume_interval_max != 0;
                            stats_request_id = stats_request->request_id;
                        }
                        StatsResponseHeader stats{};
                        stats.msg_type = MessageType::GET_STATS_RESPONSE;
                        stats.padding[0] = stats.padding[1] = stats.padding[2] = 0;
                        stats.request_id = stats_request_id;
                        stats.partition_id = static_cast<uint32_t>(g_rank);

                        {
                            // GET_STATS — lock-free with atomic acquire-loads.
                            // Node ids this server knows: the load-time nodes plus every
                            // inserted id that reached it (insert, backlink, broadcast), in
                            // both storage modes. (It reported the full-replication vector
                            // store's capacity, which stays N0 under own-only storage.)
                            stats.total_nodes = n_base_load(data);
                            uint32_t n_local_snap = n_local_load(data);
                            stats.local_nodes = n_local_snap;

                            uint64_t directed_edges = 0;
                            uint64_t allocated_neighbor_capacity = 0;
                            // A row below node_locks.size() has its adjacency chunk and its
                            // lock published (every path that grows the lock store grows adj
                            // first, and an insert does both before publishing the row), so
                            // the loop stays below it and reads each row under its lock.
                            const uint32_t rows = static_cast<uint32_t>(
                                std::min<size_t>(n_local_snap, data.node_locks.size()));
                            for (uint32_t i = 0; i < rows; i++) {
                                std::lock_guard<std::mutex> nlock(data.node_locks[i].mtx);
                                directed_edges += data.adj[i].size();
                                allocated_neighbor_capacity += data.adj[i].capacity();
                            }
                            stats.directed_edges = directed_edges;
                            stats.adjacency_serialized_bytes =
                                (static_cast<uint64_t>(n_local_snap) + directed_edges) *
                                sizeof(uint32_t);
                            stats.adjacency_allocated_neighbor_bytes =
                                allocated_neighbor_capacity * sizeof(uint32_t);
                        }

                        stats.shared_lock_count = g_lock_metrics.shared_lock_count.load();
                        stats.unique_lock_count = g_lock_metrics.unique_lock_count.load();
                        stats.shared_lock_wait_us = g_lock_metrics.shared_lock_wait_us.load();
                        stats.unique_lock_wait_us = g_lock_metrics.unique_lock_wait_us.load();
                        stats.insert_alloc_lock_count = g_lock_metrics.insert_alloc_lock_count.load();
                        stats.insert_alloc_lock_wait_us = g_lock_metrics.insert_alloc_lock_wait_us.load();
                        stats.insert_edge_lock_count = g_lock_metrics.insert_edge_lock_count.load();
                        stats.insert_edge_lock_wait_us = g_lock_metrics.insert_edge_lock_wait_us.load();
                        stats.backlink_lock_count = g_lock_metrics.backlink_lock_count.load();
                        stats.backlink_lock_wait_us = g_lock_metrics.backlink_lock_wait_us.load();
                        stats.broadcast_lock_count = g_lock_metrics.broadcast_lock_count.load();
                        stats.broadcast_lock_wait_us = g_lock_metrics.broadcast_lock_wait_us.load();

                        stats.backlink_edges_requested = g_graph_metrics.backlink_edges_requested.load();
                        stats.backlink_edges_applied = g_graph_metrics.backlink_edges_applied.load();
                        stats.backlink_skip_not_local = g_graph_metrics.backlink_skip_not_local.load();
                        stats.backlink_skip_already_exists = g_graph_metrics.backlink_skip_already_exists.load();
                        stats.backlink_skip_out_of_range = g_graph_metrics.backlink_skip_out_of_range.load();
                        stats.insert_neighbors_undelivered = g_graph_metrics.insert_neighbors_undelivered.load();

                        stats.search_shared = snapshot_lock_role(g_lock_metrics.search_shared);
                        stats.insert_alloc_unique = snapshot_lock_role(g_lock_metrics.insert_alloc_unique);
                        stats.insert_alloc_shared = snapshot_lock_role(g_lock_metrics.insert_alloc_shared);
                        stats.insert_alloc_mutex = snapshot_lock_role(g_lock_metrics.insert_alloc_mutex);
                        stats.insert_prune_shared = snapshot_lock_role(g_lock_metrics.insert_prune_shared);
                        stats.insert_edge_shared = snapshot_lock_role(g_lock_metrics.insert_edge_shared);
                        stats.backlink_unique = snapshot_lock_role(g_lock_metrics.backlink_unique);
                        stats.broadcast_unique = snapshot_lock_role(g_lock_metrics.broadcast_unique);
                        stats.search_node_lock = snapshot_lock_role(g_lock_metrics.search_node_lock);
                        stats.search_distance_us = g_lock_metrics.search_distance_us.load();
                        stats.search_beam_sort_us = g_lock_metrics.search_beam_sort_us.load();
                        stats.search_distance_count = g_lock_metrics.search_distance_count.load();
                        stats.search_neighbors_evaluated = g_lock_metrics.search_neighbors_evaluated.load();
                        stats.search_request_count = g_lock_metrics.search_request_count.load();
                        stats.search_exact_count = g_lock_metrics.search_exact_count.load();
                        stats.search_adc_count = g_lock_metrics.search_adc_count.load();
                        stats.search_rerank_count = g_lock_metrics.search_rerank_count.load();
                        stats.search_missing_code_count = g_lock_metrics.search_missing_code_count.load();
                        stats.search_adc_table_count = g_lock_metrics.search_adc_table_count.load();
                        stats.search_adc_table_us = g_lock_metrics.search_adc_table_ns.load() / 1000;
                        {
                            const StorageBytes bytes = storage_bytes(data);
                            stats.storage_mode = data.own_only ? 1u : 0u;
                            stats.storage_dim = data.dim;
                            stats.storage_pq_m = data.own_only ? data.pq_table.M : 0u;
                            stats.storage_vector_bytes = bytes.vector_bytes;
                            stats.storage_code_bytes = bytes.code_bytes;
                            stats.storage_pq_table_bytes = bytes.pq_table_bytes;
                            stats.process_rss_bytes = process_rss_bytes();
                        }
                        stats.search_total_us = metrics.search_total_us.load();
                        stats.queue_wait_us = metrics.queue_wait_us.load();
                        stats.queue_wait_count = metrics.queue_wait_count.load();
                        stats.response_queue_wait_us = metrics.response_queue_wait_us.load();
                        stats.response_queue_wait_count = metrics.response_queue_wait_count.load();
                        stats.queue_wait_interval_max_us =
                            ann::snapshot_interval_max(
                                metrics.queue_wait_interval_max_us,
                                consume_interval_max);
                        stats.response_queue_wait_interval_max_us =
                            ann::snapshot_interval_max(
                                metrics.response_queue_wait_interval_max_us,
                                consume_interval_max);

                        zmq::message_t resp_identity(work.identity.data(), work.identity.size());
                        zmq::message_t resp_empty(0);
                        zmq::message_t resp_data(&stats, sizeof(stats));

                        socket.send(resp_identity, zmq::send_flags::sndmore);
                        socket.send(resp_empty, zmq::send_flags::sndmore);
                        socket.send(resp_data, zmq::send_flags::none);
                        break;
                    }

                    // PING: Immediate response for network baseline measurement
                    if (msg_type == MessageType::PING_REQUEST) {
                        if (request.size() >= sizeof(PingRequest)) {
                            const PingRequest* ping = reinterpret_cast<const PingRequest*>(request.data());
                            PingResponse pong;
                            pong.msg_type = MessageType::PING_RESPONSE;
                            pong.padding[0] = pong.padding[1] = pong.padding[2] = 0;
                            pong.sequence = ping->sequence;
                            pong.send_time_ns = ping->send_time_ns;
                            pong.server_id = static_cast<uint32_t>(g_rank);
                            pong.padding2 = 0;

                            zmq::message_t resp_identity(work.identity.data(), work.identity.size());
                            zmq::message_t resp_empty(0);
                            zmq::message_t resp_data(&pong, sizeof(pong));

                            socket.send(resp_identity, zmq::send_flags::sndmore);
                            socket.send(resp_empty, zmq::send_flags::sndmore);
                            socket.send(resp_data, zmq::send_flags::none);
                        }
                        break;
                    }

                    // P2P: Track query origin for response routing
                    if (msg_type == MessageType::SEARCH_REQUEST && p2p_mode &&
                        request.size() >= 1 + sizeof(StateHeader)) {
                        StateHeader* hdr = reinterpret_cast<StateHeader*>(
                            static_cast<char*>(request.data()) + 1);
                        std::lock_guard<std::mutex> lock(query_origins_mutex);
                        query_origins[hdr->query_id] = work.identity;
                    }

                    work.enqueue_time = steady_clock::now();
                    work.io_processing_us = static_cast<uint32_t>(
                        duration_cast<microseconds>(work.enqueue_time - recv_end).count()
                    );

                    metrics.zmq_recv_us += work.zmq_recv_us;

                    // Route to appropriate queue based on message type. The
                    // maintenance types go only to their batch queues.
                    if (msg_type == MessageType::BATCH_BACKLINK_REQUEST) {
                        work.queue_depth = static_cast<uint32_t>(backlink_queue.size());
                        metrics.queue_depth_sum += work.queue_depth;
                        backlink_queue.push(std::move(work));
                    } else if (msg_type == MessageType::VECTOR_BROADCAST) {
                        work.queue_depth = static_cast<uint32_t>(broadcast_queue.size());
                        metrics.queue_depth_sum += work.queue_depth;
                        broadcast_queue.push(std::move(work));
                    } else if (msg_type == MessageType::INSERT_REQUEST) {
                        work.queue_depth = static_cast<uint32_t>(insert_queue.size());
                        metrics.queue_depth_sum += work.queue_depth;
                        insert_queue.push(std::move(work));
                    } else {
                        work.queue_depth = static_cast<uint32_t>(work_queue.size());
                        metrics.queue_depth_sum += work.queue_depth;
                        work_queue.push(std::move(work));
                    }
                    recv_count++;
                } catch (const zmq::error_t& e) {
                    if (e.num() != EAGAIN) {
                        std::cerr << "ZMQ recv error: " << e.what() << std::endl;
                    }
                }
            }
        }

        // Send responses
        ResponseItem resp;
        uint32_t send_this_loop = 0;
        while ((send_budget == 0 || send_this_loop < send_budget) &&
               response_queue.try_pop(resp)) {
            send_this_loop++;
            auto pop_time = steady_clock::now();
            uint32_t this_resp_queue_wait_us = static_cast<uint32_t>(
                duration_cast<microseconds>(pop_time - resp.enqueue_time).count()
            );
            // Gate: skip stale responses enqueued before the last RESET.
            // Still send the response (coord-side cleanup), but exclude from metric.
            int64_t interval_start_ns = tide::g_interval_start_ns.load(std::memory_order_acquire);
            bool resp_is_stale = resp.enqueue_time.time_since_epoch().count() < interval_start_ns;
            if (!resp_is_stale) {
                metrics.response_queue_wait_us.fetch_add(
                    this_resp_queue_wait_us, std::memory_order_relaxed);
                metrics.response_queue_wait_count.fetch_add(1, std::memory_order_relaxed);
                ann::update_interval_max(metrics.response_queue_wait_interval_max_us,
                                         this_resp_queue_wait_us);
            }
            io_timing.last_resp_queue_wait_us.store(this_resp_queue_wait_us, std::memory_order_relaxed);

            try {
                if (resp.needs_forward && p2p_mode && !peer_sockets.empty()) {
                    uint32_t target = resp.target_partition;
                    if (target == static_cast<uint32_t>(g_rank)) {
                        // Return NEED_HOP to the coordinator instead of
                        // forwarding to this rank's own socket.
                        resp.needs_forward = false;
                    } else if (target < peer_sockets.size() && peer_sockets[target]) {
                        zmq::message_t empty_msg(0);
                        zmq::message_t coord_id_msg(resp.coordinator_id.data(), resp.coordinator_id.size());
                        zmq::message_t state_msg(resp.forward_state.data(), resp.forward_state.size());

                        auto send_start = steady_clock::now();
                        peer_sockets[target]->send(empty_msg, zmq::send_flags::sndmore);
                        peer_sockets[target]->send(coord_id_msg, zmq::send_flags::sndmore);
                        peer_sockets[target]->send(state_msg, zmq::send_flags::none);
                        auto send_end = steady_clock::now();
                        uint32_t this_zmq_send_us = static_cast<uint32_t>(
                            duration_cast<microseconds>(send_end - send_start).count()
                        );
                        metrics.zmq_send_us += this_zmq_send_us;
                        io_timing.last_zmq_send_us.store(this_zmq_send_us, std::memory_order_relaxed);
                        p2p_forward_count++;
                        send_count++;
                        if (p2p_forward_count % 1000 == 0) {
                            std::cerr << "[P2P] Rank " << g_rank << " forwarded "
                                      << p2p_forward_count << " to partition " << target
                                      << " (send=" << this_zmq_send_us << "us)" << std::endl;
                        }
                    } else {
                        std::cerr << "P2P: Invalid target partition " << target << std::endl;
                    }
                }

                if (!resp.needs_forward) {
                    std::string target_id;
                    if (!resp.coordinator_id.empty()) {
                        target_id = resp.coordinator_id;
                    } else if (p2p_mode) {
                        if (resp.response.size() >= 1 + sizeof(StateResponseHeader)) {
                            StateResponseHeader* hdr = reinterpret_cast<StateResponseHeader*>(
                                resp.response.data() + 1);
                            std::lock_guard<std::mutex> lock(query_origins_mutex);
                            auto it = query_origins.find(hdr->query_id);
                            if (it != query_origins.end()) {
                                target_id = it->second;
                                query_origins.erase(it);
                            } else {
                                target_id = resp.identity;
                            }
                        } else {
                            target_id = resp.identity;
                        }
                    } else {
                        target_id = resp.identity;
                    }

                    auto send_start = steady_clock::now();

                    // Measure gap: pop_time → send_start (routing + message construction)
                    uint32_t send_prep_gap_us = static_cast<uint32_t>(
                        duration_cast<microseconds>(send_start - pop_time).count()
                    );
                    metrics.send_prep_us += send_prep_gap_us;
                    io_timing.last_send_prep_us.store(send_prep_gap_us, std::memory_order_relaxed);

                    // Patch SEARCH_RESPONSE header with actual timing values
                    // (replaces lagged values that worker put in)
                    MessageType resp_type = get_message_type(
                        resp.response.data(), resp.response.size());
                    if (resp_type == MessageType::SEARCH_RESPONSE &&
                        resp.response.size() >= 1 + sizeof(StateResponseHeader)) {
                        StateResponseHeader* hdr = reinterpret_cast<StateResponseHeader*>(
                            resp.response.data() + 1);  // skip msg_type byte
                        hdr->response_queue_wait_us = this_resp_queue_wait_us;
                        hdr->send_prep_us = send_prep_gap_us;
                    }

                    zmq::message_t id_msg(target_id.data(), target_id.size());
                    zmq::message_t empty_msg(0);
                    zmq::message_t resp_msg(resp.response.data(), resp.response.size());

                    socket.send(id_msg, zmq::send_flags::sndmore);
                    socket.send(empty_msg, zmq::send_flags::sndmore);
                    socket.send(resp_msg, zmq::send_flags::none);
                    auto send_end = steady_clock::now();
                    uint32_t this_zmq_send_us = static_cast<uint32_t>(
                        duration_cast<microseconds>(send_end - send_start).count()
                    );
                    metrics.zmq_send_us += this_zmq_send_us;
                    io_timing.last_zmq_send_us.store(this_zmq_send_us, std::memory_order_relaxed);
                    send_count++;
                }
            } catch (const zmq::error_t& e) {
                std::cerr << "ZMQ send error: " << e.what() << std::endl;
            }
        }
    }

    // Cleanup
    running = false;

    // Flush remaining batch queues before shutdown
    size_t backlink_discarded = 0;
    size_t broadcast_discarded = 0;
    while (true) {
        auto batch = backlink_queue.try_pop_n(1000);
        if (batch.empty()) break;
        backlink_discarded += batch.size();
    }
    while (true) {
        auto batch = broadcast_queue.try_pop_n(1000);
        if (batch.empty()) break;
        broadcast_discarded += batch.size();
    }
    if (backlink_discarded > 0 || broadcast_discarded > 0) {
        std::cout << "[SHUTDOWN] Discarded " << backlink_discarded
                  << " backlinks, " << broadcast_discarded
                  << " broadcasts from queues" << std::endl;
    }

    for (auto& w : workers) {
        w.join();
    }
    return 0;
}
