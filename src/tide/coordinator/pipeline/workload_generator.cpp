/**
 * Workload generator of the unified pipeline.
 *
 * Contains:
 * - generate_mixed_workloads: Generate mixed search+insert workloads
 * - send_workload: Send a workload item to server
 */

#include "coordinator_types.h"
#include "pipeline_types.h"
#include "coordinator_types.h"
#include "head_index.h"
#include "protocol/insert_protocol.h"
#include "protocol/broadcast_messages.h"
#include "workload_progress.h"

#include <iostream>
#include <chrono>
#include <unordered_map>
#include <random>
#include <atomic>
#include <mutex>

using namespace std::chrono;

namespace tide {

// Coordinator-side head index BF timing (cumulative since process start).
// Snapshot via stats command or workload completion print.
std::atomic<uint64_t> g_head_bf_us{0};
std::atomic<uint64_t> g_head_bf_calls{0};

// Helper: timed wrapper around HeadIndex::find_nearest_with_dist.
static inline std::pair<uint32_t, float> head_bf_timed(
    const ann::HeadIndex* head_index, const float* query
) {
    auto t0 = steady_clock::now();
    auto r = head_index->find_nearest_with_dist(query);
    auto t1 = steady_clock::now();
    g_head_bf_us.fetch_add(
        duration_cast<microseconds>(t1 - t0).count(),
        std::memory_order_relaxed);
    g_head_bf_calls.fetch_add(1, std::memory_order_relaxed);
    return r;
}

// Number of helper threads doing the head-index brute force in parallel.
// Default 8; override with TIDE_HEAD_BF_THREADS (1..64).
static int head_bf_helper_count() {
    if (const char* env = std::getenv("TIDE_HEAD_BF_THREADS")) {
        int n = std::atoi(env);
        if (n > 0 && n <= 64) return n;
    }
    return 8;
}

// Compute head BF for one workload item and fill its entry-related fields.
// Called from helper threads — must be thread safe modulo `item` (each
// helper writes a disjoint set of items).
static inline void apply_head_bf(
    WorkloadItem& item,
    const ann::HeadIndex* head_index,
    const uint32_t* membership,
    uint32_t n_servers
) {
    const float* q = item.query.data();
    auto [entry_node, entry_dist] = head_bf_timed(head_index, q);
    uint32_t entry_partition = owner_label_checked(membership[entry_node], n_servers, "head entry");
    item.beam.push_back({entry_node, entry_dist, 0, {0, 0, 0}});
    item.current_partition = entry_partition;
    item.entry_partition = entry_partition;
    if (item.type == WorkloadType::INSERT) {
        item.target_partition = entry_partition;
    }
}

void generate_mixed_workloads(UnifiedContext& ctx) {
    const auto* head_index = static_cast<const ann::HeadIndex*>(ctx.head_index);
    const auto materialize_start = steady_clock::now();
    auto& accounting = ctx.serving_accounting;
    auto steady_ns = [](steady_clock::time_point stamp) {
        return duration_cast<nanoseconds>(stamp.time_since_epoch()).count();
    };
    if (accounting.enabled) accounting.materialize.begin_ns = steady_ns(materialize_start);
    auto materialize_last = materialize_start;
    uint64_t materialized_reported = 0;
    std::cout << "[WORKLOAD_STAGE] stage=materialize state=start completed=0 total="
              << ctx.total_workloads << " unit=items elapsed_ms=0 item_bytes=" << sizeof(WorkloadItem)
              << " query_copy_bytes=" << uint64_t(ctx.total_workloads) * ctx.dim * sizeof(float)
              << '\n' << std::flush;
    ctx.workloads.clear();
    ctx.workloads.reserve(ctx.total_workloads);

    // ──────────────────────────────────────────────────────────────────
    // Step 1: serial init pass — assign type, query data, etc.
    // (NO head BF here; deferred to helper threads below.)
    // The interleave logic uses cumulative insert_idx/expected_inserts,
    // so this MUST stay serial. It is fast (just struct fills + memcpy).
    // ──────────────────────────────────────────────────────────────────
    uint32_t search_idx = 0;
    uint32_t insert_idx = 0;
    // IDs cross the coordinator/server boundary and server P2P state lives
    // beyond one daemon command.  Allocate a daemon-global, non-reused range
    // so a delayed reply cannot alias a later command's workload zero.
    uint32_t workload_id = g_daemon.next_workload_id.fetch_add(
        ctx.total_workloads, std::memory_order_relaxed);

    for (uint32_t i = 0; i < ctx.total_workloads; i++) {
        uint32_t expected_inserts = (ctx.n_inserts > 0) ?
            ((i + 1) * ctx.n_inserts / ctx.total_workloads) : 0;
        bool should_insert = (insert_idx < expected_inserts) && (insert_idx < ctx.n_inserts);

        if (should_insert) {
            WorkloadItem item;
            item.type = WorkloadType::INSERT;
            item.id = workload_id++;
            item.insert_sequence = insert_idx;
            item.insert_phase = InsertPhase::SEARCH_PENDING;
            item.done = false;
            item.hops = 0;
            const float* vec = ctx.insert_vectors + static_cast<size_t>(insert_idx) * ctx.dim;
            item.query.assign(vec, vec + ctx.dim);
            ctx.workloads.push_back(std::move(item));
            insert_idx++;
        } else if (search_idx < ctx.n_searches) {
            WorkloadItem item;
            item.type = WorkloadType::SEARCH;
            item.id = workload_id++;
            uint32_t q_idx = (ctx.n_available_queries > 0)
                ? (search_idx % ctx.n_available_queries)
                : search_idx;
            item.query_index = q_idx;
            // Ordinal is the position in this exact search-only execution,
            // not the possibly-cycled source query index.
            item.query_ordinal = search_idx;
            item.done = false;
            item.hops = 0;
            const float* q = ctx.queries_data + static_cast<size_t>(q_idx) * ctx.dim;
            item.query.assign(q, q + ctx.dim);
            ctx.workloads.push_back(std::move(item));
            search_idx++;
        } else if (insert_idx < ctx.n_inserts) {
            WorkloadItem item;
            item.type = WorkloadType::INSERT;
            item.id = workload_id++;
            item.insert_sequence = insert_idx;
            item.insert_phase = InsertPhase::SEARCH_PENDING;
            item.done = false;
            item.hops = 0;
            const float* vec = ctx.insert_vectors + static_cast<size_t>(insert_idx) * ctx.dim;
            item.query.assign(vec, vec + ctx.dim);
            ctx.workloads.push_back(std::move(item));
            insert_idx++;
        }
        if ((i + 1) % 4096 == 0 || i + 1 == ctx.total_workloads) {
            const auto now = steady_clock::now();
            if (workload_progress_due(ctx.workloads.size(), materialized_reported, ctx.total_workloads,
                    16384, duration_cast<milliseconds>(now - materialize_last).count())) {
                std::cout << "[WORKLOAD_STAGE] stage=materialize state=progress completed="
                          << ctx.workloads.size() << " total=" << ctx.total_workloads << " unit=items elapsed_ms="
                          << duration_cast<milliseconds>(now - materialize_start).count() << '\n' << std::flush;
                materialized_reported = ctx.workloads.size(); materialize_last = now;
            }
        }
    }

    const auto materialize_end = steady_clock::now();
    if (accounting.enabled) accounting.materialize.end_ns = steady_ns(materialize_end);
    std::cout << "  Generated workloads: " << ctx.workloads.size()
              << " (" << search_idx << " search, " << insert_idx << " insert)\n"
              << "[WORKLOAD_STAGE] stage=materialize state=end completed=" << ctx.workloads.size()
              << " total=" << ctx.total_workloads << " unit=items elapsed_ms="
              << duration_cast<milliseconds>(materialize_end - materialize_start).count() << '\n' << std::flush;

    // ──────────────────────────────────────────────────────────────────
    // Step 2: parallel head BF — N_HELPER threads. JOIN before returning.
    // Pipeline does NOT overlap with this; it starts after all entries are
    // ready. This keeps the original sequential refill_workloads semantics
    // (and the QPS measurement which starts at first dispatch).
    // ──────────────────────────────────────────────────────────────────
    auto t_bf_start = steady_clock::now();
    if (accounting.enabled) accounting.head_bf.begin_ns = steady_ns(t_bf_start);
    const int N_HELPER = head_bf_helper_count();
    const uint32_t* membership = ctx.membership;
    const uint32_t n_servers = ctx.n_servers;
    const uint32_t N = static_cast<uint32_t>(ctx.workloads.size());
    auto* workloads = &ctx.workloads;
    std::atomic<uint32_t> bf_completed{0};
    std::mutex bf_progress_mutex;
    uint64_t bf_reported = 0;
    auto bf_last = t_bf_start;
    std::cout << "[WORKLOAD_STAGE] stage=head_bf state=start completed=0 total=" << N
              << " unit=items elapsed_ms=0 threads=" << N_HELPER << '\n' << std::flush;

    std::vector<std::thread> threads;
    threads.reserve(N_HELPER);
    for (int tid = 0; tid < N_HELPER; tid++) {
        threads.emplace_back(
            [&, tid, N_HELPER, N, workloads, head_index, membership, n_servers]() {
                for (uint32_t i = static_cast<uint32_t>(tid); i < N;
                     i += static_cast<uint32_t>(N_HELPER)) {
                    apply_head_bf((*workloads)[i], head_index, membership, n_servers);
                    const auto done = bf_completed.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (done % 1024 == 0 || done == N) {
                        std::lock_guard<std::mutex> lock(bf_progress_mutex);
                        const auto now = steady_clock::now();
                        if (workload_progress_due(done, bf_reported, N, 4096,
                                duration_cast<milliseconds>(now - bf_last).count())) {
                            std::cout << "[WORKLOAD_STAGE] stage=head_bf state=progress completed=" << done
                                      << " total=" << N << " unit=items elapsed_ms="
                                      << duration_cast<milliseconds>(now - t_bf_start).count() << '\n' << std::flush;
                            bf_reported = done; bf_last = now;
                        }
                    }
                }
            });
    }
    for (auto& t : threads) t.join();
    auto t_bf_end = steady_clock::now();
    if (accounting.enabled) accounting.head_bf.end_ns = steady_ns(t_bf_end);
    auto bf_wall_us = duration_cast<microseconds>(t_bf_end - t_bf_start).count();
    std::cout << "[WORKLOAD_STAGE] stage=head_bf state=end completed=" << bf_completed.load()
              << " total=" << N << " unit=items elapsed_ms=" << bf_wall_us / 1000.0 << '\n' << std::flush;

    // Step 3: serial debug-counter pass (avoids atomic in hot path).
    if (g_debug.enabled) {
        for (const auto& item : ctx.workloads) {
            uint32_t ep = item.entry_partition;
            if (ep < g_debug.entry_partition_count.size()) {
                g_debug.entry_partition_count[ep]++;
            }
        }
    }
    std::cout << "  Head BF parallel: " << N_HELPER << " threads, " << N
              << " items, wall=" << bf_wall_us / 1000.0 << "ms\n" << std::flush;
}

// =============================================================================
// queue_insert_ops - Send all insert ops (INSERT, BROADCAST) per-server directly.
// Called after insert search completes. Each op is dispatched to its target
// server via the per-server work queue (coordinator_server_thread drains).
// =============================================================================
void queue_insert_ops(UnifiedContext& ctx, uint32_t workload_idx) {
    auto& item = ctx.workloads[workload_idx];

    // 1. Compute assigned_id
    uint32_t assigned_id = ctx.n_base + item.insert_sequence;
    item.assigned_id = assigned_id;

    // 2. Determine target partition from nearest search neighbor
    if (!item.search_neighbors.empty()) {
        item.target_partition = get_node_partition(ctx, item.search_neighbors[0]);
    }
    uint32_t target_server = owner_label_checked(item.target_partition, ctx.n_servers, "insert target");

    // 3. Dispatch INSERT_REQUEST to target server
    {
        auto insert_data = ann::serialize_insert_request(
            item.id, item.query.data(), ctx.dim,
            item.search_neighbors, item.search_distances,
            ctx.insert_R, ctx.insert_alpha,
            item.target_partition, assigned_id
        );
        ServerWorkItem swi;
        swi.request_data = std::move(insert_data);
        swi.target_server = target_server;
        swi.workload_id = item.id;
        g_daemon.server_queues[target_server]->push(std::move(swi));
    }

    // 4. Add to pending manager
    if (ctx.pending_mgr) {
        ctx.pending_mgr->add(assigned_id, item.query, item.target_partition);
    }

    // 5. Track inserted vectors
    ctx.inserted_vectors[assigned_id] = item.query;
    ctx.inserted_partitions[assigned_id] = item.target_partition;

    // 6. Optimistically update membership_vec
    if (ctx.membership_vec && assigned_id > 0) {
        if (assigned_id >= ctx.membership_vec->size()) {
            ctx.membership_vec->resize(assigned_id + 1);
        }
        (*ctx.membership_vec)[assigned_id] = item.target_partition;
    }

    // 7. Defer backlinks until INSERT_RESPONSE returns post-prune exact edges.
    if (!ctx.no_backlink) {
        item.backlinks_sent = 0;
        item.backlinks_acked = 0;
        item.backlink_done = false;
        ctx.insert_response_pending_map[item.id] = workload_idx;
    } else {
        item.backlink_done = true;
        ctx.backlink_phase_completed++;
    }

    // 8. Dispatch VECTOR_BROADCAST to all servers
    if (!ctx.no_broadcast) {
        item.broadcasts_acked = 0;
        auto broadcast_data = ann::serialize_vector_broadcast(
            assigned_id, ctx.dim, item.target_partition, item.query.data()
        );
        for (uint32_t s = 0; s < ctx.n_servers; s++) {
            ServerWorkItem swi;
            swi.request_data.assign(broadcast_data.begin(), broadcast_data.end());
            swi.target_server = s;
            swi.workload_id = item.id;
            g_daemon.server_queues[s]->push(std::move(swi));
        }
        ctx.broadcast_pending_map[assigned_id] = workload_idx;
    } else {
        item.broadcast_done = true;
    }

    // 9. Transition to INSERT_PENDING (skip INSERT_PENDING)
    item.insert_phase = InsertPhase::INSERT_PENDING;
    item.phase_start_time = steady_clock::now();

    // 10. Add to completed_inserts_pool
    if (ctx.enable_inserted_query && !item.query.empty()) {
        UnifiedContext::CompletedInsert completed_insert;
        completed_insert.insert_sequence = item.insert_sequence;
        completed_insert.vector = item.query;
        ctx.completed_inserts_pool.push(std::move(completed_insert));
    }
}

void send_workload(UnifiedContext& ctx, uint32_t workload_idx) {
    if (workload_idx >= ctx.workloads.size()) return;

    auto& item = ctx.workloads[workload_idx];

    auto& sockets = *ctx.sockets;
    const auto* head_index = static_cast<const ann::HeadIndex*>(ctx.head_index);

    // Convert SEARCH to inserted query with probability
    if (item.type == WorkloadType::SEARCH && !item.is_inserted_query &&
        ctx.enable_inserted_query && !ctx.completed_inserts_pool.empty()) {

        static thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double roll = dist(rng);

        if (roll < ctx.inserted_query_ratio) {
            auto& completed = ctx.completed_inserts_pool.front();

            if (!completed.vector.empty() && head_index) {
                item.is_inserted_query = true;
                item.insert_gt_index = completed.insert_sequence;
                item.query = std::move(completed.vector);

                auto [entry_node, entry_dist] = head_bf_timed(head_index, item.query.data());

                uint32_t entry_partition = 0;
                if (ctx.membership_vec && entry_node < ctx.membership_vec->size()) {
                    entry_partition = owner_label_checked((*ctx.membership_vec)[entry_node], ctx.n_servers, "entry node");
                } else if (ctx.membership && entry_node < ctx.n_base) {
                    entry_partition = owner_label_checked(ctx.membership[entry_node], ctx.n_servers, "entry node");
                }

                item.beam.clear();
                item.beam.push_back({entry_node, entry_dist, 0, {0, 0, 0}});
                item.current_partition = entry_partition;
                item.entry_partition = entry_partition;

                ctx.completed_inserts_pool.pop();
            } else {
                ctx.completed_inserts_pool.pop();
            }
        }
    }

    // Head-BF is computed before the pipeline starts. A cutover can
    // publish a new owner before a later query is first dispatched, so resolve
    // the already-selected global entry node against the current membership at
    // dispatch time. Subsequent hops retain their explicit next partition.
    if (item.hops == 0 && item.beam.size() == 1 && ctx.membership_vec) {
        const uint32_t entry_node = item.beam.front().node_id;
        if (entry_node < ctx.membership_vec->size()) {
            const uint32_t entry_partition =
                owner_label_checked((*ctx.membership_vec)[entry_node], ctx.n_servers, "entry node");
            item.current_partition = entry_partition;
            item.entry_partition = entry_partition;
        }
    }

    auto now = steady_clock::now();
    item.phase_start_time = now;
    if (item.start_time.time_since_epoch().count() == 0) {
        item.start_time = item.phase_start_time;

        // Track first start time for Search/Insert wall clock separation
        if (item.type == WorkloadType::SEARCH && !ctx.search_started) {
            ctx.search_first_start = now;
            ctx.search_started = true;
        } else if (item.type == WorkloadType::INSERT && !ctx.insert_started) {
            ctx.insert_first_start = now;
            ctx.insert_started = true;
        }
    }

    if (item.type == WorkloadType::SEARCH ||
        (item.type == WorkloadType::INSERT && item.insert_phase == InsertPhase::SEARCH_PENDING)) {
        // Send SEARCH request
        uint32_t L_to_use = (item.type == WorkloadType::INSERT) ? ctx.insert_L : ctx.L;
        uint32_t K_to_use = (item.type == WorkloadType::INSERT) ? ctx.insert_R : ctx.K;
        uint32_t W_to_use = (item.type == WorkloadType::INSERT) ? ctx.insert_W : ctx.W;

        // Track search parameters for debug
        if (g_debug.enabled) {
            if (item.type == WorkloadType::INSERT) {
                g_debug.insert_L = L_to_use;
                g_debug.insert_K = K_to_use;
                g_debug.insert_W = W_to_use;
            } else {
                g_debug.query_L = L_to_use;
                g_debug.query_K = K_to_use;
            }
        }

        // P2P mode support (same as search-only)
        uint32_t server_idx = owner_label_checked(item.current_partition, ctx.n_servers, "dispatch");
        uint32_t flags = ctx.p2p_mode ? ann::STATE_FLAG_P2P : 0;
        if (ctx.trace_routes) flags |= ann::STATE_FLAG_TRACE_ROUTE;

        // Measure serialization time
        auto t0 = steady_clock::now();
        auto state_data = ann::serialize_state(
            item.id, L_to_use, K_to_use, W_to_use,
            item.current_partition,
            item.query.data(), ctx.dim,
            item.beam, flags, server_idx, item.hops,
            0, 0,  // cumulative_queue_us, cumulative_processing_us
            item.seen_nodes,  // Visited set for cross-partition hop
            ctx.trace_routes ? item.route : std::vector<ann::RouteHop>{},
            0  // cumulative_distance_computations
        );
        auto t1 = steady_clock::now();
        double this_serialize_us = duration_cast<microseconds>(t1 - t0).count();
        ctx.timing.serialize_us += this_serialize_us;

        zmq::message_t empty(0);
        zmq::message_t request(state_data.data(), state_data.size());

        auto t2 = steady_clock::now();
        sockets[server_idx].send(empty, zmq::send_flags::sndmore);
        sockets[server_idx].send(request, zmq::send_flags::none);
        auto t3 = steady_clock::now();
        double this_send_us = duration_cast<microseconds>(t3 - t2).count();
        ctx.timing.send_us += this_send_us;
        ctx.timing.send_count++;

        // Record send time (pre-serialize boundary) and coordinator timing for RTT calculation
        // t0 (before serialize), not t3: the total_us window then covers every term of
        // measured_coord_total_us()
        item.send_time = t0;
        item.last_coord_serialize_us = this_serialize_us;
        item.last_coord_send_us = this_send_us;

        ctx.in_flight[item.id] = workload_idx;

    }
    // INSERT/BACKLINK/BROADCAST ops are dispatched directly by queue_insert_ops()
    // (or from response handlers) onto g_daemon.server_queues[server].
}

// =============================================================================
// create_search_work_item - Create ServerWorkItem for search (no ZMQ send)
// For multi-thread coordinator: serialize and return to caller.
// =============================================================================
ServerWorkItem create_search_work_item(UnifiedContext& ctx, uint32_t workload_idx) {
    ServerWorkItem swi;
    if (workload_idx >= ctx.workloads.size()) return swi;

    auto& item = ctx.workloads[workload_idx];
    const auto* head_index = static_cast<const ann::HeadIndex*>(ctx.head_index);

    // Convert SEARCH to inserted query
    if (item.type == WorkloadType::SEARCH && !item.is_inserted_query &&
        ctx.enable_inserted_query && !ctx.completed_inserts_pool.empty()) {
        static thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng) < ctx.inserted_query_ratio && !ctx.completed_inserts_pool.empty()) {
            auto& completed = ctx.completed_inserts_pool.front();
            if (!completed.vector.empty() && head_index) {
                item.is_inserted_query = true;
                item.insert_gt_index = completed.insert_sequence;
                item.query = std::move(completed.vector);
                auto [entry_node, entry_dist] = head_bf_timed(head_index, item.query.data());
                uint32_t entry_partition = 0;
                if (ctx.membership_vec && entry_node < ctx.membership_vec->size())
                    entry_partition = owner_label_checked((*ctx.membership_vec)[entry_node], ctx.n_servers, "entry node");
                else if (ctx.membership && entry_node < ctx.n_base)
                    entry_partition = owner_label_checked(ctx.membership[entry_node], ctx.n_servers, "entry node");
                item.beam.clear();
                item.beam.push_back({entry_node, entry_dist, 0, {0, 0, 0}});
                item.current_partition = entry_partition;
                item.entry_partition = entry_partition;
                ctx.completed_inserts_pool.pop();
            } else {
                ctx.completed_inserts_pool.pop();
            }
        }
    }

    auto now = steady_clock::now();
    item.phase_start_time = now;
    if (item.start_time.time_since_epoch().count() == 0) {
        item.start_time = now;
        if (item.type == WorkloadType::SEARCH && !ctx.search_started) {
            ctx.search_first_start = now; ctx.search_started = true;
        } else if (item.type == WorkloadType::INSERT && !ctx.insert_started) {
            ctx.insert_first_start = now; ctx.insert_started = true;
        }
    }

    uint32_t L_to_use = (item.type == WorkloadType::INSERT) ? ctx.insert_L : ctx.L;
    uint32_t K_to_use = (item.type == WorkloadType::INSERT) ? ctx.insert_R : ctx.K;
    uint32_t W_to_use = (item.type == WorkloadType::INSERT) ? ctx.insert_W : ctx.W;
    uint32_t server_idx = owner_label_checked(item.current_partition, ctx.n_servers, "dispatch");
    uint32_t flags = ctx.p2p_mode ? ann::STATE_FLAG_P2P : 0;
    if (ctx.trace_routes) flags |= ann::STATE_FLAG_TRACE_ROUTE;

    auto t0 = steady_clock::now();
    swi.request_data = ann::serialize_state(
        item.id, L_to_use, K_to_use, W_to_use,
        item.current_partition,
        item.query.data(), ctx.dim,
        item.beam, flags, server_idx, item.hops,
        0, 0, item.seen_nodes,
        ctx.trace_routes ? item.route : std::vector<ann::RouteHop>{},
        0  // cumulative_distance_computations
    );
    auto t1 = steady_clock::now();
    double this_serialize_us = duration_cast<microseconds>(t1 - t0).count();
    ctx.timing.serialize_us += this_serialize_us;
    item.last_coord_serialize_us = this_serialize_us;
    item.send_time = t1;

    swi.workload_id = item.id;
    swi.target_server = server_idx;
    ctx.in_flight[item.id] = workload_idx;

    return swi;
}

} // namespace tide
