#include "coordinator_types.h"
#include "pipeline_types.h"
#include "coordinator_types.h"
#include "head_index.h"
#include "protocol/insert_protocol.h"
#include "protocol/broadcast_messages.h"

#include <iostream>
#include <chrono>
#include <unordered_map>
#include <random>
#include <atomic>

using namespace std::chrono;

namespace tide {

// Head-index search threads: 8, or TIDE_HEAD_BF_THREADS (1..64).
static int head_bf_helper_count() {
    if (const char* env = std::getenv("TIDE_HEAD_BF_THREADS")) {
        int n = std::atoi(env);
        if (n > 0 && n <= 64) return n;
    }
    return 8;
}

// Fills the entry fields of one item. Each helper thread writes its own items.
static inline void apply_head_bf(
    WorkloadItem& item,
    const ann::HeadIndex* head_index,
    const uint32_t* membership,
    uint32_t n_servers
) {
    const float* q = item.query.data();
    auto [entry_node, entry_dist] = head_index->find_nearest_with_dist(q);
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
    auto& accounting = ctx.serving;
    auto steady_ns = [](steady_clock::time_point stamp) {
        return duration_cast<nanoseconds>(stamp.time_since_epoch()).count();
    };
    if (accounting.enabled) accounting.materialize.begin_ns = steady_ns(materialize_start);
    ctx.workloads.clear();
    ctx.workloads.reserve(ctx.total_workloads);

    // Serial pass: assign types and query rows (the insert interleave depends on order).
    uint32_t search_idx = 0;
    uint32_t insert_idx = 0;
    // Workload ids are never reused, so a late reply cannot match a later command.
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
            // position in this command, not the source query index
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
    }

    const auto materialize_end = steady_clock::now();
    if (accounting.enabled) accounting.materialize.end_ns = steady_ns(materialize_end);

    // Head-index search runs on helper threads. Dispatch starts after they join.
    auto t_bf_start = steady_clock::now();
    if (accounting.enabled) accounting.head_bf.begin_ns = steady_ns(t_bf_start);
    const int N_HELPER = head_bf_helper_count();
    const uint32_t* membership = ctx.membership;
    const uint32_t n_servers = ctx.n_servers;
    const uint32_t N = static_cast<uint32_t>(ctx.workloads.size());
    auto* workloads = &ctx.workloads;

    std::vector<std::thread> threads;
    threads.reserve(N_HELPER);
    for (int tid = 0; tid < N_HELPER; tid++) {
        threads.emplace_back(
            [&, tid, N_HELPER, N, workloads, head_index, membership, n_servers]() {
                for (uint32_t i = static_cast<uint32_t>(tid); i < N;
                     i += static_cast<uint32_t>(N_HELPER)) {
                    apply_head_bf((*workloads)[i], head_index, membership, n_servers);
                }
            });
    }
    for (auto& t : threads) t.join();
    auto t_bf_end = steady_clock::now();
    if (accounting.enabled) accounting.head_bf.end_ns = steady_ns(t_bf_end);

    // debug counters, serially
    if (g_debug.enabled) {
        for (const auto& item : ctx.workloads) {
            uint32_t ep = item.entry_partition;
            if (ep < g_debug.entry_partition_count.size()) {
                g_debug.entry_partition_count[ep]++;
            }
        }
    }
    std::cout << "  Generated workloads: " << N << " (" << search_idx << " search, " << insert_idx << " insert) in "
              << duration_cast<microseconds>(materialize_end - materialize_start).count() / 1000.0
              << " ms, head index " << duration_cast<microseconds>(t_bf_end - t_bf_start).count() / 1000.0
              << " ms (" << N_HELPER << " threads)\n" << std::flush;
}

// Sends the insert and its broadcasts once the insert's search is done.
void queue_insert_ops(UnifiedContext& ctx, uint32_t workload_idx) {
    auto& item = ctx.workloads[workload_idx];

    uint32_t assigned_id = ctx.n_base + item.insert_sequence;
    item.assigned_id = assigned_id;

    if (!item.search_neighbors.empty()) {
        item.target_partition = get_node_partition(ctx, item.search_neighbors[0]);
    }
    uint32_t target_server = owner_label_checked(item.target_partition, ctx.n_servers, "insert target");

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

    if (ctx.pending_mgr) {
        ctx.pending_mgr->add(assigned_id, item.query, item.target_partition);
    }

    ctx.inserted_vectors[assigned_id] = item.query;
    ctx.inserted_partitions[assigned_id] = item.target_partition;

    if (ctx.membership_vec && assigned_id > 0) {
        if (assigned_id >= ctx.membership_vec->size()) {
            ctx.membership_vec->resize(assigned_id + 1);
        }
        (*ctx.membership_vec)[assigned_id] = item.target_partition;
    }

    // Backlinks wait for INSERT_RESPONSE (post-prune edges).
    if (!ctx.no_backlink) {
        item.backlinks_sent = 0;
        item.backlinks_acked = 0;
        item.backlink_done = false;
        ctx.insert_response_pending_map[item.id] = workload_idx;
    } else {
        item.backlink_done = true;
        ctx.backlink_phase_completed++;
    }

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

    item.insert_phase = InsertPhase::INSERT_PENDING;
    item.phase_start_time = steady_clock::now();

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

    // Some searches become queries for inserted vectors.
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

                auto [entry_node, entry_dist] = head_index->find_nearest_with_dist(item.query.data());

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

    // Resolve the entry node's owner at dispatch: a batch may have moved it since the head search.
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
        uint32_t L_to_use = (item.type == WorkloadType::INSERT) ? ctx.insert_L : ctx.L;
        uint32_t K_to_use = (item.type == WorkloadType::INSERT) ? ctx.insert_R : ctx.K;
        uint32_t W_to_use = (item.type == WorkloadType::INSERT) ? ctx.insert_W : ctx.W;

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

        uint32_t server_idx = owner_label_checked(item.current_partition, ctx.n_servers, "dispatch");
        uint32_t flags = ctx.p2p_mode ? ann::STATE_FLAG_P2P : 0;
        if (ctx.trace_routes) flags |= ann::STATE_FLAG_TRACE_ROUTE;

        auto t0 = steady_clock::now();
        auto state_data = ann::serialize_state(
            item.id, L_to_use, K_to_use, W_to_use,
            item.current_partition,
            item.query.data(), ctx.dim,
            item.beam, flags, server_idx, item.hops,
            0, 0,  // cumulative_queue_us, cumulative_processing_us
            item.seen_nodes,
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

        // The RTT starts before serialization, as in measured_coord_total_us().
        item.send_time = t0;
        item.last_coord_serialize_us = this_serialize_us;
        item.last_coord_send_us = this_send_us;

        ctx.in_flight[item.id] = workload_idx;

    }
    // Inserts, backlinks and broadcasts are queued by queue_insert_ops() and the reply handlers.
}

// Serializes a search request for a server thread.
ServerWorkItem create_search_work_item(UnifiedContext& ctx, uint32_t workload_idx) {
    ServerWorkItem swi;
    if (workload_idx >= ctx.workloads.size()) return swi;

    auto& item = ctx.workloads[workload_idx];
    const auto* head_index = static_cast<const ann::HeadIndex*>(ctx.head_index);

    // some searches become queries for inserted vectors
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
                auto [entry_node, entry_dist] = head_index->find_nearest_with_dist(item.query.data());
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
