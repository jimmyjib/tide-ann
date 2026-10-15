// Workload pipeline: dispatch, reply handling and per-command metrics.

#include "pipeline_types.h"
#include "coordinator_types.h"
#include "controller_hooks.h"
#include "window_snapshot_wire.h"
#include "workload_progress.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <set>
#include <map>
#include <thread>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <stdexcept>

using namespace std::chrono;

namespace tide {

// Rolling Recall@K of baseline queries, for the controller's recall guard.
template <typename Item>
static void rolling_recall_accumulate(UnifiedContext& ctx, const Item& item) {
    if (ctx.self_recall || item.is_inserted_query) return;
    if (ctx.groundtruth == nullptr || item.query_index >= ctx.n_searches) {
        return;
    }
    const int32_t* gt = ctx.groundtruth +
                        static_cast<size_t>(item.query_index) * ctx.gt_cols;
    ctx.rolling_recall_sum += compute_recall(item.result_neighbors, gt, ctx.K);
    ctx.rolling_recall_count += 1;
}

std::vector<std::atomic<int32_t>> g_coord_thread_in_flight_by_origin;
std::atomic<int32_t> g_coord_thread_global_search_in_flight{0};
std::atomic<bool> g_coord_thread_p2p_polling_enabled{false};
std::atomic<uint32_t> g_coord_thread_batch_per_server{8};
std::mutex g_coord_thread_in_flight_mutex;
std::unordered_map<uint32_t, CoordinatorInFlightRecord> g_coord_thread_in_flight_records;

// Past this many hops the coordinator prints an [INFINITE_HOP] line (it keeps forwarding).
constexpr uint32_t kExternalSafetyHopCap = 100;

static void mark_insert_done_if_ready(UnifiedContext& ctx, WorkloadItem& item) {
    if (!item.backlink_done || !item.broadcast_done ||
        item.insert_phase == InsertPhase::DONE) {
        return;
    }

    item.insert_phase = InsertPhase::DONE;
    item.done = true;
    if (ctx.pending_mgr) {
        ctx.pending_mgr->set_committed(item.assigned_id);
        ctx.pending_mgr->remove(item.assigned_id);
    }
    ctx.broadcast_phase_completed++;
}

static void mark_daemon_insert_done_if_ready(WorkloadItem& item) {
    if (!item.backlink_done || !item.broadcast_done ||
        item.insert_phase == InsertPhase::DONE) {
        return;
    }

    item.insert_phase = InsertPhase::DONE;
    item.done = true;
    if (item.assigned_id != 0) {
        g_daemon.pending_mgr.set_committed(item.assigned_id);
        g_daemon.pending_mgr.remove(item.assigned_id);
    }
}

static void queue_exact_backlinks_from_insert_response(
    UnifiedContext& ctx,
    uint32_t workload_idx,
    const std::vector<ann::CrossPartitionEdge>& cross_edges
) {
    if (workload_idx >= ctx.workloads.size()) return;
    auto& item = ctx.workloads[workload_idx];

    std::unordered_map<uint32_t, std::vector<uint32_t>> server_to_to_ids;
    uint32_t exact_edges = 0;
    for (const auto& edge : cross_edges) {
        if (edge.from_partition == item.target_partition) continue;
        uint32_t server_idx = owner_label_checked(edge.from_partition, ctx.n_servers, "backlink fanout");
        server_to_to_ids[server_idx].push_back(edge.from_id);
        exact_edges++;
    }
    item.pending_backlinks = cross_edges;
    ctx.total_backlinks_sent += exact_edges;

    item.backlinks_sent = 0;
    item.backlinks_acked = 0;
    if (server_to_to_ids.empty()) {
        if (!item.backlink_done) {
            item.backlink_done = true;
            ctx.backlink_phase_completed++;
        }
        mark_insert_done_if_ready(ctx, item);
        return;
    }

    item.backlink_done = false;
    for (auto& [server_idx, to_ids] : server_to_to_ids) {
        auto batch_data = ann::serialize_batch_backlink_request(
            item.id, item.assigned_id, item.target_partition,
            ctx.dim, ctx.insert_R, ctx.insert_alpha,
            to_ids, item.query.data()
        );
        ServerWorkItem swi;
        swi.request_data = std::move(batch_data);
        swi.target_server = server_idx;
        swi.workload_id = item.id;
        g_daemon.server_queues[server_idx]->push(std::move(swi));
        item.backlinks_sent++;
        g_daemon.maintenance.backlink_batches_sent++;
        g_daemon.maintenance.backlink_edges_sent += to_ids.size();
    }
    ctx.backlink_pending_map[item.id] = workload_idx;
}

static void queue_exact_backlinks_from_daemon_insert_response(
    UnifiedContext& ctx,
    uint32_t workload_idx,
    const std::vector<ann::CrossPartitionEdge>& cross_edges
) {
    if (workload_idx >= g_daemon.pending_workloads.size()) return;
    auto& item = g_daemon.pending_workloads[workload_idx];

    std::unordered_map<uint32_t, std::vector<uint32_t>> server_to_to_ids;
    uint32_t exact_edges = 0;
    for (const auto& edge : cross_edges) {
        if (edge.from_partition == item.target_partition) continue;
        uint32_t server_idx = owner_label_checked(edge.from_partition, ctx.n_servers, "backlink fanout");
        server_to_to_ids[server_idx].push_back(edge.from_id);
        exact_edges++;
    }
    item.pending_backlinks = cross_edges;
    ctx.total_backlinks_sent += exact_edges;

    item.backlinks_sent = 0;
    item.backlinks_acked = 0;
    if (server_to_to_ids.empty()) {
        item.backlink_done = true;
        mark_daemon_insert_done_if_ready(item);
        return;
    }

    item.backlink_done = false;
    for (auto& [server_idx, to_ids] : server_to_to_ids) {
        auto batch_data = ann::serialize_batch_backlink_request(
            item.id, item.assigned_id, item.target_partition,
            ctx.dim, ctx.insert_R, ctx.insert_alpha,
            to_ids, item.query.data()
        );
        if (server_idx < g_daemon.server_queues.size() &&
            g_daemon.server_queues[server_idx]) {
            ServerWorkItem swi;
            swi.request_data = std::move(batch_data);
            swi.target_server = server_idx;
            swi.workload_id = item.id;
            g_daemon.server_queues[server_idx]->push(std::move(swi));
            item.backlinks_sent++;
            g_daemon.maintenance.backlink_batches_sent++;
            g_daemon.maintenance.backlink_edges_sent += to_ids.size();
        }
    }
    if (item.backlinks_sent == 0) {
        item.backlink_done = true;
        mark_daemon_insert_done_if_ready(item);
        return;
    }
    g_daemon.pending_backlink_map[item.id] = workload_idx;
}

static void process_insert_response_payload(
    UnifiedContext& ctx,
    const char* data,
    size_t size
) {
    ann::InsertResponseHeader hdr;
    std::vector<uint32_t> neighbors;
    std::vector<ann::CrossPartitionEdge> cross_edges;
    if (!ann::deserialize_insert_response(data, size, hdr, neighbors, cross_edges)) {
        return;
    }

    ctx.insert_rpc_phase_completed++;
    ctx.total_insert_queue_wait_us += hdr.queue_wait_us;
    ctx.total_insert_deserialize_us += hdr.deserialize_us;
    ctx.total_insert_alloc_lock_us += hdr.alloc_lock_wait_us;
    ctx.total_insert_edge_lock_us += hdr.edge_lock_wait_us;
    ctx.total_insert_prune_us += hdr.prune_us;
    ctx.total_insert_alloc_memory_us += hdr.alloc_memory_us;
    ctx.total_insert_edge_memory_us += hdr.edge_memory_us;
    ctx.total_insert_local_backlink_prune_us += hdr.local_backlink_prune_us;
    ctx.resize_full_base.count += hdr.resize_full_base_count;
    ctx.resize_full_base.time_us += hdr.resize_full_base_us;
    ctx.resize_membership.count += hdr.resize_membership_count;
    ctx.resize_membership.time_us += hdr.resize_membership_us;
    ctx.resize_adj.count += hdr.resize_adj_count;
    ctx.resize_adj.time_us += hdr.resize_adj_us;
    ctx.resize_global_to_local.count += hdr.resize_global_to_local_count;
    ctx.resize_global_to_local.time_us += hdr.resize_global_to_local_us;

    auto pit = ctx.insert_response_pending_map.find(hdr.request_id);
    if (pit == ctx.insert_response_pending_map.end()) {
        auto dpit = g_daemon.pending_insert_response_map.find(hdr.request_id);
        if (dpit == g_daemon.pending_insert_response_map.end()) {
            g_daemon.maintenance.orphan_insert_responses++;
            return;
        }

        uint32_t workload_idx = dpit->second;
        g_daemon.pending_insert_response_map.erase(dpit);
        if (workload_idx >= g_daemon.pending_workloads.size()) {
            g_daemon.maintenance.orphan_insert_responses++;
            return;
        }

        auto& item = g_daemon.pending_workloads[workload_idx];
        if (item.insert_response_seen) {
            g_daemon.maintenance.orphan_insert_responses++;
            return;
        }
        item.insert_response_seen = true;
        item.assigned_id = hdr.assigned_id;
        item.insert_us = duration_cast<microseconds>(
            steady_clock::now() - item.phase_start_time).count();

        queue_exact_backlinks_from_daemon_insert_response(ctx, workload_idx, cross_edges);
        return;
    }
    uint32_t workload_idx = pit->second;
    if (workload_idx >= ctx.workloads.size()) {
        ctx.insert_response_pending_map.erase(pit);
        g_daemon.maintenance.orphan_insert_responses++;
        return;
    }

    auto& item = ctx.workloads[workload_idx];
    if (item.insert_response_seen) {
        ctx.insert_response_pending_map.erase(pit);
        g_daemon.maintenance.orphan_insert_responses++;
        return;
    }
    item.insert_response_seen = true;
    item.assigned_id = hdr.assigned_id;
    item.insert_us = duration_cast<microseconds>(
        steady_clock::now() - item.phase_start_time).count();

    uint32_t target_server = item.target_partition;
    if (target_server < UnifiedContext::MAX_SERVERS) {
        ctx.resize_per_server[target_server].full_base.count += hdr.resize_full_base_count;
        ctx.resize_per_server[target_server].full_base.time_us += hdr.resize_full_base_us;
        ctx.resize_per_server[target_server].membership.count += hdr.resize_membership_count;
        ctx.resize_per_server[target_server].membership.time_us += hdr.resize_membership_us;
        ctx.resize_per_server[target_server].adj.count += hdr.resize_adj_count;
        ctx.resize_per_server[target_server].adj.time_us += hdr.resize_adj_us;
        ctx.resize_per_server[target_server].global_to_local.count += hdr.resize_global_to_local_count;
        ctx.resize_per_server[target_server].global_to_local.time_us += hdr.resize_global_to_local_us;
    }

    if (!ctx.no_backlink) {
        queue_exact_backlinks_from_insert_response(ctx, workload_idx, cross_edges);
    }
    ctx.insert_response_pending_map.erase(pit);
}

static void process_backlink_ack_payload(
    UnifiedContext& ctx,
    const char* data,
    size_t size
) {
    uint32_t batch_id = 0;
    if (size >= 8) std::memcpy(&batch_id, data + 4, sizeof(uint32_t));
    auto it = ctx.backlink_pending_map.find(batch_id);
    if (it == ctx.backlink_pending_map.end()) {
        auto dit = g_daemon.pending_backlink_map.find(batch_id);
        if (dit == g_daemon.pending_backlink_map.end()) {
            g_daemon.maintenance.orphan_backlink_acks++;
            return;
        }

        uint32_t idx = dit->second;
        if (idx >= g_daemon.pending_workloads.size()) {
            g_daemon.pending_backlink_map.erase(dit);
            return;
        }

        auto& item = g_daemon.pending_workloads[idx];
        item.backlinks_acked++;
        ctx.total_backlinks_acked++;
        if (item.backlinks_acked >= item.backlinks_sent) {
            item.backlink_us = duration_cast<microseconds>(
                steady_clock::now() - item.phase_start_time).count();
            item.backlink_done = true;
            g_daemon.pending_backlink_map.erase(dit);
            mark_daemon_insert_done_if_ready(item);
        }
        return;
    }

    uint32_t idx = it->second;
    if (idx >= ctx.workloads.size()) {
        ctx.backlink_pending_map.erase(it);
        return;
    }
    auto& item = ctx.workloads[idx];
    item.backlinks_acked++;
    ctx.total_backlinks_acked++;
    if (item.backlinks_acked >= item.backlinks_sent) {
        item.backlink_us = duration_cast<microseconds>(
            steady_clock::now() - item.phase_start_time).count();
        item.backlink_done = true;
        ctx.backlink_phase_completed++;
        ctx.backlink_pending_map.erase(it);
        mark_insert_done_if_ready(ctx, item);
    }
}

static void process_broadcast_ack_payload(
    UnifiedContext& ctx,
    const char* data,
    size_t size
) {
    uint32_t node_id = 0;
    if (size >= 8) std::memcpy(&node_id, data + 4, sizeof(uint32_t));
    auto it = ctx.broadcast_pending_map.find(node_id);
    if (it == ctx.broadcast_pending_map.end()) {
        auto dit = g_daemon.pending_broadcast_map.find(node_id);
        if (dit == g_daemon.pending_broadcast_map.end()) {
            g_daemon.maintenance.orphan_broadcast_acks++;
            return;
        }

        uint32_t idx = dit->second;
        if (idx >= g_daemon.pending_workloads.size()) {
            g_daemon.pending_broadcast_map.erase(dit);
            return;
        }

        auto& item = g_daemon.pending_workloads[idx];
        item.broadcasts_acked++;
        ctx.total_broadcasts_acked++;
        if (item.broadcasts_acked >= g_daemon.n_servers) {
            item.broadcast_us = duration_cast<microseconds>(
                steady_clock::now() - item.phase_start_time).count();
            item.broadcast_done = true;
            g_daemon.pending_broadcast_map.erase(dit);
            mark_daemon_insert_done_if_ready(item);
        }
        return;
    }

    uint32_t idx = it->second;
    if (idx >= ctx.workloads.size()) {
        ctx.broadcast_pending_map.erase(it);
        return;
    }
    auto& item = ctx.workloads[idx];
    item.broadcasts_acked++;
    ctx.total_broadcasts_acked++;
    if (item.broadcasts_acked >= ctx.n_servers) {
        item.broadcast_us = duration_cast<microseconds>(
            steady_clock::now() - item.phase_start_time).count();
        item.broadcast_done = true;
        ctx.broadcast_pending_map.erase(it);
        mark_insert_done_if_ready(ctx, item);
    }
}

static void handoff_pending_context_state(UnifiedContext& ctx) {
    if (ctx.insert_response_pending_map.empty() &&
        ctx.backlink_pending_map.empty() &&
        ctx.broadcast_pending_map.empty()) {
        return;
    }

    if (g_daemon.pending_insert_response_map.empty() &&
        g_daemon.pending_backlink_map.empty() &&
        g_daemon.pending_broadcast_map.empty()) {
        g_daemon.pending_workloads.clear();
    }

    uint32_t offset = static_cast<uint32_t>(g_daemon.pending_workloads.size());
    g_daemon.pending_workloads.reserve(g_daemon.pending_workloads.size() + ctx.workloads.size());
    g_daemon.pending_workloads.insert(
        g_daemon.pending_workloads.end(), ctx.workloads.begin(), ctx.workloads.end());

    for (const auto& [request_id, idx] : ctx.insert_response_pending_map) {
        g_daemon.pending_insert_response_map[request_id] = offset + idx;
    }
    for (const auto& [batch_id, idx] : ctx.backlink_pending_map) {
        g_daemon.pending_backlink_map[batch_id] = offset + idx;
    }
    for (const auto& [node_id, idx] : ctx.broadcast_pending_map) {
        g_daemon.pending_broadcast_map[node_id] = offset + idx;
    }

    ctx.insert_response_pending_map.clear();
    ctx.backlink_pending_map.clear();
    ctx.broadcast_pending_map.clear();
}

// Per-server thread: sends queued requests and receives the replies.
void coordinator_server_thread(
    uint32_t server_idx,
    zmq::socket_t& socket,
    ann::ThreadSafeQueue<ServerWorkItem>& work_queue,
    ann::ThreadSafeQueue<ServerResponse>& response_queue,
    std::vector<std::atomic<int32_t>>& in_flight_by_origin,
    std::atomic<int32_t>& global_search_in_flight,
    std::mutex& in_flight_mutex,
    std::unordered_map<uint32_t, CoordinatorInFlightRecord>& in_flight_records,
    std::atomic<bool>& running
) {
    int32_t local_control_in_flight = 0;
    bool has_deferred_work = false;
    ServerWorkItem deferred_work;

    while (running) {
        while (true) {
            int32_t search_in_flight = in_flight_by_origin[server_idx].load(std::memory_order_acquire);
            int32_t total_in_flight = search_in_flight + local_control_in_flight;
            uint32_t batch_per_server = g_coord_thread_batch_per_server.load(std::memory_order_acquire);
            if (batch_per_server == 0) batch_per_server = 1;
            // above the servers' backlink and broadcast batch sizes, so drains progress
            uint32_t control_batch_per_server = std::max<uint32_t>(batch_per_server, 1024);
            if (total_in_flight < 0) {
                std::cerr << "[ServerThread " << server_idx << "] Warning: total_in_flight="
                          << total_in_flight << "\n";
                local_control_in_flight = 0;
                total_in_flight = search_in_flight;
            }

            ServerWorkItem work;
            if (has_deferred_work) {
                work = std::move(deferred_work);
                has_deferred_work = false;
            } else if (!work_queue.try_pop(work)) {
                break;
            }

            ann::MessageType request_type = ann::get_message_type(
                work.request_data.data(), work.request_data.size());
            bool is_search_request = (request_type == ann::MessageType::SEARCH_REQUEST);
            // every non-search request gets exactly one reply
            if (is_search_request &&
                static_cast<uint32_t>(std::max<int32_t>(search_in_flight, 0)) >= batch_per_server) {
                deferred_work = std::move(work);
                has_deferred_work = true;
                break;
            }
            if (!is_search_request &&
                static_cast<uint32_t>(std::max<int32_t>(local_control_in_flight, 0)) >= control_batch_per_server) {
                deferred_work = std::move(work);
                has_deferred_work = true;
                break;
            }

            auto send_begin = steady_clock::now();
            // Register the launch before sending: in P2P the final reply can arrive on another thread first.
            if (is_search_request) {
                {
                    std::lock_guard<std::mutex> lock(in_flight_mutex);
                    in_flight_records[work.workload_id] = {server_idx, send_begin, 0.0};
                }
                in_flight_by_origin[server_idx].fetch_add(1, std::memory_order_acq_rel);
                global_search_in_flight.fetch_add(1, std::memory_order_acq_rel);
            }
            zmq::message_t empty(0);
            zmq::message_t request(work.request_data.data(), work.request_data.size());
            socket.send(empty, zmq::send_flags::sndmore);
            socket.send(request, zmq::send_flags::none);
            auto send_end = steady_clock::now();
            double this_send_us = duration_cast<microseconds>(send_end - send_begin).count();

            if (is_search_request) {
                // the reply may already have removed the record
                std::lock_guard<std::mutex> lock(in_flight_mutex);
                auto record = in_flight_records.find(work.workload_id);
                if (record != in_flight_records.end()) record->second.coord_send_us = this_send_us;
            } else {
                local_control_in_flight++;
            }
        }

        int32_t search_in_flight = in_flight_by_origin[server_idx].load(std::memory_order_acquire);
        int32_t total_in_flight = search_in_flight + local_control_in_flight;
        int32_t global_search_count = global_search_in_flight.load(std::memory_order_acquire);
        bool p2p_polling = g_coord_thread_p2p_polling_enabled.load(std::memory_order_acquire);

        // DRAIN can flush replies nothing counted, so check the socket once even when idle.
        zmq::pollitem_t poll_item = {static_cast<void*>(socket), 0, ZMQ_POLLIN, 0};
        bool idle_no_counted_work = (total_in_flight == 0 && work_queue.size() == 0 &&
                                     (!p2p_polling || global_search_count == 0));
        zmq::poll(&poll_item, 1, idle_no_counted_work
                                  ? std::chrono::milliseconds(0)
                                  : std::chrono::milliseconds(1));
        if (!(poll_item.revents & ZMQ_POLLIN)) {
            if (idle_no_counted_work) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }
            static thread_local uint32_t poll_miss_count = 0;
            poll_miss_count++;
            if (poll_miss_count % 5000 == 0 && total_in_flight > 0) {
                std::cerr << "[ST" << server_idx << "] poll miss " << poll_miss_count
                          << " in_flight=" << total_in_flight
                          << " search_in_flight=" << search_in_flight
                          << " local_control=" << local_control_in_flight
                          << " q=" << work_queue.size() << std::endl;
            }
            continue;
        }

        while (true) {
            zmq::pollitem_t check = {static_cast<void*>(socket), 0, ZMQ_POLLIN, 0};
            if (zmq::poll(&check, 1, std::chrono::milliseconds(0)) <= 0) break;

            zmq::message_t resp_empty, resp_data;
            auto recv_start = steady_clock::now();
            auto recv_result = socket.recv(resp_empty, zmq::recv_flags::dontwait);
            if (!recv_result) break;
            const auto body_result = socket.recv(resp_data, zmq::recv_flags::dontwait);
            if (!body_result) throw std::runtime_error("missing server response body");
            if (resp_data.size() && *static_cast<const uint8_t*>(resp_data.data()) ==
                    static_cast<uint8_t>(ann::MessageType::STATS_WINDOW_SNAPSHOT_RESPONSE) &&
                (resp_empty.size() || !resp_empty.more() || resp_data.more()))
                throw std::runtime_error("native snapshot must be one delimited response body");
            auto recv_end = steady_clock::now();

            ServerResponse resp;
            resp.server_idx = server_idx;
            // every known id, inserted ones included (total_inserted is atomic)
            const auto retained_bytes = window_snapshot_queue_bytes(
                resp_data.data(), resp_data.size(),
                g_daemon.n_base + g_daemon.total_inserted.load(std::memory_order_relaxed),
                static_cast<uint32_t>(g_daemon.n_servers));
            resp.response_data.assign(
                static_cast<char*>(resp_data.data()),
                static_cast<char*>(resp_data.data()) + retained_bytes);
            resp.recv_time = recv_end;
            resp.coord_recv_us = duration_cast<microseconds>(recv_end - recv_start).count();

            ann::MessageType response_type = ann::get_message_type(
                resp.response_data.data(), resp.response_data.size());
            bool matched_search_response = false;

            // Release the origin server's launch slot. Under P2P the reply can come on another socket.
            if (response_type == ann::MessageType::SEARCH_RESPONSE &&
                resp.response_data.size() > 1 + sizeof(uint32_t)) {
                uint32_t query_id;
                std::memcpy(&query_id, resp.response_data.data() + 1, sizeof(uint32_t));
                uint32_t origin_server = server_idx;
                bool found = false;
                {
                    std::lock_guard<std::mutex> lock(in_flight_mutex);
                    auto sit = in_flight_records.find(query_id);
                    if (sit != in_flight_records.end()) {
                        origin_server = sit->second.origin_server;
                        resp.send_time = sit->second.send_begin;
                        resp.coord_send_us = sit->second.coord_send_us;
                        in_flight_records.erase(sit);
                        found = true;
                    }
                }
                if (found && origin_server < in_flight_by_origin.size()) {
                    int32_t after = in_flight_by_origin[origin_server].fetch_sub(
                        1, std::memory_order_acq_rel) - 1;
                    if (after < 0) {
                        std::cerr << "[ST" << server_idx << "] shared in_flight UNDERFLOW origin="
                                  << origin_server << std::endl;
                        in_flight_by_origin[origin_server].store(0, std::memory_order_release);
                    }
                    int32_t global_after = global_search_in_flight.fetch_sub(
                        1, std::memory_order_acq_rel) - 1;
                    if (global_after < 0) {
                        std::cerr << "[ST" << server_idx
                                  << "] global_search_in_flight UNDERFLOW" << std::endl;
                        global_search_in_flight.store(0, std::memory_order_release);
                    }
                    matched_search_response = true;
                }
            }

            resp.enqueue_time = steady_clock::now();
            response_queue.push(std::move(resp));
            if (response_type == ann::MessageType::DRAIN_ACK) {
                // DRAIN_ACK closes this socket's control window.
                local_control_in_flight = 0;
            } else if (!matched_search_response &&
                       response_type != ann::MessageType::SEARCH_RESPONSE) {
                if (local_control_in_flight > 0) {
                    local_control_in_flight--;
                }
            }
        }
    }
}

void process_server_response(UnifiedContext& ctx, ServerResponse& srv_resp, uint32_t& completed) {
    // Controller traffic is handled by the controller.
    if (controller_offer_owned(srv_resp.server_idx, srv_resp.response_data)) return;
    auto process_start = steady_clock::now();
    auto recv_end = srv_resp.recv_time;
    uint32_t server_idx = srv_resp.server_idx;
    if (srv_resp.response_data.empty()) return;

    double coord_response_queue_us = 0;
    if (srv_resp.enqueue_time.time_since_epoch().count() > 0) {
        coord_response_queue_us = duration_cast<microseconds>(
            process_start - srv_resp.enqueue_time).count();
        if (coord_response_queue_us < 0) coord_response_queue_us = 0;
    }

    ann::MessageType msg_type = ann::get_message_type(
        srv_resp.response_data.data(), srv_resp.response_data.size());

    if (msg_type == ann::MessageType::SEARCH_RESPONSE) {
        ann::StateResponseHeader resp_header;
        std::vector<ann::BeamEntry> new_beam;
        std::vector<uint32_t> neighbors;
        std::vector<float> distances;
        std::vector<uint32_t> resp_seen_nodes;
        std::vector<ann::RouteHop> resp_route;

        size_t total_resp_size = srv_resp.response_data.size() - 1;
        size_t search_resp_size = total_resp_size;

        auto deser_start = steady_clock::now();
        if (!ann::deserialize_response(
                srv_resp.response_data.data() + 1,
                search_resp_size,
                resp_header, new_beam, neighbors, distances, resp_seen_nodes, resp_route)) {
            return;
        }
        auto deser_end = steady_clock::now();
        double this_deser_us = duration_cast<microseconds>(deser_end - deser_start).count();
        ctx.timing.deserialize_us += this_deser_us;

        uint32_t workload_id = resp_header.query_id;
        auto it = ctx.in_flight.find(workload_id);
        if (it == ctx.in_flight.end()) return;
        auto& item = ctx.workloads[it->second];

        if (srv_resp.send_time.time_since_epoch().count() > 0) {
            item.send_time = srv_resp.send_time;
        }
        double rtt = duration_cast<microseconds>(deser_end - item.send_time).count();
        ctx.hop_rtt_us.push_back(rtt);

        HopLatency hop_lat;
        hop_lat.from_partition = item.current_partition;
        hop_lat.to_partition = resp_header.next_partition;
        hop_lat.total_us = rtt;
        hop_lat.server_processing_us = resp_header.server_processing_us;
        hop_lat.queue_wait_us = resp_header.queue_wait_us;
        hop_lat.lock_wait_us = resp_header.shared_lock_wait_us;
        hop_lat.pq_distances = resp_header.pq_distance_count;
        hop_lat.full_distances = resp_header.full_distance_count;
        hop_lat.local_nodes_expanded = resp_header.local_nodes_expanded;
        hop_lat.top_W_count = resp_header.top_W_count;
        hop_lat.neighbors_evaluated = resp_header.neighbors_evaluated;
        hop_lat.zmq_recv_us = resp_header.zmq_recv_us;
        hop_lat.zmq_send_us = resp_header.zmq_send_us;
        hop_lat.serialize_us = resp_header.serialize_us;
        hop_lat.deserialize_us = resp_header.deserialize_us;
        hop_lat.queue_depth = resp_header.queue_depth;
        hop_lat.response_queue_wait_us = resp_header.response_queue_wait_us;
        hop_lat.io_processing_us = resp_header.io_processing_us;
        hop_lat.worker_overhead_us = resp_header.worker_overhead_us;
        hop_lat.send_prep_us = resp_header.send_prep_us;
        // the RTT starts at the server thread's send
        hop_lat.coord_serialize_us = 0;
        hop_lat.coord_send_us = srv_resp.coord_send_us;
        hop_lat.coord_response_queue_us = coord_response_queue_us;
        hop_lat.coord_recv_us = srv_resp.coord_recv_us;
        hop_lat.coord_deserialize_us = this_deser_us;
        hop_lat.network_us = hop_lat.pure_network_us();
        item.hop_latencies.push_back(hop_lat);
        ctx.requests_per_server[server_idx]++;
        item.total_distance_computations += resp_header.distance_computations;

        if (resp_header.status == ann::StateStatus::ERROR) {
            // A rejected query gets no reply. The command ends once nothing is in flight.
            ctx.in_flight.erase(workload_id);
            if (ctx.aborted_reason.empty())
                ctx.aborted_reason = "server " + std::to_string(server_idx) + " refused " +
                    std::string(item.type == WorkloadType::SEARCH ? "query " : "the search of insert workload ") +
                    std::to_string(workload_id) +
                    " (status ERROR: search state dimension differs)";
            return;
        }

        if (item.type == WorkloadType::SEARCH) {
            if (resp_header.status == ann::StateStatus::DONE) {
                if (ctx.p2p_mode) item.hops = resp_header.inter_partition_hops;
                else item.hops++;

                item.query_latency_us = duration_cast<microseconds>(recv_end - item.start_time).count();
                item.done = true;
                item.result_neighbors = std::move(neighbors);
                item.result_distances = std::move(distances);
                item.cumulative_queue_us = resp_header.cumulative_queue_us;
                item.cumulative_processing_us = resp_header.cumulative_processing_us;
                if (ctx.trace_routes) item.route = std::move(resp_route);

                // Score before queuing: the metric thread may merge pending inserts into the result.
                rolling_recall_accumulate(ctx, item);
                {
                    MetricEvent evt(MetricEventType::SEARCH_DONE, it->second);
                    if (ctx.groundtruth && item.query_index < ctx.n_searches) {
                        evt.gt_row = ctx.groundtruth + static_cast<size_t>(item.query_index) * ctx.gt_cols;
                        evt.gt_cols = ctx.gt_cols;
                    }
                    if (ctx.pending_mgr) evt.beam_snapshot = new_beam;
                    if (g_debug.enabled) evt.labels = owner_labels(ctx, item.result_neighbors);
                    ctx.metric_queue->push(std::move(evt));
                }

                ctx.in_flight.erase(workload_id);
                ctx.searches_completed++;
                ctx.search_last_end = steady_clock::now();
                completed++;

            } else if (resp_header.status == ann::StateStatus::NEED_HOP) {
                item.beam = std::move(new_beam);
                item.seen_nodes = std::move(resp_seen_nodes);
                if (ctx.trace_routes) item.route = std::move(resp_route);
                item.current_partition = resp_header.next_partition;
                item.hops++;

                auto swi = create_search_work_item(ctx, it->second);
                if (item.hops > kExternalSafetyHopCap) {
                    std::cerr << "[INFINITE_HOP] wid=" << workload_id
                              << " hops=" << item.hops
                              << " cap=" << kExternalSafetyHopCap
                              << " from=" << hop_lat.from_partition
                              << " to=" << resp_header.next_partition
                              << " seen=" << item.seen_nodes.size()
                              << " beam=" << item.beam.size()
                              << " target=" << swi.target_server
                              << std::endl;
                }
                g_daemon.server_queues[swi.target_server]->push(std::move(swi));
            }
        } else {
            // insert: search phase
            if (resp_header.status == ann::StateStatus::DONE) {
                if (ctx.p2p_mode) {
                    item.hops = resp_header.inter_partition_hops;
                } else {
                    item.hops++;
                }
                item.search_neighbors = std::move(neighbors);
                item.search_distances = std::move(distances);
                item.search_us = duration_cast<microseconds>(recv_end - item.start_time).count();
                item.cumulative_queue_us = resp_header.cumulative_queue_us;
                item.cumulative_processing_us = resp_header.cumulative_processing_us;
                if (ctx.trace_routes) item.route = std::move(resp_route);
                if (ctx.p2p_mode) {
                    item.hop_latencies.clear();
                }

                uint32_t old_target = item.target_partition;
                if (!item.search_neighbors.empty())
                    item.target_partition = get_node_partition(ctx, item.search_neighbors[0]);

                if (g_debug.enabled && item.target_partition < g_debug.final_partition_count.size()) {
                    g_debug.final_partition_count[item.target_partition]++;
                    if (old_target < g_debug.transition_matrix.size() &&
                        item.target_partition < g_debug.transition_matrix[old_target].size())
                        g_debug.transition_matrix[old_target][item.target_partition]++;
                    g_debug.insert_count++;
                    g_debug.insert_total_hops += item.hops;
                    if (item.hops < g_debug.insert_hop_distribution.size())
                        g_debug.insert_hop_distribution[item.hops]++;
                    for (size_t i = 0; i < item.search_neighbors.size(); i++) {
                        uint32_t p = get_node_partition(ctx, item.search_neighbors[i]);
                        if (p < g_debug.topK_partition_total.size())
                            g_debug.topK_partition_total[p]++;
                    }
                    MetricEvent evt(MetricEventType::INSERT_SEARCH_DONE, it->second);
                    evt.from_partition = old_target;
                    evt.labels = owner_labels(ctx, item.search_neighbors);
                    ctx.metric_queue->push(std::move(evt));
                }

                ctx.insert_search_phase_completed++;
                if (ctx.insert_search_only) {
                    item.insert_phase = InsertPhase::DONE;
                    item.done = true;
                } else {
                    uint32_t workload_idx = it->second;
                    queue_insert_ops(ctx, workload_idx);
                }
                ctx.in_flight.erase(workload_id);
                ctx.inserts_completed++;
                ctx.insert_last_end = steady_clock::now();
                completed++;

            } else if (resp_header.status == ann::StateStatus::NEED_HOP) {
                item.beam = std::move(new_beam);
                item.seen_nodes = std::move(resp_seen_nodes);
                if (ctx.trace_routes) item.route = std::move(resp_route);
                item.current_partition = resp_header.next_partition;
                item.hops++;

                auto swi = create_search_work_item(ctx, it->second);
                if (item.hops > kExternalSafetyHopCap) {
                    std::cerr << "[INFINITE_HOP_INS] wid=" << workload_id
                              << " hops=" << item.hops
                              << " cap=" << kExternalSafetyHopCap
                              << " to=" << resp_header.next_partition
                              << " seen=" << item.seen_nodes.size()
                              << " target=" << swi.target_server
                              << std::endl;
                }
                g_daemon.server_queues[swi.target_server]->push(std::move(swi));
            }
        }

    } else if (msg_type == ann::MessageType::INSERT_RESPONSE) {
        process_insert_response_payload(
            ctx,
            srv_resp.response_data.data(),
            srv_resp.response_data.size());

    } else if (msg_type == ann::MessageType::BATCH_BACKLINK_ACK) {
        process_backlink_ack_payload(
            ctx,
            srv_resp.response_data.data(),
            srv_resp.response_data.size());

    } else if (msg_type == ann::MessageType::VECTOR_BROADCAST_ACK) {
        process_broadcast_ack_payload(
            ctx,
            srv_resp.response_data.data(),
            srv_resp.response_data.size());
    }
}

void run_unified_workload(UnifiedContext& ctx) {
    install_crash_handlers();

    auto& sockets = *ctx.sockets;

    ctx.server_lock_metrics.resize(ctx.n_servers);
    ctx.requests_per_server.resize(ctx.n_servers, 0);

    if (g_coord_thread_in_flight_by_origin.size() != ctx.n_servers) {
        g_coord_thread_in_flight_by_origin = std::vector<std::atomic<int32_t>>(ctx.n_servers);
    }
    for (auto& count : g_coord_thread_in_flight_by_origin) {
        count.store(0, std::memory_order_release);
    }
    g_coord_thread_global_search_in_flight.store(0, std::memory_order_release);
    g_coord_thread_p2p_polling_enabled.store(ctx.p2p_mode, std::memory_order_release);
    g_coord_thread_batch_per_server.store(
        ctx.batch_per_server == 0 ? 1 : ctx.batch_per_server,
        std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(g_coord_thread_in_flight_mutex);
        g_coord_thread_in_flight_records.clear();
    }

    ctx.metric_queue = std::make_unique<ann::ThreadSafeQueue<MetricEvent>>();
    ctx.metric_thread_running.store(true, std::memory_order_relaxed);
    ctx.metric_thread = std::thread(run_metric_thread, std::ref(ctx));
    struct MetricThreadGuard {
        UnifiedContext& ctx;
        ~MetricThreadGuard() {
            if (!ctx.metric_thread.joinable()) return;
            MetricEvent shutdown;
            shutdown.type = MetricEventType::SHUTDOWN;
            ctx.metric_queue->push(std::move(shutdown));
            ctx.metric_thread.join();
        }
    } metric_thread_guard{ctx};

    // server threads live as long as the daemon
    if (!g_daemon.main_response_queue) {
        g_daemon.main_response_queue = std::make_unique<ann::ThreadSafeQueue<ServerResponse>>();
        g_daemon.server_queues.resize(ctx.n_servers);
        for (uint32_t i = 0; i < ctx.n_servers; i++) {
            g_daemon.server_queues[i] = std::make_unique<ann::ThreadSafeQueue<ServerWorkItem>>();
        }

        g_daemon.server_threads_running.store(true, std::memory_order_relaxed);
        for (uint32_t i = 0; i < ctx.n_servers; i++) {
            g_daemon.server_threads.emplace_back(
                coordinator_server_thread, i, std::ref(sockets[i]),
                std::ref(*g_daemon.server_queues[i]), std::ref(*g_daemon.main_response_queue),
                std::ref(g_coord_thread_in_flight_by_origin),
                std::ref(g_coord_thread_global_search_in_flight),
                std::ref(g_coord_thread_in_flight_mutex),
                std::ref(g_coord_thread_in_flight_records),
                std::ref(g_daemon.server_threads_running));
        }
    }
    auto& server_queues = g_daemon.server_queues;
    auto& main_response_queue = *g_daemon.main_response_queue;

    auto& accounting = ctx.serving;
    auto start_time = steady_clock::now();
    g_total_workloads.store(ctx.total_workloads);
    g_last_completed.store(0);

    uint32_t next_workload_idx = 0;
    uint32_t completed = 0;
    const uint32_t total = ctx.total_workloads;
    // the controller's window counters restart with each command
    static uint64_t command_sequence = 0;
    const uint64_t command_id = ++command_sequence;

    std::cout << "\nRunning workload: search=" << ctx.n_searches
              << ", insert=" << ctx.n_inserts
              << ", batch_per_server=" << ctx.batch_per_server
              << ", servers=" << ctx.n_servers << "\n";

    std::cout << "  Starting workload dispatch\n";

    uint32_t last_progress = 0;
    auto last_log_time = start_time;
    auto last_completed_time = start_time;
    uint32_t last_completed_count = 0;

    const uint32_t configured_window =
        ctx.dispatch_window > 0 ? ctx.dispatch_window : ctx.effective_batch_size;
    const size_t enqueue_window =
        std::max<size_t>(1, static_cast<size_t>(configured_window));
    auto refill_workloads = [&]() {
        // the controller pauses dispatch while it applies a batch
        if (!controller_dispatch_allowed()) return;
        // a rejected query ended the command
        if (!ctx.aborted_reason.empty()) return;
        while (ctx.in_flight.size() < enqueue_window && next_workload_idx < total) {
            auto swi = create_search_work_item(ctx, next_workload_idx++);
            g_daemon.server_queues[swi.target_server]->push(std::move(swi));
        }
    };
    refill_workloads();
    std::cout << "  Enqueue window=" << enqueue_window
              << ", initially queued=" << ctx.in_flight.size() << "\n";

    while (completed < total) {
        ServerResponse srv_resp;
        bool got_any = false;
        while (main_response_queue.try_pop(srv_resp)) {
            auto process_start = steady_clock::now();
            process_server_response(ctx, srv_resp, completed);
            auto process_end = steady_clock::now();
            ctx.timing.process_us += duration_cast<microseconds>(
                process_end - process_start).count();
            got_any = true;
        }
        ctx.timing.poll_count++;

        // A rejected query ends the command once its other queries have completed.
        if (!ctx.aborted_reason.empty() && ctx.in_flight.empty()) {
            std::cerr << "[SEARCH] " << ctx.aborted_reason << std::endl;
            break;
        }

        if (!got_any) {
            auto wait_start = steady_clock::now();
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            auto wait_end = steady_clock::now();
            ctx.timing.poll_wait_us += duration_cast<microseconds>(
                wait_end - wait_start).count();
        }

        if (controller_running()) {
            ControllerTickInputs tick_inputs;
            tick_inputs.completed_queries = ctx.searches_completed;
            tick_inputs.recall_sum = ctx.rolling_recall_sum;
            tick_inputs.recall_count = ctx.rolling_recall_count;
            tick_inputs.command_id = command_id;
            auto read_in_flight = [&](ControllerTickInputs& in) {
                size_t controller_queued = 0;
                for (const auto& queue : server_queues) {
                    if (queue) controller_queued += queue->size();
                }
                in.logical_in_flight = ctx.in_flight.size();
                in.transport_in_flight =
                    g_coord_thread_global_search_in_flight.load(
                        std::memory_order_acquire);
                in.queued_search = controller_queued;
                in.queued_responses = main_response_queue.size();
            };
            if (!controller_dispatch_allowed()) read_in_flight(tick_inputs);
            controller_tick(tick_inputs);
            if (!controller_dispatch_allowed()) {
                // a pause that nothing in flight can release ends the command
                read_in_flight(tick_inputs);
                std::string stall_reason;
                if (controller_dispatch_stalled(tick_inputs, &stall_reason)) {
                    ctx.aborted_reason =
                        "dispatch paused after an unconfirmed ownership change (" + stall_reason + "), " +
                        std::to_string(completed) + "/" + std::to_string(total) +
                        " workloads done. Run migration-accept (or migration-rollback if ACCEPT was never "
                        "sent), then controller-abort ends the terminal controller. A coordinator restart "
                        "also needs reset (or a server restart) before serving again";
                    std::cerr << "[CONTROLLER] " << ctx.aborted_reason << std::endl;
                    break;
                }
            }
        }

        refill_workloads();

        if (completed > last_completed_count) {
            last_completed_count = completed;
            last_completed_time = steady_clock::now();
        }

        auto now = steady_clock::now();
        double elapsed_since_start = duration_cast<milliseconds>(now - start_time).count();
        double stall_ms = duration_cast<milliseconds>(now - last_completed_time).count();

        // one line when nothing completed for 10 seconds
        if (stall_ms > 10000 && ctx.in_flight.size() > 0) {
            size_t queued = 0;
            for (uint32_t s = 0; s < ctx.n_servers; s++) queued += g_daemon.server_queues[s]->size();
            std::cerr << "[STALL] no completion for " << (stall_ms / 1000) << "s: completed="
                      << completed << "/" << total << " in_flight=" << ctx.in_flight.size()
                      << " server_queues=" << queued
                      << " response_queue=" << g_daemon.main_response_queue->size()
                      << " pending insert/backlink/broadcast="
                      << ctx.insert_response_pending_map.size() << "/"
                      << ctx.backlink_pending_map.size() << "/"
                      << ctx.broadcast_pending_map.size() << std::endl;
            last_completed_time = now;
        }

        if (workload_progress_due(completed, last_progress, total, 100,
                duration_cast<milliseconds>(now - last_log_time).count())) {
            uint32_t search_pending = 0, insert_pending = 0, bl_bc_pending = 0;
            for (auto& [wid, idx] : ctx.in_flight) {
                auto& item = ctx.workloads[idx];
                if (item.type == WorkloadType::SEARCH) search_pending++;
                else if (item.insert_phase == InsertPhase::SEARCH_PENDING) search_pending++;
                else if (item.insert_phase == InsertPhase::INSERT_PENDING) insert_pending++;
                else if (item.insert_phase == InsertPhase::INSERT_PENDING) bl_bc_pending++;
            }

            // for the crash handler
            g_last_completed.store(completed);
            g_in_flight_search.store(search_pending);
            g_in_flight_insert.store(insert_pending + bl_bc_pending);

            static uint32_t last_mem_log = 0;
            size_t mem_kb = get_memory_usage_kb();
            size_t peak = g_peak_memory_kb.load();
            if (mem_kb > peak) g_peak_memory_kb.store(mem_kb);

            std::cout << "  " << completed << "/" << total
                      << " [" << elapsed_since_start << "ms]"
                      << " in_flight=" << ctx.in_flight.size()
                      << " (S:" << search_pending
                      << " I:" << insert_pending
                      << " BL_BC:" << bl_bc_pending << ")"
                      << " (multi-thread)";

            if (completed >= last_mem_log + 5000) {
                std::cout << " [MEM:" << (mem_kb / 1024) << "MB]";
                last_mem_log = completed;
            }
            std::cout << "\n" << std::flush;

            last_progress = completed;
            last_log_time = now;
        }
    }

    // Backlinks and broadcasts still pending are handled by the next drain.
    handoff_pending_context_state(ctx);

    auto end_time = steady_clock::now();
    if (accounting.enabled) {
        // the serving interval ends at the last completed search
        accounting.end_ns = ann::ServingAccounting::ns_at(ctx.search_last_end);
        accounting.owner_search_begin_ns = ann::ServingAccounting::ns_at(ctx.search_first_start);
        accounting.completed_queries = ctx.searches_completed;
    }
    ctx.elapsed_ms = duration_cast<microseconds>(end_time - start_time).count() / 1000.0;

    // Join the metric thread before reading results (it merges pending inserts).
    auto metric_join_start = steady_clock::now();
    if (ctx.metric_thread.joinable()) {
        MetricEvent shutdown_evt;
        shutdown_evt.type = MetricEventType::SHUTDOWN;
        ctx.metric_queue->push(std::move(shutdown_evt));
        ctx.metric_thread.join();
    }
    auto metric_join_end = steady_clock::now();
    ctx.metric_thread_join_ms =
        duration_cast<microseconds>(metric_join_end - metric_join_start).count() / 1000.0;
    ctx.elapsed_with_metrics_ms =
        duration_cast<microseconds>(metric_join_end - start_time).count() / 1000.0;
    if (ctx.n_searches > 0) {
        uint64_t total_hops = 0;
        double total_latency_us = 0;
        uint32_t search_count = 0;

        for (const auto& item : ctx.workloads) {
            if (item.type == WorkloadType::SEARCH && item.done) {
                total_hops += item.hops;
                total_latency_us += item.query_latency_us;
                search_count++;
            }
        }

        if (search_count > 0) {
            ctx.avg_hops = static_cast<double>(total_hops) / search_count;
            ctx.avg_latency_ms = total_latency_us / search_count / 1000.0;
        }

        if (!ctx.hop_rtt_us.empty()) {
            double sum_rtt = 0;
            for (double rtt : ctx.hop_rtt_us) {
                sum_rtt += rtt;
            }
            ctx.avg_hop_latency_ms = (sum_rtt / ctx.hop_rtt_us.size()) / 1000.0;
        }
    }

    // Recall of baseline and inserted queries, and miss details.
    if (ctx.n_searches > 0) {
        double total_recall = 0;
        uint32_t search_count = 0;

        for (auto& item : ctx.workloads) {
            if (item.type == WorkloadType::SEARCH && item.done) {
                double recall = 0.0;
                const int32_t* gt = nullptr;

                if (ctx.self_recall) {
                    // a hit when the expected id is in the top K
                    uint32_t expected_id = ctx.self_ids.empty()
                        ? ctx.self_recall_base_id + item.query_index
                        : (item.query_index < ctx.self_ids.size() ? ctx.self_ids[item.query_index] : UINT32_MAX);
                    size_t k = std::min(static_cast<size_t>(ctx.K),
                                        item.result_neighbors.size());
                    bool hit = false;
                    for (size_t r = 0; r < k; r++) {
                        if (item.result_neighbors[r] == expected_id) {
                            hit = true; break;
                        }
                    }
                    recall = hit ? 1.0 : 0.0;
                    ctx.baseline_recall_sum += recall;
                    ctx.baseline_queries_completed++;
                } else if (item.is_inserted_query) {
                    if (!ctx.insert_groundtruth.empty() &&
                        item.insert_gt_index < ctx.insert_gt_rows) {
                        gt = ctx.insert_groundtruth.data() +
                             static_cast<size_t>(item.insert_gt_index) * ctx.insert_gt_cols;
                        recall = compute_recall(item.result_neighbors, gt, ctx.K);
                        ctx.inserted_recall_sum += recall;
                        ctx.inserted_queries_completed++;
                    }
                } else if (ctx.groundtruth != nullptr && item.query_index < ctx.n_searches) {
                    gt = ctx.groundtruth + static_cast<size_t>(item.query_index) * ctx.gt_cols;
                    recall = compute_recall(item.result_neighbors, gt, ctx.K);
                    ctx.baseline_recall_sum += recall;
                    ctx.baseline_queries_completed++;
                }

                total_recall += recall;

                // misses need a ground-truth row (none under self-recall)
                if (gt != nullptr && (ctx.debug_recall || recall < 1.0)) {
                    ctx.total_queries_analyzed++;

                    std::set<uint32_t> result_set(item.result_neighbors.begin(),
                        item.result_neighbors.begin() + std::min((size_t)ctx.K, item.result_neighbors.size()));
                    std::set<int32_t> gt_set(gt, gt + ctx.K);

                    std::vector<uint32_t> missed_gt;
                    for (int32_t g : gt_set) {
                        if (g >= 0 && result_set.find(static_cast<uint32_t>(g)) == result_set.end()) {
                            missed_gt.push_back(static_cast<uint32_t>(g));
                            ctx.total_gt_missed++;
                        }
                    }

                    std::vector<uint32_t> wrong_results;
                    uint32_t inserted_truly_closer = 0;

                    std::map<uint32_t, size_t> result_rank;
                    for (size_t i = 0; i < item.result_neighbors.size(); i++) {
                        result_rank[item.result_neighbors[i]] = i;
                    }

                    for (uint32_t r : result_set) {
                        if (gt_set.find(static_cast<int32_t>(r)) == gt_set.end()) {
                            wrong_results.push_back(r);
                            ctx.total_wrong_results++;
                            if (ctx.insert_threshold_id > 0 && r >= ctx.insert_threshold_id) {
                                ctx.wrong_results_inserted++;
                                auto rank_it = result_rank.find(r);
                                if (rank_it != result_rank.end()) {
                                    size_t rank = rank_it->second;
                                    if (rank < ctx.K && !missed_gt.empty()) {
                                        inserted_truly_closer++;
                                        ctx.inserted_truly_closer++;
                                    }
                                }
                            }
                        }
                    }

                    if (!missed_gt.empty()) {
                        ctx.queries_with_miss++;

                        // keep the first 20
                        if (ctx.miss_details.size() < 20) {
                            UnifiedContext::RecallMissDetail detail;
                            detail.query_idx = search_count;
                            detail.results = item.result_neighbors;
                            detail.groundtruth.assign(gt, gt + ctx.K);
                            detail.missed_gt = missed_gt;
                            detail.wrong_in_results = wrong_results;
                            detail.recall = recall;
                            ctx.miss_details.push_back(detail);
                        }
                    }
                }
                search_count++;
            }
        }
        // avg_recall: baseline queries only.
        if (ctx.baseline_queries_completed > 0) {
            ctx.avg_recall = ctx.baseline_recall_sum / ctx.baseline_queries_completed;
        } else {
            ctx.avg_recall = (search_count > 0) ? total_recall / search_count : 0;
        }
        ctx.search_qps = ctx.n_searches / (ctx.elapsed_ms / 1000.0);

        if (ctx.debug_recall && ctx.queries_with_miss > 0) {
            std::cout << "\nRecall misses:\n";
            std::cout << "Queries with miss: " << ctx.queries_with_miss << "/" << ctx.total_queries_analyzed << "\n";
            std::cout << "Total GT missed: " << ctx.total_gt_missed << "\n";
            std::cout << "Total wrong results: " << ctx.total_wrong_results;
            if (ctx.insert_threshold_id > 0) {
                std::cout << " (inserted nodes: " << ctx.wrong_results_inserted
                          << ", " << (ctx.total_wrong_results > 0 ?
                              100.0 * ctx.wrong_results_inserted / ctx.total_wrong_results : 0.0) << "%)";
            }
            std::cout << "\n";

            std::cout << "\nFirst " << std::min((size_t)5, ctx.miss_details.size()) << " misses:\n";
            for (size_t i = 0; i < std::min((size_t)5, ctx.miss_details.size()); i++) {
                auto& d = ctx.miss_details[i];
                std::cout << "Query " << d.query_idx << " (recall=" << (d.recall * 100.0) << "%):\n";
                std::cout << "  GT:      [";
                for (size_t j = 0; j < std::min((size_t)10, d.groundtruth.size()); j++) {
                    if (j > 0) std::cout << ", ";
                    std::cout << d.groundtruth[j];
                }
                std::cout << "]\n";
                std::cout << "  Results: [";
                for (size_t j = 0; j < std::min((size_t)10, d.results.size()); j++) {
                    if (j > 0) std::cout << ", ";
                    std::cout << d.results[j];
                    if (ctx.insert_threshold_id > 0 && d.results[j] >= ctx.insert_threshold_id) {
                        std::cout << "*";  // inserted node
                    }
                }
                std::cout << "]\n";
                std::cout << "  Missed:  [";
                for (size_t j = 0; j < d.missed_gt.size(); j++) {
                    if (j > 0) std::cout << ", ";
                    std::cout << d.missed_gt[j];
                }
                std::cout << "]\n";
                std::cout << "  Wrong:   [";
                for (size_t j = 0; j < d.wrong_in_results.size(); j++) {
                    if (j > 0) std::cout << ", ";
                    std::cout << d.wrong_in_results[j];
                    if (ctx.insert_threshold_id > 0 && d.wrong_in_results[j] >= ctx.insert_threshold_id) {
                        std::cout << "*";
                    }
                }
                std::cout << "]\n\n";
            }
        }
    }

    if (ctx.n_inserts > 0) {
        ctx.insert_throughput = ctx.n_inserts / (ctx.elapsed_ms / 1000.0);

        for (auto& item : ctx.workloads) {
            if (item.type == WorkloadType::INSERT) {
                ctx.total_search_phase_us += item.search_us;
                ctx.total_insert_rpc_us += item.insert_us;
                ctx.total_backlink_us += item.backlink_us;
                ctx.total_broadcast_us += item.broadcast_us;
            }
        }
    }

    if (ctx.search_started && ctx.searches_completed > 0) {
        ctx.search_wall_ms = duration_cast<microseconds>(
            ctx.search_last_end - ctx.search_first_start).count() / 1000.0;
    }
    if (ctx.insert_started && ctx.inserts_completed > 0) {
        ctx.insert_wall_ms = duration_cast<microseconds>(
            ctx.insert_last_end - ctx.insert_first_start).count() / 1000.0;
    }

    std::cout << "Workload done in " << ctx.elapsed_ms << " ms\n" << std::flush;
}

} // namespace tide
