// Server stats queries and the per-command metrics of a result.

#include "coordinator_types.h"
#include "state_protocol.h"
#include "protocol/broadcast_messages.h"
#include "metrics/statistics.h"
#include "stats_response_collector.h"

#include <chrono>
#include <map>
#include <algorithm>
#include <iostream>
#include <thread>

using namespace std::chrono;

namespace tide {

// Raw GET_STATS snapshots, through the server threads when they run.
bool query_server_stats(std::vector<ann::StatsResponseHeader>& snapshots,
                        bool consume_interval_max) {
    snapshots.assign(g_daemon.n_servers, ann::StatsResponseHeader{});
    if (g_daemon.n_servers == 0) return true;

    ann::StatsRequest stats_request{};
    stats_request.msg_type = ann::MessageType::GET_STATS_REQUEST;
    stats_request.consume_interval_max = consume_interval_max ? 1 : 0;
    const uint64_t request_id =
        g_daemon.next_stats_request_id.fetch_add(1, std::memory_order_relaxed);
    if (request_id == 0) {
        daemon_output_line("Error: GET_STATS request id overflow");
        return false;
    }
    stats_request.request_id = request_id;
    const auto* request_begin = reinterpret_cast<const char*>(&stats_request);
    const std::vector<char> request(request_begin,
                                    request_begin + sizeof(stats_request));
    if (g_daemon.main_response_queue && !g_daemon.server_queues.empty()) {
        for (size_t i = 0; i < g_daemon.n_servers; ++i) {
            if (!g_daemon.server_queues[i]) continue;
            ServerWorkItem item;
            item.request_data = request;
            item.target_server = static_cast<uint32_t>(i);
            g_daemon.server_queues[i]->push(std::move(item));
        }
    } else {
        for (size_t i = 0; i < g_daemon.n_servers; ++i) {
            zmq::message_t empty_frame(0);
            zmq::message_t request_frame(request.data(), request.size());
            g_daemon.sockets[i].send(empty_frame, zmq::send_flags::sndmore);
            g_daemon.sockets[i].send(request_frame, zmq::send_flags::none);
        }
    }

    StatsResponseCollector collector(g_daemon.n_servers, request_id);
    // Put back replies that belong to the drain or the controller.
    std::vector<ServerResponse> foreign;
    auto restore_foreign = [&]() {
        for (auto& response : foreign) {
            g_daemon.main_response_queue->push(std::move(response));
        }
        foreign.clear();
    };
    const auto deadline = steady_clock::now() + milliseconds(5000);
    while (!collector.complete() && steady_clock::now() < deadline) {
        if (g_daemon.main_response_queue && !g_daemon.server_queues.empty()) {
            ServerResponse response;
            if (g_daemon.main_response_queue->try_pop(response)) {
                if (response.response_data.empty() ||
                    ann::get_message_type(response.response_data.data(),
                                          response.response_data.size()) !=
                        ann::MessageType::GET_STATS_RESPONSE) {
                    foreign.push_back(std::move(response));
                    continue;
                }
                const auto disposition = collector.accept(
                    response.server_idx, response.response_data.data(),
                    response.response_data.size());
                if (disposition == StatsResponseDisposition::INVALID ||
                    disposition == StatsResponseDisposition::DUPLICATE) {
                    restore_foreign();
                    daemon_output_line("Error: invalid GET_STATS response");
                    return false;
                }
            } else {
                std::this_thread::sleep_for(milliseconds(1));
            }
            continue;
        }

        bool saw_response = false;
        for (size_t i = 0; i < g_daemon.n_servers; ++i) {
            if (collector.received(i)) continue;
            zmq::message_t empty_frame, response;
            if (!g_daemon.sockets[i].recv(empty_frame, zmq::recv_flags::dontwait)) continue;
            if (!g_daemon.sockets[i].recv(response, zmq::recv_flags::none)) continue;
            const auto disposition = collector.accept(
                i, static_cast<const char*>(response.data()), response.size());
            if (disposition == StatsResponseDisposition::INVALID ||
                disposition == StatsResponseDisposition::DUPLICATE) {
                daemon_output_line("Error: invalid direct GET_STATS response");
                return false;
            }
            saw_response = true;
        }
        if (!saw_response) std::this_thread::sleep_for(milliseconds(1));
    }
    restore_foreign();
    snapshots = collector.snapshots();
    return collector.complete();
}

void query_server_lock_metrics(ann::WorkloadResult& result) {
    if (g_daemon.n_servers == 0 || g_daemon.sockets.empty()) return;

    result.per_server_phase_lock_stats.resize(g_daemon.n_servers);
    result.resize_mutex_profile.per_server.resize(g_daemon.n_servers);

    auto copy_role = [](ann::WorkloadResult::LockRoleStats& dst,
                        const ann::LockRoleStatsHeader& src) {
        dst.acquisitions = src.acquisitions;
        dst.wait_us = src.wait_us;
        dst.hold_us = src.hold_us;
        dst.max_wait_us = src.max_wait_us;
        dst.max_hold_us = src.max_hold_us;
    };

    auto add_role = [](ann::WorkloadResult::LockRoleStats& dst,
                       const ann::LockRoleStatsHeader& src) {
        dst.acquisitions += src.acquisitions;
        dst.wait_us += src.wait_us;
        dst.hold_us += src.hold_us;
        if (src.max_wait_us > dst.max_wait_us) dst.max_wait_us = src.max_wait_us;
        if (src.max_hold_us > dst.max_hold_us) dst.max_hold_us = src.max_hold_us;
    };

    uint64_t total_insert_alloc_count = 0, total_insert_alloc_wait = 0;
    uint64_t total_insert_edge_count = 0, total_insert_edge_wait = 0;
    uint64_t total_backlink_count = 0, total_backlink_wait = 0;
    uint64_t total_broadcast_count = 0, total_broadcast_wait = 0;

    // does not consume the interval maxima of the next admin stats
    std::vector<ann::StatsResponseHeader> snapshots;
    const bool complete = query_server_stats(snapshots, false);
    const size_t responses = complete ? snapshots.size() : 0;
    if (!complete) {
        daemon_output_line("Warning: incomplete GET_STATS round");
    } else {
        for (size_t i = 0; i < snapshots.size(); ++i) {
            const auto* stats = &snapshots[i];

            auto& ps = result.per_server_phase_lock_stats[i];
            ps.insert_alloc_count = stats->insert_alloc_lock_count;
            ps.insert_alloc_wait_us = stats->insert_alloc_lock_count > 0 ?
                static_cast<double>(stats->insert_alloc_lock_wait_us) / stats->insert_alloc_lock_count : 0.0;
            ps.insert_edge_count = stats->insert_edge_lock_count;
            ps.insert_edge_wait_us = stats->insert_edge_lock_count > 0 ?
                static_cast<double>(stats->insert_edge_lock_wait_us) / stats->insert_edge_lock_count : 0.0;
            ps.backlink_count = stats->backlink_lock_count;
            ps.backlink_wait_us = stats->backlink_lock_count > 0 ?
                static_cast<double>(stats->backlink_lock_wait_us) / stats->backlink_lock_count : 0.0;
            ps.broadcast_count = stats->broadcast_lock_count;
            ps.broadcast_wait_us = stats->broadcast_lock_count > 0 ?
                static_cast<double>(stats->broadcast_lock_wait_us) / stats->broadcast_lock_count : 0.0;

            ps.backlink_edges_requested = stats->backlink_edges_requested;
            ps.backlink_edges_applied = stats->backlink_edges_applied;
            ps.backlink_skip_not_local = stats->backlink_skip_not_local;
            ps.backlink_skip_already_exists = stats->backlink_skip_already_exists;
            ps.backlink_skip_out_of_range = stats->backlink_skip_out_of_range;

            auto& rp = result.resize_mutex_profile.per_server[i];
            copy_role(rp.search_shared, stats->search_shared);
            copy_role(rp.insert_alloc_unique, stats->insert_alloc_unique);
            copy_role(rp.insert_alloc_shared, stats->insert_alloc_shared);
            copy_role(rp.insert_alloc_mutex, stats->insert_alloc_mutex);
            copy_role(rp.insert_prune_shared, stats->insert_prune_shared);
            copy_role(rp.insert_edge_shared, stats->insert_edge_shared);
            copy_role(rp.backlink_unique, stats->backlink_unique);
            copy_role(rp.broadcast_unique, stats->broadcast_unique);
            copy_role(rp.search_node_lock, stats->search_node_lock);

            if (result.resize_mutex_profile.search_sub_per_server.size() <= i) {
                result.resize_mutex_profile.search_sub_per_server.resize(i + 1);
            }
            auto& sps = result.resize_mutex_profile.search_sub_per_server[i];
            sps.distance_us = stats->search_distance_us;
            sps.beam_sort_us = stats->search_beam_sort_us;
            sps.distance_count = stats->search_distance_count;
            sps.neighbors_evaluated = stats->search_neighbors_evaluated;
            sps.request_count = stats->search_request_count;

            total_insert_alloc_count += stats->insert_alloc_lock_count;
            total_insert_alloc_wait += stats->insert_alloc_lock_wait_us;
            total_insert_edge_count += stats->insert_edge_lock_count;
            total_insert_edge_wait += stats->insert_edge_lock_wait_us;
            total_backlink_count += stats->backlink_lock_count;
            total_backlink_wait += stats->backlink_lock_wait_us;
            total_broadcast_count += stats->broadcast_lock_count;
            total_broadcast_wait += stats->broadcast_lock_wait_us;

            add_role(result.resize_mutex_profile.search_shared, stats->search_shared);
            add_role(result.resize_mutex_profile.insert_alloc_unique, stats->insert_alloc_unique);
            add_role(result.resize_mutex_profile.insert_alloc_shared, stats->insert_alloc_shared);
            add_role(result.resize_mutex_profile.insert_alloc_mutex, stats->insert_alloc_mutex);
            add_role(result.resize_mutex_profile.insert_prune_shared, stats->insert_prune_shared);
            add_role(result.resize_mutex_profile.insert_edge_shared, stats->insert_edge_shared);
            add_role(result.resize_mutex_profile.backlink_unique, stats->backlink_unique);
            add_role(result.resize_mutex_profile.broadcast_unique, stats->broadcast_unique);
            add_role(result.resize_mutex_profile.search_node_lock, stats->search_node_lock);

            result.resize_mutex_profile.search_sub.distance_us += stats->search_distance_us;
            result.resize_mutex_profile.search_sub.beam_sort_us += stats->search_beam_sort_us;
            result.resize_mutex_profile.search_sub.distance_count += stats->search_distance_count;
            result.resize_mutex_profile.search_sub.neighbors_evaluated += stats->search_neighbors_evaluated;
            result.resize_mutex_profile.search_sub.request_count += stats->search_request_count;
        }
    }

    result.insert_alloc_lock_count = total_insert_alloc_count;
    result.avg_insert_alloc_lock_wait_us = total_insert_alloc_count > 0 ?
        static_cast<double>(total_insert_alloc_wait) / total_insert_alloc_count : 0.0;
    result.insert_edge_lock_count = total_insert_edge_count;
    result.avg_insert_edge_lock_wait_us = total_insert_edge_count > 0 ?
        static_cast<double>(total_insert_edge_wait) / total_insert_edge_count : 0.0;
    result.backlink_lock_count = total_backlink_count;
    result.avg_backlink_lock_wait_us = total_backlink_count > 0 ?
        static_cast<double>(total_backlink_wait) / total_backlink_count : 0.0;
    result.broadcast_lock_count = total_broadcast_count;
    result.avg_broadcast_lock_wait_us = total_broadcast_count > 0 ?
        static_cast<double>(total_broadcast_wait) / total_broadcast_count : 0.0;

    daemon_output_line("[LOCK METRICS] Collected phase-specific metrics from " +
                       std::to_string(responses) + "/" + std::to_string(g_daemon.n_servers) + " servers");
}

void populate_detailed_metrics(const UnifiedContext& ctx, ann::WorkloadResult& result) {
    std::vector<double> query_latencies;
    std::vector<double> all_hop_rtts;
    std::vector<double> route_transport_residuals;

    uint64_t total_route_dist = 0, route_queries = 0;  // every server visit of every completed query
    uint64_t total_pq_dist = 0, total_full_dist = 0;
    uint64_t total_local_expanded = 0, total_top_W = 0;
    uint64_t total_neighbors_evaluated = 0;
    double total_server_processing = 0, total_queue_wait = 0;
    double total_network = 0, total_lock_wait = 0;
    uint64_t total_hops_count = 0;

    double total_zmq_recv = 0, total_zmq_send = 0;
    double total_deserialize = 0, total_serialize = 0;
    double total_queue_depth = 0, total_resp_queue_wait = 0;
    double total_io_processing = 0, total_worker_overhead = 0, total_send_prep = 0;

    double total_coord_serialize = 0, total_coord_send = 0;
    double total_coord_recv = 0, total_coord_total = 0;
    double total_coord_deserialize = 0, total_coord_response_queue = 0;

    std::map<uint32_t, std::pair<uint64_t, uint64_t>> per_hop_dist;  // hop -> (total_dist, count)
    uint64_t hop0_dist_total = 0, hop0_count = 0;
    uint64_t multihop_dist_total = 0, multihop_hops_total = 0;

    double total_pre_first_hop = 0;
    double total_inter_hop = 0;

    std::vector<double> insert_search_latencies;
    std::vector<double> insert_hop_rtts;
    double insert_total_server = 0, insert_total_queue = 0, insert_total_network = 0;
    double insert_total_resp_queue = 0, insert_total_coord_total = 0;
    double insert_total_lock_wait = 0;
    double insert_total_pre_first_hop = 0, insert_total_inter_hop = 0;
    uint64_t insert_total_hops = 0;

    for (const auto& item : ctx.workloads) {
        // insert workloads: their search phase
        if (item.type == WorkloadType::INSERT) {
            if (item.insert_phase == InsertPhase::SEARCH_PENDING) continue;

            {
                result.insert_search.count++;
                result.insert_search.total_hops += item.hops;
                result.insert_search.hop_distribution[item.hops]++;

                if (item.search_us > 0) {
                    insert_search_latencies.push_back(item.search_us);
                }

                insert_total_pre_first_hop += item.pre_first_hop_us;
                insert_total_inter_hop += item.inter_hop_us;

                if (item.hop_latencies.empty() && item.hops > 0) {
                    insert_total_hops += item.hops;
                    insert_total_server += item.cumulative_processing_us;
                    insert_total_queue += item.cumulative_queue_us;
                    double accounted = item.cumulative_processing_us + item.cumulative_queue_us;
                    insert_total_network += std::max(0.0, item.search_us - accounted);
                    if (item.hops > 0 && item.search_us > 0) {
                        double est_hop_rtt = item.search_us / item.hops;
                        for (uint32_t h = 0; h < item.hops; h++) {
                            insert_hop_rtts.push_back(est_hop_rtt);
                        }
                    }
                } else {
                    for (size_t h = 0; h < item.hop_latencies.size(); h++) {
                        const auto& hop = item.hop_latencies[h];
                        double hop_server_time = hop.measured_server_total_us() - hop.response_queue_wait_us;
                        insert_total_server += hop_server_time;
                        insert_total_queue += hop.queue_wait_us;
                        insert_total_resp_queue += hop.response_queue_wait_us;
                        insert_total_network += hop.pure_network_us();
                        insert_total_lock_wait += hop.lock_wait_us;
                        insert_total_coord_total += hop.measured_coord_total_us();
                        insert_total_hops++;

                        if (hop.total_us > 0) {
                            insert_hop_rtts.push_back(hop.total_us);
                        }
                    }
                }
            }
            continue;
        }

        // search workloads
        if (item.type != WorkloadType::SEARCH) continue;
        if (!item.done) continue;

        // Each reply carries the distance computations since the coordinator sent the state: one visit
        // without P2P, every visit up to that reply with P2P.
        total_route_dist += item.total_distance_computations;
        route_queries++;

        if (item.query_latency_us > 0) {
            query_latencies.push_back(item.query_latency_us);
        }

        total_pre_first_hop += item.pre_first_hop_us;
        total_inter_hop += item.inter_hop_us;

        result.hop_distribution[item.hops]++;

        if (!item.route.empty()) {
            std::string route_str;
            for (size_t r = 0; r < item.route.size(); r++) {
                if (r > 0) route_str += "→";
                route_str += std::to_string(item.route[r].partition);
            }
            result.route_summary[route_str]++;
            result.route_traced_count++;

            double route_processing_us = 0;
            for (size_t r = 0; r < item.route.size(); r++) {
                double processing_us = static_cast<double>(item.route[r].depart_us) -
                                       static_cast<double>(item.route[r].arrive_us);
                route_processing_us += processing_us;
                result.route_total_processing_us += processing_us;
                result.route_processing_segments++;
            }

            double route_queue_us = 0;
            if (item.cumulative_queue_us > 0) {
                route_queue_us = item.cumulative_queue_us;
            } else {
                for (const auto& hop : item.hop_latencies) {
                    route_queue_us += hop.queue_wait_us;
                }
            }
            result.route_total_queue_us += route_queue_us;

            double path_rtt_us = 0;
            double coord_overhead_us = 0;
            for (const auto& hop : item.hop_latencies) {
                path_rtt_us += hop.total_us;
                coord_overhead_us += hop.coord_serialize_us + hop.coord_send_us +
                    hop.coord_response_queue_us + hop.coord_recv_us + hop.coord_deserialize_us;
            }
            if (path_rtt_us > 0) {
                double residual_us = path_rtt_us - route_processing_us - route_queue_us - coord_overhead_us;
                result.route_transport_residual_total_us += residual_us;
                result.route_transport_residual_count++;
                if (residual_us < 0) result.route_transport_residual_negative_count++;
                route_transport_residuals.push_back(residual_us);
            }
        }

        // One record per reply the coordinator received: every visit without P2P, the final reply with P2P.
        double query_server_total = 0, query_queue_total = 0, query_network_total = 0;

        for (size_t h = 0; h < item.hop_latencies.size(); h++) {
            const auto& hop = item.hop_latencies[h];

            if (hop.total_us > 0) {
                all_hop_rtts.push_back(hop.total_us);
            }

            total_local_expanded += hop.local_nodes_expanded;
            total_top_W += hop.top_W_count;
            total_neighbors_evaluated += hop.neighbors_evaluated;

            // the kinds per visit; a P2P reply has them for its own visit only
            if (!ctx.p2p_mode) {
                total_pq_dist += hop.pq_distances;
                total_full_dist += hop.full_distances;
                per_hop_dist[h].first += hop.pq_distances + hop.full_distances;
                per_hop_dist[h].second++;
            }

            // server time excludes the response queue wait
            double hop_server_time = hop.measured_server_total_us() - hop.response_queue_wait_us;
            query_server_total += hop_server_time;
            query_queue_total += hop.queue_wait_us;
            query_network_total += hop.pure_network_us();

            total_server_processing += hop_server_time;
            total_queue_wait += hop.queue_wait_us;
            total_network += hop.pure_network_us();
            total_lock_wait += hop.lock_wait_us;

            total_zmq_recv += hop.zmq_recv_us;
            total_zmq_send += hop.zmq_send_us;
            total_deserialize += hop.deserialize_us;
            total_serialize += hop.serialize_us;
            total_queue_depth += hop.queue_depth;
            total_resp_queue_wait += hop.response_queue_wait_us;
            total_io_processing += hop.io_processing_us;
            total_worker_overhead += hop.worker_overhead_us;
            total_send_prep += hop.send_prep_us;

            total_coord_serialize += hop.coord_serialize_us;
            total_coord_send += hop.coord_send_us;
            total_coord_response_queue += hop.coord_response_queue_us;
            total_coord_recv += hop.coord_recv_us;
            total_coord_deserialize += hop.coord_deserialize_us;
            total_coord_total += hop.measured_coord_total_us();

            total_hops_count++;
        }

        if (!ctx.p2p_mode && !item.hop_latencies.empty()) {
            const auto& first_hop = item.hop_latencies[0];
            hop0_dist_total += first_hop.pq_distances + first_hop.full_distances;
            hop0_count++;

            if (item.hop_latencies.size() > 1) {
                for (size_t h = 1; h < item.hop_latencies.size(); h++) {
                    multihop_dist_total += item.hop_latencies[h].pq_distances + item.hop_latencies[h].full_distances;
                    multihop_hops_total++;
                }
            }
        }
    }

    uint32_t search_count = query_latencies.size();

    if (!route_transport_residuals.empty()) {
        std::sort(route_transport_residuals.begin(), route_transport_residuals.end());
        double sum = 0;
        for (double residual : route_transport_residuals) sum += residual;
        result.route_transport_residual_avg_per_query_us = sum / route_transport_residuals.size();
        result.route_transport_residual_min_us = route_transport_residuals.front();
        result.route_transport_residual_max_us = route_transport_residuals.back();
        result.route_transport_residual_p50_us =
            route_transport_residuals[route_transport_residuals.size() / 2];
        result.route_transport_residual_p95_us = route_transport_residuals[
            ann::nearest_rank_percentile_index(route_transport_residuals.size(), 95)];
        result.route_transport_residual_p99_us =
            route_transport_residuals[static_cast<size_t>(route_transport_residuals.size() * 0.99)];
        if (result.route_processing_segments > 0) {
            result.route_transport_residual_avg_per_hop_us =
                result.route_transport_residual_total_us / result.route_processing_segments;
        }
    }

    if (route_queries > 0) result.avg_dist_comp = static_cast<double>(total_route_dist) / route_queries;
    if (search_count == 0) goto insert_search_metrics;
    std::sort(query_latencies.begin(), query_latencies.end());
    result.min_query_latency_ms = query_latencies.front() / 1000.0;
    result.max_query_latency_ms = query_latencies.back() / 1000.0;
    result.p50_query_latency_ms = query_latencies[query_latencies.size() / 2] / 1000.0;
    result.p95_query_latency_ms = query_latencies[
        ann::nearest_rank_percentile_index(query_latencies.size(), 95)] / 1000.0;
    result.p99_query_latency_ms = query_latencies[static_cast<size_t>(query_latencies.size() * 0.99)] / 1000.0;

    result.avg_server_per_query_ms = (total_server_processing / search_count) / 1000.0;
    result.avg_worker_queue_per_query_ms = (total_queue_wait / search_count) / 1000.0;
    result.avg_response_queue_per_query_ms = (total_resp_queue_wait / search_count) / 1000.0;
    result.avg_network_per_query_ms = (total_network / search_count) / 1000.0;
    result.avg_lock_per_query_ms = (total_lock_wait / search_count) / 1000.0;
    result.avg_coord_total_per_query_ms = (total_coord_total / search_count) / 1000.0;

    result.avg_pre_first_hop_per_query_ms = (total_pre_first_hop / search_count) / 1000.0;
    result.avg_inter_hop_per_query_ms = (total_inter_hop / search_count) / 1000.0;

    if (total_hops_count > 0) {
        result.avg_server_per_hop_ms = (total_server_processing / total_hops_count) / 1000.0;
        result.avg_worker_queue_per_hop_ms = (total_queue_wait / total_hops_count) / 1000.0;
        result.avg_response_queue_per_hop_ms = (total_resp_queue_wait / total_hops_count) / 1000.0;
        result.avg_network_per_hop_ms = (total_network / total_hops_count) / 1000.0;
        result.avg_lock_per_hop_ms = (total_lock_wait / total_hops_count) / 1000.0;
        result.avg_coord_total_per_hop_ms = (total_coord_total / total_hops_count) / 1000.0;
    }

    if (!all_hop_rtts.empty()) {
        std::sort(all_hop_rtts.begin(), all_hop_rtts.end());
        result.min_hop_rtt_ms = all_hop_rtts.front() / 1000.0;
        result.max_hop_rtt_ms = all_hop_rtts.back() / 1000.0;
        result.p50_hop_rtt_ms = all_hop_rtts[all_hop_rtts.size() / 2] / 1000.0;
        result.p95_hop_rtt_ms = all_hop_rtts[
            ann::nearest_rank_percentile_index(all_hop_rtts.size(), 95)] / 1000.0;
        result.p99_hop_rtt_ms = all_hop_rtts[static_cast<size_t>(all_hop_rtts.size() * 0.99)] / 1000.0;
        result.avg_hop_rtt_ms = 0;
        for (double rtt : all_hop_rtts) result.avg_hop_rtt_ms += rtt;
        result.avg_hop_rtt_ms = (result.avg_hop_rtt_ms / all_hop_rtts.size()) / 1000.0;
    }

    if (total_hops_count > 0) {
        result.avg_zmq_recv_us = total_zmq_recv / total_hops_count;
        result.avg_zmq_send_us = total_zmq_send / total_hops_count;
        result.avg_deserialize_us = total_deserialize / total_hops_count;
        result.avg_serialize_us = total_serialize / total_hops_count;
        result.avg_queue_depth = total_queue_depth / total_hops_count;
        result.avg_resp_queue_wait_us = total_resp_queue_wait / total_hops_count;
        result.avg_io_processing_us = total_io_processing / total_hops_count;
        result.avg_worker_overhead_us = total_worker_overhead / total_hops_count;
        result.avg_send_prep_us = total_send_prep / total_hops_count;
    }

    if (total_hops_count > 0) {
        result.coord_serialize_us = total_coord_serialize / total_hops_count;
        result.coord_send_us = total_coord_send / total_hops_count;
        result.coord_response_queue_us = total_coord_response_queue / total_hops_count;
        result.coord_recv_us = total_coord_recv / total_hops_count;
        result.coord_deserialize_us = total_coord_deserialize / total_hops_count;
    }

    // Server threads record send/recv per hop. Use those sums when ctx.timing is empty.
    result.timing_serialize_us = static_cast<uint64_t>(
        ctx.timing.serialize_us > 0 ? ctx.timing.serialize_us : total_coord_serialize);
    result.timing_send_us = static_cast<uint64_t>(
        ctx.timing.send_us > 0 ? ctx.timing.send_us : total_coord_send);
    result.timing_response_queue_us = static_cast<uint64_t>(total_coord_response_queue);
    result.timing_poll_wait_us = static_cast<uint64_t>(ctx.timing.poll_wait_us);
    result.timing_recv_us = static_cast<uint64_t>(
        ctx.timing.recv_us > 0 ? ctx.timing.recv_us : total_coord_recv);
    result.timing_deserialize_us = static_cast<uint64_t>(
        ctx.timing.deserialize_us > 0 ? ctx.timing.deserialize_us : total_coord_deserialize);
    result.timing_process_us = static_cast<uint64_t>(ctx.timing.process_us);
    result.timing_send_count = ctx.timing.send_count > 0 ? ctx.timing.send_count : total_hops_count;
    result.timing_poll_count = ctx.timing.poll_count;
    result.timing_recv_count = ctx.timing.recv_count > 0 ? ctx.timing.recv_count : total_hops_count;
    result.total_hops = total_hops_count;

    result.avg_pq_dist = static_cast<double>(total_pq_dist) / search_count;
    result.avg_full_dist = static_cast<double>(total_full_dist) / search_count;

    result.local_expanded = total_local_expanded;
    result.top_W = total_top_W;
    result.local_ratio = (total_top_W > 0) ? (100.0 * total_local_expanded / total_top_W) : 0;
    result.avg_neighbors_evaluated = static_cast<double>(total_neighbors_evaluated) / search_count;

    for (const auto& [hop, data] : per_hop_dist) {
        if (data.second > 0) {
            result.per_hop_dist_avg[hop] = static_cast<double>(data.first) / data.second;
        }
    }
    result.hop0_query_count = hop0_count;
    result.avg_hop0_dist = (hop0_count > 0) ? static_cast<double>(hop0_dist_total) / hop0_count : 0;
    result.multihop_query_count = (multihop_hops_total > 0) ? search_count - hop0_count : 0;
    result.multihop_total_hops = multihop_hops_total;
    result.avg_multihop_dist_per_hop = (multihop_hops_total > 0) ?
        static_cast<double>(multihop_dist_total) / multihop_hops_total : 0;

    result.requests_per_server = ctx.requests_per_server;

    if (total_hops_count > 0 && result.avg_hop_rtt_ms > 0) {
        // excludes the wait inside zmq::poll()
        double coord_overhead = result.coord_serialize_us + result.coord_send_us +
            result.coord_response_queue_us + result.coord_recv_us + result.coord_deserialize_us;
        result.overhead_per_hop_us = coord_overhead;
        result.overhead_ratio = coord_overhead / (result.avg_hop_rtt_ms * 1000.0);
    }

    insert_search_metrics:
    if (result.insert_search.count > 0) {
        result.insert_search.avg_hops = static_cast<double>(result.insert_search.total_hops) /
                                        result.insert_search.count;

        if (insert_total_hops > 0) {
            result.insert_search.avg_server_per_hop_ms = (insert_total_server / insert_total_hops) / 1000.0;
            result.insert_search.avg_worker_queue_per_hop_ms = (insert_total_queue / insert_total_hops) / 1000.0;
            result.insert_search.avg_response_queue_per_hop_ms = (insert_total_resp_queue / insert_total_hops) / 1000.0;
            result.insert_search.avg_network_per_hop_ms = (insert_total_network / insert_total_hops) / 1000.0;
            result.insert_search.avg_lock_per_hop_ms = (insert_total_lock_wait / insert_total_hops) / 1000.0;
            result.insert_search.avg_coord_total_per_hop_ms = (insert_total_coord_total / insert_total_hops) / 1000.0;
        }

        result.insert_search.avg_server_per_query_ms = (insert_total_server / result.insert_search.count) / 1000.0;
        result.insert_search.avg_worker_queue_per_query_ms = (insert_total_queue / result.insert_search.count) / 1000.0;
        result.insert_search.avg_response_queue_per_query_ms = (insert_total_resp_queue / result.insert_search.count) / 1000.0;
        result.insert_search.avg_network_per_query_ms = (insert_total_network / result.insert_search.count) / 1000.0;
        result.insert_search.avg_lock_per_query_ms = (insert_total_lock_wait / result.insert_search.count) / 1000.0;
        result.insert_search.avg_coord_total_per_query_ms = (insert_total_coord_total / result.insert_search.count) / 1000.0;

        result.insert_search.avg_pre_first_hop_per_query_ms = (insert_total_pre_first_hop / result.insert_search.count) / 1000.0;
        result.insert_search.avg_inter_hop_per_query_ms = (insert_total_inter_hop / result.insert_search.count) / 1000.0;

        if (!insert_search_latencies.empty()) {
            std::sort(insert_search_latencies.begin(), insert_search_latencies.end());
            result.insert_search.min_latency_ms = insert_search_latencies.front() / 1000.0;
            result.insert_search.max_latency_ms = insert_search_latencies.back() / 1000.0;
            result.insert_search.p50_latency_ms = insert_search_latencies[insert_search_latencies.size() / 2] / 1000.0;
            result.insert_search.p95_latency_ms = insert_search_latencies[
                ann::nearest_rank_percentile_index(insert_search_latencies.size(), 95)] /
                1000.0;
            result.insert_search.p99_latency_ms = insert_search_latencies[
                static_cast<size_t>(insert_search_latencies.size() * 0.99)] / 1000.0;

            double sum = 0;
            for (double lat : insert_search_latencies) sum += lat;
            result.insert_search.avg_latency_ms = (sum / insert_search_latencies.size()) / 1000.0;
        }

        if (!insert_hop_rtts.empty()) {
            std::sort(insert_hop_rtts.begin(), insert_hop_rtts.end());
            result.insert_search.min_hop_rtt_ms = insert_hop_rtts.front() / 1000.0;
            result.insert_search.max_hop_rtt_ms = insert_hop_rtts.back() / 1000.0;
            result.insert_search.p50_hop_rtt_ms = insert_hop_rtts[insert_hop_rtts.size() / 2] / 1000.0;
            result.insert_search.p95_hop_rtt_ms = insert_hop_rtts[
                ann::nearest_rank_percentile_index(insert_hop_rtts.size(), 95)] / 1000.0;
            result.insert_search.p99_hop_rtt_ms = insert_hop_rtts[
                static_cast<size_t>(insert_hop_rtts.size() * 0.99)] / 1000.0;

            double sum = 0;
            for (double rtt : insert_hop_rtts) sum += rtt;
            result.insert_search.avg_hop_rtt_ms = (sum / insert_hop_rtts.size()) / 1000.0;
        }
    }

    if (ctx.n_inserts > 0) {
        result.backlink.count = ctx.total_backlinks_sent;
        result.backlink.avg_per_insert = static_cast<double>(ctx.total_backlinks_sent) / ctx.n_inserts;
        result.backlink.wall_clock_ms = ctx.total_backlink_us / 1000.0;
        result.backlink.total_lock_wait_ms = (result.avg_backlink_lock_wait_us * result.backlink_lock_count) / 1000.0;

        result.broadcast.count = ctx.total_broadcasts_acked;
        result.broadcast.wall_clock_ms = ctx.total_broadcast_us / 1000.0;
        result.broadcast.total_lock_wait_ms = (result.avg_broadcast_lock_wait_us * result.broadcast_lock_count) / 1000.0;
    }

    if (ctx.n_inserts > 0) {
        result.maintenance_backlink_pre_drain_edges = ctx.total_backlinks_sent;
        result.maintenance_backlink_combined_edges =
            result.maintenance_backlink_pre_drain_edges + result.maintenance_backlink_drain_exact_edges;
        result.maintenance_backlink_combined_avg_per_insert =
            static_cast<double>(result.maintenance_backlink_combined_edges) / ctx.n_inserts;
        result.maintenance_backlink_metric_complete = true;
        result.maintenance_broadcast_pre_drain_acks = ctx.total_broadcasts_acked;
        result.maintenance_broadcast_combined_acks =
            result.maintenance_broadcast_pre_drain_acks + result.maintenance_broadcast_drain_acks;
        result.maintenance_broadcast_metric_complete = true;
    }

    result.pending_mgr_size = ctx.pending_mgr ? ctx.pending_mgr->size() : 0;
    result.merge_search_calls = ctx.merge_search_calls;
    result.merge_search_pending_total = ctx.merge_search_pending_total;
    result.merge_search_pending_eligible = ctx.merge_search_pending_eligible;
    result.merge_search_pending_topk = ctx.merge_search_pending_topk;
    result.merge_search_result_changed = ctx.merge_search_result_changed;
    result.merge_search_wall_ms = ctx.merge_search_wall_us / 1000.0;
    result.merge_search_avg_us = ctx.merge_search_calls > 0 ?
        static_cast<double>(ctx.merge_search_wall_us) / ctx.merge_search_calls : 0.0;
}

} // namespace tide
