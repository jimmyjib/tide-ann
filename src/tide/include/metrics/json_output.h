#pragma once
/**
 * JSON Output Utilities
 *
 * Functions for converting ExperimentResult to JSON and outputting results.
 */

#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include "metrics/experiment_result.h"
#include "metrics/query_metrics.h"
#include "metrics/statistics.h"

namespace ann {

/**
 * Convert ExperimentResult to JSON string.
 */
inline std::string serving_accounting_json(const ServingAccounting& a) {
    const auto error = a.error();
    const bool valid = error.empty();
    std::ostringstream ss;
    ss << std::setprecision(17)
       << "{\"schema\":\"tide-serving-accounting-v2\",\"clock_scope\":\"coordinator-steady-ns\""
       << ",\"status\":\"" << (valid ? "VALID" : "INVALID") << "\",\"error\":\"" << error << "\""
       << ",\"begin_scope\":\"after-GT-load-before-query-and-context-preparation\""
       << ",\"end_scope\":\"last-owner-search-completion\""
       << ",\"owner_search_begin_scope\":\"first-owner-query-init-before-serialization-and-enqueue-not-transport-dispatch\""
       << ",\"included\":[\"query-context-preparation\",\"workload-materialization\",\"parallel-head-BF-wall-once\","
          "\"native-search-and-controller-work\",\"rolling-Recall\",\"in-envelope-instrumentation\"]"
       << ",\"outside_envelope\":[\"GT-load\",\"previous-resident-payload-load\",\"external-client-and-controller-startup\","
          "\"post-completion-controller-stop\",\"metric-thread-join\",\"final-Recall-digest-report\"]"
       << ",\"declared_queries\":" << a.declared_queries << ",\"completed_queries\":" << a.completed_queries
       << ",\"begin_ns\":" << a.begin_ns << ",\"end_ns\":" << a.end_ns
       << ",\"owner_search_begin_ns\":" << a.owner_search_begin_ns;
    auto span = [&](const char* name, const ServingAccounting::Span& s) {
        ss << ",\"" << name << "\":{\"begin_ns\":" << s.begin_ns << ",\"end_ns\":" << s.end_ns << '}';
    };
    span("materialize", a.materialize);
    span("head_bf", a.head_bf);
    if (valid) {
        const auto raw = a.raw_ns();
        ss << ",\"raw_elapsed_ns\":" << raw << ",\"raw_envelope_qps\":" << a.completed_queries * 1e9 / raw;
    } else {
        ss << ",\"raw_elapsed_ns\":null,\"raw_envelope_qps\":null";
    }
    return ss.str() + '}';
}

inline std::string to_json(const ExperimentResult& r) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);

    ss << "{\n";
    if (r.serving_accounting.enabled)
        ss << "  \"serving_accounting\": " << serving_accounting_json(r.serving_accounting) << ",\n";
    ss << "  \"timestamp\": \"" << r.timestamp << "\",\n";
    if (!r.description.empty()) {
        ss << "  \"description\": \"" << r.description << "\",\n";
    }
    ss << "  \"dataset\": \"" << r.dataset << "\",\n";
    ss << "  \"groundtruth_file\": \"" << r.groundtruth_file << "\",\n";
    ss << "  \"mode\": \"" << r.mode << "\",\n";
    ss << "  \"p2p_mode\": " << (r.p2p_mode ? "true" : "false") << ",\n";
    ss << "  \"observability\": {\n";
    ss << "    \"diagnostic_mode\": " << (r.diagnostic_mode ? "true" : "false") << "\n";
    ss << "  },\n";

    ss << "  \"parameters\": {\n";
    ss << "    \"queries\": " << r.queries << ",\n";
    ss << "    \"k\": " << r.k << ",\n";
    ss << "    \"L\": " << r.L << ",\n";
    ss << "    \"batch_size\": " << r.batch_size << "\n";
    ss << "  },\n";

    ss << "  \"search\": {\n";
    ss << "    \"count\": " << r.search_count << ",\n";
    ss << "    \"recall\": " << r.recall << ",\n";
    ss << "    \"qps\": " << r.qps << ",\n";
    ss << "    \"avg_hops\": " << r.avg_hops << ",\n";
    ss << "    \"avg_dist_comp\": " << r.avg_dist_comp << ",\n";
    ss << "    \"total_time_ms\": " << r.total_time_ms;
    if (!r.search_result_digest.empty()) {
        ss << ",\n";
        ss << "    \"result_digest\": {\n";
        ss << "      \"algorithm\": \"" << r.search_result_digest_algorithm << "\",\n";
        ss << "      \"value\": \"" << r.search_result_digest << "\",\n";
        ss << "      \"query_count\": " << r.search_result_digest_queries << ",\n";
        ss << "      \"ranked_neighbors_hashed\": "
           << r.search_result_digest_neighbors << ",\n";
        ss << "      \"computed_post_workload\": true\n";
        ss << "    }";
    }
    ss << "\n";
    ss << "  },\n";

    if (r.self_query) {
        ss << "  \"self_query\": {\"queries\": " << r.self_query_count
           << ", \"found_at_zero\": " << r.self_query_found_at_zero
           << ", \"found_nonzero\": " << r.self_query_found_nonzero
           << ", \"not_found\": " << r.self_query_not_found
           << ", \"top1_at_zero\": " << r.self_query_top1_at_zero
           << ", \"failures_truncated\": " << (r.self_query_failures_truncated ? "true" : "false")
           << ", \"failures\": [";
        const auto precision = ss.precision(9);
        for (size_t i = 0; i < r.self_query_failures.size(); ++i) {
            const auto& f = r.self_query_failures[i];
            ss << (i ? ", " : "") << "[" << f.id << ", " << f.rank << ", " << f.distance << "]";
        }
        ss.precision(precision);
        ss << "]},\n";
    }

    // Search/Insert wall clock separation (only if available)
    if (r.search_wall_ms > 0 || r.insert_wall_ms > 0 || r.total_with_metrics_ms > 0) {
        ss << "  \"wall_clock\": {\n";
        ss << "    \"search_ms\": " << r.search_wall_ms << ",\n";
        ss << "    \"insert_ms\": " << r.insert_wall_ms << ",\n";
        ss << "    \"search_only_qps\": " << r.search_only_qps << ",\n";
        ss << "    \"insert_only_tps\": " << r.insert_only_tps;
        if (r.total_with_metrics_ms > 0) {
            ss << ",\n";
            ss << "    \"total_with_metrics_ms\": " << r.total_with_metrics_ms << ",\n";
            ss << "    \"metric_thread_join_ms\": " << r.metric_thread_join_ms << ",\n";
            ss << "    \"search_with_metrics_qps\": " << r.search_with_metrics_qps;
        }
        ss << "\n";
        ss << "  },\n";
    }

    ss << "  \"hop_distribution\": {";
    bool first = true;
    for (const auto& [hops, count] : r.hop_distribution) {
        if (!first) ss << ", ";
        ss << "\"" << hops << "\": " << count;
        first = false;
    }
    ss << "},\n";

    // Route traces (path analysis with same-clock transport residual)
    if (r.route_traced_count > 0) {
        ss << "  \"route_traces\": {\n";
        ss << "    \"traced_queries\": " << r.route_traced_count << ",\n";
        ss << "    \"unique_routes\": " << r.route_summary.size() << ",\n";
        if (r.route_processing_segments > 0) {
            double avg_proc_us = r.route_total_processing_us / r.route_processing_segments;
            double avg_queue_us = r.route_total_queue_us / r.route_processing_segments;
            ss << "    \"avg_server_processing_per_hop_us\": " << std::fixed << std::setprecision(1) << avg_proc_us << std::setprecision(4) << ",\n";
            ss << "    \"avg_server_queue_per_hop_us\": " << std::fixed << std::setprecision(1) << avg_queue_us << std::setprecision(4) << ",\n";
        }
        if (r.route_transport_residual_count > 0) {
            ss << "    \"path_transport_residual\": {\n";
            ss << "      \"avg_per_query_us\": " << r.route_transport_residual_avg_per_query_us << ",\n";
            ss << "      \"avg_per_hop_us\": " << r.route_transport_residual_avg_per_hop_us << ",\n";
            ss << "      \"min_us\": " << r.route_transport_residual_min_us << ",\n";
            ss << "      \"p50_us\": " << r.route_transport_residual_p50_us << ",\n";
            ss << "      \"p95_us\": " << r.route_transport_residual_p95_us << ",\n";
            ss << "      \"p99_us\": " << r.route_transport_residual_p99_us << ",\n";
            ss << "      \"max_us\": " << r.route_transport_residual_max_us << ",\n";
            ss << "      \"count\": " << r.route_transport_residual_count << ",\n";
            ss << "      \"negative_count\": " << r.route_transport_residual_negative_count << "\n";
            ss << "    },\n";
        }
        ss << "    \"routes\": {";
        bool rfirst = true;
        for (const auto& [route, cnt] : r.route_summary) {
            if (!rfirst) ss << ", ";
            ss << "\"" << route << "\": " << cnt;
            rfirst = false;
        }
        ss << "}\n";
        ss << "  },\n";
    }

    // The latency block: end-to-end latency of the search workloads (of the insert workloads when no search
    // ran) and the per-hop service split. The query_latency and per_* blocks below report the search timings.
    bool latency_uses_insert = (r.search_count == 0 && r.insert_count > 0);
    double latency_avg_ms = latency_uses_insert
        ? (r.per_insert_search_ms + r.per_insert_rpc_ms)
        : r.avg_query_latency_ms;
    double latency_min_ms = latency_uses_insert
        ? (r.insert_search.min_latency_ms + r.per_insert_rpc_ms)
        : r.min_query_latency_ms;
    double latency_p50_ms = latency_uses_insert
        ? (r.insert_search.p50_latency_ms + r.per_insert_rpc_ms)
        : r.p50_query_latency_ms;
    double latency_p95_ms = latency_uses_insert
        ? (r.insert_search.p95_latency_ms + r.per_insert_rpc_ms)
        : r.p95_query_latency_ms;
    double latency_p99_ms = latency_uses_insert
        ? (r.insert_search.p99_latency_ms + r.per_insert_rpc_ms)
        : r.p99_query_latency_ms;
    double latency_max_ms = latency_uses_insert
        ? (r.insert_search.max_latency_ms + r.per_insert_rpc_ms)
        : r.max_query_latency_ms;
    double service_hop_rtt_ms = latency_uses_insert ? r.insert_search.avg_hop_rtt_ms : r.avg_hop_rtt_ms;
    double service_server_ms = latency_uses_insert ? r.insert_search.avg_server_per_hop_ms : r.avg_server_per_hop_ms;
    double service_worker_queue_ms = latency_uses_insert ? r.insert_search.avg_worker_queue_per_hop_ms : r.avg_worker_queue_per_hop_ms;
    double service_response_queue_ms = latency_uses_insert ? r.insert_search.avg_response_queue_per_hop_ms : r.avg_response_queue_per_hop_ms;
    double service_transport_ms = latency_uses_insert ? r.insert_search.avg_network_per_hop_ms : r.avg_network_per_hop_ms;
    double service_coord_ms = latency_uses_insert ? r.insert_search.avg_coord_total_per_hop_ms : r.avg_coord_total_per_hop_ms;

    ss << "  \"latency\": {\n";
    ss << "    \"end_to_end\": {\n";
    ss << "      \"scope\": \"" << (latency_uses_insert ? "insert_workloads" : "search_workloads") << "\",\n";
    ss << "      \"avg_ms\": " << latency_avg_ms << ",\n";
    ss << "      \"min_ms\": " << latency_min_ms << ",\n";
    ss << "      \"p50_ms\": " << latency_p50_ms << ",\n";
    ss << "      \"p95_ms\": " << latency_p95_ms << ",\n";
    ss << "      \"p99_ms\": " << latency_p99_ms << ",\n";
    ss << "      \"max_ms\": " << latency_max_ms << ",\n";
    ss << "      \"includes_scheduler_backlog\": true,\n";
    ss << "      \"percentiles_source\": \""
       << (latency_uses_insert ? "insert_search_percentiles_plus_avg_insert_rpc" : "search_query_latency")
       << "\"\n";
    ss << "    },\n";
    ss << "    \"service\": {\n";
    ss << "      \"scope\": \"" << (latency_uses_insert ? "insert_search_hops" : "search_hops") << "\",\n";
    ss << "      \"hop_rtt_avg_ms\": " << service_hop_rtt_ms << ",\n";
    ss << "      \"server_per_hop_ms\": " << service_server_ms << ",\n";
    ss << "      \"worker_queue_per_hop_ms\": " << service_worker_queue_ms << ",\n";
    ss << "      \"response_queue_per_hop_ms\": " << service_response_queue_ms << ",\n";
    ss << "      \"transport_residual_per_hop_ms\": " << service_transport_ms << ",\n";
    ss << "      \"coord_total_per_hop_ms\": " << service_coord_ms << "\n";
    ss << "    }\n";
    ss << "  },\n";

    // Query latency stats
    ss << "  \"query_latency\": {\n";
    ss << "    \"avg_ms\": " << r.avg_query_latency_ms << ",\n";
    ss << "    \"min_ms\": " << r.min_query_latency_ms << ",\n";
    ss << "    \"p50_ms\": " << r.p50_query_latency_ms << ",\n";
    ss << "    \"p95_ms\": " << r.p95_query_latency_ms << ",\n";
    ss << "    \"p99_ms\": " << r.p99_query_latency_ms << ",\n";
    ss << "    \"max_ms\": " << r.max_query_latency_ms << "\n";
    ss << "  },\n";

    // Per-query breakdown
    ss << "  \"per_query_breakdown\": {\n";
    ss << "    \"server_ms\": { \"total\": " << r.avg_server_per_query_ms
       << ", \"lock_wait_ms\": " << r.avg_lock_per_query_ms << " },\n";
    ss << "    \"worker_queue_ms\": " << r.avg_worker_queue_per_query_ms << ",\n";
    ss << "    \"response_queue_ms\": " << r.avg_response_queue_per_query_ms << ",\n";
    ss << "    \"network_ms\": " << r.avg_network_per_query_ms << ",\n";
    ss << "    \"coord_total_ms\": " << r.avg_coord_total_per_query_ms << ",\n";
    ss << "    \"coord_total_ms_available\": " << (r.p2p_mode ? "false" : "true") << "\n";
    ss << "  },\n";

    // Per-hop breakdown
    ss << "  \"per_hop_breakdown\": {\n";
    ss << "    \"avg_latency_ms\": " << r.avg_hop_rtt_ms << ",\n";
    ss << "    \"server_ms\": { \"total\": " << r.avg_server_per_hop_ms
       << ", \"lock_wait_ms\": " << r.avg_lock_per_hop_ms << " },\n";
    ss << "    \"worker_queue_ms\": " << r.avg_worker_queue_per_hop_ms << ",\n";
    ss << "    \"response_queue_ms\": " << r.avg_response_queue_per_hop_ms << ",\n";
    ss << "    \"network_ms\": " << r.avg_network_per_hop_ms << ",\n";
    ss << "    \"coord_total_ms\": " << r.avg_coord_total_per_hop_ms << ",\n";
    ss << "    \"coord_total_ms_available\": " << (r.p2p_mode ? "false" : "true") << "\n";
    ss << "  },\n";

    // Hop RTT stats
    ss << "  \"hop_rtt\": {\n";
    ss << "    \"avg_ms\": " << r.avg_hop_rtt_ms << ",\n";
    ss << "    \"min_ms\": " << r.min_hop_rtt_ms << ",\n";
    ss << "    \"p50_ms\": " << r.p50_hop_rtt_ms << ",\n";
    ss << "    \"p95_ms\": " << r.p95_hop_rtt_ms << ",\n";
    ss << "    \"p99_ms\": " << r.p99_hop_rtt_ms << ",\n";
    ss << "    \"max_ms\": " << r.max_hop_rtt_ms << "\n";
    ss << "  },\n";

    // Server detailed timing
    ss << "  \"server_timing_us\": {\n";
    ss << "    \"zmq_recv\": " << r.avg_zmq_recv_us << ",\n";
    ss << "    \"zmq_send\": " << r.avg_zmq_send_us << ",\n";
    ss << "    \"deserialize\": " << r.avg_deserialize_us << ",\n";
    ss << "    \"serialize\": " << r.avg_serialize_us << ",\n";
    ss << "    \"queue_depth\": " << r.avg_queue_depth << ",\n";
    ss << "    \"resp_queue_wait\": " << r.avg_resp_queue_wait_us << ",\n";
    ss << "    \"io_processing\": " << r.avg_io_processing_us << ",\n";
    ss << "    \"worker_overhead\": " << r.avg_worker_overhead_us << ",\n";
    ss << "    \"send_prep\": " << r.avg_send_prep_us << "\n";
    ss << "  },\n";

    // Coordinator timing per hop
    ss << "  \"coord_timing_per_hop_us\": {\n";
    ss << "    \"available\": " << (r.p2p_mode ? "false" : "true") << ",\n";
    ss << "    \"serialize\": " << r.coord_serialize_us << ",\n";
    ss << "    \"send\": " << r.coord_send_us << ",\n";
    ss << "    \"response_queue\": " << r.coord_response_queue_us << ",\n";
    ss << "    \"recv\": " << r.coord_recv_us << ",\n";
    ss << "    \"deserialize\": " << r.coord_deserialize_us << "\n";
    ss << "  },\n";

    // Coordinator timing totals
    ss << "  \"coord_timing_totals\": {\n";
    ss << "    \"serialize_us\": " << r.timing_serialize_us << ",\n";
    ss << "    \"send_us\": " << r.timing_send_us << ",\n";
    ss << "    \"response_queue_us\": " << r.timing_response_queue_us << ",\n";
    ss << "    \"poll_wait_us\": " << r.timing_poll_wait_us << ",\n";
    ss << "    \"recv_us\": " << r.timing_recv_us << ",\n";
    ss << "    \"deserialize_us\": " << r.timing_deserialize_us << ",\n";
    ss << "    \"process_us\": " << r.timing_process_us << ",\n";
    ss << "    \"send_count\": " << r.timing_send_count << ",\n";
    ss << "    \"poll_count\": " << r.timing_poll_count << ",\n";
    ss << "    \"recv_count\": " << r.timing_recv_count << ",\n";
    ss << "    \"total_hops\": " << r.total_hops << "\n";
    ss << "  },\n";

    // Bottleneck analysis
    ss << "  \"bottleneck\": {\n";
    ss << "    \"overhead_per_hop_us\": " << r.overhead_per_hop_us << ",\n";
    ss << "    \"overhead_ratio\": " << r.overhead_ratio << "\n";
    ss << "  },\n";

    // Expansion analysis
    ss << "  \"expansion\": {\n";
    ss << "    \"local_ratio\": " << r.local_ratio << ",\n";
    ss << "    \"local_expanded\": " << r.local_expanded << ",\n";
    ss << "    \"top_W\": " << r.top_W << ",\n";
    ss << "    \"avg_neighbors_evaluated\": " << r.avg_neighbors_evaluated << "\n";
    ss << "  },\n";

    // Distance stats
    ss << "  \"distance\": {\n";
    ss << "    \"avg_pq_dist\": " << r.avg_pq_dist << ",\n";
    ss << "    \"avg_full_dist\": " << r.avg_full_dist << "\n";
    ss << "  },\n";

    // Per-hop distance computation
    ss << "  \"per_hop_distance\": {\n";
    ss << "    \"per_hop_avg\": {";
    first = true;
    for (const auto& [hop, avg] : r.per_hop_dist_avg) {
        if (!first) ss << ", ";
        ss << "\"" << hop << "\": " << avg;
        first = false;
    }
    ss << "},\n";
    ss << "    \"hop0_query_count\": " << r.hop0_query_count << ",\n";
    ss << "    \"avg_hop0_dist\": " << r.avg_hop0_dist << ",\n";
    ss << "    \"multihop_query_count\": " << r.multihop_query_count << ",\n";
    ss << "    \"multihop_total_hops\": " << r.multihop_total_hops << ",\n";
    ss << "    \"avg_multihop_dist_per_hop\": " << r.avg_multihop_dist_per_hop << "\n";
    ss << "  },\n";

    // Server load
    ss << "  \"server_load\": [";
    for (size_t i = 0; i < r.requests_per_server.size(); i++) {
        if (i > 0) ss << ", ";
        ss << r.requests_per_server[i];
    }
    ss << "],\n";

    // Lock metrics
    ss << "  \"lock_metrics\": {\n";
    ss << "    \"total_shared_locks\": " << r.total_shared_locks << ",\n";
    ss << "    \"total_unique_locks\": " << r.total_unique_locks << ",\n";
    ss << "    \"avg_unique_lock_wait_us\": " << r.avg_unique_lock_wait_us << ",\n";
    ss << "    \"avg_shared_lock_wait_us\": " << r.avg_shared_lock_wait_us << ",\n";
    // Per-phase lock metrics
    ss << "    \"phase_specific\": {\n";
    ss << "      \"insert_alloc\": {\"count\": " << r.insert_alloc_lock_count
       << ", \"avg_wait_us\": " << r.avg_insert_alloc_lock_wait_us << "},\n";
    ss << "      \"insert_edge\": {\"count\": " << r.insert_edge_lock_count
       << ", \"avg_wait_us\": " << r.avg_insert_edge_lock_wait_us << "},\n";
    ss << "      \"backlink\": {\"count\": " << r.backlink_lock_count
       << ", \"avg_wait_us\": " << r.avg_backlink_lock_wait_us << "},\n";
    ss << "      \"broadcast\": {\"count\": " << r.broadcast_lock_count
       << ", \"avg_wait_us\": " << r.avg_broadcast_lock_wait_us << "}\n";
    ss << "    },\n";
    ss << "    \"per_server\": [";
    for (size_t i = 0; i < r.per_server_lock_stats.size(); i++) {
        if (i > 0) ss << ", ";
        auto& [shared, unique, shared_wait, unique_wait] = r.per_server_lock_stats[i];
        ss << "{\"shared\": " << shared << ", \"unique\": " << unique
           << ", \"shared_wait_us\": " << shared_wait << ", \"unique_wait_us\": " << unique_wait << "}";
    }
    ss << "],\n";
    // Per-server phase-specific lock stats
    ss << "    \"per_server_phase\": [";
    for (size_t i = 0; i < r.per_server_phase_lock_stats.size(); i++) {
        if (i > 0) ss << ", ";
        const auto& ps = r.per_server_phase_lock_stats[i];
        ss << "{\"insert_alloc\": {\"count\": " << ps.insert_alloc_count << ", \"avg_wait_us\": " << ps.insert_alloc_wait_us << "}, "
           << "\"insert_edge\": {\"count\": " << ps.insert_edge_count << ", \"avg_wait_us\": " << ps.insert_edge_wait_us << "}, "
           << "\"backlink\": {\"count\": " << ps.backlink_count << ", \"avg_wait_us\": " << ps.backlink_wait_us << "}, "
           << "\"broadcast\": {\"count\": " << ps.broadcast_count << ", \"avg_wait_us\": " << ps.broadcast_wait_us << "}, "
           << "\"graph\": {\"bl_requested\": " << ps.backlink_edges_requested
           << ", \"bl_applied\": " << ps.backlink_edges_applied
           << ", \"bl_skip_not_local\": " << ps.backlink_skip_not_local
           << ", \"bl_skip_dup\": " << ps.backlink_skip_already_exists
           << ", \"bl_skip_out_of_range\": " << ps.backlink_skip_out_of_range << "}}";
    }
    ss << "],\n";
    auto write_role = [&](const char* name, const ExperimentResult::LockRoleStats& role, bool comma) {
        ss << "      \"" << name << "\": {"
           << "\"acquisitions\": " << role.acquisitions
           << ", \"wait_us\": " << role.wait_us
           << ", \"hold_us\": " << role.hold_us
           << ", \"avg_wait_us\": " << role.avg_wait_us()
           << ", \"avg_hold_us\": " << role.avg_hold_us()
           << ", \"max_wait_us\": " << role.max_wait_us
           << ", \"max_hold_us\": " << role.max_hold_us
           << "}";
        if (comma) ss << ",";
        ss << "\n";
    };
    auto write_role_fields = [&](const ExperimentResult::LockRoleStats& role, bool comma) {
        ss << "{\"acquisitions\": " << role.acquisitions
           << ", \"wait_us\": " << role.wait_us
           << ", \"hold_us\": " << role.hold_us
           << ", \"avg_wait_us\": " << role.avg_wait_us()
           << ", \"avg_hold_us\": " << role.avg_hold_us()
           << ", \"max_wait_us\": " << role.max_wait_us
           << ", \"max_hold_us\": " << role.max_hold_us
           << "}";
        if (comma) ss << ", ";
    };
    ss << "    \"resize_mutex_profile\": {\n";
    ss << "      \"metric_semantics\": \"per actual lock acquisition; wait_us is acquire wait, hold_us is time until release; insert_alloc_mutex is g_alloc_mutex, not g_resize_mutex\",\n";
    write_role("search_shared", r.resize_mutex_profile.search_shared, true);
    write_role("insert_alloc_unique", r.resize_mutex_profile.insert_alloc_unique, true);
    write_role("insert_alloc_shared", r.resize_mutex_profile.insert_alloc_shared, true);
    write_role("insert_alloc_mutex", r.resize_mutex_profile.insert_alloc_mutex, true);
    write_role("insert_prune_shared", r.resize_mutex_profile.insert_prune_shared, true);
    write_role("insert_edge_shared", r.resize_mutex_profile.insert_edge_shared, true);
    write_role("backlink_unique", r.resize_mutex_profile.backlink_unique, true);
    write_role("broadcast_unique", r.resize_mutex_profile.broadcast_unique, true);
    write_role("search_node_lock", r.resize_mutex_profile.search_node_lock, true);
    {
        const auto& sub = r.resize_mutex_profile.search_sub;
        ss << "      \"search_sub_stage\": {"
              "\"distance_us\": " << sub.distance_us
           << ", \"beam_sort_us\": " << sub.beam_sort_us
           << ", \"distance_count\": " << sub.distance_count
           << ", \"neighbors_evaluated\": " << sub.neighbors_evaluated
           << ", \"request_count\": " << sub.request_count
           << ", \"avg_distance_us_per_request\": "
           << (sub.request_count > 0 ? static_cast<double>(sub.distance_us) / sub.request_count : 0.0)
           << ", \"avg_beam_sort_us_per_request\": "
           << (sub.request_count > 0 ? static_cast<double>(sub.beam_sort_us) / sub.request_count : 0.0)
           << ", \"avg_neighbors_per_request\": "
           << (sub.request_count > 0 ? static_cast<double>(sub.neighbors_evaluated) / sub.request_count : 0.0)
           << ", \"ns_per_distance\": "
           << (sub.distance_count > 0 ? static_cast<double>(sub.distance_us) * 1000.0 / sub.distance_count : 0.0)
           << "},\n";
    }
    ss << "      \"per_server\": [";
    for (size_t i = 0; i < r.resize_mutex_profile.per_server.size(); i++) {
        if (i > 0) ss << ", ";
        const auto& ps = r.resize_mutex_profile.per_server[i];
        ss << "{\"search_shared\": ";
        write_role_fields(ps.search_shared, true);
        ss << "\"insert_alloc_unique\": ";
        write_role_fields(ps.insert_alloc_unique, true);
        ss << "\"insert_alloc_shared\": ";
        write_role_fields(ps.insert_alloc_shared, true);
        ss << "\"insert_alloc_mutex\": ";
        write_role_fields(ps.insert_alloc_mutex, true);
        ss << "\"insert_prune_shared\": ";
        write_role_fields(ps.insert_prune_shared, true);
        ss << "\"insert_edge_shared\": ";
        write_role_fields(ps.insert_edge_shared, true);
        ss << "\"backlink_unique\": ";
        write_role_fields(ps.backlink_unique, true);
        ss << "\"broadcast_unique\": ";
        write_role_fields(ps.broadcast_unique, true);
        ss << "\"search_node_lock\": ";
        write_role_fields(ps.search_node_lock, false);
        ss << "}";
    }
    ss << "],\n";
    ss << "      \"per_server_search_sub_stage\": [";
    for (size_t i = 0; i < r.resize_mutex_profile.search_sub_per_server.size(); i++) {
        if (i > 0) ss << ", ";
        const auto& sub = r.resize_mutex_profile.search_sub_per_server[i];
        ss << "{\"server\": " << i
           << ", \"distance_and_beam_update_us\": " << sub.distance_us
           << ", \"beam_sort_us\": " << sub.beam_sort_us
           << ", \"distance_count\": " << sub.distance_count
           << ", \"neighbors_evaluated\": " << sub.neighbors_evaluated
           << ", \"request_count\": " << sub.request_count
           << ", \"ns_per_distance_and_beam_update\": "
           << (sub.distance_count > 0 ? static_cast<double>(sub.distance_us) * 1000.0 / sub.distance_count : 0.0)
           << "}";
    }
    ss << "]\n";
    ss << "    }\n";
    ss << "  },\n";

    // Insert stats (query_latency, per_query_breakdown and search included)
    ss << "  \"insert\": {\n";
    ss << "    \"count\": " << r.insert_count << ",\n";
    ss << "    \"throughput\": " << r.insert_throughput << ",\n";

    // Whole INSERT latency = search + insert_rpc (backlink/broadcast excluded)
    double insert_query_latency_avg = r.per_insert_search_ms + r.per_insert_rpc_ms;
    ss << "    \"query_latency\": {\n";
    ss << "      \"avg_ms\": " << insert_query_latency_avg << ",\n";
    ss << "      \"search_ms\": " << r.per_insert_search_ms << ",\n";
    ss << "      \"insert_rpc_ms\": " << r.per_insert_rpc_ms << ",\n";
    ss << "      \"semantics\": \"foreground_search_plus_insert_rpc\"\n";
    ss << "    },\n";

    // INSERT RPC breakdown (no hops, one server)
    ss << "    \"per_query_breakdown\": {\n";
    ss << "      \"queue_wait_ms\": " << r.per_insert_rpc_queue_wait_ms << ",\n";
    ss << "      \"server_ms\": { \"total\": " << (r.per_insert_rpc_alloc_lock_ms + r.per_insert_rpc_edge_lock_ms + r.per_insert_rpc_prune_ms + r.per_insert_rpc_alloc_memory_ms + r.per_insert_rpc_edge_memory_ms + r.per_insert_rpc_local_backlink_prune_ms)
       << ", \"lock_wait_ms\": " << (r.per_insert_rpc_alloc_lock_ms + r.per_insert_rpc_edge_lock_ms) << " },\n";
    ss << "      \"deserialize_ms\": " << r.per_insert_rpc_deserialize_ms << ",\n";
    ss << "      \"prune_ms\": " << r.per_insert_rpc_prune_ms << ",\n";
    ss << "      \"memory_ms\": " << (r.per_insert_rpc_alloc_memory_ms + r.per_insert_rpc_edge_memory_ms) << ",\n";
    ss << "      \"local_backlink_prune_ms\": " << r.per_insert_rpc_local_backlink_prune_ms << "\n";
    ss << "    },\n";

    // The insert's search phase (the same format as SEARCH)
    if (r.insert_search.count > 0) {
        ss << "    \"search\": {\n";
        ss << "      \"count\": " << r.insert_search.count << ",\n";
        ss << "      \"total_hops\": " << r.insert_search.total_hops << ",\n";
        ss << "      \"avg_hops\": " << r.insert_search.avg_hops << ",\n";
        ss << "      \"hop_distribution\": {";
        first = true;
        for (const auto& [hops, count] : r.insert_search.hop_distribution) {
            if (!first) ss << ", ";
            ss << "\"" << hops << "\": " << count;
            first = false;
        }
        ss << "},\n";
        ss << "      \"query_latency\": {\n";
        ss << "        \"avg_ms\": " << r.insert_search.avg_latency_ms << ",\n";
        ss << "        \"min_ms\": " << r.insert_search.min_latency_ms << ",\n";
        ss << "        \"p50_ms\": " << r.insert_search.p50_latency_ms << ",\n";
        ss << "        \"p95_ms\": " << r.insert_search.p95_latency_ms << ",\n";
        ss << "        \"p99_ms\": " << r.insert_search.p99_latency_ms << ",\n";
        ss << "        \"max_ms\": " << r.insert_search.max_latency_ms << "\n";
        ss << "      },\n";
        ss << "      \"per_query_breakdown\": {\n";
        ss << "        \"server_ms\": { \"total\": " << r.insert_search.avg_server_per_query_ms
           << ", \"lock_wait_ms\": " << r.insert_search.avg_lock_per_query_ms << " },\n";
        ss << "        \"worker_queue_ms\": " << r.insert_search.avg_worker_queue_per_query_ms << ",\n";
        ss << "        \"response_queue_ms\": " << r.insert_search.avg_response_queue_per_query_ms << ",\n";
        ss << "        \"network_ms\": " << r.insert_search.avg_network_per_query_ms << ",\n";
        ss << "        \"coord_total_ms\": " << r.insert_search.avg_coord_total_per_query_ms << ",\n";
        ss << "        \"coord_total_ms_available\": " << (r.p2p_mode ? "false" : "true") << "\n";
        ss << "      },\n";
        ss << "      \"per_hop_breakdown\": {\n";
        ss << "        \"avg_latency_ms\": " << r.insert_search.avg_hop_rtt_ms << ",\n";
        ss << "        \"server_ms\": { \"total\": " << r.insert_search.avg_server_per_hop_ms
           << ", \"lock_wait_ms\": " << r.insert_search.avg_lock_per_hop_ms << " },\n";
        ss << "        \"worker_queue_ms\": " << r.insert_search.avg_worker_queue_per_hop_ms << ",\n";
        ss << "        \"response_queue_ms\": " << r.insert_search.avg_response_queue_per_hop_ms << ",\n";
        ss << "        \"network_ms\": " << r.insert_search.avg_network_per_hop_ms << ",\n";
        ss << "        \"coord_total_ms\": " << r.insert_search.avg_coord_total_per_hop_ms << ",\n";
        ss << "        \"coord_total_ms_available\": " << (r.p2p_mode ? "false" : "true") << "\n";
        ss << "      },\n";
        ss << "      \"hop_rtt\": {\n";
        ss << "        \"avg_ms\": " << r.insert_search.avg_hop_rtt_ms << ",\n";
        ss << "        \"min_ms\": " << r.insert_search.min_hop_rtt_ms << ",\n";
        ss << "        \"p50_ms\": " << r.insert_search.p50_hop_rtt_ms << ",\n";
        ss << "        \"p95_ms\": " << r.insert_search.p95_hop_rtt_ms << ",\n";
        ss << "        \"p99_ms\": " << r.insert_search.p99_hop_rtt_ms << ",\n";
        ss << "        \"max_ms\": " << r.insert_search.max_hop_rtt_ms << "\n";
        ss << "      }\n";
        ss << "    },\n";
    }

    // Older fields kept (backward compatibility)
    ss << "    \"avg_backlinks\": " << r.avg_backlinks << ",\n";
    ss << "    \"total_backlinks\": " << r.total_backlinks << ",\n";
    ss << "    \"total_broadcasts\": " << r.total_broadcasts << "\n";
    ss << "  },\n";

    if (r.insert_count > 0) {
        ss << "  \"maintenance\": {\n";
        ss << "    \"backlink\": {\n";
        ss << "      \"pre_drain_edges\": " << r.maintenance_backlink_pre_drain_edges << ",\n";
        ss << "      \"drain_exact_edges\": " << r.maintenance_backlink_drain_exact_edges << ",\n";
        ss << "      \"combined_edges\": " << r.maintenance_backlink_combined_edges << ",\n";
        ss << "      \"combined_avg_per_insert\": " << r.maintenance_backlink_combined_avg_per_insert << ",\n";
        ss << "      \"metric_complete\": " << (r.maintenance_backlink_metric_complete ? "true" : "false") << "\n";
        ss << "    },\n";
        ss << "    \"broadcast\": {\n";
        ss << "      \"pre_drain_acks\": " << r.maintenance_broadcast_pre_drain_acks << ",\n";
        ss << "      \"drain_acks\": " << r.maintenance_broadcast_drain_acks << ",\n";
        ss << "      \"combined_acks\": " << r.maintenance_broadcast_combined_acks << ",\n";
        ss << "      \"metric_complete\": " << (r.maintenance_broadcast_metric_complete ? "true" : "false") << "\n";
        ss << "    }\n";
        ss << "  },\n";
    }

    // Unified pipeline timing (kept for debugging)
    ss << "  \"unified_timing\": {\n";
    ss << "    \"search_phase_ms\": " << r.search_phase_ms << ",\n";
    ss << "    \"insert_rpc_ms\": " << r.insert_rpc_ms << ",\n";
    ss << "    \"backlink_ms\": " << r.backlink_ms << ",\n";
    ss << "    \"broadcast_ms\": " << r.broadcast_ms << "\n";
    ss << "  },\n";

    // Resize stats (per-vector breakdown)
    uint32_t total_resize_count = r.resize_full_base.count + r.resize_membership.count +
                                  r.resize_adj.count + r.resize_global_to_local.count;
    double total_resize_ms = r.resize_full_base.total_ms + r.resize_membership.total_ms +
                             r.resize_adj.total_ms + r.resize_global_to_local.total_ms;
    ss << "  \"resize\": {\n";
    ss << "    \"total_count\": " << total_resize_count << ",\n";
    ss << "    \"total_ms\": " << total_resize_ms << ",\n";
    ss << "    \"by_vector\": {\n";
    ss << "      \"full_base\": {\"count\": " << r.resize_full_base.count << ", \"ms\": " << r.resize_full_base.total_ms << "},\n";
    ss << "      \"membership\": {\"count\": " << r.resize_membership.count << ", \"ms\": " << r.resize_membership.total_ms << "},\n";
    ss << "      \"adj\": {\"count\": " << r.resize_adj.count << ", \"ms\": " << r.resize_adj.total_ms << "},\n";
    ss << "      \"global_to_local\": {\"count\": " << r.resize_global_to_local.count << ", \"ms\": " << r.resize_global_to_local.total_ms << "}\n";
    ss << "    },\n";
    ss << "    \"by_server\": [\n";
    const int by_server = std::min(r.num_servers, ExperimentResult::MAX_SERVERS);
    for (int s = 0; s < by_server; s++) {
        const auto& srv = r.resize_per_server[s];
        uint32_t srv_count = srv.full_base.count + srv.membership.count + srv.adj.count + srv.global_to_local.count;
        double srv_ms = srv.full_base.total_ms + srv.membership.total_ms + srv.adj.total_ms + srv.global_to_local.total_ms;
        ss << "      {\"server\": " << s << ", \"count\": " << srv_count << ", \"ms\": " << srv_ms;
        ss << ", \"full_base\": " << srv.full_base.total_ms;
        ss << ", \"membership\": " << srv.membership.total_ms;
        ss << ", \"adj\": " << srv.adj.total_ms;
        ss << ", \"global_to_local\": " << srv.global_to_local.total_ms << "}";
        if (s < by_server - 1) ss << ",";
        ss << "\n";
    }
    ss << "    ]\n";
    ss << "  }";

    // Beam tracking (only if data exists)
    if (r.beam_total_entries > 0 || r.result_total_entries > 0) {
        ss << ",\n";
        ss << "  \"beam_tracking\": {\n";
        ss << "    \"beam_total_entries\": " << r.beam_total_entries << ",\n";
        ss << "    \"beam_inserted_entries\": " << r.beam_inserted_entries << ",\n";
        double beam_inserted_ratio = r.beam_total_entries > 0 ?
            100.0 * r.beam_inserted_entries / r.beam_total_entries : 0.0;
        ss << "    \"beam_inserted_ratio\": " << beam_inserted_ratio << ",\n";
        ss << "    \"result_total_entries\": " << r.result_total_entries << ",\n";
        ss << "    \"result_inserted_entries\": " << r.result_inserted_entries << ",\n";
        double result_inserted_ratio = r.result_total_entries > 0 ?
            100.0 * r.result_inserted_entries / r.result_total_entries : 0.0;
        ss << "    \"result_inserted_ratio\": " << result_inserted_ratio << "\n";
        ss << "  }";
    }

    // Server stats (if available)
    if (!r.server_total_nodes.empty()) {
        ss << ",\n";
        ss << "  \"server_stats\": {\n";
        ss << "    \"per_server_nodes\": [";
        for (size_t i = 0; i < r.server_total_nodes.size(); i++) {
            if (i > 0) ss << ", ";
            ss << r.server_total_nodes[i];
        }
        ss << "],\n";
        uint64_t total = 0;
        for (auto n : r.server_total_nodes) total += n;
        ss << "    \"total_nodes\": " << total << "\n";
        ss << "  }";
    }

    // ==========================================
    // Backlink Metrics (separate top-level)
    // ==========================================
    if (r.backlink.count > 0) {
        ss << ",\n";
        ss << "  \"backlink\": {\n";
        ss << "    \"count\": " << r.backlink.count << ",\n";
        ss << "    \"avg_per_insert\": " << r.backlink.avg_per_insert << ",\n";
        ss << "    \"wall_clock_ms\": " << r.backlink.wall_clock_ms << ",\n";
        ss << "    \"total_lock_wait_ms\": " << r.backlink.total_lock_wait_ms << "\n";
        ss << "  }";
    }

    // ==========================================
    // Broadcast Metrics (separate top-level)
    // ==========================================
    if (r.broadcast.count > 0) {
        ss << ",\n";
        ss << "  \"broadcast\": {\n";
        ss << "    \"count\": " << r.broadcast.count << ",\n";
        ss << "    \"wall_clock_ms\": " << r.broadcast.wall_clock_ms << ",\n";
        ss << "    \"total_lock_wait_ms\": " << r.broadcast.total_lock_wait_ms << "\n";
        ss << "  }";
    }

    // Merge search stats
    ss << ",\n";
    ss << "  \"merge_search\": {\n";
    ss << "    \"pending_mgr_size\": " << r.pending_mgr_size << ",\n";
    ss << "    \"calls\": " << r.merge_search_calls << ",\n";
    ss << "    \"pending_total\": " << r.merge_search_pending_total << ",\n";
    ss << "    \"pending_eligible\": " << r.merge_search_pending_eligible << ",\n";
    ss << "    \"pending_topk\": " << r.merge_search_pending_topk << ",\n";
    ss << "    \"result_changed\": " << r.merge_search_result_changed << ",\n";
    ss << "    \"wall_ms\": " << r.merge_search_wall_ms << ",\n";
    ss << "    \"avg_us\": " << r.merge_search_avg_us << "\n";
    ss << "  }";

    // ==========================================
    // Partition Analysis
    // ==========================================
    if (!r.partition_analysis.entry_partition.empty() || !r.partition_analysis.final_partition.empty()) {
        ss << ",\n";
        ss << "  \"partition_analysis\": {\n";
        ss << "    \"entry_partition\": {";
        first = true;
        for (const auto& [part, count] : r.partition_analysis.entry_partition) {
            if (!first) ss << ", ";
            ss << "\"" << part << "\": " << count;
            first = false;
        }
        ss << "},\n";
        ss << "    \"final_partition\": {";
        first = true;
        for (const auto& [part, count] : r.partition_analysis.final_partition) {
            if (!first) ss << ", ";
            ss << "\"" << part << "\": " << count;
            first = false;
        }
        ss << "}";
        if (!r.partition_analysis.transitions.empty()) {
            ss << ",\n    \"transitions\": [";
            for (size_t i = 0; i < r.partition_analysis.transitions.size(); i++) {
                if (i > 0) ss << ", ";
                ss << "[";
                for (size_t j = 0; j < r.partition_analysis.transitions[i].size(); j++) {
                    if (j > 0) ss << ", ";
                    ss << r.partition_analysis.transitions[i][j];
                }
                ss << "]";
            }
            ss << "]";
        }
        ss << "\n  }";
    }

    // ==========================================
    // Network Baseline (PING measurement)
    // ==========================================
    if (r.network_baseline.measured) {
        ss << ",\n";
        ss << "  \"network_baseline\": {\n";
        ss << "    \"ping_count\": " << r.network_baseline.ping_count << ",\n";
        // Original format: servers array
        ss << "    \"servers\": [\n";
        for (size_t i = 0; i < r.network_baseline.servers.size(); i++) {
            const auto& srv = r.network_baseline.servers[i];
            ss << "      {\"id\": " << srv.id
               << ", \"count\": " << srv.count
               << ", \"avg_us\": " << srv.avg_us
               << ", \"min_us\": " << srv.min_us
               << ", \"p50_us\": " << srv.p50_us
               << ", \"p95_us\": " << srv.p95_us
               << ", \"p99_us\": " << srv.p99_us
               << ", \"max_us\": " << srv.max_us << "}";
            if (i < r.network_baseline.servers.size() - 1) ss << ",";
            ss << "\n";
        }
        ss << "    ],\n";
        // Original format: overall
        ss << "    \"overall\": {"
           << "\"total_pings\": " << r.network_baseline.total_pings
           << ", \"avg_us\": " << r.network_baseline.overall_avg_us
           << ", \"min_us\": " << r.network_baseline.overall_min_us
           << ", \"p50_us\": " << r.network_baseline.overall_p50_us
           << ", \"p95_us\": " << r.network_baseline.overall_p95_us
           << ", \"p99_us\": " << r.network_baseline.overall_p99_us
           << ", \"max_us\": " << r.network_baseline.overall_max_us << "},\n";
        // Direct stats (same as overall)
        ss << "    \"direct\": {"
           << "\"avg_us\": " << r.network_baseline.direct.avg_us
           << ", \"min_us\": " << r.network_baseline.direct.min_us
           << ", \"p50_us\": " << r.network_baseline.direct.p50_us
           << ", \"p95_us\": " << r.network_baseline.direct.p95_us
           << ", \"p99_us\": " << r.network_baseline.direct.p99_us
           << ", \"max_us\": " << r.network_baseline.direct.max_us << "}\n";
        ss << "  }";
    }

    // Inserted Query Metrics
    if (r.inserted_query.count > 0) {
        ss << ",\n";
        ss << "  \"inserted_query\": {\n";
        ss << "    \"count\": " << r.inserted_query.count << ",\n";
        ss << "    \"recall\": " << r.inserted_query.recall << ",\n";
        ss << "    \"baseline_count\": " << r.inserted_query.baseline_count << "\n";
        ss << "  }";
    }

    // Debug NPY path (if available)
    if (!r.debug_npy_path.empty()) {
        ss << ",\n";
        ss << "  \"debug_npy_path\": \"" << r.debug_npy_path << "\"";
    }

    ss << "\n}\n";
    return ss.str();
}

/**
 * Output experiment result as JSON.
 * The output is wrapped with markers for easy extraction by scripts.
 */
inline void output_result(const ExperimentResult& r) {
    std::cout << "===JSON_RESULT===\n";
    std::cout << to_json(r);
    std::cout << "===JSON_END===" << std::endl;
}

/**
 * Get current timestamp in ISO format.
 */
inline std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&time));
    return std::string(buf);
}

/**
 * Populate timing fields in ExperimentResult from GlobalMetrics.
 */
inline void populate_timing_from_metrics(ExperimentResult& result, GlobalMetrics& metrics) {
    if (metrics.hop_rtt_us.empty()) return;

    // Calculate averages and min/max
    double total_rtt = 0;
    double min_rtt = 1e9, max_rtt = 0;
    for (double rtt : metrics.hop_rtt_us) {
        total_rtt += rtt;
        min_rtt = std::min(min_rtt, rtt);
        max_rtt = std::max(max_rtt, rtt);
    }
    result.avg_hop_rtt_ms = (total_rtt / metrics.hop_rtt_us.size()) / 1000.0;
    result.min_hop_rtt_ms = min_rtt / 1000.0;
    result.max_hop_rtt_ms = max_rtt / 1000.0;

    double avg_server_us = static_cast<double>(metrics.server_time_us) / metrics.hop_rtt_us.size();
    result.avg_server_per_hop_ms = avg_server_us / 1000.0;
    result.avg_network_per_hop_ms = result.avg_hop_rtt_ms - result.avg_server_per_hop_ms;

    // Percentiles
    std::sort(metrics.hop_rtt_us.begin(), metrics.hop_rtt_us.end());
    result.p50_hop_rtt_ms = metrics.hop_rtt_us[metrics.hop_rtt_us.size() / 2] / 1000.0;
    result.p95_hop_rtt_ms = metrics.hop_rtt_us[
        nearest_rank_percentile_index(metrics.hop_rtt_us.size(), 95)] / 1000.0;
    result.p99_hop_rtt_ms = metrics.hop_rtt_us[static_cast<size_t>(metrics.hop_rtt_us.size() * 0.99)] / 1000.0;

    // Distance stats
    if (result.queries > 0) {
        result.avg_pq_dist = static_cast<double>(metrics.pq_dist) / result.queries;
        result.avg_full_dist = static_cast<double>(metrics.full_dist) / result.queries;
    }

    // Expansion analysis
    result.local_expanded = metrics.local_expanded;
    result.top_W = metrics.top_W;
    result.local_ratio = (metrics.top_W > 0) ?
        (100.0 * metrics.local_expanded / metrics.top_W) : 0;
    if (result.queries > 0) {
        result.avg_neighbors_evaluated = static_cast<double>(metrics.neighbors_evaluated) / result.queries;
    }

    // Server load
    result.requests_per_server = metrics.requests_per_server;
}

}  // namespace ann
