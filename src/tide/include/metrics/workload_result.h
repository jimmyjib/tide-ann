// Result of one workload command (recall, insert, mixed), serialized by to_json().
#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <cstdint>

namespace ann {

// Interval of one recall command (coordinator steady clock) and the setup spans inside it.
struct ServingAccounting {
    struct Span { uint64_t begin_ns = 0, end_ns = 0; };
    bool enabled = false;
    uint64_t begin_ns = 0, end_ns = 0, owner_search_begin_ns = 0;
    uint64_t declared_queries = 0, completed_queries = 0;
    Span materialize, head_bf;

    static uint64_t ns_at(std::chrono::steady_clock::time_point value) {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            value.time_since_epoch()).count());
    }
    static uint64_t now_ns() { return ns_at(std::chrono::steady_clock::now()); }

    bool complete() const {
        if (!enabled) return false;
        if (!begin_ns || end_ns <= begin_ns || !owner_search_begin_ns ||
            owner_search_begin_ns < begin_ns || owner_search_begin_ns > end_ns)
            return false;
        if (!declared_queries || completed_queries != declared_queries) return false;
        uint64_t previous = begin_ns;
        auto check = [&](const Span& span) {
            if (!span.begin_ns || span.begin_ns < previous || span.end_ns < span.begin_ns ||
                span.end_ns > owner_search_begin_ns) return false;
            previous = span.end_ns;
            return true;
        };
        return check(materialize) && check(head_bf);
    }
    uint64_t elapsed_ns() const { return end_ns - begin_ns; }  // only when complete()
};

struct WorkloadResult {
    ServingAccounting serving;
    std::string timestamp;
    std::string description;
    std::string dataset;
    std::string groundtruth_file;
    std::string mode;  // "search", "insert", "mixed"
    bool diagnostic_mode = false;  // insert --search-only (no graph mutation)

    uint32_t queries = 0;
    uint32_t k = 0;
    uint32_t L = 0;
    uint32_t batch_size = 0;

    uint32_t search_count = 0;
    double recall = 0.0;
    double qps = 0.0;
    double avg_hops = 0.0;
    double total_time_ms = 0.0;
    double avg_dist_comp = 0.0;
    // digest of the ranked top-K results, computed after the timed part
    std::string search_result_digest_algorithm;
    std::string search_result_digest;
    uint32_t search_result_digest_queries = 0;
    uint64_t search_result_digest_neighbors = 0;

    // recall --self-ids: found_at_zero counts queries whose node is in its own top-K at distance 0.
    // failures lists (id, rank or -1, distance or -1) of the others, at most kSelfQueryFailureLimit.
    bool self_query = false;
    uint32_t self_query_count = 0;
    uint32_t self_query_found_at_zero = 0;
    uint32_t self_query_found_nonzero = 0;
    uint32_t self_query_not_found = 0;
    uint32_t self_query_top1_at_zero = 0;
    struct SelfQueryFailure { uint32_t id; int32_t rank; float distance; };
    std::vector<SelfQueryFailure> self_query_failures;
    bool self_query_failures_truncated = false;

    double search_wall_ms = 0.0;   // first search start to last search end
    double insert_wall_ms = 0.0;   // first insert start to last insert end
    double search_only_qps = 0.0;  // n_searches / search_wall_ms
    double insert_only_tps = 0.0;  // n_inserts / insert_wall_ms
    double total_with_metrics_ms = 0.0;     // including the metric-thread join
    double search_with_metrics_qps = 0.0;   // n_searches / total_with_metrics_ms
    double metric_thread_join_ms = 0.0;     // end-of-phase wait for the metric thread

    std::map<uint32_t, uint32_t> hop_distribution;

    // query latency (ms)
    double avg_query_latency_ms = 0.0;
    double min_query_latency_ms = 0.0;
    double p50_query_latency_ms = 0.0;
    double p95_query_latency_ms = 0.0;
    double p99_query_latency_ms = 0.0;
    double max_query_latency_ms = 0.0;

    // per query (ms)
    double avg_server_per_query_ms = 0.0;         // excluding the response queue
    double avg_worker_queue_per_query_ms = 0.0;
    double avg_response_queue_per_query_ms = 0.0;
    double avg_network_per_query_ms = 0.0;
    double avg_lock_per_query_ms = 0.0;           // included in server_ms
    double avg_coord_total_per_query_ms = 0.0;    // measured_coord_total_us

    double avg_pre_first_hop_per_query_ms = 0.0;  // start to first send
    double avg_inter_hop_per_query_ms = 0.0;      // coordinator time between hops

    // per hop (ms)
    double avg_hop_latency_ms = 0.0;
    double avg_server_per_hop_ms = 0.0;           // excluding the response queue
    double avg_worker_queue_per_hop_ms = 0.0;
    double avg_response_queue_per_hop_ms = 0.0;
    double avg_network_per_hop_ms = 0.0;
    double avg_lock_per_hop_ms = 0.0;             // included in server_ms
    double avg_coord_total_per_hop_ms = 0.0;      // measured_coord_total_us

    // hop RTT (ms)
    double avg_hop_rtt_ms = 0.0;
    double min_hop_rtt_ms = 0.0;
    double p50_hop_rtt_ms = 0.0;
    double p95_hop_rtt_ms = 0.0;
    double p99_hop_rtt_ms = 0.0;
    double max_hop_rtt_ms = 0.0;

    // server timing per hop (us)
    double avg_zmq_recv_us = 0.0;
    double avg_zmq_send_us = 0.0;
    double avg_deserialize_us = 0.0;
    double avg_serialize_us = 0.0;
    double avg_queue_depth = 0.0;
    double avg_resp_queue_wait_us = 0.0;
    double avg_io_processing_us = 0.0;  // receive end to enqueue
    double avg_worker_overhead_us = 0.0;  // unmeasured worker time
    double avg_send_prep_us = 0.0;  // pop to send start

    // coordinator timing per hop (us)
    double coord_serialize_us = 0.0;
    double coord_send_us = 0.0;
    double coord_response_queue_us = 0.0;  // server thread to main thread handoff
    double coord_recv_us = 0.0;
    double coord_deserialize_us = 0.0;

    // coordinator timing totals (us)
    uint64_t timing_serialize_us = 0;
    uint64_t timing_send_us = 0;
    uint64_t timing_response_queue_us = 0;
    uint64_t timing_poll_wait_us = 0;
    uint64_t timing_recv_us = 0;
    uint64_t timing_deserialize_us = 0;
    uint64_t timing_process_us = 0;
    uint64_t timing_send_count = 0;
    uint64_t timing_poll_count = 0;
    uint64_t timing_recv_count = 0;
    uint64_t total_hops = 0;

    double overhead_per_hop_us = 0.0;
    double overhead_ratio = 0.0;

    double local_ratio = 0.0;           // local_expanded / top_W * 100
    uint64_t local_expanded = 0;
    uint64_t top_W = 0;
    double avg_neighbors_evaluated = 0.0;

    double avg_pq_dist = 0.0;
    double avg_full_dist = 0.0;

    // hop index -> average distance evaluations
    std::map<uint32_t, double> per_hop_dist_avg;
    uint32_t hop0_query_count = 0;
    double avg_hop0_dist = 0.0;
    uint32_t multihop_query_count = 0;
    uint32_t multihop_total_hops = 0;
    double avg_multihop_dist_per_hop = 0.0;

    std::vector<uint64_t> requests_per_server;

    // lock totals
    uint64_t total_shared_locks = 0;
    uint64_t total_unique_locks = 0;
    double avg_unique_lock_wait_us = 0.0;
    double avg_shared_lock_wait_us = 0.0;

    // per insert phase: allocation, edges, backlinks, broadcasts
    uint64_t insert_alloc_lock_count = 0;
    double avg_insert_alloc_lock_wait_us = 0.0;
    uint64_t insert_edge_lock_count = 0;
    double avg_insert_edge_lock_wait_us = 0.0;
    uint64_t backlink_lock_count = 0;
    double avg_backlink_lock_wait_us = 0.0;
    uint64_t broadcast_lock_count = 0;
    double avg_broadcast_lock_wait_us = 0.0;

    // per server: shared and unique counts and waits
    std::vector<std::tuple<uint64_t, uint64_t, double, double>> per_server_lock_stats;

    // per server and insert phase: counts and waits
    struct PhaseSpecificLockStats {
        uint64_t insert_alloc_count = 0;
        double insert_alloc_wait_us = 0.0;
        uint64_t insert_edge_count = 0;
        double insert_edge_wait_us = 0.0;
        uint64_t backlink_count = 0;
        double backlink_wait_us = 0.0;
        uint64_t broadcast_count = 0;
        double broadcast_wait_us = 0.0;
        uint64_t backlink_edges_requested = 0;
        uint64_t backlink_edges_applied = 0;
        uint64_t backlink_skip_not_local = 0;
        uint64_t backlink_skip_already_exists = 0;
        uint64_t backlink_skip_out_of_range = 0;
    };
    std::vector<PhaseSpecificLockStats> per_server_phase_lock_stats;

    struct LockRoleStats {
        uint64_t acquisitions = 0;
        uint64_t wait_us = 0;
        uint64_t hold_us = 0;
        uint64_t max_wait_us = 0;
        uint64_t max_hold_us = 0;

        double avg_wait_us() const {
            return acquisitions > 0 ? static_cast<double>(wait_us) / acquisitions : 0.0;
        }
        double avg_hold_us() const {
            return acquisitions > 0 ? static_cast<double>(hold_us) / acquisitions : 0.0;
        }
    };

    struct ResizeMutexProfile {
        LockRoleStats search_shared;
        LockRoleStats insert_alloc_unique;
        LockRoleStats insert_alloc_shared;
        LockRoleStats insert_alloc_mutex;
        LockRoleStats insert_prune_shared;
        LockRoleStats insert_edge_shared;
        LockRoleStats backlink_unique;
        LockRoleStats broadcast_unique;
        LockRoleStats search_node_lock;  // search-path per-node mutex

        // search totals
        struct SearchSubStage {
            uint64_t distance_us = 0;
            uint64_t beam_sort_us = 0;
            uint64_t distance_count = 0;
            uint64_t neighbors_evaluated = 0;
            uint64_t request_count = 0;
        };
        SearchSubStage search_sub;
        std::vector<SearchSubStage> search_sub_per_server;

        struct PerServer {
            LockRoleStats search_shared;
            LockRoleStats insert_alloc_unique;
            LockRoleStats insert_alloc_shared;
            LockRoleStats insert_alloc_mutex;
            LockRoleStats insert_prune_shared;
            LockRoleStats insert_edge_shared;
            LockRoleStats backlink_unique;
            LockRoleStats broadcast_unique;
            LockRoleStats search_node_lock;  // search-path per-node mutex
        };
        std::vector<PerServer> per_server;
    };
    ResizeMutexProfile resize_mutex_profile;

    uint32_t insert_count = 0;
    double insert_throughput = 0.0;
    double avg_backlinks = 0.0;       // pre-drain totals
    uint32_t total_backlinks = 0;
    uint32_t total_broadcasts = 0;

    // insert results count before the drain and the drain reply counts the rest
    uint32_t maintenance_backlink_pre_drain_edges = 0;
    uint32_t maintenance_backlink_drain_exact_edges = 0;
    uint32_t maintenance_backlink_combined_edges = 0;
    double maintenance_backlink_combined_avg_per_insert = 0.0;
    bool maintenance_backlink_metric_complete = false;
    uint32_t maintenance_broadcast_pre_drain_acks = 0;
    uint32_t maintenance_broadcast_drain_acks = 0;
    uint32_t maintenance_broadcast_combined_acks = 0;
    bool maintenance_broadcast_metric_complete = false;

    uint32_t workloads_generated = 0;
    uint32_t search_phase_completed = 0;
    uint32_t insert_rpc_completed = 0;
    uint32_t backlink_phase_completed = 0;
    uint32_t broadcast_phase_completed = 0;

    // cumulative pipeline timing (ms)
    double search_phase_ms = 0.0;
    double insert_rpc_ms = 0.0;
    double backlink_ms = 0.0;
    double broadcast_ms = 0.0;

    // per insert (ms)
    double per_insert_search_ms = 0.0;
    double per_insert_rpc_ms = 0.0;
    double per_insert_backlink_ms = 0.0;
    double per_insert_broadcast_ms = 0.0;
    double per_insert_total_ms = 0.0;

    // lock wait per insert (ms)
    double per_insert_alloc_lock_wait_ms = 0.0;
    double per_insert_edge_lock_wait_ms = 0.0;
    double per_insert_backlink_lock_wait_ms = 0.0;
    double per_insert_broadcast_lock_wait_ms = 0.0;
    double per_insert_total_lock_wait_ms = 0.0;

    // INSERT request timing per insert (ms)
    double per_insert_rpc_queue_wait_ms = 0.0;
    double per_insert_rpc_deserialize_ms = 0.0;
    double per_insert_rpc_alloc_lock_ms = 0.0;
    double per_insert_rpc_edge_lock_ms = 0.0;
    double per_insert_rpc_prune_ms = 0.0;
    double per_insert_rpc_alloc_memory_ms = 0.0;
    struct ResizeVectorStats {
        uint32_t count = 0;
        double total_ms = 0.0;
    };
    ResizeVectorStats resize_full_base;
    ResizeVectorStats resize_membership;
    ResizeVectorStats resize_adj;
    ResizeVectorStats resize_global_to_local;
    // per server
    static constexpr int MAX_SERVERS = 64;  // the supported server counts (2..64)
    struct PerServerResize {
        ResizeVectorStats full_base;
        ResizeVectorStats membership;
        ResizeVectorStats adj;
        ResizeVectorStats global_to_local;
    };
    PerServerResize resize_per_server[MAX_SERVERS] = {};
    int num_servers = 0;
    double per_insert_rpc_edge_memory_ms = 0.0;
    double per_insert_rpc_local_backlink_prune_ms = 0.0;
    double per_insert_rpc_network_rtt_ms = 0.0;

    bool p2p_mode = false;

    // route_summary[route] = count, with route keys joining server ids by the U+2192 arrow
    std::map<std::string, uint32_t> route_summary;
    uint32_t route_traced_count = 0;  // queries with non-empty route
    // Per-route timing. Server time is measured locally. The transport residual is the path RTT minus
    // measured server and coordinator time.
    double route_total_processing_us = 0;
    double route_total_queue_us = 0;
    uint32_t route_processing_segments = 0;
    double route_transport_residual_total_us = 0;
    double route_transport_residual_avg_per_query_us = 0;
    double route_transport_residual_avg_per_hop_us = 0;
    double route_transport_residual_min_us = 0;
    double route_transport_residual_p50_us = 0;
    double route_transport_residual_p95_us = 0;
    double route_transport_residual_p99_us = 0;
    double route_transport_residual_max_us = 0;
    uint32_t route_transport_residual_count = 0;
    uint32_t route_transport_residual_negative_count = 0;

    uint64_t beam_total_entries = 0;
    uint64_t beam_inserted_entries = 0;
    uint64_t result_total_entries = 0;
    uint64_t result_inserted_entries = 0;

    // nodes per server after the command
    std::vector<uint32_t> server_total_nodes;

    // --debug results file, empty when none was written
    std::string debug_npy_path;

    // queries for inserted vectors
    struct InsertedQueryMetrics {
        uint32_t count = 0;
        double recall = 0.0;             // percent
        uint32_t baseline_count = 0;     // n_searches - count
    };
    InsertedQueryMetrics inserted_query;

    // search phase of inserts, per hop and per query
    struct InsertSearchMetrics {
        uint32_t count = 0;                          // completed insert searches
        uint32_t total_hops = 0;
        double avg_hops = 0.0;
        std::map<uint32_t, uint32_t> hop_distribution;

        double avg_server_per_hop_ms = 0.0;          // excluding the response queue
        double avg_worker_queue_per_hop_ms = 0.0;
        double avg_response_queue_per_hop_ms = 0.0;
        double avg_network_per_hop_ms = 0.0;
        double avg_lock_per_hop_ms = 0.0;            // included in server_ms
        double avg_coord_total_per_hop_ms = 0.0;

        double avg_server_per_query_ms = 0.0;
        double avg_worker_queue_per_query_ms = 0.0;
        double avg_response_queue_per_query_ms = 0.0;
        double avg_network_per_query_ms = 0.0;
        double avg_lock_per_query_ms = 0.0;
        double avg_coord_total_per_query_ms = 0.0;

        double avg_pre_first_hop_per_query_ms = 0.0;
        double avg_inter_hop_per_query_ms = 0.0;

        double avg_latency_ms = 0.0;                 // end to end
        double min_latency_ms = 0.0;
        double p50_latency_ms = 0.0;
        double p95_latency_ms = 0.0;
        double p99_latency_ms = 0.0;
        double max_latency_ms = 0.0;

        double avg_hop_rtt_ms = 0.0;
        double min_hop_rtt_ms = 0.0;
        double p50_hop_rtt_ms = 0.0;
        double p95_hop_rtt_ms = 0.0;
        double p99_hop_rtt_ms = 0.0;
        double max_hop_rtt_ms = 0.0;
    };
    InsertSearchMetrics insert_search;

    struct BacklinkMetrics {
        uint32_t count = 0;
        double avg_per_insert = 0.0;
        double wall_clock_ms = 0.0;        // longest request
        double total_lock_wait_ms = 0.0;   // summed over requests
    };
    BacklinkMetrics backlink;

    struct BroadcastMetrics {
        uint32_t count = 0;                // inserts x servers
        double wall_clock_ms = 0.0;        // longest request
        double total_lock_wait_ms = 0.0;   // summed over requests
    };
    BroadcastMetrics broadcast;

    // entry and result partitions of the searches
    struct PartitionAnalysis {
        std::map<uint32_t, uint32_t> entry_partition;
        std::map<uint32_t, uint32_t> final_partition;
        std::vector<std::vector<uint32_t>> transitions;
    };
    PartitionAnalysis partition_analysis;

    // PING round trips
    struct NetworkBaseline {
        bool measured = false;
        uint32_t ping_count = 0;  // per server

        // per server
        struct PerServerStats {
            uint32_t id = 0;
            uint32_t count = 0;
            double avg_us = 0.0;
            double min_us = 0.0;
            double p50_us = 0.0;
            double p95_us = 0.0;
            double p99_us = 0.0;
            double max_us = 0.0;
        };
        std::vector<PerServerStats> servers;

        // overall
        uint32_t total_pings = 0;
        double overall_avg_us = 0.0;
        double overall_min_us = 0.0;
        double overall_p50_us = 0.0;
        double overall_p95_us = 0.0;
        double overall_p99_us = 0.0;
        double overall_max_us = 0.0;

        // same as overall
        struct ModeStats {
            double avg_us = 0.0, min_us = 0.0, p50_us = 0.0, p95_us = 0.0;
            double p99_us = 0.0, max_us = 0.0;
        };
        ModeStats direct;
    };
    NetworkBaseline network_baseline;

    uint32_t pending_mgr_size = 0;  // pending inserts at metric collection time
    uint64_t merge_search_calls = 0;
    uint64_t merge_search_pending_total = 0;
    uint64_t merge_search_pending_eligible = 0;
    uint64_t merge_search_pending_topk = 0;
    uint64_t merge_search_result_changed = 0;
    double merge_search_wall_ms = 0.0;
    double merge_search_avg_us = 0.0;
};

}  // namespace ann
