#pragma once
/**
 * Experiment Result Structure
 *
 * All experiment results should be collected into this struct
 * for consistent JSON formatting.
 */

#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <cstdint>

namespace ann {

// Whole-serving accounting of one search command: one coordinator steady-clock
// envelope from query/context preparation to the last owner completion, with
// the two preparation spans it contains (workload materialization and the
// head-index brute force). Spans are inside the envelope, never added to it.
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

    std::string error() const {
        if (!enabled) return "not_search_command";
        if (!begin_ns || end_ns <= begin_ns || !owner_search_begin_ns ||
            owner_search_begin_ns < begin_ns || owner_search_begin_ns > end_ns)
            return "missing_or_unordered_envelope";
        if (!declared_queries || completed_queries != declared_queries)
            return "unproven_completed_query_scope";
        uint64_t previous = begin_ns;
        auto check = [&](const Span& span) {
            if (!span.begin_ns || span.begin_ns < previous || span.end_ns < span.begin_ns ||
                span.end_ns > owner_search_begin_ns) return false;
            previous = span.end_ns;
            return true;
        };
        if (!check(materialize) || !check(head_bf)) return "missing_overlapping_or_unordered_span";
        return {};
    }
    uint64_t raw_ns() const { return end_ns - begin_ns; } // only after error().empty()
};

/**
 * Experiment result structure for JSON output.
 * All experiment results should be collected into this struct and output via
 * output_result() for consistent JSON formatting.
 */
struct ExperimentResult {
    ServingAccounting serving_accounting;
    // Basic info
    std::string timestamp;
    std::string description;  // User-provided experiment description
    std::string dataset;
    std::string groundtruth_file;
    std::string mode;  // "search", "insert", "mixed"
    bool diagnostic_mode = false;  // insert --search-only (no graph mutation)

    // Query parameters
    uint32_t queries = 0;
    uint32_t k = 0;
    uint32_t L = 0;
    uint32_t batch_size = 0;

    // Search metrics
    uint32_t search_count = 0;  // Number of search queries
    double recall = 0.0;
    double qps = 0.0;
    double avg_hops = 0.0;
    double total_time_ms = 0.0;
    double avg_dist_comp = 0.0;
    // Deterministic digest of ranked top-K search results. Computed after the
    // timed workload/metric-thread join and therefore excluded from QPS/latency.
    std::string search_result_digest_algorithm;
    std::string search_result_digest;
    uint32_t search_result_digest_queries = 0;
    uint64_t search_result_digest_neighbors = 0;

    // Self-query check (recall --self-ids): each query is the vector of its
    // expected node; found_at_zero = the node is in its own top-K at distance
    // exactly 0.0. failures: (id, rank or -1, distance or -1) of every other
    // query, at most kSelfQueryFailureLimit of them.
    bool self_query = false;
    uint32_t self_query_count = 0;
    uint32_t self_query_found_at_zero = 0;
    uint32_t self_query_found_nonzero = 0;
    uint32_t self_query_not_found = 0;
    uint32_t self_query_top1_at_zero = 0;
    struct SelfQueryFailure { uint32_t id; int32_t rank; float distance; };
    std::vector<SelfQueryFailure> self_query_failures;
    bool self_query_failures_truncated = false;

    // Search/Insert wall clock separation
    double search_wall_ms = 0.0;   // Wall clock for search workloads (first start to last end)
    double insert_wall_ms = 0.0;   // Wall clock for insert workloads (first start to last end)
    double search_only_qps = 0.0;  // n_searches / search_wall_ms
    double insert_only_tps = 0.0;  // n_inserts / insert_wall_ms
    double total_with_metrics_ms = 0.0;     // run_unified_workload wall including metric-thread join
    double search_with_metrics_qps = 0.0;   // n_searches / total_with_metrics_ms
    double metric_thread_join_ms = 0.0;     // end-of-phase wait for the metric thread

    // Hop distribution
    std::map<uint32_t, uint32_t> hop_distribution;

    // Query latency stats (in ms)
    double avg_query_latency_ms = 0.0;
    double min_query_latency_ms = 0.0;
    double p50_query_latency_ms = 0.0;
    double p95_query_latency_ms = 0.0;
    double p99_query_latency_ms = 0.0;
    double max_query_latency_ms = 0.0;

    // Per-query breakdown (in ms)
    double avg_server_per_query_ms = 0.0;         // Server processing (excl. response_queue)
    double avg_worker_queue_per_query_ms = 0.0;   // Work queue wait
    double avg_response_queue_per_query_ms = 0.0; // Response queue wait
    double avg_network_per_query_ms = 0.0;
    double avg_lock_per_query_ms = 0.0;           // Lock wait (included in server_ms)
    double avg_coord_total_per_query_ms = 0.0;    // Coordinator total (serialize+send+poll_to_recv+recv+deser)

    // Gap analysis: pre-first-hop and inter-hop timing (in ms, per query)
    double avg_pre_first_hop_per_query_ms = 0.0;  // Time from start to first hop send
    double avg_inter_hop_per_query_ms = 0.0;      // Cumulative inter-hop processing time

    // Per-hop breakdown (in ms)
    double avg_hop_latency_ms = 0.0;
    double avg_server_per_hop_ms = 0.0;           // Server processing (excl. response_queue)
    double avg_worker_queue_per_hop_ms = 0.0;     // Work queue wait
    double avg_response_queue_per_hop_ms = 0.0;   // Response queue wait
    double avg_network_per_hop_ms = 0.0;
    double avg_lock_per_hop_ms = 0.0;             // Lock wait (included in server_ms)
    double avg_coord_total_per_hop_ms = 0.0;      // Coordinator total (serialize+send+poll_to_recv+recv+deser)

    // Hop RTT stats (in ms)
    double avg_hop_rtt_ms = 0.0;
    double min_hop_rtt_ms = 0.0;
    double p50_hop_rtt_ms = 0.0;
    double p95_hop_rtt_ms = 0.0;
    double p99_hop_rtt_ms = 0.0;
    double max_hop_rtt_ms = 0.0;

    // Server detailed timing (in us, per request/hop)
    double avg_zmq_recv_us = 0.0;
    double avg_zmq_send_us = 0.0;
    double avg_deserialize_us = 0.0;
    double avg_serialize_us = 0.0;
    double avg_queue_depth = 0.0;
    double avg_resp_queue_wait_us = 0.0;
    double avg_io_processing_us = 0.0;  // I/O thread processing (recv_end → enqueue)
    double avg_worker_overhead_us = 0.0;  // Unmeasured gaps in worker thread (dispatch + pre-serialize)
    double avg_send_prep_us = 0.0;  // Gap: pop_time → send_start (routing + msg construction)

    // Coordinator timing per hop (in us)
    double coord_serialize_us = 0.0;
    double coord_send_us = 0.0;
    double coord_response_queue_us = 0.0;  // server-thread -> main-thread handoff
    // (no poll-to-recv field: the wait inside zmq::poll() is not part of the coordinator total)
    double coord_recv_us = 0.0;
    double coord_deserialize_us = 0.0;

    // Coordinator timing totals (in us)
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

    // Bottleneck analysis
    double overhead_per_hop_us = 0.0;
    double overhead_ratio = 0.0;

    // Expansion analysis
    double local_ratio = 0.0;           // local_expanded / top_W * 100
    uint64_t local_expanded = 0;
    uint64_t top_W = 0;
    double avg_neighbors_evaluated = 0.0;

    // Distance stats
    double avg_pq_dist = 0.0;
    double avg_full_dist = 0.0;

    // Per-hop distance computation (hop index -> avg dist)
    std::map<uint32_t, double> per_hop_dist_avg;
    uint32_t hop0_query_count = 0;
    double avg_hop0_dist = 0.0;
    uint32_t multihop_query_count = 0;
    uint32_t multihop_total_hops = 0;
    double avg_multihop_dist_per_hop = 0.0;

    // Server load
    std::vector<uint64_t> requests_per_server;

    // Lock metrics (cumulative totals)
    uint64_t total_shared_locks = 0;
    uint64_t total_unique_locks = 0;
    double avg_unique_lock_wait_us = 0.0;
    double avg_shared_lock_wait_us = 0.0;

    // Per-phase lock metrics (insert allocation, insert edges, backlinks, broadcasts)
    uint64_t insert_alloc_lock_count = 0;
    double avg_insert_alloc_lock_wait_us = 0.0;
    uint64_t insert_edge_lock_count = 0;
    double avg_insert_edge_lock_wait_us = 0.0;
    uint64_t backlink_lock_count = 0;
    double avg_backlink_lock_wait_us = 0.0;
    uint64_t broadcast_lock_count = 0;
    double avg_broadcast_lock_wait_us = 0.0;

    // Per-server lock stats: [server_idx] -> (shared_locks, unique_locks, shared_wait_us, unique_wait_us)
    std::vector<std::tuple<uint64_t, uint64_t, double, double>> per_server_lock_stats;

    // Per-server per-phase lock stats
    // [server_idx] -> (insert_alloc_count, insert_alloc_wait_us, insert_edge_count, insert_edge_wait_us,
    //                  backlink_count, backlink_wait_us, broadcast_count, broadcast_wait_us)
    struct PhaseSpecificLockStats {
        uint64_t insert_alloc_count = 0;
        double insert_alloc_wait_us = 0.0;
        uint64_t insert_edge_count = 0;
        double insert_edge_wait_us = 0.0;
        uint64_t backlink_count = 0;
        double backlink_wait_us = 0.0;
        uint64_t broadcast_count = 0;
        double broadcast_wait_us = 0.0;
        // Graph metrics
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

        // search internal sub-stage cumulative timings + counts.
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

    // Insert stats (optional)
    uint32_t insert_count = 0;
    double insert_throughput = 0.0;
    double avg_backlinks = 0.0;       // Backward-compatible pre-drain alias
    uint32_t total_backlinks = 0;     // Backward-compatible pre-drain alias
    uint32_t total_broadcasts = 0;    // Backward-compatible pre-drain alias

    // Deferred maintenance accounting. Insert JSON contains pre-drain counts;
    // explicit drain JSON carries drain counts. Summaries compare combined values.
    uint32_t maintenance_backlink_pre_drain_edges = 0;
    uint32_t maintenance_backlink_drain_exact_edges = 0;
    uint32_t maintenance_backlink_combined_edges = 0;
    double maintenance_backlink_combined_avg_per_insert = 0.0;
    bool maintenance_backlink_metric_complete = false;
    uint32_t maintenance_broadcast_pre_drain_acks = 0;
    uint32_t maintenance_broadcast_drain_acks = 0;
    uint32_t maintenance_broadcast_combined_acks = 0;
    bool maintenance_broadcast_metric_complete = false;

    // Workload generation and phase completion stats (for debugging)
    uint32_t workloads_generated = 0;
    uint32_t search_phase_completed = 0;
    uint32_t insert_rpc_completed = 0;
    uint32_t backlink_phase_completed = 0;
    uint32_t broadcast_phase_completed = 0;

    // Unified pipeline timing (in ms) - cumulative totals
    double search_phase_ms = 0.0;
    double insert_rpc_ms = 0.0;
    double backlink_ms = 0.0;
    double broadcast_ms = 0.0;

    // Per-insert timing breakdown (in ms) - averages per insert
    // Calculated from cumulative totals / insert_count
    double per_insert_search_ms = 0.0;
    double per_insert_rpc_ms = 0.0;
    double per_insert_backlink_ms = 0.0;
    double per_insert_broadcast_ms = 0.0;
    double per_insert_total_ms = 0.0;

    // Per-insert lock wait breakdown (in ms) - averages per insert
    // Calculated from phase lock metrics
    double per_insert_alloc_lock_wait_ms = 0.0;
    double per_insert_edge_lock_wait_ms = 0.0;
    double per_insert_backlink_lock_wait_ms = 0.0;
    double per_insert_broadcast_lock_wait_ms = 0.0;
    double per_insert_total_lock_wait_ms = 0.0;

    // INSERT RPC timing breakdown (per-request, average per insert, in ms)
    double per_insert_rpc_queue_wait_ms = 0.0;
    double per_insert_rpc_deserialize_ms = 0.0;
    double per_insert_rpc_alloc_lock_ms = 0.0;
    double per_insert_rpc_edge_lock_ms = 0.0;
    double per_insert_rpc_prune_ms = 0.0;
    double per_insert_rpc_alloc_memory_ms = 0.0;
    // Per-vector resize stats
    struct ResizeVectorStats {
        uint32_t count = 0;
        double total_ms = 0.0;
    };
    ResizeVectorStats resize_full_base;
    ResizeVectorStats resize_membership;
    ResizeVectorStats resize_adj;
    ResizeVectorStats resize_global_to_local;
    // Per-server resize stats
    static constexpr int MAX_SERVERS = 64;  // the supported server counts (2..64)
    struct PerServerResize {
        ResizeVectorStats full_base;
        ResizeVectorStats membership;
        ResizeVectorStats adj;
        ResizeVectorStats global_to_local;
    };
    PerServerResize resize_per_server[MAX_SERVERS] = {};
    int num_servers = 0;  // Actual number of servers used
    double per_insert_rpc_edge_memory_ms = 0.0;
    double per_insert_rpc_local_backlink_prune_ms = 0.0;
    double per_insert_rpc_network_rtt_ms = 0.0;  // Gap analysis: network RTT for INSERT RPC

    // P2P mode flag
    bool p2p_mode = false;

    // P2P route traces: per-query partition path
    // route_summary[route_string] = count (e.g. "0→2→1" = 523)
    std::map<std::string, uint32_t> route_summary;
    uint32_t route_traced_count = 0;  // queries with non-empty route
    // Per-route timing. Server processing/queue are same-process measurements;
    // transport residual is coordinator-clock path RTT minus measured server and
    // coordinator overhead. It intentionally replaces cross-server clock deltas.
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

    // Beam tracking for inserted nodes analysis
    uint64_t beam_total_entries = 0;
    uint64_t beam_inserted_entries = 0;
    uint64_t result_total_entries = 0;
    uint64_t result_inserted_entries = 0;

    // Server stats (total nodes per server after workload)
    std::vector<uint32_t> server_total_nodes;

    // Path of the --debug results NPY file (empty when none was written)
    std::string debug_npy_path;

    // ==========================================
    // Inserted Query Metrics
    // ==========================================
    // Metrics for queries that use inserted vectors
    struct InsertedQueryMetrics {
        uint32_t count = 0;              // Number of inserted queries executed
        double recall = 0.0;             // Inserted query recall (%)
        uint32_t baseline_count = 0;     // Number of baseline queries (= n_searches - count)
    };
    InsertedQueryMetrics inserted_query;

    // ==========================================
    // INSERT Search Metrics
    // ==========================================
    // Detailed metrics of the search phase of an INSERT
    // (per_insert_search_ms measures the total time only; this is the per-hop breakdown)
    struct InsertSearchMetrics {
        uint32_t count = 0;                          // completed INSERT searches
        uint32_t total_hops = 0;                     // total hops
        double avg_hops = 0.0;                       // average hops
        std::map<uint32_t, uint32_t> hop_distribution;  // hop distribution

        // Per-hop breakdown (sum of all hops / hop count)
        double avg_server_per_hop_ms = 0.0;          // per-hop server processing (excl. response_queue)
        double avg_worker_queue_per_hop_ms = 0.0;    // per-hop work queue wait
        double avg_response_queue_per_hop_ms = 0.0;  // per-hop response queue wait
        double avg_network_per_hop_ms = 0.0;         // per-hop network time
        double avg_lock_per_hop_ms = 0.0;            // per-hop lock wait (included in server_ms)
        double avg_coord_total_per_hop_ms = 0.0;     // per-hop coordinator total

        // Per-query breakdown (sum of all hops / query count) - for gap analysis
        double avg_server_per_query_ms = 0.0;        // per-query server processing (excl. response_queue)
        double avg_worker_queue_per_query_ms = 0.0;  // per-query work queue wait
        double avg_response_queue_per_query_ms = 0.0;// per-query response queue wait
        double avg_network_per_query_ms = 0.0;       // per-query network time
        double avg_lock_per_query_ms = 0.0;          // per-query lock wait
        double avg_coord_total_per_query_ms = 0.0;   // per-query coordinator total

        // Gap analysis: pre-first-hop and inter-hop timing (in ms, per query)
        double avg_pre_first_hop_per_query_ms = 0.0; // from start to the first hop send
        double avg_inter_hop_per_query_ms = 0.0;     // accumulated processing time between hops

        // Latency percentiles (query latency = end-to-end)
        double avg_latency_ms = 0.0;                 // average search latency
        double min_latency_ms = 0.0;
        double p50_latency_ms = 0.0;
        double p95_latency_ms = 0.0;
        double p99_latency_ms = 0.0;
        double max_latency_ms = 0.0;

        // Hop RTT percentiles (per-hop RTT)
        double avg_hop_rtt_ms = 0.0;
        double min_hop_rtt_ms = 0.0;
        double p50_hop_rtt_ms = 0.0;
        double p95_hop_rtt_ms = 0.0;
        double p99_hop_rtt_ms = 0.0;
        double max_hop_rtt_ms = 0.0;
    };
    InsertSearchMetrics insert_search;

    // ==========================================
    // Backlink Metrics (separate from insert)
    // ==========================================
    // Async backlink phase - separate top-level metric
    struct BacklinkMetrics {
        uint32_t count = 0;                // Total backlink count
        double avg_per_insert = 0.0;       // Average backlinks per insert
        double wall_clock_ms = 0.0;        // Wall clock time (max of parallel requests)
        double total_lock_wait_ms = 0.0;   // Cumulative lock wait (sum of all requests)
    };
    BacklinkMetrics backlink;

    // ==========================================
    // Broadcast Metrics (separate from insert)
    // ==========================================
    // Async broadcast phase - separate top-level metric
    struct BroadcastMetrics {
        uint32_t count = 0;                // Total broadcast count (n_inserts × n_servers)
        double wall_clock_ms = 0.0;        // Wall clock time (max of parallel requests)
        double total_lock_wait_ms = 0.0;   // Cumulative lock wait (sum of all requests)
    };
    BroadcastMetrics broadcast;

    // ==========================================
    // Partition Analysis
    // ==========================================
    // Partition distribution during the search
    struct PartitionAnalysis {
        std::map<uint32_t, uint32_t> entry_partition;    // partition distribution of the entry points
        std::map<uint32_t, uint32_t> final_partition;    // partition distribution of the final results
        // transitions[from][to] = count
        std::vector<std::vector<uint32_t>> transitions;
    };
    PartitionAnalysis partition_analysis;

    // ==========================================
    // Network Baseline (PING measurement)
    // ==========================================
    struct NetworkBaseline {
        bool measured = false;
        uint32_t ping_count = 0;  // Pings per server

        // Per-server stats (original format)
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

        // Overall stats (original format)
        uint32_t total_pings = 0;
        double overall_avg_us = 0.0;
        double overall_min_us = 0.0;
        double overall_p50_us = 0.0;
        double overall_p95_us = 0.0;
        double overall_p99_us = 0.0;
        double overall_max_us = 0.0;

        // Direct stats (same as overall, for compatibility)
        struct ModeStats {
            double avg_us = 0.0, min_us = 0.0, p50_us = 0.0, p95_us = 0.0;
            double p99_us = 0.0, max_us = 0.0;
        };
        ModeStats direct;
    };
    NetworkBaseline network_baseline;

    // Merge search stats
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
