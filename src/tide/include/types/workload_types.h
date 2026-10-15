#pragma once
/**
 * Unified Workload Types
 *
 * Types for the unified async pipeline that handles both search and insert.
 */

#include <vector>
#include <unordered_map>
#include <map>
#include <queue>
#include <deque>
#include <chrono>
#include <cstdint>
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>
#include <string>
#include <zmq.hpp>
#include "state_protocol.h"
#include "types/timing_types.h"
#include "types/pending_insert.h"
#include "types/search_context.h"
#include "thread_pool.h"  // For ThreadSafeQueue
#include "metrics/experiment_result.h"

namespace tide {

// Producer-consumer queue carrying workload indices that have completed
// head-index BF and are ready for dispatch. Producers = head_bf helper
// threads (workload_generator). Consumer = main pipeline loop
// (pipeline_core::refill_workloads). Mutex-based; contention is low: helpers
// push once per head-index search and main pops only when
// in_flight < dispatch_window.
struct ReadyQueue {
    std::mutex mu;
    std::deque<uint32_t> q;
    std::atomic<uint32_t> total_pushed{0};

    void push(uint32_t i) {
        std::lock_guard<std::mutex> g(mu);
        q.push_back(i);
        total_pushed.fetch_add(1, std::memory_order_release);
    }
    bool try_pop(uint32_t& out) {
        std::lock_guard<std::mutex> g(mu);
        if (q.empty()) return false;
        out = q.front();
        q.pop_front();
        return true;
    }
    size_t size() {
        std::lock_guard<std::mutex> g(mu);
        return q.size();
    }
};

using namespace std::chrono;
using ann::BeamEntry;
using ann::CrossPartitionEdge;

/**
 * ServerWorkItem: Work item sent from main thread to per-server thread.
 */
struct ServerWorkItem {
    std::vector<char> request_data;                    // Serialized SEARCH_REQUEST
    uint32_t workload_id = 0;
    uint32_t target_server = 0;
};

/**
 * ServerResponse: Response from per-server thread back to main thread.
 */
struct ServerResponse {
    uint32_t server_idx = 0;
    std::vector<char> response_data;
    steady_clock::time_point send_time;                // Pre-serialize boundary for RTT accounting
    steady_clock::time_point recv_time;
    steady_clock::time_point enqueue_time;
    double coord_send_us = 0;
    double coord_recv_us = 0;
};

/**
 * InsertPhase: Tracks the current phase of an insert operation
 */
enum class InsertPhase : uint8_t {
    SEARCH_PENDING = 0,
    INSERT_PENDING = 1,  // Insert out-edge + backlink + broadcast
    DONE = 2
};

/**
 * WorkloadType: Distinguishes search vs insert workloads
 */
enum class WorkloadType : uint8_t {
    SEARCH = 0,
    INSERT = 1
};

/**
 * MetricEventType: Types of events sent to the metric thread
 */
enum class MetricEventType : uint8_t {
    SEARCH_DONE = 0,        // Search workload completed
    SEARCH_HOP = 1,         // Search hop state logging
    INSERT_SEARCH_DONE = 2, // Insert search phase completed
    INSERT_HOP = 3,         // Insert search hop state
    SHUTDOWN = 255          // Sentinel for thread termination
};

/**
 * MetricEvent: Event sent from search thread to metric thread
 *
 * This allows post-processing (logging, merge, tracking) to happen
 * off the critical path, improving timing measurement accuracy.
 */
struct MetricEvent {
    MetricEventType type = MetricEventType::SEARCH_DONE;
    uint32_t workload_idx = 0;

    // HOP events: beam snapshot (copied before beam is moved)
    std::vector<ann::BeamEntry> beam_snapshot;
    uint32_t hop_num = 0;
    uint32_t from_partition = 0;
    uint32_t to_partition = 0;

    // SEARCH_DONE: GT reference for recall logging
    const int32_t* gt_row = nullptr;
    uint32_t gt_cols = 0;

    // --debug only: the owner label of each id the log line names (the result
    // neighbours of a DONE event, the beam entries of a HOP event), resolved on
    // the pipeline thread, which owns the membership vector and resizes it.
    std::vector<uint32_t> labels;

    MetricEvent() = default;
    MetricEvent(MetricEventType t, uint32_t idx) : type(t), workload_idx(idx) {}
};

/**
 * WorkloadItem: Unified structure for tracking both search and insert workloads
 */
struct WorkloadItem {
    // Common fields
    WorkloadType type;
    uint32_t id;
    uint32_t query_index = UINT32_MAX;  // For SEARCH: actual index into queries array (for GT lookup)
    // Position in this search-only command (not the cycled source row); set
    // by the workload generator, read by no other code at present.
    uint32_t query_ordinal = UINT32_MAX;
    std::vector<float> query;
    std::vector<BeamEntry> beam;
    std::vector<uint32_t> seen_nodes;  // Visited set for cross-partition hop
    uint32_t current_partition;
    uint32_t entry_partition;
    uint32_t hops;
    bool done;

    // Search results
    std::vector<uint32_t> result_neighbors;
    std::vector<float> result_distances;

    // Insert-specific fields
    InsertPhase insert_phase;
    uint32_t insert_sequence;  // INSERT-only sequence number (0, 1, 2, ... for inserts)
    uint32_t assigned_id;
    std::vector<uint32_t> search_neighbors;
    std::vector<float> search_distances;
    std::vector<CrossPartitionEdge> pending_backlinks;
    uint32_t backlinks_sent;
    uint32_t backlinks_acked;
    bool backlink_done = false;
    uint32_t broadcasts_acked;
    bool broadcast_done = false;
    uint32_t target_partition;
    bool insert_response_seen = false;

    // Timing
    steady_clock::time_point start_time;
    steady_clock::time_point phase_start_time;
    steady_clock::time_point send_time;
    double search_us = 0;
    double insert_us = 0;
    double backlink_us = 0;
    double broadcast_us = 0;

    double last_coord_serialize_us = 0;
    double last_coord_send_us = 0;

    std::vector<HopLatency> hop_latencies;
    uint32_t total_distance_computations = 0;
    double query_latency_us = 0;

    // P2P cumulative metrics (from final DONE response, used when hop_latencies is empty)
    uint32_t cumulative_queue_us = 0;
    uint32_t cumulative_processing_us = 0;

    // P2P route trace (partition IDs + per-hop timing)
    std::vector<ann::RouteHop> route;

    // Gap analysis: timing breakdown for bottom-up vs top-down matching
    double pre_first_hop_us = 0;  // Time from start to first hop send
    double inter_hop_us = 0;      // Cumulative inter-hop processing time
    steady_clock::time_point last_recv_time;  // For inter-hop calculation

    // Inserted node query support
    bool is_inserted_query = false;     // Is this a query for an inserted node?
    uint32_t insert_gt_index = 0;       // Index into insert_groundtruth (= insert_sequence of source)

    WorkloadItem()
        : type(WorkloadType::SEARCH), id(0), current_partition(0), hops(0), done(false),
          insert_phase(InsertPhase::SEARCH_PENDING), insert_sequence(0), assigned_id(0),
          backlinks_sent(0), backlinks_acked(0), broadcasts_acked(0), target_partition(0) {}
};

/**
 * UnifiedContext: Context structure for unified pipeline execution
 */
struct UnifiedContext {
    // === Workload tracking ===
    std::vector<WorkloadItem> workloads;
    std::unordered_map<uint32_t, uint32_t> in_flight;

    // === Input parameters ===
    uint32_t n_searches;
    uint32_t n_inserts;
    uint32_t total_workloads;
    uint32_t insert_ratio;

    // Search parameters
    uint32_t L;
    uint32_t K;
    uint32_t W;
    uint32_t effective_batch_size;
    uint32_t dispatch_window = 0;
    bool p2p_mode = false;
    bool debug_mode = false;
    bool no_backlink = false;
    bool no_broadcast = false;
    bool insert_search_only = false;
    bool no_merge = false;
    bool trace_routes = true;

    // Insert parameters
    uint32_t insert_R;
    uint32_t insert_L;
    uint32_t insert_W;
    float insert_alpha;
    uint32_t insert_start_id = 0;

    // Data references
    uint32_t dim;
    uint32_t n_servers;
    uint32_t n_base;
    const float* queries_data;
    uint32_t n_available_queries = 0;  // Cycle queries_data when n_searches > this
    const float* insert_vectors;
    const uint32_t* membership;
    std::vector<uint32_t>* membership_vec = nullptr;
    const int32_t* groundtruth;
    uint32_t gt_cols;

    // Self-recall mode: SEARCH workloads use queries from --query-file (e.g., the
    // same insert.npy used during a prior insert phase). Recall is
    // computed as 1.0 iff (self_recall_base_id + query_index) appears in
    // result_neighbors[0..K]. Skips GT file entirely.
    bool self_recall = false;
    uint32_t self_recall_base_id = 0;
    // Self-query mode (recall --self-ids FILE): query row i is the vector of
    // node self_ids[i], the expected id (instead of base + query_index).
    std::vector<uint32_t> self_ids;

    // ZMQ resources (non-owning)
    std::vector<zmq::socket_t>* sockets;
    std::vector<zmq::pollitem_t>* poll_items;

    // Head index reference
    const void* head_index;

    // PendingInsertManager
    PendingInsertManager* pending_mgr = nullptr;

    // Parallel head-index brute force + pipelined dispatch.
    // generate_mixed_workloads_async() spawns N_HELPER threads that compute
    // head BF for each workload and push the index to `ready_q` as soon as it
    // is ready. The main pipeline loop (refill_workloads in pipeline_core)
    // pops from `ready_q` instead of incrementing a sequential index, so
    // dispatch begins immediately and overlaps with the rest of the head BF.
    // `head_bf_threads` are joined by run_unified_workload after the main loop
    // terminates.
    ReadyQueue ready_q;
    std::vector<std::thread> head_bf_threads;

    // === Output metrics ===
    ann::ServingAccounting serving_accounting;
    double elapsed_ms = 0;
    double avg_recall = 0;
    double search_qps = 0;
    double insert_throughput = 0;

    // Wall clock separation
    std::chrono::steady_clock::time_point search_first_start;
    std::chrono::steady_clock::time_point search_last_end;
    std::chrono::steady_clock::time_point insert_first_start;
    std::chrono::steady_clock::time_point insert_last_end;
    bool search_started = false;
    bool insert_started = false;
    double search_wall_ms = 0;
    double insert_wall_ms = 0;

    // Timing breakdown
    TimingMetrics timing;
    // (no poll-to-recv timing: the wait inside zmq::poll() is not part of the coordinator total)
    double total_search_phase_us = 0;
    double total_insert_rpc_us = 0;
    double total_backlink_us = 0;
    double total_broadcast_us = 0;

    // INSERT RPC timing breakdown
    double total_insert_queue_wait_us = 0;
    double total_insert_deserialize_us = 0;
    double total_insert_alloc_lock_us = 0;
    double total_insert_edge_lock_us = 0;
    double total_insert_prune_us = 0;
    double total_insert_alloc_memory_us = 0;

    struct ResizeStatsAggregated {
        uint32_t count = 0;
        double time_us = 0;
    };
    ResizeStatsAggregated resize_full_base;
    ResizeStatsAggregated resize_membership;
    ResizeStatsAggregated resize_adj;
    ResizeStatsAggregated resize_global_to_local;

    static constexpr int MAX_SERVERS = 64;  // the supported server counts (2..64)
    struct PerServerResizeStats {
        ResizeStatsAggregated full_base;
        ResizeStatsAggregated membership;
        ResizeStatsAggregated adj;
        ResizeStatsAggregated global_to_local;
    };
    PerServerResizeStats resize_per_server[MAX_SERVERS] = {};
    double total_insert_edge_memory_us = 0;
    double total_insert_local_backlink_prune_us = 0;
    double total_insert_network_rtt_us = 0;  // Network RTT for INSERT RPC (gap analysis)

    // Gap analysis: pre-first-hop and inter-hop timing
    double total_pre_first_hop_us = 0;       // Search: time before first hop
    double total_inter_hop_us = 0;           // Search: cumulative inter-hop processing
    double total_insert_search_pre_first_hop_us = 0;  // Insert search phase
    double total_insert_search_inter_hop_us = 0;

    // Hop metrics
    double avg_hops = 0;
    double avg_latency_ms = 0;
    double avg_hop_latency_ms = 0;

    // Counters
    uint32_t searches_completed = 0;
    uint32_t inserts_completed = 0;
    uint32_t total_backlinks_sent = 0;
    uint32_t total_backlinks_acked = 0;
    uint32_t total_broadcasts_acked = 0;

    // Phase completion counters
    uint32_t insert_search_phase_completed = 0;
    uint32_t insert_rpc_phase_completed = 0;
    uint32_t backlink_phase_completed = 0;
    uint32_t broadcast_phase_completed = 0;

    std::vector<uint64_t> requests_per_server;
    std::vector<double> hop_rtt_us;
    std::vector<SearchContext::ServerLockMetrics> server_lock_metrics;

    // Inserted vector tracking
    std::map<uint32_t, std::vector<float>> inserted_vectors;
    std::map<uint32_t, uint32_t> inserted_partitions;

    // Beam tracking
    uint32_t insert_threshold_id = 0;
    uint64_t beam_total_entries = 0;
    uint64_t beam_inserted_entries = 0;
    uint64_t result_total_entries = 0;
    uint64_t result_inserted_entries = 0;

    // Recall miss analysis
    bool debug_recall = false;
    uint64_t total_queries_analyzed = 0;
    uint64_t queries_with_miss = 0;
    uint64_t total_gt_missed = 0;
    uint64_t total_wrong_results = 0;
    uint64_t wrong_results_inserted = 0;
    uint64_t gt_missed_displaced = 0;
    uint64_t inserted_truly_closer = 0;

    struct RecallMissDetail {
        uint32_t query_idx;
        std::vector<uint32_t> results;
        std::vector<uint32_t> groundtruth;
        std::vector<uint32_t> missed_gt;
        std::vector<uint32_t> wrong_in_results;
        double recall;
    };
    std::vector<RecallMissDetail> miss_details;

    // Async phase tracking (ID-based matching, not FIFO)
    // backlink: batch_id (= workload_idx) → workload_idx
    // broadcast: assigned_id (= node_id) → workload_idx
    // insert response: request_id → workload_idx, used for exact async backlinks
    std::unordered_map<uint32_t, uint32_t> backlink_pending_map;
    std::unordered_map<uint32_t, uint32_t> broadcast_pending_map;
    std::unordered_map<uint32_t, uint32_t> insert_response_pending_map;

    // ==========================================
    // Inserted Node Query Support
    // ==========================================
    // Pool of completed inserts available for search queries
    struct CompletedInsert {
        uint32_t insert_sequence;       // 0, 1, 2, ... (insert order)
        std::vector<float> vector;      // Inserted vector
    };
    std::queue<CompletedInsert> completed_inserts_pool;

    // Insert GT (pre-computed file)
    std::vector<int32_t> insert_groundtruth;  // [n_inserts * K], row-major
    uint32_t insert_gt_rows = 0;              // n_inserts
    uint32_t insert_gt_cols = 0;              // K

    // Recall tracking (separated)
    uint32_t baseline_queries_completed = 0;
    double baseline_recall_sum = 0.0;
    uint32_t inserted_queries_completed = 0;
    double inserted_recall_sum = 0.0;

    // Rolling Recall@K scored at completion time for the runtime controller's
    // optional recall guard (window deltas of these two counters). Baseline
    // queries only; inserted and self-recall queries keep their end-of-run
    // accounting above. Zero when the command has no ground truth.
    double rolling_recall_sum = 0.0;
    uint64_t rolling_recall_count = 0;

    // Set when the main loop ended before every workload completed: the runtime
    // controller froze dispatch after an ownership transaction it could not
    // confirm, and nothing was left in flight. The runner reports it as an error.
    std::string aborted_reason;

    // Merge-search accounting. These counters make the pending-insert overlay
    // auditable in mixed workloads where pending inserts may lag visibility.
    uint64_t merge_search_calls = 0;
    uint64_t merge_search_pending_total = 0;
    uint64_t merge_search_pending_eligible = 0;
    uint64_t merge_search_pending_topk = 0;
    uint64_t merge_search_result_changed = 0;
    uint64_t merge_search_wall_us = 0;

    // Settings
    double inserted_query_ratio = 0.10;  // 10% of searches use inserted vectors
    bool enable_inserted_query = false;  // Enabled only when --insert-gt is provided

    // Metric thread support
    // Metric thread handles: log_search_result, log_hop_state, merge_into_results, beam tracking
    // This separates I/O-heavy operations from the critical network path.
    std::unique_ptr<ann::ThreadSafeQueue<MetricEvent>> metric_queue;
    std::atomic<bool> metric_thread_running{false};
    std::thread metric_thread;
    double metric_thread_join_ms = 0.0;
    double elapsed_with_metrics_ms = 0.0;

    // === Coordinator multi-thread ===
    uint32_t batch_per_server = 8;
};

}  // namespace tide
