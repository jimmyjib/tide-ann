// Workload items and the per-command context of the coordinator pipeline.
#pragma once

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
#include "thread_pool.h"
#include "metrics/workload_result.h"

namespace tide {

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

// main thread -> per-server thread
struct ServerWorkItem {
    std::vector<char> request_data;
    uint32_t workload_id = 0;
    uint32_t target_server = 0;
};

// per-server thread -> main thread
struct ServerResponse {
    uint32_t server_idx = 0;
    std::vector<char> response_data;
    steady_clock::time_point send_time;                // before serialization (RTT start)
    steady_clock::time_point recv_time;
    steady_clock::time_point enqueue_time;
    double coord_send_us = 0;
    double coord_recv_us = 0;
};

enum class InsertPhase : uint8_t {
    SEARCH_PENDING = 0,
    INSERT_PENDING = 1,  // insert, backlinks and broadcasts
    DONE = 2
};

enum class WorkloadType : uint8_t {
    SEARCH = 0,
    INSERT = 1
};

enum class MetricEventType : uint8_t {
    SEARCH_DONE = 0,
    INSERT_SEARCH_DONE = 2,
    SHUTDOWN = 255
};

// Event for the metric thread.
struct MetricEvent {
    MetricEventType type = MetricEventType::SEARCH_DONE;
    uint32_t workload_idx = 0;

    // SEARCH_DONE: final beam, for the pending-insert merge
    std::vector<ann::BeamEntry> beam_snapshot;
    // INSERT_SEARCH_DONE: entry server of the insert's search
    uint32_t from_partition = 0;

    // SEARCH_DONE: ground-truth row
    const int32_t* gt_row = nullptr;
    uint32_t gt_cols = 0;

    // --debug: owner label of each id the log line names, resolved on the pipeline thread
    std::vector<uint32_t> labels;

    MetricEvent() = default;
    MetricEvent(MetricEventType t, uint32_t idx) : type(t), workload_idx(idx) {}
};

struct WorkloadItem {
    WorkloadType type;
    uint32_t id;
    uint32_t query_index = UINT32_MAX;  // query row, also the ground-truth row
    // position in this command
    uint32_t query_ordinal = UINT32_MAX;
    std::vector<float> query;
    std::vector<BeamEntry> beam;
    std::vector<uint32_t> seen_nodes;  // visited set
    uint32_t current_partition;
    uint32_t entry_partition;
    uint32_t hops;
    bool done;

    std::vector<uint32_t> result_neighbors;
    std::vector<float> result_distances;

    // inserts
    InsertPhase insert_phase;
    uint32_t insert_sequence;  // 0, 1, 2, ... in insert order
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

    // P2P: cumulative timing from the final reply
    uint32_t cumulative_queue_us = 0;
    uint32_t cumulative_processing_us = 0;

    // route: servers visited, with per-hop timing
    std::vector<ann::RouteHop> route;

    double pre_first_hop_us = 0;  // start to first send
    double inter_hop_us = 0;      // coordinator time between hops
    steady_clock::time_point last_recv_time;

    bool is_inserted_query = false;     // queries an inserted vector
    uint32_t insert_gt_index = 0;       // row of insert_groundtruth

    WorkloadItem()
        : type(WorkloadType::SEARCH), id(0), current_partition(0), hops(0), done(false),
          insert_phase(InsertPhase::SEARCH_PENDING), insert_sequence(0), assigned_id(0),
          backlinks_sent(0), backlinks_acked(0), broadcasts_acked(0), target_partition(0) {}
};

// Per-command state of the coordinator pipeline.
struct UnifiedContext {
    std::vector<WorkloadItem> workloads;
    std::unordered_map<uint32_t, uint32_t> in_flight;

    uint32_t n_searches;
    uint32_t n_inserts;
    uint32_t total_workloads;
    uint32_t insert_ratio;

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

    uint32_t insert_R;
    uint32_t insert_L;
    uint32_t insert_W;
    float insert_alpha;
    uint32_t insert_start_id = 0;

    uint32_t dim;
    uint32_t n_servers;
    uint32_t n_base;
    const float* queries_data;
    uint32_t n_available_queries = 0;  // queries repeat beyond this
    const float* insert_vectors;
    const uint32_t* membership;
    std::vector<uint32_t>* membership_vec = nullptr;
    const int32_t* groundtruth;
    uint32_t gt_cols;

    // self-recall: the expected id of query i is self_recall_base_id + i (no ground-truth file)
    bool self_recall = false;
    uint32_t self_recall_base_id = 0;
    // recall --self-ids: query row i is the vector of node self_ids[i]
    std::vector<uint32_t> self_ids;

    // not owned
    std::vector<zmq::socket_t>* sockets;
    std::vector<zmq::pollitem_t>* poll_items;

    const void* head_index;

    PendingInsertManager* pending_mgr = nullptr;

    ReadyQueue ready_q;
    std::vector<std::thread> head_bf_threads;

    ann::ServingAccounting serving;
    double elapsed_ms = 0;
    double avg_recall = 0;
    double search_qps = 0;
    double insert_throughput = 0;

    std::chrono::steady_clock::time_point search_first_start;
    std::chrono::steady_clock::time_point search_last_end;
    std::chrono::steady_clock::time_point insert_first_start;
    std::chrono::steady_clock::time_point insert_last_end;
    bool search_started = false;
    bool insert_started = false;
    double search_wall_ms = 0;
    double insert_wall_ms = 0;

    TimingMetrics timing;
    double total_search_phase_us = 0;
    double total_insert_rpc_us = 0;
    double total_backlink_us = 0;
    double total_broadcast_us = 0;

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
    double total_insert_network_rtt_us = 0;

    double total_pre_first_hop_us = 0;
    double total_inter_hop_us = 0;
    double total_insert_search_pre_first_hop_us = 0;
    double total_insert_search_inter_hop_us = 0;

    double avg_hops = 0;
    double avg_latency_ms = 0;
    double avg_hop_latency_ms = 0;

    uint32_t searches_completed = 0;
    uint32_t inserts_completed = 0;
    uint32_t total_backlinks_sent = 0;
    uint32_t total_backlinks_acked = 0;
    uint32_t total_broadcasts_acked = 0;

    uint32_t insert_search_phase_completed = 0;
    uint32_t insert_rpc_phase_completed = 0;
    uint32_t backlink_phase_completed = 0;
    uint32_t broadcast_phase_completed = 0;

    std::vector<uint64_t> requests_per_server;
    std::vector<double> hop_rtt_us;
    std::vector<SearchContext::ServerLockMetrics> server_lock_metrics;

    std::map<uint32_t, std::vector<float>> inserted_vectors;
    std::map<uint32_t, uint32_t> inserted_partitions;

    uint32_t insert_threshold_id = 0;
    uint64_t beam_total_entries = 0;
    uint64_t beam_inserted_entries = 0;
    uint64_t result_total_entries = 0;
    uint64_t result_inserted_entries = 0;

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

    // pending maintenance by id: backlink batch, broadcast node id, insert request id
    std::unordered_map<uint32_t, uint32_t> backlink_pending_map;
    std::unordered_map<uint32_t, uint32_t> broadcast_pending_map;
    std::unordered_map<uint32_t, uint32_t> insert_response_pending_map;

    // completed inserts that searches can query
    struct CompletedInsert {
        uint32_t insert_sequence;
        std::vector<float> vector;
    };
    std::queue<CompletedInsert> completed_inserts_pool;

    std::vector<int32_t> insert_groundtruth;  // [rows x cols], row-major
    uint32_t insert_gt_rows = 0;
    uint32_t insert_gt_cols = 0;

    uint32_t baseline_queries_completed = 0;
    double baseline_recall_sum = 0.0;
    uint32_t inserted_queries_completed = 0;
    double inserted_recall_sum = 0.0;

    // rolling Recall@K of baseline queries, for the recall guard
    double rolling_recall_sum = 0.0;
    uint64_t rolling_recall_count = 0;

    // set when the command ended before every workload completed
    std::string aborted_reason;

    // pending-insert merge counters
    uint64_t merge_search_calls = 0;
    uint64_t merge_search_pending_total = 0;
    uint64_t merge_search_pending_eligible = 0;
    uint64_t merge_search_pending_topk = 0;
    uint64_t merge_search_result_changed = 0;
    uint64_t merge_search_wall_us = 0;

    double inserted_query_ratio = 0.10;  // share of searches that query inserted vectors
    bool enable_inserted_query = false;  // only with --insert-gt

    // metric thread: CSV logs, pending-insert merge and beam tracking
    std::unique_ptr<ann::ThreadSafeQueue<MetricEvent>> metric_queue;
    std::atomic<bool> metric_thread_running{false};
    std::thread metric_thread;
    double metric_thread_join_ms = 0.0;
    double elapsed_with_metrics_ms = 0.0;

    uint32_t batch_per_server = 8;
};

}  // namespace tide
