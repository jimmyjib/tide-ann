#pragma once
/**
 * Coordinator types and global declarations.
 */

#include <vector>
#include <string>
#include <memory>
#include <queue>
#include <unordered_map>
#include <atomic>

#include <zmq.hpp>

#include "head_index.h"
#include "tide_types.h"
#include "owner_label.h"
#include "observability.h"

namespace tide {

// =============================================================================
// DaemonState - Persistent coordinator state
// =============================================================================
struct DaemonState {
    // Data
    std::vector<float> base;
    std::vector<float> queries_data;
    std::vector<uint32_t> membership;
    // The membership file loaded at startup. reset reloads it, like the servers
    // reload their load-time partitions, so moved owners return too.
    std::string membership_path;
    std::vector<int32_t> groundtruth;
    uint32_t n_base = 0, dim = 0;
    // The servers' storage mode, from the start-up cluster check: hybrid PQ
    // own-only storage (each server holds only its own nodes' vectors, so a MOVE
    // ships the moved node's vector to its target) or full replication.
    bool own_only_storage = false;
    uint32_t n_queries = 0;
    uint32_t gt_rows = 0, gt_cols = 0;

    // Head Index
    ann::HeadIndex head_index;
    // The head index loaded at start (the runtime's ghost sample); reset
    // restores it (recreate-head replaces head_index only).
    ann::HeadIndex head_index_at_load;

    // Servers
    std::vector<std::string> servers;
    uint32_t n_servers = 0;

    // ZMQ resources
    std::unique_ptr<zmq::context_t> context;
    std::vector<zmq::socket_t> sockets;
    std::vector<zmq::pollitem_t> poll_items;

    // Coordinator server threads (daemon lifetime)
    std::vector<std::unique_ptr<ann::ThreadSafeQueue<tide::ServerWorkItem>>> server_queues;
    std::unique_ptr<ann::ThreadSafeQueue<tide::ServerResponse>> main_response_queue;
    std::vector<std::thread> server_threads;
    std::atomic<bool> server_threads_running{false};

    // Server-side P2P origin state is keyed by workload ID and survives a
    // daemon command.  Never recycle IDs between commands.
    std::atomic<uint32_t> next_workload_id{0};
    // Request identity for GET_STATS rounds. Zero is reserved: a one-byte
    // request carries no id, and the collectors never accept id 0.
    std::atomic<uint64_t> next_stats_request_id{1};

    // Persistent PendingInsertManager (survives across workload commands)
    tide::PendingInsertManager pending_mgr;

    // Parameters (from initial command line)
    std::string data_dir;
    uint32_t L = 50, K = 10, W = 8;
    uint32_t insert_R = 64, insert_L = 64, insert_W = 8;
    float insert_alpha = 1.2f;
    uint32_t batch_size = 32;
    uint32_t dispatch_window = 256;  // 0 means use batch_size.
    bool p2p_mode = false;
    bool debug_mode = false;  // --debug: per-hop debug CSVs and the recall-miss analysis
    bool no_backlink = false;
    bool no_broadcast = false;
    bool insert_search_only = false;
    bool no_merge = false;  // diagnostic: skip the coordinator merge search
    bool trace_routes = true;  // diagnostic route telemetry; disable for throughput measurements

    // Insert tracking. Atomic: the command thread adds each insert command's
    // count (and reset clears it) while the long-lived server-reply thread
    // reads n_base + total_inserted for the snapshot queue bound on every
    // received message. No ordering is implied beyond the
    // value itself.
    std::atomic<uint32_t> total_inserted{0};
    std::vector<float> all_inserted_vectors;

    // Pending async state (moved from UnifiedContext after workload ends).
    // After an insert workload, some items may still be in the INSERT_PENDING
    // phase. The drain settles them: the coordinator sends DRAIN to the
    // servers and polls for their ACKs.
    std::unordered_map<uint32_t, uint32_t> pending_backlink_map;   // batch_id → workload_idx
    std::unordered_map<uint32_t, uint32_t> pending_broadcast_map;  // node_id → workload_idx
    std::unordered_map<uint32_t, uint32_t> pending_insert_response_map;  // request_id → workload_idx
    std::vector<tide::WorkloadItem> pending_workloads;

    // Insert maintenance accounting since daemon start. Never reset: the
    // servers' backlink counters (GET_STATS backlink_edges_requested) are not
    // reset either, so once every queued backlink batch is applied the sum of
    // the servers' counters equals backlink_edges_sent. An orphan is a
    // maintenance reply (insert response, backlink or broadcast ACK) that has
    // no pending entry when it arrives: it came after the drain barrier had
    // already settled its insert, or twice. Only the daemon thread touches it.
    struct MaintenanceAccounting {
        uint64_t backlink_edges_sent = 0;
        uint64_t backlink_batches_sent = 0;
        uint64_t orphan_insert_responses = 0;
        uint64_t orphan_backlink_acks = 0;
        uint64_t orphan_broadcast_acks = 0;
    } maintenance;

    // Network baseline results (stored from ping command)
    struct BaselineResult {
        uint32_t ping_count = 0;  // Pings per server

        // Per-server stats (original format)
        struct PerServerStats {
            uint32_t id = 0;
            uint32_t count = 0;
            double avg_us = 0, min_us = 0, p50_us = 0, p95_us = 0;
            double p99_us = 0, max_us = 0;
        };
        std::vector<PerServerStats> servers;

        // Overall stats (original format)
        uint32_t total_pings = 0;
        double overall_avg_us = 0, overall_min_us = 0, overall_p50_us = 0;
        double overall_p95_us = 0, overall_p99_us = 0, overall_max_us = 0;

        // Direct stats (same as overall, for compatibility)
        struct ModeStats {
            double avg_us = 0, min_us = 0, p50_us = 0, p95_us = 0;
            double p99_us = 0, max_us = 0;
        };
        ModeStats direct;

        bool measured = false;
    } baseline;

    // Client API socket (ZMQ REP for request-response)
    std::unique_ptr<zmq::socket_t> client_socket;
    std::string client_endpoint = "tcp://*:29699";
};

// =============================================================================
// Global variables (defined in tide_coordinator.cpp)
// =============================================================================
extern DaemonState g_daemon;
extern std::ostream* g_current_out;

// =============================================================================
// Function declarations - daemon_state.cpp
// =============================================================================
void reset_socket_state();
void daemon_output(const std::string& msg);
void daemon_output_line(const std::string& msg);
std::vector<uint32_t> get_partition_node_counts();
std::string generate_debug_log_path(const std::string& mode, uint32_t count);
void stop_daemon_server_threads();
// The cluster the coordinator starts with, from one GET_STATS reply per server
// (rank order): every server must answer as its rank, run the same storage mode
// (full replication or hybrid PQ own-only: an own-only server drops every beam
// entry a full-replication server scored without the exact flag, so a mixed
// cluster would lose results), with the coordinator's vector dimension (the
// dimension of its base.npy; an own-only server's comes from its PQ pivots), and
// own-only servers with one PQ code size. False with a message otherwise;
// `summary` names the agreed mode, dimension and code size.
bool cluster_storage_consistent(const std::vector<ann::StatsResponseHeader>& replies, uint32_t servers,
                                uint32_t dim, std::string* summary, std::string* error);

// =============================================================================
// Function declarations - metrics.cpp
// =============================================================================
void query_server_lock_metrics(ann::ExperimentResult& result);
bool query_server_stats(std::vector<ann::StatsResponseHeader>& snapshots,
                        bool consume_interval_max);
void populate_detailed_metrics(const UnifiedContext& ctx, ann::ExperimentResult& result);

// =============================================================================
// Function declarations - workload_runners.cpp
// =============================================================================
void run_search_workload(uint32_t n_queries, const std::string& gt_path);
// Extended variant supporting --query-file and --self-recall. When query_file
// is non-empty, loads queries from PATH (skipping query_offset rows) and uses
// those instead of g_daemon.queries_data. When self_recall is true, recall is
// computed against expected_id = (self_recall_base_id ? self_recall_base_id
// : g_daemon.n_base) + query_index — gt_path is ignored.
// With self_ids (recall --self-ids), query row i of query_file is the vector
// of node self_ids[i]: the expected id of that query, and the result JSON gets a
// "self_query" block (found in its own top-K at distance exactly 0.0, per query).
// search_l: this command's search list size (recall --search-l; 0 = the
// start-up L). The reply's "L" names the size used.
void run_search_workload_ex(uint32_t n_queries, const std::string& gt_path,
                            const std::string& query_file, uint32_t query_offset,
                            bool self_recall, uint32_t self_recall_base_id,
                            const std::vector<uint32_t>& self_ids = {}, uint32_t search_l = 0);
void run_insert_workload(uint32_t n_insert, const std::string& insert_file = "",
                         uint32_t start_id = 0);
void run_mixed_workload(uint32_t n_queries, uint32_t insert_ratio, const std::string& gt_path,
                        const std::string& insert_gt_path = "",
                        const std::string& insert_file = "", uint32_t insert_start_id = 0,
                        double inserted_query_ratio = 0.10);

// =============================================================================
// Function declarations - admin_commands.cpp
// =============================================================================
bool send_admin_command(ann::MessageType cmd_type);
bool process_daemon_command(const std::string& cmd);
// Every admin command process_daemon_command accepts (the start-up banner and the unknown-command reply).
inline constexpr const char* kAdminCommands =
    "baseline, recall, insert, mixed, drain, recreate-head, status, stats, ping, reset, membership-check, "
    "migration-apply, migration-accept, migration-rollback, controller-start, controller-status, "
    "controller-stop, controller-abort, shutdown, quit, exit";
std::string run_ping_baseline(uint32_t count, uint32_t pipelined_batch = 0);

} // namespace tide
