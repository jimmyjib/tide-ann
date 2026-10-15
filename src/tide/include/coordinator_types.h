#pragma once

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

// State that lives as long as the coordinator process.
struct DaemonState {
    std::vector<float> base;
    std::vector<float> queries_data;
    std::vector<uint32_t> membership;
    // membership file loaded at start (reset reloads it)
    std::string membership_path;
    std::vector<int32_t> groundtruth;
    uint32_t n_base = 0, dim = 0;
    // storage mode from the start-up check: PQ mode (a MOVE sends the vector) or full vectors
    bool own_only_storage = false;
    uint32_t n_queries = 0;
    uint32_t gt_rows = 0, gt_cols = 0;

    ann::HeadIndex head_index;
    // head index loaded at start from ghost/ghost_indices.npy (reset restores it)
    ann::HeadIndex head_index_at_load;

    std::vector<std::string> servers;
    uint32_t n_servers = 0;

    std::unique_ptr<zmq::context_t> context;
    std::vector<zmq::socket_t> sockets;
    std::vector<zmq::pollitem_t> poll_items;

    // server threads live as long as the daemon
    std::vector<std::unique_ptr<ann::ThreadSafeQueue<tide::ServerWorkItem>>> server_queues;
    std::unique_ptr<ann::ThreadSafeQueue<tide::ServerResponse>> main_response_queue;
    std::vector<std::thread> server_threads;
    std::atomic<bool> server_threads_running{false};

    // workload ids are never reused across commands
    std::atomic<uint32_t> next_workload_id{0};
    // GET_STATS request id (0 is reserved)
    std::atomic<uint64_t> next_stats_request_id{1};

    // pending inserts, kept across commands
    tide::PendingInsertManager pending_mgr;

    // start-up parameters
    std::string data_dir;
    uint32_t L = 50, K = 10, W = 8;
    uint32_t insert_R = 64, insert_L = 64, insert_W = 8;
    float insert_alpha = 1.2f;
    uint32_t batch_size = 32;
    uint32_t dispatch_window = 256;  // 0: batch_size
    bool p2p_mode = false;
    bool debug_mode = false;  // --debug
    bool no_backlink = false;
    bool no_broadcast = false;
    bool insert_search_only = false;
    bool no_merge = false;  // skip the pending-insert merge
    bool trace_routes = true;  // route traces
    bool result_digest = false;  // recall/mixed --digest

    // atomic: the server-reply thread reads n_base + total_inserted
    std::atomic<uint32_t> total_inserted{0};
    std::vector<float> all_inserted_vectors;

    // pending maintenance left by the last insert command, settled by the next drain
    std::unordered_map<uint32_t, uint32_t> pending_backlink_map;   // batch_id -> workload_idx
    std::unordered_map<uint32_t, uint32_t> pending_broadcast_map;  // node_id -> workload_idx
    std::unordered_map<uint32_t, uint32_t> pending_insert_response_map;  // request_id -> workload_idx
    std::vector<tide::WorkloadItem> pending_workloads;

    // Maintenance totals since start (never reset). An orphan is a reply with no pending entry.
    struct MaintenanceAccounting {
        uint64_t backlink_edges_sent = 0;
        uint64_t backlink_batches_sent = 0;
        uint64_t orphan_insert_responses = 0;
        uint64_t orphan_backlink_acks = 0;
        uint64_t orphan_broadcast_acks = 0;
    } maintenance;

    // ping results, reported with the next result
    struct BaselineResult {
        uint32_t ping_count = 0;  // per server

        // per server
        struct PerServerStats {
            uint32_t id = 0;
            uint32_t count = 0;
            double avg_us = 0, min_us = 0, p50_us = 0, p95_us = 0;
            double p99_us = 0, max_us = 0;
        };
        std::vector<PerServerStats> servers;

        // overall
        uint32_t total_pings = 0;
        double overall_avg_us = 0, overall_min_us = 0, overall_p50_us = 0;
        double overall_p95_us = 0, overall_p99_us = 0, overall_max_us = 0;

        // same as overall
        struct ModeStats {
            double avg_us = 0, min_us = 0, p50_us = 0, p95_us = 0;
            double p99_us = 0, max_us = 0;
        };
        ModeStats direct;

        bool measured = false;
    } baseline;

    // admin socket (REP)
    std::unique_ptr<zmq::socket_t> client_socket;
    std::string client_endpoint = "tcp://*:29699";
};

extern DaemonState g_daemon;
extern std::ostream* g_current_out;

void reset_socket_state();
void daemon_output(const std::string& msg);
void daemon_output_line(const std::string& msg);
std::vector<uint32_t> get_partition_node_counts();
std::string generate_debug_log_path(const std::string& mode, uint32_t count);
void stop_daemon_server_threads();
// Checks the start-up stats replies: every server answers as its rank with one storage mode, the
// coordinator's dimension and one PQ code size. `summary` names them.
bool cluster_storage_consistent(const std::vector<ann::StatsResponseHeader>& replies, uint32_t servers,
                                uint32_t dim, std::string* summary, std::string* error);

void query_server_lock_metrics(ann::WorkloadResult& result);
bool query_server_stats(std::vector<ann::StatsResponseHeader>& snapshots,
                        bool consume_interval_max);
void populate_detailed_metrics(const UnifiedContext& ctx, ann::WorkloadResult& result);

void run_search_workload(uint32_t n_queries, const std::string& gt_path);
// recall: optional --query-file (from query_offset), self-recall or --self-ids, and --search-l
// (search_l, 0 = the start-up L).
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

bool send_admin_command(ann::MessageType cmd_type);
bool process_daemon_command(const std::string& cmd);
// admin commands, listed at start and in the unknown-command reply
inline constexpr const char* kAdminCommands =
    "recall, insert, mixed, drain, recreate-head, status, stats, ping, reset, membership-check, "
    "migration-apply, migration-accept, migration-rollback, controller-start, controller-status, "
    "controller-stop, controller-abort, shutdown, quit, exit";
std::string run_ping_baseline(uint32_t count, uint32_t pipelined_batch = 0);

} // namespace tide
