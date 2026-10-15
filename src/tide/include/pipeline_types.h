#pragma once
/**
 * Pipeline types and global declarations of the unified pipeline.
 */

#include <vector>
#include <string>
#include <fstream>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <unordered_map>

#include "tide_types.h"
#include "owner_label.h"
#include "state_protocol.h"
#include "metrics/experiment_result.h"

namespace tide {

struct CoordinatorInFlightRecord {
    uint32_t origin_server = 0;
    steady_clock::time_point send_begin;
    double coord_send_us = 0;
};

// =============================================================================
// DebugState - All debug-related variables in one place
// =============================================================================
struct DebugState {
    // Toggle
    bool enabled = false;

    // Partition distribution tracking
    std::vector<uint32_t> entry_partition_count;
    std::vector<uint32_t> final_partition_count;
    std::vector<std::vector<uint32_t>> transition_matrix;  // entry -> final
    std::vector<uint64_t> topK_partition_total;

    // Insert tracking
    uint32_t insert_count = 0;
    uint32_t insert_total_hops = 0;
    std::vector<uint32_t> insert_hop_distribution;
    uint32_t insert_L = 0, insert_K = 0, insert_W = 0;

    // Query tracking
    uint32_t query_count = 0;
    uint32_t query_total_hops = 0;
    std::vector<uint32_t> query_result_partition;
    uint32_t query_L = 0, query_K = 0;

    // File logging
    std::ofstream log_file;
    std::ofstream hop_file;
    std::mutex log_mutex;

    // Context (the owner labels arrive in each MetricEvent; the metric thread
    // never reads the membership, which the pipeline thread grows)
    uint32_t n_servers = 4;
};

// =============================================================================
// Global variables (defined in debug_utils.cpp)
// =============================================================================
extern DebugState g_debug;
extern std::atomic<uint32_t> g_last_completed;
extern std::atomic<uint32_t> g_total_workloads;
extern std::atomic<uint32_t> g_in_flight_search;
extern std::atomic<uint32_t> g_in_flight_insert;
extern std::atomic<size_t> g_peak_memory_kb;

// =============================================================================
// Function declarations - debug_utils.cpp
// =============================================================================
void crash_handler(int sig);
void install_crash_handlers();
size_t get_memory_usage_kb();

uint32_t get_node_partition(const UnifiedContext& ctx, uint32_t node_id);
// The owner label of every id, resolved on the pipeline thread (which owns the
// membership vector and grows it as inserts are dispatched) for the metric
// thread's --debug logs, which never read the membership themselves.
std::vector<uint32_t> owner_labels(const UnifiedContext& ctx, const std::vector<uint32_t>& ids);
std::vector<uint32_t> owner_labels(const UnifiedContext& ctx, const std::vector<ann::BeamEntry>& beam);

// The CSV headers name one partition column per server (n_servers columns), as
// the rows write them.
std::string debug_log_header(uint32_t n_servers);
std::string debug_hop_header(uint32_t n_servers);
void open_debug_log(const std::string& path, uint32_t n_servers);
void close_debug_log();
// labels[i] is the owner of beam[i] / neighbors[i] (owner_labels).
void log_hop_state(const char* type, uint32_t id, uint32_t hop_num,
                   uint32_t current_partition, uint32_t next_partition,
                   const std::vector<ann::BeamEntry>& beam, const std::vector<uint32_t>& labels);
void log_search_result(const char* type, uint32_t id, uint32_t entry_part, uint32_t final_part,
                       uint32_t hops, const std::vector<uint32_t>& neighbors, const std::vector<uint32_t>& labels,
                       const std::vector<float>& distances, uint32_t K,
                       const int32_t* gt_row, uint32_t gt_k);

// Opens the --debug logs when log_path is not empty and a membership is given.
void enable_partition_debug(uint32_t n_servers, const std::string& log_path,
                            const std::vector<uint32_t>* membership_vec);
void disable_partition_debug();
void print_partition_debug_stats();
void populate_debug_metrics(ann::ExperimentResult& result);

// =============================================================================
// Function declarations - workload_generator.cpp
// =============================================================================
void generate_mixed_workloads(UnifiedContext& ctx);
void send_workload(UnifiedContext& ctx, uint32_t workload_idx);
ServerWorkItem create_search_work_item(UnifiedContext& ctx, uint32_t workload_idx);
void queue_insert_ops(UnifiedContext& ctx, uint32_t workload_idx);

// =============================================================================
// Function declarations - pipeline_core.cpp
// =============================================================================
void process_response(UnifiedContext& ctx, uint32_t server_idx, uint32_t& completed);
void process_server_response(UnifiedContext& ctx, ServerResponse& resp, uint32_t& completed);
void coordinator_server_thread(
    uint32_t server_idx, zmq::socket_t& socket,
    ann::ThreadSafeQueue<ServerWorkItem>& work_queue,
    ann::ThreadSafeQueue<ServerResponse>& response_queue,
    std::vector<std::atomic<int32_t>>& in_flight_by_origin,
    std::atomic<int32_t>& global_search_in_flight,
    std::mutex& in_flight_mutex,
    std::unordered_map<uint32_t, CoordinatorInFlightRecord>& in_flight_records,
    std::atomic<bool>& running);
void run_unified_workload(UnifiedContext& ctx);

// =============================================================================
// Function declarations - metric_thread.cpp
// =============================================================================
void process_metric_event(UnifiedContext& ctx, const MetricEvent& evt);
void run_metric_thread(UnifiedContext& ctx);

} // namespace tide
