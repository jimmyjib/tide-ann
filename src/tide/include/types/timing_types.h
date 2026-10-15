// Coordinator timing, per-hop latency and query state.
#pragma once

#include <vector>
#include <chrono>
#include <cstdint>
#include <set>
#include <algorithm>
#include "state_protocol.h"

namespace tide {

using namespace std::chrono;
using ann::BeamEntry;

// coordinator overhead of one command
struct TimingMetrics {
    double serialize_us = 0;
    double send_us = 0;
    double poll_wait_us = 0;
    double recv_us = 0;
    double deserialize_us = 0;
    double process_us = 0;
    uint32_t send_count = 0;
    uint32_t recv_count = 0;
    uint32_t poll_count = 0;
};

struct HopLatency {
    uint32_t from_partition;
    uint32_t to_partition;
    double total_us;
    double server_processing_us;
    double queue_wait_us;
    double network_us;
    double lock_wait_us = 0;

    uint32_t pq_distances = 0;
    uint32_t full_distances = 0;
    uint32_t local_nodes_expanded = 0;
    uint32_t top_W_count = 0;
    uint32_t neighbors_evaluated = 0;

    // server side
    uint32_t zmq_recv_us = 0;
    uint32_t zmq_send_us = 0;
    uint32_t serialize_us = 0;
    uint32_t deserialize_us = 0;
    uint32_t queue_depth = 0;
    uint32_t response_queue_wait_us = 0;
    uint32_t io_processing_us = 0;  // receive end to enqueue
    uint32_t worker_overhead_us = 0;  // unmeasured worker time
    uint32_t send_prep_us = 0;  // pop to send start

    // coordinator side
    double coord_serialize_us = 0;
    double coord_send_us = 0;
    double coord_response_queue_us = 0;  // server thread to main thread handoff
    double coord_recv_us = 0;
    double coord_deserialize_us = 0;

    double measured_server_total_us() const {
        // server_processing_us includes deserialize_us.
        // zmq_send_us is excluded: the value on the wire belongs to the previous request.
        return server_processing_us + zmq_recv_us +
               serialize_us + response_queue_wait_us + io_processing_us +
               worker_overhead_us + send_prep_us;
    }

    double measured_coord_total_us() const {
        // coord_send_us and the poll wait overlap server time, so they are not subtracted.
        return coord_serialize_us + coord_response_queue_us +
               coord_recv_us + coord_deserialize_us;
    }

    double pure_network_us() const {
        return total_us - queue_wait_us - measured_server_total_us() - measured_coord_total_us();
    }
};

struct QueryState {
    uint32_t query_id;
    std::vector<float> query;
    std::vector<BeamEntry> beam;
    uint32_t current_partition;
    uint32_t hops;
    uint32_t total_distance_computations;
    bool done;
    std::vector<uint32_t> result_neighbors;
    std::vector<float> result_distances;

    steady_clock::time_point query_start_time;
    steady_clock::time_point send_time;
    double query_latency_us = 0;

    double last_coord_serialize_us = 0;
    double last_coord_send_us = 0;

    std::vector<HopLatency> hop_latencies;

    uint32_t total_pq_dist = 0;
    uint32_t total_full_dist = 0;
    uint32_t total_local_expanded = 0;
    uint32_t total_top_W = 0;
    uint32_t total_neighbors_evaluated = 0;
    uint64_t total_server_time_us = 0;
    uint64_t total_queue_wait_us = 0;
    uint64_t total_lock_wait_us = 0;
    std::vector<uint32_t> partition_path;

    // P2P
    uint32_t p2p_cumulative_queue_us = 0;
    uint32_t p2p_cumulative_processing_us = 0;
};

// Recall@k against a ground-truth row.
inline double compute_recall(const std::vector<uint32_t>& result, const int32_t* gt, uint32_t k) {
    std::set<int32_t> gt_set(gt, gt + k);
    int hits = 0;
    for (uint32_t i = 0; i < std::min(static_cast<uint32_t>(result.size()), k); i++) {
        if (gt_set.count(static_cast<int32_t>(result[i]))) hits++;
    }
    return static_cast<double>(hits) / k;
}

}  // namespace tide
