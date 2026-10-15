#pragma once
/**
 * Query and Global Metrics
 *
 * Structures for tracking per-query and aggregate metrics.
 */

#include <vector>
#include <cstdint>
#include <algorithm>

namespace ann {

/**
 * Extended query metrics collected during search.
 * Used by coordinators to track per-query detailed statistics.
 */
struct QueryMetrics {
    uint32_t total_pq_dist = 0;
    uint32_t total_full_dist = 0;
    uint32_t total_local_expanded = 0;
    uint32_t total_top_W = 0;
    uint32_t total_neighbors_evaluated = 0;
    uint64_t total_server_time_us = 0;
    std::vector<uint32_t> partition_path;  // Track visited partitions
};

/**
 * Global aggregated metrics across all queries.
 */
struct GlobalMetrics {
    uint64_t pq_dist = 0;
    uint64_t full_dist = 0;
    uint64_t local_expanded = 0;
    uint64_t top_W = 0;
    uint64_t neighbors_evaluated = 0;
    uint64_t server_time_us = 0;
    std::vector<uint64_t> requests_per_server;
    std::vector<double> hop_rtt_us;

    void init_servers(size_t n_servers) {
        requests_per_server.resize(n_servers, 0);
    }

    void add_hop_rtt(double rtt) {
        hop_rtt_us.push_back(rtt);
    }

    void add_server_request(size_t server_idx) {
        if (server_idx < requests_per_server.size()) {
            requests_per_server[server_idx]++;
        }
    }
};

}  // namespace ann
