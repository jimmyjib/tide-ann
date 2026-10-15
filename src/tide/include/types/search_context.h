#pragma once
/**
 * Search Context Types
 *
 * Context structure for search phase execution.
 * Includes ServerLockMetrics used by both SearchContext and UnifiedContext.
 */

#include <vector>
#include <cstdint>
#include <zmq.hpp>
#include "types/timing_types.h"
#include "types/pending_insert.h"

namespace tide {

/**
 * SearchContext: Context structure for search phase execution
 */
struct SearchContext {
    // === Input parameters ===
    uint32_t n_queries;
    uint32_t L;
    uint32_t K;
    uint32_t W;
    uint32_t effective_batch_size;
    bool p2p_mode;
    bool debug_mode;

    // Data references
    uint32_t dim;
    uint32_t n_servers;
    const int32_t* groundtruth;
    uint32_t gt_cols;
    const uint32_t* membership;

    // ZMQ resources (non-owning)
    std::vector<zmq::socket_t>* sockets;
    std::vector<zmq::pollitem_t>* poll_items;

    // Query states (modified by function)
    std::vector<QueryState>* query_states;

    // PendingInsertManager for merge
    PendingInsertManager* pending_mgr = nullptr;

    // === Output metrics ===
    double avg_recall = 0.0;
    double avg_hops = 0.0;
    double avg_hop_rtt = 0.0;
    double elapsed_ms = 0.0;
    double qps = 0.0;

    TimingMetrics timing;

    // Global extended metrics
    uint64_t global_pq_dist = 0;
    uint64_t global_full_dist = 0;
    uint64_t global_local_expanded = 0;
    uint64_t global_top_W = 0;
    uint64_t global_neighbors_evaluated = 0;
    uint64_t global_server_time_us = 0;
    uint64_t global_queue_wait_us = 0;

    std::vector<uint64_t> requests_per_server;
    std::vector<double> hop_rtt_us;

    /**
     * ServerLockMetrics: Per-server lock contention metrics
     * Used by both SearchContext and UnifiedContext
     */
    struct ServerLockMetrics {
        uint64_t shared_lock_count = 0;
        uint64_t unique_lock_count = 0;
        uint64_t shared_lock_wait_us = 0;
        uint64_t unique_lock_wait_us = 0;

        // Phase-specific lock metrics
        uint64_t insert_alloc_lock_count = 0;
        uint64_t insert_alloc_lock_wait_us = 0;
        uint64_t insert_edge_lock_count = 0;
        uint64_t insert_edge_lock_wait_us = 0;
        uint64_t backlink_lock_count = 0;
        uint64_t backlink_lock_wait_us = 0;
        uint64_t broadcast_lock_count = 0;
        uint64_t broadcast_lock_wait_us = 0;

        double avg_shared_wait_us() const {
            return shared_lock_count > 0 ?
                static_cast<double>(shared_lock_wait_us) / shared_lock_count : 0.0;
        }
        double avg_unique_wait_us() const {
            return unique_lock_count > 0 ?
                static_cast<double>(unique_lock_wait_us) / unique_lock_count : 0.0;
        }
        double avg_insert_alloc_wait_us() const {
            return insert_alloc_lock_count > 0 ?
                static_cast<double>(insert_alloc_lock_wait_us) / insert_alloc_lock_count : 0.0;
        }
        double avg_insert_edge_wait_us() const {
            return insert_edge_lock_count > 0 ?
                static_cast<double>(insert_edge_lock_wait_us) / insert_edge_lock_count : 0.0;
        }
        double avg_backlink_wait_us() const {
            return backlink_lock_count > 0 ?
                static_cast<double>(backlink_lock_wait_us) / backlink_lock_count : 0.0;
        }
        double avg_broadcast_wait_us() const {
            return broadcast_lock_count > 0 ?
                static_cast<double>(broadcast_lock_wait_us) / broadcast_lock_count : 0.0;
        }
    };

    std::vector<ServerLockMetrics> server_lock_metrics;
};

}  // namespace tide
