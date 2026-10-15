#pragma once

/**
 * Thread Pool Utilities for Multi-threaded Servers.
 *
 * Common structures and classes of the multi-threaded server.
 */

#include <mutex>
#include <condition_variable>
#include <queue>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <memory>
#include <unistd.h>

// Forward declaration for MessageType
namespace ann {
enum class MessageType : uint8_t;
}

namespace ann {

/**
 * Work item for thread pool.
 * Represents an incoming request to be processed by a worker thread.
 */
struct WorkItem {
    std::string identity;  // ZMQ routing identity
    std::vector<char> request;  // Serialized request data
    std::chrono::steady_clock::time_point enqueue_time;
    // Detailed timing
    uint32_t zmq_recv_us = 0;  // Time spent in ZMQ recv()
    uint32_t io_processing_us = 0;  // Time between recv_end and enqueue (I/O thread processing)
    uint32_t queue_depth = 0;  // Queue depth when enqueued
    // P2P fields (peer forwarding)
    std::string coordinator_id;  // Original coordinator identity for DONE response
    bool from_peer = false;      // True if received from peer server
    // Message type for unified worker dispatch (Unified Pipeline)
    MessageType msg_type = MessageType::SEARCH_REQUEST;
};

/**
 * Response item from workers.
 * Represents a completed response to be sent back.
 */
struct ResponseItem {
    std::string identity;  // ZMQ routing identity
    std::vector<char> response;  // Serialized response data
    std::chrono::steady_clock::time_point enqueue_time;  // For response queue wait measurement
    // P2P fields (peer forwarding)
    std::string coordinator_id;       // Original coordinator identity
    bool needs_forward = false;       // True if should forward to peer
    uint32_t target_partition = 0;    // Partition to forward to
    std::vector<char> forward_state;  // Serialized state for forwarding
};

/**
 * Shared timing state between I/O thread and workers.
 * Used for 1-request-delayed timing values.
 */
struct IOTimingState {
    std::atomic<uint32_t> last_zmq_send_us{0};
    std::atomic<uint32_t> last_resp_queue_wait_us{0};
    std::atomic<uint32_t> last_send_prep_us{0};  // Gap: pop_time → send_start (routing + msg construction)
};

/**
 * Thread-safe queue for producer-consumer pattern.
 */
template<typename T>
class ThreadSafeQueue {
public:
    void set_signal_fd(int fd) { signal_fd_ = fd; }

    void push(T item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(item));
        }
        cv_.notify_one();
        if (signal_fd_ >= 0) {
            char byte = 1;
            ::write(signal_fd_, &byte, 1);  // wake up poll, ignore failure (pipe full = already signaled)
        }
    }

    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool wait_pop(T& item, int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                         [this] { return !queue_.empty(); })) {
            item = std::move(queue_.front());
            queue_.pop();
            return true;
        }
        return false;
    }

    bool wait_until_nonempty(int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(
            lock, std::chrono::milliseconds(timeout_ms),
            [this] { return !queue_.empty(); });
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    /**
     * Try to pop up to n items at once.
     * Returns actual number of items popped (may be less than n if queue has fewer).
     */
    std::vector<T> try_pop_n(size_t n) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<T> result;
        size_t to_pop = std::min(n, queue_.size());
        result.reserve(to_pop);
        for (size_t i = 0; i < to_pop; i++) {
            result.push_back(std::move(queue_.front()));
            queue_.pop();
        }
        return result;
    }

    /**
     * Get condition variable for external waiting.
     * Used for wait_for_any_queue pattern.
     */
    std::condition_variable& get_cv() {
        return cv_;
    }

    std::mutex& get_mutex() {
        return mutex_;
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    int signal_fd_ = -1;
};

/**
 * Timing metrics for performance monitoring.
 * Used to track various timing statistics across worker threads.
 */
struct TimingMetrics {
    std::atomic<uint64_t> queue_wait_us{0};      // Time spent waiting in work queue
    std::atomic<uint64_t> queue_wait_count{0};   // Work items included in queue_wait_us
    // Largest work-queue wait since the previous consuming GET_STATS snapshot.
    std::atomic<uint64_t> queue_wait_interval_max_us{0};
    std::atomic<uint64_t> deserialize_us{0};     // Time to deserialize request
    std::atomic<uint64_t> search_total_us{0};    // Total search time
    std::atomic<uint64_t> pq_distance_us{0};     // PQ distance computation time
    std::atomic<uint64_t> disk_read_us{0};       // Disk I/O time
    std::atomic<uint64_t> rerank_us{0};          // Re-ranking time
    std::atomic<uint64_t> beam_sort_us{0};       // Beam sorting time
    std::atomic<uint64_t> serialize_us{0};       // Response serialization time
    std::atomic<uint64_t> response_queue_us{0};  // Time to push to response queue
    std::atomic<uint64_t> request_count{0};      // Number of processed requests
    std::atomic<uint64_t> pq_distance_count{0};  // Number of PQ distance computations
    std::atomic<uint64_t> full_distance_count{0}; // Number of full distance computations
    std::atomic<uint64_t> disk_read_count{0};    // Number of disk reads
    // I/O thread level metrics (aggregate only)
    std::atomic<uint64_t> zmq_recv_us{0};        // ZMQ recv() time
    std::atomic<uint64_t> zmq_send_us{0};        // ZMQ send() time
    std::atomic<uint64_t> queue_depth_sum{0};    // Sum of queue depths at enqueue
    std::atomic<uint64_t> response_queue_wait_us{0};  // Time waiting in response queue
    std::atomic<uint64_t> response_queue_wait_count{0}; // Responses included in wait total
    // Largest response-queue wait since the previous consuming GET_STATS snapshot.
    std::atomic<uint64_t> response_queue_wait_interval_max_us{0};
    std::atomic<uint64_t> send_prep_us{0};       // Gap: pop_time → send_start (routing + msg construction)

    void reset() {
        queue_wait_us = 0;
        queue_wait_count = 0;
        queue_wait_interval_max_us = 0;
        deserialize_us = 0;
        search_total_us = 0;
        pq_distance_us = 0;
        disk_read_us = 0;
        rerank_us = 0;
        beam_sort_us = 0;
        serialize_us = 0;
        response_queue_us = 0;
        request_count = 0;
        pq_distance_count = 0;
        full_distance_count = 0;
        disk_read_count = 0;
        zmq_recv_us = 0;
        zmq_send_us = 0;
        queue_depth_sum = 0;
        response_queue_wait_us = 0;
        response_queue_wait_count = 0;
        response_queue_wait_interval_max_us = 0;
        send_prep_us = 0;
    }
};

/**
 * Lock metrics for tracking shared_mutex contention.
 * Used to measure lock wait times during concurrent Search + Insert workloads.
 *
 * Tracks the global g_resize_mutex in tide_server.cpp:
 * - shared_lock: Used by search workers (concurrent reads allowed)
 * - unique_lock: Used by RESET, which clears and reloads the stores
 * Node-level locking (node_locks[i].mtx) protects individual adjacency list modifications.
 *
 * High lock wait times indicate contention between search and resize operations.
 */
struct LockRoleMetrics {
    std::atomic<uint64_t> acquisitions{0};
    std::atomic<uint64_t> wait_us{0};
    std::atomic<uint64_t> hold_us{0};
    std::atomic<uint64_t> max_wait_us{0};
    std::atomic<uint64_t> max_hold_us{0};

    static void update_max(std::atomic<uint64_t>& target, uint64_t value) {
        uint64_t current = target.load(std::memory_order_relaxed);
        while (current < value &&
               !target.compare_exchange_weak(
                   current, value,
                   std::memory_order_relaxed,
                   std::memory_order_relaxed)) {
        }
    }

    void record(uint64_t wait, uint64_t hold) {
        acquisitions.fetch_add(1, std::memory_order_relaxed);
        wait_us.fetch_add(wait, std::memory_order_relaxed);
        hold_us.fetch_add(hold, std::memory_order_relaxed);
        update_max(max_wait_us, wait);
        update_max(max_hold_us, hold);
    }

    void reset() {
        acquisitions = 0;
        wait_us = 0;
        hold_us = 0;
        max_wait_us = 0;
        max_hold_us = 0;
    }
};

struct LockMetrics {
    // Acquisition counts
    std::atomic<uint64_t> shared_lock_count{0};   // Number of shared_lock acquisitions
    std::atomic<uint64_t> unique_lock_count{0};   // Number of unique_lock acquisitions

    // Wait times (microseconds) - time spent waiting to acquire locks
    std::atomic<uint64_t> shared_lock_wait_us{0}; // Total wait time for shared_lock
    std::atomic<uint64_t> unique_lock_wait_us{0}; // Total wait time for unique_lock

    // Phase-specific metrics for insert latency breakdown
    std::atomic<uint64_t> insert_alloc_lock_count{0};   // Insert allocation phase
    std::atomic<uint64_t> insert_alloc_lock_wait_us{0};
    std::atomic<uint64_t> insert_edge_lock_count{0};    // Insert edge addition phase
    std::atomic<uint64_t> insert_edge_lock_wait_us{0};
    std::atomic<uint64_t> backlink_lock_count{0};       // Backlink phase
    std::atomic<uint64_t> backlink_lock_wait_us{0};
    std::atomic<uint64_t> broadcast_lock_count{0};      // Broadcast phase
    std::atomic<uint64_t> broadcast_lock_wait_us{0};

    // Role-level lock profile. These counters are per acquisition and include
    // hold time, unlike the phase counters above, which are sometimes
    // normalized by batch item count.
    LockRoleMetrics search_shared;
    LockRoleMetrics insert_alloc_unique;
    LockRoleMetrics insert_alloc_shared;
    LockRoleMetrics insert_alloc_mutex;
    LockRoleMetrics insert_prune_shared;
    LockRoleMetrics insert_edge_shared;
    LockRoleMetrics backlink_unique;
    LockRoleMetrics broadcast_unique;
    // per-node mutex acquired by search path (process_state)
    // protecting adj snapshot reads. Reveals contention against insert/backlink
    // node-lock holders.
    LockRoleMetrics search_node_lock;

    // search-internal sub-stage cumulative metrics. All
    // counters reset to 0 in reset(). distance_us/beam_sort_us in microseconds.
    std::atomic<uint64_t> search_distance_us{0};
    std::atomic<uint64_t> search_beam_sort_us{0};
    std::atomic<uint64_t> search_distance_count{0};
    std::atomic<uint64_t> search_neighbors_evaluated{0};
    std::atomic<uint64_t> search_request_count{0};
    // The kinds of search_distance_count (distance evaluations): neighbours
    // scored against a held vector (exact) or a PQ code (adc), and own-only
    // re-ranks (the exact re-score of an expanded entry); exact + adc + rerank
    // = search_distance_count. A neighbour with neither vector nor code
    // (missing code, scored FLT_MAX) is counted apart and is no distance work.
    std::atomic<uint64_t> search_exact_count{0};
    std::atomic<uint64_t> search_adc_count{0};
    std::atomic<uint64_t> search_rerank_count{0};
    std::atomic<uint64_t> search_missing_code_count{0};
    // Own-only: ADC query tables built (at most one per request) and their build
    // time in ns (kept out of search_distance_us).
    std::atomic<uint64_t> search_adc_table_count{0};
    std::atomic<uint64_t> search_adc_table_ns{0};

    void reset() {
        shared_lock_count = 0;
        unique_lock_count = 0;
        shared_lock_wait_us = 0;
        unique_lock_wait_us = 0;
        // Reset phase-specific metrics
        insert_alloc_lock_count = 0;
        insert_alloc_lock_wait_us = 0;
        insert_edge_lock_count = 0;
        insert_edge_lock_wait_us = 0;
        backlink_lock_count = 0;
        backlink_lock_wait_us = 0;
        broadcast_lock_count = 0;
        broadcast_lock_wait_us = 0;
        search_shared.reset();
        insert_alloc_unique.reset();
        insert_alloc_shared.reset();
        insert_alloc_mutex.reset();
        insert_prune_shared.reset();
        insert_edge_shared.reset();
        backlink_unique.reset();
        broadcast_unique.reset();
        search_node_lock.reset();
        search_distance_us = 0;
        search_beam_sort_us = 0;
        search_distance_count = 0;
        search_neighbors_evaluated = 0;
        search_request_count = 0;
        search_exact_count = 0;
        search_adc_count = 0;
        search_rerank_count = 0;
        search_missing_code_count = 0;
        search_adc_table_count = 0;
        search_adc_table_ns = 0;
    }

    // Helper: average wait time per shared_lock acquisition
    double avg_shared_wait_us() const {
        uint64_t count = shared_lock_count.load();
        return count > 0 ? static_cast<double>(shared_lock_wait_us.load()) / count : 0.0;
    }

    // Helper: average wait time per unique_lock acquisition
    double avg_unique_wait_us() const {
        uint64_t count = unique_lock_count.load();
        return count > 0 ? static_cast<double>(unique_lock_wait_us.load()) / count : 0.0;
    }

    // Phase-specific average wait helpers
    double avg_insert_alloc_wait_us() const {
        uint64_t count = insert_alloc_lock_count.load();
        return count > 0 ? static_cast<double>(insert_alloc_lock_wait_us.load()) / count : 0.0;
    }
    double avg_insert_edge_wait_us() const {
        uint64_t count = insert_edge_lock_count.load();
        return count > 0 ? static_cast<double>(insert_edge_lock_wait_us.load()) / count : 0.0;
    }
    double avg_backlink_wait_us() const {
        uint64_t count = backlink_lock_count.load();
        return count > 0 ? static_cast<double>(backlink_lock_wait_us.load()) / count : 0.0;
    }
    double avg_broadcast_wait_us() const {
        uint64_t count = broadcast_lock_count.load();
        return count > 0 ? static_cast<double>(broadcast_lock_wait_us.load()) / count : 0.0;
    }
};

/**
 * Graph modification metrics.
 * Tracks cross-partition backlink outcomes to monitor graph quality.
 */
struct GraphMetrics {
    std::atomic<uint64_t> backlink_edges_requested{0};
    std::atomic<uint64_t> backlink_edges_applied{0};
    std::atomic<uint64_t> backlink_skip_not_local{0};
    std::atomic<uint64_t> backlink_skip_already_exists{0};
    // Edges whose target id was at or beyond this server's n_base when the
    // batch ran (an inserted node it has not learned yet): dropped.
    std::atomic<uint64_t> backlink_skip_out_of_range{0};
    // Insert candidates this server had not received completely when the
    // insert ran (an inserted id of another server whose broadcast or backlink
    // was still queued here: no owner label, vector or code yet), dropped
    // before the prune, plus any selected neighbour without an owner label.
    std::atomic<uint64_t> insert_neighbors_undelivered{0};

    void reset() {
        backlink_edges_requested = 0;
        backlink_edges_applied = 0;
        backlink_skip_not_local = 0;
        backlink_skip_already_exists = 0;
        backlink_skip_out_of_range = 0;
        insert_neighbors_undelivered = 0;
    }
};

}  // namespace ann
