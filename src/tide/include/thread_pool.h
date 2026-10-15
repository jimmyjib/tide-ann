#pragma once

#include <mutex>
#include <condition_variable>
#include <queue>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <memory>
#include <unistd.h>

namespace ann {
enum class MessageType : uint8_t;
}

namespace ann {

// A request received by the I/O thread.
struct WorkItem {
    std::string identity;  // ZMQ routing identity
    std::vector<char> request;
    std::chrono::steady_clock::time_point enqueue_time;
    uint32_t zmq_recv_us = 0;
    uint32_t io_processing_us = 0;  // receive end to enqueue
    uint32_t queue_depth = 0;  // at enqueue
    std::string coordinator_id;  // P2P: where the final reply goes
    bool from_peer = false;
    // message type for the worker dispatch
    MessageType msg_type = MessageType::SEARCH_REQUEST;
};

// A reply from a worker to the I/O thread.
struct ResponseItem {
    std::string identity;  // ZMQ routing identity
    std::vector<char> response;
    std::chrono::steady_clock::time_point enqueue_time;
    std::string coordinator_id;       // P2P
    bool needs_forward = false;       // P2P: forward forward_state to target_partition
    uint32_t target_partition = 0;
    std::vector<char> forward_state;
};

// Timing the I/O thread measured on the previous reply, reported by the workers.
struct IOTimingState {
    std::atomic<uint32_t> last_zmq_send_us{0};
    std::atomic<uint32_t> last_resp_queue_wait_us{0};
    std::atomic<uint32_t> last_send_prep_us{0};  // pop to send start
};

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
            ::write(signal_fd_, &byte, 1);  // a full pipe is already signaled
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

    // up to n items
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

// Worker and I/O thread timing totals (microseconds unless named otherwise).
struct TimingMetrics {
    std::atomic<uint64_t> queue_wait_us{0};
    std::atomic<uint64_t> queue_wait_count{0};
    // largest wait since the previous consuming GET_STATS
    std::atomic<uint64_t> queue_wait_interval_max_us{0};
    std::atomic<uint64_t> deserialize_us{0};
    std::atomic<uint64_t> search_total_us{0};
    std::atomic<uint64_t> beam_sort_us{0};
    std::atomic<uint64_t> serialize_us{0};
    std::atomic<uint64_t> response_queue_us{0};  // push to the response queue
    std::atomic<uint64_t> request_count{0};
    std::atomic<uint64_t> pq_distance_count{0};
    std::atomic<uint64_t> full_distance_count{0};
    // I/O thread
    std::atomic<uint64_t> zmq_recv_us{0};
    std::atomic<uint64_t> zmq_send_us{0};
    std::atomic<uint64_t> queue_depth_sum{0};    // queue depths at enqueue
    std::atomic<uint64_t> response_queue_wait_us{0};
    std::atomic<uint64_t> response_queue_wait_count{0};
    // largest wait since the previous consuming GET_STATS
    std::atomic<uint64_t> response_queue_wait_interval_max_us{0};
    std::atomic<uint64_t> send_prep_us{0};       // pop to send start

    void reset() {
        queue_wait_us = 0;
        queue_wait_count = 0;
        queue_wait_interval_max_us = 0;
        deserialize_us = 0;
        search_total_us = 0;
        beam_sort_us = 0;
        serialize_us = 0;
        response_queue_us = 0;
        request_count = 0;
        pq_distance_count = 0;
        full_distance_count = 0;
        zmq_recv_us = 0;
        zmq_send_us = 0;
        queue_depth_sum = 0;
        response_queue_wait_us = 0;
        response_queue_wait_count = 0;
        response_queue_wait_interval_max_us = 0;
        send_prep_us = 0;
    }
};

// Acquisitions, wait and hold time of one lock role.
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

// Lock wait metrics (shared/unique) reported in GET_STATS.
struct LockMetrics {
    std::atomic<uint64_t> shared_lock_count{0};
    std::atomic<uint64_t> unique_lock_count{0};
    std::atomic<uint64_t> shared_lock_wait_us{0};
    std::atomic<uint64_t> unique_lock_wait_us{0};

    // per insert phase
    std::atomic<uint64_t> insert_alloc_lock_count{0};
    std::atomic<uint64_t> insert_alloc_lock_wait_us{0};
    std::atomic<uint64_t> insert_edge_lock_count{0};
    std::atomic<uint64_t> insert_edge_lock_wait_us{0};
    std::atomic<uint64_t> backlink_lock_count{0};
    std::atomic<uint64_t> backlink_lock_wait_us{0};
    std::atomic<uint64_t> broadcast_lock_count{0};
    std::atomic<uint64_t> broadcast_lock_wait_us{0};

    // per lock role, per acquisition, with hold time
    LockRoleMetrics search_shared;
    LockRoleMetrics insert_alloc_unique;
    LockRoleMetrics insert_alloc_shared;
    LockRoleMetrics insert_alloc_mutex;
    LockRoleMetrics insert_prune_shared;
    LockRoleMetrics insert_edge_shared;
    LockRoleMetrics backlink_unique;
    LockRoleMetrics broadcast_unique;
    // node lock taken by search to copy a row
    LockRoleMetrics search_node_lock;

    // search totals
    std::atomic<uint64_t> search_distance_us{0};
    std::atomic<uint64_t> search_beam_sort_us{0};
    std::atomic<uint64_t> search_distance_count{0};
    std::atomic<uint64_t> search_neighbors_evaluated{0};
    std::atomic<uint64_t> search_request_count{0};
    // exact + adc + rerank = search_distance_count (missing codes are no distance work)
    std::atomic<uint64_t> search_exact_count{0};
    std::atomic<uint64_t> search_adc_count{0};
    std::atomic<uint64_t> search_rerank_count{0};
    std::atomic<uint64_t> search_missing_code_count{0};
    // PQ mode: ADC tables built (at most one per request) and their build time, not in search_distance_us
    std::atomic<uint64_t> search_adc_table_count{0};
    std::atomic<uint64_t> search_adc_table_ns{0};

    void reset() {
        shared_lock_count = 0;
        unique_lock_count = 0;
        shared_lock_wait_us = 0;
        unique_lock_wait_us = 0;
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

    double avg_shared_wait_us() const {
        uint64_t count = shared_lock_count.load();
        return count > 0 ? static_cast<double>(shared_lock_wait_us.load()) / count : 0.0;
    }

    double avg_unique_wait_us() const {
        uint64_t count = unique_lock_count.load();
        return count > 0 ? static_cast<double>(unique_lock_wait_us.load()) / count : 0.0;
    }

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

// Backlink and insert-candidate outcomes.
struct GraphMetrics {
    std::atomic<uint64_t> backlink_edges_requested{0};
    std::atomic<uint64_t> backlink_edges_applied{0};
    std::atomic<uint64_t> backlink_skip_not_local{0};
    std::atomic<uint64_t> backlink_skip_already_exists{0};
    // dropped: target id not known here yet
    std::atomic<uint64_t> backlink_skip_out_of_range{0};
    // insert candidates skipped because they had not fully reached this server yet
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
