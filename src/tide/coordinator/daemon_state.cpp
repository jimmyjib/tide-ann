/**
 * Daemon state helpers of the coordinator.
 *
 * Contains:
 * - reset_socket_state: Drain leftover messages
 * - daemon_output: Output to both stdout and response stream
 * - get_partition_node_counts: Count nodes per partition
 * - generate_debug_log_path: Generate debug log file path
 */

#include "coordinator_types.h"

#include <iostream>
#include <sstream>
#include <chrono>

namespace tide {

// Reset ZMQ socket state between commands to prevent state pollution
void reset_socket_state() {
    // After the first workload, coordinator_server_thread owns every DEALER
    // socket for the daemon lifetime.  libzmq sockets are not thread-safe:
    // polling a socket here while its owner thread is polling the same socket
    // can corrupt the internal ypipe and abort.  All later admin commands use
    // server_queues/main_response_queue, so there is nothing for the daemon
    // main thread to drain or rebuild while those queues exist.
    if (g_daemon.main_response_queue && !g_daemon.server_queues.empty()) {
        return;
    }

    // Drain any remaining messages from all sockets
    for (auto& socket : g_daemon.sockets) {
        zmq::message_t msg;
        int drained = 0;
        while (true) {
            try {
                auto result = socket.recv(msg, zmq::recv_flags::dontwait);
                if (!result.has_value() || result.value() == 0) break;
                drained++;
            } catch (...) {
                break;
            }
        }
    }

    // Reset poll_items to ensure clean state
    g_daemon.poll_items.clear();
    for (auto& sock : g_daemon.sockets) {
        zmq::pollitem_t item = {static_cast<void*>(sock), 0, ZMQ_POLLIN, 0};
        g_daemon.poll_items.push_back(item);
    }
}

// Output helper - writes to both stdout and g_current_out (if set)
void daemon_output(const std::string& msg) {
    std::cout << msg << std::flush;
    if (g_current_out) {
        *g_current_out << msg;
    }
}

void daemon_output_line(const std::string& msg) {
    daemon_output(msg + "\n");
}

// Helper: Calculate partition node counts from membership array
std::vector<uint32_t> get_partition_node_counts() {
    std::vector<uint32_t> counts(g_daemon.n_servers, 0);
    for (size_t i = 0; i < g_daemon.membership.size(); i++) {
        uint32_t part = owner_label_checked(g_daemon.membership[i], g_daemon.n_servers, "partition node counts");
        counts[part]++;
    }
    return counts;
}

// Helper: Generate verbose debug log path
// Format: debug_{dataset}_{mode}_{timestamp}.csv, relative: the --debug files go to the coordinator's
// working directory (--data-dir may be mounted read-only).
std::string generate_debug_log_path(const std::string& mode, uint32_t count) {
    // Dataset name = the last component of data_dir (e.g. "demo" from "/data/demo")
    std::string dataset = g_daemon.data_dir;
    size_t last_slash = dataset.rfind('/');
    if (last_slash != std::string::npos) {
        dataset = dataset.substr(last_slash + 1);
    }

    // Generate timestamp (format: YYYYMMDD_HHMMSS)
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    char ts[20];
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm);

    // Build filename
    std::ostringstream oss;
    oss << "debug_" << dataset << "_" << mode << count << "_" << ts << ".csv";
    return oss.str();
}

void stop_daemon_server_threads() {
    g_daemon.server_threads_running.store(false, std::memory_order_release);
    for (auto& thread : g_daemon.server_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    g_daemon.server_threads.clear();
    // Switch later admin commands to their direct-socket fallback. Keeping
    // these queues allocated would enqueue shutdown work with no consumer.
    g_daemon.server_queues.clear();
    g_daemon.main_response_queue.reset();
}

bool cluster_storage_consistent(const std::vector<ann::StatsResponseHeader>& replies, uint32_t servers,
                                uint32_t dim, std::string* summary, std::string* error) {
    auto fail = [&](const std::string& why) { if (error) *error = why; return false; };
    if (replies.size() != servers || servers == 0)
        return fail("expected " + std::to_string(servers) + " stats replies, got " + std::to_string(replies.size()));
    auto mode_name = [](uint32_t mode) {
        return mode == 0 ? std::string("full") : mode == 1 ? std::string("pq") : "mode " + std::to_string(mode);
    };
    std::ostringstream modes;
    bool mixed = false, other_dim = false, other_m = false, bad_mode = false;
    for (uint32_t rank = 0; rank < servers; ++rank) {
        const auto& r = replies[rank];
        if (r.partition_id != rank)
            return fail("server " + std::to_string(rank) + " answered as partition " + std::to_string(r.partition_id));
        modes << (rank ? ", " : "") << rank << ": " << mode_name(r.storage_mode) << " dim " << r.storage_dim;
        if (r.storage_mode == 1) modes << " M " << r.storage_pq_m;
        bad_mode |= r.storage_mode > 1;
        mixed |= r.storage_mode != replies[0].storage_mode;
        other_dim |= r.storage_dim != dim;
        other_m |= r.storage_mode == 1 && r.storage_pq_m != replies[0].storage_pq_m;
    }
    if (bad_mode) return fail("unknown storage mode (" + modes.str() + ")");
    if (mixed)
        return fail("mixed storage modes (" + modes.str() + "): start every server with --pq, or every server "
                    "with --full-vectors");
    if (other_dim)
        return fail("vector dimension differs from the coordinator's base.npy dimension " + std::to_string(dim) +
                    " (" + modes.str() + ")");
    if (other_m) return fail("PQ code sizes differ (" + modes.str() + ")");
    if (summary) {
        *summary = mode_name(replies[0].storage_mode) + ", dim " + std::to_string(dim);
        if (replies[0].storage_mode == 1) *summary += ", PQ M " + std::to_string(replies[0].storage_pq_m);
    }
    if (error) error->clear();
    return true;
}

} // namespace tide
