#pragma once

/**
 * Coordinator Utilities.
 *
 * Common functions of the coordinator and the servers.
 */

#include <string>
#include <vector>
#include <sstream>
#include <set>
#include <algorithm>
#include <cstdint>
#include <zmq.hpp>

namespace ann {

/**
 * Parse comma-separated server addresses.
 *
 * @param servers_str Comma-separated string (e.g., "tcp://host1:port,tcp://host2:port")
 * @return Vector of individual server addresses
 */
inline std::vector<std::string> parse_servers(const std::string& servers_str) {
    std::vector<std::string> servers;
    std::stringstream ss(servers_str);
    std::string addr;
    while (std::getline(ss, addr, ',')) {
        servers.push_back(addr);
    }
    return servers;
}

/**
 * Compute recall@k metric.
 *
 * @param result Retrieved neighbors
 * @param groundtruth True nearest neighbors
 * @param k Number of neighbors to consider
 * @return Recall value (0.0 to 1.0)
 */
inline double compute_recall(const std::vector<uint32_t>& result,
                             const std::vector<uint32_t>& groundtruth,
                             uint32_t k) {
    std::set<uint32_t> gt_set(groundtruth.begin(),
                               groundtruth.begin() + std::min(static_cast<size_t>(k), groundtruth.size()));
    size_t hits = 0;
    for (size_t i = 0; i < std::min(static_cast<size_t>(k), result.size()); i++) {
        if (gt_set.count(result[i])) hits++;
    }
    return static_cast<double>(hits) / k;
}

/**
 * Compute recall using signed groundtruth (some datasets use int32_t).
 */
inline double compute_recall_signed(const std::vector<uint32_t>& result,
                                    const std::vector<int32_t>& groundtruth,
                                    uint32_t k) {
    std::set<uint32_t> gt_set;
    for (size_t i = 0; i < std::min(static_cast<size_t>(k), groundtruth.size()); i++) {
        if (groundtruth[i] >= 0) {
            gt_set.insert(static_cast<uint32_t>(groundtruth[i]));
        }
    }
    size_t hits = 0;
    for (size_t i = 0; i < std::min(static_cast<size_t>(k), result.size()); i++) {
        if (gt_set.count(result[i])) hits++;
    }
    return static_cast<double>(hits) / k;
}

/**
 * ZMQ connection settings for high-throughput servers and coordinators.
 */
struct ZmqSettings {
    int sndhwm = 10000;    // Send high water mark
    int rcvhwm = 10000;    // Receive high water mark
    int sndtimeo = 1000;   // Send timeout (ms)
    int rcvtimeo = 0;      // Receive timeout (0 = non-blocking)
    int linger = 0;        // Linger time on close

    // Preset for servers (ROUTER socket)
    static ZmqSettings server() {
        return {10000, 10000, 100, 10, 0};
    }

    // Preset for coordinators (DEALER socket)
    static ZmqSettings coordinator() {
        return {10000, 10000, 1000, 0, 0};
    }
};

/**
 * Apply ZMQ settings to a socket.
 */
inline void apply_zmq_settings(zmq::socket_t& socket, const ZmqSettings& settings) {
    socket.set(zmq::sockopt::linger, settings.linger);
    socket.set(zmq::sockopt::rcvtimeo, settings.rcvtimeo);
    socket.set(zmq::sockopt::sndtimeo, settings.sndtimeo);
    socket.set(zmq::sockopt::sndhwm, settings.sndhwm);
    socket.set(zmq::sockopt::rcvhwm, settings.rcvhwm);
}

/**
 * Create and configure a ROUTER socket for servers.
 */
inline zmq::socket_t create_server_socket(zmq::context_t& ctx, int port,
                                          const ZmqSettings& settings = ZmqSettings::server()) {
    zmq::socket_t socket(ctx, zmq::socket_type::router);
    apply_zmq_settings(socket, settings);
    socket.bind("tcp://*:" + std::to_string(port));
    return socket;
}

/**
 * Create and configure a DEALER socket for coordinators.
 */
inline zmq::socket_t create_coordinator_socket(zmq::context_t& ctx,
                                                const std::string& server_addr,
                                                const std::string& identity,
                                                const ZmqSettings& settings = ZmqSettings::coordinator()) {
    zmq::socket_t socket(ctx, zmq::socket_type::dealer);
    apply_zmq_settings(socket, settings);
    socket.set(zmq::sockopt::routing_id, identity);
    socket.connect(server_addr);
    return socket;
}

/**
 * Coordinator-level timing metrics.
 * Coordinator-side timing of one search.
 */
struct CoordinatorTimingMetrics {
    double serialize_us = 0;     // Serialization time
    double send_us = 0;          // Time to send to socket
    double poll_wait_us = 0;     // Time spent in zmq_poll (waiting)
    double recv_us = 0;          // Time to receive from socket
    double deserialize_us = 0;   // Deserialization time
    double process_us = 0;       // Time to process response and update state
    uint32_t send_count = 0;     // Number of sends
    uint32_t recv_count = 0;     // Number of receives
    uint32_t poll_count = 0;     // Number of poll calls

    // No poll-to-recv counters: the coordinator's wait inside zmq::poll() is not
    // part of measured_coord_total_us (see there).
};

/**
 * Per-hop/per-server timing breakdown.
 * Used for detailed latency analysis.
 */
struct DetailedHopTiming {
    // Basic timing
    double total_us = 0;              // Total round-trip time
    double server_processing_us = 0;  // Server-reported processing time
    double queue_wait_us = 0;         // Server-reported queue wait time
    double network_us = 0;            // Calculated: total - measured_server_total

    // Server I/O thread timing (from response)
    double zmq_recv_us = 0;           // Server: ZMQ recv() time
    double zmq_send_us = 0;           // Server: ZMQ send() time
    double serialize_us = 0;          // Server: serialize time
    double deserialize_us = 0;        // Server: deserialize time
    double response_queue_wait_us = 0; // Server: response queue wait
    uint32_t queue_depth = 0;         // Queue depth when enqueued

    // Coordinator-side timing
    double coord_serialize_us = 0;    // Coordinator: serialize time
    double coord_send_us = 0;         // Coordinator: send() time
    // (no poll-to-recv field: see measured_coord_total_us)
    double coord_recv_us = 0;         // Coordinator: recv() time
    double coord_deserialize_us = 0;  // Coordinator: deserialize time

    /**
     * Calculate measured server total (all server-side overhead).
     */
    double measured_server_total_us() const {
        return server_processing_us + zmq_recv_us + zmq_send_us +
               serialize_us + deserialize_us + response_queue_wait_us;
    }

    /**
     * Calculate measured coordinator total (all coord-side overhead).
     */
    double measured_coord_total_us() const {
        // coord_send_us is excluded: the wall-clock window around socket.send()
        // overlaps server and wire activity that queue_wait_us +
        // measured_server_total_us already count, so including it would count
        // the same time twice (and could make the derived network time negative).
        // The coordinator's wait inside zmq::poll() for the next reply is
        // excluded for the same reason: it overlaps the server's work on the
        // in-flight hop. The cost is a small positive bias (about 1 us) on
        // pure_network_us.
        return coord_serialize_us +
               coord_recv_us + coord_deserialize_us;
    }

    /**
     * Calculate pure network time (actual TCP RTT estimate).
     * This subtracts all measured overhead from total RTT.
     */
    double pure_network_us() const {
        return total_us - queue_wait_us - measured_server_total_us() - measured_coord_total_us();
    }
};

/**
 * Helper to print detailed timing breakdown.
 */
inline void print_detailed_timing_breakdown(const DetailedHopTiming& t, std::ostream& os = std::cout) {
    os << "    Total RTT:           " << t.total_us / 1000.0 << " ms\n"
       << "    Queue wait:          " << t.queue_wait_us / 1000.0 << " ms\n"
       << "    Server processing:   " << t.server_processing_us / 1000.0 << " ms\n"
       << "    Server I/O:\n"
       << "      - ZMQ recv:        " << t.zmq_recv_us << " us\n"
       << "      - ZMQ send:        " << t.zmq_send_us << " us\n"
       << "      - Serialize:       " << t.serialize_us << " us\n"
       << "      - Deserialize:     " << t.deserialize_us << " us\n"
       << "      - Resp queue wait: " << t.response_queue_wait_us << " us\n"
       << "    Coordinator I/O:\n"
       << "      - Serialize:       " << t.coord_serialize_us << " us\n"
       << "      - Send:            " << t.coord_send_us << " us\n"
       << "      - Recv:            " << t.coord_recv_us << " us\n"
       << "      - Deserialize:     " << t.coord_deserialize_us << " us\n"
       << "    ---------------------------------\n"
       << "    Measured server:     " << t.measured_server_total_us() / 1000.0 << " ms\n"
       << "    Measured coord:      " << t.measured_coord_total_us() / 1000.0 << " ms\n"
       << "    Pure network:        " << t.pure_network_us() / 1000.0 << " ms\n";
}

}  // namespace ann
