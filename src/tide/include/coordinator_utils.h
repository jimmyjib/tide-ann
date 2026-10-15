// Socket and recall helpers shared by the coordinator and the servers.
#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <set>
#include <algorithm>
#include <cstdint>
#include <zmq.hpp>

namespace ann {

// "tcp://host1:port,tcp://host2:port" -> addresses
inline std::vector<std::string> parse_servers(const std::string& servers_str) {
    std::vector<std::string> servers;
    std::stringstream ss(servers_str);
    std::string addr;
    while (std::getline(ss, addr, ',')) {
        servers.push_back(addr);
    }
    return servers;
}

// Recall@k in [0, 1].
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

// Recall@k against int32 ground truth.
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

struct ZmqSettings {
    int sndhwm = 10000;
    int rcvhwm = 10000;
    int sndtimeo = 1000;   // ms
    int rcvtimeo = 0;      // 0 = non-blocking
    int linger = 0;

    // server ROUTER socket
    static ZmqSettings server() {
        return {10000, 10000, 100, 10, 0};
    }

    // coordinator DEALER socket
    static ZmqSettings coordinator() {
        return {10000, 10000, 1000, 0, 0};
    }
};

inline void apply_zmq_settings(zmq::socket_t& socket, const ZmqSettings& settings) {
    socket.set(zmq::sockopt::linger, settings.linger);
    socket.set(zmq::sockopt::rcvtimeo, settings.rcvtimeo);
    socket.set(zmq::sockopt::sndtimeo, settings.sndtimeo);
    socket.set(zmq::sockopt::sndhwm, settings.sndhwm);
    socket.set(zmq::sockopt::rcvhwm, settings.rcvhwm);
}

inline zmq::socket_t create_server_socket(zmq::context_t& ctx, int port,
                                          const ZmqSettings& settings = ZmqSettings::server()) {
    zmq::socket_t socket(ctx, zmq::socket_type::router);
    apply_zmq_settings(socket, settings);
    socket.bind("tcp://*:" + std::to_string(port));
    return socket;
}

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

// Coordinator timing of one search.
struct CoordinatorTimingMetrics {
    double serialize_us = 0;
    double send_us = 0;
    double poll_wait_us = 0;     // in zmq_poll
    double recv_us = 0;
    double deserialize_us = 0;
    double process_us = 0;
    uint32_t send_count = 0;
    uint32_t recv_count = 0;
    uint32_t poll_count = 0;
};

// Round trip of one hop split into server, coordinator and network time.
struct DetailedHopTiming {
    double total_us = 0;              // round trip
    double server_processing_us = 0;
    double queue_wait_us = 0;
    double network_us = 0;            // total - measured_server_total

    // server I/O thread
    double zmq_recv_us = 0;
    double zmq_send_us = 0;
    double serialize_us = 0;
    double deserialize_us = 0;
    double response_queue_wait_us = 0;
    uint32_t queue_depth = 0;         // at enqueue

    // coordinator
    double coord_serialize_us = 0;
    double coord_send_us = 0;
    double coord_recv_us = 0;
    double coord_deserialize_us = 0;

    double measured_server_total_us() const {
        return server_processing_us + zmq_recv_us + zmq_send_us +
               serialize_us + deserialize_us + response_queue_wait_us;
    }

    double measured_coord_total_us() const {
        // see timing_types.h
        return coord_serialize_us +
               coord_recv_us + coord_deserialize_us;
    }

    // round trip minus every measured overhead
    double pure_network_us() const {
        return total_us - queue_wait_us - measured_server_total_us() - measured_coord_total_us();
    }
};

}  // namespace ann
