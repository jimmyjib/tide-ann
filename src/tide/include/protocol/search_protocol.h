#pragma once

#include <vector>
#include <cstring>
#include <string>
#include "protocol/message_types.h"

namespace ann {

struct StateHeader {
    uint32_t query_id;
    uint32_t L;
    uint32_t K;
    uint32_t W;
    uint32_t current_partition;
    uint32_t beam_size;
    uint32_t dim;
    uint32_t flags;
    uint32_t origin_server;
    uint32_t hop_count;
    uint32_t cumulative_queue_us;
    uint32_t cumulative_processing_us;
    // P2P: the coordinator sees only the last reply, so the state carries the distance count
    uint32_t cumulative_distance_computations;
} __attribute__((packed));

static_assert(sizeof(StateHeader) == 52, "StateHeader must be 52 bytes");

constexpr uint32_t STATE_FLAG_P2P = 0x1;
// route traces are opt-in
constexpr uint32_t STATE_FLAG_TRACE_ROUTE = 0x2;

struct RouteHop {
    uint32_t partition;
    uint64_t arrive_us;   // steady clock (us) at receive
    uint64_t depart_us;   // steady clock (us) at send
} __attribute__((packed));
static_assert(sizeof(RouteHop) == 20, "RouteHop must be 20 bytes");

enum class StateStatus : uint32_t {
    NEED_HOP = 0,
    DONE = 1,
    ERROR = 2
};

struct StateResponseHeader {
    uint32_t query_id;
    StateStatus status;
    uint32_t next_partition;
    uint32_t beam_size;
    uint32_t result_count;
    uint32_t distance_computations;
    uint32_t inter_partition_hops;
    uint32_t io_processing_us;  // receive end to enqueue

    uint32_t server_processing_us;
    uint32_t pq_distance_count;
    uint32_t full_distance_count;
    uint32_t beam_size_in;
    uint32_t local_nodes_expanded;
    uint32_t top_W_count;
    uint32_t neighbors_evaluated;
    uint32_t queue_wait_us;

    // P2P cumulative timing
    uint32_t cumulative_queue_us;
    uint32_t cumulative_processing_us;

    uint32_t zmq_recv_us;
    uint32_t zmq_send_us;
    uint32_t serialize_us;
    uint32_t deserialize_us;
    uint32_t queue_depth;
    uint32_t response_queue_wait_us;

    uint32_t shared_lock_count;
    uint32_t unique_lock_count;
    uint32_t shared_lock_wait_us;
    uint32_t unique_lock_wait_us;

    // unmeasured worker time
    uint32_t worker_overhead_us;
    uint32_t send_prep_us;  // pop to send start, from the previous reply
} __attribute__((packed));

static_assert(sizeof(StateResponseHeader) == 120, "StateResponseHeader must be 120 bytes");

constexpr uint32_t STATE_WIRE_MAGIC = 0x54494445;
// Version of the message structs below, printed by wire_format_string() with their sizes.
constexpr uint32_t STATE_WIRE_VERSION = 6;

// Message formats of this build: wire version and struct sizes.
inline std::string wire_format_string() {
    return "tide-state-wire|magic=" +
           std::to_string(STATE_WIRE_MAGIC) +
           "|version=" + std::to_string(STATE_WIRE_VERSION) +
           "|state_header=" + std::to_string(sizeof(StateHeader)) +
           "|response_header=" +
           std::to_string(sizeof(StateResponseHeader)) +
           "|beam=" + std::to_string(sizeof(BeamEntry)) +
           "|route=" + std::to_string(sizeof(RouteHop)) +
           "|stats_request=" + std::to_string(sizeof(StatsRequest)) +
           "|stats_response=" + std::to_string(sizeof(StatsResponseHeader)) +
           "|migration_ack=" +
           std::to_string(sizeof(MembershipMigrationAck)) +
           "|membership_check=" +
           std::to_string(sizeof(MembershipCheckResponse)) +
           "|stats_window_begin=" +
           std::to_string(sizeof(StatsWindowBeginRequest)) +
           "|stats_window_ack=" +
           std::to_string(sizeof(StatsWindowAck));
}

struct ResponseMetrics {
    uint32_t server_processing_us = 0;
    uint32_t pq_distance_count = 0;
    uint32_t full_distance_count = 0;
    uint32_t beam_size_in = 0;
    uint32_t local_nodes_expanded = 0;
    uint32_t top_W_count = 0;
    uint32_t neighbors_evaluated = 0;
    uint32_t queue_wait_us = 0;
    uint32_t cumulative_queue_us = 0;
    uint32_t cumulative_processing_us = 0;
    uint32_t zmq_recv_us = 0;
    uint32_t zmq_send_us = 0;
    uint32_t serialize_us = 0;
    uint32_t deserialize_us = 0;
    uint32_t queue_depth = 0;
    uint32_t response_queue_wait_us = 0;
    uint32_t shared_lock_count = 0;
    uint32_t unique_lock_count = 0;
    uint32_t shared_lock_wait_us = 0;
    uint32_t unique_lock_wait_us = 0;
    uint32_t io_processing_us = 0;  // receive end to enqueue
    uint32_t worker_overhead_us = 0;  // unmeasured worker time
    uint32_t send_prep_us = 0;  // pop to send start, from the previous reply
};

inline std::vector<char> serialize_state(
    uint32_t query_id,
    uint32_t L, uint32_t K, uint32_t W,
    uint32_t current_partition,
    const float* query, uint32_t dim,
    const std::vector<BeamEntry>& beam,
    uint32_t flags = 0,
    uint32_t origin_server = 0,
    uint32_t hop_count = 0,
    uint32_t cumulative_queue_us = 0,
    uint32_t cumulative_processing_us = 0,
    const std::vector<uint32_t>& seen_nodes = {},
    const std::vector<RouteHop>& route = {},
    uint32_t cumulative_distance_computations = 0
) {
    size_t header_size = sizeof(StateHeader);
    size_t query_size = dim * sizeof(float);
    size_t beam_bytes = beam.size() * sizeof(BeamEntry);
    size_t seen_bytes = seen_nodes.size() * sizeof(uint32_t);
    size_t route_bytes = route.size() * sizeof(RouteHop);
    size_t total_size = 1 + header_size + query_size + beam_bytes +
                        sizeof(uint32_t) + seen_bytes +
                        sizeof(uint32_t) + route_bytes;

    std::vector<char> buffer(total_size);
    char* ptr = buffer.data();

    *ptr = static_cast<char>(MessageType::SEARCH_REQUEST);
    ptr++;

    StateHeader header;
    header.query_id = query_id;
    header.L = L;
    header.K = K;
    header.W = W;
    header.current_partition = current_partition;
    header.beam_size = static_cast<uint32_t>(beam.size());
    header.dim = dim;
    header.flags = flags;
    header.origin_server = origin_server;
    header.hop_count = hop_count;
    header.cumulative_queue_us = cumulative_queue_us;
    header.cumulative_processing_us = cumulative_processing_us;
    header.cumulative_distance_computations =
        cumulative_distance_computations;
    std::memcpy(ptr, &header, header_size);
    ptr += header_size;

    std::memcpy(ptr, query, query_size);
    ptr += query_size;

    if (!beam.empty()) {
        std::memcpy(ptr, beam.data(), beam_bytes);
        ptr += beam_bytes;
    }

    // seen nodes
    uint32_t seen_count = static_cast<uint32_t>(seen_nodes.size());
    std::memcpy(ptr, &seen_count, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    if (!seen_nodes.empty()) {
        std::memcpy(ptr, seen_nodes.data(), seen_bytes);
        ptr += seen_bytes;
    }

    // route
    uint32_t route_count = static_cast<uint32_t>(route.size());
    std::memcpy(ptr, &route_count, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    if (!route.empty()) {
        std::memcpy(ptr, route.data(), route_bytes);
    }

    return buffer;
}

// State a server forwards after a P2P NEED_HOP: hop_count + 1 and the cumulative counters advanced
// by this hop (wrapping like the 32-bit wire fields).
inline std::vector<char> serialize_forward_state(
    const StateHeader& incoming,
    uint32_t next_partition,
    const float* query,
    const std::vector<BeamEntry>& beam,
    uint32_t hop_queue_us,
    uint32_t hop_processing_us,
    uint32_t hop_distance_computations,
    const std::vector<uint32_t>& seen_nodes,
    const std::vector<RouteHop>& route
) {
    return serialize_state(
        incoming.query_id,
        incoming.L, incoming.K, incoming.W,
        next_partition,
        query, incoming.dim,
        beam,
        incoming.flags,
        incoming.origin_server,
        incoming.hop_count + 1,
        incoming.cumulative_queue_us + hop_queue_us,
        incoming.cumulative_processing_us + hop_processing_us,
        seen_nodes,
        route,
        incoming.cumulative_distance_computations + hop_distance_computations);
}

inline bool deserialize_state(
    const char* data, size_t size,
    StateHeader& header,
    std::vector<float>& query,
    std::vector<BeamEntry>& beam,
    std::vector<uint32_t>& seen_nodes,
    std::vector<RouteHop>& route
) {
    if (size < sizeof(StateHeader)) return false;

    const char* ptr = data;
    const char* end = data + size;
    std::memcpy(&header, ptr, sizeof(StateHeader));
    ptr += sizeof(StateHeader);

    size_t expected_size = sizeof(StateHeader) +
                           header.dim * sizeof(float) +
                           header.beam_size * sizeof(BeamEntry);
    if (size < expected_size) return false;

    query.resize(header.dim);
    std::memcpy(query.data(), ptr, header.dim * sizeof(float));
    ptr += header.dim * sizeof(float);

    beam.resize(header.beam_size);
    if (header.beam_size > 0) {
        std::memcpy(beam.data(), ptr, header.beam_size * sizeof(BeamEntry));
        ptr += header.beam_size * sizeof(BeamEntry);
    }

    // seen nodes (optional)
    seen_nodes.clear();
    if (ptr + sizeof(uint32_t) <= end) {
        uint32_t seen_count = 0;
        std::memcpy(&seen_count, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);
        if (seen_count > 0 && ptr + seen_count * sizeof(uint32_t) <= end) {
            seen_nodes.resize(seen_count);
            std::memcpy(seen_nodes.data(), ptr, seen_count * sizeof(uint32_t));
            ptr += seen_count * sizeof(uint32_t);
        }
    }

    // route (optional)
    route.clear();
    if (ptr + sizeof(uint32_t) <= end) {
        uint32_t route_count = 0;
        std::memcpy(&route_count, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);
        if (route_count > 0 && ptr + route_count * sizeof(RouteHop) <= end) {
            route.resize(route_count);
            std::memcpy(route.data(), ptr, route_count * sizeof(RouteHop));
        }
    }

    return true;
}

inline std::vector<char> serialize_response(
    uint32_t query_id,
    StateStatus status,
    uint32_t next_partition,
    const std::vector<BeamEntry>& beam,
    const std::vector<uint32_t>& neighbors,
    const std::vector<float>& distances,
    uint32_t distance_computations,
    uint32_t inter_partition_hops,
    const ResponseMetrics& metrics = ResponseMetrics{},
    const std::vector<uint32_t>& seen_nodes = {},
    const std::vector<RouteHop>& route = {}
) {
    StateResponseHeader header;
    header.query_id = query_id;
    header.status = status;
    header.next_partition = next_partition;
    header.beam_size = static_cast<uint32_t>(beam.size());
    header.result_count = static_cast<uint32_t>(neighbors.size());
    header.distance_computations = distance_computations;
    header.inter_partition_hops = inter_partition_hops;
    header.io_processing_us = metrics.io_processing_us;

    header.server_processing_us = metrics.server_processing_us;
    header.pq_distance_count = metrics.pq_distance_count;
    header.full_distance_count = metrics.full_distance_count;
    header.beam_size_in = metrics.beam_size_in;
    header.local_nodes_expanded = metrics.local_nodes_expanded;
    header.top_W_count = metrics.top_W_count;
    header.neighbors_evaluated = metrics.neighbors_evaluated;
    header.queue_wait_us = metrics.queue_wait_us;
    header.cumulative_queue_us = metrics.cumulative_queue_us;
    header.cumulative_processing_us = metrics.cumulative_processing_us;
    header.zmq_recv_us = metrics.zmq_recv_us;
    header.zmq_send_us = metrics.zmq_send_us;
    header.serialize_us = metrics.serialize_us;
    header.deserialize_us = metrics.deserialize_us;
    header.queue_depth = metrics.queue_depth;
    header.response_queue_wait_us = metrics.response_queue_wait_us;
    header.shared_lock_count = metrics.shared_lock_count;
    header.unique_lock_count = metrics.unique_lock_count;
    header.shared_lock_wait_us = metrics.shared_lock_wait_us;
    header.unique_lock_wait_us = metrics.unique_lock_wait_us;
    header.worker_overhead_us = metrics.worker_overhead_us;
    header.send_prep_us = metrics.send_prep_us;

    size_t seen_bytes = seen_nodes.size() * sizeof(uint32_t);
    size_t route_bytes = route.size() * sizeof(RouteHop);
    size_t total_size = 1 + sizeof(StateResponseHeader) +
                        beam.size() * sizeof(BeamEntry) +
                        neighbors.size() * sizeof(uint32_t) +
                        distances.size() * sizeof(float) +
                        sizeof(uint32_t) + seen_bytes +
                        sizeof(uint32_t) + route_bytes;

    std::vector<char> buffer(total_size);
    char* ptr = buffer.data();

    *ptr = static_cast<char>(MessageType::SEARCH_RESPONSE);
    ptr++;

    std::memcpy(ptr, &header, sizeof(StateResponseHeader));
    ptr += sizeof(StateResponseHeader);

    if (!beam.empty()) {
        std::memcpy(ptr, beam.data(), beam.size() * sizeof(BeamEntry));
        ptr += beam.size() * sizeof(BeamEntry);
    }

    if (!neighbors.empty()) {
        std::memcpy(ptr, neighbors.data(), neighbors.size() * sizeof(uint32_t));
        ptr += neighbors.size() * sizeof(uint32_t);
    }

    if (!distances.empty()) {
        std::memcpy(ptr, distances.data(), distances.size() * sizeof(float));
        ptr += distances.size() * sizeof(float);
    }

    // seen nodes
    uint32_t seen_count = static_cast<uint32_t>(seen_nodes.size());
    std::memcpy(ptr, &seen_count, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    if (!seen_nodes.empty()) {
        std::memcpy(ptr, seen_nodes.data(), seen_bytes);
        ptr += seen_bytes;
    }

    // route
    uint32_t route_count = static_cast<uint32_t>(route.size());
    std::memcpy(ptr, &route_count, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    if (!route.empty()) {
        std::memcpy(ptr, route.data(), route_bytes);
    }

    return buffer;
}

// Reply to a rejected search state (another dimension): status ERROR, no beam, no result.
inline std::vector<char> serialize_search_error(uint32_t query_id, uint32_t hop_count) {
    return serialize_response(query_id, StateStatus::ERROR, UINT32_MAX, {}, {}, {}, 0, hop_count + 1);
}

inline bool deserialize_response(
    const char* data, size_t size,
    StateResponseHeader& header,
    std::vector<BeamEntry>& beam,
    std::vector<uint32_t>& neighbors,
    std::vector<float>& distances,
    std::vector<uint32_t>& seen_nodes,
    std::vector<RouteHop>& route,
    size_t* consumed_out = nullptr
) {
    if (size < sizeof(StateResponseHeader)) return false;

    const char* ptr = data;
    const char* end = data + size;
    std::memcpy(&header, ptr, sizeof(StateResponseHeader));
    ptr += sizeof(StateResponseHeader);

    beam.resize(header.beam_size);
    if (header.beam_size > 0) {
        std::memcpy(beam.data(), ptr, header.beam_size * sizeof(BeamEntry));
        ptr += header.beam_size * sizeof(BeamEntry);
    }

    if (header.status == StateStatus::DONE && header.result_count > 0) {
        neighbors.resize(header.result_count);
        distances.resize(header.result_count);
        std::memcpy(neighbors.data(), ptr, header.result_count * sizeof(uint32_t));
        ptr += header.result_count * sizeof(uint32_t);
        std::memcpy(distances.data(), ptr, header.result_count * sizeof(float));
        ptr += header.result_count * sizeof(float);
    }

    // seen nodes (optional)
    seen_nodes.clear();
    if (ptr + sizeof(uint32_t) <= end) {
        uint32_t seen_count = 0;
        std::memcpy(&seen_count, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);
        if (seen_count > 0 && ptr + seen_count * sizeof(uint32_t) <= end) {
            seen_nodes.resize(seen_count);
            std::memcpy(seen_nodes.data(), ptr, seen_count * sizeof(uint32_t));
            ptr += seen_count * sizeof(uint32_t);
        }
    }

    // route (optional)
    route.clear();
    if (ptr + sizeof(uint32_t) <= end) {
        uint32_t route_count = 0;
        std::memcpy(&route_count, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);
        if (route_count > 0 && ptr + route_count * sizeof(RouteHop) <= end) {
            route.resize(route_count);
            std::memcpy(route.data(), ptr, route_count * sizeof(RouteHop));
        }
    }

    return true;
}

}  // namespace ann
