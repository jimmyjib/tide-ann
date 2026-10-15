// Backlink and vector broadcast messages of the insert path.
#pragma once

#include <vector>
#include <cstring>
#include "protocol/message_types.h"

namespace ann {

// Backlinks of one inserted node to several nodes of one server:
// [BatchBacklinkHeader][to_ids u32[num_edges]][vector float[dim]]. Each to_id gains from_id.
struct BatchBacklinkHeader {
    MessageType msg_type;     // BATCH_BACKLINK_REQUEST
    uint8_t padding[3];
    uint32_t batch_id;        // echoed in the ACK
    uint32_t num_edges;
    uint32_t from_id;         // the inserted node
    uint32_t from_partition;  // its owner
    uint32_t dim;
    uint32_t R;               // pruning degree
    float alpha;              // pruning alpha
} __attribute__((packed));

static_assert(sizeof(BatchBacklinkHeader) == 32, "BatchBacklinkHeader must be 32 bytes");

struct BatchBacklinkAckHeader {
    MessageType msg_type;     // BATCH_BACKLINK_ACK
    uint8_t padding[3];
    uint32_t batch_id;
    uint32_t edges_processed;
    uint32_t lock_wait_us;
    uint32_t processing_us;   // excluding lock wait
} __attribute__((packed));

static_assert(sizeof(BatchBacklinkAckHeader) == 20, "BatchBacklinkAckHeader must be 20 bytes");

inline std::vector<char> serialize_batch_backlink_request(
    uint32_t batch_id,
    uint32_t from_id,
    uint32_t from_partition,
    uint32_t dim,
    uint32_t R,
    float alpha,
    const std::vector<uint32_t>& to_ids,
    const float* vector
) {
    size_t total_size = sizeof(BatchBacklinkHeader) +
                        to_ids.size() * sizeof(uint32_t) +
                        dim * sizeof(float);
    std::vector<char> buffer(total_size);

    BatchBacklinkHeader header;
    header.msg_type = MessageType::BATCH_BACKLINK_REQUEST;
    std::memset(header.padding, 0, sizeof(header.padding));
    header.batch_id = batch_id;
    header.num_edges = static_cast<uint32_t>(to_ids.size());
    header.from_id = from_id;
    header.from_partition = from_partition;
    header.dim = dim;
    header.R = R;
    header.alpha = alpha;

    char* ptr = buffer.data();
    std::memcpy(ptr, &header, sizeof(BatchBacklinkHeader));
    ptr += sizeof(BatchBacklinkHeader);

    std::memcpy(ptr, to_ids.data(), to_ids.size() * sizeof(uint32_t));
    ptr += to_ids.size() * sizeof(uint32_t);

    std::memcpy(ptr, vector, dim * sizeof(float));

    return buffer;
}

inline std::vector<char> serialize_batch_backlink_ack(
    uint32_t batch_id,
    uint32_t edges_processed,
    uint32_t lock_wait_us,
    uint32_t processing_us
) {
    std::vector<char> buffer(sizeof(BatchBacklinkAckHeader));

    BatchBacklinkAckHeader header;
    header.msg_type = MessageType::BATCH_BACKLINK_ACK;
    std::memset(header.padding, 0, sizeof(header.padding));
    header.batch_id = batch_id;
    header.edges_processed = edges_processed;
    header.lock_wait_us = lock_wait_us;
    header.processing_us = processing_us;

    std::memcpy(buffer.data(), &header, sizeof(BatchBacklinkAckHeader));
    return buffer;
}

// Vector of an inserted node for every server (no edges): [VectorBroadcastHeader][vector float[dim]]
struct VectorBroadcastHeader {
    MessageType msg_type;     // VECTOR_BROADCAST
    uint8_t padding[3];       // zero
    uint32_t node_id;
    uint32_t dim;
    uint32_t home_partition;  // owner of the node
} __attribute__((packed));

static_assert(sizeof(VectorBroadcastHeader) == 16, "VectorBroadcastHeader must be 16 bytes");

struct VectorBroadcastAckHeader {
    MessageType msg_type;     // VECTOR_BROADCAST_ACK
    uint8_t padding[3];
    uint32_t node_id;         // echoed
} __attribute__((packed));

static_assert(sizeof(VectorBroadcastAckHeader) == 8, "VectorBroadcastAckHeader must be 8 bytes");

inline std::vector<char> serialize_vector_broadcast(
    uint32_t node_id,
    uint32_t dim,
    uint32_t home_partition,
    const float* vector
) {
    size_t total_size = sizeof(VectorBroadcastHeader) + dim * sizeof(float);
    std::vector<char> buffer(total_size);

    VectorBroadcastHeader header;
    header.msg_type = MessageType::VECTOR_BROADCAST;
    std::memset(header.padding, 0, sizeof(header.padding));
    header.node_id = node_id;
    header.dim = dim;
    header.home_partition = home_partition;

    std::memcpy(buffer.data(), &header, sizeof(VectorBroadcastHeader));
    if (vector != nullptr) {
        std::memcpy(buffer.data() + sizeof(VectorBroadcastHeader), vector, dim * sizeof(float));
    }

    return buffer;
}

// SEARCH_REQUEST for an empty message
inline MessageType get_message_type(const char* data, size_t size) {
    if (size < 1) return MessageType::SEARCH_REQUEST;

    uint8_t type_val = static_cast<uint8_t>(data[0]);
    return static_cast<MessageType>(type_val);
}

}  // namespace ann
