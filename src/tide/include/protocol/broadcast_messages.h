#pragma once
/**
 * Broadcast and Backlink Protocol Messages
 *
 * Messages for cross-partition coordination during insert operations.
 */

#include <vector>
#include <cstring>
#include "protocol/message_types.h"

namespace ann {

/**
 * Batch backlink request header - for adding multiple back-edges in a single request.
 * Bundles edges going to the same partition for efficient processing.
 *
 * Layout: [BatchBacklinkHeader][to_ids uint32_t[num_edges]][vector float[dim]]
 *
 * from_id: The newly inserted node (backlink source)
 * to_ids[i]: Nodes that should add from_id to their adjacency lists
 *
 * Example: INSERT 500000 → neighbors [100, 200, 300] (all in partition 0)
 *          from_id=500000, to_ids=[100,200,300]
 *          → adj[100], adj[200], adj[300] will add 500000
 */
struct BatchBacklinkHeader {
    MessageType msg_type;     // BATCH_BACKLINK_REQUEST
    uint8_t padding[3];       // Padding for alignment
    uint32_t batch_id;        // For ACK matching
    uint32_t num_edges;       // Number of to_ids in this batch
    uint32_t from_id;         // New node ID (the inserted node)
    uint32_t from_partition;  // Partition where from_id was inserted
    uint32_t dim;             // Vector dimension
    uint32_t R;               // Max out-degree for robust pruning
    float alpha;              // Alpha parameter for robust pruning
} __attribute__((packed));

static_assert(sizeof(BatchBacklinkHeader) == 32, "BatchBacklinkHeader must be 32 bytes");

/**
 * Batch backlink ACK header - acknowledgement for batch backlink request.
 */
struct BatchBacklinkAckHeader {
    MessageType msg_type;     // BATCH_BACKLINK_ACK
    uint8_t padding[3];       // Padding for alignment
    uint32_t batch_id;        // Matches the request batch_id
    uint32_t edges_processed; // Number of edges actually processed
    uint32_t lock_wait_us;    // Total lock wait time for this batch
    uint32_t processing_us;   // Processing time (excluding lock wait)
} __attribute__((packed));

static_assert(sizeof(BatchBacklinkAckHeader) == 20, "BatchBacklinkAckHeader must be 20 bytes");

/**
 * Serialize batch backlink request.
 * Includes multiple to_ids and the vector data.
 */
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

/**
 * Serialize batch backlink ACK.
 */
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

/**
 * Vector broadcast header - for broadcasting vector data to all partitions.
 * Unlike backlinks, this only stores vector data without adding any edges.
 * This ensures all partitions can compute distances for inserted vectors.
 *
 * Layout: [VectorBroadcastHeader][vector float[dim]]
 */
struct VectorBroadcastHeader {
    MessageType msg_type;     // VECTOR_BROADCAST
    uint8_t padding[3];       // zero
    uint32_t node_id;         // ID of the inserted node
    uint32_t dim;             // Vector dimension
    uint32_t home_partition;  // Partition where the node was inserted
} __attribute__((packed));

static_assert(sizeof(VectorBroadcastHeader) == 16, "VectorBroadcastHeader must be 16 bytes");

/**
 * Vector broadcast ACK - echoes node_id for matching.
 */
struct VectorBroadcastAckHeader {
    MessageType msg_type;     // VECTOR_BROADCAST_ACK
    uint8_t padding[3];
    uint32_t node_id;         // Echoed from request for ACK matching
} __attribute__((packed));

static_assert(sizeof(VectorBroadcastAckHeader) == 8, "VectorBroadcastAckHeader must be 8 bytes");

/**
 * Serialize vector broadcast request.
 */
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

/**
 * Check message type from raw buffer.
 * Returns SEARCH_REQUEST for an empty message.
 */
inline MessageType get_message_type(const char* data, size_t size) {
    if (size < 1) return MessageType::SEARCH_REQUEST;

    // First byte is MessageType (uint8_t)
    uint8_t type_val = static_cast<uint8_t>(data[0]);
    return static_cast<MessageType>(type_val);
}

}  // namespace ann
