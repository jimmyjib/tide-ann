#pragma once

#include <vector>
#include <cstring>
#include "protocol/message_types.h"

namespace ann {

struct InsertHeader {
    MessageType msg_type;
    uint8_t padding[3];
    uint32_t request_id;
    uint32_t dim;
    uint32_t num_candidates;
    uint32_t R;
    float alpha;
    uint32_t target_partition;
    uint32_t assigned_global_id;
} __attribute__((packed));

static_assert(sizeof(InsertHeader) == 32, "InsertHeader must be 32 bytes");

// insert response with timing
struct InsertResponseHeader {
    MessageType msg_type;
    uint8_t padding[3];
    uint32_t request_id;
    uint32_t assigned_id;
    uint32_t success;
    uint32_t num_neighbors;
    uint32_t num_cross_partition;
    uint32_t processing_us;
    uint32_t queue_wait_us;
    uint32_t deserialize_us;
    uint32_t alloc_lock_wait_us;
    uint32_t edge_lock_wait_us;
    uint32_t prune_us;
    uint32_t alloc_memory_us;
    // resize counts per store
    uint32_t resize_full_base_count;
    uint32_t resize_full_base_us;
    uint32_t resize_membership_count;
    uint32_t resize_membership_us;
    uint32_t resize_adj_count;
    uint32_t resize_adj_us;
    uint32_t resize_global_to_local_count;
    uint32_t resize_global_to_local_us;
    uint32_t edge_memory_us;
    uint32_t local_backlink_prune_us;
} __attribute__((packed));

static_assert(sizeof(InsertResponseHeader) == 92, "InsertResponseHeader must be 92 bytes");

struct CrossPartitionEdge {
    uint32_t from_id;
    uint32_t to_id;
    uint32_t from_partition;
    uint32_t to_partition;
} __attribute__((packed));

static_assert(sizeof(CrossPartitionEdge) == 16, "CrossPartitionEdge must be 16 bytes");

// resize counts of one store
struct ResizeStatsPerVector {
    uint32_t count = 0;
    uint32_t time_us = 0;
};

struct InsertTimingBreakdown {
    uint32_t queue_wait_us = 0;
    uint32_t deserialize_us = 0;
    uint32_t alloc_lock_wait_us = 0;
    uint32_t edge_lock_wait_us = 0;
    uint32_t prune_us = 0;
    uint32_t alloc_memory_us = 0;
    ResizeStatsPerVector resize_full_base;
    ResizeStatsPerVector resize_membership;
    ResizeStatsPerVector resize_adj;
    ResizeStatsPerVector resize_global_to_local;
    uint32_t edge_memory_us = 0;
    uint32_t local_backlink_prune_us = 0;
};

inline std::vector<char> serialize_insert_request(
    uint32_t request_id,
    const float* vector,
    uint32_t dim,
    const std::vector<uint32_t>& candidates,
    const std::vector<float>& distances,
    uint32_t R,
    float alpha,
    uint32_t target_partition,
    uint32_t assigned_global_id
) {
    uint32_t num_candidates = static_cast<uint32_t>(candidates.size());
    size_t total_size = sizeof(InsertHeader) +
                        dim * sizeof(float) +
                        num_candidates * sizeof(uint32_t) +
                        num_candidates * sizeof(float);
    std::vector<char> buffer(total_size);
    char* ptr = buffer.data();

    InsertHeader header;
    header.msg_type = MessageType::INSERT_REQUEST;
    header.request_id = request_id;
    header.dim = dim;
    header.num_candidates = num_candidates;
    header.R = R;
    header.alpha = alpha;
    header.target_partition = target_partition;
    header.assigned_global_id = assigned_global_id;

    std::memcpy(ptr, &header, sizeof(InsertHeader));
    ptr += sizeof(InsertHeader);

    std::memcpy(ptr, vector, dim * sizeof(float));
    ptr += dim * sizeof(float);

    std::memcpy(ptr, candidates.data(), num_candidates * sizeof(uint32_t));
    ptr += num_candidates * sizeof(uint32_t);

    std::memcpy(ptr, distances.data(), num_candidates * sizeof(float));

    return buffer;
}

inline bool deserialize_insert_request(
    const char* data, size_t size,
    InsertHeader& header,
    std::vector<float>& vector,
    std::vector<uint32_t>& candidates,
    std::vector<float>& distances
) {
    if (size < sizeof(InsertHeader)) return false;

    std::memcpy(&header, data, sizeof(InsertHeader));
    if (header.msg_type != MessageType::INSERT_REQUEST) return false;

    size_t expected_size = sizeof(InsertHeader) +
                           header.dim * sizeof(float) +
                           header.num_candidates * sizeof(uint32_t) +
                           header.num_candidates * sizeof(float);
    if (size < expected_size) return false;

    const char* ptr = data + sizeof(InsertHeader);

    vector.resize(header.dim);
    std::memcpy(vector.data(), ptr, header.dim * sizeof(float));
    ptr += header.dim * sizeof(float);

    candidates.resize(header.num_candidates);
    std::memcpy(candidates.data(), ptr, header.num_candidates * sizeof(uint32_t));
    ptr += header.num_candidates * sizeof(uint32_t);

    distances.resize(header.num_candidates);
    std::memcpy(distances.data(), ptr, header.num_candidates * sizeof(float));

    return true;
}

inline std::vector<char> serialize_insert_response(
    uint32_t request_id,
    uint32_t assigned_id,
    bool success,
    const std::vector<uint32_t>& neighbors,
    const std::vector<CrossPartitionEdge>& cross_partition_edges,
    uint32_t processing_us,
    const InsertTimingBreakdown& timing = InsertTimingBreakdown{}
) {
    size_t total_size = sizeof(InsertResponseHeader) +
                        neighbors.size() * sizeof(uint32_t) +
                        cross_partition_edges.size() * sizeof(CrossPartitionEdge);

    std::vector<char> buffer(total_size);
    char* ptr = buffer.data();

    InsertResponseHeader header;
    std::memset(&header, 0, sizeof(header));
    header.msg_type = MessageType::INSERT_RESPONSE;
    header.request_id = request_id;
    header.assigned_id = assigned_id;
    header.success = success ? 1 : 0;
    header.num_neighbors = static_cast<uint32_t>(neighbors.size());
    header.num_cross_partition = static_cast<uint32_t>(cross_partition_edges.size());
    header.processing_us = processing_us;
    header.queue_wait_us = timing.queue_wait_us;
    header.deserialize_us = timing.deserialize_us;
    header.alloc_lock_wait_us = timing.alloc_lock_wait_us;
    header.edge_lock_wait_us = timing.edge_lock_wait_us;
    header.prune_us = timing.prune_us;
    header.alloc_memory_us = timing.alloc_memory_us;
    header.resize_full_base_count = timing.resize_full_base.count;
    header.resize_full_base_us = timing.resize_full_base.time_us;
    header.resize_membership_count = timing.resize_membership.count;
    header.resize_membership_us = timing.resize_membership.time_us;
    header.resize_adj_count = timing.resize_adj.count;
    header.resize_adj_us = timing.resize_adj.time_us;
    header.resize_global_to_local_count = timing.resize_global_to_local.count;
    header.resize_global_to_local_us = timing.resize_global_to_local.time_us;
    header.edge_memory_us = timing.edge_memory_us;
    header.local_backlink_prune_us = timing.local_backlink_prune_us;

    std::memcpy(ptr, &header, sizeof(InsertResponseHeader));
    ptr += sizeof(InsertResponseHeader);

    if (!neighbors.empty()) {
        std::memcpy(ptr, neighbors.data(), neighbors.size() * sizeof(uint32_t));
        ptr += neighbors.size() * sizeof(uint32_t);
    }

    if (!cross_partition_edges.empty()) {
        std::memcpy(ptr, cross_partition_edges.data(),
                    cross_partition_edges.size() * sizeof(CrossPartitionEdge));
    }

    return buffer;
}

inline bool deserialize_insert_response(
    const char* data, size_t size,
    InsertResponseHeader& header,
    std::vector<uint32_t>& neighbors,
    std::vector<CrossPartitionEdge>& cross_partition_edges
) {
    if (size < sizeof(InsertResponseHeader)) return false;

    std::memcpy(&header, data, sizeof(InsertResponseHeader));
    if (header.msg_type != MessageType::INSERT_RESPONSE) return false;

    const char* ptr = data + sizeof(InsertResponseHeader);

    neighbors.resize(header.num_neighbors);
    if (header.num_neighbors > 0) {
        std::memcpy(neighbors.data(), ptr, header.num_neighbors * sizeof(uint32_t));
        ptr += header.num_neighbors * sizeof(uint32_t);
    }

    cross_partition_edges.resize(header.num_cross_partition);
    if (header.num_cross_partition > 0) {
        std::memcpy(cross_partition_edges.data(), ptr,
                    header.num_cross_partition * sizeof(CrossPartitionEdge));
    }

    return true;
}

}  // namespace ann
