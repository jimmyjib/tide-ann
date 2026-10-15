#pragma once
/**
 * Message Types and Basic Protocol Structures
 *
 * Core types used by all protocol messages.
 */

#include <cstdint>
#include <cstddef>
#include <atomic>

namespace ann {

/**
 * MessageType - First byte of every message for type disambiguation.
 * Values are stable on the wire. Unused (reserved) values: 4-5, 10-11,
 * 22-28, 34, 37-52, 57, 59-62 and 68.
 */
enum class MessageType : uint8_t {
    SEARCH_REQUEST = 0,
    SEARCH_RESPONSE = 1,
    INSERT_REQUEST = 2,
    INSERT_RESPONSE = 3,
    VECTOR_BROADCAST = 6,
    VECTOR_BROADCAST_ACK = 7,
    RESET_REQUEST = 8,
    RESET_ACK = 9,
    GET_STATS_REQUEST = 12,
    GET_STATS_RESPONSE = 13,
    PING_REQUEST = 14,
    PING_RESPONSE = 15,
    BATCH_BACKLINK_REQUEST = 16,
    BATCH_BACKLINK_ACK = 17,
    DRAIN_REQUEST = 18,    // Flush residual backlink/broadcast queues
    DRAIN_ACK = 19,
    SHUTDOWN_REQUEST = 20, // Graceful, quiescent server shutdown
    SHUTDOWN_ACK = 21,
    MIGRATION_PREPARE_ACK = 29,
    MIGRATION_COMMIT_REQUEST = 30,
    MIGRATION_COMMIT_ACK = 31,
    MIGRATION_ROLLBACK_REQUEST = 32,
    MIGRATION_ROLLBACK_ACK = 33,
    MIGRATION_REFRESH_RESPONSE = 35,
    MIGRATION_PREPARE_INLINE_REQUEST = 36,
    MIGRATION_ACCEPT_REQUEST = 53,
    MIGRATION_ACCEPT_ACK = 54,
    STATS_WINDOW_BEGIN_REQUEST = 55,
    STATS_WINDOW_BEGIN_ACK = 56,
    // HARVEST_ACK is the receipt of a validated snapshot frame, coordinator-internal.
    STATS_WINDOW_HARVEST_ACK = 58,
    STATS_WINDOW_SNAPSHOT_REQUEST = 63,
    STATS_WINDOW_SNAPSHOT_RESPONSE = 64,
    STATS_WINDOW_STOP_REQUEST = 65,
    STATS_WINDOW_STOP_ACK = 66,
    MIGRATION_REFRESH_INLINE_REQUEST = 67,
    // membership-check: the coordinator's owner of every id (type, u32 count,
    // count x u32); each server answers how its own state compares.
    MEMBERSHIP_CHECK_REQUEST = 69,
    MEMBERSHIP_CHECK_RESPONSE = 70
};

enum class AdminAckDisposition : uint8_t {
    ACCEPT,
    DUPLICATE,
    UNMATCHED,
    INVALID_SERVER
};

// Replies of the insert maintenance path. Their owners are the pipeline and the
// drain, which account for each one against its pending entry; any other
// exchange that pops one from the shared response queue must put it back.
inline constexpr bool is_maintenance_reply(MessageType type) {
    return type == MessageType::INSERT_RESPONSE || type == MessageType::BATCH_BACKLINK_ACK ||
           type == MessageType::VECTOR_BROADCAST_ACK;
}

// Whether an admin exchange (send_admin_command) puts back the replies that are
// not its ACK. RESET does not: it clears the pending maps and the servers reload,
// so a pre-reset maintenance reply has no owner afterwards, and a broadcast ACK
// (keyed by node id, which restarts at n_base) could be credited to a new insert.
inline constexpr bool admin_command_requeues_foreign_replies(MessageType request) {
    return request != MessageType::RESET_REQUEST;
}

inline constexpr MessageType admin_ack_type(MessageType request) {
    switch (request) {
        case MessageType::RESET_REQUEST: return MessageType::RESET_ACK;
        case MessageType::DRAIN_REQUEST: return MessageType::DRAIN_ACK;
        case MessageType::SHUTDOWN_REQUEST: return MessageType::SHUTDOWN_ACK;
        case MessageType::MIGRATION_PREPARE_INLINE_REQUEST:
            return MessageType::MIGRATION_PREPARE_ACK;
        case MessageType::MIGRATION_COMMIT_REQUEST:
            return MessageType::MIGRATION_COMMIT_ACK;
        case MessageType::MIGRATION_ROLLBACK_REQUEST:
            return MessageType::MIGRATION_ROLLBACK_ACK;
        case MessageType::MIGRATION_ACCEPT_REQUEST:
            return MessageType::MIGRATION_ACCEPT_ACK;
        case MessageType::STATS_WINDOW_SNAPSHOT_REQUEST:
            return MessageType::STATS_WINDOW_SNAPSHOT_RESPONSE;
        case MessageType::STATS_WINDOW_STOP_REQUEST:
            return MessageType::STATS_WINDOW_STOP_ACK;
        case MessageType::MIGRATION_REFRESH_INLINE_REQUEST:
            return MessageType::MIGRATION_REFRESH_RESPONSE;
        case MessageType::MEMBERSHIP_CHECK_REQUEST:
            return MessageType::MEMBERSHIP_CHECK_RESPONSE;
        default: return request;
    }
}

struct MembershipMigrationAck {
    MessageType msg_type;
    uint8_t status;  // 0 = success, nonzero = rejected
    // MIGRATION_ACCEPT_ACK: the attempt id its request carried (the
    // coordinator discards the ACK of an earlier, timed-out attempt); 0 for
    // every other type.
    uint16_t request_id;
    uint32_t server_rank;
    uint64_t epoch_from;
    uint64_t epoch_to;
    uint64_t move_count;
} __attribute__((packed));

static_assert(sizeof(MembershipMigrationAck) == 32,
              "MembershipMigrationAck must be 32 bytes");


// A server's answer to MEMBERSHIP_CHECK_REQUEST (membership_check in
// membership_migration.h): its own state against the coordinator's owners.
struct MembershipCheckResponse {
    MessageType msg_type;
    uint8_t status;     // 0 compared; 1 not quiescent; 2 malformed request
    uint8_t own_only;   // the server's storage mode
    uint8_t pool_ok;    // every free row slot unmapped, with a zero-capacity row
    uint32_t server_rank;
    uint32_t ids;             // ids compared (the request's count)
    uint32_t server_ids;      // ids the server knows (n_base)
    uint32_t n_local;         // local row slots allocated
    uint32_t free_slots;      // slots in the free row-slot pool
    uint32_t first_mismatch;  // the first id that failed a check, UINT32_MAX for none
    uint32_t padding;
    uint64_t ownership_epoch;
    uint64_t owned;                  // ids the coordinator assigns to this server
    uint64_t mapped;                 // ids with a valid local row here
    uint64_t vectors;                // own-only: ids whose original vector is here
    uint64_t membership_mismatches;  // mem_at differs from the coordinator's owner
    uint64_t mapping_mismatches;     // a row here differs from ownership here
    uint64_t vector_mismatches;      // own-only: a vector here differs from ownership here
    uint64_t missing_codes;          // own-only: ids without a PQ code
    uint64_t vector_digest;          // own-only: sum of node_vector_hash over the vectors held here
} __attribute__((packed));

static_assert(sizeof(MembershipCheckResponse) == 104,
              "MembershipCheckResponse must be 104 bytes");

struct MembershipMigrationRefreshHeader {
    MessageType msg_type;
    uint8_t status;
    uint8_t padding[2];
    uint32_t server_rank;
    uint64_t epoch_from;
    uint64_t epoch_to;
    uint64_t move_count;
    uint64_t payload_bytes;
} __attribute__((packed));

static_assert(sizeof(MembershipMigrationRefreshHeader) == 40,
              "MembershipMigrationRefreshHeader must be 40 bytes");

inline constexpr AdminAckDisposition classify_admin_ack(
    MessageType expected,
    MessageType received,
    std::size_t server_index,
    std::size_t server_count,
    bool already_acked
) {
    if (server_index >= server_count) {
        return AdminAckDisposition::INVALID_SERVER;
    }
    if (received != expected) return AdminAckDisposition::UNMATCHED;
    if (already_acked) return AdminAckDisposition::DUPLICATE;
    return AdminAckDisposition::ACCEPT;
}

inline constexpr bool shutdown_state_is_quiescent(
    std::size_t work_queued,
    std::size_t insert_queued,
    std::size_t backlink_queued,
    std::size_t broadcast_queued,
    std::size_t response_queued,
    uint32_t maintenance_in_flight,
    uint32_t search_in_flight
) {
    return work_queued == 0 && insert_queued == 0 &&
           backlink_queued == 0 && broadcast_queued == 0 &&
           response_queued == 0 && maintenance_in_flight == 0 &&
           search_in_flight == 0;
}

class SearchInFlightOwnership {
public:
    explicit SearchInFlightOwnership(
        std::atomic<uint32_t>& counter) noexcept
        : counter_(counter) {}
    ~SearchInFlightOwnership() { release(); }

    SearchInFlightOwnership(const SearchInFlightOwnership&) = delete;
    SearchInFlightOwnership& operator=(
        const SearchInFlightOwnership&) = delete;

    void acquire_if_search(MessageType type) noexcept {
        if (type == MessageType::SEARCH_REQUEST && !owned_) {
            counter_.fetch_add(1, std::memory_order_acq_rel);
            owned_ = true;
        }
    }

    void release() noexcept {
        if (owned_) {
            counter_.fetch_sub(1, std::memory_order_acq_rel);
            owned_ = false;
        }
    }

    bool owned() const noexcept { return owned_; }

private:
    std::atomic<uint32_t>& counter_;
    bool owned_ = false;
};

class InFlightOwnership {
public:
    explicit InFlightOwnership(
        std::atomic<uint32_t>& counter) noexcept
        : counter_(counter) {}
    ~InFlightOwnership() { release(); }

    InFlightOwnership(const InFlightOwnership&) = delete;
    InFlightOwnership& operator=(const InFlightOwnership&) = delete;

    void acquire() noexcept {
        if (!owned_) {
            counter_.fetch_add(1, std::memory_order_acq_rel);
            owned_ = true;
        }
    }

    void release() noexcept {
        if (owned_) {
            counter_.fetch_sub(1, std::memory_order_acq_rel);
            owned_ = false;
        }
    }

    bool owned() const noexcept { return owned_; }

private:
    std::atomic<uint32_t>& counter_;
    bool owned_ = false;
};

/**
 * BeamEntry: node_id + distance + expanded flag (12 bytes)
 */
struct BeamEntry {
    uint32_t node_id;
    float distance;
    uint8_t expanded;
    uint8_t padding[3];
} __attribute__((packed));

static_assert(sizeof(BeamEntry) == 12, "BeamEntry must be 12 bytes");

// padding[0] of a BeamEntry carries flags (state wire version 6). Under hybrid
// PQ own-only storage a beam entry's distance is either exact (scored against
// the node's original vector on a server that holds it) or an ADC estimate
// (scored against its code); an exact entry is never re-scored. Full
// replication leaves the byte zero (every distance there is exact). Entries the
// coordinator creates start with the byte zero.
inline constexpr uint8_t kBeamEntryExactDistance = 1u;
inline bool beam_entry_exact(const BeamEntry& entry) {
    return (entry.padding[0] & kBeamEntryExactDistance) != 0;
}
inline void set_beam_entry_exact(BeamEntry& entry, bool exact) {
    entry.padding[0] = static_cast<uint8_t>(exact ? (entry.padding[0] | kBeamEntryExactDistance)
                                                  : (entry.padding[0] & ~kBeamEntryExactDistance));
}

/**
 * PING request/response for network baseline measurement
 * Minimal payload for measuring pure ZMQ + network RTT
 */
struct PingRequest {
    MessageType msg_type;  // PING_REQUEST
    uint8_t padding[3];
    uint32_t sequence;     // Sequence number for matching
    uint64_t send_time_ns; // High-resolution timestamp (coordinator fills this)
} __attribute__((packed));

struct PingResponse {
    MessageType msg_type;  // PING_RESPONSE
    uint8_t padding[3];
    uint32_t sequence;
    uint64_t send_time_ns; // Echo back the original send time
    uint32_t server_id;    // Which server responded
    uint32_t padding2;
    // Padding to match SEARCH response size (~800 bytes)
    // This tests if packet size affects network baseline
    uint8_t size_padding[776];  // 24 + 776 = 800 bytes
} __attribute__((packed));

static_assert(sizeof(PingRequest) == 16, "PingRequest must be 16 bytes");
static_assert(sizeof(PingResponse) == 800, "PingResponse must be 800 bytes");

/**
 * GET_STATS request. Ordinary observations preserve interval maxima so that
 * only an explicit consuming snapshot (the admin stats command) resets them.
 */
struct StatsRequest {
    MessageType msg_type;          // GET_STATS_REQUEST
    uint8_t consume_interval_max;  // 0 = observe, nonzero = take and reset
    uint8_t padding[2];
    // Nonzero coordinator-generated identity. A response is valid only for
    // the collector round that issued this exact request.
    uint64_t request_id;
} __attribute__((packed));

static_assert(sizeof(StatsRequest) == 12, "StatsRequest must be 12 bytes");

/**
 * Server statistics response header
 */
struct LockRoleStatsHeader {
    uint64_t acquisitions;
    uint64_t wait_us;
    uint64_t hold_us;
    uint64_t max_wait_us;
    uint64_t max_hold_us;
};

struct StatsResponseHeader {
    MessageType msg_type;
    uint8_t padding[3];
    uint64_t request_id;
    uint32_t partition_id;
    uint32_t total_nodes;
    uint32_t local_nodes;
    uint64_t directed_edges;
    // Logical DiskANN row payload: one uint32 degree plus one uint32 per
    // directed neighbor for every local adjacency row. File header excluded.
    uint64_t adjacency_serialized_bytes;
    // Allocated payload in inner neighbor vectors only. ChunkedArray outer
    // storage, vector-object, allocator, and node-lock overhead are excluded.
    uint64_t adjacency_allocated_neighbor_bytes;

    // Lock metrics (aggregate)
    uint64_t shared_lock_count;
    uint64_t unique_lock_count;
    uint64_t shared_lock_wait_us;
    uint64_t unique_lock_wait_us;

    // Phase-specific lock metrics
    uint64_t insert_alloc_lock_count;
    uint64_t insert_alloc_lock_wait_us;
    uint64_t insert_edge_lock_count;
    uint64_t insert_edge_lock_wait_us;
    uint64_t backlink_lock_count;
    uint64_t backlink_lock_wait_us;
    uint64_t broadcast_lock_count;
    uint64_t broadcast_lock_wait_us;

    // Graph metrics
    uint64_t backlink_edges_requested;
    uint64_t backlink_edges_applied;
    uint64_t backlink_skip_not_local;
    uint64_t backlink_skip_already_exists;
    // requested = applied + skip_not_local + skip_already_exists + skip_out_of_range.
    uint64_t backlink_skip_out_of_range;
    // Insert candidates not yet received here (no owner label, vector or code:
    // their broadcast or backlink was still queued), dropped before the prune.
    uint64_t insert_neighbors_undelivered;

    // Per-acquisition lock profile. These fields intentionally separate wait
    // from hold time and keep acquisition count independent of batch size.
    LockRoleStatsHeader search_shared;
    LockRoleStatsHeader insert_alloc_unique;
    LockRoleStatsHeader insert_alloc_shared;
    LockRoleStatsHeader insert_alloc_mutex;
    LockRoleStatsHeader insert_prune_shared;
    LockRoleStatsHeader insert_edge_shared;
    LockRoleStatsHeader backlink_unique;
    LockRoleStatsHeader broadcast_unique;
    LockRoleStatsHeader search_node_lock;  // search-path per-node mutex

    // search internal sub-stage cumulative timings (μs) and counts.
    uint64_t search_distance_us;
    uint64_t search_beam_sort_us;
    uint64_t search_distance_count;
    uint64_t search_neighbors_evaluated;
    uint64_t search_request_count;

    // Absolute cumulative server timing snapshots. Coordinator workload
    // reporting derives interval values by subtracting snapshots.
    uint64_t search_total_us;
    uint64_t queue_wait_us;
    uint64_t queue_wait_count;
    uint64_t response_queue_wait_us;
    uint64_t response_queue_wait_count;
    // Interval gauges. A GET_STATS request with consume_interval_max set
    // atomically takes and resets these maxima; observing requests preserve
    // them. Unlike the fields above, consumers must never subtract them.
    uint64_t queue_wait_interval_max_us;
    uint64_t response_queue_wait_interval_max_us;

    // The kinds of search_distance_count (cumulative, like it): exact + adc +
    // rerank = distance count. A missing code (a neighbour scored FLT_MAX: no
    // vector, no code yet) is counted apart and is not distance work.
    uint64_t search_exact_count;
    uint64_t search_adc_count;
    uint64_t search_rerank_count;
    uint64_t search_missing_code_count;
    // Own-only: ADC query tables built (at most one per request) and their build
    // time, which is outside search_distance_us (it is part of the request time).
    uint64_t search_adc_table_count;
    uint64_t search_adc_table_us;

    // Storage: mode 0 full replication, 1 hybrid PQ own-only; the allocated
    // bytes of the vector store (full_base + delta, or the own vectors), of the
    // PQ codes and of the PQ tables; the process's resident set size.
    uint32_t storage_mode;
    uint32_t storage_dim;
    uint32_t storage_pq_m;
    uint32_t storage_reserved;
    uint64_t storage_vector_bytes;
    uint64_t storage_code_bytes;
    uint64_t storage_pq_table_bytes;
    uint64_t process_rss_bytes;
};

// The request version and the window's node count N (ids [0, N) are
// counted).
struct StatsWindowBeginRequest {
    MessageType msg_type;
    uint8_t version;    // tide::kStatsWindowBeginVersion (1)
    uint8_t pad[2];     // zero
    uint32_t nodes;     // the window's node count; the server refuses another N
    uint64_t window_epoch;
} __attribute__((packed));

static_assert(sizeof(StatsWindowBeginRequest) == 16,
              "stats window begin request layout drifted");

struct StatsWindowAck {
    MessageType msg_type;   // BEGIN_ACK or HARVEST_ACK
    uint8_t status;         // 0 = success, nonzero = rejected
    uint8_t reserved[2];
    uint32_t server_rank;
    uint64_t window_epoch;
    uint64_t total_w;       // sum of W counters at the snapshot (0 on begin)
    uint64_t reserved_total; // zero
    // Cumulative search counters at the instant of this ack. Differencing
    // the begin ack from the harvest ack yields the window's measured
    // coefficients (c_d, c_r) and hop count with no further exchange.
    uint64_t search_request_count;
    uint64_t search_distance_count;
    uint64_t search_distance_us;
    uint64_t search_total_us;
    // The exact / ADC / re-rank split of the search work (cumulative, as
    // above; exact + adc + rerank = search_distance_count), so a controller can
    // price the two distance kinds separately, and the ADC query tables built
    // (at most one per request; their time is in search_total_us, outside
    // search_distance_us: the request component of the price).
    uint64_t search_exact_count;
    uint64_t search_adc_count;
    uint64_t search_rerank_count;
    uint64_t search_adc_table_count;
} __attribute__((packed));

static_assert(sizeof(StatsWindowAck) == 96,
              "stats window ack layout drifted");

}  // namespace ann
