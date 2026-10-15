#pragma once

#include <cstdint>
#include <cstddef>
#include <atomic>

namespace ann {

// First byte of every message. Values are fixed on the wire. Reserved: 4-5, 10-11, 22-28, 34,
// 37-52, 57, 59-62 and 68.
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
    DRAIN_REQUEST = 18,    // apply partial backlink and broadcast batches
    DRAIN_ACK = 19,
    SHUTDOWN_REQUEST = 20, // quiescent shutdown
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
    // HARVEST_ACK: a validated snapshot frame (coordinator only)
    STATS_WINDOW_HARVEST_ACK = 58,
    STATS_WINDOW_SNAPSHOT_REQUEST = 63,
    STATS_WINDOW_SNAPSHOT_RESPONSE = 64,
    STATS_WINDOW_STOP_REQUEST = 65,
    STATS_WINDOW_STOP_ACK = 66,
    MIGRATION_REFRESH_INLINE_REQUEST = 67,
    // membership-check request: type, u32 count, count x u32 owners
    MEMBERSHIP_CHECK_REQUEST = 69,
    MEMBERSHIP_CHECK_RESPONSE = 70
};

enum class AdminAckDisposition : uint8_t {
    ACCEPT,
    DUPLICATE,
    UNMATCHED,
    INVALID_SERVER
};

// Insert maintenance replies. Any other exchange that pops one from the shared queue puts it back.
inline constexpr bool is_maintenance_reply(MessageType type) {
    return type == MessageType::INSERT_RESPONSE || type == MessageType::BATCH_BACKLINK_ACK ||
           type == MessageType::VECTOR_BROADCAST_ACK;
}

// Whether an admin exchange puts back replies that are not its ACK (RESET drops them).
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
    // MIGRATION_ACCEPT_ACK: the attempt id of its request, 0 for other types
    uint16_t request_id;
    uint32_t server_rank;
    uint64_t epoch_from;
    uint64_t epoch_to;
    uint64_t move_count;
} __attribute__((packed));

static_assert(sizeof(MembershipMigrationAck) == 32,
              "MembershipMigrationAck must be 32 bytes");


// reply to MEMBERSHIP_CHECK_REQUEST (membership_check in membership_migration.h)
struct MembershipCheckResponse {
    MessageType msg_type;
    uint8_t status;     // 0 compared, 1 not quiescent, 2 malformed request
    uint8_t own_only;   // storage mode
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
    uint64_t vectors;                // PQ mode: ids whose original vector is here
    uint64_t membership_mismatches;  // mem_at differs from the coordinator's owner
    uint64_t mapping_mismatches;     // a row here differs from ownership here
    uint64_t vector_mismatches;      // PQ mode: a vector here differs from ownership here
    uint64_t missing_codes;          // PQ mode: ids without a PQ code
    uint64_t vector_digest;          // PQ mode: sum of node_vector_hash over the vectors held here
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

// node_id, distance and flags (12 bytes)
struct BeamEntry {
    uint32_t node_id;
    float distance;
    uint8_t expanded;
    uint8_t padding[3];
} __attribute__((packed));

static_assert(sizeof(BeamEntry) == 12, "BeamEntry must be 12 bytes");

// padding[0] flags. In PQ mode an entry's distance is exact or an ADC estimate, and an exact entry is
// never re-scored. Zero with full vectors and for entries the coordinator creates.
inline constexpr uint8_t kBeamEntryExactDistance = 1u;
inline bool beam_entry_exact(const BeamEntry& entry) {
    return (entry.padding[0] & kBeamEntryExactDistance) != 0;
}
inline void set_beam_entry_exact(BeamEntry& entry, bool exact) {
    entry.padding[0] = static_cast<uint8_t>(exact ? (entry.padding[0] | kBeamEntryExactDistance)
                                                  : (entry.padding[0] & ~kBeamEntryExactDistance));
}

// PING / PONG for the network baseline.
struct PingRequest {
    MessageType msg_type;  // PING_REQUEST
    uint8_t padding[3];
    uint32_t sequence;
    uint64_t send_time_ns; // set by the coordinator
} __attribute__((packed));

struct PingResponse {
    MessageType msg_type;  // PING_RESPONSE
    uint8_t padding[3];
    uint32_t sequence;
    uint64_t send_time_ns; // echoed
    uint32_t server_id;
    uint32_t padding2;
    // pad to ~800 bytes (search response size)
    uint8_t size_padding[776];  // 24 + 776 = 800
} __attribute__((packed));

static_assert(sizeof(PingRequest) == 16, "PingRequest must be 16 bytes");
static_assert(sizeof(PingResponse) == 800, "PingResponse must be 800 bytes");

// GET_STATS request. Only a consuming request (admin stats) resets the interval maxima.
struct StatsRequest {
    MessageType msg_type;          // GET_STATS_REQUEST
    uint8_t consume_interval_max;  // 0 = observe, nonzero = take and reset
    uint8_t padding[2];
    // nonzero id, and a reply is valid only for the round that sent it
    uint64_t request_id;
} __attribute__((packed));

static_assert(sizeof(StatsRequest) == 12, "StatsRequest must be 12 bytes");

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
    // adjacency payload: one u32 degree plus one u32 per neighbor per row
    uint64_t adjacency_serialized_bytes;
    // allocated neighbor vectors only
    uint64_t adjacency_allocated_neighbor_bytes;

    uint64_t shared_lock_count;
    uint64_t unique_lock_count;
    uint64_t shared_lock_wait_us;
    uint64_t unique_lock_wait_us;

    // per insert phase
    uint64_t insert_alloc_lock_count;
    uint64_t insert_alloc_lock_wait_us;
    uint64_t insert_edge_lock_count;
    uint64_t insert_edge_lock_wait_us;
    uint64_t backlink_lock_count;
    uint64_t backlink_lock_wait_us;
    uint64_t broadcast_lock_count;
    uint64_t broadcast_lock_wait_us;

    uint64_t backlink_edges_requested;
    uint64_t backlink_edges_applied;
    uint64_t backlink_skip_not_local;
    uint64_t backlink_skip_already_exists;
    // requested = applied + skip_not_local + skip_already_exists + skip_out_of_range.
    uint64_t backlink_skip_out_of_range;
    // insert candidates skipped because they had not reached this server yet
    uint64_t insert_neighbors_undelivered;

    // per lock role: acquisitions, wait and hold time
    LockRoleStatsHeader search_shared;
    LockRoleStatsHeader insert_alloc_unique;
    LockRoleStatsHeader insert_alloc_shared;
    LockRoleStatsHeader insert_alloc_mutex;
    LockRoleStatsHeader insert_prune_shared;
    LockRoleStatsHeader insert_edge_shared;
    LockRoleStatsHeader backlink_unique;
    LockRoleStatsHeader broadcast_unique;
    LockRoleStatsHeader search_node_lock;  // search-path per-node mutex

    // search totals (us) and counts
    uint64_t search_distance_us;
    uint64_t search_beam_sort_us;
    uint64_t search_distance_count;
    uint64_t search_neighbors_evaluated;
    uint64_t search_request_count;

    // cumulative (the coordinator subtracts snapshots)
    uint64_t search_total_us;
    uint64_t queue_wait_us;
    uint64_t queue_wait_count;
    uint64_t response_queue_wait_us;
    uint64_t response_queue_wait_count;
    // maxima since the last consuming request, never subtracted
    uint64_t queue_wait_interval_max_us;
    uint64_t response_queue_wait_interval_max_us;

    // exact + adc + rerank = search_distance_count (missing codes are no distance work)
    uint64_t search_exact_count;
    uint64_t search_adc_count;
    uint64_t search_rerank_count;
    uint64_t search_missing_code_count;
    // PQ mode: ADC tables built (at most one per request), whose time is not in search_distance_us
    uint64_t search_adc_table_count;
    uint64_t search_adc_table_us;

    // storage mode (0 full vectors, 1 PQ), allocated store bytes and resident set size
    uint32_t storage_mode;
    uint32_t storage_dim;
    uint32_t storage_pq_m;
    uint32_t storage_reserved;
    uint64_t storage_vector_bytes;
    uint64_t storage_code_bytes;
    uint64_t storage_pq_table_bytes;
    uint64_t process_rss_bytes;
};

// version and the window's node count N
struct StatsWindowBeginRequest {
    MessageType msg_type;
    uint8_t version;    // tide::kStatsWindowBeginVersion (1)
    uint8_t pad[2];     // zero
    uint32_t nodes;     // the window's node count
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
    // cumulative search counters (the coordinator takes BEGIN-to-HARVEST differences)
    uint64_t search_request_count;
    uint64_t search_distance_count;
    uint64_t search_distance_us;
    uint64_t search_total_us;
    // exact / ADC / re-rank split and ADC tables built, cumulative
    uint64_t search_exact_count;
    uint64_t search_adc_count;
    uint64_t search_rerank_count;
    uint64_t search_adc_table_count;
} __attribute__((packed));

static_assert(sizeof(StatsWindowAck) == 96,
              "stats window ack layout drifted");

}  // namespace ann
