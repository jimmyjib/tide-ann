#pragma once
/**
 * Server types and global declarations.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <limits>
#include <thread>

#include "chunked_array.h"
#include "state_protocol.h"
#include "thread_pool.h"
#include "distance.h"
#include "pq.h"

namespace tide {

// Import commonly used types from ann namespace
using ann::BeamEntry;
using ann::StateStatus;
using ann::CrossPartitionEdge;
using ann::LockMetrics;
using ann::GraphMetrics;
using ann::WorkItem;
using ann::ResponseItem;
using ann::TimingMetrics;
using ann::compute_distance;

// =============================================================================
// ServerData - Global server state (read-only after init except during insert)
// =============================================================================
// Per-node lock for adjacency list protection (cache-line aligned)
struct alignas(64) NodeLock {
    std::mutex mtx;
};

struct ServerData {
    // Chunked storage:
    // - `full_base` is the immutable static section loaded from disk.
    //   Size fixed at `base_static_capacity * dim` after load.
    // - Inserted nodes (node_id >= base_static_capacity) live in the
    //   ChunkedVectorStore delta section, which uses a fixed-size atomic
    //   pointer array as outer container (no reallocation). Lazy chunk alloc.
    // - The outer container is reallocation-free, so search can read chunk
    //   pointers concurrently with broadcast / insert without any lock.
    static constexpr uint32_t FULL_BASE_SEGMENT_NODES = ChunkedVectorStore::CHUNK_NODES;
    std::vector<float> full_base;             // static section (loaded from disk)
    ChunkedVectorStore full_base_delta;       // delta section (lock-free)
    // Write state of each delta vector, by id - base_static_capacity
    // (store_replicated_vector): 0 empty, 1 being written, 2 complete.
    ChunkedCounterStore full_base_delta_state;
    uint32_t base_static_capacity = 0;        // = full_base.size() / dim after load (fixed)

    // Membership chunked the same way. Init value UINT32_MAX.
    ChunkedArray<uint32_t, ChunkedVectorStore::CHUNK_NODES,
                 ChunkedVectorStore::MAX_CHUNKS> membership_delta;

    // Graph/partition state chunked for lock-free access.
    //
    // Per plan:
    //   - adj            : ChunkedArray<std::vector<uint32_t>, 10000, 1000>
    //                      (10K inner vectors per chunk; 10M nodes max)
    //   - global_to_local: ChunkedArray<uint32_t> (100K per chunk; 100M max)
    //   - local_to_global: ChunkedArray<uint32_t>
    //   - n_local        : atomic<uint32_t> (fetch_add for unique local_idx)
    //
    // adj inner vector push_back is still protected by per-node `node_locks`.
    // The chunked outer means: search/insert can read at(i) without any global
    // lock; concurrent allocations of new chunks are serialized by the
    // ChunkedArray internal mutex (rare path).
    std::vector<uint32_t> membership;
    ChunkedArray<std::vector<uint32_t>, 10000, 1000> adj;
    ChunkedArray<uint32_t> global_to_local;   // init = UINT32_MAX
    ChunkedArray<uint32_t> local_to_global;   // init = UINT32_MAX
    std::atomic<uint32_t> n_local{0};
    std::atomic<uint32_t> n_base{0};         // Atomic for lock-free bump
    uint32_t dim;
    uint32_t my_partition;
    uint32_t n_servers;
    bool use_partition_graph;                // true if using partition-specific graph

    // Per-node locks, indexed by local_idx: chunked and never moved, so
    // searches read size() and node_locks[i] while an insert grows the store
    // (ensure_node_locks); the capacity matches adj's (10M rows).
    ChunkedSlots<NodeLock, 10000, 1000> node_locks;

    // deferred pruning slack. Prune triggers when adj.size() > R + edge_overload
    // instead of size() > R. Prune still reduces to R neighbors; only the trigger
    // threshold is delayed, so robust_prune calls drop by ≈1/(edge_overload+1).
    // Set via --edge-overload CLI flag at server startup (default 16).
    uint32_t edge_overload = 16;

    // Free row slots (adjacency released, no mapping) that a migration PREPARE
    // and an insert reuse before appending; a rollback returns the reused
    // slots here. Only freed rows enter it (the source rows an ACCEPT
    // frees). PREPARE, ROLLBACK, ACCEPT and RESET touch it under the exclusive
    // admin lock with every queue drained; insert batches run
    // concurrently under the shared admin lock and take a slot under
    // free_slot_mutex (insert_row_slot, handlers.cpp).
    std::vector<uint32_t> free_slot_pool;
    std::mutex free_slot_mutex;
    ChunkedArray<uint64_t> migration_slot_generations; // absent segment means 0

    // Runtime window statistics (stats_window.h). Relaxed per-node counters:
    // the static table stats_window_w for the load-time ids [0, N0) (allocated
    // zeroed at load, N0 = stats_window_static_nodes) and the chunked delta
    // table stats_window_delta for the ids inserted after load (index id - N0),
    // grown at STATS_WINDOW_BEGIN, never on the insert, backlink or broadcast
    // paths (a controller phase fences inserts out). BEGIN freezes the window's
    // node count N (stats_window_nodes, at most the ids this server knows),
    // zeroes [0, N) and activates the window mid-serving; a SNAPSHOT request
    // deactivates it and answers with one snapshot frame. A counter of v counts
    // the distance evaluations of v's expansions executed HERE.
    std::unique_ptr<std::atomic<uint32_t>[]> stats_window_w;
    uint32_t stats_window_static_nodes = 0;   // N0; load and RESET only (admin)
    ChunkedCounterStore stats_window_delta;   // ids >= N0, by id - N0
    std::atomic<uint32_t> stats_window_nodes{0};
    // Release store at BEGIN after the table is grown and zeroed and N stored;
    // the search path's acquire load then sees both.
    std::atomic<bool> stats_window_active{false};
    std::atomic<uint64_t> stats_window_epoch{0};
    // Serializes BEGIN, the snapshot cut and STOP (never taken on the search path).
    std::mutex stats_window_mutex;

    // ---- Storage mode -------------------------------------------------------
    // Full replication (own_only false): every server holds every vector
    // (full_base for the load-time ids, full_base_delta for inserted ids) and
    // every distance is exact.
    //
    // Hybrid PQ own-only storage (own_only true, the server's --pq mode): a
    // server holds the original vectors of the nodes it owns, in `full_own`
    // indexed by LOCAL slot (the row index of adj / local_to_global), and a PQ
    // code of every node (`pq_codes_static` for the load-time ids, row = id;
    // `pq_codes_delta` for inserted ids). full_base stays empty. A node's vector
    // is present here exactly when global_to_local maps it to a slot: an owned
    // node, or the target row of a migration that installed the shipped vector
    // before publishing the mapping. Distances against a present vector are
    // exact; against a code they are ADC (query vs code) or code-to-code.
    // own_only is set before the first load and RESET keeps it; the stores are
    // rebuilt by load_server_data. The membership static section stays N0 =
    // base_static_capacity (the load-time node count from membership.npy).
    bool own_only = false;
    ann::PQTable pq_table;                 // pq_pivots.bin (own_only only)
    std::vector<uint8_t> pq_codes_static;  // [base_static_capacity * M], pq_codes.bin
    ChunkedCodeStore pq_codes_delta;       // codes of inserted ids, by id - base_static_capacity
    ChunkedVectorStore full_own;           // own vectors by local slot
};

// Helpers for n_local atomic access.
inline uint32_t n_local_load(const ServerData& d,
                              std::memory_order mo = std::memory_order_acquire) {
    return d.n_local.load(mo);
}
// fetch_add returns previous value — caller gets a unique local_idx slot.
inline uint32_t n_local_alloc(ServerData& d) {
    return d.n_local.fetch_add(1, std::memory_order_acq_rel);
}
inline void n_local_set(ServerData& d, uint32_t value) {
    if (value == 0) {
        d.free_slot_pool.clear();
        d.migration_slot_generations.clear();
    }
    d.n_local.store(value, std::memory_order_release);
}

inline uint64_t migration_slot_generation(const ServerData& d, uint32_t slot) {
    const auto* segment = d.migration_slot_generations.segment(
        slot / decltype(d.migration_slot_generations)::chunk_size);
    return segment ? segment[slot % decltype(d.migration_slot_generations)::chunk_size] : 0;
}

// Grow node_locks to at least 'target' locks. Thread-safe; an existing lock
// never moves, so search/insert reading node_locks[i] for i below an observed
// size() stays safe while another thread grows the store. Beyond the capacity
// (adj's 10M rows) it throws std::length_error.
inline void ensure_node_locks(ServerData& d, uint32_t target) {
    if (!d.node_locks.grow_to(target)) throw std::length_error("node locks: capacity exceeded or allocation failed");
}

// Helpers for n_base atomic access.
inline uint32_t n_base_load(const ServerData& d,
                            std::memory_order mo = std::memory_order_acquire) {
    return d.n_base.load(mo);
}

// CAS-loop max bump. Use release ordering so that prior vec/mem writes are
// visible to readers that subsequently observe the updated n_base via acquire.
inline void n_base_bump(ServerData& d, uint32_t want) {
    uint32_t cur = d.n_base.load(std::memory_order_relaxed);
    while (cur < want &&
           !d.n_base.compare_exchange_weak(cur, want,
                                            std::memory_order_release,
                                            std::memory_order_relaxed)) {}
}

// Direct set — used by init / reset paths only.
inline void n_base_set(ServerData& d, uint32_t value) {
    d.n_base.store(value, std::memory_order_release);
}

// Checked global -> local lookup: UINT32_MAX when the id has no mapping here,
// including an id whose global_to_local chunk was never allocated on this
// server (an id another server created). ChunkedArray::at does not check.
inline uint32_t g2l_checked(const ServerData& d, uint32_t gid) {
    using G2L = decltype(d.global_to_local);
    const uint32_t* segment = d.global_to_local.segment(gid / G2L::chunk_size);
    return segment ? segment[gid % G2L::chunk_size] : UINT32_MAX;
}

// Own-only: the original vector of `id` when this server holds it (a mapped
// local slot whose full_own chunk exists), else nullptr.
inline const float* own_vec(const ServerData& d, uint32_t id) {
    const uint32_t slot = g2l_checked(d, id);
    if (slot == UINT32_MAX ||
        !d.full_own.segment_allocated(slot / ChunkedVectorStore::CHUNK_NODES)) return nullptr;
    return d.full_own.vec(slot);
}

// Full replication: the vector of `id` when it is complete here, else nullptr.
// A load-time id's vector always is; an inserted id's only once its first
// delivery has published it (store_replicated_vector: state 2, read with
// acquire), so a reader never sees the zeroed chunk or a copy in progress.
inline const float* replicated_vec(const ServerData& d, uint32_t id) {
    if (id < d.base_static_capacity) return d.full_base.data() + static_cast<size_t>(id) * d.dim;
    const uint32_t delta = id - d.base_static_capacity;
    const auto* state = d.full_base_delta_state.counter(delta);
    if (state == nullptr || state->load(std::memory_order_acquire) != 2 ||
        !d.full_base_delta.segment_allocated(delta / ChunkedVectorStore::CHUNK_NODES)) return nullptr;
    return d.full_base_delta.vec(delta);
}

// The vector this server stores for `id`, or nullptr: the own vector under
// own-only storage, the complete replicated vector under full replication.
// The distance helpers below read vectors only through it.
inline const float* stored_vec(const ServerData& d, uint32_t id) {
    return d.own_only ? own_vec(d, id) : replicated_vec(d, id);
}

// Vector access helpers — chunked-array backed delta section.
// Full replication: never null for an id whose chunk exists (callers pass known
// ids that were delivered; the distance helpers use stored_vec instead).
// Own-only: the own vector or nullptr for a node held only as a code, so every
// own-only caller null-checks or goes through the distance helpers below.
inline float* server_vec(ServerData& d, uint32_t id) {
    if (d.own_only) return const_cast<float*>(own_vec(d, id));
    if (id < d.base_static_capacity) {
        return d.full_base.data() + static_cast<size_t>(id) * d.dim;
    }
    return d.full_base_delta.vec(id - d.base_static_capacity);
}

inline const float* server_vec(const ServerData& d, uint32_t id) {
    if (d.own_only) return own_vec(d, id);
    if (id < d.base_static_capacity) {
        return d.full_base.data() + static_cast<size_t>(id) * d.dim;
    }
    return d.full_base_delta.vec(id - d.base_static_capacity);
}

// Ensure a chunk exists that covers `node_id`. Thread-safe (atomic pointer
// store + small mutex for allocation). Returns true if at least one new
// chunk was allocated. A no-op under own_only: own vectors live in full_own
// (by local slot), every other node only as a code.
inline bool server_ensure_vec_capacity(ServerData& d, uint32_t node_id) {
    if (d.own_only) return false;
    if (node_id < d.base_static_capacity) return false;
    uint32_t delta = node_id - d.base_static_capacity;
    size_t prev = d.full_base_delta.allocated_chunks();
    d.full_base_delta.ensure_capacity_for(delta);
    return d.full_base_delta.allocated_chunks() > prev;
}

// Full replication: stores the vector of an inserted id once. Its insert (on
// the home server), a backlink and a VECTOR_BROADCAST all deliver it, and a
// backlink and a broadcast of the same id can run at the same time on two
// workers. The first delivery claims the id (state 0 -> 1), copies the vector
// and publishes state 2 (release); any other delivery skips the copy and waits
// (acquire) until the state is 2. On return every caller may publish the id
// (an edge, n_base) with its vector complete, and a search never reads a vector
// that is being rewritten. True when this call wrote it; false when another
// delivery did, under own-only storage, for a load-time id, or when the chunk
// could not be allocated.
inline bool store_replicated_vector(ServerData& d, uint32_t id, const float* vec) {
    if (d.own_only || id < d.base_static_capacity) return false;
    const uint32_t delta = id - d.base_static_capacity;
    if (!d.full_base_delta.ensure_capacity_for(delta) || !d.full_base_delta_state.ensure_capacity_for(delta))
        return false;
    std::atomic<uint32_t>& state = *d.full_base_delta_state.counter(delta);
    uint32_t expected = 0;
    if (state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
        std::copy(vec, vec + d.dim, d.full_base_delta.vec(delta));
        state.store(2, std::memory_order_release);
        return true;
    }
    while (state.load(std::memory_order_acquire) != 2) std::this_thread::yield();
    return false;
}

// Membership access helpers — chunked the same way as full_base.
// The label of an inserted id (the delta section) is written by every path
// that delivers the id: its insert, and on the other servers a backlink and a
// VECTOR_BROADCAST, which two workers can apply at the same time; searches read
// it meanwhile. Delta labels are therefore relaxed atomic loads and stores
// (every delivery stores the same owner). The static section changes only
// under the exclusive admin lock (migration).
inline uint32_t mem_at(const ServerData& d, uint32_t id) {
    if (id < d.base_static_capacity) {
        return d.membership[id];
    }
    return __atomic_load_n(&d.membership_delta.at(id - d.base_static_capacity), __ATOMIC_RELAXED);
}

inline void mem_set(ServerData& d, uint32_t id, uint32_t value) {
    if (id < d.base_static_capacity) {
        d.membership[id] = value;
        return;
    }
    __atomic_store_n(&d.membership_delta.at(id - d.base_static_capacity), value, __ATOMIC_RELAXED);
}

// The owner label of `id` as this server knows it, UINT32_MAX when it knows
// none yet: an inserted id whose membership chunk is absent here, or whose label
// no delivery has stored. Unlike mem_at it never indexes an absent chunk.
inline uint32_t mem_label_checked(const ServerData& d, uint32_t id) {
    if (id < d.base_static_capacity) return id < d.membership.size() ? d.membership[id] : UINT32_MAX;
    using Mem = decltype(d.membership_delta);
    const uint32_t delta = id - d.base_static_capacity;
    const uint32_t* segment = d.membership_delta.segment(delta / Mem::chunk_size);
    return segment ? __atomic_load_n(&segment[delta % Mem::chunk_size], __ATOMIC_RELAXED) : UINT32_MAX;
}

// Total membership capacity across static + chunks.
inline size_t mem_size(const ServerData& d) {
    return static_cast<size_t>(d.base_static_capacity)
         + d.membership_delta.total_capacity();
}

// Ensure a chunk exists that covers node_id for membership. Pre-fills new
// chunks with UINT32_MAX. Thread-safe.
inline bool server_ensure_mem_capacity(ServerData& d, uint32_t node_id) {
    if (node_id < d.base_static_capacity) return false;
    uint32_t delta = node_id - d.base_static_capacity;
    size_t prev = d.membership_delta.allocated_chunks();
    d.membership_delta.ensure_capacity_for(delta, UINT32_MAX);
    return d.membership_delta.allocated_chunks() > prev;
}

// =============================================================================
// Hybrid PQ distances (own-only storage)
// =============================================================================

// The PQ code of `id` (own_only only): a load-time id's row of pq_codes_static,
// an inserted id's READY delta code, else nullptr (full replication, or a delta
// code not stored yet: the broadcast/backlink that carries it is still queued).
inline const uint8_t* pq_code(const ServerData& d, uint32_t id) {
    if (!d.own_only || d.pq_table.M == 0) return nullptr;
    if (id < d.base_static_capacity) {
        const size_t offset = static_cast<size_t>(id) * d.pq_table.M;
        return offset + d.pq_table.M <= d.pq_codes_static.size() ? d.pq_codes_static.data() + offset : nullptr;
    }
    return d.pq_codes_delta.code(id - d.base_static_capacity);
}

// Whether this server has received id `id` completely: a load-time id always
// has; an inserted id once its owner label (below K) and, under own-only
// storage, its PQ code (READY) or, under full replication, its vector (state 2)
// are stored here. Until then the broadcast or backlink that carries it is still
// queued here or being applied: an insert drops such a candidate instead of
// reading it (batch_process_inserts), and never links it into a row.
inline bool delivered_here(const ServerData& d, uint32_t id) {
    if (id < d.base_static_capacity) return true;
    if (mem_label_checked(d, id) >= d.n_servers) return false;
    return d.own_only ? pq_code(d, id) != nullptr : replicated_vec(d, id) != nullptr;
}

// Stores the code of an inserted id once (own_only; a load-time id's code is in
// pq_codes_static already). Idempotent: a second delivery of the same vector
// (insert, backlink, broadcast) finds the code READY or being written and skips.
inline void store_pq_code(ServerData& d, uint32_t gid, const float* vec) {
    if (!d.own_only || gid < d.base_static_capacity) return;
    d.pq_codes_delta.write_once(gid - d.base_static_capacity,
                                [&](uint8_t* out) { d.pq_table.encode(vec, out); });
}

// Copies an own vector into full_own at local slot `slot` (own_only). The caller
// publishes global_to_local[gid] = slot only after this returns.
inline void store_own_vector(ServerData& d, uint32_t slot, const float* vec) {
    d.full_own.ensure_capacity_for(slot);
    std::copy(vec, vec + d.dim, d.full_own.vec(slot));
}

// A full vector this server does not store but knows for the current operation
// (a backlink carries its source's vector): used instead of the code of `id`.
struct KnownVector {
    uint32_t id = UINT32_MAX;
    const float* vec = nullptr;
};

// Squared-L2 distance between two nodes with the best data this server holds:
// both vectors present -> exact; one vector and the other's code -> ADC of the
// vector against the code; two codes -> code-to-code table; otherwise FLT_MAX
// (a delta code not stored yet). Full replication takes the exact branch, with
// the same argument order as the original prune code, for every vector it holds
// complete; an inserted id whose vector is not complete here is FLT_MAX.
inline float node_pair_distance(const ServerData& d, uint32_t a, uint32_t b,
                                const KnownVector* known = nullptr) {
    const float* va = (known && known->id == a) ? known->vec : stored_vec(d, a);
    const float* vb = (known && known->id == b) ? known->vec : stored_vec(d, b);
    if (va && vb) return compute_distance(va, vb, d.dim);
    if (!d.own_only) return std::numeric_limits<float>::max();
    const uint8_t* cb = vb ? nullptr : pq_code(d, b);
    if (va && cb) return d.pq_table.adc_distance_direct(va, cb);
    const uint8_t* ca = va ? nullptr : pq_code(d, a);
    if (vb && ca) return d.pq_table.adc_distance_direct(vb, ca);
    if (ca && cb) return d.pq_table.sym_distance(ca, cb);
    return std::numeric_limits<float>::max();
}

// The query's ADC table [M][K], built on first use in one process_state call
// (a hop whose neighbours are all held as vectors never builds it). The buffer
// is the caller's thread-local scratch. The build is timed (build_ns) so the
// search can keep it out of the per-distance price: one build is a fixed cost
// of the hop (M x K sub-distances, about 256 exact distances at M = 24, sub_dim 4).
class LazyAdcTable {
public:
    LazyAdcTable(const ann::PQTable& pq, const float* query, std::vector<float>& buffer)
        : pq_(pq), query_(query), buffer_(buffer) {}
    const float* get() {
        if (!built_) {
            const auto start = std::chrono::steady_clock::now();
            buffer_.resize(static_cast<size_t>(pq_.M) * ann::PQTable::K);
            pq_.build_query_table(query_, buffer_.data());
            built_ = true;
            build_ns_ = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count());
        }
        return buffer_.data();
    }
    bool built() const { return built_; }
    uint64_t build_ns() const { return build_ns_; }
private:
    const ann::PQTable& pq_;
    const float* query_;
    std::vector<float>& buffer_;
    bool built_ = false;
    uint64_t build_ns_ = 0;
};

struct HybridScoreCounts {
    uint32_t exact = 0;    // neighbours scored against a present vector
    uint32_t adc = 0;      // neighbours scored against their code
    uint32_t missing = 0;  // neither: FLT_MAX
};

// How hybrid_score scored a neighbour.
enum : uint8_t { kScoreAdc = 0, kScoreExact = 1, kScoreMissing = 2 };

// Scores `ids` against the query (the search's neighbour scorer). A neighbour
// whose vector this server holds is scored exactly (kind[i] = kScoreExact);
// otherwise its code is scored with the query's ADC table (kScoreAdc); a node
// with neither gets FLT_MAX (kScoreMissing: no distance was evaluated, so it is
// no distance work). "Held" means the vector is present (stored_vec), not the
// membership label. Under full replication every vector is present except an
// inserted id whose vector is not complete here (its chunk absent, or its
// delivery still queued or being copied), which gets FLT_MAX.
inline HybridScoreCounts hybrid_score(const ServerData& d, const float* query, const std::vector<uint32_t>& ids,
                                      LazyAdcTable& adc, std::vector<float>& distances,
                                      std::vector<uint8_t>& kind) {
    HybridScoreCounts counts;
    distances.resize(ids.size());
    kind.assign(ids.size(), kScoreAdc);
    for (size_t i = 0; i < ids.size(); ++i) {
        const uint32_t id = ids[i];
        const float* vec = stored_vec(d, id);
        if (vec) {
            distances[i] = compute_distance(query, vec, d.dim);
            kind[i] = kScoreExact;
            ++counts.exact;
        } else if (const uint8_t* code = pq_code(d, id)) {
            distances[i] = d.pq_table.adc_distance(adc.get(), code);
            ++counts.adc;
        } else {
            distances[i] = std::numeric_limits<float>::max();
            kind[i] = kScoreMissing;
            ++counts.missing;
        }
    }
    return counts;
}

// Allocated bytes of the stores, for the stats reply (memory per server).
struct StorageBytes {
    uint64_t vector_bytes = 0;    // full_base + full_base_delta (with state words), or full_own
    uint64_t code_bytes = 0;      // pq_codes_static + pq_codes_delta (with state bytes)
    uint64_t pq_table_bytes = 0;  // centroids + code-to-code table
    uint64_t own_vectors = 0;     // own_only: mapped local slots (vectors in use)
};
inline StorageBytes storage_bytes(const ServerData& d) {
    StorageBytes b;
    if (d.own_only) {
        b.vector_bytes = static_cast<uint64_t>(d.full_own.total_node_capacity()) * d.dim * sizeof(float);
        b.code_bytes = d.pq_codes_static.size() + d.pq_codes_delta.allocated_bytes();
        b.pq_table_bytes = (d.pq_table.centroids.size() + d.pq_table.sym_table.size()) * sizeof(float);
    } else {
        b.vector_bytes = static_cast<uint64_t>(d.full_base.size()) * sizeof(float) +
            static_cast<uint64_t>(d.full_base_delta.total_node_capacity()) * d.dim * sizeof(float) +
            static_cast<uint64_t>(d.full_base_delta_state.total_capacity()) * sizeof(uint32_t);
    }
    return b;
}

// =============================================================================
// CapacityGrowth - chunk allocations made by ensure_capacity (insert path)
// =============================================================================
struct ResizeStats {
    uint32_t count = 0;
    uint32_t time_us = 0;
};

struct CapacityGrowth {
    ResizeStats resize_full_base;
    ResizeStats resize_membership;
    ResizeStats resize_adj;
    ResizeStats resize_global_to_local;
};

// =============================================================================
// ProcessMetrics - Metrics for process_state
// =============================================================================
struct ProcessMetrics {
    // Distance evaluations = exact + adc + rerank (a missing code is FLT_MAX,
    // not an evaluation); the window counter W counts the same work per
    // expanded node: its scored neighbours plus its own re-rank.
    uint32_t distance_computations = 0;
    uint32_t exact_distances = 0;     // neighbours scored against a present vector
    uint32_t adc_distances = 0;       // neighbours scored against their PQ code
    uint32_t missing_codes = 0;       // neighbours with neither (FLT_MAX): no work
    uint32_t rerank_distances = 0;    // own-only: expanded entries re-scored exactly
    uint32_t adc_tables = 0;          // own-only: ADC query tables built (0 or 1 per hop)
    uint64_t adc_table_ns = 0;        // their build time, outside distance_us
    uint32_t local_nodes_expanded = 0;
    uint32_t top_W_count = 0;
    uint32_t neighbors_evaluated = 0;
    uint32_t beam_size_in = 0;
    uint64_t distance_us = 0;
    uint64_t beam_sort_us = 0;
};

constexpr bool valid_search_source(uint32_t node_id, uint32_t n_base,
                                   uint32_t local_idx,
                                   uint32_t n_local) noexcept {
    return node_id < n_base && local_idx != UINT32_MAX && local_idx < n_local;
}

// =============================================================================
// Global variables (defined in tide_server.cpp)
// =============================================================================
// 2-tier lock: g_resize_mutex (outer vector resize) + node_locks (per-node adj)
// Lock ordering: g_resize_mutex → node_locks[i] (ascending local_idx)
extern std::shared_mutex g_resize_mutex;
extern std::mutex g_alloc_mutex;
extern LockMetrics g_lock_metrics;
extern GraphMetrics g_graph_metrics;
extern std::string g_data_dir;
extern int g_rank;
extern uint32_t g_start_node;

// =============================================================================
// Function declarations - Graph Operations (graph_ops.cpp)
// =============================================================================

// Load server data from disk
uint32_t load_server_data(ServerData& data, const std::string& data_dir, int rank);

// RESET: clears every store (vectors, own vectors, PQ codes and tables,
// membership, adjacency, both mappings, node locks, the free row-slot pool and
// slot generations) and reloads them, and the window-statistics table, from
// `data_dir` with load_server_data. The start-up mode (own_only, rank, K) is
// kept. The caller excludes search and maintenance (exclusive locks, drained
// queues). Returns the start node.
uint32_t reset_server_data(ServerData& data, const std::string& data_dir, int rank);

// Ensure capacity for a given global_id (grow the chunked stores if needed)
void ensure_capacity(ServerData& data, uint32_t global_id, CapacityGrowth& result);

// RobustPrune (Algorithm 3 from FreshDiskANN). Occlusion distances come from
// node_pair_distance (exact, ADC or code-to-code under own-only storage);
// `known` supplies a full vector this server does not store (a backlink source).
std::vector<uint32_t> robust_prune(
    const ServerData& data,
    uint32_t p,
    std::vector<std::pair<uint32_t, float>>& candidates,
    uint32_t R,
    float alpha,
    const KnownVector* known = nullptr
);

// =============================================================================
// Function declarations - Handlers (handlers.cpp)
// =============================================================================

// BeamSearch state processing
void process_state(
    ServerData& data,
    const float* query, uint32_t dim,
    uint32_t L, uint32_t K, uint32_t W,
    std::vector<BeamEntry>& beam,
    StateStatus& status,
    uint32_t& next_partition,
    std::vector<uint32_t>& result_neighbors,
    std::vector<float>& result_distances,
    ProcessMetrics& pmetrics,
    std::vector<bool>& seen,
    std::vector<uint32_t>& seen_nodes,
    uint32_t query_id,
    uint32_t hop_count
);

// =============================================================================
// Batch processing functions (server-side batching)
// =============================================================================

// Batch process multiple backlink requests with single lock acquisition
std::vector<ResponseItem> batch_process_backlinks(
    ServerData& data,
    std::vector<WorkItem>& batch,
    TimingMetrics& metrics
);

// Batch process multiple broadcast requests with single lock acquisition
std::vector<ResponseItem> batch_process_broadcasts(
    ServerData& data,
    std::vector<WorkItem>& batch,
    TimingMetrics& metrics
);

// The row slot of an inserted node (batch_process_inserts): the most
// recently freed slot of the pool (free_slot_pool, the rows an ACCEPT
// freed), its migration generation bumped as a PREPARE reuse does, or a new
// slot appended at n_local when the pool is empty (or its last entry is not a
// sound free row). Insert batches run concurrently, so the pool is taken under
// free_slot_mutex.
uint32_t insert_row_slot(ServerData& data);

// Batch process multiple insert requests with single g_resize_mutex acquisition
std::vector<ResponseItem> batch_process_inserts(
    ServerData& data,
    std::vector<WorkItem>& batch,
    TimingMetrics& metrics
);

} // namespace tide
