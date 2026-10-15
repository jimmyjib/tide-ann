#pragma once

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

using ann::BeamEntry;
using ann::StateStatus;
using ann::CrossPartitionEdge;
using ann::LockMetrics;
using ann::GraphMetrics;
using ann::WorkItem;
using ann::ResponseItem;
using ann::TimingMetrics;
using ann::compute_distance;

// per-node adjacency lock (cache-line aligned)
struct alignas(64) NodeLock {
    std::mutex mtx;
};

struct ServerData {
    // Load-time vectors are in full_base. Inserted ids are in full_base_delta, whose chunks never
    // move, so search reads them without a lock.
    static constexpr uint32_t FULL_BASE_SEGMENT_NODES = ChunkedVectorStore::CHUNK_NODES;
    std::vector<float> full_base;             // static section (loaded from disk)
    ChunkedVectorStore full_base_delta;       // delta section (lock-free)
    // write state of each delta vector: 0 empty, 1 being written, 2 complete
    ChunkedCounterStore full_base_delta_state;
    uint32_t base_static_capacity = 0;        // load-time node count N0

    // inserted ids' owner labels, initialized to UINT32_MAX
    ChunkedArray<uint32_t, ChunkedVectorStore::CHUNK_NODES,
                 ChunkedVectorStore::MAX_CHUNKS> membership_delta;

    // Chunked graph stores. Readers need no global lock, and adjacency rows change under node_locks.
    std::vector<uint32_t> membership;
    ChunkedArray<std::vector<uint32_t>, 10000, 1000> adj;
    ChunkedArray<uint32_t> global_to_local;   // init = UINT32_MAX
    ChunkedArray<uint32_t> local_to_global;   // init = UINT32_MAX
    std::atomic<uint32_t> n_local{0};
    std::atomic<uint32_t> n_base{0};
    uint32_t dim;
    uint32_t my_partition;
    uint32_t n_servers;
    bool use_partition_graph;

    // per-node locks by local row, which never move (capacity matches adj, 10M rows)
    ChunkedSlots<NodeLock, 10000, 1000> node_locks;

    // prune when a row exceeds R + edge_overload (--edge-overload), back to R
    uint32_t edge_overload = 16;

    // Freed row slots (no mapping) reused by inserts and migrations. Migration steps and RESET change
    // it under the exclusive admin lock. Insert batches take a slot under free_slot_mutex.
    std::vector<uint32_t> free_slot_pool;
    std::mutex free_slot_mutex;
    ChunkedArray<uint64_t> migration_slot_generations; // absent segment means 0

    // Window statistics (stats_window.h): relaxed per-node counters of the distance evaluations of
    // each node's expansions on this server. Load-time ids use stats_window_w, inserted ids
    // stats_window_delta (grown at BEGIN). BEGIN fixes the node count N and zeroes [0, N).
    std::unique_ptr<std::atomic<uint32_t>[]> stats_window_w;
    uint32_t stats_window_static_nodes = 0;   // N0, set at load and RESET only (admin)
    ChunkedCounterStore stats_window_delta;   // ids >= N0, by id - N0
    std::atomic<uint32_t> stats_window_nodes{0};
    // release store at BEGIN, so the search path's acquire load sees the zeroed table and N
    std::atomic<bool> stats_window_active{false};
    std::atomic<uint64_t> stats_window_epoch{0};
    // serializes BEGIN, the snapshot and STOP (never taken by search)
    std::mutex stats_window_mutex;

    // own_only (--pq): owned nodes' vectors in full_own (by local slot), PQ codes of every node.
    // Otherwise every vector in full_base/full_base_delta.
    bool own_only = false;
    ann::PQTable pq_table;                 // pq_pivots.bin (PQ mode)
    std::vector<uint8_t> pq_codes_static;  // [base_static_capacity * M], pq_codes.bin
    ChunkedCodeStore pq_codes_delta;       // codes of inserted ids, by id - base_static_capacity
    ChunkedVectorStore full_own;           // own vectors by local slot
};

inline uint32_t n_local_load(const ServerData& d,
                              std::memory_order mo = std::memory_order_acquire) {
    return d.n_local.load(mo);
}
// returns the previous value, a unique local slot
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

// Grows node_locks. Existing locks never move. Throws std::length_error beyond 10M rows.
inline void ensure_node_locks(ServerData& d, uint32_t target) {
    if (!d.node_locks.grow_to(target)) throw std::length_error("node locks: capacity exceeded or allocation failed");
}

inline uint32_t n_base_load(const ServerData& d,
                            std::memory_order mo = std::memory_order_acquire) {
    return d.n_base.load(mo);
}

// release: writes before the bump are visible to readers that see the new n_base
inline void n_base_bump(ServerData& d, uint32_t want) {
    uint32_t cur = d.n_base.load(std::memory_order_relaxed);
    while (cur < want &&
           !d.n_base.compare_exchange_weak(cur, want,
                                            std::memory_order_release,
                                            std::memory_order_relaxed)) {}
}

// init and reset only
inline void n_base_set(ServerData& d, uint32_t value) {
    d.n_base.store(value, std::memory_order_release);
}

// global -> local, UINT32_MAX when unmapped or the chunk is absent
inline uint32_t g2l_checked(const ServerData& d, uint32_t gid) {
    using G2L = decltype(d.global_to_local);
    const uint32_t* segment = d.global_to_local.segment(gid / G2L::chunk_size);
    return segment ? segment[gid % G2L::chunk_size] : UINT32_MAX;
}

// PQ mode: the vector of `id` if this server holds it, else nullptr
inline const float* own_vec(const ServerData& d, uint32_t id) {
    const uint32_t slot = g2l_checked(d, id);
    if (slot == UINT32_MAX ||
        !d.full_own.segment_allocated(slot / ChunkedVectorStore::CHUNK_NODES)) return nullptr;
    return d.full_own.vec(slot);
}

// Full vectors: the vector of `id` once complete here (state 2, acquire), else nullptr.
inline const float* replicated_vec(const ServerData& d, uint32_t id) {
    if (id < d.base_static_capacity) return d.full_base.data() + static_cast<size_t>(id) * d.dim;
    const uint32_t delta = id - d.base_static_capacity;
    const auto* state = d.full_base_delta_state.counter(delta);
    if (state == nullptr || state->load(std::memory_order_acquire) != 2 ||
        !d.full_base_delta.segment_allocated(delta / ChunkedVectorStore::CHUNK_NODES)) return nullptr;
    return d.full_base_delta.vec(delta);
}

// the vector this server stores for `id`, or nullptr
inline const float* stored_vec(const ServerData& d, uint32_t id) {
    return d.own_only ? own_vec(d, id) : replicated_vec(d, id);
}

// Vector of `id`. In PQ mode nullptr for nodes held only as a code.
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

// Makes sure a delta chunk covers node_id. True if one was allocated (no-op in PQ mode).
inline bool server_ensure_vec_capacity(ServerData& d, uint32_t node_id) {
    if (d.own_only) return false;
    if (node_id < d.base_static_capacity) return false;
    uint32_t delta = node_id - d.base_static_capacity;
    size_t prev = d.full_base_delta.allocated_chunks();
    d.full_base_delta.ensure_capacity_for(delta);
    return d.full_base_delta.allocated_chunks() > prev;
}

// Full vectors: stores an inserted id's vector once. The first delivery writes it (state 0 -> 1 -> 2),
// later ones wait for state 2. True when this call wrote it.
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

// Inserted ids' labels are relaxed atomics: several deliveries store the same owner while searches
// read. The load-time labels change only under the exclusive admin lock.
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

// owner label of `id`, UINT32_MAX when not known here yet (never reads an absent chunk)
inline uint32_t mem_label_checked(const ServerData& d, uint32_t id) {
    if (id < d.base_static_capacity) return id < d.membership.size() ? d.membership[id] : UINT32_MAX;
    using Mem = decltype(d.membership_delta);
    const uint32_t delta = id - d.base_static_capacity;
    const uint32_t* segment = d.membership_delta.segment(delta / Mem::chunk_size);
    return segment ? __atomic_load_n(&segment[delta % Mem::chunk_size], __ATOMIC_RELAXED) : UINT32_MAX;
}

inline size_t mem_size(const ServerData& d) {
    return static_cast<size_t>(d.base_static_capacity)
         + d.membership_delta.total_capacity();
}

// covers node_id in the membership, new chunks filled with UINT32_MAX (thread-safe)
inline bool server_ensure_mem_capacity(ServerData& d, uint32_t node_id) {
    if (node_id < d.base_static_capacity) return false;
    uint32_t delta = node_id - d.base_static_capacity;
    size_t prev = d.membership_delta.allocated_chunks();
    d.membership_delta.ensure_capacity_for(delta, UINT32_MAX);
    return d.membership_delta.allocated_chunks() > prev;
}


// PQ code of `id`, or nullptr (full vectors, or a delta code not stored yet)
inline const uint8_t* pq_code(const ServerData& d, uint32_t id) {
    if (!d.own_only || d.pq_table.M == 0) return nullptr;
    if (id < d.base_static_capacity) {
        const size_t offset = static_cast<size_t>(id) * d.pq_table.M;
        return offset + d.pq_table.M <= d.pq_codes_static.size() ? d.pq_codes_static.data() + offset : nullptr;
    }
    return d.pq_codes_delta.code(id - d.base_static_capacity);
}

// True once this server has an inserted id's owner label and its code (PQ mode) or vector.
// Inserts skip candidates that are not complete yet.
inline bool delivered_here(const ServerData& d, uint32_t id) {
    if (id < d.base_static_capacity) return true;
    if (mem_label_checked(d, id) >= d.n_servers) return false;
    return d.own_only ? pq_code(d, id) != nullptr : replicated_vec(d, id) != nullptr;
}

// stores an inserted id's code once (later deliveries skip it)
inline void store_pq_code(ServerData& d, uint32_t gid, const float* vec) {
    if (!d.own_only || gid < d.base_static_capacity) return;
    d.pq_codes_delta.write_once(gid - d.base_static_capacity,
                                [&](uint8_t* out) { d.pq_table.encode(vec, out); });
}

// The caller publishes global_to_local[gid] = slot only after this returns.
inline void store_own_vector(ServerData& d, uint32_t slot, const float* vec) {
    d.full_own.ensure_capacity_for(slot);
    std::copy(vec, vec + d.dim, d.full_own.vec(slot));
}

// a vector this server does not store but has for this operation (a backlink's source)
struct KnownVector {
    uint32_t id = UINT32_MAX;
    const float* vec = nullptr;
};

// Squared L2 with the best data held: exact, ADC (vector vs code), code-to-code, else FLT_MAX.
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

// ADC table of the query, built on first use in a search step and timed separately.
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
    uint32_t exact = 0;    // neighbors scored against a held vector
    uint32_t adc = 0;      // neighbors scored against their code
    uint32_t missing = 0;  // neither: FLT_MAX
};

// how hybrid_score scored a neighbor
enum : uint8_t { kScoreAdc = 0, kScoreExact = 1, kScoreMissing = 2 };

// Scores `ids` against the query: exact for held vectors, ADC for codes, FLT_MAX (no work) if
// neither is here.
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

// allocated bytes of the stores, for the stats reply
struct StorageBytes {
    uint64_t vector_bytes = 0;    // full_base + full_base_delta (with state words), or full_own
    uint64_t code_bytes = 0;      // pq_codes_static + pq_codes_delta (with state bytes)
    uint64_t pq_table_bytes = 0;  // centroids + code-to-code table
    uint64_t own_vectors = 0;     // PQ mode: mapped local slots
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

// chunk allocations made by ensure_capacity
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

struct ProcessMetrics {
    // distance evaluations = exact + adc + rerank (W(v) counts the same work per expanded node)
    uint32_t distance_computations = 0;
    uint32_t exact_distances = 0;     // neighbors scored against a held vector
    uint32_t adc_distances = 0;       // neighbors scored against their PQ code
    uint32_t missing_codes = 0;       // neighbors with neither (FLT_MAX): no work
    uint32_t rerank_distances = 0;    // PQ mode: expanded entries re-scored exactly
    uint32_t adc_tables = 0;          // PQ mode: ADC tables built (0 or 1 per hop)
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

// g_resize_mutex: taken only by RESET.
extern std::shared_mutex g_resize_mutex;
extern std::mutex g_alloc_mutex;
extern LockMetrics g_lock_metrics;
extern GraphMetrics g_graph_metrics;
extern std::string g_data_dir;
extern int g_rank;
extern uint32_t g_start_node;


uint32_t load_server_data(ServerData& data, const std::string& data_dir, int rank);

// Clears every store and reloads it from data_dir in the start-up mode. The caller excludes all
// other work. Returns the start node.
uint32_t reset_server_data(ServerData& data, const std::string& data_dir, int rank);

void ensure_capacity(ServerData& data, uint32_t global_id, CapacityGrowth& result);

// FreshDiskANN Algorithm 3. `known` is a vector this server does not store (a backlink source).
std::vector<uint32_t> robust_prune(
    const ServerData& data,
    uint32_t p,
    std::vector<std::pair<uint32_t, float>>& candidates,
    uint32_t R,
    float alpha,
    const KnownVector* known = nullptr
);


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


std::vector<ResponseItem> batch_process_backlinks(
    ServerData& data,
    std::vector<WorkItem>& batch,
    TimingMetrics& metrics
);

std::vector<ResponseItem> batch_process_broadcasts(
    ServerData& data,
    std::vector<WorkItem>& batch,
    TimingMetrics& metrics
);

// Row slot for an inserted node: a freed slot from the pool, or a new one at n_local.
uint32_t insert_row_slot(ServerData& data);

std::vector<ResponseItem> batch_process_inserts(
    ServerData& data,
    std::vector<WorkItem>& batch,
    TimingMetrics& metrics
);

} // namespace tide
