#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <new>

namespace tide {

// =============================================================================
// ChunkedArray<T> — reallocation-free chunked storage.
//
// Design:
//   - Outer container = fixed-size atomic pointer array (no reallocation ever).
//   - Each slot points to a heap-allocated chunk of CHUNK_SIZE T elements,
//     lazily allocated on first use.
//   - at(idx) is a hot-path acquire load; safe for concurrent readers vs
//     concurrent ensure_capacity callers.
//   - ensure_segment / ensure_capacity_for use a single mutex to serialize
//     chunk allocation; the actual store of the new chunk pointer is a
//     release atomic store, paired with the acquire load on the read side.
//   - Disjoint slot writes by different threads are safe — each chunk slot
//     is plain memory after the pointer is published.
// =============================================================================
template <typename T, std::size_t CHUNK_SIZE = 100000, std::size_t MAX_CHUNKS = 1000>
class ChunkedArray {
    static_assert(CHUNK_SIZE > 0, "CHUNK_SIZE must be positive");
    static_assert(MAX_CHUNKS > 0, "MAX_CHUNKS must be positive");

public:
    using value_type = T;
    static constexpr std::size_t chunk_size = CHUNK_SIZE;
    static constexpr std::size_t max_chunks = MAX_CHUNKS;
    static constexpr std::size_t max_capacity = CHUNK_SIZE * MAX_CHUNKS;

    ChunkedArray() = default;
    ~ChunkedArray() { clear(); }

    ChunkedArray(const ChunkedArray&) = delete;
    ChunkedArray& operator=(const ChunkedArray&) = delete;

    // Hot path: acquire-load the chunk pointer, index into it.
    T& at(std::size_t idx) {
        std::size_t seg = idx / CHUNK_SIZE;
        std::size_t off = idx % CHUNK_SIZE;
        return chunks_[seg].load(std::memory_order_acquire)[off];
    }
    const T& at(std::size_t idx) const {
        std::size_t seg = idx / CHUNK_SIZE;
        std::size_t off = idx % CHUNK_SIZE;
        return chunks_[seg].load(std::memory_order_acquire)[off];
    }

    // Convenience: same semantics as at(), so call sites can write
    // `data.adj[i]`.
    T& operator[](std::size_t idx) { return at(idx); }
    const T& operator[](std::size_t idx) const { return at(idx); }

    // Returns the raw chunk pointer for segment `seg`, or nullptr if not
    // allocated. Caller must use acquire ordering — provided by this load.
    T* segment(std::size_t seg) const {
        if (seg >= MAX_CHUNKS) return nullptr;
        return chunks_[seg].load(std::memory_order_acquire);
    }

    // Allocate a single segment if not already present. `init` is used to
    // initialize each element of a newly-allocated chunk. Idempotent and
    // thread-safe; multiple concurrent callers will serialize on alloc_mtx_
    // and only one will actually allocate.
    bool ensure_segment(std::size_t seg, const T& init = T{}) {
        if (seg >= MAX_CHUNKS) return false;
        if (chunks_[seg].load(std::memory_order_acquire) != nullptr) return true;
        std::lock_guard<std::mutex> g(alloc_mtx_);
        if (chunks_[seg].load(std::memory_order_relaxed) != nullptr) return true;
        T* chunk = new T[CHUNK_SIZE];
        for (std::size_t i = 0; i < CHUNK_SIZE; i++) chunk[i] = init;
        chunks_[seg].store(chunk, std::memory_order_release);
        n_allocated_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // Allocate all segments up to and including the segment that covers `idx`.
    bool ensure_capacity_for(std::size_t idx, const T& init = T{}) {
        std::size_t seg_needed = idx / CHUNK_SIZE;
        if (seg_needed >= MAX_CHUNKS) return false;
        for (std::size_t s = 0; s <= seg_needed; s++) {
            if (!ensure_segment(s, init)) return false;
        }
        return true;
    }

    // Number of allocated segments (for capacity reporting).
    std::size_t allocated_chunks() const {
        return n_allocated_.load(std::memory_order_acquire);
    }

    // Total addressable capacity (= allocated_chunks * CHUNK_SIZE).
    std::size_t total_capacity() const {
        return allocated_chunks() * CHUNK_SIZE;
    }

    // Free all chunks. NOT thread-safe with respect to concurrent at/ensure.
    // Caller must ensure exclusive access (admin RESET path).
    void clear() {
        std::lock_guard<std::mutex> g(alloc_mtx_);
        for (std::size_t s = 0; s < MAX_CHUNKS; s++) {
            T* p = chunks_[s].exchange(nullptr, std::memory_order_acq_rel);
            delete[] p;
        }
        n_allocated_.store(0, std::memory_order_release);
    }

private:
    std::atomic<T*> chunks_[MAX_CHUNKS]{};
    std::atomic<std::size_t> n_allocated_{0};
    mutable std::mutex alloc_mtx_;
};

// =============================================================================
// ChunkedSlots<T> — default-constructed elements that never move (the per-node
// mutexes), lazily chunked, with a published size.
//
// size() and operator[] are lock-free: a reader that sees i < size() (acquire
// load) finds element i's chunk published and the element constructed.
// grow_to(n) allocates every chunk covering [0, n) under a mutex and then
// publishes n (release store); it never moves or reallocates anything a reader
// can reach. (A std::deque cannot be read while it grows: emplace_back rewrites
// the members size() and operator[] read and may reallocate its block map.)
// clear() is NOT thread-safe with readers (admin RESET path only).
// =============================================================================
template <typename T, std::size_t CHUNK_SIZE = 10000, std::size_t MAX_CHUNKS = 1000>
class ChunkedSlots {
    static_assert(CHUNK_SIZE > 0 && MAX_CHUNKS > 0, "ChunkedSlots needs positive sizes");

public:
    static constexpr std::size_t chunk_size = CHUNK_SIZE;
    static constexpr std::size_t max_size = CHUNK_SIZE * MAX_CHUNKS;

    ChunkedSlots() = default;
    ~ChunkedSlots() { clear(); }

    ChunkedSlots(const ChunkedSlots&) = delete;
    ChunkedSlots& operator=(const ChunkedSlots&) = delete;

    std::size_t size() const { return size_.load(std::memory_order_acquire); }

    // Element i, for i below a size() the caller observed.
    T& operator[](std::size_t i) {
        return chunks_[i / CHUNK_SIZE].load(std::memory_order_acquire)[i % CHUNK_SIZE];
    }
    const T& operator[](std::size_t i) const {
        return chunks_[i / CHUNK_SIZE].load(std::memory_order_acquire)[i % CHUNK_SIZE];
    }

    // Grows to at least n elements. Idempotent and thread-safe; false (nothing
    // published) when n exceeds max_size or a chunk allocation failed.
    bool grow_to(std::size_t n) {
        if (size() >= n) return true;
        if (n > max_size) return false;
        std::lock_guard<std::mutex> g(mutex_);
        if (size_.load(std::memory_order_relaxed) >= n) return true;
        for (std::size_t seg = 0; seg * CHUNK_SIZE < n; ++seg) {
            if (chunks_[seg].load(std::memory_order_relaxed) != nullptr) continue;
            T* chunk = new (std::nothrow) T[CHUNK_SIZE];
            if (chunk == nullptr) return false;
            chunks_[seg].store(chunk, std::memory_order_release);
        }
        size_.store(n, std::memory_order_release);
        return true;
    }

    // Free every chunk. NOT thread-safe with respect to concurrent readers or
    // grow_to (admin RESET path only).
    void clear() {
        std::lock_guard<std::mutex> g(mutex_);
        size_.store(0, std::memory_order_release);
        for (std::size_t s = 0; s < MAX_CHUNKS; s++)
            delete[] chunks_[s].exchange(nullptr, std::memory_order_acq_rel);
    }

private:
    std::atomic<T*> chunks_[MAX_CHUNKS]{};
    std::atomic<std::size_t> size_{0};
    std::mutex mutex_;
};

// =============================================================================
// ChunkedVectorStore — float storage for fixed-dim vectors, lazily chunked.
//
// Used for the delta section of `full_base`. Each chunk holds CHUNK_NODES
// nodes' worth of floats (= CHUNK_NODES * dim floats). The dim is set at
// construction (or via init()) since dim is runtime-known.
// =============================================================================
class ChunkedVectorStore {
public:
    static constexpr std::size_t CHUNK_NODES = 100000;
    static constexpr std::size_t MAX_CHUNKS = 1000;
    static constexpr std::size_t max_node_capacity = CHUNK_NODES * MAX_CHUNKS;

    ChunkedVectorStore() = default;
    explicit ChunkedVectorStore(std::size_t dim) : dim_(dim) {}
    ~ChunkedVectorStore() { clear(); }

    ChunkedVectorStore(const ChunkedVectorStore&) = delete;
    ChunkedVectorStore& operator=(const ChunkedVectorStore&) = delete;

    void init(std::size_t dim) { dim_ = dim; }
    std::size_t dim() const { return dim_; }

    // Hot path: returns pointer to the dim floats for node `node_idx` (within
    // the delta region — caller offsets node_idx by base_static_capacity).
    float* vec(std::size_t node_idx) {
        std::size_t seg = node_idx / CHUNK_NODES;
        std::size_t off = (node_idx % CHUNK_NODES) * dim_;
        return chunks_[seg].load(std::memory_order_acquire) + off;
    }
    const float* vec(std::size_t node_idx) const {
        std::size_t seg = node_idx / CHUNK_NODES;
        std::size_t off = (node_idx % CHUNK_NODES) * dim_;
        return chunks_[seg].load(std::memory_order_acquire) + off;
    }

    // Allocate segment if not present. Idempotent + thread-safe.
    bool ensure_segment(std::size_t seg) {
        if (seg >= MAX_CHUNKS) return false;
        if (chunks_[seg].load(std::memory_order_acquire) != nullptr) return true;
        std::lock_guard<std::mutex> g(alloc_mtx_);
        if (chunks_[seg].load(std::memory_order_relaxed) != nullptr) return true;
        float* chunk = new float[CHUNK_NODES * dim_]();  // zero-init
        chunks_[seg].store(chunk, std::memory_order_release);
        n_allocated_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool ensure_capacity_for(std::size_t node_idx) {
        std::size_t seg_needed = node_idx / CHUNK_NODES;
        if (seg_needed >= MAX_CHUNKS) return false;
        for (std::size_t s = 0; s <= seg_needed; s++) {
            if (!ensure_segment(s)) return false;
        }
        return true;
    }

    // Returns the segment index covering a given node_idx; useful for
    // checking whether a chunk exists without forcing allocation.
    bool segment_allocated(std::size_t seg) const {
        if (seg >= MAX_CHUNKS) return false;
        return chunks_[seg].load(std::memory_order_acquire) != nullptr;
    }

    std::size_t allocated_chunks() const {
        return n_allocated_.load(std::memory_order_acquire);
    }

    std::size_t total_node_capacity() const {
        return allocated_chunks() * CHUNK_NODES;
    }

    void clear() {
        std::lock_guard<std::mutex> g(alloc_mtx_);
        for (std::size_t s = 0; s < MAX_CHUNKS; s++) {
            float* p = chunks_[s].exchange(nullptr, std::memory_order_acq_rel);
            delete[] p;
        }
        n_allocated_.store(0, std::memory_order_release);
    }

private:
    std::atomic<float*> chunks_[MAX_CHUNKS]{};
    std::atomic<std::size_t> n_allocated_{0};
    mutable std::mutex alloc_mtx_;
    std::size_t dim_ = 0;
};

// =============================================================================
// ChunkedCounterStore — std::atomic<uint32_t> counters, lazily chunked.
//
// The window statistics of the ids inserted after load: one relaxed
// counter per id, indexed by the delta index (global id minus the load-time
// node count). ChunkedArray<std::atomic<uint32_t>> does not compile (an
// atomic is neither copyable nor assignable, which ensure_segment's fill
// needs), so this is its own small class. A chunk is published (release
// store) only after every counter in it is zero; counter() returns nullptr
// for an absent chunk, so a reader never indexes unallocated memory. Growth
// (ensure_capacity_for) and zero() may run while other threads count:
// allocation is serialized by a mutex and never moves a published chunk.
// =============================================================================
class ChunkedCounterStore {
public:
    static constexpr std::size_t CHUNK_NODES = 100000;
    static constexpr std::size_t MAX_CHUNKS = 1000;
    using Counter = std::atomic<uint32_t>;

    ChunkedCounterStore() = default;
    ~ChunkedCounterStore() { clear(); }

    ChunkedCounterStore(const ChunkedCounterStore&) = delete;
    ChunkedCounterStore& operator=(const ChunkedCounterStore&) = delete;

    // The counter of delta index `idx`, or nullptr when its chunk is absent. Lock-free.
    Counter* counter(std::size_t idx) const {
        const std::size_t seg = idx / CHUNK_NODES;
        if (seg >= MAX_CHUNKS) return nullptr;
        Counter* chunk = chunks_[seg].load(std::memory_order_acquire);
        return chunk ? chunk + idx % CHUNK_NODES : nullptr;
    }

    // Allocates every chunk up to the one covering `idx` (zeroed). Idempotent and
    // thread-safe; false when `idx` is beyond the capacity or allocation failed.
    bool ensure_capacity_for(std::size_t idx) {
        const std::size_t seg_needed = idx / CHUNK_NODES;
        if (seg_needed >= MAX_CHUNKS) return false;
        for (std::size_t seg = 0; seg <= seg_needed; ++seg) {
            if (chunks_[seg].load(std::memory_order_acquire) != nullptr) continue;
            std::lock_guard<std::mutex> g(alloc_mtx_);
            if (chunks_[seg].load(std::memory_order_relaxed) != nullptr) continue;
            Counter* chunk = new (std::nothrow) Counter[CHUNK_NODES];
            if (chunk == nullptr) return false;
            for (std::size_t i = 0; i < CHUNK_NODES; ++i) chunk[i].store(0, std::memory_order_relaxed);
            chunks_[seg].store(chunk, std::memory_order_release);
            n_allocated_.fetch_add(1, std::memory_order_relaxed);
        }
        return true;
    }

    // Zeroes the counters [0, count) of the allocated chunks (relaxed stores).
    void zero(std::size_t count) {
        for (std::size_t seg = 0; seg < MAX_CHUNKS && seg * CHUNK_NODES < count; ++seg) {
            Counter* chunk = chunks_[seg].load(std::memory_order_acquire);
            if (chunk == nullptr) continue;
            const std::size_t end = std::min(CHUNK_NODES, count - seg * CHUNK_NODES);
            for (std::size_t i = 0; i < end; ++i) chunk[i].store(0, std::memory_order_relaxed);
        }
    }

    std::size_t allocated_chunks() const {
        return n_allocated_.load(std::memory_order_acquire);
    }
    std::size_t total_capacity() const { return allocated_chunks() * CHUNK_NODES; }

    // Free all chunks. NOT thread-safe with respect to concurrent counter() or
    // ensure_capacity_for (admin RESET path only).
    void clear() {
        std::lock_guard<std::mutex> g(alloc_mtx_);
        for (std::size_t s = 0; s < MAX_CHUNKS; s++)
            delete[] chunks_[s].exchange(nullptr, std::memory_order_acq_rel);
        n_allocated_.store(0, std::memory_order_release);
    }

private:
    std::atomic<Counter*> chunks_[MAX_CHUNKS]{};
    std::atomic<std::size_t> n_allocated_{0};
    mutable std::mutex alloc_mtx_;
};

// =============================================================================
// ChunkedCodeStore — uint8_t storage for fixed-length PQ codes, lazily chunked,
// with a per-node ready state.
//
// Mirrors ChunkedVectorStore but stores `code_bytes` (= PQ M) bytes per node.
// Used for the codes of the nodes inserted after load (own and remote): every
// server keeps an M-byte code for every node, while full vectors are kept only
// for the nodes it owns. node_idx is the delta index (global id minus the
// load-time node count).
//
// A chunk is zero-initialised, and a zero code is a valid code (every chunk at
// centroid 0), so a slot that was allocated but not yet written must not be
// read as a code. Each node therefore has a state byte: EMPTY -> WRITING (one
// writer won the compare-exchange) -> READY (release store after the bytes).
// code() returns nullptr until READY (acquire load). Several maintenance paths
// deliver the same node's vector (insert, backlink, broadcast); the first writer
// encodes, the others skip, so the bytes are written exactly once.
// =============================================================================
class ChunkedCodeStore {
public:
    static constexpr std::size_t CHUNK_NODES = 100000;
    static constexpr std::size_t MAX_CHUNKS = 1000;
    static constexpr uint8_t EMPTY = 0, WRITING = 1, READY = 2;

    ChunkedCodeStore() = default;
    ~ChunkedCodeStore() { clear(); }

    ChunkedCodeStore(const ChunkedCodeStore&) = delete;
    ChunkedCodeStore& operator=(const ChunkedCodeStore&) = delete;

    void init(std::size_t code_bytes) { code_bytes_ = code_bytes; }
    std::size_t code_bytes() const { return code_bytes_; }

    // The code of delta node `node_idx`, or nullptr when its chunk is absent or
    // the code is not READY yet. Lock-free.
    const uint8_t* code(std::size_t node_idx) const {
        const std::size_t seg = node_idx / CHUNK_NODES;
        if (seg >= MAX_CHUNKS) return nullptr;
        uint8_t* chunk = codes_[seg].load(std::memory_order_acquire);
        std::atomic<uint8_t>* state = states_[seg].load(std::memory_order_acquire);
        if (!chunk || !state) return nullptr;
        const std::size_t off = node_idx % CHUNK_NODES;
        if (state[off].load(std::memory_order_acquire) != READY) return nullptr;
        return chunk + off * code_bytes_;
    }

    bool ready(std::size_t node_idx) const { return code(node_idx) != nullptr; }

    // Writes the code of `node_idx` once: `fill(dest)` writes code_bytes bytes.
    // Returns true when this call wrote it, false when it was already READY,
    // is being written by another thread, or the chunk could not be allocated.
    template <typename Fill>
    bool write_once(std::size_t node_idx, Fill&& fill) {
        const std::size_t seg = node_idx / CHUNK_NODES;
        if (!ensure_segment(seg)) return false;
        const std::size_t off = node_idx % CHUNK_NODES;
        std::atomic<uint8_t>& state = states_[seg].load(std::memory_order_acquire)[off];
        uint8_t expected = EMPTY;
        if (!state.compare_exchange_strong(expected, WRITING, std::memory_order_acq_rel)) return false;
        fill(codes_[seg].load(std::memory_order_acquire) + off * code_bytes_);
        state.store(READY, std::memory_order_release);
        return true;
    }

    // Allocate segment if not present. Idempotent + thread-safe. The state
    // array is published before the code chunk, and readers require both.
    bool ensure_segment(std::size_t seg) {
        if (seg >= MAX_CHUNKS || code_bytes_ == 0) return false;
        if (codes_[seg].load(std::memory_order_acquire) != nullptr) return true;
        std::lock_guard<std::mutex> g(alloc_mtx_);
        if (codes_[seg].load(std::memory_order_relaxed) != nullptr) return true;
        auto* state = new std::atomic<uint8_t>[CHUNK_NODES];
        for (std::size_t i = 0; i < CHUNK_NODES; ++i) state[i].store(EMPTY, std::memory_order_relaxed);
        auto* chunk = new uint8_t[CHUNK_NODES * code_bytes_]();
        states_[seg].store(state, std::memory_order_release);
        codes_[seg].store(chunk, std::memory_order_release);
        n_allocated_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    std::size_t allocated_chunks() const {
        return n_allocated_.load(std::memory_order_acquire);
    }

    // Allocated bytes: code bytes plus one state byte per node, per chunk.
    std::size_t allocated_bytes() const {
        return allocated_chunks() * CHUNK_NODES * (code_bytes_ + 1);
    }

    // Free all chunks. NOT thread-safe with respect to concurrent readers or
    // writers (admin RESET path only).
    void clear() {
        std::lock_guard<std::mutex> g(alloc_mtx_);
        for (std::size_t s = 0; s < MAX_CHUNKS; s++) {
            delete[] codes_[s].exchange(nullptr, std::memory_order_acq_rel);
            delete[] states_[s].exchange(nullptr, std::memory_order_acq_rel);
        }
        n_allocated_.store(0, std::memory_order_release);
    }

private:
    std::atomic<uint8_t*> codes_[MAX_CHUNKS]{};
    std::atomic<std::atomic<uint8_t>*> states_[MAX_CHUNKS]{};
    std::atomic<std::size_t> n_allocated_{0};
    mutable std::mutex alloc_mtx_;
    std::size_t code_bytes_ = 0;
};

}  // namespace tide
