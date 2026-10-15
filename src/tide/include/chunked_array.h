#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <new>

namespace tide {

// Fixed outer array of lazily allocated chunks, with lock-free at() and chunk allocation under a mutex.
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

    // same as at()
    T& operator[](std::size_t idx) { return at(idx); }
    const T& operator[](std::size_t idx) const { return at(idx); }

    // chunk of segment `seg`, or nullptr (acquire load)
    T* segment(std::size_t seg) const {
        if (seg >= MAX_CHUNKS) return nullptr;
        return chunks_[seg].load(std::memory_order_acquire);
    }

    // allocates one chunk filled with `init` (idempotent and thread-safe)
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

    // allocates every chunk up to the one covering idx
    bool ensure_capacity_for(std::size_t idx, const T& init = T{}) {
        std::size_t seg_needed = idx / CHUNK_SIZE;
        if (seg_needed >= MAX_CHUNKS) return false;
        for (std::size_t s = 0; s <= seg_needed; s++) {
            if (!ensure_segment(s, init)) return false;
        }
        return true;
    }

    std::size_t allocated_chunks() const {
        return n_allocated_.load(std::memory_order_acquire);
    }

    std::size_t total_capacity() const {
        return allocated_chunks() * CHUNK_SIZE;
    }

    // not thread-safe: RESET only
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

// Elements that never move (the per-node mutexes), lazily chunked. size() and operator[] are
// lock-free. grow_to() allocates under a mutex, then publishes the size. clear() is not thread-safe.
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

    // element i, for i below an observed size()
    T& operator[](std::size_t i) {
        return chunks_[i / CHUNK_SIZE].load(std::memory_order_acquire)[i % CHUNK_SIZE];
    }
    const T& operator[](std::size_t i) const {
        return chunks_[i / CHUNK_SIZE].load(std::memory_order_acquire)[i % CHUNK_SIZE];
    }

    // false (nothing published) beyond max_size or when an allocation fails
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

    // not thread-safe: RESET only
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

// Lazily chunked float storage for vectors of one dimension (the delta section of full_base).
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

    // node_idx is a delta index (id - base_static_capacity)
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

    // idempotent and thread-safe
    bool ensure_segment(std::size_t seg) {
        if (seg >= MAX_CHUNKS) return false;
        if (chunks_[seg].load(std::memory_order_acquire) != nullptr) return true;
        std::lock_guard<std::mutex> g(alloc_mtx_);
        if (chunks_[seg].load(std::memory_order_relaxed) != nullptr) return true;
        float* chunk = new float[CHUNK_NODES * dim_]();
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

// Lazily chunked atomic counters (window statistics of inserted ids, by delta index). A chunk is
// published only after it is zeroed. counter() returns nullptr for an absent chunk.
class ChunkedCounterStore {
public:
    static constexpr std::size_t CHUNK_NODES = 100000;
    static constexpr std::size_t MAX_CHUNKS = 1000;
    using Counter = std::atomic<uint32_t>;

    ChunkedCounterStore() = default;
    ~ChunkedCounterStore() { clear(); }

    ChunkedCounterStore(const ChunkedCounterStore&) = delete;
    ChunkedCounterStore& operator=(const ChunkedCounterStore&) = delete;

    // nullptr when the chunk is absent (lock-free)
    Counter* counter(std::size_t idx) const {
        const std::size_t seg = idx / CHUNK_NODES;
        if (seg >= MAX_CHUNKS) return nullptr;
        Counter* chunk = chunks_[seg].load(std::memory_order_acquire);
        return chunk ? chunk + idx % CHUNK_NODES : nullptr;
    }

    // allocates zeroed chunks up to idx (false beyond capacity or on allocation failure)
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

    // zeroes [0, count) of the allocated chunks (relaxed stores)
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

    // not thread-safe: RESET only
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

// Lazily chunked PQ codes of inserted ids (by delta index), with a state byte per node:
// EMPTY -> WRITING -> READY. code() returns nullptr until READY. The first writer encodes.
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

    // nullptr when the chunk is absent or the code is not READY (lock-free)
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

    // Writes the code once. True when this call wrote it.
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

    // idempotent and thread-safe, publishing the state array before the code chunk
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

    // code bytes plus one state byte per node
    std::size_t allocated_bytes() const {
        return allocated_chunks() * CHUNK_NODES * (code_bytes_ + 1);
    }

    // not thread-safe: RESET only
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
