// Stress test of ChunkedArray, ChunkedVectorStore, ChunkedCodeStore and ChunkedSlots.

#include "chunked_array.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

using tide::ChunkedArray;
using tide::ChunkedVectorStore;

static void check(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::abort();
    }
}

// single thread, uint32_t elements
static void test_single_uint32() {
    ChunkedArray<uint32_t, 1000, 100> arr;

    check(arr.allocated_chunks() == 0, "initial allocated_chunks == 0");
    check(arr.segment(0) == nullptr, "initial segment(0) == nullptr");

    // First access requires ensure
    arr.ensure_capacity_for(0, UINT32_MAX);
    check(arr.allocated_chunks() == 1, "after ensure(0): 1 chunk");
    check(arr.at(0) == UINT32_MAX, "init value applied");
    check(arr.at(999) == UINT32_MAX, "init value applied at chunk end");

    // Write + read back
    arr.at(0) = 42;
    arr.at(500) = 7;
    arr.at(999) = 123;
    check(arr.at(0) == 42, "write 0");
    check(arr.at(500) == 7, "write mid");
    check(arr.at(999) == 123, "write end");

    // Cross-chunk
    arr.ensure_capacity_for(2500, UINT32_MAX);
    check(arr.allocated_chunks() == 3, "after ensure(2500): 3 chunks");
    check(arr.at(2500) == UINT32_MAX, "second chunk init");
    arr.at(2500) = 99;
    check(arr.at(2500) == 99, "cross-chunk write");

    arr.clear();
    check(arr.allocated_chunks() == 0, "after clear: 0 chunks");

    std::printf("  test_single_uint32 OK\n");
}

// Concurrent ensure_capacity: threads allocate segments and write disjoint slots. Reads see the
// init values.
static void test_concurrent_ensure() {
    constexpr size_t N_THREADS = 8;
    constexpr size_t N_OPS = 20000;
    constexpr size_t CHUNK = 1000;
    constexpr size_t MAX_CHUNKS = 200;
    ChunkedArray<uint32_t, CHUNK, MAX_CHUNKS> arr;

    std::vector<std::thread> threads;
    std::atomic<size_t> errors{0};

    for (size_t t = 0; t < N_THREADS; t++) {
        threads.emplace_back([&, t]() {
            // Each thread writes to its own region within shared chunks:
            //   thread t writes to slots [t*N_OPS .. (t+1)*N_OPS) but the
            //   chunk that covers that range may still be lazily allocated
            //   by any thread (alloc race).
            size_t base = t * N_OPS;
            for (size_t i = 0; i < N_OPS; i++) {
                size_t idx = base + i;
                arr.ensure_capacity_for(idx, UINT32_MAX);
                arr.at(idx) = static_cast<uint32_t>(idx);
            }
            // Read-back verification (same thread reading own writes).
            for (size_t i = 0; i < N_OPS; i++) {
                size_t idx = base + i;
                if (arr.at(idx) != static_cast<uint32_t>(idx)) errors++;
            }
        });
    }
    for (auto& th : threads) th.join();

    check(errors.load() == 0, "all writes readable");
    check(arr.allocated_chunks() <= MAX_CHUNKS, "allocated_chunks <= MAX");
    check(arr.allocated_chunks() > 0, "at least one chunk allocated");

    std::printf("  test_concurrent_ensure OK (allocated=%zu)\n", arr.allocated_chunks());
}

// concurrent writes to disjoint slots
static void test_disjoint_writes() {
    constexpr size_t N_THREADS = 8;
    constexpr size_t N_SLOTS_PER_THREAD = 10000;
    ChunkedArray<uint32_t, 1000, 100> arr;

    // Pre-allocate all chunks needed.
    arr.ensure_capacity_for(N_THREADS * N_SLOTS_PER_THREAD - 1, 0);

    std::vector<std::thread> threads;
    for (size_t t = 0; t < N_THREADS; t++) {
        threads.emplace_back([&, t]() {
            size_t base = t * N_SLOTS_PER_THREAD;
            for (size_t i = 0; i < N_SLOTS_PER_THREAD; i++) {
                arr.at(base + i) = static_cast<uint32_t>(base + i);
            }
        });
    }
    for (auto& th : threads) th.join();

    // Verify every slot has its expected value.
    for (size_t t = 0; t < N_THREADS; t++) {
        for (size_t i = 0; i < N_SLOTS_PER_THREAD; i++) {
            size_t idx = t * N_SLOTS_PER_THREAD + i;
            check(arr.at(idx) == static_cast<uint32_t>(idx), "disjoint slot value");
        }
    }
    std::printf("  test_disjoint_writes OK\n");
}

// ChunkedVectorStore, single thread
static void test_vector_store_single() {
    constexpr size_t DIM = 128;
    ChunkedVectorStore store(DIM);

    check(store.dim() == DIM, "dim set");
    check(store.allocated_chunks() == 0, "initial 0 chunks");

    store.ensure_capacity_for(0);
    check(store.allocated_chunks() == 1, "1 chunk after ensure(0)");

    // Write a vector
    float* v0 = store.vec(0);
    for (size_t i = 0; i < DIM; i++) v0[i] = static_cast<float>(i);

    // Read back
    const float* v0r = store.vec(0);
    for (size_t i = 0; i < DIM; i++) {
        check(v0r[i] == static_cast<float>(i), "vec read back");
    }

    // Different node in same chunk
    float* v500 = store.vec(500);
    for (size_t i = 0; i < DIM; i++) v500[i] = static_cast<float>(i * 2);
    const float* v500r = store.vec(500);
    for (size_t i = 0; i < DIM; i++) {
        check(v500r[i] == static_cast<float>(i * 2), "second vec read back");
    }

    // Verify v0 unchanged
    for (size_t i = 0; i < DIM; i++) {
        check(v0r[i] == static_cast<float>(i), "v0 unchanged");
    }

    // Cross-chunk
    store.ensure_capacity_for(150000);
    check(store.allocated_chunks() == 2, "2 chunks after cross-chunk ensure");

    std::printf("  test_vector_store_single OK\n");
}

// ChunkedVectorStore, concurrent disjoint writes (as broadcasts do)
static void test_vector_store_concurrent() {
    constexpr size_t DIM = 128;
    constexpr size_t N_THREADS = 8;
    constexpr size_t N_NODES_PER_THREAD = 5000;
    ChunkedVectorStore store(DIM);

    // Pre-allocate enough chunks.
    store.ensure_capacity_for(N_THREADS * N_NODES_PER_THREAD - 1);

    std::vector<std::thread> threads;
    for (size_t t = 0; t < N_THREADS; t++) {
        threads.emplace_back([&, t]() {
            size_t base = t * N_NODES_PER_THREAD;
            for (size_t i = 0; i < N_NODES_PER_THREAD; i++) {
                float* v = store.vec(base + i);
                for (size_t d = 0; d < DIM; d++) {
                    v[d] = static_cast<float>((base + i) * 1000 + d);
                }
            }
        });
    }
    for (auto& th : threads) th.join();

    // Verify
    for (size_t t = 0; t < N_THREADS; t++) {
        for (size_t i = 0; i < N_NODES_PER_THREAD; i++) {
            size_t node = t * N_NODES_PER_THREAD + i;
            const float* v = store.vec(node);
            for (size_t d = 0; d < DIM; d++) {
                float expected = static_cast<float>(node * 1000 + d);
                check(v[d] == expected, "vec value");
            }
        }
    }
    std::printf("  test_vector_store_concurrent OK\n");
}

// vector<uint32_t> elements (as adj)
static void test_vector_inner() {
    ChunkedArray<std::vector<uint32_t>, 100, 10> arr;
    arr.ensure_capacity_for(50);

    auto& v = arr.at(50);
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);

    check(arr.at(50).size() == 3, "inner vector size");
    check(arr.at(50)[0] == 1, "inner[0]");
    check(arr.at(50)[2] == 3, "inner[2]");

    // Different slot
    arr.at(51).push_back(99);
    check(arr.at(51).size() == 1, "neighbor slot size");
    check(arr.at(50).size() == 3, "original slot unchanged");

    std::printf("  test_vector_inner OK\n");
}

// ChunkedCodeStore: writers race on write_once for the same ids across two chunks while readers poll
// code(). One writer wins each id, and a reader sees nullptr or the complete code.
static void test_code_store_write_once_concurrent() {
    constexpr size_t CODE = 16, IDS = 2000, WRITERS = 6, READERS = 3;
    constexpr size_t FIRST = tide::ChunkedCodeStore::CHUNK_NODES - IDS / 2;  // spans two chunks
    tide::ChunkedCodeStore store;
    store.init(CODE);
    std::atomic<size_t> wins{0};
    std::atomic<bool> stop{false};
    std::atomic<size_t> torn{0};
    std::vector<std::thread> threads;
    for (size_t w = 0; w < WRITERS; ++w) {
        threads.emplace_back([&, w]() {
            for (size_t k = 0; k < IDS; ++k) {
                const size_t id = FIRST + (k * 7 + w * 13) % IDS;
                if (store.write_once(id, [id](uint8_t* out) {
                        for (size_t b = 0; b < CODE; ++b) out[b] = static_cast<uint8_t>((id + b) & 0xff);
                    }))
                    wins.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (size_t r = 0; r < READERS; ++r) {
        threads.emplace_back([&]() {
            while (!stop.load(std::memory_order_acquire)) {
                for (size_t id = FIRST; id < FIRST + IDS; ++id) {
                    const uint8_t* code = store.code(id);
                    if (!code) continue;
                    for (size_t b = 0; b < CODE; ++b)
                        if (code[b] != static_cast<uint8_t>((id + b) & 0xff)) torn.fetch_add(1);
                }
            }
        });
    }
    for (size_t w = 0; w < WRITERS; ++w) threads[w].join();
    stop.store(true, std::memory_order_release);
    for (size_t t = WRITERS; t < threads.size(); ++t) threads[t].join();
    check(wins.load() == IDS, "exactly one writer per id");
    check(torn.load() == 0, "no reader saw a partial code");
    for (size_t id = FIRST; id < FIRST + IDS; ++id) check(store.ready(id), "every id ready");
    check(store.allocated_chunks() == 2, "both chunks allocated lazily");
    std::printf("  test_code_store_write_once_concurrent OK\n");
}

// ChunkedSlots: growth publishes constructed elements that never move. Growth beyond max_size fails.
struct alignas(64) Slot {
    std::mutex mtx;
    uint64_t value = 0;
};

static void test_slots_single() {
    tide::ChunkedSlots<Slot, 100, 10> slots;
    check(slots.size() == 0, "slots start empty");
    check(slots.grow_to(1) && slots.size() == 1 && slots[0].value == 0, "one constructed slot");
    Slot* first = &slots[0];
    check(slots.grow_to(250) && slots.size() == 250, "growth across chunks");
    check(&slots[0] == first, "a slot never moves");
    check(reinterpret_cast<uintptr_t>(&slots[249]) % alignof(Slot) == 0, "over-aligned slots stay aligned");
    check(slots.grow_to(10) && slots.size() == 250, "growing to a smaller size is a no-op");
    check(!slots.grow_to(1001) && slots.size() == 250, "beyond max_size: refused, nothing published");
    check(slots.grow_to(1000) && slots.size() == 1000, "up to max_size");
    slots.clear();
    check(slots.size() == 0, "clear empties the store");
    std::printf("  test_slots_single OK\n");
}

// ChunkedSlots grown by two writers while three readers lock published slots.
static void test_slots_grow_while_locking() {
    constexpr size_t TARGET = 20000, READERS = 3, WRITERS = 2;
    tide::ChunkedSlots<Slot, 1000, 100> slots;
    check(slots.grow_to(1), "first slot");
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> locked{0};
    std::vector<std::thread> threads;
    for (size_t w = 0; w < WRITERS; ++w) {
        threads.emplace_back([&, w]() {
            for (size_t n = 2 + w; n <= TARGET; n += WRITERS) {
                check(slots.grow_to(n), "grow while readers lock");
                std::lock_guard<std::mutex> lock(slots[n - 1].mtx);
                ++slots[n - 1].value;
            }
        });
    }
    for (size_t r = 0; r < READERS; ++r) {
        threads.emplace_back([&, r]() {
            uint64_t state = r + 1;
            while (!stop.load(std::memory_order_acquire)) {
                const size_t size = slots.size();
                state = state * 6364136223846793005ull + 1442695040888963407ull;
                Slot& slot = slots[(state >> 33) % size];
                std::lock_guard<std::mutex> lock(slot.mtx);
                ++slot.value;
                locked.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (size_t w = 0; w < WRITERS; ++w) threads[w].join();
    stop.store(true, std::memory_order_release);
    for (size_t t = WRITERS; t < threads.size(); ++t) threads[t].join();
    check(slots.size() == TARGET, "grown to the target");
    uint64_t total = 0;
    for (size_t i = 0; i < TARGET; ++i) total += slots[i].value;
    check(total == locked.load() + (TARGET - 1), "every increment made under its slot's lock was kept");
    std::printf("  test_slots_grow_while_locking OK (%llu reader locks)\n", (unsigned long long)locked.load());
}

int main() {
    std::printf("ChunkedArray stress tests:\n");
    test_single_uint32();
    test_concurrent_ensure();
    test_disjoint_writes();
    test_vector_store_single();
    test_vector_store_concurrent();
    test_vector_inner();
    test_code_store_write_once_concurrent();
    test_slots_single();
    test_slots_grow_while_locking();
    std::printf("PASS\n");
    return 0;
}
