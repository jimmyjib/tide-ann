// Concurrent searches, inserts and window controls in both storage modes (a race test under TSan).
#include "server_types.h"
#include "membership_migration.h"
#include "stats_window.h"
#include "protocol/broadcast_messages.h"
#include "protocol/insert_protocol.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace tide {
std::shared_mutex g_resize_mutex;
std::mutex g_alloc_mutex;
LockMetrics g_lock_metrics;
GraphMetrics g_graph_metrics;
std::string g_data_dir;
int g_rank = 0;
uint32_t g_start_node = 0;
}

namespace {

using namespace tide;
using Clock = std::chrono::steady_clock;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "CHECK FAILED: %s\n", what);
        std::abort();
    }
}

constexpr uint32_t kDim = 8, kM = 4, kServers = 2, kRank = 0;
constexpr uint32_t kNodes = 3000;           // load-time ids
constexpr uint32_t kDegree = 12, kR = 12;   // load-time row length, insert R
constexpr float kAlpha = 1.2f;
constexpr uint32_t kMovedOut = 64;          // rows an ACCEPT frees before the stress
constexpr uint32_t kNewIds = 1440;          // inserted ids, every third one the other server's
constexpr uint32_t kBatch = 8;
constexpr uint32_t kCandidates = 16, kBacklinkTargets = 4;
constexpr uint32_t kSearchThreads = 3, kMaintenanceThreads = 3;
constexpr uint32_t kL = 24, kK = 10, kW = 4;
constexpr uint32_t kStart = 0;              // entry node (owned here, never moved)
constexpr uint32_t kTotalIds = kNodes + kNewIds;

struct Lcg {
    uint64_t state;
    uint32_t next() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(state >> 33);
    }
};

// Integer coordinates in [0, 16): every coordinate pair is a centroid of the
// pivots below, so a code reconstructs its vector and the codes are exact.
std::vector<float> random_vector(Lcg& rng) {
    std::vector<float> v(kDim);
    for (auto& x : v) x = static_cast<float>(rng.next() % 16);
    return v;
}

ann::PQTable integer_pivots() {
    ann::PQTable pq;
    pq.dim = kDim; pq.M = kM; pq.sub_dim = kDim / kM;
    pq.centroids.assign(static_cast<size_t>(kM) * ann::PQTable::K * pq.sub_dim, 0.0f);
    for (uint32_t m = 0; m < kM; ++m) {
        for (uint32_t k = 0; k < ann::PQTable::K; ++k) {
            float* c = pq.centroid_ptr_mut(m, k);
            c[0] = static_cast<float>(k % 16);
            c[1] = static_cast<float>(k / 16);
        }
    }
    pq.build_sym_table();
    return pq;
}

// The coordinator's view, fixed before any thread starts: every id's vector and owner.
struct World {
    std::vector<std::vector<float>> vectors;
    std::vector<uint32_t> owners;
};

World make_world() {
    World world;
    Lcg rng{0x7a11ull};
    for (uint32_t id = 0; id < kTotalIds; ++id) {
        world.vectors.push_back(random_vector(rng));
        const bool load_time = id < kNodes;
        world.owners.push_back(load_time ? (id % 4 == 3 ? 1u : 0u) : ((id - kNodes) % 3 == 2 ? 1u : 0u));
    }
    return world;
}

float l2(const std::vector<float>& a, const std::vector<float>& b) {
    return ann::l2_distance_squared_scalar(a.data(), b.data(), kDim);
}

void make_server(ServerData& data, const World& world, bool own_only) {
    data.own_only = own_only;
    data.dim = kDim; data.my_partition = kRank; data.n_servers = kServers;
    data.use_partition_graph = true;
    data.base_static_capacity = kNodes;
    data.membership.assign(world.owners.begin(), world.owners.begin() + kNodes);
    n_base_set(data, kNodes);
    data.global_to_local.ensure_capacity_for(kNodes - 1, UINT32_MAX);
    if (own_only) {
        data.pq_table = integer_pivots();
        data.pq_codes_static.resize(static_cast<size_t>(kNodes) * kM);
        for (uint32_t id = 0; id < kNodes; ++id)
            data.pq_table.encode(world.vectors[id].data(), data.pq_codes_static.data() + static_cast<size_t>(id) * kM);
        data.pq_codes_delta.init(kM);
        data.full_own.init(kDim);
    } else {
        data.full_base_delta.init(kDim);
        for (uint32_t id = 0; id < kNodes; ++id)
            data.full_base.insert(data.full_base.end(), world.vectors[id].begin(), world.vectors[id].end());
    }
    Lcg rng{0x9e3779b9ull};
    uint32_t slot = 0;
    for (uint32_t id = 0; id < kNodes; ++id) {
        if (world.owners[id] != kRank) continue;
        std::vector<uint32_t> row;
        while (row.size() < kDegree) {
            const uint32_t v = rng.next() % kNodes;
            if (v != id && std::find(row.begin(), row.end(), v) == row.end()) row.push_back(v);
        }
        data.adj.ensure_capacity_for(slot);
        data.local_to_global.ensure_capacity_for(slot, UINT32_MAX);
        if (own_only) store_own_vector(data, slot, world.vectors[id].data());
        data.adj.at(slot) = row;
        data.local_to_global.at(slot) = id;
        data.global_to_local.at(id) = slot;
        ++slot;
    }
    n_local_set(data, slot);
    ensure_node_locks(data, slot);
    stats_window_allocate(data, kNodes);
}

// Moves kMovedOut owned load-time rows to the other server through PREPARE,
// COMMIT and ACCEPT (the server runs these under its exclusive admin lock):
// their slots enter the free pool. Returns the freed slots.
std::set<uint32_t> free_rows_by_accept(ServerData& data, World& world) {
    MembershipMigrationPlan plan;
    plan.n_servers = kServers; plan.epoch_from = 0; plan.epoch_to = 1;
    std::set<uint32_t> freed;
    std::vector<uint32_t> moved;  // ACCEPT drops the plan, so the ids are kept here
    for (uint32_t id = 1000; plan.moves.size() < kMovedOut; ++id) {
        if (world.owners[id] != kRank) continue;
        MembershipMove move;
        move.global_id = id; move.from_partition = kRank; move.to_partition = 1;
        move.adjacency = data.adj.at(g2l_checked(data, id));
        move.adjacency_digest = migration_row_digest(move.adjacency);
        plan.moves.push_back(move);
        moved.push_back(id);
        freed.insert(g2l_checked(data, id));
    }
    MembershipMigrationJournal journal;
    std::vector<uint64_t> accepted;
    std::string error;
    check(prepare_membership_migration(data, plan.moves, 0, 1, journal, error), "PREPARE of the moved rows");
    check(commit_membership_migration(data, plan.moves, journal, error), "COMMIT of the moved rows");
    check(accept_membership_migration_history(data, plan, journal, accepted, error), "ACCEPT frees the rows");
    for (const uint32_t id : moved) world.owners[id] = 1;
    check(data.free_slot_pool.size() == kMovedOut &&
          std::set<uint32_t>(data.free_slot_pool.begin(), data.free_slot_pool.end()) == freed,
          "the freed slots are the pool");
    return freed;
}

ann::WorkItem work_item(std::vector<char> request, ann::MessageType type) {
    ann::WorkItem item;
    item.request = std::move(request);
    item.msg_type = type;
    item.enqueue_time = Clock::now();
    return item;
}

std::vector<char> broadcast_request(uint32_t id, const std::vector<float>& vec, uint32_t home) {
    std::vector<char> request(sizeof(ann::VectorBroadcastHeader) + kDim * sizeof(float));
    ann::VectorBroadcastHeader header{};
    header.msg_type = ann::MessageType::VECTOR_BROADCAST;
    header.node_id = id; header.dim = kDim; header.home_partition = home;
    std::memcpy(request.data(), &header, sizeof(header));
    std::memcpy(request.data() + sizeof(header), vec.data(), kDim * sizeof(float));
    return request;
}

struct Queues {
    ann::ThreadSafeQueue<ann::WorkItem> insert, backlink, broadcast;
};

// The ids whose insert has been applied here, as the coordinator learns them
// (from the insert responses). The producer draws insert candidates and
// backlink targets from them.
struct Applied {
    std::mutex mutex;
    std::vector<uint32_t> ids;
    void add(uint32_t id) { std::lock_guard<std::mutex> lock(mutex); ids.push_back(id); }
    uint32_t pick(Lcg& rng, uint32_t fallback) {
        std::lock_guard<std::mutex> lock(mutex);
        return ids.empty() ? fallback : ids[rng.next() % ids.size()];
    }
    bool has(uint32_t id) {
        std::lock_guard<std::mutex> lock(mutex);
        return std::find(ids.begin(), ids.end(), id) != ids.end();
    }
};

void wait_for(const std::function<bool()>& ready, const char* what) {
    const auto deadline = Clock::now() + std::chrono::seconds(60);
    while (!ready()) {
        check(Clock::now() < deadline, what);
        std::this_thread::yield();
    }
}

// steady-clock time intervals [begin, end) in ns: a search request, a maintenance batch.
using Interval = std::pair<int64_t, int64_t>;
int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

// Per search thread, the searches whose interval intersects a batch interval.
std::vector<uint64_t> searches_overlapping(std::vector<Interval> batches,
                                           const std::vector<std::vector<Interval>>& searches) {
    std::sort(batches.begin(), batches.end());
    std::vector<Interval> merged;
    for (const auto& b : batches) {
        if (!merged.empty() && b.first <= merged.back().second) merged.back().second = std::max(merged.back().second, b.second);
        else merged.push_back(b);
    }
    std::vector<uint64_t> overlapped(searches.size(), 0);
    for (size_t index = 0; index < searches.size(); ++index) {
        for (const auto& q : searches[index]) {
            // The first merged batch ending after the search begins. It overlaps if it begins before the search ends.
            const auto it = std::lower_bound(merged.begin(), merged.end(), q.first,
                                             [](const Interval& m, int64_t t) { return m.second <= t; });
            if (it != merged.end() && it->first < q.second) ++overlapped[index];
        }
    }
    return overlapped;
}

// An owned load-time id still owned here.
uint32_t owned_load_id(const World& world, Lcg& rng) {
    for (;;) {
        const uint32_t id = rng.next() % kNodes;
        if (world.owners[id] == kRank) return id;
    }
}

// Returns how many inserts named an id this server had not received (each held-back id once).
uint32_t produce(ServerData& data, const World& world, Queues& queues, Applied& applied) {
    Lcg rng{0xc0ffeeull};
    std::vector<uint32_t> sent_remote;  // other servers' ids whose broadcast and backlink are queued (or applied)
    std::vector<uint32_t> held_back;    // other servers' ids not sent yet: the next own insert names one
    uint32_t named_undelivered = 0;
    // Another server's insert: its vector arrives here twice (a broadcast and a backlink).
    auto send_remote = [&](uint32_t id) {
        const auto& vec = world.vectors[id];
        std::vector<uint32_t> targets;
        uint32_t tries = 0;
        while (targets.size() < kBacklinkTargets) {
            const bool from_applied = targets.size() % 2 && ++tries < 16;
            const uint32_t t = from_applied ? applied.pick(rng, owned_load_id(world, rng)) : owned_load_id(world, rng);
            if (std::find(targets.begin(), targets.end(), t) == targets.end()) targets.push_back(t);
        }
        queues.broadcast.push(work_item(broadcast_request(id, vec, world.owners[id]), ann::MessageType::VECTOR_BROADCAST));
        queues.backlink.push(work_item(ann::serialize_batch_backlink_request(id - kNodes, id, world.owners[id], kDim, kR,
                                                                             kAlpha, targets, vec.data()),
                                       ann::MessageType::BATCH_BACKLINK_REQUEST));
        sent_remote.push_back(id);
    };
    for (uint32_t i = 0; i < kNewIds; ++i) {
        const uint32_t id = kNodes + i;
        const auto& vec = world.vectors[id];
        if (world.owners[id] == kRank) {
            // An insert owned here: the coordinator's search result is the candidate list.
            std::vector<uint32_t> candidates;
            // Few applied or sent ids yet (a draw can repeat a candidate): fall back to load-time ids.
            uint32_t tries = 0, remote_tries = 0;
            while (candidates.size() < kCandidates) {
                const bool from_applied = candidates.size() % 4 == 3 && ++tries < 16;
                const bool from_remote = candidates.size() % 4 == 1 && !sent_remote.empty() && ++remote_tries < 16;
                const uint32_t c = from_applied ? applied.pick(rng, rng.next() % kNodes)
                                 : from_remote ? sent_remote[rng.next() % sent_remote.size()] : rng.next() % kNodes;
                if (std::find(candidates.begin(), candidates.end(), c) == candidates.end()) candidates.push_back(c);
            }
            uint32_t undelivered = UINT32_MAX;
            if (!held_back.empty()) {
                undelivered = held_back.back();
                held_back.pop_back();
                candidates.push_back(undelivered);
            }
            std::vector<float> distances;
            for (uint32_t c : candidates) distances.push_back(std::sqrt(l2(vec, world.vectors[c])));
            queues.insert.push(work_item(ann::serialize_insert_request(i, vec.data(), kDim, candidates, distances,
                                                                       kR, kAlpha, kRank, id),
                                         ann::MessageType::INSERT_REQUEST));
            if (undelivered != UINT32_MAX) {
                // The insert named an id this server has not received: once it is applied, its row
                // does not name that id (dropped, not linked). Only then is the id delivered.
                wait_for([&] { return applied.has(id); }, "the insert naming an undelivered id was applied");
                const uint32_t slot = g2l_checked(data, id);
                check(slot != UINT32_MAX, "the insert has its row");
                {
                    std::lock_guard<std::mutex> lock(data.node_locks[slot].mtx);
                    const auto& row = data.adj.at(slot);
                    check(std::find(row.begin(), row.end(), undelivered) == row.end(),
                          "an insert never links an id this server has not received");
                }
                ++named_undelivered;
                send_remote(undelivered);
            }
        } else if ((i / 3) % 16 == 0) {
            held_back.push_back(id);
        } else {
            send_remote(id);
        }
    }
    for (const uint32_t id : held_back) send_remote(id);
    return named_undelivered;
}

// One worker's maintenance loop: insert, backlink and broadcast batches in turn
// (the worker's rotation), until the producer is done and the queues are empty.
void maintain(ServerData& data, Queues& queues, Applied& applied, std::atomic<bool>& produced,
              std::vector<Interval>& batches) {
    ann::TimingMetrics metrics;
    for (uint32_t kind = 0;; kind = (kind + 1) % 3) {
        bool any = false;
        for (uint32_t offset = 0; offset < 3 && !any; ++offset) {
            const uint32_t k = (kind + offset) % 3;
            auto& queue = k == 0 ? queues.insert : k == 1 ? queues.backlink : queues.broadcast;
            auto batch = queue.try_pop_n(kBatch);
            if (batch.empty()) continue;
            any = true;
            const int64_t begin = now_ns();
            if (k == 0) {
                const auto responses = batch_process_inserts(data, batch, metrics);
                check(responses.size() == batch.size(), "an insert response per request");
                for (const auto& response : responses) {
                    // Every neighbor the insert linked was delivered here, and every edge it reports
                    // to the coordinator carries an owner label below K.
                    ann::InsertResponseHeader header{};
                    std::vector<uint32_t> neighbors;
                    std::vector<CrossPartitionEdge> edges;
                    check(ann::deserialize_insert_response(response.response.data(), response.response.size(),
                                                           header, neighbors, edges), "an insert response");
                    for (const uint32_t n : neighbors) check(delivered_here(data, n), "a linked neighbor was delivered");
                    for (const auto& edge : edges) check(edge.from_partition < kServers, "an edge's owner label is below K");
                }
                for (const auto& item : batch) {
                    ann::InsertHeader header; std::vector<float> v; std::vector<uint32_t> c; std::vector<float> d;
                    check(ann::deserialize_insert_request(item.request.data(), item.request.size(), header, v, c, d),
                          "own insert request");
                    applied.add(header.assigned_global_id);
                }
            } else if (k == 1) {
                check(batch_process_backlinks(data, batch, metrics).size() == batch.size(), "a backlink ack each");
            } else {
                check(batch_process_broadcasts(data, batch, metrics).size() == batch.size(), "a broadcast ack each");
            }
            batches.emplace_back(begin, now_ns());
        }
        if (!any) {
            if (produced.load(std::memory_order_acquire) && queues.insert.size() == 0 &&
                queues.backlink.size() == 0 && queues.broadcast.size() == 0) return;
            std::this_thread::yield();
        }
    }
}

struct SearchTotals {
    std::atomic<uint64_t> requests{0};
};

// One search request as a worker runs it, the other server's hops played by
// marking the entries it would expand: process_state until DONE.
uint64_t search_once(ServerData& data, const World& world, Lcg& rng, std::vector<bool>& seen,
                     std::vector<uint32_t>& seen_nodes) {
    const auto query = random_vector(rng);
    std::vector<ann::BeamEntry> beam{{kStart, l2(query, world.vectors[kStart]), 0, {0, 0, 0}}};
    uint64_t work = 0;
    ann::StateStatus status = ann::StateStatus::ERROR;
    std::vector<uint32_t> ids;
    std::vector<float> distances;
    for (uint32_t hop = 0; hop < 256; ++hop) {
        uint32_t next = UINT32_MAX;
        ProcessMetrics metrics;
        process_state(data, query.data(), kDim, kL, kK, kW, beam, status, next, ids, distances, metrics, seen,
                      seen_nodes, 0, hop);
        work += metrics.distance_computations;
        if (status == ann::StateStatus::DONE) break;
        check(status == ann::StateStatus::NEED_HOP && next == 1, "a search ends DONE or hops to the other server");
        // The other server expands the top W entries that are not expanded (its own).
        uint32_t marked = 0;
        for (auto& entry : beam) {
            if (entry.expanded) continue;
            entry.expanded = 1;
            ann::set_beam_entry_exact(entry, true);
            if (++marked == kW) break;
        }
    }
    check(status == ann::StateStatus::DONE, "every search finishes");
    const uint32_t n_base = n_base_load(data);
    for (size_t i = 0; i < ids.size(); ++i) {
        check(ids[i] < n_base, "a result id is known here");
        check(i == 0 || distances[i - 1] <= distances[i], "results in distance order");
    }
    for (uint32_t v : seen_nodes) if (v < seen.size()) seen[v] = false;
    seen_nodes.clear();
    return work;
}

void search_loop(ServerData& data, const World& world, uint32_t index, std::atomic<bool>& stop,
                 std::atomic<uint32_t>& started, SearchTotals& totals, std::vector<Interval>& intervals) {
    Lcg rng{0x5ea7c4ull + index};
    std::vector<bool> seen(n_base_load(data), false);
    std::vector<uint32_t> seen_nodes;
    started.fetch_add(1);
    uint32_t done = 0;
    while (!stop.load(std::memory_order_acquire) || done < 2) {
        const int64_t begin = now_ns();
        search_once(data, world, rng, seen, seen_nodes);
        intervals.emplace_back(begin, now_ns());
        ++done;
        totals.requests.fetch_add(1);
    }
}

void check_end_state(ServerData& data, const World& world, bool own_only, uint32_t n_local_before,
                     const std::set<uint32_t>& freed) {
    uint32_t local = 0, remote = 0;
    std::set<uint32_t> slots;
    const uint32_t n_local = n_local_load(data);
    for (uint32_t id = kNodes; id < kTotalIds; ++id) {
        const auto& vec = world.vectors[id];
        std::vector<uint8_t> code(kM);
        if (own_only) data.pq_table.encode(vec.data(), code.data());
        check(mem_at(data, id) == world.owners[id], "every inserted id carries its owner");
        if (own_only)
            check(pq_code(data, id) && std::equal(code.begin(), code.end(), pq_code(data, id)),
                  "every inserted id's code, written once");
        if (world.owners[id] == kRank) {
            ++local;
            const uint32_t slot = g2l_checked(data, id);
            check(slot < n_local && data.local_to_global.at(slot) == id, "an own insert maps to its slot");
            check(slots.insert(slot).second, "no two inserts share a slot");
            const float* stored = own_only ? own_vec(data, id) : server_vec(data, id);
            check(stored && std::equal(vec.begin(), vec.end(), stored), "an own insert's vector, unchanged");
            std::lock_guard<std::mutex> lock(data.node_locks[slot].mtx);
            check(!data.adj.at(slot).empty(), "an own insert has its row");
        } else {
            ++remote;
            check(g2l_checked(data, id) == UINT32_MAX, "another server's id has no row here");
            if (!own_only) {
                const float* stored = server_vec(data, id);
                check(stored && std::equal(vec.begin(), vec.end(), stored), "a replicated vector, unchanged");
            }
        }
    }
    check(local == kNewIds - kNewIds / 3 && remote == kNewIds / 3, "the id split");
    check(n_base_load(data) == kTotalIds, "n_base covers every id");
    check(n_local == n_local_before + local - kMovedOut, "inserts reused every freed slot before appending");
    check(data.free_slot_pool.empty(), "the pool is empty");
    for (uint32_t slot : freed) check(slots.count(slot) == 1, "every freed slot went to an insert");
    for (uint32_t slot = 0; slot < n_local; ++slot) {
        std::lock_guard<std::mutex> lock(data.node_locks[slot].mtx);
        for (uint32_t v : data.adj.at(slot)) check(v < kTotalIds, "every edge names a known id");
    }
    const auto counts = membership_check(data, world.owners.data(), kTotalIds);
    const bool sound = counts.pool_ok && counts.free_slots == 0 && counts.n_local == n_local &&
        counts.membership_mismatches == 0 && counts.mapping_mismatches == 0 && counts.vector_mismatches == 0 &&
        counts.missing_codes == 0;
    if (!sound) {
        const uint32_t id = counts.first_mismatch;
        const uint32_t slot = id < kTotalIds ? g2l_checked(data, id) : UINT32_MAX;
        std::fprintf(stderr,
                     "membership-check: n_local %u (expected %u), free %u, pool_ok %d, mismatches: membership %llu, "
                     "mapping %llu, vector %llu, missing codes %llu; first id %u: owner %u, label %u, slot %u, "
                     "slot's id %u\n",
                     counts.n_local, n_local, counts.free_slots, counts.pool_ok ? 1 : 0,
                     (unsigned long long)counts.membership_mismatches, (unsigned long long)counts.mapping_mismatches,
                     (unsigned long long)counts.vector_mismatches, (unsigned long long)counts.missing_codes, id,
                     id < kTotalIds ? world.owners[id] : UINT32_MAX, id < kTotalIds ? mem_at(data, id) : UINT32_MAX,
                     slot, slot < n_local ? data.local_to_global.at(slot) : UINT32_MAX);
    }
    check(sound, "membership-check of every id is sound");
    const auto& g = g_graph_metrics;
    check(g.backlink_edges_requested.load() == (kNewIds / 3) * kBacklinkTargets &&
          g.backlink_edges_requested.load() == g.backlink_edges_applied.load() +
              g.backlink_skip_already_exists.load() + g.backlink_skip_not_local.load() +
              g.backlink_skip_out_of_range.load() &&
          g.backlink_skip_not_local.load() == 0 && g.backlink_skip_out_of_range.load() == 0,
          "every backlink edge requested landed or already existed");
}

WindowSnapshotRequest snapshot_request(uint64_t epoch, uint32_t nodes) {
    WindowSnapshotRequest request;
    request.epoch = epoch; request.nodes = nodes;
    request.sections = static_cast<uint16_t>(kWindowStructureOwnerHist | kWindowStructureNbWork);
    return request;
}

void run(bool own_only) {
    World world = make_world();
    auto owned = std::make_unique<ServerData>();
    ServerData& data = *owned;
    make_server(data, world, own_only);
    const auto freed = free_rows_by_accept(data, world);
    const uint32_t n_local_before = n_local_load(data);
    g_graph_metrics.reset();

    std::atomic<bool> stop{false}, produced{false};
    std::atomic<uint32_t> started{0};
    SearchTotals totals;
    std::vector<std::vector<Interval>> search_intervals(kSearchThreads);
    std::vector<std::thread> searches;
    for (uint32_t t = 0; t < kSearchThreads; ++t)
        searches.emplace_back(search_loop, std::ref(data), std::cref(world), t, std::ref(stop), std::ref(started),
                              std::ref(totals), std::ref(search_intervals[t]));
    wait_for([&] { return started.load() == kSearchThreads; }, "search threads started");

    // Part 1: the insert pipeline while searches run.
    Queues queues;
    Applied applied;
    std::vector<std::vector<Interval>> batch_intervals(kMaintenanceThreads);
    std::vector<std::thread> maintenance;
    for (uint32_t t = 0; t < kMaintenanceThreads; ++t)
        maintenance.emplace_back(maintain, std::ref(data), std::ref(queues), std::ref(applied), std::ref(produced),
                                 std::ref(batch_intervals[t]));
    const uint32_t named_undelivered = produce(data, world, queues, applied);
    produced.store(true, std::memory_order_release);
    for (auto& thread : maintenance) thread.join();
    check_end_state(data, world, own_only, n_local_before, freed);
    check(named_undelivered > 0 && g_graph_metrics.insert_neighbors_undelivered.load() >= named_undelivered,
          "candidates not yet on this server dropped and counted");

    // Part 2, with inserts kept out: window controls on this (the I/O) thread while searches count.
    const uint32_t nodes = n_base_load(data);
    ann::StatsWindowAck counters{}; counters.server_rank = kRank;
    for (uint64_t epoch = 1; epoch <= 6; ++epoch) {
        check(stats_window_begin_local(data, stats_window_begin_request(epoch, nodes)), "BEGIN while searching");
        const uint64_t before = totals.requests.load();
        wait_for([&] { return totals.requests.load() >= before + 6; }, "searches during the window");
        if (epoch % 3 == 0) {
            WindowStopRequest stop_request; stop_request.epoch = epoch; stop_request.ranks = kServers;
            check(!stats_window_stop_local(data, stop_request, kRank).status, "STOP while searching");
        } else {
            auto captured = stats_window_capture_snapshot(data, snapshot_request(epoch, nodes), counters);
            check(!captured.ack.status && captured.nodes == nodes, "SNAPSHOT cut while searching");
            StatsWindowSnapshotTask task;
            check(task.start("window", std::move(captured), -1), "encoder admitted");
            StatsWindowSnapshotReply reply;
            check(task.finish(reply, true) && reply.response.size() > sizeof(WindowSnapshotHeader),
                  "the encoded frame");
        }
    }
    stop.store(true, std::memory_order_release);
    for (auto& thread : searches) thread.join();
    std::vector<Interval> batches;
    for (const auto& thread : batch_intervals) batches.insert(batches.end(), thread.begin(), thread.end());
    const std::vector<uint64_t> per_thread = searches_overlapping(batches, search_intervals);
    check(per_thread.size() == kSearchThreads &&
              std::all_of(per_thread.begin(), per_thread.end(), [](uint64_t c) { return c >= 1; }),
          "every search thread searched during a maintenance batch");
    uint64_t overlapped = 0;
    std::string per_thread_text;
    for (const uint64_t c : per_thread) {
        overlapped += c;
        per_thread_text += (per_thread_text.empty() ? "" : "/") + std::to_string(c);
    }

    // Quiescent: one window counts exactly the distance evaluations of its searches.
    check(stats_window_begin_local(data, stats_window_begin_request(7, nodes)), "quiescent BEGIN");
    Lcg rng{0xa11ce5ull};
    std::vector<bool> seen(nodes, false);
    std::vector<uint32_t> seen_nodes;
    uint64_t work = 0;
    for (uint32_t i = 0; i < 32; ++i) work += search_once(data, world, rng, seen, seen_nodes);
    const auto captured = stats_window_capture_snapshot(data, snapshot_request(7, nodes), counters);
    check(!captured.ack.status && captured.ack.total_w == work && work > 0,
          "the window counts each evaluation of its searches once");
    std::printf("server_concurrency %s OK (%llu searches, %llu of them overlapping one of %zu maintenance batches, "
                "per search thread %s; %u inserts named an id not yet received, %llu candidates dropped)\n",
                own_only ? "own-only" : "full-replication", (unsigned long long)totals.requests.load(),
                (unsigned long long)overlapped, batches.size(), per_thread_text.c_str(), named_undelivered,
                (unsigned long long)g_graph_metrics.insert_neighbors_undelivered.load());
}

// Full replication: several workers deliver the same inserted ids' vectors at
// once (store_replicated_vector, as a backlink and a broadcast of one id do):
// exactly one delivery writes each vector, and every delivery finds it
// complete when the call returns.
void replicated_vector_written_once() {
    const World world = make_world();
    auto owned = std::make_unique<ServerData>();
    ServerData& data = *owned;
    make_server(data, world, false);
    constexpr uint32_t kDeliveries = 4;
    std::atomic<uint32_t> writes{0}, incomplete{0};
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < kDeliveries; ++t) {
        threads.emplace_back([&, t] {
            for (uint32_t i = 0; i < kNewIds; ++i) {
                const uint32_t id = kNodes + (i * 7 + t * 13) % kNewIds;
                const auto& vec = world.vectors[id];
                if (store_replicated_vector(data, id, vec.data())) writes.fetch_add(1);
                const float* stored = server_vec(data, id);
                if (!std::equal(vec.begin(), vec.end(), stored)) incomplete.fetch_add(1);
            }
        });
    }
    for (auto& thread : threads) thread.join();
    check(writes.load() == kNewIds, "exactly one delivery writes each vector");
    check(incomplete.load() == 0, "every delivery returns with the vector complete");
    std::printf("replicated_vector_written_once OK\n");
}

// Each part runs in its own child process, so a failure (or a race report) in
// one does not hide the others' outcomes.
bool isolated(const char* name, const std::function<void()>& body) {
    std::fflush(stdout);
    const pid_t child = fork();
    check(child >= 0, "fork");
    if (child == 0) {
        body();
        std::fflush(stdout);
        _exit(0);
    }
    int status = 0;
    check(waitpid(child, &status, 0) == child, "wait");
    const bool ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    if (!ok) std::fprintf(stderr, "server_concurrency %s failed\n", name);
    return ok;
}

}  // namespace

int main() {
    const bool once = isolated("replicated-vector-once", replicated_vector_written_once);
    const bool own_only = isolated("own-only", [] { run(true); });
    const bool full = isolated("full-replication", [] { run(false); });
    if (!once || !own_only || !full) return 1;
    std::printf("ALL SERVER CONCURRENCY TESTS PASSED\n");
    return 0;
}
