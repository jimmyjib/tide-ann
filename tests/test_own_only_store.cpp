// PQ mode stores: distances, code and vector stores, search re-rank, loader, RESET and inserts.
#include "server_types.h"
#include "membership_migration.h"
#include "npy_writer.h"
#include "protocol/broadcast_messages.h"
#include "protocol/insert_protocol.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#define CHECK(expr) do { if (!(expr)) { std::cerr << "CHECK failed at line " << __LINE__ << ": " #expr "\n"; return 1; } } while (0)

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

constexpr uint32_t kDim = 4, kM = 2, kNodes = 6, kServers = 2;
constexpr float kMax = std::numeric_limits<float>::max();
// Owners of the six load-time ids (rank 0 owns 0, 1 and 4).
const std::vector<uint32_t> kMembership = {0, 0, 1, 1, 0, 1};
// Every coordinate is an integer, and so is every centroid coordinate below, so
// each squared distance is exact in float whatever the summation order.
const std::vector<std::vector<float>> kVectors = {
    {0, 0, 0, 0}, {1, 1, 2, 2}, {3, 0, 1, 4}, {2, 2, 2, 2}, {5, 5, 0, 0}, {1, 4, 4, 1}};
const std::vector<std::vector<uint32_t>> kRows = {{1, 2}, {4, 3}, {0}, {1}, {}, {4}};

// Pivots whose centroid k of each 2-dim chunk is (k % 16, k / 16): every vector
// above lies exactly on a centroid, so its code reconstructs it.
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

std::vector<uint8_t> encode_all(const ann::PQTable& pq) {
    std::vector<uint8_t> codes(kNodes * kM);
    for (uint32_t id = 0; id < kNodes; ++id) pq.encode(kVectors[id].data(), codes.data() + id * kM);
    return codes;
}

// Rank `rank` of the two-server layout, in PQ or full-vectors mode.
void make_server(ServerData& data, uint32_t rank, bool own_only) {
    data.own_only = own_only;
    data.dim = kDim; data.my_partition = rank; data.n_servers = kServers;
    data.base_static_capacity = kNodes;
    data.membership = kMembership;
    n_base_set(data, kNodes);
    data.global_to_local.ensure_capacity_for(kNodes - 1, UINT32_MAX);
    if (own_only) {
        data.pq_table = integer_pivots();
        data.pq_codes_static = encode_all(data.pq_table);
        data.pq_codes_delta.init(kM);
        data.full_own.init(kDim);
    } else {
        data.full_base_delta.init(kDim);
        for (const auto& v : kVectors) data.full_base.insert(data.full_base.end(), v.begin(), v.end());
    }
    uint32_t slot = 0;
    for (uint32_t id = 0; id < kNodes; ++id) {
        if (kMembership[id] != rank) continue;
        data.adj.ensure_capacity_for(slot);
        data.local_to_global.ensure_capacity_for(slot, UINT32_MAX);
        if (own_only) store_own_vector(data, slot, kVectors[id].data());
        data.adj.at(slot) = kRows[id];
        data.local_to_global.at(slot) = id;
        data.global_to_local.at(id) = slot;
        ++slot;
    }
    n_local_set(data, slot);
    ensure_node_locks(data, slot);
}

float exact(uint32_t a, uint32_t b) {
    return ann::l2_distance_squared_scalar(kVectors[a].data(), kVectors[b].data(), kDim);
}

int distance_cases() {
    ServerData data;
    make_server(data, 0, true);
    const auto& pq = data.pq_table;
    // Both held (0, 1): exact. One held (0 owned, 2 remote), either order: ADC of
    // the held vector against the other's code. Both remote (2, 3): code-to-code.
    CHECK(node_pair_distance(data, 0, 1) == compute_distance(kVectors[0].data(), kVectors[1].data(), kDim));
    CHECK(node_pair_distance(data, 0, 2) == pq.adc_distance_direct(kVectors[0].data(), pq_code(data, 2)));
    CHECK(node_pair_distance(data, 2, 0) == pq.adc_distance_direct(kVectors[0].data(), pq_code(data, 2)));
    CHECK(node_pair_distance(data, 2, 3) == pq.sym_distance(pq_code(data, 2), pq_code(data, 3)));
    // On centroids every estimate is exact.
    CHECK(node_pair_distance(data, 0, 2) == exact(0, 2) && node_pair_distance(data, 2, 3) == exact(2, 3));
    // A known vector (a backlink's source) replaces that node's code.
    const KnownVector known{2, kVectors[2].data()};
    CHECK(node_pair_distance(data, 0, 2, &known) == compute_distance(kVectors[0].data(), kVectors[2].data(), kDim));
    // An inserted id (>= N0) without a stored code: FLT_MAX in every case.
    const uint32_t inserted = kNodes + 1;
    server_ensure_mem_capacity(data, inserted);
    CHECK(pq_code(data, inserted) == nullptr);
    CHECK(node_pair_distance(data, 0, inserted) == kMax && node_pair_distance(data, inserted, 2) == kMax);
    store_pq_code(data, inserted, kVectors[3].data());
    CHECK(node_pair_distance(data, 0, inserted) == exact(0, 3));
    // Full replication: every pair exact, as the prune code always computed it.
    ServerData full;
    make_server(full, 0, false);
    CHECK(node_pair_distance(full, 2, 3) == compute_distance(kVectors[2].data(), kVectors[3].data(), kDim));
    CHECK(pq_code(full, 2) == nullptr);
    return 0;
}

int stores_and_ready_flag() {
    ServerData data;
    make_server(data, 0, true);
    const uint32_t id = kNodes + 3;
    server_ensure_mem_capacity(data, id);
    // Not READY before the first write, and READY after it with the encoded bytes.
    CHECK(!data.pq_codes_delta.ready(id - kNodes) && pq_code(data, id) == nullptr);
    store_pq_code(data, id, kVectors[5].data());
    CHECK(data.pq_codes_delta.ready(id - kNodes));
    std::vector<uint8_t> expected(kM);
    data.pq_table.encode(kVectors[5].data(), expected.data());
    CHECK(std::equal(expected.begin(), expected.end(), pq_code(data, id)));
    // Idempotent: a second delivery (another vector even) never rewrites it.
    store_pq_code(data, id, kVectors[4].data());
    CHECK(std::equal(expected.begin(), expected.end(), pq_code(data, id)));
    CHECK(!data.pq_codes_delta.write_once(id - kNodes, [](uint8_t* out) { out[0] = 99; }));
    CHECK(std::equal(expected.begin(), expected.end(), pq_code(data, id)));
    // A load-time id's code is static, so storing one is a no-op.
    const std::vector<uint8_t> before(data.pq_codes_static);
    store_pq_code(data, 2, kVectors[0].data());
    CHECK(data.pq_codes_static == before && data.pq_codes_delta.allocated_chunks() == 1);
    // An own vector is reachable only once the mapping is published.
    data.adj.ensure_capacity_for(3);
    store_own_vector(data, 3, kVectors[5].data());
    CHECK(own_vec(data, id) == nullptr);
    data.global_to_local.ensure_capacity_for(id, UINT32_MAX);
    data.global_to_local.at(id) = 3;
    CHECK(own_vec(data, id) != nullptr && std::equal(kVectors[5].begin(), kVectors[5].end(), own_vec(data, id)));
    // The checked lookup of an id whose mapping chunk was never allocated here.
    CHECK(g2l_checked(data, 250000) == UINT32_MAX && own_vec(data, 250000) == nullptr);
    return 0;
}

int hybrid_scorer() {
    ServerData data;
    make_server(data, 0, true);
    const float query[kDim] = {1, 2, 3, 4};
    std::vector<float> buffer, distances;
    std::vector<uint8_t> flags;
    const uint32_t missing = kNodes + 2;
    server_ensure_mem_capacity(data, missing);
    {
        // Held vectors only: exact, and the ADC table is never built.
        LazyAdcTable adc(data.pq_table, query, buffer);
        const auto counts = hybrid_score(data, query, {1, 4}, adc, distances, flags);
        CHECK(counts.exact == 2 && counts.adc == 0 && counts.missing == 0 && !adc.built());
        CHECK(flags == (std::vector<uint8_t>{1, 1}));
        CHECK(distances[0] == compute_distance(query, kVectors[1].data(), kDim));
    }
    LazyAdcTable adc(data.pq_table, query, buffer);
    const auto counts = hybrid_score(data, query, {1, 2, missing, 4}, adc, distances, flags);
    CHECK(counts.exact == 2 && counts.adc == 1 && counts.missing == 1 && adc.built());
    CHECK(flags == (std::vector<uint8_t>{kScoreExact, kScoreAdc, kScoreMissing, kScoreExact}));
    CHECK(distances[1] == data.pq_table.adc_distance(adc.get(), pq_code(data, 2)));
    CHECK(distances[2] == kMax);
    // Full replication: exact everywhere. An id beyond the stored capacity is FLT_MAX.
    ServerData full;
    make_server(full, 0, false);
    LazyAdcTable none(full.pq_table, query, buffer);
    const auto all = hybrid_score(full, query, {2, 3, 999999}, none, distances, flags);
    CHECK(all.exact == 2 && all.adc == 0 && all.missing == 1 && !none.built());
    CHECK(distances[0] == compute_distance(query, kVectors[2].data(), kDim) && distances[2] == kMax);
    return 0;
}

struct Hop {
    ann::StateStatus status = ann::StateStatus::ERROR;
    uint32_t next = 0;
    std::vector<uint32_t> ids;
    std::vector<float> distances;
    ProcessMetrics metrics;
};

Hop hop(ServerData& data, const float* query, std::vector<ann::BeamEntry>& beam, std::vector<uint32_t>& seen_nodes) {
    Hop out;
    std::vector<bool> seen(16, false);
    for (uint32_t id : seen_nodes) seen[id] = true;
    process_state(data, query, kDim, /*L=*/8, /*K=*/4, /*W=*/4, beam, out.status, out.next, out.ids,
                  out.distances, out.metrics, seen, seen_nodes, 0, 0);
    return out;
}

int search_rerank_and_done() {
    ServerData rank0, rank1;
    make_server(rank0, 0, true);
    make_server(rank1, 1, true);
    const float query[kDim] = {1, 1, 1, 1};
    // The coordinator's entry for node 0 carries no exact flag.
    std::vector<ann::BeamEntry> beam{{0, 123.0f, 0, {0, 0, 0}}};
    std::vector<uint32_t> seen_nodes;
    // Rank 0 expands 0 (re-ranked), 1 and 4, while 2 and 3 are rank 1's (NEED_HOP).
    auto first = hop(rank0, query, beam, seen_nodes);
    CHECK(first.status == ann::StateStatus::NEED_HOP && first.next == 1);
    CHECK(first.metrics.rerank_distances == 1);
    // Distance work = exact + ADC + re-rank (W(v) and the price count re-ranks).
    CHECK(first.metrics.exact_distances + first.metrics.adc_distances + first.metrics.rerank_distances ==
          first.metrics.distance_computations);
    CHECK(first.metrics.adc_distances >= 1);
    CHECK(first.metrics.adc_tables == 1);  // one ADC table for the hop
    for (const auto& entry : beam) {
        const bool held = kMembership[entry.node_id] == 0;
        // Every rank-0 node is exact. Rank 1's carry ADC estimates, unexpanded.
        CHECK(ann::beam_entry_exact(entry) == held);
        CHECK(held ? entry.expanded == 1 : entry.expanded == 0);
        if (held) CHECK(entry.distance == compute_distance(query, kVectors[entry.node_id].data(), kDim));
    }
    // Rank 1 re-ranks its entries on expansion. At DONE every distance is exact.
    auto second = hop(rank1, query, beam, seen_nodes);
    CHECK(second.status == ann::StateStatus::DONE);
    CHECK(second.metrics.rerank_distances >= 1);
    CHECK(second.metrics.exact_distances + second.metrics.adc_distances + second.metrics.rerank_distances ==
          second.metrics.distance_computations);
    CHECK(!second.ids.empty());
    for (size_t i = 0; i < second.ids.size(); ++i) {
        CHECK(second.distances[i] == std::sqrt(compute_distance(query, kVectors[second.ids[i]].data(), kDim)));
    }
    // An entry marked expanded without being expanded (membership unknown: id 9
    // is beyond membership) keeps a non-exact distance: dropped at DONE under
    // PQ mode, kept under full replication.
    std::vector<ann::BeamEntry> stale{{4, 0.0f, 0, {0, 0, 0}}, {9, 0.5f, 0, {0, 0, 0}}};
    std::vector<uint32_t> seen_stale{4, 9};
    auto dropped = hop(rank0, query, stale, seen_stale);
    CHECK(dropped.status == ann::StateStatus::DONE && dropped.ids == std::vector<uint32_t>{4});
    ServerData full;
    make_server(full, 0, false);
    std::vector<ann::BeamEntry> stale_full{{4, 0.0f, 0, {0, 0, 0}}, {9, 0.5f, 0, {0, 0, 0}}};
    std::vector<uint32_t> seen_full{4, 9};
    auto kept = hop(full, query, stale_full, seen_full);
    CHECK(kept.status == ann::StateStatus::DONE && kept.ids.size() == 2);
    for (const auto& entry : stale_full) CHECK(entry.padding[0] == 0);  // full replication never sets the flag
    return 0;
}

int prune_matches_full_replication() {
    // On centroid-exact vectors ADC and code-to-code equal the exact distances,
    // so RobustPrune keeps the same neighbors in both storage modes.
    ServerData own, full;
    make_server(own, 0, true);
    make_server(full, 0, false);
    for (float alpha : {1.0f, 1.2f, 2.0f}) {
        for (uint32_t p : {0u, 1u, 4u}) {
            std::vector<std::pair<uint32_t, float>> own_candidates, full_candidates;
            for (uint32_t id = 0; id < kNodes; ++id) {
                own_candidates.emplace_back(id, node_pair_distance(own, p, id));
                full_candidates.emplace_back(id, node_pair_distance(full, p, id));
            }
            CHECK(own_candidates == full_candidates);
            CHECK(robust_prune(own, p, own_candidates, 3, alpha) == robust_prune(full, p, full_candidates, 3, alpha));
        }
    }
    return 0;
}

int maintenance_paths() {
    ServerData data;
    make_server(data, 0, true);
    tide::g_graph_metrics.reset();
    // A backlink from inserted node 7 (owned by rank 1) to 0: rank 0 keeps only
    // the code of 7, and the edge lands.
    const uint32_t from = kNodes + 1;
    std::vector<WorkItem> batch(1);
    batch[0].request = ann::serialize_batch_backlink_request(1, from, 1, kDim, 64, 1.2f, {0}, kVectors[3].data());
    batch[0].msg_type = ann::MessageType::BATCH_BACKLINK_REQUEST;
    TimingMetrics metrics;
    CHECK(batch_process_backlinks(data, batch, metrics).size() == 1);
    CHECK(pq_code(data, from) != nullptr && own_vec(data, from) == nullptr && mem_at(data, from) == 1);
    CHECK(std::find(data.adj.at(0).begin(), data.adj.at(0).end(), from) != data.adj.at(0).end());
    // A broadcast of the same node again: the code stays as it was.
    std::vector<uint8_t> code(pq_code(data, from), pq_code(data, from) + kM);
    std::vector<char> broadcast(sizeof(ann::VectorBroadcastHeader) + kDim * sizeof(float));
    ann::VectorBroadcastHeader header{};
    header.msg_type = ann::MessageType::VECTOR_BROADCAST;
    header.node_id = from; header.dim = kDim; header.home_partition = 1;
    std::memcpy(broadcast.data(), &header, sizeof(header));
    std::memcpy(broadcast.data() + sizeof(header), kVectors[0].data(), kDim * sizeof(float));
    std::vector<WorkItem> broadcasts(1);
    broadcasts[0].request = broadcast;
    batch_process_broadcasts(data, broadcasts, metrics);
    CHECK(std::equal(code.begin(), code.end(), pq_code(data, from)));
    // Messages of another dimension are rejected unapplied (never read past the
    // vector): a broadcast and a backlink of dimension 2 for inserted id 9.
    const uint32_t other = kNodes + 3;
    header.node_id = other; header.dim = 2;
    std::memcpy(broadcast.data(), &header, sizeof(header));
    broadcasts[0].request = broadcast;
    batch_process_broadcasts(data, broadcasts, metrics);
    server_ensure_mem_capacity(data, other);
    CHECK(pq_code(data, other) == nullptr && mem_at(data, other) == UINT32_MAX);
    const float short_vector[2] = {1, 1};
    batch[0].request = ann::serialize_batch_backlink_request(2, other, 1, 2, 64, 1.2f, {0}, short_vector);
    const uint64_t requested = g_graph_metrics.backlink_edges_requested.load();
    CHECK(batch_process_backlinks(data, batch, metrics).size() == 1);
    CHECK(pq_code(data, other) == nullptr && mem_at(data, other) == UINT32_MAX &&
          g_graph_metrics.backlink_edges_requested.load() == requested);
    // An insert of dimension 2 is answered as invalid and allocates nothing.
    const uint32_t n_local_before = n_local_load(data);
    std::vector<WorkItem> inserts(1);
    inserts[0].request = ann::serialize_insert_request(1, short_vector, 2, {0}, {1.0f}, 4, 1.2f, 0, other);
    CHECK(batch_process_inserts(data, inserts, metrics).size() == 1);
    CHECK(n_local_load(data) == n_local_before && own_vec(data, other) == nullptr);
    return 0;
}

// An insert reuses a row slot an ACCEPT freed. Rank 0
// moves node 1 (slot 1) to rank 1 through the server's PREPARE, COMMIT and
// ACCEPT, and slot 1 enters the pool with generation 1. The next insert (id 6,
// owned here) takes slot 1 (generation 2) instead of appending. Its vector
// (in PQ mode the own vector and code, with full replication the replicated vector),
// mapping and row are in place, the pool is empty and membership-check finds
// every mapping, vector and the pool sound. The insert after it appends slot 3.
int insert_reuses_pooled_slot() {
    for (const bool own_only : {true, false}) {
        ServerData data;
        make_server(data, 0, own_only);
        MembershipMigrationPlan plan;
        plan.n_servers = kServers; plan.epoch_from = 0; plan.epoch_to = 1;
        MembershipMove move;
        move.global_id = 1; move.from_partition = 0; move.to_partition = 1;
        move.adjacency = kRows[1]; move.adjacency_digest = migration_row_digest(move.adjacency);
        plan.moves.push_back(move);
        MembershipMigrationJournal journal;
        std::vector<uint64_t> accepted;
        std::string error;
        CHECK(prepare_membership_migration(data, plan.moves, 0, 1, journal, error));
        CHECK(commit_membership_migration(data, plan.moves, journal, error));
        CHECK(accept_membership_migration_history(data, plan, journal, accepted, error));
        CHECK(data.free_slot_pool == std::vector<uint32_t>({1}) && migration_slot_generation(data, 1) == 1 &&
              n_local_load(data) == 3 && g2l_checked(data, 1) == UINT32_MAX && data.adj.at(1).capacity() == 0);
        TimingMetrics metrics;
        const float first[kDim] = {2, 3, 4, 5}, second[kDim] = {4, 4, 1, 1};
        std::vector<WorkItem> inserts(1);
        inserts[0].request = ann::serialize_insert_request(1, first, kDim, {0, 4}, {1.0f, 2.0f}, 4, 1.2f, 0, kNodes);
        CHECK(batch_process_inserts(data, inserts, metrics).size() == 1);
        CHECK(n_local_load(data) == 3 && data.free_slot_pool.empty() && g2l_checked(data, kNodes) == 1 &&
              data.local_to_global.at(1) == kNodes && migration_slot_generation(data, 1) == 2 &&
              mem_at(data, kNodes) == 0 && !data.adj.at(1).empty());
        const float* stored = server_vec(data, kNodes);
        CHECK(stored != nullptr && std::equal(first, first + kDim, stored));
        if (own_only) CHECK(pq_code(data, kNodes) != nullptr);
        inserts[0].request = ann::serialize_insert_request(2, second, kDim, {0, kNodes}, {1.0f, 2.0f}, 4, 1.2f, 0,
                                                           kNodes + 1);
        CHECK(batch_process_inserts(data, inserts, metrics).size() == 1);
        CHECK(n_local_load(data) == 4 && g2l_checked(data, kNodes + 1) == 3 && data.local_to_global.at(3) == kNodes + 1);
        std::vector<uint32_t> owners = kMembership;
        owners[1] = 1; owners.push_back(0); owners.push_back(0);
        const auto counts = membership_check(data, owners.data(), kNodes + 2);
        CHECK(counts.pool_ok && counts.free_slots == 0 && counts.n_local == 4 && counts.membership_mismatches == 0 &&
              counts.mapping_mismatches == 0 && counts.vector_mismatches == 0 && counts.missing_codes == 0);
    }
    return 0;
}

// RESET (reset_server_data, the server's RESET handler) in PQ mode:
// after an insert (own vector + code + mapping), a backlink (a delta code and an
// edge), a freed row slot and an overwritten own vector, every store is reloaded
// exactly as a fresh start: the own vectors by slot, the static codes, no delta
// code, membership N0, the rows and mappings, the free pool and the statistics
// table, and own_only stays on.
int own_only_reset() {
    char pattern[] = "/tmp/own-only-reset-XXXXXX";
    const char* made = mkdtemp(pattern);
    CHECK(made != nullptr);
    const std::string root = made, partition = root + "/partition_0";
    std::filesystem::create_directories(partition);
    const auto pq = integer_pivots();
    ann::write_npy_uint32(root + "/membership.npy", kMembership);
    pq.save(root + "/pq_pivots.bin");
    ann::write_pq_codes(root + "/pq_codes.bin", encode_all(pq), kNodes, kM, pq.pivots_hash());
    const std::vector<uint32_t> l2g = {0, 1, 4};
    ann::write_npy_uint32(partition + "/local_to_global.npy", l2g);
    ann::write_diskann_adj(partition + "/adj", {kRows[0], kRows[1], kRows[4]}, 0);
    std::vector<float> rows;
    for (uint32_t id : l2g) rows.insert(rows.end(), kVectors[id].begin(), kVectors[id].end());
    ann::write_npy_float(partition + "/vectors.npy", rows, 3, kDim);
    auto start = [&](ServerData& data) {
        data.own_only = true; data.my_partition = 0; data.n_servers = kServers;
        return load_server_data(data, root, 0);
    };
    ServerData fresh, data;
    const uint32_t fresh_start = start(fresh);
    start(data);
    TimingMetrics metrics;
    // An insert of id 6 (owned here), a backlink from inserted id 7 (rank 1's) to 0.
    const float inserted[kDim] = {2, 3, 4, 5};
    std::vector<WorkItem> inserts(1);
    inserts[0].request = ann::serialize_insert_request(1, inserted, kDim, {0, 1}, {1.0f, 2.0f}, 4, 1.2f, 0, kNodes);
    CHECK(batch_process_inserts(data, inserts, metrics).size() == 1);
    CHECK(n_local_load(data) == 4 && own_vec(data, kNodes) != nullptr && pq_code(data, kNodes) != nullptr);
    std::vector<WorkItem> backlinks(1);
    backlinks[0].request = ann::serialize_batch_backlink_request(1, kNodes + 1, 1, kDim, 64, 1.2f, {0},
                                                                 kVectors[3].data());
    CHECK(batch_process_backlinks(data, backlinks, metrics).size() == 1);
    CHECK(pq_code(data, kNodes + 1) != nullptr && mem_at(data, kNodes + 1) == 1);
    data.free_slot_pool.push_back(3);
    std::fill(data.full_own.vec(2), data.full_own.vec(2) + kDim, 9.0f);  // node 4's own vector
    data.stats_window_w[0].store(5);
    CHECK(reset_server_data(data, root, 0) == fresh_start);
    CHECK(data.own_only && data.dim == kDim && data.pq_table.M == kM);
    CHECK(n_base_load(data) == kNodes && n_local_load(data) == 3 && data.base_static_capacity == kNodes);
    CHECK(mem_size(data) == kNodes && data.membership == kMembership);
    CHECK(data.pq_codes_static == fresh.pq_codes_static && data.pq_codes_delta.allocated_chunks() == 0);
    CHECK(pq_code(data, kNodes) == nullptr && pq_code(data, kNodes + 1) == nullptr);
    CHECK(data.free_slot_pool.empty() && data.full_own.total_node_capacity() ==
          fresh.full_own.total_node_capacity());
    for (uint32_t id = 0; id < kNodes; ++id) {
        CHECK(g2l_checked(data, id) == g2l_checked(fresh, id));
        const float* a = own_vec(data, id);
        const float* b = own_vec(fresh, id);
        CHECK((a == nullptr) == (b == nullptr));
        if (a) CHECK(std::equal(a, a + kDim, kVectors[id].data()));
    }
    CHECK(g2l_checked(data, kNodes) == UINT32_MAX && own_vec(data, kNodes) == nullptr);
    for (uint32_t slot = 0; slot < 3; ++slot)
        CHECK(data.adj.at(slot) == fresh.adj.at(slot) && data.local_to_global.at(slot) == l2g[slot]);
    CHECK(data.stats_window_nodes == kNodes && data.stats_window_w[0].load() == 0);
    std::filesystem::remove_all(root);
    return 0;
}

// A raw NPY file (version 1.0) of the given dtype and shape holding `data`.
void write_raw_npy(const std::string& path, const std::string& descr, const std::string& shape,
                   const void* data, size_t bytes) {
    std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
    while ((10 + header.size() + 1) % 64 != 0) header += ' ';
    header += '\n';
    std::ofstream out(path, std::ios::binary);
    out.write("\x93NUMPY\x01\x00", 8);
    const uint16_t length = static_cast<uint16_t>(header.size());
    out.write(reinterpret_cast<const char*>(&length), 2);
    out << header;
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}

int own_only_loader() {
    char pattern[] = "/tmp/own-only-load-XXXXXX";
    const char* made = mkdtemp(pattern);
    CHECK(made != nullptr);
    const std::string root = made, partition = root + "/partition_0";
    std::filesystem::create_directories(partition);
    const auto pq = integer_pivots();
    ann::write_npy_uint32(root + "/membership.npy", kMembership);
    pq.save(root + "/pq_pivots.bin");
    ann::write_pq_codes(root + "/pq_codes.bin", encode_all(pq), kNodes, kM, pq.pivots_hash());
    const std::vector<uint32_t> l2g = {0, 1, 4};
    ann::write_npy_uint32(partition + "/local_to_global.npy", l2g);
    ann::write_diskann_adj(partition + "/adj", {kRows[0], kRows[1], kRows[4]}, 0);
    auto write_vectors = [&](const std::vector<uint32_t>& ids) {
        std::vector<float> rows;
        for (uint32_t id : ids) rows.insert(rows.end(), kVectors[id].begin(), kVectors[id].end());
        ann::write_npy_float(partition + "/vectors.npy", rows, static_cast<uint32_t>(ids.size()), kDim);
    };
    write_vectors(l2g);
    auto load = [&](ServerData& data) {
        data.own_only = true; data.my_partition = 0; data.n_servers = kServers;
        return load_server_data(data, root, 0);
    };
    {
        // base.npy is absent: PQ mode never opens it.
        ServerData data;
        load(data);
        CHECK(n_base_load(data) == kNodes && data.dim == kDim && data.base_static_capacity == kNodes);
        CHECK(data.full_base.empty() && data.pq_table.M == kM && data.pq_codes_static == encode_all(pq));
        CHECK(n_local_load(data) == 3 && own_vec(data, 4) != nullptr && own_vec(data, 2) == nullptr);
        CHECK(std::equal(kVectors[4].begin(), kVectors[4].end(), own_vec(data, 4)));
        CHECK(data.stats_window_nodes == kNodes);
    }
    auto refused = [&](const char* what) {
        ServerData data;
        try { load(data); } catch (const std::runtime_error& e) {
            if (std::string(e.what()).find(what) != std::string::npos) return true;
            std::cerr << "unexpected refusal: " << e.what() << "\n";
            return false;
        }
        return false;
    };
    // An int64 membership.npy or local_to_global.npy is rejected, never read as raw
    // uint32 words (labels interleaved with zeros that would pass the label check).
    {
        const std::vector<int64_t> wide(kMembership.begin(), kMembership.end());
        write_raw_npy(root + "/membership.npy", "<i8", "(6,)", wide.data(), wide.size() * sizeof(int64_t));
        CHECK(refused("is not int32 or uint32"));
        ann::write_npy_uint32(root + "/membership.npy", kMembership);
        const std::vector<int64_t> wide_l2g(l2g.begin(), l2g.end());
        write_raw_npy(partition + "/local_to_global.npy", "<i8", "(3,)", wide_l2g.data(),
                      wide_l2g.size() * sizeof(int64_t));
        CHECK(refused("is not int32 or uint32"));
        // A file shorter than its shape is rejected too.
        write_raw_npy(partition + "/local_to_global.npy", "<u4", "(3,)", l2g.data(), 2 * sizeof(uint32_t));
        CHECK(refused("shorter than its shape"));
        ann::write_npy_uint32(partition + "/local_to_global.npy", l2g);
        ServerData data;
        load(data);
        CHECK(n_local_load(data) == 3 && mem_at(data, 2) == 1);
    }
    // Rows in another order than local_to_global are rejected.
    write_vectors({1, 0, 4});
    CHECK(refused("not in local_to_global order"));
    write_vectors({0, 1});
    CHECK(refused("is not local_to_global.npy's"));
    write_vectors(l2g);
    // Codes of other pivots, then a missing codes file and missing pivots: no fallback.
    ann::write_pq_codes(root + "/pq_codes.bin", encode_all(pq), kNodes, kM, pq.pivots_hash() ^ 1);
    CHECK(refused("other pivots"));
    std::filesystem::remove(root + "/pq_codes.bin");
    CHECK(refused("pq_codes.bin"));
    std::filesystem::remove(root + "/pq_pivots.bin");
    CHECK(refused("missing or invalid PQ pivots"));
    std::filesystem::remove_all(root);
    return 0;
}

}  // namespace

// An insert whose candidates name an inserted id this server has not received yet
// (its broadcast and backlink still queued here): the candidate is dropped and
// counted, never read. No row and no cross-partition edge names it, and every
// edge's owner label is below K. Once delivered, the id is a candidate like any
// other. Both storage modes.
int undelivered_candidates_are_dropped() {
    for (const bool own_only : {true, false}) {
        ServerData data;
        make_server(data, 0, own_only);
        tide::g_graph_metrics.reset();
        TimingMetrics metrics;
        // Another server's inserted id: chunks exist (a later insert grew them), nothing delivered.
        const uint32_t queued = kNodes, labeled = kNodes + 1, own = kNodes + 2;
        server_ensure_mem_capacity(data, own);
        server_ensure_vec_capacity(data, own);
        CHECK(!delivered_here(data, queued) && delivered_here(data, 2) && delivered_here(data, 0));
        // A label alone is not a delivery: the vector or code is still missing.
        mem_set(data, labeled, 1);
        CHECK(!delivered_here(data, labeled));
        // Full replication: the zeroed slot of an undelivered vector is never read.
        if (!own_only) {
            CHECK(node_pair_distance(data, 0, queued) == kMax && stored_vec(data, queued) == nullptr);
            std::vector<float> buffer, distances;
            std::vector<uint8_t> flags;
            LazyAdcTable none(data.pq_table, kVectors[0].data(), buffer);
            const auto counts = hybrid_score(data, kVectors[0].data(), {1, queued}, none, distances, flags);
            CHECK(counts.exact == 1 && counts.missing == 1 && distances[1] == kMax);
        }
        // The insert of `own` (owned here) names 1 (own), 2 (rank 1) and the two undelivered ids.
        std::vector<WorkItem> inserts(1);
        inserts[0].request = ann::serialize_insert_request(1, kVectors[3].data(), kDim, {1, queued, 2, labeled},
                                                           {1.0f, 1.5f, 2.0f, 2.5f}, 4, 1.2f, 0, own);
        const auto responses = batch_process_inserts(data, inserts, metrics);
        CHECK(responses.size() == 1);
        ann::InsertResponseHeader header{};
        std::vector<uint32_t> neighbors;
        std::vector<CrossPartitionEdge> edges;
        CHECK(ann::deserialize_insert_response(responses[0].response.data(), responses[0].response.size(), header,
                                               neighbors, edges));
        CHECK(std::find(neighbors.begin(), neighbors.end(), queued) == neighbors.end() &&
              std::find(neighbors.begin(), neighbors.end(), labeled) == neighbors.end());
        CHECK(std::find(neighbors.begin(), neighbors.end(), 2u) != neighbors.end());
        for (const auto& edge : edges) CHECK(edge.from_partition < kServers && edge.from_id != queued);
        CHECK(tide::g_graph_metrics.insert_neighbors_undelivered.load() == 2);
        const uint32_t slot = g2l_checked(data, own);
        CHECK(slot != UINT32_MAX && std::find(data.adj.at(slot).begin(), data.adj.at(slot).end(), queued) ==
                                        data.adj.at(slot).end());
        // The broadcast arrives: the id is delivered and the next insert may link it.
        std::vector<char> broadcast(sizeof(ann::VectorBroadcastHeader) + kDim * sizeof(float));
        ann::VectorBroadcastHeader bh{};
        bh.msg_type = ann::MessageType::VECTOR_BROADCAST;
        bh.node_id = queued; bh.dim = kDim; bh.home_partition = 1;
        std::memcpy(broadcast.data(), &bh, sizeof(bh));
        std::memcpy(broadcast.data() + sizeof(bh), kVectors[5].data(), kDim * sizeof(float));
        std::vector<WorkItem> broadcasts(1);
        broadcasts[0].request = broadcast;
        batch_process_broadcasts(data, broadcasts, metrics);
        CHECK(delivered_here(data, queued));
        if (!own_only)
            CHECK(node_pair_distance(data, 0, queued) == compute_distance(kVectors[0].data(), kVectors[5].data(), kDim));
        inserts[0].request = ann::serialize_insert_request(2, kVectors[5].data(), kDim, {queued, 1},
                                                           {0.0f, 1.0f}, 4, 1.2f, 0, own + 1);
        const auto second = batch_process_inserts(data, inserts, metrics);
        CHECK(ann::deserialize_insert_response(second[0].response.data(), second[0].response.size(), header,
                                               neighbors, edges));
        CHECK(std::find(neighbors.begin(), neighbors.end(), queued) != neighbors.end());
        for (const auto& edge : edges) CHECK(edge.from_partition < kServers);
        CHECK(tide::g_graph_metrics.insert_neighbors_undelivered.load() == 2);
    }
    return 0;
}

int main() {
    if (distance_cases() != 0) return 1;
    if (stores_and_ready_flag() != 0) return 1;
    if (hybrid_scorer() != 0) return 1;
    if (search_rerank_and_done() != 0) return 1;
    if (prune_matches_full_replication() != 0) return 1;
    if (maintenance_paths() != 0) return 1;
    if (own_only_loader() != 0) return 1;
    if (own_only_reset() != 0) return 1;
    if (insert_reuses_pooled_slot() != 0) return 1;
    if (undelivered_candidates_are_dropped() != 0) return 1;
    std::cout << "own-only store tests passed\n";
    return 0;
}
