/**
 * Graph operations of the server.
 *
 * - load_server_data: Load partition data from disk
 * - ensure_capacity: grow the chunked stores for a new global id
 * - robust_prune: α-RNG based pruning (FreshDiskANN Algorithm 3)
 *
 * The insert itself is batch_process_inserts in handlers.cpp.
 */

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <atomic>
#include <stdexcept>
#include <string>

#include "server_types.h"
#include "distance.h"
#include "npy_reader.h"
#include "owner_label.h"
#include "stats_window.h"

using namespace std::chrono;

namespace tide {

// Import npy_reader functions from ann namespace
using ann::read_npy_float;
using ann::read_npy_uint32;
using ann::PartitionGraph;
using ann::read_partition_graph;

// =============================================================================
// load_server_data - Load partition data from disk
// =============================================================================
namespace {

// Loads the own-only stores (hybrid PQ storage, --pq) after the partition graph:
// pq_pivots.bin, pq_codes.bin (one code per load-time id, encoded with exactly
// these pivots) and partition_<rank>/vectors.npy (row i = the vector of
// local_to_global[i]). Every file is required: there is no fallback to base.npy.
void load_own_only_stores(ServerData& data, const std::string& data_dir, const std::string& partition_dir,
                          uint32_t n_membership, uint32_t n_local) {
    const std::string pivots_path = data_dir + "/pq_pivots.bin";
    const std::string codes_path = data_dir + "/pq_codes.bin";
    const std::string vectors_path = partition_dir + "/vectors.npy";
    const std::string hint = " (own-only storage, --pq; start with --full-vectors to load base.npy instead)";
    if (!data.pq_table.load(pivots_path))
        throw std::runtime_error("missing or invalid PQ pivots " + pivots_path + hint);
    const uint32_t M = data.pq_table.M;
    data.dim = data.pq_table.dim;
    data.pq_codes_static = ann::read_pq_codes(codes_path, n_membership, M, data.pq_table.pivots_hash());
    uint32_t rows = 0, cols = 0;
    std::vector<float> own;
    try {
        own = ann::read_npy_float_matrix_checked(vectors_path, rows, cols);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string(e.what()) + hint);
    }
    if (rows != n_local || cols != data.dim)
        throw std::runtime_error(vectors_path + ": shape " + std::to_string(rows) + " x " + std::to_string(cols) +
                                 " is not local_to_global.npy's " + std::to_string(n_local) + " rows x dim " +
                                 std::to_string(data.dim));
    data.full_own.init(data.dim);
    if (n_local > 0 && !data.full_own.ensure_capacity_for(n_local - 1))
        throw std::runtime_error("own vector store capacity exhausted");
    for (uint32_t slot = 0; slot < n_local; ++slot) {
        std::copy(own.data() + static_cast<size_t>(slot) * data.dim,
                  own.data() + static_cast<size_t>(slot + 1) * data.dim, data.full_own.vec(slot));
    }
    // Row order: vectors.npy must be in local_to_global order. A deterministic
    // sample of up to 1,024 rows is encoded with the pivots; each code must equal
    // the stored code of local_to_global[row]. A file in another order matches
    // almost nowhere; >= 99 % tolerates a centroid tie decided differently by
    // another build of the encoder.
    const uint32_t sample = std::min<uint32_t>(n_local, 1024);
    uint32_t matched = 0;
    std::vector<uint8_t> code(M);
    for (uint32_t k = 0; k < sample; ++k) {
        const uint32_t slot = static_cast<uint32_t>((static_cast<uint64_t>(k) * n_local) / sample);
        const uint32_t gid = data.local_to_global.at(slot);
        if (gid >= n_membership) throw std::runtime_error("local_to_global names an id beyond membership.npy");
        data.pq_table.encode(data.full_own.vec(slot), code.data());
        if (std::equal(code.begin(), code.end(), data.pq_codes_static.data() + static_cast<size_t>(gid) * M))
            ++matched;
    }
    if (static_cast<uint64_t>(matched) * 100 < static_cast<uint64_t>(sample) * 99)
        throw std::runtime_error(vectors_path + ": rows are not in local_to_global order (" +
                                 std::to_string(matched) + " of " + std::to_string(sample) +
                                 " sampled rows encode to their node's stored code)");
    data.pq_codes_delta.init(M);
}

}  // namespace

uint32_t load_server_data(ServerData& data, const std::string& data_dir, int rank) {
    // Membership: the load-time node count N0 in both storage modes. Every owner
    // label must name one of the K = n_servers servers: a label at or above K is
    // never folded onto another server (owner_label.h), the start fails.
    uint32_t n_membership = 0;
    data.membership = read_npy_uint32(data_dir + "/membership.npy", n_membership);
    std::string label_error;
    if (data.my_partition >= data.n_servers)
        throw std::runtime_error("rank " + std::to_string(data.my_partition) + " is not below the server count " +
                                 std::to_string(data.n_servers));
    if (!owner_labels_valid(data.membership, data.n_servers, &label_error))
        throw std::runtime_error(data_dir + "/membership.npy: " + label_error + " (--num-servers)");

    if (data.own_only) {
        // Own-only: no replicated vectors; N0 comes from membership.npy and the
        // dimension from the pivots (load_own_only_stores).
        data.full_base.clear();
        n_base_set(data, n_membership);
        data.base_static_capacity = n_membership;
    } else {
        // Full replication: every vector from base.npy, which must cover membership.
        std::string base_path = data_dir + "/base.npy";
        uint32_t n_rows, n_cols;
        data.full_base = read_npy_float(base_path, n_rows, n_cols);
        if (n_rows != n_membership)
            throw std::runtime_error(base_path + " has " + std::to_string(n_rows) + " rows, membership.npy " +
                                     std::to_string(n_membership));
        n_base_set(data, n_rows);
        data.dim = n_cols;
        // Lock in the static portion size. Any node_id < this index reads from
        // `full_base`. Anything >= it reads/writes from `full_base_delta`.
        data.base_static_capacity = n_rows;
        data.full_base_delta.init(n_cols);
    }

    // Load partition-specific graph
    data.use_partition_graph = true;
    std::string partition_dir = data_dir + "/partition_" + std::to_string(rank);
    PartitionGraph pg = read_partition_graph(partition_dir, n_base_load(data));

    // Copy from the std::vector that read_partition_graph returns into the
    // ChunkedArray-backed storage.
    n_local_set(data, pg.n_local);
    // adj: ensure chunks covering [0, n_local), then copy each inner vector.
    if (pg.n_local > 0) {
        data.adj.ensure_capacity_for(pg.n_local - 1);
        for (uint32_t i = 0; i < pg.n_local; i++) {
            data.adj.at(i) = std::move(pg.adj[i]);
        }
    }
    // global_to_local: indexed by global_id; populate from pg.
    if (!pg.global_to_local.empty()) {
        data.global_to_local.ensure_capacity_for(pg.global_to_local.size() - 1, UINT32_MAX);
        for (uint32_t gid = 0; gid < pg.global_to_local.size(); gid++) {
            data.global_to_local.at(gid) = pg.global_to_local[gid];
        }
    }
    // local_to_global: indexed by local_idx.
    if (pg.n_local > 0) {
        data.local_to_global.ensure_capacity_for(pg.n_local - 1, UINT32_MAX);
        for (uint32_t lid = 0; lid < pg.n_local; lid++) {
            data.local_to_global.at(lid) = pg.local_to_global[lid];
        }
    }

    if (data.own_only) {
        load_own_only_stores(data, data_dir, partition_dir, n_membership, pg.n_local);
    }

    // Window-statistics counters: the static table of the load-time ids,
    // zeroed; the delta table of inserted ids is grown at each BEGIN.
    stats_window_allocate(data, n_membership);

    // Initialize per-node locks for local nodes (pre-grow via mutex helper).
    ensure_node_locks(data, pg.n_local);

    return pg.start_node;
}

uint32_t reset_server_data(ServerData& data, const std::string& data_dir, int rank) {
    data.full_base.clear();
    data.full_base_delta.clear();
    data.full_base_delta_state.clear();
    // Own-only stores (own_only itself is the start-up mode and stays): rebuilt
    // by load_server_data below.
    data.full_own.clear();
    data.pq_codes_delta.clear();
    std::vector<uint8_t>().swap(data.pq_codes_static);
    data.pq_table = ann::PQTable{};
    data.base_static_capacity = 0;
    data.adj.clear();
    data.membership.clear();
    data.membership_delta.clear();
    data.global_to_local.clear();
    data.local_to_global.clear();
    data.node_locks.clear();
    // The window counters of inserted ids (load_server_data reallocates the
    // static table and empties this one as well).
    data.stats_window_delta.clear();
    // The free row-slot pool names slots of the rows cleared above
    // (n_local_set(0) clears it and the slot generations as well).
    data.free_slot_pool.clear();
    n_base_set(data, 0);
    n_local_set(data, 0);
    return load_server_data(data, data_dir, rank);
}

// =============================================================================
// ensure_capacity - Resize outer vectors for a global ID.
// MUST be called under g_resize_mutex exclusive lock.
// =============================================================================
void ensure_capacity(ServerData& data, uint32_t global_id, CapacityGrowth& result) {
    n_base_bump(data, global_id + 1);
    size_t target = n_base_load(data);

    // full_base / membership chunked, thread-safe ensure_capacity_for.
    {
        auto t0 = steady_clock::now();
        bool grew = server_ensure_vec_capacity(data, static_cast<uint32_t>(target - 1));
        auto t1 = steady_clock::now();
        if (grew) {
            result.resize_full_base.count = 1;
            result.resize_full_base.time_us = static_cast<uint32_t>(duration_cast<microseconds>(t1 - t0).count());
        }
    }
    {
        auto t0 = steady_clock::now();
        bool grew = server_ensure_mem_capacity(data, static_cast<uint32_t>(target - 1));
        auto t1 = steady_clock::now();
        if (grew) {
            result.resize_membership.count = 1;
            result.resize_membership.time_us = static_cast<uint32_t>(duration_cast<microseconds>(t1 - t0).count());
        }
    }
    // global_to_local is a ChunkedArray indexed by global id; ensure_capacity_for
    // is thread-safe (atomic chunk pointer store + small mutex for alloc). adj
    // and the per-node locks are indexed by local row, not by global id: an
    // insert grows them for its own row (insert_row_slot) before publishing it,
    // so resize_adj stays 0.
    {
        auto t0 = steady_clock::now();
        data.global_to_local.ensure_capacity_for(static_cast<size_t>(target - 1), UINT32_MAX);
        auto t1 = steady_clock::now();
        result.resize_global_to_local.time_us = static_cast<uint32_t>(duration_cast<microseconds>(t1 - t0).count());
    }
}

// =============================================================================
// robust_prune - Algorithm 3 from FreshDiskANN paper
// =============================================================================
std::vector<uint32_t> robust_prune(
    const ServerData& data,
    uint32_t p,
    std::vector<std::pair<uint32_t, float>>& candidates,
    uint32_t R,
    float alpha,
    const KnownVector* known
) {
    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<uint32_t> neighbors;
    neighbors.reserve(R);

    for (const auto& [c_id, c_dist] : candidates) {
        if (c_id == p) continue;
        if (neighbors.size() >= R) break;

        bool keep = true;
        for (uint32_t n_id : neighbors) {
            // Exact when this server holds both vectors (always under full
            // replication), else ADC or code-to-code (own-only storage). A
            // distance with no data is FLT_MAX: the candidate is not occluded by
            // that neighbour and is judged on its own c_dist.
            float n_c_dist = node_pair_distance(data, n_id, c_id, known);
            if (alpha * n_c_dist <= c_dist) {
                keep = false;
                break;
            }
        }

        if (keep) {
            neighbors.push_back(c_id);
        }
    }

    return neighbors;
}

} // namespace tide
