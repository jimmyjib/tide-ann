/**
 * Request handlers of the server.
 *
 * Contains:
 * - process_state: BeamSearch state processing
 * - batch_process_inserts / batch_process_backlinks / batch_process_broadcasts:
 *   the insert pipeline's maintenance batches (INSERT_REQUEST,
 *   BATCH_BACKLINK_REQUEST, VECTOR_BROADCAST)
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <atomic>
#include <cmath>
#include <map>
#include <unordered_map>

#include "server_types.h"
#include "owner_label.h"
#include "stats_window.h"
#include "distance.h"
#include "protocol/broadcast_messages.h"

using namespace std::chrono;

namespace tide {

uint32_t insert_row_slot(ServerData& data) {
    {
        std::lock_guard<std::mutex> lock(data.free_slot_mutex);
        if (!data.free_slot_pool.empty()) {
            const uint32_t slot = data.free_slot_pool.back();
            // A pooled slot is a freed row: below n_local, no mapping, an empty
            // row of zero capacity (the PREPARE pool check). Anything else is
            // left in the pool for membership-check to report; the insert appends.
            const bool sound = slot < n_local_load(data) &&
                data.local_to_global.segment(slot / decltype(data.local_to_global)::chunk_size) &&
                data.adj.segment(slot / decltype(data.adj)::chunk_size) &&
                data.local_to_global.at(slot) == UINT32_MAX && data.adj.at(slot).empty() &&
                data.migration_slot_generations.ensure_capacity_for(slot);
            const uint64_t generation = sound ? migration_slot_generation(data, slot) : 0;
            // Keep one generation step for a later migration (as PREPARE does).
            if (sound && generation < UINT64_MAX - 1) {
                data.free_slot_pool.pop_back();
                data.migration_slot_generations.at(slot) = generation + 1;
                return slot;
            }
        }
    }
    // Append. Every store of row `cur` (adjacency, local_to_global, the own
    // vector under own-only storage) and its lock exist before n_local makes
    // the row visible, so a reader bounded by n_local or by node_locks.size()
    // never meets an unallocated chunk. Concurrent insert batches race for the
    // row with a compare-exchange; the loser allocates the next row and tries
    // again. Nothing is published when a store is full (UINT32_MAX).
    uint32_t cur = n_local_load(data);
    for (;;) {
        if (!data.adj.ensure_capacity_for(cur) || !data.local_to_global.ensure_capacity_for(cur, UINT32_MAX) ||
            (data.own_only && !data.full_own.ensure_capacity_for(cur)) ||
            !data.node_locks.grow_to(static_cast<size_t>(cur) + 1))
            return UINT32_MAX;
        if (data.n_local.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel, std::memory_order_acquire))
            return cur;
    }
}

static size_t broadcast_request_size(uint32_t dim) {
    return sizeof(ann::VectorBroadcastHeader) + static_cast<size_t>(dim) * sizeof(float);
}

static void fill_broadcast_ack(ResponseItem& resp, uint32_t node_id) {
    ann::VectorBroadcastAckHeader ack_hdr;
    ack_hdr.msg_type = ann::MessageType::VECTOR_BROADCAST_ACK;
    ack_hdr.padding[0] = ack_hdr.padding[1] = ack_hdr.padding[2] = 0;
    ack_hdr.node_id = node_id;
    resp.response.assign(reinterpret_cast<char*>(&ack_hdr),
                         reinterpret_cast<char*>(&ack_hdr) + sizeof(ack_hdr));
}

// =============================================================================
// process_state - BeamSearch state processing
// =============================================================================
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
) {
    (void)query_id; (void)hop_count;  // kept in the signature for the callers' tracing
    // Reset metrics
    pmetrics = ProcessMetrics{};
    pmetrics.beam_size_in = static_cast<uint32_t>(beam.size());

    // Hybrid PQ own-only storage (server_types.h): a neighbour whose vector this
    // server holds is scored exactly, any other against its code with the
    // query's ADC table (built on first use, in a thread-local buffer), and
    // every entry expanded here is re-scored exactly unless its distance is
    // already exact (BeamEntry flag). DONE fires only when every beam entry is
    // expanded, so the returned top-K carries exact distances; an entry marked
    // expanded without being expanded (unknown membership, an id beyond n_base,
    // no valid local row) can still hold an estimate and is dropped at DONE.
    // Full replication scores everything exactly and never touches the flag.
    //
    // Work accounting (W(v), the distance count and the distance timer): one
    // distance evaluation per re-rank and per neighbour scored exactly or with
    // ADC, all timed inside the distance timer; a neighbour with neither a
    // vector nor a code (FLT_MAX, kScoreMissing) is no distance work. The ADC
    // query table, built at most once per hop (process_state call), is counted
    // separately (adc_tables) and its build time is taken out of the distance
    // time, so it falls in the request component of the price (METRICS.md).
    const bool own_only = data.own_only;
    static thread_local std::vector<float> adc_buffer;
    LazyAdcTable adc(data.pq_table, query, adc_buffer);
    std::vector<uint8_t> neighbor_kind;
    auto record_adc_table = [&]() {
        pmetrics.adc_tables = adc.built() ? 1u : 0u;
        pmetrics.adc_table_ns = adc.build_ns();
    };

    uint32_t n = n_base_load(data);
    uint32_t my_partition = data.my_partition;

    uint32_t n_servers = data.n_servers;
    auto& adj = data.adj;  // ChunkedArray<vector<uint32_t>>; uses operator[]
    // Membership is chunked — use mem_at(data, id). Cache membership_size
    // (total capacity including chunks) once per call since it changes only
    // under unique_lock and we hold shared_lock here.
    size_t membership_size = mem_size(data);

    // Resize seen vector if n_base has grown (due to inserts)
    if (seen.size() < n) {
        seen.resize(n, false);
    }

    // The worker thread clears and restores `seen` around every request (stale state of
    // the previous query never leaks into this one).
    for (const auto& entry : beam) {
        if (entry.node_id < n && entry.node_id < seen.size()) {
            if (!seen[entry.node_id]) {
                seen[entry.node_id] = true;
                seen_nodes.push_back(entry.node_id);
            }
        }
    }

    while (true) {
        // Get top W unexplored nodes
        std::vector<std::pair<size_t, BeamEntry*>> top_W;
        for (size_t i = 0; i < beam.size() && top_W.size() < W; i++) {
            if (!beam[i].expanded) {
                top_W.emplace_back(i, &beam[i]);
            }
        }

        if (top_W.empty()) {
            // Search complete - return results
            status = StateStatus::DONE;
            if (own_only) {
                beam.erase(std::remove_if(beam.begin(), beam.end(),
                                          [](const BeamEntry& e) { return !ann::beam_entry_exact(e); }),
                           beam.end());
            }

            auto t_sort_start = steady_clock::now();
            std::sort(beam.begin(), beam.end(),
                [](const BeamEntry& a, const BeamEntry& b) {
                    return a.distance < b.distance;
                });
            auto t_sort_end = steady_clock::now();
            pmetrics.beam_sort_us += duration_cast<nanoseconds>(t_sort_end - t_sort_start).count();

            size_t k = std::min(static_cast<size_t>(K), beam.size());
            result_neighbors.resize(k);
            result_distances.resize(k);
            for (size_t i = 0; i < k; i++) {
                result_neighbors[i] = beam[i].node_id;
                result_distances[i] = std::sqrt(beam[i].distance);
            }
            record_adc_table();
            return;
        }

        // Check which nodes are local
        std::vector<std::pair<size_t, BeamEntry*>> local_nodes;
        for (auto& [idx, entry] : top_W) {
            if (entry->node_id >= membership_size) {
                // No membership entry for this id: mark it expanded (the search terminates)
                entry->expanded = 1;
                continue;
            }
            uint32_t node_partition = mem_at(data, entry->node_id);
            if (node_partition == UINT32_MAX) {
                // Owner unknown (sentinel): mark it expanded and skip it
                entry->expanded = 1;
                continue;
            }
            // A label at or above K is corrupt state: stop, never fold it (owner_label.h).
            const bool is_local =
                owner_label_checked(node_partition, n_servers, "search expansion") == my_partition;
            if (is_local) {
                local_nodes.emplace_back(idx, entry);
            }
        }

        // Track metrics for observability
        pmetrics.top_W_count += static_cast<uint32_t>(top_W.size());
        pmetrics.local_nodes_expanded += static_cast<uint32_t>(local_nodes.size());

        if (local_nodes.empty()) {
            // Are all top-W entries of unknown owner (sentinel)?
            bool all_unknown = true;
            for (auto& [idx, entry] : top_W) {
                if (entry->node_id < membership_size &&
                    mem_at(data, entry->node_id) != UINT32_MAX) {
                    all_unknown = false;
                    break;
                }
            }
            if (all_unknown) {
                // All top_W unknown — mark expanded, re-loop outer while for forward progress
                for (auto& [idx, entry] : top_W) {
                    entry->expanded = 1;
                }
                continue;  // outer while-loop
            }
            // Hop to first known-membership entry
            status = StateStatus::NEED_HOP;
            uint32_t hop_node_id = 0;
            uint32_t hop_partition = 0;
            for (auto& [idx, entry] : top_W) {
                if (entry->node_id < membership_size &&
                    mem_at(data, entry->node_id) != UINT32_MAX) {
                    hop_node_id = entry->node_id;
                    hop_partition = mem_at(data, hop_node_id);
                    break;
                }
            }
            next_partition = owner_label_checked(hop_partition, n_servers, "search hop");

            auto t_sort_start = steady_clock::now();
            std::sort(beam.begin(), beam.end(),
                [](const BeamEntry& a, const BeamEntry& b) {
                    return a.distance < b.distance;
                });
            auto t_sort_end = steady_clock::now();
            pmetrics.beam_sort_us += duration_cast<nanoseconds>(t_sort_end - t_sort_start).count();
            record_adc_table();
            return;
        }

        // Collect all neighbors that need distance computation
        std::vector<uint32_t> neighbors_to_eval;
        // Window statistics: per-expansion spans into neighbors_to_eval,
        // recorded for every query.
        // (source global id, start offset); the end offset is the next
        // span's start (or the final vector size). span_reranked: whether
        // that expansion re-ranks its source (one more evaluation for it).
        // Acquire: pairs with BEGIN's release store, so an active window's
        // node count and zeroed counters are visible (stats_window.cpp).
        const bool window_active = data.stats_window_active.load(
            std::memory_order_acquire);
        std::vector<std::pair<uint32_t, uint32_t>> window_spans;
        std::vector<uint8_t> span_reranked;
        if (window_active) {
            window_spans.reserve(local_nodes.size());
            span_reranked.reserve(local_nodes.size());
        }
        // Expanded entries held here whose distance is an estimate: re-scored
        // exactly below, inside the distance timer.
        std::vector<std::pair<BeamEntry*, const float*>> reranks;
        uint32_t n_base_snap = n_base_load(data);
        uint32_t n_local_snap = n_local_load(data);
        for (auto& [idx, entry] : local_nodes) {
            entry->expanded = 1;
            if (entry->node_id >= n_base_snap) continue;
            // Checked: an id another server created may have no mapping chunk here.
            uint32_t local_idx = g2l_checked(data, entry->node_id);
            if (!valid_search_source(entry->node_id, n_base_snap,
                                     local_idx, n_local_snap)) continue;
            bool rerank = false;
            if (own_only && !ann::beam_entry_exact(*entry)) {
                // Re-rank: this server holds the expanded node's vector.
                if (const float* vec = own_vec(data, entry->node_id)) {
                    reranks.emplace_back(entry, vec);
                    rerank = true;
                }
            }
            if (window_active) {
                window_spans.emplace_back(
                    static_cast<uint32_t>(entry->node_id),
                    static_cast<uint32_t>(neighbors_to_eval.size()));
                span_reranked.push_back(rerank ? 1 : 0);
            }
            std::vector<uint32_t> neighbors_snapshot;
            // A mapped row always has its lock (the lock store grows before a row
            // is published); a row beyond the lock store is never read unlocked.
            if (local_idx < data.node_locks.size()) {
                // time how long search waits for the per-node
                // mutex held by insert/backlink writers, and how long search itself
                // holds it for the adj snapshot copy.
                auto t_lock_start = steady_clock::now();
                std::lock_guard<std::mutex> nlock(data.node_locks[local_idx].mtx);
                auto t_lock_acquired = steady_clock::now();
                neighbors_snapshot = adj[local_idx];
                auto t_lock_release = steady_clock::now();
                uint64_t wait_us = duration_cast<microseconds>(
                    t_lock_acquired - t_lock_start).count();
                uint64_t hold_us = duration_cast<microseconds>(
                    t_lock_release - t_lock_acquired).count();
                g_lock_metrics.search_node_lock.record(wait_us, hold_us);
            }
            for (uint32_t nb : neighbors_snapshot) {
                if (nb >= seen.size()) {
                    seen.resize(nb + 1, false);
                }
                if (!seen[nb]) {
                    neighbors_to_eval.push_back(nb);
                    seen[nb] = true;
                    seen_nodes.push_back(nb);
                }
            }
        }

        if (neighbors_to_eval.empty() && reranks.empty()) {
            // No distance call is required for this frontier.
            continue;
        }

        // Distance work of this frontier, timed together: the re-ranks, then
        // every new neighbour scored exact against a held vector, else ADC
        // against its code, else FLT_MAX (hybrid_score, server_types.h). The
        // ADC table's build time (at most once per hop) is subtracted.
        auto t_dist_start = steady_clock::now();
        const uint64_t table_ns_before = adc.build_ns();
        for (auto& [entry, vec] : reranks) {
            entry->distance = compute_distance(query, vec, dim);
            ann::set_beam_entry_exact(*entry, true);
        }
        pmetrics.rerank_distances += static_cast<uint32_t>(reranks.size());

        std::vector<float> neighbor_dists;
        HybridScoreCounts scored;
        if (!neighbors_to_eval.empty()) {
            // Track neighbors evaluated
            pmetrics.neighbors_evaluated += static_cast<uint32_t>(neighbors_to_eval.size());
            scored = hybrid_score(data, query, neighbors_to_eval, adc, neighbor_dists, neighbor_kind);
        } else {
            neighbor_kind.clear();
        }
        // Re-ranks plus scored neighbours; a missing code is no distance work.
        pmetrics.distance_computations += static_cast<uint32_t>(reranks.size()) + scored.exact + scored.adc;
        pmetrics.exact_distances += scored.exact;
        pmetrics.adc_distances += scored.adc;
        pmetrics.missing_codes += scored.missing;
        // Window statistics: every expansion's re-rank and scored neighbours.
        if (window_active && neighbor_kind.size() == neighbors_to_eval.size()) {
            stats_window_count_work(data, window_spans, neighbor_kind, span_reranked);
        }

        if (neighbors_to_eval.empty()) {
            pmetrics.distance_us += duration_cast<nanoseconds>(steady_clock::now() - t_dist_start).count();
            // A re-rank-only frontier changed distances in place: keep the beam
            // ordered for the next frontier.
            std::sort(beam.begin(), beam.end(),
                [](const BeamEntry& a, const BeamEntry& b) { return a.distance < b.distance; });
            continue;
        }

        // Update beam with computed distances
        for (size_t i = 0; i < neighbors_to_eval.size(); i++) {
            uint32_t nb = neighbors_to_eval[i];
            float nb_dist = neighbor_dists[i];
            BeamEntry candidate{nb, nb_dist, 0, {0, 0, 0}};
            if (own_only && neighbor_kind[i] == kScoreExact) ann::set_beam_entry_exact(candidate, true);

            if (beam.size() < L) {
                beam.push_back(candidate);
            } else {
                auto worst_it = std::max_element(beam.begin(), beam.end(),
                    [](const BeamEntry& a, const BeamEntry& b) {
                        return a.distance < b.distance;
                    });
                if (nb_dist < worst_it->distance) {
                    *worst_it = candidate;
                }
            }
        }
        auto t_dist_end = steady_clock::now();
        const uint64_t elapsed = duration_cast<nanoseconds>(t_dist_end - t_dist_start).count();
        const uint64_t table_ns = adc.build_ns() - table_ns_before;
        pmetrics.distance_us += elapsed > table_ns ? elapsed - table_ns : 0;

        // Sort beam
        auto t_sort_start = steady_clock::now();
        std::sort(beam.begin(), beam.end(),
            [](const BeamEntry& a, const BeamEntry& b) {
                return a.distance < b.distance;
            });
        auto t_sort_end = steady_clock::now();
        pmetrics.beam_sort_us += duration_cast<nanoseconds>(t_sort_end - t_sort_start).count();

        if (beam.size() > L) {
            beam.resize(L);
        }
    }
}

// =============================================================================
// batch_process_backlinks - Batch process BATCH_BACKLINK_REQUESTs under single lock
// =============================================================================
std::vector<ResponseItem> batch_process_backlinks(
    ServerData& data,
    std::vector<WorkItem>& batch,
    TimingMetrics& metrics
) {
    std::vector<ResponseItem> responses;
    responses.reserve(batch.size());
    if (batch.empty()) return responses;

    // Lock-free batch backlinks. Each iteration's inner adj
    // modification is protected by per-node lock.
    auto t_lock_acquired = steady_clock::now();
    uint64_t lock_wait_us = 0;
    g_lock_metrics.backlink_lock_count += batch.size();

    // Optimization: reuse prune_candidates buffer across the entire batch.
    // Avoids per-prune heap alloc. The vector is cleared+reserved on each use.
    std::vector<std::pair<uint32_t, float>> prune_candidates;

    for (auto& work : batch) {
        ResponseItem resp;
        resp.identity = work.identity;
        resp.coordinator_id = work.coordinator_id;
        resp.needs_forward = false;

        if (work.request.size() < sizeof(ann::BatchBacklinkHeader)) {
            resp.response = ann::serialize_batch_backlink_ack(0, 0, 0, 0);
            responses.push_back(std::move(resp));
            continue;
        }

        ann::BatchBacklinkHeader* hdr = reinterpret_cast<ann::BatchBacklinkHeader*>(
            const_cast<char*>(work.request.data()));
        const uint32_t* to_ids = reinterpret_cast<const uint32_t*>(
            work.request.data() + sizeof(ann::BatchBacklinkHeader));
        const float* vec_data = reinterpret_cast<const float*>(
            work.request.data() + sizeof(ann::BatchBacklinkHeader) +
            hdr->num_edges * sizeof(uint32_t));

        // The message carries num_edges ids and one vector of this server's
        // dimension; anything else is refused unapplied (0 edges processed).
        const size_t expected_size = sizeof(ann::BatchBacklinkHeader) +
            static_cast<size_t>(hdr->num_edges) * sizeof(uint32_t) + static_cast<size_t>(data.dim) * sizeof(float);
        if (hdr->dim != data.dim || work.request.size() < expected_size) {
            std::cerr << "[BACKLINK] refused: dimension " << hdr->dim << " (server " << data.dim << "), "
                      << work.request.size() << " bytes for " << hdr->num_edges << " edges" << std::endl;
            resp.response = ann::serialize_batch_backlink_ack(hdr->batch_id, 0, 0, 0);
            responses.push_back(std::move(resp));
            continue;
        }

        // Store vector data for from_id (the newly inserted node). It is owned
        // by another server: full replication copies the vector, own-only
        // storage keeps only its code (the prune below uses the message's
        // vector for every distance to from_id).
        uint32_t from_id = hdr->from_id;
        // Capacity + vec + mem before n_base bump (release). The id
        // mapping grows too: every server then holds a global_to_local
        // chunk for every id it knows, so a later lookup of an id another
        // server created never meets an absent chunk (lookups stay checked).
        server_ensure_mem_capacity(data, from_id);
        data.global_to_local.ensure_capacity_for(from_id, UINT32_MAX);
        if (data.own_only) {
            store_pq_code(data, from_id, vec_data);
        } else {
            // Written once: a broadcast of the same id may be applied concurrently.
            store_replicated_vector(data, from_id, vec_data);
        }
        mem_set(data, from_id, owner_label_checked(hdr->from_partition, data.n_servers, "backlink owner"));
        n_base_bump(data, from_id + 1);
        const KnownVector from_vector{from_id, vec_data};
        const KnownVector* known = data.own_only ? &from_vector : nullptr;

        // Optimization: snapshot atomic counters once per work item (not per to_id).
        const uint32_t n_base_snap = n_base_load(data);
        const uint32_t n_local_snap = n_local_load(data);
        const uint32_t hdr_R = hdr->R;
        const float hdr_alpha = hdr->alpha;
        const uint32_t edge_ovl = data.edge_overload;

        // Add back-edges for each to_id
        uint32_t edges_processed = 0;
        g_graph_metrics.backlink_edges_requested += hdr->num_edges;
        for (uint32_t i = 0; i < hdr->num_edges; i++) {
            uint32_t to_id = to_ids[i];
            if (to_id < n_base_snap) {
                // Checked: to_id may be an id another server created (no mapping chunk here).
                uint32_t local_idx = g2l_checked(data, to_id);
                if (local_idx != UINT32_MAX && local_idx < n_local_snap) {
                    std::lock_guard<std::mutex> nlock(data.node_locks[local_idx].mtx);
                    auto& neighbors = data.adj[local_idx];
                    if (std::find(neighbors.begin(), neighbors.end(), from_id) == neighbors.end()) {
                        g_graph_metrics.backlink_edges_applied++;
                        neighbors.push_back(from_id);
                        if (hdr_R > 0 && neighbors.size() > hdr_R + edge_ovl) {
                            prune_candidates.clear();
                            prune_candidates.reserve(neighbors.size());
                            // to_id is owned here, so its vector is present: exact
                            // against held vectors and from_id (the message's
                            // vector), ADC against the codes of the others.
                            for (uint32_t n_id : neighbors) {
                                float d = node_pair_distance(data, to_id, n_id, known);
                                prune_candidates.emplace_back(n_id, d);
                            }
                            neighbors = robust_prune(data, to_id, prune_candidates, hdr_R, hdr_alpha, known);
                        }
                    } else {
                        g_graph_metrics.backlink_skip_already_exists++;
                    }
                } else {
                    g_graph_metrics.backlink_skip_not_local++;
                }
            } else {
                // The target id is not (yet) known here: the edge is dropped.
                // Counted so requested = applied + the three skips exactly.
                g_graph_metrics.backlink_skip_out_of_range++;
            }
            edges_processed++;
        }

        auto t_done = steady_clock::now();
        uint32_t processing_us = static_cast<uint32_t>(
            duration_cast<microseconds>(t_done - t_lock_acquired).count());
        resp.response = ann::serialize_batch_backlink_ack(
            hdr->batch_id, edges_processed, static_cast<uint32_t>(lock_wait_us), processing_us);
        responses.push_back(std::move(resp));
    }

    metrics.request_count += batch.size();
    auto t_lock_release = steady_clock::now();
    g_lock_metrics.backlink_unique.record(
        lock_wait_us,
        duration_cast<microseconds>(t_lock_release - t_lock_acquired).count());
    return responses;
}

// =============================================================================
// batch_process_inserts - Batch process INSERT_REQUESTs
//   1. Exclusive lock briefly → resize if needed → release
//   2. Shared lock + alloc_mutex → allocate all nodes
//   3. Shared lock → prune all + collect per-node operations
//   4. Per-node lock → set adj + back-edges + prune (grouped by node)
// =============================================================================
std::vector<ResponseItem> batch_process_inserts(
    ServerData& data,
    std::vector<WorkItem>& batch,
    TimingMetrics& metrics
) {
    std::vector<ResponseItem> responses;
    responses.reserve(batch.size());
    if (batch.empty()) return responses;

    auto batch_pickup_time = steady_clock::now();

    auto avg_u32 = [](uint64_t total, uint32_t count) -> uint32_t {
        return count > 0 ? static_cast<uint32_t>(total / count) : 0;
    };

    // Per-insert state
    struct InsertState {
        uint32_t new_id = 0;
        uint32_t new_local_idx = 0;
        uint32_t R = 0;
        float alpha = 1.0f;
        uint32_t request_id = 0;
        std::vector<float> vector;
        std::vector<uint32_t> candidates;
        std::vector<float> distances;
        std::vector<uint32_t> out_neighbors;
        std::vector<CrossPartitionEdge> cross_partition_edges;
        uint32_t deserialize_us = 0;
        uint32_t prune_us = 0;
        bool valid = false;
    };
    std::vector<InsertState> states(batch.size());

    // Deserialize all + init responses
    uint32_t max_new_id = 0;
    uint32_t valid_count = 0;
    for (size_t b = 0; b < batch.size(); b++) {
        ResponseItem resp;
        resp.identity = batch[b].identity;
        resp.coordinator_id = batch[b].coordinator_id;
        resp.needs_forward = false;
        responses.push_back(std::move(resp));

        auto t_deser_start = steady_clock::now();
        ann::InsertHeader hdr;
        auto& st = states[b];
        if (!ann::deserialize_insert_request(
                batch[b].request.data(), batch[b].request.size(),
                hdr, st.vector, st.candidates, st.distances)) {
            continue;
        }
        if (st.vector.size() != data.dim) {
            // Never read past the request's vector: refused (an invalid insert response).
            std::cerr << "[INSERT] refused: vector dimension " << st.vector.size() << " (server " << data.dim
                      << ")" << std::endl;
            continue;
        }
        auto t_deser_end = steady_clock::now();
        st.deserialize_us = static_cast<uint32_t>(
            duration_cast<microseconds>(t_deser_end - t_deser_start).count());
        st.new_id = hdr.assigned_global_id;
        st.R = hdr.R;
        st.alpha = hdr.alpha;
        st.request_id = hdr.request_id;
        st.valid = true;
        valid_count++;
        if (st.new_id > max_new_id) max_new_id = st.new_id;
    }

    auto t_start = steady_clock::now();

    CapacityGrowth resize_result;
    uint64_t alloc_lock_wait_us = 0;
    uint64_t alloc_memory_us = 0;
    uint64_t edge_lock_wait_us = 0;
    uint64_t edge_node_lock_wait_us = 0;
    uint64_t edge_memory_us = 0;
    uint64_t local_prune_total_us = 0;

    // --- Step 1+2: lock-free allocation via ChunkedArray + atomic n_local ---
    {
        auto t_mem_start = steady_clock::now();
        // ensure_capacity is thread-safe (chunked).
        ensure_capacity(data, max_new_id, resize_result);

        for (auto& st : states) {
            if (!st.valid) continue;
            n_base_bump(data, st.new_id + 1);
            // A freed row slot from the pool (ACCEPT freed it), else a new one
            // whose stores and lock insert_row_slot allocated before publishing it.
            st.new_local_idx = insert_row_slot(data);
            if (st.new_local_idx == UINT32_MAX) {
                // A per-server store is full (10,000,000 rows): refused, nothing published.
                std::cerr << "[INSERT] refused: no row slot for id " << st.new_id << " (row capacity reached)"
                          << std::endl;
                st.valid = false;
                --valid_count;
                continue;
            }
            if (data.own_only) {
                // The own vector and the code are stored before the mapping is
                // published: a reader that finds the slot finds the vector.
                store_own_vector(data, st.new_local_idx, st.vector.data());
                store_pq_code(data, st.new_id, st.vector.data());
            }
            data.global_to_local[st.new_id] = st.new_local_idx;
            data.local_to_global[st.new_local_idx] = st.new_id;
            mem_set(data, st.new_id, data.my_partition);
            if (!data.own_only) {
                store_replicated_vector(data, st.new_id, st.vector.data());
            }
        }
        auto t_mem_end = steady_clock::now();
        g_lock_metrics.insert_alloc_shared.record(
            0,
            duration_cast<microseconds>(t_mem_end - t_mem_start).count());
        alloc_memory_us = duration_cast<microseconds>(t_mem_end - t_mem_start).count();
    }

    if (valid_count > 0) {
        g_lock_metrics.insert_alloc_lock_count += valid_count;
        g_lock_metrics.insert_alloc_lock_wait_us += alloc_lock_wait_us;
    }

    // --- Step 3: Shared lock -> prune all + collect per-node ops ---
    struct NodeOp {
        enum Type { SET_ADJ, ADD_EDGE };
        Type type;
        uint32_t node_id;
        std::vector<uint32_t>* adj_ptr;
        uint32_t R;
        float alpha;
    };
    // Optimization: unordered_map for O(1) avg lookup (vs std::map red-black tree).
    std::unordered_map<uint32_t, std::vector<NodeOp>> node_ops;
    if (valid_count > 0) {
        uint32_t typical_R = 64;
        for (auto& st : states) { if (st.valid) { typical_R = st.R; break; } }
        node_ops.reserve(valid_count * (typical_R + 1));
    }

    {
        // Lock-free batch prune (chunked storage, no resize).
        auto t_lock_acquired = steady_clock::now();

        // Optimization: snapshot atomic / function-call results once per batch.
        const uint32_t n_base_snap = n_base_load(data);
        const uint32_t n_local_snap = n_local_load(data);
        const uint32_t my_partition_snap = data.my_partition;

        // Optimization: reuse cands buffer across all valid inserts in the batch.
        std::vector<std::pair<uint32_t, float>> cands;

        for (auto& st : states) {
            if (!st.valid) continue;

            cands.clear();
            cands.reserve(st.candidates.size());
            for (size_t i = 0; i < st.candidates.size() && i < st.distances.size(); i++) {
                // A candidate this server has not received completely (an id another
                // server inserted, whose broadcast or backlink is still queued here or
                // being applied) has no complete vector or code and no owner label
                // here: dropped and counted, never read. The prune would read a vector
                // being written, and its cross-partition edge would carry no owner.
                if (!delivered_here(data, st.candidates[i])) {
                    g_graph_metrics.insert_neighbors_undelivered++;
                    continue;
                }
                float d = st.distances[i];
                cands.emplace_back(st.candidates[i], d * d);
            }
            auto t_prune_start = steady_clock::now();
            st.out_neighbors = robust_prune(data, st.new_id, cands, st.R, st.alpha);
            auto t_prune_end = steady_clock::now();
            st.prune_us = static_cast<uint32_t>(
                duration_cast<microseconds>(t_prune_end - t_prune_start).count());

            // Every kept neighbour has an owner label here (the candidates were
            // delivered, and a label never returns to unset); a row and an edge
            // never name an id without one, so the check below only guards it.
            st.out_neighbors.erase(
                std::remove_if(st.out_neighbors.begin(), st.out_neighbors.end(), [&](uint32_t neighbor_id) {
                    if (mem_label_checked(data, neighbor_id) < data.n_servers) return false;
                    g_graph_metrics.insert_neighbors_undelivered++;
                    return true;
                }),
                st.out_neighbors.end());

            node_ops[st.new_local_idx].push_back(
                {NodeOp::SET_ADJ, 0, &st.out_neighbors, st.R, st.alpha});

            for (uint32_t neighbor_id : st.out_neighbors) {
                const uint32_t np = mem_label_checked(data, neighbor_id);
                if (np == my_partition_snap) {
                    if (neighbor_id >= n_base_snap) continue;
                    // Checked: the mapping chunk of an id created elsewhere may be absent.
                    uint32_t nlx = g2l_checked(data, neighbor_id);
                    if (nlx != UINT32_MAX && nlx < n_local_snap) {
                        node_ops[nlx].push_back(
                            {NodeOp::ADD_EDGE, st.new_id, nullptr, st.R, st.alpha});
                    }
                } else {
                    CrossPartitionEdge edge;
                    edge.from_id = neighbor_id;
                    edge.to_id = st.new_id;
                    edge.from_partition = np;
                    edge.to_partition = my_partition_snap;
                    st.cross_partition_edges.push_back(edge);
                }
            }
        }
        auto t_lock_release = steady_clock::now();
        g_lock_metrics.insert_prune_shared.record(
            0,
            duration_cast<microseconds>(t_lock_release - t_lock_acquired).count());
    }

    // --- Step 4: Per-node lock -> process all ops per node ---
    {
        // Outer chunked, no shared_lock needed; per-node lock only.
        auto t_lock_acquired = steady_clock::now();

        auto t_mem_start = steady_clock::now();
        for (auto& [local_idx, ops] : node_ops) {
            auto t_node_lock_start = steady_clock::now();
            std::unique_lock<std::mutex> nlock(data.node_locks[local_idx].mtx);
            auto t_node_lock_acquired = steady_clock::now();
            uint64_t node_lock_wait_us = duration_cast<microseconds>(
                t_node_lock_acquired - t_node_lock_start).count();
            edge_lock_wait_us += node_lock_wait_us;
            edge_node_lock_wait_us += node_lock_wait_us;

            auto& adj = data.adj[local_idx];
            uint32_t global_id = data.local_to_global[local_idx];

            bool modified = false;
            uint32_t max_R = 0;
            float max_alpha = 1.0f;

            for (auto& op : ops) {
                if (op.type == NodeOp::SET_ADJ) {
                    adj = *op.adj_ptr;
                } else {
                    if (std::find(adj.begin(), adj.end(), op.node_id) == adj.end()) {
                        adj.push_back(op.node_id);
                        modified = true;
                    }
                }
                if (op.R > max_R) max_R = op.R;
                if (op.alpha > max_alpha) max_alpha = op.alpha;
            }

            if (modified && max_R > 0 && adj.size() > max_R + data.edge_overload) {
                std::vector<std::pair<uint32_t, float>> prune_cands;
                prune_cands.reserve(adj.size());
                for (uint32_t adj_id : adj) {
                    // global_id is owned here: exact against held vectors, ADC
                    // against the codes of remote neighbours.
                    float d = node_pair_distance(data, global_id, adj_id);
                    prune_cands.emplace_back(adj_id, d);
                }
                auto t_prune_start = steady_clock::now();
                adj = robust_prune(data, global_id, prune_cands, max_R, max_alpha);
                auto t_prune_end = steady_clock::now();
                local_prune_total_us += duration_cast<microseconds>(
                    t_prune_end - t_prune_start).count();
            }
        }
        auto t_mem_end = steady_clock::now();
        uint64_t total_edge_time_us = duration_cast<microseconds>(
            t_mem_end - t_mem_start).count();
        edge_memory_us = total_edge_time_us > local_prune_total_us + edge_node_lock_wait_us
            ? total_edge_time_us - local_prune_total_us - edge_node_lock_wait_us
            : 0;
        g_lock_metrics.insert_edge_shared.record(
            0,
            duration_cast<microseconds>(t_mem_end - t_lock_acquired).count());
    }

    if (valid_count > 0) {
        g_lock_metrics.insert_edge_lock_count += valid_count;
        g_lock_metrics.insert_edge_lock_wait_us += edge_lock_wait_us;
    }

    // Build responses
    auto t_end = steady_clock::now();
    uint32_t processing_us = static_cast<uint32_t>(
        duration_cast<microseconds>(t_end - t_start).count());

    bool emitted_resize_stats = false;
    for (size_t b = 0; b < batch.size(); b++) {
        auto& st = states[b];
        ann::InsertTimingBreakdown timing;
        timing.queue_wait_us = static_cast<uint32_t>(
            duration_cast<microseconds>(batch_pickup_time - batch[b].enqueue_time).count());
        timing.deserialize_us = st.deserialize_us;
        timing.alloc_lock_wait_us = avg_u32(alloc_lock_wait_us, valid_count);
        timing.edge_lock_wait_us = avg_u32(edge_lock_wait_us, valid_count);
        timing.prune_us = st.prune_us;
        timing.alloc_memory_us = avg_u32(alloc_memory_us, valid_count);
        timing.edge_memory_us = avg_u32(edge_memory_us, valid_count);
        timing.local_backlink_prune_us = avg_u32(local_prune_total_us, valid_count);

        // Resize is a batch-level event. Put it on one response so coordinator
        // totals stay correct instead of multiplying by batch size.
        if (st.valid && !emitted_resize_stats) {
            timing.resize_full_base.count = resize_result.resize_full_base.count;
            timing.resize_full_base.time_us = resize_result.resize_full_base.time_us;
            timing.resize_membership.count = resize_result.resize_membership.count;
            timing.resize_membership.time_us = resize_result.resize_membership.time_us;
            timing.resize_adj.count = resize_result.resize_adj.count;
            timing.resize_adj.time_us = resize_result.resize_adj.time_us;
            timing.resize_global_to_local.count = resize_result.resize_global_to_local.count;
            timing.resize_global_to_local.time_us = resize_result.resize_global_to_local.time_us;
            emitted_resize_stats = true;
        }

        responses[b].response = ann::serialize_insert_response(
            st.request_id, st.new_id, st.valid,
            st.out_neighbors, st.cross_partition_edges,
            processing_us, timing);
    }

    metrics.request_count += batch.size();
    return responses;
}

// =============================================================================
// batch_process_broadcasts - Batch process multiple VECTOR_BROADCASTs
// =============================================================================
std::vector<ResponseItem> batch_process_broadcasts(
    ServerData& data,
    std::vector<WorkItem>& batch,
    TimingMetrics& metrics
) {
    std::vector<ResponseItem> responses;
    responses.reserve(batch.size());

    if (batch.empty()) return responses;

    const size_t full_size = broadcast_request_size(data.dim);
    uint32_t apply_count = 0;
    for (auto& work : batch) {
        if (work.request.size() >= full_size) apply_count++;
    }

    // Lock-free batch broadcast. No global lock acquired; each iteration
    // writes to disjoint chunked-storage slots and bumps n_base with release.
    steady_clock::time_point t_batch_start = steady_clock::now();
    if (apply_count > 0) {
        g_lock_metrics.broadcast_lock_count += apply_count;
    }

    for (auto& work : batch) {
        ResponseItem resp;
        resp.identity = work.identity;
        resp.coordinator_id = work.coordinator_id;
        resp.needs_forward = false;

        if (work.request.size() < sizeof(ann::VectorBroadcastHeader)) {
            fill_broadcast_ack(resp, 0);
            responses.push_back(std::move(resp));
            continue;
        }

        ann::VectorBroadcastHeader* vb_hdr = reinterpret_cast<ann::VectorBroadcastHeader*>(
            const_cast<char*>(work.request.data()));

        uint32_t node_id = vb_hdr->node_id;
        uint32_t home_partition = vb_hdr->home_partition;

        if (vb_hdr->dim != data.dim) {
            std::cerr << "[BROADCAST] refused: dimension " << vb_hdr->dim << " (server " << data.dim << ")"
                      << std::endl;
        } else if (work.request.size() >= full_size) {
            const float* vec_data = reinterpret_cast<const float*>(
                work.request.data() + sizeof(ann::VectorBroadcastHeader));

            // full_base + membership both chunked with atomic outer arrays.
            // ensure_capacity is idempotent + thread-safe; the fast path is a
            // single atomic load. Own-only storage keeps only the code, on the
            // home server too (its INSERT stores the own vector).
            server_ensure_mem_capacity(data, node_id);
            // The id mapping grows with the membership (see the backlink path).
            data.global_to_local.ensure_capacity_for(node_id, UINT32_MAX);
            if (data.own_only) {
                store_pq_code(data, node_id, vec_data);
            } else {
                // Written once: a backlink of the same id may be applied concurrently.
                store_replicated_vector(data, node_id, vec_data);
            }
            mem_set(data, node_id, owner_label_checked(home_partition, data.n_servers, "broadcast owner"));

            // Bump n_base after vec/mem writes (release ordering, so
            // search threads that observe the new n_base also see vec/mem).
            n_base_bump(data, node_id + 1);
        }

        // A truncated request, or one of another dimension, is acknowledged without being applied.
        fill_broadcast_ack(resp, node_id);
        responses.push_back(std::move(resp));
    }

    metrics.request_count += batch.size();
    if (apply_count > 0) {
        auto t_batch_end = steady_clock::now();
        g_lock_metrics.broadcast_unique.record(
            0,
            duration_cast<microseconds>(t_batch_end - t_batch_start).count());
    }
    return responses;
}

} // namespace tide
