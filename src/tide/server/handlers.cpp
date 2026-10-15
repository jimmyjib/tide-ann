// Server request handlers: search steps and the insert, backlink and broadcast batches.

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
            // A pooled slot must be a freed row (below n_local, unmapped, empty). Anything else stays pooled.
            const bool sound = slot < n_local_load(data) &&
                data.local_to_global.segment(slot / decltype(data.local_to_global)::chunk_size) &&
                data.adj.segment(slot / decltype(data.adj)::chunk_size) &&
                data.local_to_global.at(slot) == UINT32_MAX && data.adj.at(slot).empty() &&
                data.migration_slot_generations.ensure_capacity_for(slot);
            const uint64_t generation = sound ? migration_slot_generation(data, slot) : 0;
            // keep one generation step for a later migration
            if (sound && generation < UINT64_MAX - 1) {
                data.free_slot_pool.pop_back();
                data.migration_slot_generations.at(slot) = generation + 1;
                return slot;
            }
        }
    }
    // Append: every store of the row and its lock exist before n_local publishes it. Insert batches
    // race for the row with a compare-exchange. Nothing is published when a store is full.
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

// One search step over the beam entries this server owns.
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
    (void)query_id; (void)hop_count;
    pmetrics = ProcessMetrics{};
    pmetrics.beam_size_in = static_cast<uint32_t>(beam.size());

    // PQ mode: exact distance for held vectors, ADC for codes, and expanded entries are re-ranked, so
    // the returned top-K has exact distances. Distance work counts re-ranks and scored neighbors, but the
    // ADC table build time is not part of it (docs/metrics.md).
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
    auto& adj = data.adj;
    // read the membership size once per call
    size_t membership_size = mem_size(data);

    // n_base grows with inserts
    if (seen.size() < n) {
        seen.resize(n, false);
    }

    for (const auto& entry : beam) {
        if (entry.node_id < n && entry.node_id < seen.size()) {
            if (!seen[entry.node_id]) {
                seen[entry.node_id] = true;
                seen_nodes.push_back(entry.node_id);
            }
        }
    }

    while (true) {
        std::vector<std::pair<size_t, BeamEntry*>> top_W;
        for (size_t i = 0; i < beam.size() && top_W.size() < W; i++) {
            if (!beam[i].expanded) {
                top_W.emplace_back(i, &beam[i]);
            }
        }

        if (top_W.empty()) {
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

        std::vector<std::pair<size_t, BeamEntry*>> local_nodes;
        for (auto& [idx, entry] : top_W) {
            if (entry->node_id >= membership_size) {
                // no membership entry: mark it expanded
                entry->expanded = 1;
                continue;
            }
            uint32_t node_partition = mem_at(data, entry->node_id);
            if (node_partition == UINT32_MAX) {
                // unknown owner: mark it expanded and skip it
                entry->expanded = 1;
                continue;
            }
            // a label >= K is corrupt: stop (owner_label.h)
            const bool is_local =
                owner_label_checked(node_partition, n_servers, "search expansion") == my_partition;
            if (is_local) {
                local_nodes.emplace_back(idx, entry);
            }
        }

        pmetrics.top_W_count += static_cast<uint32_t>(top_W.size());
        pmetrics.local_nodes_expanded += static_cast<uint32_t>(local_nodes.size());

        if (local_nodes.empty()) {
            // are all top-W entries of unknown owner?
            bool all_unknown = true;
            for (auto& [idx, entry] : top_W) {
                if (entry->node_id < membership_size &&
                    mem_at(data, entry->node_id) != UINT32_MAX) {
                    all_unknown = false;
                    break;
                }
            }
            if (all_unknown) {
                // mark them expanded and loop again
                for (auto& [idx, entry] : top_W) {
                    entry->expanded = 1;
                }
                continue;
            }
            // hop to the first entry with a known owner
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

        std::vector<uint32_t> neighbors_to_eval;
        // Window statistics: per expansion, (source id, start offset) into neighbors_to_eval and whether
        // it re-ranks its source. The acquire pairs with BEGIN's release store (stats_window.cpp).
        const bool window_active = data.stats_window_active.load(
            std::memory_order_acquire);
        std::vector<std::pair<uint32_t, uint32_t>> window_spans;
        std::vector<uint8_t> span_reranked;
        if (window_active) {
            window_spans.reserve(local_nodes.size());
            span_reranked.reserve(local_nodes.size());
        }
        // expanded entries with estimated distances, re-scored below
        std::vector<std::pair<BeamEntry*, const float*>> reranks;
        uint32_t n_base_snap = n_base_load(data);
        uint32_t n_local_snap = n_local_load(data);
        for (auto& [idx, entry] : local_nodes) {
            entry->expanded = 1;
            if (entry->node_id >= n_base_snap) continue;
            // an id created elsewhere may have no mapping chunk here
            uint32_t local_idx = g2l_checked(data, entry->node_id);
            if (!valid_search_source(entry->node_id, n_base_snap,
                                     local_idx, n_local_snap)) continue;
            bool rerank = false;
            if (own_only && !ann::beam_entry_exact(*entry)) {
                // re-rank: this server holds the vector
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
            // a mapped row always has its lock
            if (local_idx < data.node_locks.size()) {
                // time the wait for the node lock and the hold for the row copy
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
            // no distance work for this frontier
            continue;
        }

        // Distance work, timed together: re-ranks, then exact, ADC or FLT_MAX per neighbor (hybrid_score).
        // The ADC table build time is subtracted.
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
            pmetrics.neighbors_evaluated += static_cast<uint32_t>(neighbors_to_eval.size());
            scored = hybrid_score(data, query, neighbors_to_eval, adc, neighbor_dists, neighbor_kind);
        } else {
            neighbor_kind.clear();
        }
        // re-ranks plus scored neighbors (a missing code is no work)
        pmetrics.distance_computations += static_cast<uint32_t>(reranks.size()) + scored.exact + scored.adc;
        pmetrics.exact_distances += scored.exact;
        pmetrics.adc_distances += scored.adc;
        pmetrics.missing_codes += scored.missing;
        // window statistics per expansion
        if (window_active && neighbor_kind.size() == neighbors_to_eval.size()) {
            stats_window_count_work(data, window_spans, neighbor_kind, span_reranked);
        }

        if (neighbors_to_eval.empty()) {
            pmetrics.distance_us += duration_cast<nanoseconds>(steady_clock::now() - t_dist_start).count();
            // keep the beam ordered after a re-rank-only frontier
            std::sort(beam.begin(), beam.end(),
                [](const BeamEntry& a, const BeamEntry& b) { return a.distance < b.distance; });
            continue;
        }

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

// Applies backlink batches. Each row is changed under its node lock.
std::vector<ResponseItem> batch_process_backlinks(
    ServerData& data,
    std::vector<WorkItem>& batch,
    TimingMetrics& metrics
) {
    std::vector<ResponseItem> responses;
    responses.reserve(batch.size());
    if (batch.empty()) return responses;

    auto t_lock_acquired = steady_clock::now();
    uint64_t lock_wait_us = 0;
    g_lock_metrics.backlink_lock_count += batch.size();

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

        // a message of the wrong size or dimension is rejected (0 edges)
        const size_t expected_size = sizeof(ann::BatchBacklinkHeader) +
            static_cast<size_t>(hdr->num_edges) * sizeof(uint32_t) + static_cast<size_t>(data.dim) * sizeof(float);
        if (hdr->dim != data.dim || work.request.size() < expected_size) {
            std::cerr << "[BACKLINK] rejected: dimension " << hdr->dim << " (server " << data.dim << "), "
                      << work.request.size() << " bytes for " << hdr->num_edges << " edges" << std::endl;
            resp.response = ann::serialize_batch_backlink_ack(hdr->batch_id, 0, 0, 0);
            responses.push_back(std::move(resp));
            continue;
        }

        // The new node is owned elsewhere: full replication copies its vector, PQ mode keeps its code.
        uint32_t from_id = hdr->from_id;
        // grow capacity, vector, membership and id mapping before the n_base release store
        server_ensure_mem_capacity(data, from_id);
        data.global_to_local.ensure_capacity_for(from_id, UINT32_MAX);
        if (data.own_only) {
            store_pq_code(data, from_id, vec_data);
        } else {
            // written once, since a broadcast of the same id may run concurrently
            store_replicated_vector(data, from_id, vec_data);
        }
        mem_set(data, from_id, owner_label_checked(hdr->from_partition, data.n_servers, "backlink owner"));
        n_base_bump(data, from_id + 1);
        const KnownVector from_vector{from_id, vec_data};
        const KnownVector* known = data.own_only ? &from_vector : nullptr;

        const uint32_t n_base_snap = n_base_load(data);
        const uint32_t n_local_snap = n_local_load(data);
        const uint32_t hdr_R = hdr->R;
        const float hdr_alpha = hdr->alpha;
        const uint32_t edge_ovl = data.edge_overload;

        uint32_t edges_processed = 0;
        g_graph_metrics.backlink_edges_requested += hdr->num_edges;
        for (uint32_t i = 0; i < hdr->num_edges; i++) {
            uint32_t to_id = to_ids[i];
            if (to_id < n_base_snap) {
                // to_id may have been created elsewhere (no mapping chunk here)
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
                            // to_id is owned here: exact for held vectors and from_id, ADC for the others
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
                // unknown id: dropped, counted
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

// batch_process_inserts: allocate rows, prune, then apply edges per node.
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
            // a request with a short vector is rejected
            std::cerr << "[INSERT] rejected: vector dimension " << st.vector.size() << " (server " << data.dim
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

    // allocate rows
    {
        auto t_mem_start = steady_clock::now();
        ensure_capacity(data, max_new_id, resize_result);

        for (auto& st : states) {
            if (!st.valid) continue;
            n_base_bump(data, st.new_id + 1);
            // a freed row from the pool, or a new one with its stores and lock allocated
            st.new_local_idx = insert_row_slot(data);
            if (st.new_local_idx == UINT32_MAX) {
                // a store is full (10,000,000 rows): rejected, nothing published
                std::cerr << "[INSERT] rejected: no row slot for id " << st.new_id << " (row capacity reached)"
                          << std::endl;
                st.valid = false;
                --valid_count;
                continue;
            }
            if (data.own_only) {
                // vector and code are stored before the mapping is published
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

    // prune every insert and collect the per-node edge changes
    struct NodeOp {
        enum Type { SET_ADJ, ADD_EDGE };
        Type type;
        uint32_t node_id;
        std::vector<uint32_t>* adj_ptr;
        uint32_t R;
        float alpha;
    };
    std::unordered_map<uint32_t, std::vector<NodeOp>> node_ops;
    if (valid_count > 0) {
        uint32_t typical_R = 64;
        for (auto& st : states) { if (st.valid) { typical_R = st.R; break; } }
        node_ops.reserve(valid_count * (typical_R + 1));
    }

    {
        auto t_lock_acquired = steady_clock::now();

        const uint32_t n_base_snap = n_base_load(data);
        const uint32_t n_local_snap = n_local_load(data);
        const uint32_t my_partition_snap = data.my_partition;

        std::vector<std::pair<uint32_t, float>> cands;

        for (auto& st : states) {
            if (!st.valid) continue;

            cands.clear();
            cands.reserve(st.candidates.size());
            for (size_t i = 0; i < st.candidates.size() && i < st.distances.size(); i++) {
                // A candidate this server has not fully received yet has no vector, code or owner label here:
                // it is skipped and counted.
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

            // every kept neighbor has an owner label here, so the check below is only a guard
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
                    // the mapping chunk of an id created elsewhere may be absent
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

    // apply the edge changes per node
    {
        // only the node lock is needed
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
                    // global_id is owned here: exact for held vectors, ADC for remote neighbors
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

        // a resize is reported on one response of the batch
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

// Applies broadcast batches without a global lock. Each id writes its own slots.
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
            std::cerr << "[BROADCAST] rejected: dimension " << vb_hdr->dim << " (server " << data.dim << ")"
                      << std::endl;
        } else if (work.request.size() >= full_size) {
            const float* vec_data = reinterpret_cast<const float*>(
                work.request.data() + sizeof(ann::VectorBroadcastHeader));

            // PQ mode keeps only the code, on the home server too (its INSERT stores the vector).
            server_ensure_mem_capacity(data, node_id);
            // the id mapping grows with the membership
            data.global_to_local.ensure_capacity_for(node_id, UINT32_MAX);
            if (data.own_only) {
                store_pq_code(data, node_id, vec_data);
            } else {
                // written once, since a backlink of the same id may run concurrently
                store_replicated_vector(data, node_id, vec_data);
            }
            mem_set(data, node_id, owner_label_checked(home_partition, data.n_servers, "broadcast owner"));

            // publish n_base after the vector and membership writes (release)
            n_base_bump(data, node_id + 1);
        }

        // a truncated request or one of another dimension is acknowledged, not applied
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
