// Inserts not yet visible on every server, merged into search results until the drain.
#pragma once

#include <vector>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include "protocol/message_types.h"

namespace tide {

using ann::BeamEntry;

enum class InsertStatus : uint8_t {
    INSERTED = 0,     // backlinks pending
    BROADCASTING = 1,
    COMMITTED = 2     // visible everywhere
};

struct PendingInsert {
    uint32_t node_id;
    std::vector<float> vector;
    uint32_t home_partition;
    InsertStatus status;
};

struct PendingMergeStats {
    uint64_t pending_total = 0;
    uint64_t pending_eligible = 0;
    uint64_t pending_topk = 0;
    bool result_changed = false;
};

// pending inserts by id (thread-safe)
class PendingInsertManager {
public:
    void clear() {
        std::unique_lock lock(mutex_);
        pending_.clear();
    }

    void add(uint32_t node_id, const std::vector<float>& vector, uint32_t partition) {
        std::unique_lock lock(mutex_);
        pending_[node_id] = PendingInsert{node_id, vector, partition, InsertStatus::INSERTED};
    }

    void set_committed(uint32_t node_id) {
        std::unique_lock lock(mutex_);
        auto it = pending_.find(node_id);
        if (it != pending_.end()) {
            it->second.status = InsertStatus::COMMITTED;
        }
    }

    void remove(uint32_t node_id) {
        std::unique_lock lock(mutex_);
        pending_.erase(node_id);
    }

    // Merges the beam and the pending inserts into the top K. Ids >= exclude_threshold are left out
    // (for an inserted query: itself and later inserts, which its ground truth does not contain).
    PendingMergeStats merge_with_beam(const float* query, uint32_t dim,
                                      const std::vector<BeamEntry>& beam,
                                      std::vector<uint32_t>& neighbors,
                                      std::vector<float>& distances, uint32_t K,
                                      uint32_t exclude_threshold = UINT32_MAX) {
        PendingMergeStats stats;
        std::shared_lock lock(mutex_);
        if (pending_.empty()) return stats;

        // Distances are squared L2, but result_distances hold square roots and are squared back.
        std::vector<std::pair<float, uint32_t>> candidates;
        candidates.reserve((beam.empty() ? neighbors.size() : beam.size()) + pending_.size());

        std::vector<uint32_t> before = neighbors;
        if (!beam.empty()) {
            for (const auto& entry : beam) {
                if (entry.node_id >= exclude_threshold) continue;
                candidates.emplace_back(entry.distance, entry.node_id);
            }
        } else {
            for (size_t i = 0; i < neighbors.size(); ++i) {
                if (neighbors[i] >= exclude_threshold) continue;
                float dist = distances[i];
                candidates.emplace_back(dist * dist, neighbors[i]);
            }
        }

        for (const auto& [nid, pi] : pending_) {
            stats.pending_total++;
            if (pi.status == InsertStatus::COMMITTED) continue;
            if (nid >= exclude_threshold) continue;
            stats.pending_eligible++;
            float dist = compute_l2_squared(query, pi.vector, dim);
            candidates.emplace_back(dist, nid);
        }

        std::sort(candidates.begin(), candidates.end());

        neighbors.clear();
        distances.clear();
        std::unordered_set<uint32_t> emitted;
        emitted.reserve(K * 2 + 1);
        for (const auto& [dist, nid] : candidates) {
            if (nid >= exclude_threshold) continue;
            if (!emitted.insert(nid).second) continue;
            neighbors.push_back(nid);
            distances.push_back(std::sqrt(std::max(dist, 0.0f)));
            if (pending_.find(nid) != pending_.end()) stats.pending_topk++;
            if (neighbors.size() >= K) break;
        }

        stats.result_changed = before != neighbors;
        return stats;
    }

    size_t size() const {
        std::shared_lock lock(mutex_);
        return pending_.size();
    }

    bool contains(uint32_t node_id) const {
        std::shared_lock lock(mutex_);
        return pending_.find(node_id) != pending_.end();
    }

private:
    static float compute_l2_squared(const float* q, const std::vector<float>& v, uint32_t dim) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) {
            float diff = q[i] - v[i];
            sum += diff * diff;
        }
        return sum;
    }

    std::map<uint32_t, PendingInsert> pending_;
    mutable std::shared_mutex mutex_;
};

}  // namespace tide
