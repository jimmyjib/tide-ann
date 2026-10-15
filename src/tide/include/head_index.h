#pragma once

/**
 * Head index: an in-memory sample of the graph (about 1% of the nodes) that
 * picks the start nodes of every search.
 *
 * This class loads the ghost graph (a 1% sample) and finds the sample
 * nodes nearest to a query by brute force; with 10K samples a brute-force
 * scan takes about 0.1 ms.
 */

#include <vector>
#include <string>
#include <cstdint>
#include <limits>
#include <random>
#include <algorithm>

#include "npy_reader.h"
#include "distance.h"

namespace ann {

class HeadIndex {
public:
    /**
     * Load head index (ghost graph) from data directory.
     *
     * @param data_dir Directory containing ghost/ghost_indices.npy
     * @param base Full base vectors (n_base x dim)
     * @param n_base Number of base vectors
     * @param dim Vector dimension
     */
    void load(const std::string& data_dir,
              const float* base, uint32_t n_base, uint32_t dim) {
        dim_ = dim;

        // Load ghost indices (original global IDs of sampled nodes)
        uint32_t n_ghost;
        global_ids_ = read_npy_int64_as_uint32(data_dir + "/ghost/ghost_indices.npy", n_ghost);
        n_samples_ = n_ghost;

        // Extract ghost vectors from base
        samples_.resize(static_cast<size_t>(n_samples_) * dim_);
        for (uint32_t i = 0; i < n_samples_; i++) {
            uint32_t global_id = global_ids_[i];
            const float* src = base + static_cast<size_t>(global_id) * dim_;
            float* dst = samples_.data() + static_cast<size_t>(i) * dim_;
            for (uint32_t d = 0; d < dim_; d++) {
                dst[d] = src[d];
            }
        }
    }

    /**
     * Find the nearest sample to the query (brute-force).
     *
     * @param query Query vector
     * @return Global ID of the nearest sample node
     */
    uint32_t find_nearest(const float* query) const {
        uint32_t best_idx = 0;
        float best_dist = std::numeric_limits<float>::max();

        for (uint32_t i = 0; i < n_samples_; i++) {
            const float* sample = samples_.data() + static_cast<size_t>(i) * dim_;
            float dist = l2_distance_squared(query, sample, dim_);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = i;
            }
        }

        return global_ids_[best_idx];
    }

    /**
     * Find nearest sample and return both global ID and distance.
     */
    std::pair<uint32_t, float> find_nearest_with_dist(const float* query) const {
        uint32_t best_idx = 0;
        float best_dist = std::numeric_limits<float>::max();

        for (uint32_t i = 0; i < n_samples_; i++) {
            const float* sample = samples_.data() + static_cast<size_t>(i) * dim_;
            float dist = l2_distance_squared(query, sample, dim_);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = i;
            }
        }

        return {global_ids_[best_idx], best_dist};
    }

    uint32_t n_samples() const { return n_samples_; }
    uint32_t dim() const { return dim_; }

    /**
     * Rebuild head index from current base vectors.
     * Used after insert operations to include newly inserted nodes.
     *
     * @param base Full base vectors (including inserted)
     * @param n_base Total number of vectors
     * @param dim Vector dimension
     * @param membership Partition assignment for each node
     * @param n_partitions Number of partitions
     * @param sample_ratio Sampling ratio (default: 1%)
     */
    void rebuild(const float* base, uint32_t n_base, uint32_t dim,
                 const std::vector<uint32_t>& membership, uint32_t n_partitions,
                 float sample_ratio = 0.01f) {
        dim_ = dim;

        // Sample from each partition to maintain balance
        std::vector<std::vector<uint32_t>> partition_nodes(n_partitions);
        for (uint32_t i = 0; i < n_base; i++) {
            uint32_t p = membership[i] % n_partitions;
            partition_nodes[p].push_back(i);
        }

        // Clear and rebuild
        global_ids_.clear();
        std::mt19937 rng(42);  // Fixed seed for reproducibility

        for (uint32_t p = 0; p < n_partitions; p++) {
            auto& nodes = partition_nodes[p];
            uint32_t n_sample = std::max(1u, static_cast<uint32_t>(nodes.size() * sample_ratio));

            // Shuffle and take first n_sample
            std::shuffle(nodes.begin(), nodes.end(), rng);
            for (uint32_t i = 0; i < n_sample && i < nodes.size(); i++) {
                global_ids_.push_back(nodes[i]);
            }
        }

        n_samples_ = static_cast<uint32_t>(global_ids_.size());

        // Extract vectors
        samples_.resize(static_cast<size_t>(n_samples_) * dim_);
        for (uint32_t i = 0; i < n_samples_; i++) {
            uint32_t global_id = global_ids_[i];
            const float* src = base + static_cast<size_t>(global_id) * dim_;
            float* dst = samples_.data() + static_cast<size_t>(i) * dim_;
            for (uint32_t d = 0; d < dim_; d++) {
                dst[d] = src[d];
            }
        }
    }

private:
    std::vector<float> samples_;      // Sampled vectors (n_samples x dim)
    std::vector<uint32_t> global_ids_; // Original global IDs
    uint32_t n_samples_ = 0;
    uint32_t dim_ = 0;
};

}  // namespace ann
