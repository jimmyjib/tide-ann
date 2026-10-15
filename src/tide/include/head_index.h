// Head index: brute-force search over a ~1% sample (ghost/ghost_indices.npy) to pick each query's
// start node.
#pragma once

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
    // sample ids from data_dir/ghost/ghost_indices.npy, vectors from base
    void load(const std::string& data_dir,
              const float* base, uint32_t n_base, uint32_t dim) {
        dim_ = dim;

        uint32_t count = 0;
        global_ids_ = read_npy_int64_as_uint32(data_dir + "/ghost/ghost_indices.npy", count);
        n_samples_ = count;

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

    // global id of the nearest sample
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

    // nearest sample and its distance
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

    // Resamples sample_ratio of every partition's nodes (after inserts).
    void rebuild(const float* base, uint32_t n_base, uint32_t dim,
                 const std::vector<uint32_t>& membership, uint32_t n_partitions,
                 float sample_ratio = 0.01f) {
        dim_ = dim;

        std::vector<std::vector<uint32_t>> partition_nodes(n_partitions);
        for (uint32_t i = 0; i < n_base; i++) {
            uint32_t p = membership[i] % n_partitions;
            partition_nodes[p].push_back(i);
        }

        global_ids_.clear();
        std::mt19937 rng(42);

        for (uint32_t p = 0; p < n_partitions; p++) {
            auto& nodes = partition_nodes[p];
            uint32_t n_sample = std::max(1u, static_cast<uint32_t>(nodes.size() * sample_ratio));

            std::shuffle(nodes.begin(), nodes.end(), rng);
            for (uint32_t i = 0; i < n_sample && i < nodes.size(); i++) {
                global_ids_.push_back(nodes[i]);
            }
        }

        n_samples_ = static_cast<uint32_t>(global_ids_.size());

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
    std::vector<float> samples_;      // n_samples x dim
    std::vector<uint32_t> global_ids_;
    uint32_t n_samples_ = 0;
    uint32_t dim_ = 0;
};

}  // namespace ann
