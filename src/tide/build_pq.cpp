/**
 * tide-build-pq — train a Product Quantization codebook from a base .npy and save it.
 *
 * Usage:
 *   tide-build-pq <base.npy> <out_pivots.bin> [M] [iters] [--codes <out_codes.bin>]
 *
 * --codes also encodes every base row with the trained pivots and writes
 * pq_codes.bin (pq.h: magic PQCD, n, M, the pivots' hash, n * M bytes), the file
 * a server loads in own-only (--pq) mode instead of encoding at start.
 *
 * Defaults:
 *   M     : chosen so sub_dim ~= 8 (M = dim/8 when 8 | dim; otherwise the
 *           largest divisor of dim that yields sub_dim closest to 8, falling
 *           back to a divisor that is <= dim). For sift dim=128 -> M=16,
 *           gist dim=960 -> M=120.
 *   iters : 15
 *
 * Training samples up to 256000 rows from the base (stride-sampled if larger).
 *
 * SELF-TEST: encode a sample of base vectors, then over random (i, j) pairs
 * compare adc_distance_direct(vec_i, code_j) vs exact l2_distance_squared(
 * vec_i, vec_j); print the mean relative error. Validates the codebook.
 *
 * Training and encoding (pq.h) are ported algorithmically from the DiskANN
 * reference implementation (src/pq.cpp). DiskANN is Copyright (c) Microsoft
 * Corporation, MIT License: see the NOTICE file.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>

#include "npy_reader.h"  // ann::read_npy_float
#include "distance.h"    // ann::l2_distance_squared
#include "pq.h"          // ann::PQTable, ann::train_pq

namespace {

// Pick M so that sub_dim = dim / M is as close to 8 as possible, requiring
// dim % M == 0. Prefer exact 8; else search divisors of dim.
uint32_t choose_M(uint32_t dim) {
    if (dim % 8u == 0u) return dim / 8u;
    // Find divisor d of dim minimizing |d - 8|; M = dim / d.
    uint32_t best_sub = 1, best_diff = 0xFFFFFFFFu;
    for (uint32_t d = 1; d <= dim; ++d) {
        if (dim % d != 0u) continue;
        uint32_t diff = (d > 8u) ? (d - 8u) : (8u - d);
        if (diff < best_diff) { best_diff = diff; best_sub = d; }
    }
    return dim / best_sub;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> positional;
    std::string codes_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--codes") {
            if (i + 1 >= argc || !codes_path.empty()) {
                std::fprintf(stderr, "[build_pq] --codes needs one output path\n");
                return 1;
            }
            codes_path = argv[++i];
        } else {
            positional.push_back(arg);
        }
    }
    if (positional.size() < 2 || positional.size() > 4) {
        std::fprintf(stderr,
            "Usage: %s <base.npy> <out_pivots.bin> [M] [iters] [--codes <out_codes.bin>]\n", argv[0]);
        return 1;
    }
    const std::string base_path = positional[0];
    const std::string out_path = positional[1];

    uint32_t rows = 0, cols = 0;
    std::vector<float> base;
    try {
        base = ann::read_npy_float(base_path, rows, cols);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[build_pq] failed to read base: %s\n", e.what());
        return 1;
    }
    const uint32_t dim = cols;
    const size_t n = rows;
    if (n == 0 || dim == 0) {
        std::fprintf(stderr, "[build_pq] empty base (rows=%u cols=%u)\n", rows, cols);
        return 1;
    }

    uint32_t M = (positional.size() >= 3)
        ? static_cast<uint32_t>(std::strtoul(positional[2].c_str(), nullptr, 10)) : choose_M(dim);
    uint32_t iters = (positional.size() >= 4)
        ? static_cast<uint32_t>(std::strtoul(positional[3].c_str(), nullptr, 10)) : 15u;

    if (M == 0 || dim % M != 0u) {
        std::fprintf(stderr,
            "[build_pq] invalid M=%u for dim=%u (require dim %% M == 0). "
            "Suggested M=%u (sub_dim~8).\n", M, dim, choose_M(dim));
        return 1;
    }
    const uint32_t sub_dim = dim / M;

    // ---- Build training sample (stride-sample up to 256000 rows) -----------
    const size_t MAX_TRAIN = 256000;
    size_t train_n;
    std::vector<float> train;
    const float* train_ptr = nullptr;
    if (n <= MAX_TRAIN) {
        train_n = n;
        train_ptr = base.data();
    } else {
        // Stride so that we take ~MAX_TRAIN evenly spaced rows.
        size_t stride = n / MAX_TRAIN;       // >= 1
        if (stride < 1) stride = 1;
        train.reserve(MAX_TRAIN * dim);
        for (size_t i = 0; i < n && (train.size() / dim) < MAX_TRAIN; i += stride) {
            const float* src = base.data() + i * dim;
            train.insert(train.end(), src, src + dim);
        }
        train_n = train.size() / dim;
        train_ptr = train.data();
    }

    // ---- Train -------------------------------------------------------------
    ann::PQTable pq;
    try {
        ann::train_pq(train_ptr, train_n, dim, M, ann::PQTable::K, iters, pq);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[build_pq] train_pq failed: %s\n", e.what());
        return 1;
    }

    // ---- Save --------------------------------------------------------------
    try {
        pq.save(out_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[build_pq] save failed: %s\n", e.what());
        return 1;
    }

    // ---- SELF-TEST: mean relative error of ADC vs exact squared L2 ---------
    // Encode a sample of base vectors, then over random (i,j) pairs compare
    // adc_distance_direct(vec_i, code_j) against l2_distance_squared(vec_i,vec_j).
    {
        const size_t SAMPLE = std::min<size_t>(n, 20000);
        // Stride-sample SAMPLE rows from base as the "encoded set".
        std::vector<size_t> sample_idx;
        sample_idx.reserve(SAMPLE);
        {
            size_t stride = (n + SAMPLE - 1) / SAMPLE;  // ceil(n/SAMPLE) >= 1
            if (stride < 1) stride = 1;
            for (size_t i = 0; i < n && sample_idx.size() < SAMPLE; i += stride) {
                sample_idx.push_back(i);
            }
        }
        const size_t S = sample_idx.size();

        // Encode the sampled vectors.
        std::vector<uint8_t> codes(S * pq.code_bytes());
        for (size_t s = 0; s < S; ++s) {
            pq.encode(base.data() + sample_idx[s] * dim, codes.data() + s * pq.code_bytes());
        }

        // Random pairs (a, b) over the sample; a provides full vector (query),
        // b provides the PQ code. Skip self-pairs and ~zero exact distances.
        const size_t N_PAIRS = std::min<size_t>(200000, S * 50);
        std::mt19937 rng(987654321u);
        std::uniform_int_distribution<size_t> pickS(0, S - 1);

        double sum_rel = 0.0;
        size_t counted = 0;
        for (size_t t = 0; t < N_PAIRS; ++t) {
            size_t a = pickS(rng);
            size_t b = pickS(rng);
            if (a == b) continue;
            const float* va = base.data() + sample_idx[a] * dim;
            const float* vb = base.data() + sample_idx[b] * dim;
            float exact = ann::l2_distance_squared(va, vb, dim);
            if (exact <= 1e-6f) continue;  // avoid div-by-~0
            float approx = pq.adc_distance_direct(va, codes.data() + b * pq.code_bytes());
            sum_rel += std::fabs(static_cast<double>(approx) - static_cast<double>(exact))
                       / static_cast<double>(exact);
            counted++;
        }
        double mean_rel_err = (counted > 0) ? (sum_rel / static_cast<double>(counted)) : 0.0;

        std::printf("[build_pq] M=%u sub_dim=%u K=%u dim=%u base_n=%zu train_n=%zu "
                    "iters=%u code_bytes=%zu pairs=%zu mean_rel_err=%.6f\n",
                    M, sub_dim, ann::PQTable::K, dim, n, train_n, iters,
                    pq.code_bytes(), counted, mean_rel_err);
        std::printf("[build_pq] saved codebook -> %s\n", out_path.c_str());
    }

    // ---- Codes of every base row (--codes) ---------------------------------
    if (!codes_path.empty()) {
        if (n > UINT32_MAX) {
            std::fprintf(stderr, "[build_pq] base has more than 2^32 rows\n");
            return 1;
        }
        // Reload the saved pivots so the codes are those of the file a server loads.
        ann::PQTable saved;
        if (!saved.load(out_path) || saved.centroids != pq.centroids) {
            std::fprintf(stderr, "[build_pq] the saved pivots do not reload identically\n");
            return 1;
        }
        std::vector<uint8_t> codes(n * saved.code_bytes());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
            saved.encode(base.data() + static_cast<size_t>(i) * dim, codes.data() + static_cast<size_t>(i) * M);
        }
        try {
            ann::write_pq_codes(codes_path, codes, static_cast<uint32_t>(n), M, saved.pivots_hash());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[build_pq] codes write failed: %s\n", e.what());
            return 1;
        }
        std::printf("[build_pq] encoded %zu rows -> %s (pivots_hash %016llx)\n", n, codes_path.c_str(),
                    static_cast<unsigned long long>(saved.pivots_hash()));
    }
    std::printf("{\"build_pq\": {\"dim\": %u, \"M\": %u, \"sub_dim\": %u, \"K\": %u, \"iters\": %u, "
                "\"base_rows\": %zu, \"train_rows\": %zu, \"codes\": %s, \"pivots_hash\": \"%016llx\"}}\n",
                dim, M, sub_dim, ann::PQTable::K, iters, n, train_n, codes_path.empty() ? "false" : "true",
                static_cast<unsigned long long>(pq.pivots_hash()));
    return 0;
}
