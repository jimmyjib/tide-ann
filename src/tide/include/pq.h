#pragma once

/**
 * Product Quantization (PQ) — header-only, self-contained.
 *
 * Standard PQ (NOT OPQ). Ported algorithmically from the DiskANN reference
 * (src/pq.cpp: generate_pq_pivots / generate_pq_data_from_pivots_simplified /
 * FixedChunkPQTable::populate_chunk_distances / l2_distance), but with a clean,
 * minimal, equal-size-chunk implementation and our own simple codebook format
 * and k-means. No MKL, no external deps. OpenMP optional. DiskANN is
 * Copyright (c) Microsoft Corporation, MIT License: see the NOTICE file.
 *
 * FILES
 *   pq_pivots.bin : u32 magic 0x50515442 (bytes 42 54 51 50), u32 dim, u32 M, u32 K (=256), then the
 *                   centroids as float32 [M][K][sub_dim] (PQTable::save/load).
 *   pq_codes.bin  : u32 magic 0x44435150 (bytes 50 51 43 44), u32 n, u32 M, u32 zero, u64 pivots_hash()
 *                   of the pivots the codes were encoded with, then n * M code
 *                   bytes, row i = node id i (write_pq_codes/read_pq_codes).
 *
 * CORRECTNESS NOTES
 *   - ann::l2_distance_squared returns SQUARED L2 (AVX2). Every distance produced
 *     by this module is therefore a SQUARED-L2 approximation. We never take sqrt.
 *     This is consistent with the graph code's robust_prune / compute_distance, which also
 *     operate on squared L2.
 *   - Equal-size chunks only: require dim % M == 0, sub_dim = dim / M.
 *     (sift dim=128 -> M=16 -> sub_dim=8;  gist dim=960 -> M=120 -> sub_dim=8.)
 *   - K = 256 (NUM_PQ_CENTROIDS), so each chunk code fits in 1 byte (uint8_t).
 *   - Determinism: k-means uses a fixed RNG seed (std::mt19937(1234 + m)).
 *
 * LAYOUT
 *   centroids: flat [M][K][sub_dim], element (m,k,d) at
 *              centroids[(m*K + k)*sub_dim + d].
 *   sym_table: flat [M][K][K], element (m,i,j) at
 *              sym_table[(m*K + i)*K + j]  ==  squared L2 between centroid i and
 *              centroid j within chunk m.  (Symmetric: [m][i][j] == [m][j][i].)
 *   code:      M bytes; code[m] = centroid index (0..K-1) for chunk m.
 *   ADC table: flat [M][K], element (m,k) at table[m*K + k] = squared L2 between
 *              the query's chunk-m sub-vector and centroid k of chunk m.
 */

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <random>
#include <limits>
#include <algorithm>

#include "distance.h"  // ann::l2_distance_squared (SQUARED L2)

namespace ann {

class PQTable {
public:
    uint32_t dim = 0;       // full vector dimension
    uint32_t M = 0;         // number of chunks (== code length in bytes)
    uint32_t sub_dim = 0;   // dim / M  (equal-size chunks)
    static constexpr uint32_t K = 256;  // centroids per chunk (1 byte/code)

    // [M][K][sub_dim] : centroids[(m*K + k)*sub_dim + d]
    std::vector<float> centroids;
    // [M][K][K] : sym_table[(m*K + i)*K + j] = squared L2 between centroids i,j in chunk m
    std::vector<float> sym_table;

    static constexpr uint32_t MAGIC = 0x50515442u;  // little-endian u32: bytes on disk 42 54 51 50

    PQTable() = default;

    // ---- helpers ------------------------------------------------------------

    // Pointer to centroid k of chunk m (sub_dim floats).
    inline const float* centroid_ptr(uint32_t m, uint32_t k) const {
        return centroids.data() + (static_cast<size_t>(m) * K + k) * sub_dim;
    }
    inline float* centroid_ptr_mut(uint32_t m, uint32_t k) {
        return centroids.data() + (static_cast<size_t>(m) * K + k) * sub_dim;
    }

    size_t code_bytes() const { return M; }

    // ---- 8. build_sym_table -------------------------------------------------
    // Fill sym_table from centroids. M*K*K floats (gist M=120 -> ~31 MB).
    void build_sym_table() {
        sym_table.assign(static_cast<size_t>(M) * K * K, 0.0f);
        for (uint32_t m = 0; m < M; ++m) {
            float* st = sym_table.data() + static_cast<size_t>(m) * K * K;
            for (uint32_t i = 0; i < K; ++i) {
                const float* ci = centroid_ptr(m, i);
                // Diagonal is 0; fill i<->j symmetrically.
                for (uint32_t j = i + 1; j < K; ++j) {
                    float d = l2_distance_squared(ci, centroid_ptr(m, j), sub_dim);
                    st[static_cast<size_t>(i) * K + j] = d;
                    st[static_cast<size_t>(j) * K + i] = d;
                }
            }
        }
    }

    // ---- 2. save ------------------------------------------------------------
    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("PQTable::save cannot create: " + path);
        uint32_t magic = MAGIC, d = dim, m = M, k = K;
        f.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        f.write(reinterpret_cast<const char*>(&d), sizeof(d));
        f.write(reinterpret_cast<const char*>(&m), sizeof(m));
        f.write(reinterpret_cast<const char*>(&k), sizeof(k));
        f.write(reinterpret_cast<const char*>(centroids.data()),
                static_cast<std::streamsize>(centroids.size() * sizeof(float)));
        if (!f) throw std::runtime_error("PQTable::save write failed: " + path);
    }

    // ---- 1. load ------------------------------------------------------------
    // Returns false on header magic/K mismatch or short read.
    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        uint32_t magic = 0, d = 0, m = 0, k = 0;
        f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        f.read(reinterpret_cast<char*>(&d), sizeof(d));
        f.read(reinterpret_cast<char*>(&m), sizeof(m));
        f.read(reinterpret_cast<char*>(&k), sizeof(k));
        if (!f) return false;
        if (magic != MAGIC) return false;
        if (k != K) return false;            // K must be 256 (1 byte/code)
        if (m == 0 || d == 0 || d % m != 0) return false;  // equal-size chunks

        dim = d;
        M = m;
        sub_dim = d / m;

        size_t n_floats = static_cast<size_t>(M) * K * sub_dim;
        centroids.resize(n_floats);
        f.read(reinterpret_cast<char*>(centroids.data()),
               static_cast<std::streamsize>(n_floats * sizeof(float)));
        if (static_cast<size_t>(f.gcount()) != n_floats * sizeof(float)) return false;

        build_sym_table();
        return true;
    }

    // ---- 3. encode ----------------------------------------------------------
    // For each chunk m: argmin over K centroids of squared L2 (sub_dim).
    // code_out must have M bytes. Cost ~ M*K*sub_dim = K*dim.
    void encode(const float* vec, uint8_t* code_out) const {
        for (uint32_t m = 0; m < M; ++m) {
            const float* sub = vec + static_cast<size_t>(m) * sub_dim;
            float best = std::numeric_limits<float>::max();
            uint32_t best_k = 0;
            for (uint32_t k = 0; k < K; ++k) {
                float d = l2_distance_squared(sub, centroid_ptr(m, k), sub_dim);
                if (d < best) { best = d; best_k = k; }
            }
            code_out[m] = static_cast<uint8_t>(best_k);
        }
    }

    // ---- 4. build_query_table -----------------------------------------------
    // ADC table [M][K]: table_out[m*K + k] = squared L2 between query chunk m and
    // centroid k of chunk m. table_out must hold M*K floats. Call once per query.
    void build_query_table(const float* query, float* table_out) const {
        for (uint32_t m = 0; m < M; ++m) {
            const float* sub = query + static_cast<size_t>(m) * sub_dim;
            float* row = table_out + static_cast<size_t>(m) * K;
            for (uint32_t k = 0; k < K; ++k) {
                row[k] = l2_distance_squared(sub, centroid_ptr(m, k), sub_dim);
            }
        }
    }

    // ---- 5. adc_distance (asymmetric, prebuilt table) -----------------------
    // Sum_m query_table[m*K + code[m]]. SQUARED L2 approx (full query vs code).
    inline float adc_distance(const float* query_table, const uint8_t* code) const {
        float sum = 0.0f;
        for (uint32_t m = 0; m < M; ++m) {
            sum += query_table[static_cast<size_t>(m) * K + code[m]];
        }
        return sum;
    }

    // ---- 6. adc_distance_direct (asymmetric, on the fly) --------------------
    // Sum_m squared L2 between query chunk m and centroid code[m]. For single
    // calls without a prebuilt table. SQUARED L2 approx.
    inline float adc_distance_direct(const float* query, const uint8_t* code) const {
        float sum = 0.0f;
        for (uint32_t m = 0; m < M; ++m) {
            const float* sub = query + static_cast<size_t>(m) * sub_dim;
            sum += l2_distance_squared(sub, centroid_ptr(m, code[m]), sub_dim);
        }
        return sum;
    }

    // ---- 7. sym_distance (symmetric, code vs code) --------------------------
    // Sum_m sym_table[(m*K + codeA[m])*K + codeB[m]]. SQUARED L2 approx.
    inline float sym_distance(const uint8_t* codeA, const uint8_t* codeB) const {
        float sum = 0.0f;
        for (uint32_t m = 0; m < M; ++m) {
            const float* st = sym_table.data() + static_cast<size_t>(m) * K * K;
            sum += st[static_cast<size_t>(codeA[m]) * K + codeB[m]];
        }
        return sum;
    }

    // ---- 9. decode ----------------------------------------------------------
    // The reconstruction of a code: centroid code[m] of every chunk m. out holds dim floats.
    void decode(const uint8_t* code, float* out) const {
        for (uint32_t m = 0; m < M; ++m) {
            std::memcpy(out + static_cast<size_t>(m) * sub_dim, centroid_ptr(m, code[m]),
                        sub_dim * sizeof(float));
        }
    }

    // ---- 10. pivots_hash ----------------------------------------------------
    // FNV-1a 64 over dim, M, K (u32 little-endian) and the centroid bytes: the
    // identity a codes file records, so codes encoded with other pivots are refused.
    uint64_t pivots_hash() const {
        uint64_t h = 1469598103934665603ULL;
        auto byte = [&h](uint8_t b) { h ^= b; h *= 1099511628211ULL; };
        for (uint32_t word : {dim, M, K}) {
            for (uint32_t shift = 0; shift < 32; shift += 8) byte(static_cast<uint8_t>(word >> shift));
        }
        const auto* bytes = reinterpret_cast<const uint8_t*>(centroids.data());
        for (size_t i = 0; i < centroids.size() * sizeof(float); ++i) byte(bytes[i]);
        return h;
    }
};

// =============================================================================
// pq_codes.bin: precomputed codes of every load-time node.
// =============================================================================
inline constexpr uint32_t kPQCodesMagic = 0x44435150u;  // little-endian u32: bytes on disk 50 51 43 44

inline void write_pq_codes(const std::string& path, const std::vector<uint8_t>& codes, uint32_t n,
                           uint32_t M, uint64_t pivots_hash) {
    if (M == 0 || codes.size() != static_cast<size_t>(n) * M)
        throw std::runtime_error("write_pq_codes: codes size is not n * M");
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("write_pq_codes cannot create: " + path);
    const uint32_t header[4] = {kPQCodesMagic, n, M, 0};
    f.write(reinterpret_cast<const char*>(header), sizeof(header));
    f.write(reinterpret_cast<const char*>(&pivots_hash), sizeof(pivots_hash));
    f.write(reinterpret_cast<const char*>(codes.data()), static_cast<std::streamsize>(codes.size()));
    if (!f) throw std::runtime_error("write_pq_codes write failed: " + path);
}

// Reads a codes file and refuses a wrong magic, a code length other than
// `expected_M`, a row count other than `expected_n`, codes of other pivots
// (`expected_hash`) and a size that is not exactly the header plus n * M bytes.
inline std::vector<uint8_t> read_pq_codes(const std::string& path, uint32_t expected_n, uint32_t expected_M,
                                          uint64_t expected_hash) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open PQ codes file " + path);
    const std::streamoff size = f.tellg();
    f.seekg(0);
    uint32_t header[4]{};
    uint64_t hash = 0;
    f.read(reinterpret_cast<char*>(header), sizeof(header));
    f.read(reinterpret_cast<char*>(&hash), sizeof(hash));
    if (!f || header[0] != kPQCodesMagic) throw std::runtime_error(path + ": not a PQ codes file");
    if (header[2] != expected_M) throw std::runtime_error(path + ": code length differs from the pivots' M");
    if (header[1] != expected_n) throw std::runtime_error(path + ": row count differs from membership.npy");
    if (hash != expected_hash) throw std::runtime_error(path + ": encoded with other pivots (hash mismatch)");
    const uint64_t bytes = uint64_t{header[1]} * header[2];
    if (static_cast<uint64_t>(size) != sizeof(header) + sizeof(hash) + bytes)
        throw std::runtime_error(path + ": size is not the header plus n * M bytes");
    std::vector<uint8_t> codes(static_cast<size_t>(bytes));
    f.read(reinterpret_cast<char*>(codes.data()), static_cast<std::streamsize>(bytes));
    if (!f) throw std::runtime_error(path + ": truncated codes");
    return codes;
}

// =============================================================================
// train_pq — per-chunk Lloyd's k-means, deterministic.
//
//   data    : n * dim floats, row-major.
//   For each chunk m (sub_dim = dim/M):
//     - init: K distinct random sub-vectors (RNG = mt19937(1234 + m)).
//     - iterate `iters` times: assign each of n points to nearest centroid
//       (squared L2 on sub_dim), recompute centroid as mean; reseed empty
//       clusters to a random point.
//   Then out.build_sym_table().
//
//   K is fixed at PQTable::K (256); the K argument must equal that. dim % M == 0
//   is required. If n < K for a chunk, extra centroids duplicate existing points
//   (handled by the distinct-pick loop falling back to with-replacement).
// =============================================================================
inline void train_pq(const float* data, size_t n, uint32_t dim, uint32_t M,
                      uint32_t K, uint32_t iters, PQTable& out) {
    if (M == 0 || dim % M != 0) {
        throw std::runtime_error("train_pq: require dim % M == 0 (equal-size chunks)");
    }
    if (K != PQTable::K) {
        throw std::runtime_error("train_pq: K must equal PQTable::K (256)");
    }
    if (n == 0) {
        throw std::runtime_error("train_pq: n == 0");
    }

    out.dim = dim;
    out.M = M;
    out.sub_dim = dim / M;
    const uint32_t sub_dim = out.sub_dim;
    out.centroids.assign(static_cast<size_t>(M) * K * sub_dim, 0.0f);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int64_t mm = 0; mm < static_cast<int64_t>(M); ++mm) {
        const uint32_t m = static_cast<uint32_t>(mm);
        const size_t chunk_off = static_cast<size_t>(m) * sub_dim;

        // Local per-chunk centroid buffer: [K][sub_dim].
        std::vector<float> cent(static_cast<size_t>(K) * sub_dim, 0.0f);
        std::mt19937 rng(1234u + m);

        // ---- init: pick K distinct random sub-vectors as centroids ----------
        {
            std::uniform_int_distribution<size_t> pick(0, n - 1);
            if (n >= K) {
                // Distinct point indices.
                std::vector<size_t> chosen;
                chosen.reserve(K);
                // Use a set-like guard via sorted check; n can be large so use a
                // hash-free approach: sample then dedup by re-sampling.
                std::vector<bool> used;  // only feasible if n not enormous; guard below
                bool use_bitmap = (n <= (size_t)50'000'000);
                if (use_bitmap) used.assign(n, false);
                while (chosen.size() < K) {
                    size_t idx = pick(rng);
                    if (use_bitmap) {
                        if (used[idx]) continue;
                        used[idx] = true;
                    }
                    chosen.push_back(idx);
                }
                for (uint32_t k = 0; k < K; ++k) {
                    const float* src = data + chosen[k] * dim + chunk_off;
                    std::memcpy(cent.data() + static_cast<size_t>(k) * sub_dim, src,
                                sub_dim * sizeof(float));
                }
            } else {
                // n < K: seed first n centroids from the n points, remainder from
                // random (with replacement) points so all K slots are valid.
                for (uint32_t k = 0; k < K; ++k) {
                    size_t idx = (k < n) ? k : pick(rng);
                    const float* src = data + idx * dim + chunk_off;
                    std::memcpy(cent.data() + static_cast<size_t>(k) * sub_dim, src,
                                sub_dim * sizeof(float));
                }
            }
        }

        // ---- Lloyd iterations ------------------------------------------------
        std::vector<double> accum(static_cast<size_t>(K) * sub_dim);
        std::vector<size_t> counts(K);
        std::uniform_int_distribution<size_t> pick(0, n - 1);

        for (uint32_t it = 0; it < iters; ++it) {
            std::fill(accum.begin(), accum.end(), 0.0);
            std::fill(counts.begin(), counts.end(), 0);

            // Assignment + accumulation.
            for (size_t p = 0; p < n; ++p) {
                const float* sub = data + p * dim + chunk_off;
                float best = std::numeric_limits<float>::max();
                uint32_t best_k = 0;
                for (uint32_t k = 0; k < K; ++k) {
                    float d = l2_distance_squared(sub, cent.data() + static_cast<size_t>(k) * sub_dim, sub_dim);
                    if (d < best) { best = d; best_k = k; }
                }
                counts[best_k]++;
                double* acc = accum.data() + static_cast<size_t>(best_k) * sub_dim;
                for (uint32_t d = 0; d < sub_dim; ++d) acc[d] += sub[d];
            }

            // Update centroids; reseed empty clusters.
            for (uint32_t k = 0; k < K; ++k) {
                if (counts[k] > 0) {
                    double inv = 1.0 / static_cast<double>(counts[k]);
                    const double* acc = accum.data() + static_cast<size_t>(k) * sub_dim;
                    float* c = cent.data() + static_cast<size_t>(k) * sub_dim;
                    for (uint32_t d = 0; d < sub_dim; ++d) {
                        c[d] = static_cast<float>(acc[d] * inv);
                    }
                } else {
                    // Empty cluster: reseed to a random point's sub-vector.
                    size_t idx = pick(rng);
                    const float* src = data + idx * dim + chunk_off;
                    std::memcpy(cent.data() + static_cast<size_t>(k) * sub_dim, src,
                                sub_dim * sizeof(float));
                }
            }
        }

        // Store into out.centroids ([M][K][sub_dim]).
        std::memcpy(out.centroids.data() + (static_cast<size_t>(m) * K) * sub_dim,
                    cent.data(),
                    static_cast<size_t>(K) * sub_dim * sizeof(float));
    }

    out.build_sym_table();
}

}  // namespace ann
