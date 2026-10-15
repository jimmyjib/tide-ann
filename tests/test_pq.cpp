// Product quantization: file round trips, training, ADC and code-to-code distances, decoding.
#include "pq.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#define CHECK(expr) do { if (!(expr)) { std::cerr << "CHECK failed at line " << __LINE__ << ": " #expr "\n"; return 1; } } while (0)

namespace {

constexpr uint32_t kDim = 32, kM = 8, kRows = 4096, kClusters = 16;

// A Gaussian mixture: kClusters centers N(0, 4^2) (fixed seed, shared by the
// base and the queries), points center + N(0, 1) drawn with `seed`.
std::vector<float> mixture(uint32_t rows, uint32_t seed) {
    std::mt19937 center_rng(7);
    std::normal_distribution<float> center(0.0f, 4.0f), noise(0.0f, 1.0f);
    std::vector<float> centers(kClusters * kDim);
    for (auto& value : centers) value = center(center_rng);
    std::mt19937 rng(seed);
    std::vector<float> data(static_cast<size_t>(rows) * kDim);
    std::uniform_int_distribution<uint32_t> pick(0, kClusters - 1);
    for (uint32_t i = 0; i < rows; ++i) {
        const uint32_t c = pick(rng);
        for (uint32_t d = 0; d < kDim; ++d) data[static_cast<size_t>(i) * kDim + d] = centers[c * kDim + d] + noise(rng);
    }
    return data;
}

double relative(double a, double b) { return std::fabs(a - b) / std::max(1e-12, std::fabs(b)); }

std::string temp_path(const char* leaf) {
    return "/tmp/test_pq_" + std::to_string(getpid()) + "_" + leaf;
}

int train_and_round_trip(const std::vector<float>& data, ann::PQTable& pq) {
    ann::train_pq(data.data(), kRows, kDim, kM, ann::PQTable::K, 10, pq);
    CHECK(pq.dim == kDim && pq.M == kM && pq.sub_dim == kDim / kM);
    CHECK(pq.centroids.size() == static_cast<size_t>(kM) * ann::PQTable::K * pq.sub_dim);

    // Deterministic training: a second run gives bit-identical pivots.
    ann::PQTable again;
    ann::train_pq(data.data(), kRows, kDim, kM, ann::PQTable::K, 10, again);
    CHECK(again.centroids == pq.centroids && again.pivots_hash() == pq.pivots_hash());

    // dim % M != 0 and K != 256 are rejected.
    bool threw = false;
    try { ann::PQTable bad; ann::train_pq(data.data(), kRows, kDim, 5, ann::PQTable::K, 1, bad); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { ann::PQTable bad; ann::train_pq(data.data(), kRows, kDim, kM, 128, 1, bad); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    // Pivots file: save -> load is exact and rebuilds the same table.
    const std::string path = temp_path("pivots.bin");
    pq.save(path);
    ann::PQTable loaded;
    CHECK(loaded.load(path));
    CHECK(loaded.dim == pq.dim && loaded.M == pq.M && loaded.centroids == pq.centroids &&
          loaded.sym_table == pq.sym_table && loaded.pivots_hash() == pq.pivots_hash());

    // A wrong magic, a wrong K and a truncated file are rejected.
    std::vector<char> bytes;
    {
        std::ifstream in(path, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    auto write = [&](const std::vector<char>& content) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(content.data(), static_cast<std::streamsize>(content.size()));
    };
    auto bad_magic = bytes; bad_magic[0] ^= 0x5A; write(bad_magic);
    CHECK(!ann::PQTable().load(path));
    auto bad_k = bytes; bad_k[12] = static_cast<char>(128); bad_k[13] = 0; write(bad_k);
    CHECK(!ann::PQTable().load(path));
    auto truncated = bytes; truncated.resize(bytes.size() - 7); write(truncated);
    CHECK(!ann::PQTable().load(path));
    std::remove(path.c_str());

    // The hash names the pivots: one changed centroid value changes it.
    ann::PQTable changed = pq;
    changed.centroids[17] += 1.0f;
    CHECK(changed.pivots_hash() != pq.pivots_hash());
    return 0;
}

int tables_and_adc(const std::vector<float>& data, const ann::PQTable& pq) {
    // Code-to-code table: symmetric with a zero diagonal.
    const uint32_t K = ann::PQTable::K;
    for (uint32_t m = 0; m < pq.M; ++m) {
        const float* table = pq.sym_table.data() + static_cast<size_t>(m) * K * K;
        for (uint32_t i = 0; i < K; ++i) {
            CHECK(table[static_cast<size_t>(i) * K + i] == 0.0f);
            for (uint32_t j = 0; j < K; ++j)
                CHECK(table[static_cast<size_t>(i) * K + j] == table[static_cast<size_t>(j) * K + i]);
        }
    }

    std::vector<uint8_t> codes(static_cast<size_t>(kRows) * pq.M);
    for (uint32_t i = 0; i < kRows; ++i) pq.encode(data.data() + static_cast<size_t>(i) * kDim, codes.data() + i * pq.M);

    const auto queries = mixture(64, 99);
    std::vector<float> table(static_cast<size_t>(pq.M) * K), decoded_a(kDim), decoded_b(kDim);
    double overlap_sum = 0.0;
    for (uint32_t q = 0; q < 64; ++q) {
        const float* query = queries.data() + static_cast<size_t>(q) * kDim;
        pq.build_query_table(query, table.data());
        std::vector<std::pair<float, uint32_t>> exact(kRows), adc(kRows);
        for (uint32_t i = 0; i < kRows; ++i) {
            const uint8_t* code = codes.data() + i * pq.M;
            const float with_table = pq.adc_distance(table.data(), code);
            const float direct = pq.adc_distance_direct(query, code);
            // ADC through the table agrees with the direct form within 1e-5 relative.
            CHECK(relative(with_table, direct) <= 1e-5);
            // ADC is the exact distance to the code's reconstruction.
            pq.decode(code, decoded_a.data());
            CHECK(relative(direct, ann::l2_distance_squared_scalar(query, decoded_a.data(), kDim)) <= 1e-5);
            exact[i] = {ann::l2_distance_squared(query, data.data() + static_cast<size_t>(i) * kDim, kDim), i};
            adc[i] = {with_table, i};
        }
        // The code-to-code distance is the distance between the reconstructions.
        const uint8_t* a = codes.data() + (q * 37 % kRows) * pq.M;
        const uint8_t* b = codes.data() + (q * 91 % kRows) * pq.M;
        pq.decode(a, decoded_a.data());
        pq.decode(b, decoded_b.data());
        const double reconstructed = ann::l2_distance_squared_scalar(decoded_a.data(), decoded_b.data(), kDim);
        CHECK(std::fabs(pq.sym_distance(a, b) - reconstructed) <= 1e-5 * std::max(1.0, reconstructed));

        std::partial_sort(exact.begin(), exact.begin() + 10, exact.end());
        std::partial_sort(adc.begin(), adc.begin() + 100, adc.end());
        uint32_t hits = 0;
        for (uint32_t k = 0; k < 10; ++k) {
            for (uint32_t j = 0; j < 100; ++j) {
                if (adc[j].second == exact[k].second) { ++hits; break; }
            }
        }
        overlap_sum += hits / 10.0;
    }
    const double overlap = overlap_sum / 64.0;
    std::cout << "exact top-10 within ADC top-100: " << overlap << "\n";
    CHECK(overlap >= 0.8);
    return 0;
}

int codes_file(const std::vector<float>& data, const ann::PQTable& pq) {
    const uint32_t n = 100;
    std::vector<uint8_t> codes(static_cast<size_t>(n) * pq.M);
    for (uint32_t i = 0; i < n; ++i) pq.encode(data.data() + static_cast<size_t>(i) * kDim, codes.data() + i * pq.M);
    const std::string path = temp_path("codes.bin");
    ann::write_pq_codes(path, codes, n, pq.M, pq.pivots_hash());
    CHECK(ann::read_pq_codes(path, n, pq.M, pq.pivots_hash()) == codes);
    auto refused = [&](uint32_t expected_n, uint32_t expected_m, uint64_t hash, const char* what) {
        try { ann::read_pq_codes(path, expected_n, expected_m, hash); }
        catch (const std::runtime_error& e) { return std::string(e.what()).find(what) != std::string::npos; }
        return false;
    };
    CHECK(refused(n + 1, pq.M, pq.pivots_hash(), "row count"));
    CHECK(refused(n, pq.M * 2, pq.pivots_hash(), "code length"));
    CHECK(refused(n, pq.M, pq.pivots_hash() ^ 1, "other pivots"));
    {
        std::ofstream out(path, std::ios::binary | std::ios::app);
        out.put('x');
    }
    CHECK(refused(n, pq.M, pq.pivots_hash(), "size"));
    std::remove(path.c_str());
    bool threw = false;
    try { ann::write_pq_codes(path, codes, n + 1, pq.M, 0); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    return 0;
}

}  // namespace

int main() {
    const auto data = mixture(kRows, 11);
    ann::PQTable pq;
    if (train_and_round_trip(data, pq) != 0) return 1;
    if (tables_and_adc(data, pq) != 0) return 1;
    if (codes_file(data, pq) != 0) return 1;
    std::cout << "pq tests passed\n";
    return 0;
}
