#pragma once

#include <cstdint>
#include <cmath>

// SIMD support detection: the x86 path uses AVX2 loads and FMA (_mm256_fmadd_ps), so it needs both.
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define ANN_USE_AVX2 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define ANN_USE_NEON 1
#endif

namespace ann {

#ifdef ANN_USE_AVX2
/**
 * AVX2-optimized squared L2 distance.
 * Processes 8 floats at a time.
 */
inline float l2_distance_squared_avx2(const float* a, const float* b, uint32_t dim) {
    __m256 sum = _mm256_setzero_ps();

    uint32_t i = 0;
    // Process 8 floats at a time
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);  // FMA: sum += diff * diff
    }

    // Horizontal sum of 8 floats in sum
    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 sum128 = _mm_add_ps(sum_low, sum_high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);

    // Handle remainder
    for (; i < dim; i++) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }

    return result;
}
#endif

#ifdef ANN_USE_NEON
/**
 * ARM NEON-optimized squared L2 distance.
 * Processes 4 floats at a time.
 */
inline float l2_distance_squared_neon(const float* a, const float* b, uint32_t dim) {
    float32x4_t sum = vdupq_n_f32(0.0f);

    uint32_t i = 0;
    // Process 4 floats at a time
    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        sum = vmlaq_f32(sum, diff, diff);  // FMA: sum += diff * diff
    }

    // Horizontal sum of 4 floats
    float32x2_t sum2 = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    sum2 = vpadd_f32(sum2, sum2);
    float result = vget_lane_f32(sum2, 0);

    // Handle remainder
    for (; i < dim; i++) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }

    return result;
}
#endif

/**
 * L2 (Euclidean) distance between two vectors.
 */
inline float l2_distance(const float* a, const float* b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

/**
 * Squared L2 distance (faster, no sqrt).
 * Uses SIMD if available (AVX2 on x86, NEON on ARM), falls back to scalar otherwise.
 */
inline float l2_distance_squared(const float* a, const float* b, uint32_t dim) {
#ifdef ANN_USE_AVX2
    return l2_distance_squared_avx2(a, b, dim);
#elif defined(ANN_USE_NEON)
    return l2_distance_squared_neon(a, b, dim);
#else
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
#endif
}

/**
 * Scalar version of squared L2 distance (for comparison).
 */
inline float l2_distance_squared_scalar(const float* a, const float* b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Function pointer type for distance functions
using DistanceFunc = float (*)(const float*, const float*, uint32_t);

//=============================================================================
// Unified Distance Function (compile-time selection)
//=============================================================================

/**
 * Compute distance between two vectors: squared L2, the only metric. Cosine
 * data is normalised to unit length offline (tools/prepare/normalize_vectors.py),
 * where squared L2 = 2 - 2 cos, so the L2 order is the cosine order and PQ (whose
 * distances are squared L2) shares one scale with the exact distances.
 *
 * Note: This function is used in the search algorithms and is the main entry point
 * for distance computation in the distributed system.
 */
inline float compute_distance(const float* a, const float* b, uint32_t dim) {
    return l2_distance_squared(a, b, dim);
}

} // namespace ann
