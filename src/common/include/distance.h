#pragma once

#include <cstdint>
#include <cmath>

// The x86 path needs both AVX2 and FMA.
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define ANN_USE_AVX2 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define ANN_USE_NEON 1
#endif

namespace ann {

#ifdef ANN_USE_AVX2
inline float l2_distance_squared_avx2(const float* a, const float* b, uint32_t dim) {
    __m256 sum = _mm256_setzero_ps();

    uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 sum128 = _mm_add_ps(sum_low, sum_high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);

    for (; i < dim; i++) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }

    return result;
}
#endif

#ifdef ANN_USE_NEON
inline float l2_distance_squared_neon(const float* a, const float* b, uint32_t dim) {
    float32x4_t sum = vdupq_n_f32(0.0f);

    uint32_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        sum = vmlaq_f32(sum, diff, diff);
    }

    float32x2_t sum2 = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    sum2 = vpadd_f32(sum2, sum2);
    float result = vget_lane_f32(sum2, 0);

    for (; i < dim; i++) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }

    return result;
}
#endif

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

// Scalar reference version.
inline float l2_distance_squared_scalar(const float* a, const float* b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Squared L2 only. Cosine data is normalized offline (tools/prepare/normalize_vectors.py).
inline float compute_distance(const float* a, const float* b, uint32_t dim) {
    return l2_distance_squared(a, b, dim);
}

} // namespace ann
