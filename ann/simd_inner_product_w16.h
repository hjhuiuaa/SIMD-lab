// simd_inner_product_w16.h — NEON 每次循环处理 16 个 float（四路 float32x4）
// 尾部交给 simd_inner_product_w8.h（8→4→标量）

#pragma once

#include <cstddef>

#include "simd_inner_product_w8.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace simd_f32_w16 {

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

inline float horizontal_sum_f32x4(float32x4_t v) {
    float32x2_t lo = vget_low_f32(v);
    float32x2_t hi = vget_high_f32(v);
    float32x2_t s2 = vadd_f32(lo, hi);
    float32x2_t s1 = vpadd_f32(s2, s2);
    return vget_lane_f32(s1, 0);
}

inline float inner_product_neon_w16(const float* a, const float* b, std::size_t n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    std::size_t i = 0;

    for (; i + 16 <= n; i += 16) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t va2 = vld1q_f32(a + i + 8);
        float32x4_t va3 = vld1q_f32(a + i + 12);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        float32x4_t vb2 = vld1q_f32(b + i + 8);
        float32x4_t vb3 = vld1q_f32(b + i + 12);
#if defined(__ARM_FEATURE_FMA) && (__ARM_FEATURE_FMA == 1)
        acc0 = vfmaq_f32(acc0, va0, vb0);
        acc1 = vfmaq_f32(acc1, va1, vb1);
        acc2 = vfmaq_f32(acc2, va2, vb2);
        acc3 = vfmaq_f32(acc3, va3, vb3);
#else
        acc0 = vmlaq_f32(acc0, va0, vb0);
        acc1 = vmlaq_f32(acc1, va1, vb1);
        acc2 = vmlaq_f32(acc2, va2, vb2);
        acc3 = vmlaq_f32(acc3, va3, vb3);
#endif
    }

    float sum = horizontal_sum_f32x4(acc0) + horizontal_sum_f32x4(acc1)
              + horizontal_sum_f32x4(acc2) + horizontal_sum_f32x4(acc3);
    if (i < n) {
        sum += simd_f32_w8::inner_product_neon_w8(a + i, b + i, n - i);
    }
    return sum;
}

#else

inline float inner_product_neon_w16(const float* a, const float* b, std::size_t n) {
    return simd_f32_w8::inner_product_neon_w8(a, b, n);
}

#endif

} // namespace simd_f32_w16
