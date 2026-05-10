// simd_inner_product_w8.h — NEON 每次循环处理 8 个 float（双 float32x4）
// 与 simd_inner_product.h（4 路）对照实验用；尾部调用 4 路实现收尾

#pragma once

#include <cstddef>

#include "simd_inner_product.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace simd_f32_w8 {

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

inline float horizontal_sum_f32x4(float32x4_t v) {
    float32x2_t lo = vget_low_f32(v);
    float32x2_t hi = vget_high_f32(v);
    float32x2_t s2 = vadd_f32(lo, hi);
    float32x2_t s1 = vpadd_f32(s2, s2);
    return vget_lane_f32(s1, 0);
}

// sum_j a[j]*b[j]：主循环每次 8 元组；剩余长度交给 simd_f32::inner_product_neon（4 路+标量）
inline float inner_product_neon_w8(const float* a, const float* b, std::size_t n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    std::size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
#if defined(__ARM_FEATURE_FMA) && (__ARM_FEATURE_FMA == 1)
        acc0 = vfmaq_f32(acc0, va0, vb0);
        acc1 = vfmaq_f32(acc1, va1, vb1);
#else
        acc0 = vmlaq_f32(acc0, va0, vb0);
        acc1 = vmlaq_f32(acc1, va1, vb1);
#endif
    }

    float sum = horizontal_sum_f32x4(acc0) + horizontal_sum_f32x4(acc1);
    if (i < n) {
        sum += simd_f32::inner_product_neon(a + i, b + i, n - i);
    }
    return sum;
}

#else

inline float inner_product_neon_w8(const float* a, const float* b, std::size_t n) {
    return simd_f32::inner_product_neon(a, b, n);
}

#endif

} // namespace simd_f32_w8
