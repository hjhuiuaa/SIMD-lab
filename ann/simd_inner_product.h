// simd_inner_product.h — aarch64 NEON 内积封装（鲲鹏 / ARMv8）
// 与实验手册「先封装 SIMD 再做 flat」一致；不修改 flat_scan.h

#pragma once

#include <cstddef>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace simd_f32 {

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

inline float horizontal_sum_f32x4(float32x4_t v) {
    float32x2_t lo = vget_low_f32(v);
    float32x2_t hi = vget_high_f32(v);
    float32x2_t s2 = vadd_f32(lo, hi);
    float32x2_t s1 = vpadd_f32(s2, s2);
    return vget_lane_f32(s1, 0);
}

// sum_i a[i]*b[i]，主循环 4 路 FMA，尾部标量
inline float inner_product_neon(const float* a, const float* b, std::size_t n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    std::size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
#if defined(__ARM_FEATURE_FMA) && (__ARM_FEATURE_FMA == 1)
        acc = vfmaq_f32(acc, va, vb);
#else
        acc = vmlaq_f32(acc, va, vb);
#endif
    }

    float sum = horizontal_sum_f32x4(acc);
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

#else

inline float inner_product_neon(const float* a, const float* b, std::size_t n) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

#endif

} // namespace simd_f32
