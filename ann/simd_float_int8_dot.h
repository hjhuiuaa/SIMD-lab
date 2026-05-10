// simd_float_int8_dot.h — float 查询向量 · int8 数据库向量（SQ 距离核）

#pragma once

#include <cstddef>
#include <cmath>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace simd_mix {

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

inline float horizontal_sum_f32x4(float32x4_t v) {
    float32x2_t lo = vget_low_f32(v);
    float32x2_t hi = vget_high_f32(v);
    float32x2_t s2 = vadd_f32(lo, hi);
    float32x2_t s1 = vpadd_f32(s2, s2);
    return vget_lane_f32(s1, 0);
}

// sum_j q[j] * (float)xq[j]，xq 为 int8
inline float dot_float_int8_neon(const float* q, const int8_t* xq, std::size_t n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    std::size_t j = 0;

    for (; j + 8 <= n; j += 8) {
        float32x4_t q0 = vld1q_f32(q + j);
        float32x4_t q1 = vld1q_f32(q + j + 4);
        int8x8_t xv = vld1_s8(xq + j);
        int16x8_t xw = vmovl_s8(xv);
        float32x4_t xf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(xw)));
        float32x4_t xf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(xw)));
#if defined(__ARM_FEATURE_FMA) && (__ARM_FEATURE_FMA == 1)
        acc = vfmaq_f32(acc, q0, xf0);
        acc = vfmaq_f32(acc, q1, xf1);
#else
        acc = vmlaq_f32(acc, q0, xf0);
        acc = vmlaq_f32(acc, q1, xf1);
#endif
    }

    float sum = horizontal_sum_f32x4(acc);
    for (; j < n; ++j) {
        sum += q[j] * static_cast<float>(xq[j]);
    }
    return sum;
}

#else

inline float dot_float_int8_neon(const float* q, const int8_t* xq, std::size_t n) {
    float s = 0.0f;
    for (std::size_t j = 0; j < n; ++j) {
        s += q[j] * static_cast<float>(xq[j]);
    }
    return s;
}

#endif

} // namespace simd_mix
