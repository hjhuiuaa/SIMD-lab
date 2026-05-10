// flat_search_simd_w16.h — Flat + IP，内积使用 16 路展开（simd_inner_product_w16.h）

#pragma once

#include <queue>
#include <cstdint>
#include <cstddef>

#include "simd_inner_product_w16.h"

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_simd_w16(
    float* base,
    float* query,
    std::size_t base_number,
    std::size_t vecdim,
    std::size_t k)
{
    std::priority_queue<std::pair<float, uint32_t>> q;

    for (std::size_t i = 0; i < base_number; ++i) {
        const float* row = base + i * vecdim;
        float ip = simd_f32_w16::inner_product_neon_w16(row, query, vecdim);
        float dis = 1.0f - ip;

        if (q.size() < k) {
            q.push({dis, static_cast<uint32_t>(i)});
        } else if (dis < q.top().first) {
            q.push({dis, static_cast<uint32_t>(i)});
            q.pop();
        }
    }
    return q;
}
