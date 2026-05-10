// flat_search_simd.h — IP 距离 top-k，内积使用 simd_inner_product.h
// 原始 flat_scan.h 禁止修改；评测时在 main.cc 中改为包含本文件并调用 flat_search_simd

#pragma once

#include <queue>
#include <cstdint>
#include <cstddef>

#include "simd_inner_product.h"

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_simd(
    float* base,
    float* query,
    std::size_t base_number,
    std::size_t vecdim,
    std::size_t k)
{
    std::priority_queue<std::pair<float, uint32_t>> q;

    for (std::size_t i = 0; i < base_number; ++i) {
        const float* row = base + i * vecdim;
        float ip = simd_f32::inner_product_neon(row, query, vecdim);
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
