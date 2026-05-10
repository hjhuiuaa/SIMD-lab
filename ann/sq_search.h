// sq_search.h — 标量量化 SQ：逐库向量 int8 + 行尺度，内积近似 dot(q,x) ≈ scale[i]*dot(q,qx_i)

#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <queue>
#include <algorithm>

#include "simd_float_int8_dot.h"

struct SqIndex {
    std::size_t n = 0;
    std::size_t d = 0;
    std::vector<int8_t> codes;   // n * d 行主序
    std::vector<float> row_scale; // 每行一个 scale
};

inline void sq_build_from_base(const float* base, std::size_t base_number, std::size_t vecdim, SqIndex& out)
{
    out.n = base_number;
    out.d = vecdim;
    out.codes.resize(base_number * vecdim);
    out.row_scale.resize(base_number);

    for (std::size_t i = 0; i < base_number; ++i) {
        const float* row = base + i * vecdim;
        int8_t* dst = &out.codes[i * vecdim];
        float amax = 1e-8f;
        for (std::size_t j = 0; j < vecdim; ++j) {
            amax = std::max(amax, std::fabs(row[j]));
        }
        float s = amax / 127.0f;
        if (s < 1e-12f) {
            out.row_scale[i] = 1.0f;
            std::fill(dst, dst + vecdim, 0);
            continue;
        }
        out.row_scale[i] = s;
        for (std::size_t j = 0; j < vecdim; ++j) {
            int v = static_cast<int>(std::lrint(row[j] / s));
            if (v > 127) {
                v = 127;
            }
            if (v < -127) {
                v = -127;
            }
            dst[j] = static_cast<int8_t>(v);
        }
    }
}

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_sq_simd(
    const SqIndex& idx,
    float* query,
    std::size_t base_number,
    std::size_t vecdim,
    std::size_t k)
{
    (void)base_number;
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (std::size_t i = 0; i < idx.n; ++i) {
        const int8_t* row = &idx.codes[i * vecdim];
        float ip = idx.row_scale[i] * simd_mix::dot_float_int8_neon(query, row, vecdim);
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
