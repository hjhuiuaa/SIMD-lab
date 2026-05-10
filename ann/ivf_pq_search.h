// ivf_pq_search.h — IVF（倒排）+ PQ：粗聚类缩小候选，再用 PQ-ADC 排序
// 粗量化：全维 L2 k-means（nlist 个 centroid）；查询：nprobe 个最近质心列表并集 → subset PQ

#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "pq_search.h"

struct IvfPqIndex {
    PqIndex pq;
    std::size_t nlist = 0;
    std::size_t d = 0;
    std::vector<float> coarse_centroids;              // nlist * d
    std::vector<std::vector<uint32_t>> inverted_lists; // size nlist
};

namespace ivf_detail {

inline float l2sq_full(const float* a, const float* b, std::size_t d)
{
    return pq_detail::l2sq_sub(a, b, d);
}

// 全库 L2 k-means，得到 coarse_centroids 与 assign[i] ∈ [0,nlist)
inline void kmeans_l2_coarse(
    const float* base,
    std::size_t n,
    std::size_t d,
    std::size_t nlist,
    int max_iter,
    uint32_t seed,
    std::vector<float>& centroids,
    std::vector<uint32_t>& assign)
{
    centroids.resize(nlist * d);
    assign.resize(n);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::size_t> uni(0, n - 1);

    for (std::size_t j = 0; j < nlist; ++j) {
        std::size_t pick = uni(rng);
        std::memcpy(&centroids[j * d], base + pick * d, d * sizeof(float));
    }

    std::vector<std::size_t> counts(nlist);
    std::vector<float> accum(nlist * d);

    for (int it = 0; it < max_iter; ++it) {
        counts.assign(nlist, 0);
        accum.assign(nlist * d, 0.0f);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
        for (long long ii = 0; ii < static_cast<long long>(n); ++ii) {
            const std::size_t i = static_cast<std::size_t>(ii);
            const float* v = base + i * d;
            float best = std::numeric_limits<float>::max();
            std::size_t bj = 0;
            for (std::size_t j = 0; j < nlist; ++j) {
                const float t = l2sq_full(v, &centroids[j * d], d);
                if (t < best) {
                    best = t;
                    bj = j;
                }
            }
            assign[i] = static_cast<uint32_t>(bj);
        }
#else
        for (std::size_t i = 0; i < n; ++i) {
            const float* v = base + i * d;
            float best = std::numeric_limits<float>::max();
            std::size_t bj = 0;
            for (std::size_t j = 0; j < nlist; ++j) {
                const float t = l2sq_full(v, &centroids[j * d], d);
                if (t < best) {
                    best = t;
                    bj = j;
                }
            }
            assign[i] = static_cast<uint32_t>(bj);
        }
#endif

        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t j = assign[i];
            counts[j]++;
            float* acc = &accum[j * d];
            const float* v = base + i * d;
            for (std::size_t t = 0; t < d; ++t) {
                acc[t] += v[t];
            }
        }

        for (std::size_t j = 0; j < nlist; ++j) {
            if (counts[j] == 0) {
                std::size_t pick = uni(rng);
                std::memcpy(&centroids[j * d], base + pick * d, d * sizeof(float));
                continue;
            }
            const float inv = 1.0f / static_cast<float>(counts[j]);
            float* dst = &centroids[j * d];
            const float* src = &accum[j * d];
            for (std::size_t t = 0; t < d; ++t) {
                dst[t] = src[t] * inv;
            }
        }
    }
}

} // namespace ivf_detail

inline void ivf_pq_build_from_base(
    const float* base,
    std::size_t base_number,
    std::size_t vecdim,
    IvfPqIndex& out,
    std::size_t nlist,
    int coarse_kmeans_iter = 12,
    std::size_t pq_m = 8,
    std::size_t pq_ks = 256,
    int pq_kmeans_iter = 15,
    uint32_t seed = 424242u)
{
    out.nlist = nlist;
    out.d = vecdim;
    out.inverted_lists.assign(nlist, {});

    std::vector<uint32_t> assign;
    ivf_detail::kmeans_l2_coarse(
        base,
        base_number,
        vecdim,
        nlist,
        coarse_kmeans_iter,
        seed,
        out.coarse_centroids,
        assign);

    for (std::size_t i = 0; i < base_number; ++i) {
        const std::size_t j = assign[i];
        out.inverted_lists[j].push_back(static_cast<uint32_t>(i));
    }

    pq_build_from_base(
        base,
        base_number,
        vecdim,
        out.pq,
        pq_m,
        pq_ks,
        pq_kmeans_iter,
        seed + 17u);
}

// IVF 候选 id：nprobe 个最近粗质心对应倒排链并集（不排序）。
// 说明：此处 query–粗质心为全维 L²，当前标量实现；PQ-ADC 阶段见 pq_search.h 的 LUT（NEON+OpenMP）。
inline std::vector<uint32_t> ivf_pq_collect_candidates(
    const IvfPqIndex& idx,
    float* query,
    std::size_t nprobe)
{
    std::vector<uint32_t> cand;
    std::size_t np = nprobe;
    if (np > idx.nlist) {
        np = idx.nlist;
    }
    if (np == 0) {
        np = 1;
    }

    const std::size_t nlist = idx.nlist;
    const std::size_t d = idx.d;

    std::vector<std::pair<float, std::size_t>> dist_idx(nlist);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
    for (long long jj = 0; jj < static_cast<long long>(nlist); ++jj) {
        const std::size_t j = static_cast<std::size_t>(jj);
        const float t = ivf_detail::l2sq_full(query, &idx.coarse_centroids[j * d], d);
        dist_idx[j] = {t, j};
    }
#else
    for (std::size_t j = 0; j < nlist; ++j) {
        const float t = ivf_detail::l2sq_full(query, &idx.coarse_centroids[j * d], d);
        dist_idx[j] = {t, j};
    }
#endif

    std::partial_sort(
        dist_idx.begin(),
        dist_idx.begin() + static_cast<std::ptrdiff_t>(np),
        dist_idx.end(),
        [](const std::pair<float, std::size_t>& a, const std::pair<float, std::size_t>& b) {
            return a.first < b.first;
        });

    std::size_t cand_cap = 0;
    for (std::size_t t = 0; t < np; ++t) {
        const std::size_t list_id = dist_idx[t].second;
        cand_cap += idx.inverted_lists[list_id].size();
    }
    cand.reserve(cand_cap);
    for (std::size_t t = 0; t < np; ++t) {
        const std::size_t list_id = dist_idx[t].second;
        const auto& L = idx.inverted_lists[list_id];
        cand.insert(cand.end(), L.begin(), L.end());
    }
    return cand;
}

// IVF-PQ：query 到各 coarse centroid 的 L2，取 nprobe 最小；候选 = 并集倒排链；PQ-ADC Top-K
inline std::priority_queue<std::pair<float, uint32_t>> ivf_pq_search_adc(
    const IvfPqIndex& idx,
    float* query,
    std::size_t k,
    std::size_t nprobe)
{
    std::vector<uint32_t> cand = ivf_pq_collect_candidates(idx, query, nprobe);

    if (cand.empty()) {
        return std::priority_queue<std::pair<float, uint32_t>>();
    }

    if (cand.size() < k) {
        return flat_search_pq_adc_simd(idx.pq, query, idx.pq.n, idx.d, k);
    }

    return flat_search_pq_adc_simd_subset(idx.pq, query, cand.data(), cand.size(), k);
}
