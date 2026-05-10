// top_r_ip_rerank.h — PQ-ADC 在候选集中取 Top-R，再用全精度 IP（NEON w8）重排得 Top-k
// 与 GT 一致：dis = 1 - dot(q, x)

#pragma once

#include <algorithm>
#include <cstdint>
#include <queue>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "ivf_pq_search.h"
#include "flat_search_simd.h"
#include "simd_inner_product.h"
#include "sq_search.h"

namespace top_r_ip_rerank {

// ---------- PQ-SDC 粗排（对称码本内积表），供两阶段 SDC → 全精度 IP ----------

inline float pq_dis_from_sdc_qcode(
    const PqIndex& idx,
    const uint8_t* qcode,
    std::size_t vec_id)
{
    const float* tab = idx.sdc_ip.data();
    float ip = 0.0f;
    const uint8_t* cx = &idx.codes[vec_id * idx.m];
    for (std::size_t s = 0; s < idx.m; ++s) {
        const std::size_t qi = qcode[s];
        const std::size_t xi = cx[s];
        ip += tab[((s * idx.ks) + qi) * idx.ks + xi];
    }
    return 1.0f - ip;
}

// 已得到查询 PQ 码时，在 cand 上用 SDC 分数保留 Top-R（大顶堆，堆顶最差）
inline std::vector<uint32_t> pq_select_top_r_among_candidates_sdc_from_qcode(
    const PqIndex& idx,
    const uint8_t* qcode,
    const uint32_t* cand,
    std::size_t ncand,
    std::size_t R)
{
    if (ncand == 0 || idx.sdc_ip.size() != idx.m * idx.ks * idx.ks) {
        return {};
    }
    const std::size_t cap = std::min(R, ncand);
    std::priority_queue<std::pair<float, uint32_t>> heap;
    for (std::size_t t = 0; t < ncand; ++t) {
        const std::size_t vid = static_cast<std::size_t>(cand[t]);
        const float dis = pq_dis_from_sdc_qcode(idx, qcode, vid);
        if (heap.size() < cap) {
            heap.push({dis, static_cast<uint32_t>(vid)});
        } else if (dis < heap.top().first) {
            heap.push({dis, static_cast<uint32_t>(vid)});
            heap.pop();
        }
    }
    std::vector<uint32_t> out;
    out.reserve(heap.size());
    while (!heap.empty()) {
        out.push_back(heap.top().second);
        heap.pop();
    }
    return out;
}

inline std::vector<uint32_t> pq_select_top_r_among_candidates_sdc(
    const PqIndex& idx,
    float* query,
    std::size_t vecdim,
    const uint32_t* cand,
    std::size_t ncand,
    std::size_t R)
{
    std::vector<uint8_t> qcode(idx.m);
    pq_detail::pq_encode_row(
        query,
        vecdim,
        idx.m,
        idx.ks,
        idx.sub,
        idx.codebooks.data(),
        qcode.data());
    return pq_select_top_r_among_candidates_sdc_from_qcode(idx, qcode.data(), cand, ncand, R);
}

inline void pq_build_adc_lut(const PqIndex& idx, float* query, std::vector<float>& lut)
{
    const std::size_t lut_sz = idx.m * idx.ks;
    lut.resize(lut_sz);

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 64)
    for (long long sc = 0; sc < static_cast<long long>(lut_sz); ++sc) {
        const std::size_t s = static_cast<std::size_t>(sc) / idx.ks;
        const std::size_t c = static_cast<std::size_t>(sc) % idx.ks;
        const float* qseg = query + s * idx.sub;
        const float* cb = &idx.codebooks[((s * idx.ks) + c) * idx.sub];
        lut[static_cast<std::size_t>(sc)] = simd_f32::inner_product_neon(qseg, cb, idx.sub);
    }
#else
    for (std::size_t s = 0; s < idx.m; ++s) {
        const float* qseg = query + s * idx.sub;
        for (std::size_t c = 0; c < idx.ks; ++c) {
            const float* cb = &idx.codebooks[((s * idx.ks) + c) * idx.sub];
            lut[s * idx.ks + c] = simd_f32::inner_product_neon(qseg, cb, idx.sub);
        }
    }
#endif
}

inline float pq_dis_from_lut(const PqIndex& idx, const std::vector<float>& lut, std::size_t vec_id)
{
    float ip = 0.0f;
    for (std::size_t s = 0; s < idx.m; ++s) {
        const uint8_t code = idx.codes[vec_id * idx.m + s];
        ip += lut[s * idx.ks + code];
    }
    return 1.0f - ip;
}

// 在 cand 中用 PQ-ADC 分数维护大顶堆，保留「距离最小」的 R 个 id（堆顶为当前最差）
inline std::vector<uint32_t> pq_select_top_r_among_candidates(
    const PqIndex& idx,
    float* query,
    const uint32_t* cand,
    std::size_t ncand,
    std::size_t R)
{
    if (ncand == 0) {
        return {};
    }
    const std::size_t cap = std::min(R, ncand);

    std::vector<float> lut;
    pq_build_adc_lut(idx, query, lut);

    std::priority_queue<std::pair<float, uint32_t>> heap;
    for (std::size_t t = 0; t < ncand; ++t) {
        const std::size_t vid = static_cast<std::size_t>(cand[t]);
        const float dis = pq_dis_from_lut(idx, lut, vid);
        if (heap.size() < cap) {
            heap.push({dis, static_cast<uint32_t>(vid)});
        } else if (dis < heap.top().first) {
            heap.push({dis, static_cast<uint32_t>(vid)});
            heap.pop();
        }
    }

    std::vector<uint32_t> out;
    out.reserve(heap.size());
    while (!heap.empty()) {
        out.push_back(heap.top().second);
        heap.pop();
    }
    return out;
}

// 全精度 IP Top-k（与 flat_scan 度量一致，内积用 w8）
inline std::priority_queue<std::pair<float, uint32_t>> ip_rerank_topk(
    float* base,
    float* query,
    std::size_t vecdim,
    const uint32_t* ids,
    std::size_t nids,
    std::size_t k)
{
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (std::size_t t = 0; t < nids; ++t) {
        const std::size_t i = static_cast<std::size_t>(ids[t]);
        const float* row = base + i * vecdim;
        const float ip = simd_f32::inner_product_neon(row, query, vecdim);
        const float dis = 1.0f - ip;
        if (q.size() < k) {
            q.push({dis, static_cast<uint32_t>(i)});
        } else if (dis < q.top().first) {
            q.push({dis, static_cast<uint32_t>(i)});
            q.pop();
        }
    }
    return q;
}

// ---------- SQ：全库 SQ-SIMD 粗排 Top-p → 全精度 IP Top-k（与指导书两阶段思路一致）----------

inline std::priority_queue<std::pair<float, uint32_t>> sq_fullscan_top_p_ip_rerank(
    const SqIndex& idx,
    float* base,
    float* query,
    std::size_t base_number,
    std::size_t vecdim,
    std::size_t k,
    std::size_t p)
{
    if (p >= base_number) {
        return flat_search_simd(base, query, base_number, vecdim, k);
    }
    if (p < k) {
        p = k;
    }

    std::priority_queue<std::pair<float, uint32_t>> heap;
    for (std::size_t i = 0; i < base_number; ++i) {
        const int8_t* row = &idx.codes[i * vecdim];
        const float ip = idx.row_scale[i] * simd_mix::dot_float_int8_neon(query, row, vecdim);
        const float dis = 1.0f - ip;
        if (heap.size() < p) {
            heap.push({dis, static_cast<uint32_t>(i)});
        } else if (dis < heap.top().first) {
            heap.push({dis, static_cast<uint32_t>(i)});
            heap.pop();
        }
    }

    std::vector<uint32_t> top_p;
    top_p.reserve(heap.size());
    while (!heap.empty()) {
        top_p.push_back(heap.top().second);
        heap.pop();
    }
    return ip_rerank_topk(base, query, vecdim, top_p.data(), top_p.size(), k);
}

// 全库：R>=N 时与 mode 0 完全一致（全精度 IP Top-k）；否则 PQ-ADC 取 Top-R 再全精度重排
inline std::priority_queue<std::pair<float, uint32_t>> pq_fullscan_top_r_ip_rerank(
    const PqIndex& idx,
    float* base,
    float* query,
    std::size_t base_number,
    std::size_t vecdim,
    std::size_t k,
    std::size_t R)
{
    if (R >= base_number) {
        return flat_search_simd(base, query, base_number, vecdim, k);
    }
    if (R < k) {
        R = k;
    }
    std::vector<uint32_t> all_ids(base_number);
    for (std::size_t i = 0; i < base_number; ++i) {
        all_ids[i] = static_cast<uint32_t>(i);
    }
    std::vector<uint32_t> top_r = pq_select_top_r_among_candidates(
        idx,
        query,
        all_ids.data(),
        base_number,
        R);
    return ip_rerank_topk(base, query, vecdim, top_r.data(), top_r.size(), k);
}

// 全库：Fast Scan 粗排（与 PQ-ADC 分数一致，仅扫描实现不同）取 Top-R → 全精度 IP Top-k。
// 无 FastScan 布局时退回 pq_fullscan_top_r_ip_rerank（普通 ADC 粗排）。
inline std::priority_queue<std::pair<float, uint32_t>> pq_fullscan_fastscan_top_r_ip_rerank(
    const PqIndex& idx,
    float* base,
    float* query,
    std::size_t base_number,
    std::size_t vecdim,
    std::size_t k,
    std::size_t R)
{
    if (R >= base_number) {
        return flat_search_simd(base, query, base_number, vecdim, k);
    }
    if (R < k) {
        R = k;
    }
    if (idx.codes_fs.size() != idx.n * idx.m || idx.fs_block_off.empty()) {
        return pq_fullscan_top_r_ip_rerank(idx, base, query, base_number, vecdim, k, R);
    }

    std::vector<float> lut(idx.m * idx.ks);
    pq_adc_fill_lut_simd(idx, query, lut.data());

    std::vector<float> dis_all(base_number);
    pq_adc_fastscan_fill_dis_all(idx, lut.data(), base_number, dis_all.data());

    std::priority_queue<std::pair<float, uint32_t>> heap;
    for (std::size_t i = 0; i < base_number; ++i) {
        const float dis = dis_all[i];
        if (heap.size() < R) {
            heap.push({dis, static_cast<uint32_t>(i)});
        } else if (dis < heap.top().first) {
            heap.push({dis, static_cast<uint32_t>(i)});
            heap.pop();
        }
    }
    std::vector<uint32_t> top_r;
    top_r.reserve(heap.size());
    while (!heap.empty()) {
        top_r.push_back(heap.top().second);
        heap.pop();
    }
    return ip_rerank_topk(base, query, vecdim, top_r.data(), top_r.size(), k);
}

// 全库：PQ-SDC 粗排 Top-R → 全精度 IP Top-k（SDC recall 低时常用）
inline std::priority_queue<std::pair<float, uint32_t>> pq_fullscan_sdc_top_r_ip_rerank(
    const PqIndex& idx,
    float* base,
    float* query,
    std::size_t base_number,
    std::size_t vecdim,
    std::size_t k,
    std::size_t R)
{
    if (R >= base_number) {
        return flat_search_simd(base, query, base_number, vecdim, k);
    }
    if (R < k) {
        R = k;
    }
    std::vector<uint32_t> all_ids(base_number);
    for (std::size_t i = 0; i < base_number; ++i) {
        all_ids[i] = static_cast<uint32_t>(i);
    }
    std::vector<uint32_t> top_r = pq_select_top_r_among_candidates_sdc(
        idx,
        query,
        vecdim,
        all_ids.data(),
        base_number,
        R);
    return ip_rerank_topk(base, query, vecdim, top_r.data(), top_r.size(), k);
}

// IVF 候选并集上：PQ-SDC Top-R → 全精度 IP Top-k
inline std::priority_queue<std::pair<float, uint32_t>> ivf_pq_sdc_top_r_ip_rerank(
    const IvfPqIndex& idx,
    float* base,
    float* query,
    std::size_t vecdim,
    std::size_t k,
    std::size_t nprobe,
    std::size_t R)
{
    std::vector<uint32_t> cand = ivf_pq_collect_candidates(idx, query, nprobe);
    if (cand.empty()) {
        return std::priority_queue<std::pair<float, uint32_t>>();
    }
    if (cand.size() < k) {
        return ip_rerank_topk(base, query, vecdim, cand.data(), cand.size(), k);
    }
    if (R < k) {
        R = k;
    }
    const std::size_t r_eff = std::min(R, cand.size());
    std::vector<uint32_t> top_r = pq_select_top_r_among_candidates_sdc(
        idx.pq,
        query,
        vecdim,
        cand.data(),
        cand.size(),
        r_eff);
    return ip_rerank_topk(base, query, vecdim, top_r.data(), top_r.size(), k);
}

// IVF 候选并集上：PQ Top-R → 全精度 IP Top-k
inline std::priority_queue<std::pair<float, uint32_t>> ivf_pq_adc_top_r_ip_rerank(
    const IvfPqIndex& idx,
    float* base,
    float* query,
    std::size_t vecdim,
    std::size_t k,
    std::size_t nprobe,
    std::size_t R)
{
    std::vector<uint32_t> cand = ivf_pq_collect_candidates(idx, query, nprobe);
    if (cand.empty()) {
        return std::priority_queue<std::pair<float, uint32_t>>();
    }
    if (cand.size() < k) {
        return ip_rerank_topk(base, query, vecdim, cand.data(), cand.size(), k);
    }
    if (R < k) {
        R = k;
    }
    const std::size_t r_eff = std::min(R, cand.size());
    std::vector<uint32_t> top_r = pq_select_top_r_among_candidates(
        idx.pq,
        query,
        cand.data(),
        cand.size(),
        r_eff);
    return ip_rerank_topk(base, query, vecdim, top_r.data(), top_r.size(), k);
}

} // namespace top_r_ip_rerank
