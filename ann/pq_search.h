// pq_search.h — 乘积量化 PQ：m 段 × Ks 聚类中心，ADC 近似内积
// dot(q,x) ≈ sum_s dot(q_seg, C[s][code_x[s]])

#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <queue>
#include <algorithm>
#include <random>
#include <cstring>

#include "simd_inner_product.h"

struct PqIndex {
    std::size_t n = 0;
    std::size_t d = 0;
    std::size_t m = 8;           // 段数（96 维常用 8）
    std::size_t ks = 256;        // 每段码字个数
    std::size_t sub = 0;         // d / m
    std::vector<float> codebooks; // [m][ks][sub]，行主序：((s*ks)+k)*sub + j
    std::vector<uint8_t> codes;   // [n][m]
    // SDC：对称分量内积表 T_s[i][j] = dot(C[s,i], C[s,j])，行主序下标 ((s*ks+i)*ks+j)
    std::vector<float> sdc_ip;
    // Fast Scan：码字按块重排（见 pq_build_fastscan_layout），与 codes 信息量相同
    std::size_t fs_bs = 16;
    std::vector<uint8_t> codes_fs;
    std::vector<std::size_t> fs_block_off; // 长度 nblocks+1，codes_fs 中每块起始字节
};

namespace pq_detail {

inline float l2sq_sub(const float* a, const float* b, std::size_t sub)
{
    float t = 0.0f;
    for (std::size_t j = 0; j < sub; ++j) {
        float df = a[j] - b[j];
        t += df * df;
    }
    return t;
}

inline void pq_train_segment(
    const float* train_base,
    std::size_t ntrain,
    std::size_t d,
    std::size_t seg,
    std::size_t sub,
    std::size_t ks,
    int max_iter,
    uint32_t seed,
    float* centroids_out)
{
    std::mt19937 rng(seed + static_cast<uint32_t>(seg) * 997u);
    std::uniform_int_distribution<std::size_t> uni(0, ntrain - 1);

    for (std::size_t k = 0; k < ks; ++k) {
        std::size_t pick = uni(rng);
        const float* src = train_base + pick * d + seg * sub;
        std::memcpy(centroids_out + k * sub, src, sub * sizeof(float));
    }

    std::vector<std::size_t> counts(ks);
    std::vector<float> accum(ks * sub);

    for (int it = 0; it < max_iter; ++it) {
        counts.assign(ks, 0);
        accum.assign(ks * sub, 0.0f);

        for (std::size_t i = 0; i < ntrain; ++i) {
            const float* v = train_base + i * d + seg * sub;
            float best = 1e30f;
            std::size_t bk = 0;
            for (std::size_t k = 0; k < ks; ++k) {
                float t = l2sq_sub(v, centroids_out + k * sub, sub);
                if (t < best) {
                    best = t;
                    bk = k;
                }
            }
            counts[bk]++;
            float* acc = &accum[bk * sub];
            for (std::size_t j = 0; j < sub; ++j) {
                acc[j] += v[j];
            }
        }

        for (std::size_t k = 0; k < ks; ++k) {
            if (counts[k] == 0) {
                std::size_t pick = uni(rng);
                const float* src = train_base + pick * d + seg * sub;
                std::memcpy(centroids_out + k * sub, src, sub * sizeof(float));
                continue;
            }
            float inv = 1.0f / static_cast<float>(counts[k]);
            float* dst = centroids_out + k * sub;
            const float* src = &accum[k * sub];
            for (std::size_t j = 0; j < sub; ++j) {
                dst[j] = src[j] * inv;
            }
        }
    }
}

inline void pq_encode_row(
    const float* row,
    std::size_t d,
    std::size_t m,
    std::size_t ks,
    std::size_t sub,
    const float* codebooks_all,
    uint8_t* codes_out)
{
    for (std::size_t s = 0; s < m; ++s) {
        const float* v = row + s * sub;
        const float* cent = codebooks_all + (s * ks) * sub;
        float best = 1e30f;
        std::size_t bk = 0;
        for (std::size_t k = 0; k < ks; ++k) {
            float t = l2sq_sub(v, cent + k * sub, sub);
            if (t < best) {
                best = t;
                bk = k;
            }
        }
        codes_out[s] = static_cast<uint8_t>(bk);
    }
}

} // namespace pq_detail

// ADC 查找表 LUT[s][c] = dot(q_seg_s, codebook[s][c])，与 flat SIMD 共用 inner_product_neon。
// OpenMP：对全部 (s,c) 任务并行，等价于「PQ 编码在 n=1 时对 Ks 个 centroid 并行算距离/内积」。
inline void pq_adc_fill_lut_simd(const PqIndex& idx, const float* query, float* lut /* length m*ks */)
{
    const std::size_t lut_sz = idx.m * idx.ks;
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

// 定义在后文；Fast Scan 回退路径需先见声明
inline std::priority_queue<std::pair<float, uint32_t>> flat_search_pq_adc_simd(
    const PqIndex& idx,
    float* query,
    std::size_t base_number,
    std::size_t vecdim,
    std::size_t k);

// 构建 SDC 查表（离线一次）：每段 Ks×Ks 个 sub 维内积，内层用 NEON
inline void pq_fill_sdc_ip_tables(PqIndex& idx)
{
    idx.sdc_ip.resize(idx.m * idx.ks * idx.ks);
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
    for (long long ss = 0; ss < static_cast<long long>(idx.m); ++ss) {
        const std::size_t s = static_cast<std::size_t>(ss);
        for (std::size_t i = 0; i < idx.ks; ++i) {
            const float* ci = &idx.codebooks[((s * idx.ks) + i) * idx.sub];
            for (std::size_t j = 0; j < idx.ks; ++j) {
                const float* cj = &idx.codebooks[((s * idx.ks) + j) * idx.sub];
                idx.sdc_ip[((s * idx.ks) + i) * idx.ks + j] =
                    simd_f32::inner_product_neon(ci, cj, idx.sub);
            }
        }
    }
#else
    for (std::size_t s = 0; s < idx.m; ++s) {
        for (std::size_t i = 0; i < idx.ks; ++i) {
            const float* ci = &idx.codebooks[((s * idx.ks) + i) * idx.sub];
            for (std::size_t j = 0; j < idx.ks; ++j) {
                const float* cj = &idx.codebooks[((s * idx.ks) + j) * idx.sub];
                idx.sdc_ip[((s * idx.ks) + i) * idx.ks + j] =
                    simd_f32::inner_product_neon(ci, cj, idx.sub);
            }
        }
    }
#endif
}

// SDC 打分：已得到查询 PQ 码 qcode，近似 dot(q,x) ≈ sum_s T_s[qcode[s], codes_x[s]]
inline std::priority_queue<std::pair<float, uint32_t>> flat_search_pq_sdc_from_codes(
    const PqIndex& idx,
    const uint8_t* qcode,
    std::size_t base_number,
    std::size_t k)
{
    if (idx.sdc_ip.size() != idx.m * idx.ks * idx.ks) {
        std::cerr << "[pq_sdc] sdc_ip not built\n";
        return std::priority_queue<std::pair<float, uint32_t>>();
    }
    const float* tab = idx.sdc_ip.data();
    std::vector<float> dis_all(base_number);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
    for (long long ii = 0; ii < static_cast<long long>(base_number); ++ii) {
        const std::size_t i = static_cast<std::size_t>(ii);
        float ip = 0.0f;
        const uint8_t* cx = &idx.codes[i * idx.m];
        for (std::size_t s = 0; s < idx.m; ++s) {
            const std::size_t qi = qcode[s];
            const std::size_t xi = cx[s];
            ip += tab[((s * idx.ks) + qi) * idx.ks + xi];
        }
        dis_all[i] = 1.0f - ip;
    }
#else
    for (std::size_t i = 0; i < base_number; ++i) {
        float ip = 0.0f;
        const uint8_t* cx = &idx.codes[i * idx.m];
        for (std::size_t s = 0; s < idx.m; ++s) {
            const std::size_t qi = qcode[s];
            const std::size_t xi = cx[s];
            ip += tab[((s * idx.ks) + qi) * idx.ks + xi];
        }
        dis_all[i] = 1.0f - ip;
    }
#endif

    std::priority_queue<std::pair<float, uint32_t>> q;
    for (std::size_t i = 0; i < base_number; ++i) {
        const float dis = dis_all[i];
        if (q.size() < k) {
            q.push({dis, static_cast<uint32_t>(i)});
        } else if (dis < q.top().first) {
            q.push({dis, static_cast<uint32_t>(i)});
            q.pop();
        }
    }
    return q;
}

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_pq_sdc(
    const PqIndex& idx,
    float* query,
    std::size_t base_number,
    std::size_t vecdim,
    std::size_t k)
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
    return flat_search_pq_sdc_from_codes(idx, qcode.data(), base_number, k);
}

// Fast Scan 布局（Jégou et al. PQ fast scan 思路）：按向量块遍历，块内对每段 s 连续存放 blen 个码，
// 使扫描 inner loop 对固定 s 顺序读码，整块可用 NEON 一次加载 16×uint8。
inline void pq_build_fastscan_layout(PqIndex& idx)
{
    const std::size_t bs = idx.fs_bs;
    const std::size_t n = idx.n;
    const std::size_t m = idx.m;
    if (n == 0 || m == 0 || bs == 0) {
        return;
    }
    const std::size_t nblocks = (n + bs - 1) / bs;
    idx.fs_block_off.resize(nblocks + 1);
    idx.fs_block_off[0] = 0;
    idx.codes_fs.resize(n * m);
    std::size_t out = 0;
    for (std::size_t bi = 0; bi < nblocks; ++bi) {
        const std::size_t b0 = bi * bs;
        const std::size_t blen = std::min(bs, n - b0);
        for (std::size_t s = 0; s < m; ++s) {
            for (std::size_t j = 0; j < blen; ++j) {
                idx.codes_fs[out++] = idx.codes[(b0 + j) * m + s];
            }
        }
        idx.fs_block_off[bi + 1] = out;
    }
}

namespace pq_fs_detail {

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

inline void accum_segment_lut_16(
    const float* lut_row,
    const uint8_t* codes16,
    float* acc)
{
    uint8x16_t vb = vld1q_u8(codes16);
    alignas(16) uint8_t tmp[16];
    vst1q_u8(tmp, vb);
    for (int t = 0; t < 16; ++t) {
        acc[t] += lut_row[tmp[t]];
    }
}

#endif

inline void scan_block_adc(
    const PqIndex& idx,
    const float* lut,
    std::size_t blen,
    std::size_t p_byte,
    float* acc_out)
{
    const std::size_t m = idx.m;
    const std::size_t ks = idx.ks;
    for (std::size_t t = 0; t < blen; ++t) {
        acc_out[t] = 0.0f;
    }
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    if (blen == 16) {
        for (std::size_t s = 0; s < m; ++s) {
            const float* lut_row = lut + s * ks;
            accum_segment_lut_16(lut_row, idx.codes_fs.data() + p_byte + s * 16, acc_out);
        }
        return;
    }
#endif
    for (std::size_t s = 0; s < m; ++s) {
        const float* lut_row = lut + s * ks;
        const uint8_t* row = idx.codes_fs.data() + p_byte + s * blen;
        for (std::size_t j = 0; j < blen; ++j) {
            acc_out[j] += lut_row[row[j]];
        }
    }
}

} // namespace pq_fs_detail

// 已构建 LUT；用 Fast Scan 布局写全库近似距离 dis[i] = 1 - sum_s LUT[s,code]，与 flat_search_pq_adc_simd 数值一致。
inline void pq_adc_fastscan_fill_dis_all(
    const PqIndex& idx,
    const float* lut,
    std::size_t base_number,
    float* dis_out)
{
    if (idx.codes_fs.size() != idx.n * idx.m || idx.fs_block_off.empty()) {
        return;
    }
    const std::size_t bs = idx.fs_bs;
    const std::size_t nblocks = (base_number + bs - 1) / bs;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
    for (long long bi = 0; bi < static_cast<long long>(nblocks); ++bi) {
        alignas(16) float acc_buf[16];
        const std::size_t b0 = static_cast<std::size_t>(bi) * bs;
        const std::size_t blen = std::min(bs, base_number - b0);
        const std::size_t p_byte = idx.fs_block_off[static_cast<std::size_t>(bi)];
        pq_fs_detail::scan_block_adc(idx, lut, blen, p_byte, acc_buf);
        for (std::size_t j = 0; j < blen; ++j) {
            dis_out[b0 + j] = 1.0f - acc_buf[j];
        }
    }
#else
    for (std::size_t bi = 0; bi < nblocks; ++bi) {
        alignas(16) float acc_buf[16];
        const std::size_t b0 = bi * bs;
        const std::size_t blen = std::min(bs, base_number - b0);
        const std::size_t p_byte = idx.fs_block_off[bi];
        pq_fs_detail::scan_block_adc(idx, lut, blen, p_byte, acc_buf);
        for (std::size_t j = 0; j < blen; ++j) {
            dis_out[b0 + j] = 1.0f - acc_buf[j];
        }
    }
#endif
}

// PQ-ADC Fast Scan：与 flat_search_pq_adc_simd 数值一致；仅码字布局与内层访存顺序不同。
inline std::priority_queue<std::pair<float, uint32_t>> flat_search_pq_adc_fastscan_simd(
    const PqIndex& idx,
    float* query,
    std::size_t base_number,
    std::size_t vecdim,
    std::size_t k)
{
    (void)vecdim;
    if (idx.codes_fs.size() != idx.n * idx.m || idx.fs_block_off.empty()) {
        std::cerr << "[pq_fastscan] layout missing; fallback to standard ADC\n";
        return flat_search_pq_adc_simd(idx, query, base_number, vecdim, k);
    }

    const std::size_t lut_sz = idx.m * idx.ks;
    std::vector<float> lut(lut_sz);
    pq_adc_fill_lut_simd(idx, query, lut.data());

    std::vector<float> dis_all(base_number);
    pq_adc_fastscan_fill_dis_all(idx, lut.data(), base_number, dis_all.data());

    std::priority_queue<std::pair<float, uint32_t>> q;
    for (std::size_t i = 0; i < base_number; ++i) {
        const float dis = dis_all[i];
        if (q.size() < k) {
            q.push({dis, static_cast<uint32_t>(i)});
        } else if (dis < q.top().first) {
            q.push({dis, static_cast<uint32_t>(i)});
            q.pop();
        }
    }
    return q;
}

inline void pq_build_from_base(
    const float* base,
    std::size_t base_number,
    std::size_t vecdim,
    PqIndex& out,
    std::size_t pq_m = 8,
    std::size_t pq_ks = 256,
    int kmeans_iter = 15,
    uint32_t seed = 12345u)
{
    if (vecdim % pq_m != 0) {
        std::cerr << "[pq] vecdim must be divisible by m\n";
        return;
    }
    out.n = base_number;
    out.d = vecdim;
    out.m = pq_m;
    out.ks = pq_ks;
    out.sub = vecdim / pq_m;
    out.codebooks.resize(out.m * out.ks * out.sub);
    out.codes.resize(out.n * out.m);

    const std::size_t ntrain = base_number;
    for (std::size_t s = 0; s < out.m; ++s) {
        float* cb = &out.codebooks[(s * out.ks) * out.sub];
        pq_detail::pq_train_segment(
            base,
            ntrain,
            vecdim,
            s,
            out.sub,
            out.ks,
            kmeans_iter,
            static_cast<uint32_t>(seed + static_cast<uint32_t>(s)),
            cb);
    }

    for (std::size_t i = 0; i < base_number; ++i) {
        pq_detail::pq_encode_row(
            base + i * vecdim,
            vecdim,
            out.m,
            out.ks,
            out.sub,
            out.codebooks.data(),
            &out.codes[i * out.m]);
    }

    pq_fill_sdc_ip_tables(out);
    pq_build_fastscan_layout(out);
}

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_pq_adc_simd(
    const PqIndex& idx,
    float* query,
    std::size_t base_number,
    std::size_t vecdim,
    std::size_t k)
{
    (void)vecdim;
    // ADC：LUT[s][c] = dot(q_seg, C[s,c])；检索热点并行：LUT 预计算 + 全库打分并行，Top-K 单线程归并
    const std::size_t lut_sz = idx.m * idx.ks;
    std::vector<float> lut(lut_sz);
    pq_adc_fill_lut_simd(idx, query, lut.data());

    std::vector<float> dis_all(base_number);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
    for (long long ii = 0; ii < static_cast<long long>(base_number); ++ii) {
        const std::size_t i = static_cast<std::size_t>(ii);
        float ip = 0.0f;
        for (std::size_t s = 0; s < idx.m; ++s) {
            const uint8_t code = idx.codes[i * idx.m + s];
            ip += lut[s * idx.ks + code];
        }
        dis_all[i] = 1.0f - ip;
    }
#else
    for (std::size_t i = 0; i < base_number; ++i) {
        float ip = 0.0f;
        for (std::size_t s = 0; s < idx.m; ++s) {
            const uint8_t code = idx.codes[i * idx.m + s];
            ip += lut[s * idx.ks + code];
        }
        dis_all[i] = 1.0f - ip;
    }
#endif

    std::priority_queue<std::pair<float, uint32_t>> q;
    for (std::size_t i = 0; i < base_number; ++i) {
        const float dis = dis_all[i];
        if (q.size() < k) {
            q.push({dis, static_cast<uint32_t>(i)});
        } else if (dis < q.top().first) {
            q.push({dis, static_cast<uint32_t>(i)});
            q.pop();
        }
    }
    return q;
}

// 仅对候选 id 做 PQ-ADC Top-K（与 flat_search_pq_adc_simd 打分一致）
inline std::priority_queue<std::pair<float, uint32_t>> flat_search_pq_adc_simd_subset(
    const PqIndex& idx,
    float* query,
    const uint32_t* cand,
    std::size_t ncand,
    std::size_t k)
{
    const std::size_t lut_sz = idx.m * idx.ks;
    std::vector<float> lut(lut_sz);
    pq_adc_fill_lut_simd(idx, query, lut.data());

    std::priority_queue<std::pair<float, uint32_t>> q;
    for (std::size_t t = 0; t < ncand; ++t) {
        const std::size_t i = static_cast<std::size_t>(cand[t]);
        float ip = 0.0f;
        for (std::size_t s = 0; s < idx.m; ++s) {
            const uint8_t code = idx.codes[i * idx.m + s];
            ip += lut[s * idx.ks + code];
        }
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
