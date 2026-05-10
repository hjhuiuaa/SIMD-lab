#include <queue>
#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <cstdlib>
#include <cstring>
#include <future>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_search_simd.h"
#include "flat_search_simd_w8.h"
#include "flat_search_simd_w16.h"
#include "sq_search.h"
#include "pq_search.h"
#include "ivf_pq_search.h"
#include "top_r_ip_rerank.h"
// 可以自行添加需要的头文件

using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}


int main(int argc, char *argv[])
{
    // 检索模式：0=w4，3=w8，4=w16，1=SQ，2=PQ，5=IVF-PQ，6=PQ-SDC，7=PQ-ADC FastScan
    // Top-p SQ 粗排 + 全精度 IP：./main 1 <p>（p 省略则仅 SQ 单阶段）
    // Top-R：./main 2 <R> ；./main 7 <R>（FastScan 粗排）；./main 5 nlist nprobe <R>
    // PQ-SDC：./main 6 [R] [pipe] — R 为两阶段粗排候选数（SDC Top-R → 全精度 IP Top-k）；pipe 可选
    int search_mode = 0;
    if (argc >= 2) {
        search_mode = std::atoi(argv[1]);
    }
    bool pq_sdc_pipeline = false;
    std::size_t sdc_ip_rerank_R = 0;
    if (search_mode == 6) {
        for (int a = 2; a < argc; ++a) {
            if (std::strcmp(argv[a], "pipe") == 0) {
                pq_sdc_pipeline = true;
            } else {
                const int v = std::atoi(argv[a]);
                if (v > 0) {
                    sdc_ip_rerank_R = static_cast<std::size_t>(v);
                }
            }
        }
    }

    std::size_t ivf_nlist = 256;
    std::size_t ivf_nprobe = 8;
    if (search_mode == 5) {
        if (argc >= 3) {
            ivf_nlist = static_cast<std::size_t>(std::atoi(argv[2]));
        }
        if (argc >= 4) {
            ivf_nprobe = static_cast<std::size_t>(std::atoi(argv[3]));
        }
        if (ivf_nlist < 2) {
            ivf_nlist = 2;
        }
        if (ivf_nprobe < 1) {
            ivf_nprobe = 1;
        }
    }

    std::size_t sq_top_p = 0;
    if (search_mode == 1 && argc >= 3) {
        sq_top_p = static_cast<std::size_t>(std::atoi(argv[2]));
    }

    std::size_t top_r_rerank = 0;
    if ((search_mode == 2 || search_mode == 7) && argc >= 3) {
        top_r_rerank = static_cast<std::size_t>(std::atoi(argv[2]));
    }
    if (search_mode == 5 && argc >= 5) {
        top_r_rerank = static_cast<std::size_t>(std::atoi(argv[4]));
    }

    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    // Windows 本地：相对 exe 工作目录下的 anndata/；Linux 服务器：课程默认 /anndata/
#if defined(_WIN32)
    std::string data_path = "anndata/";
#else
    std::string data_path = "/anndata/";
#endif
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    // 只测试前2000条查询
    test_number = 2000;

    const size_t k = 10;

    std::vector<SearchResult> results;
    results.resize(test_number);

    // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
    // 要保存的目录必须是files/*
    // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
    // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
    // 下面是一个构建hnsw索引的示例
    // build_index(base, base_number, vecdim);

    SqIndex sq_idx;
    PqIndex pq_idx;
    IvfPqIndex ivf_pq_idx;

    if (search_mode == 1) {
        struct timeval t0, t1;
        gettimeofday(&t0, NULL);
        sq_build_from_base(base, base_number, vecdim, sq_idx);
        gettimeofday(&t1, NULL);
        int64_t build_us = (t1.tv_sec - t0.tv_sec) * 1000000LL + (t1.tv_usec - t0.tv_usec);
        std::cerr << "[sq] index build time (us): " << build_us << "\n";
    } else if (search_mode == 2 || search_mode == 6 || search_mode == 7) {
        struct timeval t0, t1;
        gettimeofday(&t0, NULL);
        pq_build_from_base(base, base_number, vecdim, pq_idx);
        gettimeofday(&t1, NULL);
        int64_t build_us = (t1.tv_sec - t0.tv_sec) * 1000000LL + (t1.tv_usec - t0.tv_usec);
        std::cerr << "[pq] index build time (us): " << build_us << "\n";
    } else if (search_mode == 5) {
        struct timeval t0, t1;
        gettimeofday(&t0, NULL);
        ivf_pq_build_from_base(base, base_number, vecdim, ivf_pq_idx, ivf_nlist);
        gettimeofday(&t1, NULL);
        int64_t build_us = (t1.tv_sec - t0.tv_sec) * 1000000LL + (t1.tv_usec - t0.tv_usec);
        std::cerr << "[ivf_pq] nlist=" << ivf_nlist << " nprobe=" << ivf_nprobe
                  << " build time (us): " << build_us << "\n";
    }

    std::cerr << "[search] mode=" << search_mode
              << " (0=w4 3=w8 4=w16 1=sq 2=pq 5=ivf_pq 6=pq_sdc 7=pq_fastscan)";
    if (pq_sdc_pipeline) {
        std::cerr << " pq_sdc_encode_pipeline";
    }
    if (sdc_ip_rerank_R > 0) {
        std::cerr << " pq_sdc_top_r_ip_rerank R=" << sdc_ip_rerank_R;
    }
    if (sq_top_p > 0) {
        std::cerr << " sq_top_p_ip_rerank p=" << sq_top_p;
    }
    if (top_r_rerank > 0) {
        if (search_mode == 7) {
            std::cerr << " pq_fastscan_top_r_ip_rerank R=" << top_r_rerank;
        } else {
            std::cerr << " top_r_ip_rerank R=" << top_r_rerank;
        }
    }
    std::cerr << "\n";

    std::vector<uint32_t> sdc_all_ids;
    if (search_mode == 6 && sdc_ip_rerank_R > 0 && pq_sdc_pipeline) {
        sdc_all_ids.resize(base_number);
        for (std::size_t ii = 0; ii < base_number; ++ii) {
            sdc_all_ids[ii] = static_cast<uint32_t>(ii);
        }
    }

    std::vector<uint8_t> sdc_buf0;
    std::vector<uint8_t> sdc_buf1;
    int sdc_slot = 0;
    if (search_mode == 6 && pq_sdc_pipeline) {
        sdc_buf0.resize(pq_idx.m);
        sdc_buf1.resize(pq_idx.m);
        pq_detail::pq_encode_row(
            test_query,
            vecdim,
            pq_idx.m,
            pq_idx.ks,
            pq_idx.sub,
            pq_idx.codebooks.data(),
            sdc_buf0.data());
        sdc_slot = 0;
    }

    // 查询测试代码
    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        // 该文件已有代码中你只能修改该函数的调用方式
        // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。
        std::priority_queue<std::pair<float, uint32_t>> res;
        if (search_mode == 0) {
            res = flat_search_simd(base, test_query + i * vecdim, base_number, vecdim, k);
        } else if (search_mode == 3) {
            res = flat_search_simd_w8(base, test_query + i * vecdim, base_number, vecdim, k);
        } else if (search_mode == 4) {
            res = flat_search_simd_w16(base, test_query + i * vecdim, base_number, vecdim, k);
        } else if (search_mode == 1) {
            if (sq_top_p > 0) {
                res = top_r_ip_rerank::sq_fullscan_top_p_ip_rerank(
                    sq_idx,
                    base,
                    test_query + i * vecdim,
                    base_number,
                    vecdim,
                    k,
                    sq_top_p);
            } else {
                res = flat_search_sq_simd(sq_idx, test_query + i * vecdim, base_number, vecdim, k);
            }
        } else if (search_mode == 2) {
            if (top_r_rerank > 0) {
                res = top_r_ip_rerank::pq_fullscan_top_r_ip_rerank(
                    pq_idx,
                    base,
                    test_query + i * vecdim,
                    base_number,
                    vecdim,
                    k,
                    top_r_rerank);
            } else {
                res = flat_search_pq_adc_simd(pq_idx, test_query + i * vecdim, base_number, vecdim, k);
            }
        } else if (search_mode == 7) {
            if (top_r_rerank > 0) {
                res = top_r_ip_rerank::pq_fullscan_fastscan_top_r_ip_rerank(
                    pq_idx,
                    base,
                    test_query + i * vecdim,
                    base_number,
                    vecdim,
                    k,
                    top_r_rerank);
            } else {
                res = flat_search_pq_adc_fastscan_simd(
                    pq_idx, test_query + i * vecdim, base_number, vecdim, k);
            }
        } else if (search_mode == 6) {
            if (sdc_ip_rerank_R > 0) {
                if (pq_sdc_pipeline) {
                    std::future<void> fut;
                    if (i + 1 < test_number) {
                        const int oth = 1 - sdc_slot;
                        float* qp = test_query + (i + 1) * vecdim;
                        uint8_t* dst = (oth == 0) ? sdc_buf0.data() : sdc_buf1.data();
                        PqIndex* idxp = &pq_idx;
                        fut = std::async(std::launch::async, [qp, vecdim, idxp, dst]() {
                            pq_detail::pq_encode_row(
                                qp,
                                vecdim,
                                idxp->m,
                                idxp->ks,
                                idxp->sub,
                                idxp->codebooks.data(),
                                dst);
                        });
                    }
                    uint8_t* cur = (sdc_slot == 0) ? sdc_buf0.data() : sdc_buf1.data();
                    std::vector<uint32_t> top_r = top_r_ip_rerank::pq_select_top_r_among_candidates_sdc_from_qcode(
                        pq_idx,
                        cur,
                        sdc_all_ids.data(),
                        base_number,
                        sdc_ip_rerank_R);
                    res = top_r_ip_rerank::ip_rerank_topk(
                        base,
                        test_query + i * vecdim,
                        vecdim,
                        top_r.data(),
                        top_r.size(),
                        k);
                    if (i + 1 < test_number) {
                        fut.wait();
                    }
                    sdc_slot = 1 - sdc_slot;
                } else {
                    res = top_r_ip_rerank::pq_fullscan_sdc_top_r_ip_rerank(
                        pq_idx,
                        base,
                        test_query + i * vecdim,
                        base_number,
                        vecdim,
                        k,
                        sdc_ip_rerank_R);
                }
            } else if (pq_sdc_pipeline) {
                std::future<void> fut;
                if (i + 1 < test_number) {
                    const int oth = 1 - sdc_slot;
                    float* qp = test_query + (i + 1) * vecdim;
                    uint8_t* dst = (oth == 0) ? sdc_buf0.data() : sdc_buf1.data();
                    PqIndex* idxp = &pq_idx;
                    fut = std::async(std::launch::async, [qp, vecdim, idxp, dst]() {
                        pq_detail::pq_encode_row(
                            qp,
                            vecdim,
                            idxp->m,
                            idxp->ks,
                            idxp->sub,
                            idxp->codebooks.data(),
                            dst);
                    });
                }
                uint8_t* cur = (sdc_slot == 0) ? sdc_buf0.data() : sdc_buf1.data();
                res = flat_search_pq_sdc_from_codes(pq_idx, cur, base_number, k);
                if (i + 1 < test_number) {
                    fut.wait();
                }
                sdc_slot = 1 - sdc_slot;
            } else {
                res = flat_search_pq_sdc(pq_idx, test_query + i * vecdim, base_number, vecdim, k);
            }
        } else if (search_mode == 5) {
            if (top_r_rerank > 0) {
                res = top_r_ip_rerank::ivf_pq_adc_top_r_ip_rerank(
                    ivf_pq_idx,
                    base,
                    test_query + i * vecdim,
                    vecdim,
                    k,
                    ivf_nprobe,
                    top_r_rerank);
            } else {
                res = ivf_pq_search_adc(ivf_pq_idx, test_query + i * vecdim, k, ivf_nprobe);
            }
        } else {
            std::cerr << "unknown search_mode\n";
            return 1;
        }

        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (res.size()) {   
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc/k;

        results[i] = {recall, diff};
    }

    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
    return 0;
}
