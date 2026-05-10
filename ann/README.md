# ANN SIMD 检索实验（Lab2）

基于 SIMD（NEON / 可移植回退）与 OpenMP 的近似最近邻检索 demo：Flat、SQ、PQ（ADC / SDC / FastScan）、IVF-PQ，以及 Top-$p$ / Top-$R$ 全精度重排。

## 编译

```bash
g++ main.cc -O2 -fopenmp -std=c++11 -o main
```

目标平台需支持所用 SIMD 路径（如 aarch64 上开启 NEON）。使用 `std::async` 的 SDC 流水线模式在多数环境下需链接 pthread（若链接报错可加上 `-pthread`）。

## 数据

默认从以下路径读取二进制向量（`fbin` / ground truth）：

| 平台 | 目录 |
|------|------|
| Linux | `/anndata/` |
| Windows | 可执行文件工作目录下的 `anndata/` |

需包含例如：`DEEP100K.query.fbin`、`DEEP100K.base.100k.fbin`、`DEEP100K.gt.query.100k.top100.bin`。

程序将**仅评测前 2000 条查询**（见 `main.cc` 中 `test_number`）。

## 运行：`./main <mode> [参数 ...]`

第一个参数 `mode` 决定检索算法；其余参数按模式可选。

| mode | 含义 | 命令行示例 | 说明 |
|------|------|------------|------|
| **0** | Flat，内积，SIMD **4** 路 | `./main 0` | 无额外参数 |
| **3** | Flat，内积，SIMD **8** 路 | `./main 3` | 无额外参数 |
| **4** | Flat，内积，SIMD **16** 路 | `./main 4` | 无额外参数 |
| **1** | **SQ**（int8 量化 + SIMD 点积） | `./main 1`<br>`./main 1 <p>` | 无 `p`：单阶段 SQ Top-$k$。<br>有 `p`：先 SQ 全库粗排保留 Top-$p$，再对候选做**全精度内积**重排为 Top-$k$ |
| **2** | **PQ-ADC**（非对称，查 LUT + 全库扫描） | `./main 2`<br>`./main 2 <R>` | 无 `R`：仅 ADC 粗排 Top-$k$。<br>有 `R`：ADC 粗排后取 Top-$R$，再**全精度 IP** 重排为 Top-$k$ |
| **5** | **IVF-PQ**（倒排 + PQ-ADC 于候选子集） | `./main 5`<br>`./main 5 <nlist> <nprobe>`<br>`./main 5 <nlist> <nprobe> <R>` | 缺省：`nlist=256`，`nprobe=8`。<br>第 4 个参数 `R>0` 时：候选集上 ADC + **Top-$R$ 全精度 IP** 重排 |
| **6** | **PQ-SDC**（对称码本内积表） | `./main 6`<br>`./main 6 <R>`<br>`./main 6 <R> pipe`<br>`./main 6 pipe <R>` | `R>0`：SDC 粗排 Top-$R$ → 全精度 IP Top-$k$。<br>`pipe`：双缓冲流水线，异步预编码下一条查询的 PQ 码（可与 `R` 同时使用） |
| **7** | **PQ-ADC + FastScan**（码字块布局 + SIMD 扫描） | `./main 7`<br>`./main 7 <R>` | 无 `R`：仅 FastScan 粗排 Top-$k$（无全精度重排）。<br>有 `R`：FastScan 粗排 Top-$R$ → 全精度 IP Top-$k$。未构建 FastScan 布局时会回退到普通 ADC |

### 参数小结

- **mode 1**：`[p]` = SQ 粗排保留候选数，再全精度重排（`p` 省略则只跑 SQ）。
- **mode 2 / 7**：`[R]` = PQ 粗排后全精度重排的候选数（省略则只做 PQ 粗排 Top-$k$）。
- **mode 5**：`nlist`、`nprobe` 控制 IVF；`[R]` 为可选第 4 参数。
- **mode 6**：`[R]` 与关键字 **`pipe`** 可单独或同时出现，顺序不限（除 `argv[1]` 必须为 `6`）。

## 输出

标准错误流打印索引用时、模式与关键选项；标准输出打印平均 **recall@k** 与平均 **latency（μs）**。
