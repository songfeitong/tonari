# Benchmark

本文回答三个问题：结果是否与独立实现一致、常见真实 workload 中的端到端成本是多少、体系变大后性能如何变化。这里保留足以理解结论的结果；完整统计、环境、revision 和 binary/data hashes 位于 `benchmarks/results/*.json`。

## 统一测量口径

所有 backend 使用相同 structures、cutoff、PBC、pair 方向和 zero-shift self policy。输出统一转换为 `(source, target, Sx, Sy, Sz)` keys 后做 exact comparison，输出顺序不参与比较。

CPU 单线程对比固定为单核、单线程；多线程 scaling 另行标明。Vesin 复用同一个 `NeighborList` 且关闭 sorting。CUDA 以真实 PyTorch batch 调用本项目；Vesin 的公开接口一次处理一个 structure，因此 baseline 在 batch 内逐结构调用并拼接。Dense PyTorch baseline 直接构造 `N²` 或 `N² × images` candidates，只在内存可控的 representative workloads 上运行。

计时排除数据读取和 H2D，包含公开 one-shot API 的输入处理、native geometry/search、必要分配与同步。每项均 warmup 并重复测量，表中报告 median。2026-09-14 的 CUDA batch-size sweep 对每个 batch 单独同步计时；历史 epoch 表对整轮调用同步计时，不能把 epoch 时间除以 batch 数当作逐 batch 延迟中位数。CPU frequency policy、软件版本和计时统计由正式 JSON 记录。

## CUDA batch-size scaling（2026-09-14）

这组测量回答“一个 batch 构图需要多久”，覆盖 QMugs population 和 Matbench 的 `batch_size=8/16/32/64/128/256/512/1024`。硬件为 NVIDIA RTX PRO 6000 Blackwell，geometry 为 float32，cutoff 为 5 Å，输出为 `PS`，使用 `algorithm="auto"`、full list、无 zero-shift self、无 sorting。Tonari 在一次 native 调用中处理整个 batch；Vesin 0.6.1 使用 CUDA 逐结构调用，包含 batch index offset 和输出拼接成本。

每个数据集、每档 batch size 使用 16 个完整 batch。Seed 为 `20260914` 到 `20260929`，各自生成无放回随机排列并取前 B 个结构；同一 seed 在不同 batch size 下使用嵌套前缀。QMugs 从原有 4,096-molecule population sample 取样，Matbench 从原有 1,536-structure sample 取样。不同 batch 之间允许重叠，batch 内不重复；尤其 Matbench 的 bs=1024 不使用 512-structure 尾批。它们是不同组成的 batch 样本，不是 16 份独立数据集。

每个 batch、每个 backend 先 warmup 2 次，再测量 7 次，每次公开调用前后执行 CUDA synchronize，输出释放在计时区间外。重复轮次交替 backend 顺序。表中中心值为 16 个 batch 各自 7 次耗时中位数的中位数；括号内 P10–P90 描述这些 batch 中位数的分布，包含输入组成与运行波动，不是置信区间。该统计不是整轮 throughput 折算。数据加载与 H2D 均在计时外。

### QMugs population

| Batch size | tonari，ms（P10–P90） | Vesin CUDA 逐结构调用，ms（P10–P90） | Vesin / tonari |
| --: | --: | --: | --: |
| 8 | 0.1049（0.1038–0.1085） | 1.8818（1.8246–1.9440） | 17.9× |
| 16 | 0.1115（0.1101–0.1147） | 3.7275（3.6515–3.7711） | 33.4× |
| 32 | 0.1178（0.1058–0.1195） | 7.3834（7.1084–7.4487） | 62.7× |
| 64 | 0.1150（0.1109–0.1167） | 14.0836（13.9515–14.1730） | 122.5× |
| 128 | 0.1301（0.1294–0.1310） | 28.0850（27.8990–28.3484） | 215.9× |
| 256 | 0.1469（0.1449–0.1613） | 56.0223（55.9059–56.3899） | 381.4× |
| 512 | 0.2259（0.2232–0.2270） | 112.1984（111.6949–112.6445） | 496.7× |
| 1024 | 0.3490（0.3067–0.3806） | 225.4216（224.4515–225.8059） | 645.9× |

### Matbench

| Batch size | tonari，ms（P10–P90） | Vesin CUDA 逐结构调用，ms（P10–P90） | Vesin / tonari |
| --: | --: | --: | --: |
| 8 | 0.1178（0.1068–0.1944） | 2.3459（2.0458–2.8427） | 19.9× |
| 16 | 0.2109（0.1460–0.2495） | 4.8244（4.5520–6.0055） | 22.9× |
| 32 | 0.2678（0.1960–0.3414） | 9.3117（8.9474–10.7066） | 34.8× |
| 64 | 0.4168（0.3511–0.4949） | 19.1085（18.2073–20.3290） | 45.8× |
| 128 | 0.7053（0.6234–0.7707） | 38.3143（37.1092–40.0708） | 54.3× |
| 256 | 1.3329（1.2143–1.4284） | 77.4621（73.2262–79.8904） | 58.1× |
| 512 | 2.7795（2.6816–2.9701） | 152.6286（148.6937–155.1866） | 54.9× |
| 1024 | 7.5129（7.2910–7.7547） | 303.1113（301.0874–305.1597） | 40.3× |

全部 256 个实测 batch 均与 Vesin 的 pair keys 精确一致，共比较 102,323,440 条 keys；此计数包含重叠抽样，不能解释为同样数量的独立结构或独立邻接关系。

QMugs 从 bs=8 到 1024，batch size 增长 128 倍，tonari 延迟从 0.1049 ms 增至 0.3490 ms，约增长 3.3 倍。Matbench 同一区间从 0.1178 ms 增至 7.5129 ms，约增长 63.8 倍；特别是 bs=512 到 1024 耗时增长约 2.7 倍。这组数据支持原生 batch 相对逐结构 CUDA 调用的优势，但不支持“所有体系的 batch 耗时都近乎不变”的表述；曲线变化的原因需要另外 profiling，不能仅从耗时推断。

完整 batch IDs、原始 samples、P10/P90、pair-key correctness、软件版本、Git revision 和 native binary/data hashes 记录在 [`rtx-pro-6000-blackwell-batch-scaling-20260914.json`](../benchmarks/results/rtx-pro-6000-blackwell-batch-scaling-20260914.json)。bs=16 使用相同 binary、seed、输入前缀和计时口径单独补测，原始记录保存在 [`rtx-pro-6000-blackwell-batch-scaling-bs16-20260914.json`](../benchmarks/results/rtx-pro-6000-blackwell-batch-scaling-bs16-20260914.json)，保留独立的测量 revision 与环境信息。下面的历史 epoch、representative batch 和 supercell 表保留各自原始测量，不能与这组新数据拼成同一条曲线；本次没有复测单体系 supercell scaling。

## CPU 多线程 scaling（2026-09-14 NumPy 重测）

三个固定 workload 分别为 Matbench 的 1,536 个晶体、QMugs population 的 4,096 个分子，以及同一 Matbench 晶体派生的 32,768-atom 超胞。使用 Threadripper PRO 9975WX 的八个物理核（affinity 0–7），比较 1、2、4、8 线程。两家均使用 NumPy 输入输出、float64、5 Å cutoff、full list，无 zero-shift self，返回完整 `ijS`。Torch 只参与计时外的数据准备，不进入被测构图接口。

Tonari 每个 workload 执行一次原生 NumPy batch 调用。Vesin 复用一个 `NeighborList` 配置对象，按相同 `n_threads` 每次重新搜索；大 batch 逐结构调用，计时包含全局索引偏移和 NumPy 输出拼接，单结构直接返回 `compute` 结果。输出释放、数据准备和正确性验证均在计时外。每档每个 backend warmup 至少 1 秒，测量 11 次，报告中位数；OMP/OpenBLAS/MKL 背景线程固定为 1。

| Workload | Threads | tonari NumPy | tonari speedup | Vesin NumPy | Vesin speedup |
| --- | --: | --: | --: | --: | --: |
| Matbench，1,536 structures | 1 | 159.813 ms | 1.00× | 230.785 ms | 1.00× |
|  | 2 | 86.001 ms | 1.86× | 224.796 ms | 1.03× |
|  | 4 | 63.996 ms | 2.50× | 212.813 ms | 1.08× |
|  | 8 | 45.212 ms | 3.53× | 202.008 ms | 1.14× |
| QMugs，4,096 structures | 1 | 147.148 ms | 1.00× | 164.568 ms | 1.00× |
|  | 2 | 81.049 ms | 1.82× | 226.418 ms | 0.73× |
|  | 4 | 59.815 ms | 2.46× | 256.403 ms | 0.64× |
|  | 8 | 42.672 ms | 3.45× | 258.728 ms | 0.64× |
| Single periodic structure，32,768 atoms | 1 | 25.679 ms | 1.00× | 11.879 ms | 1.00× |
|  | 2 | 16.284 ms | 1.58× | 8.871 ms | 1.34× |
|  | 4 | 12.689 ms | 2.02× | 7.045 ms | 1.69× |
|  | 8 | 11.259 ms | 2.28× | 4.918 ms | 2.42× |

Tonari 在 Matbench、QMugs 固定大 batch 上的 8 线程加速分别为 3.53×、3.45×，在单个大超胞上为 2.28×。Vesin 在大 batch 中没有外层结构并行，增加的是单个 structure 内部线程；这个执行模型差异必须与曲线一起解释，不能推断外层自行并行的 Vesin 也有同样表现。单个大超胞上 Vesin 在各线程档仍更快。

两家在全部三个 workload、每一档线程数下均逐项精确比较了 canonical `(i,j,Sx,Sy,Sz)`，共覆盖 8,482,022 条 workload pair keys，并为各 backend/thread 组合保存 SHA-256；所有结果一致。图表采用[NumPy 重测记录](../benchmarks/results/threadripper-pro-9975wx-cpu-thread-scaling-numpy-20260914.json)，包含原始 samples、版本、affinity、frequency policy 和 binary/data hashes。之前的 Torch 记录保留用于历史追溯，已不用于当前 CPU 多线程图。单线程三方图虽然同样使用 NumPy，但固定 core 31，本图 affinity 为 0–7；各曲线使用各自当次测量，不能拼接不同 affinity 下的数值。

PDF：[单个大体系](../artifacts/single-structure-cpu-thread-scaling.pdf)、[QMugs 固定 batch](../artifacts/qmugs-cpu-thread-scaling.pdf)、[Matbench 固定 batch](../artifacts/matbench-cpu-thread-scaling.pdf)。横轴为 CPU 线程数，工作量固定，纵轴为线性耗时刻度。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.run_cpu_thread_scaling \
  --threads 1,2,4,8 --cpus 0,1,2,3,4,5,6,7 \
  --repeats 11 --warmup-seconds 1 --require-clean \
  --output runs/cpu-thread-scaling-numpy-20260914.json
python artifacts/plot_cpu_thread_scaling.py
```

默认 `cpu_threads=None` 仍在 CPU 上解析为 1。已有 DataLoader workers 或 DDP 时应规划内部线程，见 [CPU 多线程](cpu-multithreading.md)。

## 周期晶体：matbench_mp_e_form

周期 workload 来自 ColabFit `matbench_mp_e_form`。`scripts/prepare_matbench.py` 使用固定数据 revision 和 seed，按 atom count、cell anisotropy、cell angles 与 composition 做分层抽样，得到 1,536 个 full-PBC structures。样本覆盖 1–444 atoms 和 1,343 个不同 reduced formulas。

Raw Parquet 与派生 cache 位于 Git-ignored `cache/`；仓库保存下载来源、SHA、抽样方法和全部 source IDs。Scaling workload 对样本中的一个 64-atom 晶体做整数 supercell repetition，因此仍属于真实结构派生 workload。

CPU 与 CUDA 在全部 1,536 个 structures、2,780,158 个 pair keys 上与 Vesin exact match。Representative CUDA batch 的 43,842 个 keys 还与独立 dense baseline exact match。

### CPU

硬件为 AMD Ryzen Threadripper PRO 9975WX 的单个固定 core，geometry 使用 float64，cutoff 为 5 Å。

| Workload | Atoms | Pairs | tonari | Vesin reused | Vesin / tonari |
| --- | --: | --: | --: | --: | --: |
| 1,536-structure epoch | 75,238 | 2,780,158 | 126.457 ms | 247.577 ms | 1.96× |
| 真实结构，1×1×1 | 64 | 744 | 0.0361 ms | 0.0450 ms | 1.24× |
| 派生 supercell，2×2×2 | 512 | 5,952 | 0.2285 ms | 0.2298 ms | 1.01× |
| 派生 supercell，3×3×3 | 1,728 | 20,088 | 1.1557 ms | 0.7194 ms | 0.62× |
| 派生 supercell，4×4×4 | 4,096 | 47,616 | 3.0518 ms | 1.6634 ms | 0.55× |
| 派生 supercell，8×8×8 | 32,768 | 380,928 | 24.9314 ms | 13.2429 ms | 0.53× |

真实 epoch 由大量小结构组成，本项目的低固定成本占优。约 512 atoms 进入当前机器的 crossover，之后 Vesin 的成熟 CPU cell list 更快；32,768 原子时 Vesin 约快 1.88×。这说明本项目适合常见小结构与 one-shot calls，但不能解释为所有尺度上的 CPU cell-list 优势。

### 单体系 CPU 三方复测（2026-09-14）

使用相同 Matbench 64-atom 晶体派生的六个超胞，三家均使用 NumPy 输入输出、float64、5 Å cutoff、full list，无 zero-shift self，返回 `ijS`。固定 Threadripper PRO 9975WX 的 core 31，tonari 和 Vesin 显式设为单线程，OMP/OpenBLAS/MKL 也设为单线程。ASE 使用 `primitive_neighbor_list`，保留其默认分箱参数和原生输出排序，不再使用旧 pair-options 测试中的 `PrimitiveNeighborList` 逐原子提取/Torch 转换 adapter。

计时包含一次公开 API 调用的分配、搜索和输出构造，输出释放、数据准备、超胞生成及 correctness comparison 在计时外。Vesin 复用一个 `NeighborList` 配置对象，但每次重新搜索；三家均不使用 Verlet/skin 缓存。各 backend 先 warmup 至少 1 秒，然后轮换顺序测 11 次，表中是中位数。

| Atoms | Pairs | tonari NumPy | Vesin NumPy | ASE primitive_neighbor_list |
| --: | --: | --: | --: | --: |
| 64 | 744 | 0.0713 ms | 0.0674 ms | 1.2320 ms |
| 512 | 5,952 | 0.4088 ms | 0.2889 ms | 5.8109 ms |
| 1,728 | 20,088 | 1.6916 ms | 0.8205 ms | 20.1134 ms |
| 4,096 | 47,616 | 4.2524 ms | 1.7862 ms | 49.3511 ms |
| 13,824 | 160,704 | 12.3045 ms | 5.5745 ms | 180.1876 ms |
| 32,768 | 380,928 | 25.6869 ms | 12.8554 ms | 453.5265 ms |

六个尺寸的三方 canonical pair keys 全部 exact match，共 616,032 条。32,768 原子时，tonari 约比 ASE 快 17.7×，Vesin 约比 tonari 快 2.0×。这支持 CPU 大体系上 Vesin 更快的结论；不要将 NumPy API 的新测量与上方历史 Torch adapter 计时拼成同一条曲线。

[完整原始记录](../benchmarks/results/threadripper-pro-9975wx-single-structure-20260914.json)保留每次计时、环境、CPU frequency policy、数据与 native binary hashes；[PDF 图](../artifacts/single-structure-cpu-latency.pdf)展示同一组测量。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.run_cpu_single_structure \
  --output runs/cpu-single-structure-20260914.json
python artifacts/plot_cpu_single_structure.py
```

### CUDA

硬件为 NVIDIA RTX PRO 6000 Blackwell，geometry 使用 float32，主 workload 为 `batch_size=32`。

| Workload | Atoms | tonari | Vesin/structure | Dense PyTorch |
| --- | --: | --: | --: | --: |
| 1,536-structure epoch | 75,238 | 10.706 ms | 456.085 ms | — |
| Median 32-structure batch | 1,126 | 0.1943 ms | 8.9252 ms | 42.7491 ms |
| 真实结构，1×1×1 | 64 | 0.0773 ms | 0.2876 ms | 0.6895 ms |
| 派生 supercell，2×2×2 | 512 | 0.1175 ms | 0.2762 ms | 6.6082 ms |
| 派生 supercell，3×3×3 | 1,728 | 0.1204 ms | 0.2874 ms | 73.1263 ms |
| 派生 supercell，8×8×8 | 32,768 | 0.2119 ms | 0.6132 ms | skipped |

CUDA 的主要优势来自整个 batch 一次进入 native pipeline。Vesin baseline 的单结构 API 需要逐结构调用；dense baseline 则会 materialize 大量 candidate tensors。二者与本项目代表了不同的执行模型，因此表格既是性能比较，也是 batching strategy 的比较。

32,768-atom case 另有 Nsight Systems profile。Machine-readable summary 与 kernel、memory、CUDA API、NVTX aggregates 位于 `benchmarks/results/`；raw trace 保持 Git ignored。Profile 用于确认时间确实花在预期 kernels 与边界上，不作为公共性能承诺。

## 有限分子：QMugs

有限体系 workload 来自 [QMugs](https://doi.org/10.3929/ethz-b-000482129)。准备脚本从 665,911 个 ChEMBL 分子的 1,992,984 个 conformers 中，为每个分子选择 GFN2-xTB 能量最低的 conformer；能量只用于选择，不进入 benchmark cache。

固定 seed 产生两个互不重叠的 4,096-molecule samples：population sample 保留自然大小分布，总原子数中位数为 52；size-balanced sample 在八个重原子区间各取 512 个分子，总原子数最高 221。数据许可和 attribution 见 [`benchmarks/data/QMUGS_ATTRIBUTION.md`](../benchmarks/data/QMUGS_ATTRIBUTION.md)。

CPU 与 CUDA 在全部 8,192 个分子、15,144,842 个 pair keys 上与 Vesin exact match。九个 representative CUDA batches 的 1,322,646 个 keys 还与 finite dense baseline exact match。

### CPU

| Workload | Structures | Atoms | Pairs | tonari | Vesin reused |
| --- | --: | --: | --: | --: | --: |
| Population epoch | 4,096 | 226,648 | 5,320,936 | 133.910 ms | 174.752 ms |
| Size-balanced epoch | 4,096 | 339,795 | 9,823,906 | 247.193 ms | 286.736 ms |
| 4–10 heavy atoms | 512 | 9,096 | 138,600 | 5.127 ms | 12.359 ms |
| 31–40 heavy atoms | 512 | 31,782 | 766,106 | 18.769 ms | 23.257 ms |
| 81–100 heavy atoms | 512 | 90,409 | 3,051,622 | 74.421 ms | 79.295 ms |

本项目在两个 epoch 和所有 size bins 中均不慢于 Vesin；优势随分子变大而收窄，50 个以上 heavy atoms 时基本接近。QMugs 最大结构仍只有 221 atoms，因此它没有进入晶体 supercell workload 中 Vesin 明显占优的大体系区间。

### CUDA

| Workload | Atoms | Pairs | tonari | Vesin/structure | Dense PyTorch |
| --- | --: | --: | --: | --: | --: |
| Population epoch，bs=8 | 226,648 | 5,320,936 | 37.522 ms | 906.184 ms | — |
| Population epoch，bs=64 | 226,648 | 5,320,936 | 5.015 ms | 903.370 ms | — |
| Population epoch，bs=128 | 226,648 | 5,320,936 | 2.848 ms | 900.984 ms | — |
| Population representative batch | 3,494 | 80,992 | 0.0828 ms | 13.9544 ms | 0.2989 ms |
| 81–100-heavy-atom batch | 11,409 | 384,320 | 0.1192 ms | 16.4433 ms | 0.7914 ms |

这些分子最多 221 atoms，因此 CUDA 主要展示 batch amortization，而不是单个巨大分子的 scaling。随着 batch size 增大，native calls 从数百次降到数十次；Vesin 仍需逐结构执行，因此总时间基本不随 batch size 改变。

## Half list 与 self pair

额外 benchmark 在 256 个 Matbench 晶体上比较 full/half 与 self 组合。Tonari、Vesin 和 ASE 的 normalized pair keys 在四种模式下全部 exact match。

| CPU mode      |   Pairs |   Output |    tonari |     Vesin |        ASE |
| ------------- | ------: | -------: | --------: | --------: | ---------: |
| Full，no self | 505,336 | 14.15 MB | 23.099 ms | 44.532 ms | 703.616 ms |
| Half，no self | 252,668 |  7.07 MB | 21.375 ms | 43.301 ms | 532.243 ms |

Half list 把 pair count 和 output bytes 精确减半，说明它是 native candidate policy，而不是 Python 后处理。ASE 在这里主要承担独立 correctness reference 的角色，没有参与 32,768-atom scaling benchmark。

完整 CUDA pair-mode结果位于 `benchmarks/results/rtx-pro-6000-blackwell-pair-options.json`；half list 同样把输出减半，并显著降低 peak allocation。

## 可复现入口

数据准备与运行脚本：

```text
scripts/prepare_matbench.py
scripts/prepare_qmugs.py
benchmarks/run_cpu_benchmark.py
benchmarks/run_cpu_single_structure.py
benchmarks/run_cpu_thread_scaling.py
benchmarks/run_cuda_benchmark.py
benchmarks/run_cuda_batch_scaling.py
benchmarks/run_qmugs_cpu_benchmark.py
benchmarks/run_qmugs_cuda_benchmark.py
benchmarks/run_pair_options_cpu_benchmark.py
benchmarks/run_pair_options_cuda_benchmark.py
```

在仓库根目录、安装当前源码后，复现新的 CUDA batch-size sweep：

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.run_cuda_batch_scaling \
  --batch-sizes 8 16 32 64 128 256 512 1024 \
  --batches 16 --repeats 7 --warmup 2 --seed 20260914 \
  --output runs/cuda-batch-scaling-20260914.json
```

固定数据来源、selection IDs 与抽样规则：

```text
benchmarks/data/matbench_mp_e_form_sample.json
benchmarks/data/qmugs_sample.json
benchmarks/data/qmugs_sample_structures.csv
```

正式 records 位于 `benchmarks/results/`。每个 JSON 记录当次软件、硬件、Git revision、data/cache/native-extension hashes 与完整统计；文档不重复这些容易变旧的 provenance 字段。

## 结论边界

这些数字只适用于当前 workstation、软件版本、5 Å cutoff 和对应 workload。CPU 结果会受 affinity、frequency policy、dtype 与结构分布影响；CUDA 结果会受 batch composition、GPU、PyTorch/CUDA 版本与显存压力影响。Benchmark 排除 data loading/H2D，也不把静态缓存与动态 one-shot search 混为同一成本口径。结果用于理解工程取舍，不写成 unit-test threshold，也不构成跨机器性能承诺。
