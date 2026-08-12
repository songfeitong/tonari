# Benchmark

本文回答三个问题：结果是否与独立实现一致、常见真实 workload 中的端到端成本是多少、体系变大后性能如何变化。这里保留足以理解结论的结果；完整统计、环境、revision 和 binary/data hashes 位于 `benchmarks/results/*.json`。

## 统一测量口径

所有 backend 使用相同 structures、cutoff、PBC、pair 方向和 zero-shift self policy。输出统一转换为 `(source, target, Sx, Sy, Sz)` keys 后做 exact comparison，输出顺序不参与比较。

CPU 固定为单核、单线程，Vesin 复用同一个 `NeighborList` 且关闭 sorting。CUDA 以真实 PyTorch batch 调用本项目；Vesin 的公开接口一次处理一个 structure，因此 baseline 在 batch 内逐结构调用并拼接。Dense PyTorch baseline 直接构造 `N²` 或 `N² × images` candidates，只在内存可控的 representative workloads 上运行。

计时排除数据读取和 H2D，包含公开 one-shot API 的输入处理、native geometry/search、必要分配与同步。每项均 warmup 并重复测量，表中报告 median。CPU frequency policy、软件版本和计时统计由正式 JSON 记录。

## CPU 多线程 scaling

多线程测量使用 AMD Ryzen Threadripper PRO 9975WX 上固定的八个 cores、float64 和 5 Å cutoff，显式比较 `num_threads=1/2/4/8`。Matbench 的 1,536 个晶体和 QMugs population 的 4,096 个分子分别合并成一个 `batch_ptr` batch，用于观察大量独立 structures；大型单体系沿用真实 Matbench 晶体派生的 32,768-atom supercell。三个 workload 的 8,482,022 个 pair keys 都与 Vesin 0.6.1 exact match。

| Workload | Threads | tonari | Speedup | Vesin | Vesin speedup |
| --- | --: | --: | --: | --: | --: |
| Matbench，1,536-structure batch | 1 | 156.361 ms | 1.00× | 222.251 ms | 1.00× |
|  | 2 | 81.206 ms | 1.93× | 219.969 ms | 1.01× |
|  | 4 | 57.938 ms | 2.70× | 204.586 ms | 1.09× |
|  | 8 | 39.383 ms | 3.97× | 195.727 ms | 1.14× |
| QMugs population，4,096-structure batch | 1 | 151.763 ms | 1.00× | 146.076 ms | 1.00× |
|  | 2 | 89.073 ms | 1.70× | 211.450 ms | 0.69× |
|  | 4 | 58.612 ms | 2.59× | 241.881 ms | 0.60× |
|  | 8 | 34.252 ms | 4.43× | 241.787 ms | 0.60× |
| Matbench-derived，32,768 atoms | 1 | 21.532 ms | 1.00× | 12.067 ms | 1.00× |
|  | 2 | 14.769 ms | 1.46× | 8.978 ms | 1.34× |
|  | 4 | 12.372 ms | 1.74× | 6.724 ms | 1.79× |
|  | 8 | 10.219 ms | 2.11× | 5.586 ms | 2.16× |

Tonari 对两个大 batch 的收益来自跨 structure 调度；Vesin 的公开 API 一次只接收一个 structure，因此 baseline 复用一个 `NeighborList`、把相同 `n_threads` 传给每次调用，但仍需顺序遍历 structures。Baseline 对每个 structure 的输出只计数而不额外拼接，这略微有利于 Vesin。这个比较包含两种公开执行模型的差异，并不表示调用方不能在 Vesin 外层另行组织并行。对 32,768-atom 单体系，两者都直接使用相同线程数执行一次调用；Tonari 获得 2.11× scaling，但 Vesin 在所有线程数下仍更快，说明现有 CPU cell-list 的常数差距没有被多线程掩盖。

默认 `num_threads=1` 的意义仍是控制资源而非自动追求最快 wall time。已有 DataLoader workers 或 DDP 进程时，显式增加内部线程可能像 QMugs 上的 Vesin baseline 一样让小任务的调度成本超过收益；组合建议见 [CPU 多线程](cpu-multithreading.md)。完整 samples、frequency policy、affinity、revision 与 binary/data hashes 位于 `benchmarks/results/threadripper-pro-9975wx-cpu-thread-scaling.json`。

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
benchmarks/run_cpu_thread_scaling.py
benchmarks/run_cuda_benchmark.py
benchmarks/run_qmugs_cpu_benchmark.py
benchmarks/run_qmugs_cuda_benchmark.py
benchmarks/run_pair_options_cpu_benchmark.py
benchmarks/run_pair_options_cuda_benchmark.py
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
