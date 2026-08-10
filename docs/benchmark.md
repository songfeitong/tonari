# 真实材料 benchmark

## 数据集与抽样

主要数据源是 ColabFit `matbench_mp_e_form`，dataset ID 为 `DS_5drebe4tktiu_0`。Source page 当前报告 132,741 个 configurations，而任务早期描述写的是 132,752；本仓库记录实际观察值，不静默修正这一差异。固定输入是 `colabfit/Matbench_mp_e_form` revision `9880d5b9b62877ec5aa14d1a4c2a9ff4ee870b8d` 的 `co/co_0.parquet`，大小 128,655,162 bytes，SHA-256 为 `4b815791cc31862895b23cda7339d96217c37815c8f183949dc59b3035ee2afd`。

`scripts/prepare_matbench.py` 使用 seed `20260809` 确定性选择 1,536 个 structures。脚本按 atom count、cell-vector length anisotropy、最大 absolute inter-vector cosine 和 element count 使用固定 strata，并用 `sha256(f"{seed}:{configuration_id}")` 决定每个 stratum 内的稳定顺序，再在 strata 间 round-robin。样本覆盖 948 个 occupied strata 和 1,343 个 unique reduced formulas，原子数范围为 1–444。所有源晶体都是 full PBC。

已提交 manifest 保存 source IDs、compositions、atom counts、cell metrics、strata、source revision 和 selection method；raw Parquet 与 tensor cache 位于 Git-ignored `cache/`。当前 manifest SHA-256 为 `3475c391b8def6bf599a57a7f1c938113eabcbfb0f2b89f7d31da648c6f7b413`，cache SHA-256 为 `25644bf26a8c305c91d3e52f3ea8fb12c3c58dede830bf5625b341527179b58c`。没有下载 OMat24，也没有保存 energy/force labels。Scaling workload 只对样本中的 64-atom structure `CO_8661596785617876616983344` 做整数 supercell repetition，因此仍是真实结构派生 workload。

## 正确性口径

每个 backend 的输出都转换为完整的 `(source, target, Sx, Sy, Sz)` key set，并在排序后比较；输出顺序不属于契约。Vesin shift 施加在它的 second atom 上，所以对照映射为 `source=vesin_second`、`target=vesin_first`、`cell_shifts=vesin_shift`。CPU 和 CUDA 都在计时前对全部 1,536 个 structures、2,780,158 个 pairs 与 Vesin 做 exact differential validation，结果全部一致；CUDA 的 median batch 另与独立 dense PyTorch baseline 精确比较了 43,842 个 pairs。

这里比较的是 neighbor identity，不比较浮点 `displacements` 的 backend 舍入顺序。Production tests 另外固定 strict cutoff、onsite exclusion、periodic self-images、multiple images、partial PBC、未 wrap representatives、rank-deficient cells、CPU/CUDA/NumPy/Torch 一致性和一致单位缩放不变性。

## CPU 方法

CPU workload 使用标准 map-style PyTorch `Dataset` 与 `DataLoader(batch_size=1, shuffle=True, num_workers=0)`；shuffle generator seed 与抽样 seed 相同。DataLoader 先 materialize 为确定顺序的 structure list，计时排除数据读取，只包含公开 one-shot `find_neighbors` 的输入处理、periodic metadata、native search、精确输出分配和 Python/C++ boundary。

tonari 与 Vesin 均固定在 CPU 31，且均为单线程。Vesin 使用一个跨所有重复复用的 `NeighborList(cutoff=5.0, full_list=True, sorted=False, n_threads=1)`，不承担 object reconstruction。Geometry 使用 float64；每个 backend/workload 至少 warmup 2 秒，epoch 计时 11 次，scaling cases 计时 12 次；JSON 保存全部 samples、minimum、median 和 maximum。

正式复现命令为：

```bash
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' \
/home/ftsong/projects/elfes-workspace/elfes/.venv/bin/python \
benchmarks/run_cpu_benchmark.py --cpu 31 --repeats 11 \
  --warmup-seconds 2 --require-clean \
  --output benchmarks/results/threadripper-pro-9975wx-cpu.json
```

## CPU 结果

硬件为 AMD Ryzen Threadripper PRO 9975WX 32-Cores，软件为 Python 3.12.3、PyTorch 2.12.1+cu130、Vesin 0.6.1。正式 record 是 `benchmarks/results/threadripper-pro-9975wx-cpu.json`；implementation revision 为 `052f207403d3c5c058dd844ad546115b99790500`，CPU extension SHA-256 为 `f6a193c29c97a2f86faf2e8901d6b97178f3ef26fa52b7288c57a64126764e95`。

| Workload | Atoms | Pairs | tonari CPU | Vesin CPU reused | Vesin / tonari |
| --- | --: | --: | --: | --: | --: |
| 1,536-structure DataLoader epoch | 75,238 | 2,780,158 | 143.999 ms | 248.459 ms | 1.73× |
| 真实结构，1×1×1 | 64 | 744 | 0.0419 ms | 0.0454 ms | 1.08× |
| 派生 supercell，2×2×2 | 512 | 5,952 | 0.2379 ms | 0.2296 ms | 0.97× |
| 派生 supercell，3×3×3 | 1,728 | 20,088 | 1.1043 ms | 0.7152 ms | 0.65× |
| 派生 supercell，4×4×4 | 4,096 | 47,616 | 2.9374 ms | 1.6503 ms | 0.56× |
| 派生 supercell，6×6×6 | 13,824 | 160,704 | 10.1893 ms | 5.4640 ms | 0.54× |
| 派生 supercell，8×8×8 | 32,768 | 380,928 | 24.0689 ms | 13.0912 ms | 0.54× |

真实样本中大量 structure 很小，固定调用成本占主导，所以完整 epoch 上 tonari 领先 1.73×。单独放大后，约 512 atoms 已到本机交叉区，Vesin 在更大单体系上明显更快。正确结论是“tonari CPU 对常见小材料体系和许多高频 one-shot calls 有优势”，而不是“tonari CPU 的 cell list 全面超过 Vesin”。

CPU exhaustive/cell-list crossover 按 `N² × image_count` 判断。开发期在完整 epoch 上扫描 candidate limits 2,048、8,192、16,384、32,768、131,072，对应 quick-run medians 约 138.8、115.4、115.0、117.0、162.2 ms，因此选择 16,384。这个 threshold 属于内部策略，不是公共 API 契约；开发 quick runs 也不与正式固定-core表格混用。

## CUDA 方法

CUDA 使用同一 1,536-structure sample 和真实 `DataLoader(batch_size=32)`；完整 batch 先一次 transfer 到 GPU。计时排除数据读取和 H2D，包含公开 one-shot `find_neighbors`、metadata、同步、分配与 CUDA work。Geometry 使用 float32，cutoff 为 5 Å。

Vesin 0.6.1 baseline 接收 CUDA tensors，但必须逐 structure 调用后再拼接；它的 `NeighborList` 在重复之间复用。Dense baseline 独立复现 Equiformer/FairChem-style `N² × padded periodic images` tensor expansion，并校正为 tonari 的 strict cutoff 与 onsite policy；它只用于方法对照，不复制上游源码。所有 backend 都显式同步，报告 median，并保留 Torch allocator peak。超过 150 million estimated candidates 的 dense scaling case 跳过，以避免无信息的 OOM。

正式复现命令为：

```bash
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=1 \
/home/ftsong/projects/elfes-workspace/elfes/.venv/bin/python \
benchmarks/run_cuda_benchmark.py --require-clean \
  --output benchmarks/results/rtx-pro-6000-blackwell.json
```

## CUDA 结果

硬件是一张 NVIDIA RTX PRO 6000 Blackwell Workstation Edition，compute capability 12.0；软件为 Python 3.12.3、PyTorch 2.12.1+cu130 和 Vesin 0.6.1。正式 record 是 `benchmarks/results/rtx-pro-6000-blackwell.json`；implementation revision 为 `01eeac5683c2871d572338de437716d0689f5e50`，CUDA extension SHA-256 为 `78c0af2e407fac90a332e227af5f73212d8c23cf0930756fcedaeffa3a4e495c`。

| Workload | Atoms | tonari CUDA | Vesin GPU/structure | Vesin / tonari | Dense PyTorch | Dense / tonari |
| --- | --: | --: | --: | --: | --: | --: |
| 1,536-structure DataLoader epoch | 75,238 | 12.107 ms | 494.393 ms | 40.84× | skipped | — |
| Median 32-structure batch | 1,126 | 0.2252 ms | 9.2855 ms | 41.23× | 42.7812 ms | 189.97× |
| 真实结构，1×1×1 | 64 | 0.0969 ms | 0.3733 ms | 3.85× | 0.6898 ms | 7.12× |
| 派生 supercell，2×2×2 | 512 | 0.1444 ms | 0.6293 ms | 4.36× | 6.6022 ms | 45.71× |
| 派生 supercell，3×3×3 | 1,728 | 0.1442 ms | 0.8262 ms | 5.73× | 73.1652 ms | 507.27× |
| 派生 supercell，4×4×4 | 4,096 | 0.1486 ms | 0.8118 ms | 5.46× | skipped | — |
| 派生 supercell，6×6×6 | 13,824 | 0.1811 ms | 1.0332 ms | 5.71× | skipped | — |
| 派生 supercell，8×8×8 | 32,768 | 0.2539 ms | 1.4912 ms | 5.87× | skipped | — |

CUDA 的核心优势不是单个距离公式更神奇，而是整个 batch 一次进入 native pipeline、候选不 materialize、cell list 的中间状态保持在 device、输出精确分配。Median batch 中 dense baseline 额外占用约 6.9 GB Torch allocator memory，tonari 约 1.3 MB；这解释了为什么“进入模型前动态构邻居”在常见训练工作流中可以成为很小的 overhead，而不是显存和 latency 瓶颈。

CUDA 正式数字相对早期原型显著改善，但不能仅凭 source-level diff 把全部收益归因于某一个 kernel。当前证据能确定的是：同一真实 workload、同一语义和同一验证口径下，最终 package/native boundary 与 batched pipeline 的端到端耗时如表所示。

## CUDA profiling

最终 32,768-atom workload 使用 Nsight Systems 记录 3 次 warmup 和 20 次 profile calls。23 次 calls 的全部 CUDA kernels 合计 3.119 ms，即每次约 0.1356 ms；memory operations 每次约 0.0061 ms；20-call NVTX range 合计 6.062 ms，即每次约 0.3031 ms。主要 kernels 的平均耗时为 wrapping/AABB 50.26 µs、pair counting 34.19 µs、pair writing 37.49 µs、periodic image insertion 6.40 µs。

复现命令为：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. \
nsys profile --trace=cuda,nvtx --force-overwrite=true \
  --output=runs/nsys-matbench-32768-final \
  /home/ftsong/projects/elfes-workspace/elfes/.venv/bin/python \
  benchmarks/profile_case.py --factor 8 --iterations 20
```

Machine-readable summary 和完整 kernel、memory operation、CUDA API、NVTX CSV 位于 `benchmarks/results/`；体积较大的 raw `.nsys-rep` 保持 Git ignored。较早实验把 fused atomic bounds 拆成独立 reductions，GPU work 与 NVTX range 都回退，因此已经撤销；summary 保留该反例，避免只记录成功尝试。

## 结论边界

所有数字只代表当前 workstation、固定 software revision、5 Å cutoff 和上述 workload。CPU 对 affinity、frequency warmup、dtype 与 structure distribution 敏感；CUDA 对 batch composition、GPU、PyTorch/CUDA version 和 memory pressure 敏感。静态缓存与动态 one-shot search 也不是同一种成本口径。Benchmark 数字不会写成 unit-test 阈值，也不构成跨机器性能承诺。
