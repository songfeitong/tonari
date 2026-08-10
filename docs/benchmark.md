# 真实结构 benchmark

## 周期晶体：数据集与抽样

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

Vesin 0.6.1 baseline 接收 CUDA tensors，但必须逐 structure 调用后再拼接。历史 Matbench 正式 record 的实现每个 DataLoader batch 构造一个 `NeighborList`，再在该 batch 内复用；因此 object construction 包含在每次 batch call 中。后续 QMugs runner 改为跨全部 batches/repeats 复用同一个对象，避免把这项固定成本算给竞争对手。Dense baseline 独立复现 Equiformer/FairChem-style `N² × padded periodic images` tensor expansion，并校正为 tonari 的 strict cutoff 与 onsite policy；它只用于方法对照，不复制上游源码。所有 backend 都显式同步，报告 median，并保留 Torch allocator peak。超过 150 million estimated candidates 的 dense scaling case 跳过，以避免无信息的 OOM。

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

## 有限分子：QMugs 数据集与抽样

有限分子 workload 使用 [QMugs 原始数据集](https://doi.org/10.3929/ethz-b-000482129)，即从 ChEMBL 27 提取并经过几何优化与 sanity checks 的药物样分子。官方 `summary.csv` 为 2,026,848,085 bytes，SHA-256 为 `b6d7b54fa4d290ceace81c644f20b2ddfd68c21ecb1f4b5c00e8913cd608bcfd`；`structures.tar.gz` 为 7,180,016,346 bytes，SHA-256 为 `264102bf1c036d077a72ab558168be4c5c6054e6aeecb8a7768be36df87ad46b`。数据集页面标注 CC BY-SA 3.0，ChEMBL attribution 固定为 ChEMBL 27。

`scripts/prepare_qmugs.py` 下载并验证这两个官方文件，流式扫描 summary 与压缩 tar，不解压完整 structure tree。实际观察到 665,911 个唯一 ChEMBL 分子和 1,992,984 个 conformers；每个分子按 `GFN2_TOTAL_ENERGY` 最小值、再按 conformer ID 确定性选择一个 conformer。能量只用于 conformer selection，不写入 sample cache 或 selection table。

Seed `20260810` 产生两个互不重复的 workload。Population sample 取全体 ChEMBL IDs 的 `sha256(f'{seed}:{chembl_id}')` 最小 4,096 个，保持自然大小分布；它包含 226,648 atoms，总原子数 minimum/median/maximum 为 13/52/209，重原子数为 7/30/100。Size-balanced sample 再从 population 外按重原子数 4–10、11–20、21–30、31–40、41–50、51–65、66–80、81–100 八档各取稳定哈希最小的 512 个；它包含 339,795 atoms，总原子数 minimum/median/maximum 为 6/71/221。

仓库中的 `benchmarks/data/qmugs_sample.json` 只保存 source、license、sampling contract 与 selection filename，`qmugs_sample_structures.csv` 一行保存一个 ChEMBL/conformer source ID、atom counts、composition 和几何摘要。对应 SHA-256 分别为 `a764a23b001873f5b7f879212fcc324acfa57fa3d9918abdbba295d0664b462f` 与 `80e6d6ee9e8f80c63ca688da5d7c3d2577cb377c54885203d9f296f9ee6f5fdc`。Git-ignored deterministic NPZ cache 为 6.0 MB，SHA-256 为 `fa451cef3debb93a26ea42bc7740489e98124a56684aec755073606462b09c75`；NPZ entries 使用固定 ZIP metadata，因此相同 inputs 与 seed 会得到逐字节相同的 cache。

## QMugs correctness 与方法

QMugs structures 是 finite systems：每个 sample 的 `cell` 为零、`pbc` 全 false，cutoff 为 5 Å。CPU 使用 float64、固定 CPU 31、单线程和真实 `DataLoader(batch_size=1, shuffle=True, num_workers=0)`；CUDA 使用 float32、完整 batch 一次 H2D 后计时，主 workload 为 `batch_size=64`，另以 8/32/64/128 做 population batch-size scaling。与周期 benchmark 相同，data loading 和 H2D 不计时，公开 one-shot API、native metadata/search、allocation、必要同步与 Python/C++ boundary 计时。

CPU Vesin baseline 在全部调用和 repeats 间复用一个 `NeighborList(cutoff=5.0, full_list=True, sorted=False, n_threads=1)`。CUDA Vesin baseline 也复用一个 `NeighborList`，但 Vesin API 一次处理一个 structure，因此每个 DataLoader batch 内仍逐 structure compute 并拼接。Finite dense PyTorch baseline 独立 materialize 每个 structure 的 `N²` pairs，使用相同 strict cutoff 和 onsite exclusion；完整 epoch 不运行 dense，以免把不必要的中间张量成本重复数千次，主 batch 与八个 size bins 的 representative batch 均运行并计时。

CPU 与 CUDA 在全部 8,192 个分子、15,144,842 个完整 `(source, target, 0, 0, 0)` keys 上与 Vesin exact match。Population representative batch 与八个 size-bin representative batches 共 1,322,646 个 keys，全部与 dense baseline exact match。输出顺序仍不属于比较口径。

正式复现命令为：

```bash
/home/ftsong/.local/bin/uv sync --frozen --group dev
sudo sh -c 'echo performance > /sys/devices/system/cpu/cpu31/cpufreq/scaling_governor; echo performance > /sys/devices/system/cpu/cpu31/cpufreq/energy_performance_preference'
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' .venv/bin/python \
benchmarks/run_qmugs_cpu_benchmark.py --cpu 31 --repeats 11 \
  --warmup-seconds 2 --require-clean \
  --output benchmarks/results/threadripper-pro-9975wx-qmugs-cpu.json
sudo sh -c 'echo powersave > /sys/devices/system/cpu/cpu31/cpufreq/scaling_governor; echo balance_performance > /sys/devices/system/cpu/cpu31/cpufreq/energy_performance_preference'

PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=1 \
/home/ftsong/projects/elfes-workspace/elfes/.venv/bin/python \
benchmarks/run_qmugs_cuda_benchmark.py --repeats 11 --require-clean \
  --output benchmarks/results/rtx-pro-6000-blackwell-qmugs.json
```

CPU 正式 JSON 的 clean implementation revision 为 `dca205966ab5643451b0f4c7d97cbc7c11123c57`，Python 为 3.12.13，CPU extension SHA-256 为 `a2466a5b24d9a427fcb8076a7eb286d8e3d52db22f12aaac5792b6f812a7d302`。Runner 固定 CPU31 和 Torch/Vesin 单线程，并在 JSON 中记录 `amd-pstate-epp` driver、`performance` governor/EPP、boost 与 frequency bounds；runner 本身不修改系统 policy，上述命令在测量后恢复原始设置。CUDA 正式 JSON 的 clean implementation revision 为 `70a09d2fbe737a61677d68b3f5fbf1b685f2610e`，Python 为 3.12.3，CUDA extension SHA-256 为 `78c0af2e407fac90a332e227af5f73212d8c23cf0930756fcedaeffa3a4e495c`。

## QMugs CPU 结果

| Workload | Structures | Atoms | Pairs | tonari CPU | Vesin CPU reused | Vesin / tonari |
| --- | --: | --: | --: | --: | --: | --: |
| Population epoch | 4,096 | 226,648 | 5,320,936 | 169.700 ms | 169.554 ms | 1.00× |
| Size-balanced epoch | 4,096 | 339,795 | 9,823,906 | 303.687 ms | 281.308 ms | 0.93× |
| 4–10 heavy atoms | 512 | 9,096 | 138,600 | 9.917 ms | 11.868 ms | 1.20× |
| 11–20 heavy atoms | 512 | 15,846 | 297,044 | 12.862 ms | 14.283 ms | 1.11× |
| 21–30 heavy atoms | 512 | 23,953 | 515,654 | 17.396 ms | 17.997 ms | 1.03× |
| 31–40 heavy atoms | 512 | 31,782 | 766,106 | 22.943 ms | 22.666 ms | 0.99× |
| 41–50 heavy atoms | 512 | 41,717 | 1,103,462 | 31.166 ms | 29.780 ms | 0.96× |
| 51–65 heavy atoms | 512 | 55,512 | 1,658,682 | 46.267 ms | 42.553 ms | 0.92× |
| 66–80 heavy atoms | 512 | 71,480 | 2,292,736 | 69.588 ms | 58.151 ms | 0.84× |
| 81–100 heavy atoms | 512 | 90,409 | 3,051,622 | 92.743 ms | 77.884 ms | 0.84× |

QMugs 把 Matbench scaling 结论放到了更真实的 finite-molecule 分布上：4–30-heavy-atom bins 中 `tonari` 领先 1.03–1.20×，31–40 档基本进入 crossover，之后 Vesin 的成熟 CPU cell list 逐渐领先。Population headline 保持自然大小分布并恰好打平；size-balanced workload 刻意让八个区间等权，因而更突出大分子尾部并由 Vesin 领先约 1.08×。Reviewer 发现首版 CPU JSON 没记录 power policy 且无法复现后，本表在明确记录的 Python 与 CPU frequency policy 下重新测量；旧的 1.16× headline 已撤回。

## QMugs CUDA 结果

| Workload | Atoms | Pairs | tonari CUDA | Vesin GPU/structure | Vesin / tonari | Dense PyTorch | Dense / tonari |
| --- | --: | --: | --: | --: | --: | --: | --: |
| Population epoch，bs=8 | 226,648 | 5,320,936 | 45.471 ms | 923.253 ms | 20.30× | skipped | — |
| Population epoch，bs=32 | 226,648 | 5,320,936 | 12.309 ms | 915.940 ms | 74.41× | skipped | — |
| Population epoch，bs=64 | 226,648 | 5,320,936 | 6.730 ms | 914.764 ms | 135.93× | skipped | — |
| Population epoch，bs=128 | 226,648 | 5,320,936 | 4.049 ms | 913.805 ms | 225.68× | skipped | — |
| Size-balanced epoch，bs=64 | 339,795 | 9,823,906 | 7.346 ms | 934.513 ms | 127.21× | skipped | — |
| Population representative batch，bs=64 | 3,494 | 80,992 | 0.1128 ms | 14.1688 ms | 125.56× | 0.3031 ms | 2.69× |
| 4–10-heavy-atom representative batch | 1,167 | 17,758 | 0.1094 ms | 12.6545 ms | 115.71× | 0.2652 ms | 2.43× |
| 21–30-heavy-atom representative batch | 3,001 | 65,154 | 0.1124 ms | 14.3014 ms | 127.19× | 0.2691 ms | 2.39× |
| 51–65-heavy-atom representative batch | 6,945 | 211,056 | 0.1264 ms | 15.2762 ms | 120.88× | 0.4107 ms | 3.25× |
| 81–100-heavy-atom representative batch | 11,409 | 384,320 | 0.1425 ms | 15.9462 ms | 111.92× | 0.7740 ms | 5.43× |

QMugs 分子最多 221 atoms，因此 CUDA 全部走 fused exhaustive path；本轮刻意不通过复制单分子制造大体系 scaling。Batch-size sweep 是更符合分子 GNN 的 scaling 轴：同一 4,096 个分子从 bs=8 增加到 bs=128 时，调用次数从 512 降到 32，时间从 45.471 ms 降到 4.049 ms。Vesin 必须逐结构执行，batch size 对其约 914–923 ms 的总时间影响很小。

Finite dense baseline 不承担 periodic image padding，所以与 `tonari` 的差距远小于 Matbench 代表 batch 的约 190×；它仍要 materialize `N²` candidates，并随分子尺寸从 2.43×扩大到 5.43×。Population representative batch 的 Torch allocator peak 为 `tonari` 2.32 MB、dense 18.85 MB；81–100-heavy-atom batch 为 10.91 MB 对 176.16 MB。Vesin 的 native temporary allocations 仍可能不完全出现在 Torch allocator peak 中。

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
