# 真实材料 benchmark

## 数据集与抽样

主要数据源是 ColabFit `matbench_mp_e_form`，dataset ID 为 `DS_5drebe4tktiu_0`。Source page 当前报告 132,741 个 configurations，而任务早期描述写的是 132,752；本仓库记录实际观察到的源数据值，不静默修正这一差异。固定的 Parquet object 是 `colabfit/Matbench_mp_e_form` revision `9880d5b9b62877ec5aa14d1a4c2a9ff4ee870b8d`、path `co/co_0.parquet`、128,655,162 bytes，SHA-256 为 `4b815791cc31862895b23cda7339d96217c37815c8f183949dc59b3035ee2afd`。

`scripts/prepare_matbench.py` 使用 seed `20260809` 确定性选择 1,536 个 structures。脚本对 atom count、cell-vector length anisotropy、最大 absolute inter-vector cosine 和 element count 使用固定 strata，以 `sha256(f"{seed}:{configuration_id}")` 为每个 stratum 内的稳定顺序，并在排序后的 strata 间 round-robin。样本覆盖 948 个 occupied strata 和 1,343 个 unique reduced formulas。Atom-count 在 0/10/25/50/75/90/95/99/100% 的 quantiles 分别为 1、4、9、24、58、134、176、322.6、444；cell anisotropy 范围为 1.0–52.27，cell skew 范围为 0.0–0.9966。所有源晶体都是 full PBC。已提交的 992 KiB manifest 包含 source configuration IDs、可用时的 Matbench names、compositions、atom counts、cell metrics、strata、source revision 和 selection method；123 MiB raw Parquet 与 1.5 MiB tensor cache 保持 Git ignored。

DataLoader 使用 `batch_size=32`、同一 seed 的 deterministic shuffle、标准 map-style `Dataset`、把 tensors 与 `ptr` 拼接起来的 custom collate、pinned CPU memory，以及每个完整 batch 一次 CUDA transfer。计时阶段遍历预先 transfer 到 CUDA 的 batches，因此明确排除 DataLoader 与 host-to-device transfer。每种 timing 都包含 warmup 和 device synchronization；完整 epoch 重复 7 次，single-batch/scaling cases 至少重复 12 次，表格报告 median。

## Baselines 与验证

Vesin baseline 使用 Vesin 0.6.1 `NeighborList(cutoff, full_list=True)`，在 CUDA tensors 上逐 structure 调用，再拼接结果并映射到本 API 的 `(source, target, S)` 方向。Dense baseline 独立实现 Equiformer/FairChem pattern：在 PyTorch 中 materialize batch-local `N^2` atom pairs 和 batch-wide padded image range；其 cutoff 与 onsite semantics 调整为被测 API 的精确定义。这是 style-equivalent exact-semantics baseline，并不声称未修改的 upstream Equiformer 会生成相同 edge keys，因为 upstream 使用 inclusive cutoff 和额外的 near-zero filtering。

计时前，production CUDA output 在全部 48 个 batches 上与 Vesin 精确一致：1,536 个 structures、2,780,158 个完整 five-component keys。Candidate count 位于 median 的 batch 还与独立 dense baseline 对 43,842 条 edges 精确一致。比较不依赖 output ordering。Equiformer V3 reference checkout revision 为 `a7300c58df683dc99cb48027d5bfd4c887486c48`；benchmark 的 Vesin version 为 0.6.1。

## 结果

硬件是一张 NVIDIA RTX PRO 6000 Blackwell Workstation Edition，compute capability 12.0；软件为 PyTorch 2.12.1+cu130、Python 3.12.3，geometry 使用 float32，cutoff 为 5.0 Å。运行前两张 GPU 均为空闲状态，实际选择 GPU 1。完整 machine-readable record 位于 `benchmarks/results/rtx-pro-6000-blackwell.json`，其中记录的实现 revision 为 `c9b5dfa3300bec5cee8a75fb6cc06fc8aa6b5de9`。

| Workload | Atoms | 新 batch CUDA | Vesin GPU/structure | Vesin / 新实现 | Dense PyTorch | Dense / 新实现 |
| --- | --: | --: | --: | --: | --: | --: |
| 1,536-structure DataLoader epoch | 75,238 | 34.127 ms | 489.831 ms | 14.35× | skipped | — |
| Median 32-structure batch | 1,126 | 0.821 ms | 9.299 ms | 11.33× | 42.757 ms | 52.10× |
| 真实结构，1×1×1 | 64 | 0.223 ms | 0.378 ms | 1.69× | 0.699 ms | 3.14× |
| 派生 supercell，2×2×2 | 512 | 0.294 ms | 0.702 ms | 2.39× | 6.672 ms | 22.66× |
| 派生 supercell，3×3×3 | 1,728 | 0.344 ms | 0.942 ms | 2.73× | 73.211 ms | 212.56× |
| 派生 supercell，4×4×4 | 4,096 | 0.346 ms | 0.920 ms | 2.66× | skipped | — |
| 派生 supercell，6×6×6 | 13,824 | 0.357 ms | 1.108 ms | 3.11× | skipped | — |
| 派生 supercell，8×8×8 | 32,768 | 0.409 ms | 1.635 ms | 3.99× | skipped | — |

Scaling source 是真实 64-atom configuration `CO_8661596785617876616983344`；只有 integer supercell repetition 是派生操作。Edge count 从 744 增至 380,928。Dense candidate estimate 从 1×1×1 的 110,592 增至 8×8×8 的 28,991,029,248，因此超过 150 million-candidate safety limit 的 runs 被跳过，避免无意义的 out-of-memory 风险。在 median batch 中，dense baseline 的 PyTorch allocator additional memory 为 6,915,438,080 bytes，新路径为 1,494,528 bytes；到 3×3×3 时，dense 使用 11,710,352,384 bytes。Vesin 的 allocator peak 不包含所有 native temporary allocations，不能解读为它的完整 memory footprint。

## Profiling 与解释

Nsight Systems 在 32,768-atom 派生 workload 上测得每次调用约 0.126 ms 的 CUDA kernels。最大的 kernels 是融合 representative wrapping 与 Cartesian bounds 的 48.3 µs、cell-list edge writing 的 35.5 µs，以及 edge counting 的 28.1 µs。一次实验把融合 atomic bounds 替换为独立 CUB block reduction，结果 bounds work 合计增至 106.4 µs，20-call NVTX range 也从 9.256 ms 增至 10.179 ms，因此该改动已撤销。由此可以排除 CUDA query kernels 中仍隐藏着尚未检查的更大 device hotspot。Machine-readable comparison 位于 `benchmarks/results/nsys-matbench-32768-summary.json`；raw profiler traces 保持 ignored，存放在 `runs/`。

目前 one-shot overhead 主要来自 metadata 与 exact-size allocation synchronization。当 `ptr/cells/pbc/cutoff` 不变时，reusable metadata API 可能节省实测约 0.177 ms，但 tensor mutation 下的安全行为需要明确 ownership/invalidation design，因此没有加入隐式 identity-based cache。Output count synchronization 可以通过 over-allocation 或 allocator/workspace contract 减少，但这会改变 memory behavior 和 public API。它们是有测量依据的后续方向，而不是臆测的 kernel micro-optimization。

这些数字是当前 workstation evidence，不是可移植的 performance guarantee 或测试阈值。Crossover 会随 atom-count distribution、density、cutoff、cell geometry、dtype、GPU、CUDA/PyTorch versions 以及 metadata 能否复用而变化。新路径在报告的每种 workload 上都更快，但最小 single-structure advantage 只有 1.69×；最强收益来自 heterogeneous batch parallelism，它消除了数千次逐结构 launches。
