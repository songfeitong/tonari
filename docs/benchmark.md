# 真实材料 benchmark

## 数据集与抽样

主要数据源是 ColabFit `matbench_mp_e_form`，dataset ID 为 `DS_5drebe4tktiu_0`。Source page 当前报告 132,741 个 configurations，而任务早期描述写的是 132,752；本仓库记录实际观察到的源数据值，不静默修正这一差异。固定 Parquet object 是 `colabfit/Matbench_mp_e_form` revision `9880d5b9b62877ec5aa14d1a4c2a9ff4ee870b8d`、path `co/co_0.parquet`、128,655,162 bytes，SHA-256 为 `4b815791cc31862895b23cda7339d96217c37815c8f183949dc59b3035ee2afd`。

`scripts/prepare_matbench.py` 使用 seed `20260809` 确定性选择 1,536 个 structures。脚本对 atom count、cell-vector length anisotropy、最大 absolute inter-vector cosine 和 element count 使用固定 strata，以 `sha256(f"{seed}:{configuration_id}")` 为每个 stratum 内的稳定顺序，并在排序后的 strata 间 round-robin。样本覆盖 948 个 occupied strata 和 1,343 个 unique reduced formulas。Atom-count 在 0/10/25/50/75/90/95/99/100% quantiles 分别为 1、4、9、24、58、134、176、322.6、444；cell anisotropy 范围为 1.0–52.27，cell skew 范围为 0.0–0.9966。所有源晶体都是 full PBC。

已提交的约 992 KiB manifest 包含 source configuration IDs、可用时的 Matbench names、compositions、atom counts、cell metrics、strata、source revision 和 selection method；约 123 MiB raw Parquet 与 1.5 MiB tensor cache 位于 Git-ignored `cache/`。没有下载 OMat24，也没有保留 energy/force labels。Scaling workload 只对样本中的真实 64-atom configuration `CO_8661596785617876616983344` 做整数 supercell repetition，因此属于真实结构派生 workload，不是随机点云。

## CPU benchmark 方法

CPU workload 使用标准 map-style PyTorch `Dataset` 和 `DataLoader(batch_size=1, shuffle=True, num_workers=0)`，shuffle generator seed 与抽样 seed 相同。DataLoader 先 materialize 为按真实顺序排列的 `StructureBatch` list，计时明确排除数据读取，只包含公开 one-shot `radius_graph_pbc` 调用；每次调用都包含 metadata、native search、output allocation 与 Python/C++ boundary。

本实现与 Vesin 都固定在 CPU 31 上运行。Backend 均为单线程：本实现内部不启 thread pool，Vesin 明确使用 `n_threads=1`。Vesin baseline 对每个 workload 构造一个 `NeighborList(cutoff=5.0, full_list=True, sorted=False)` 并在所有重复之间复用，因此它不承担 object reconstruction，属于对 Vesin有利的公平 baseline。Geometry 使用 float64；每个 backend/workload 先持续 warmup 至少 2 秒，再计时 11 次，scaling cases 至少 12 次；JSON 保存每个 sample、minimum、median 和 maximum。

正确性计时前，1,536 个 structures 的 production keys 与 Vesin 全量逐 structure exact comparison。方向统一为 `(source=vesin_second, target=vesin_first, S=vesin_shift)`，然后 lexicographically sort 完整 `(source, target, Sx, Sy, Sz)`，不比较 backend output order。最终 2,780,158 条 keys 全部一致。

CPU 正式复现命令为：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' \
/home/ftsong/projects/elfes-workspace/elfes/.venv/bin/python \
benchmarks/run_cpu_benchmark.py --cpu 31 --repeats 11 \
  --output benchmarks/results/threadripper-pro-9975wx-cpu.json
```

## CPU 结果

硬件为 AMD Ryzen Threadripper PRO 9975WX 32-Cores，32 个 physical cores、每 core 一个 hardware thread；正式进程 affinity 为 `[31]`。软件为 Python 3.12.3、PyTorch 2.12.1+cu130、Vesin 0.6.1。Machine-readable record 为 `benchmarks/results/threadripper-pro-9975wx-cpu.json`，implementation/method revision 为 `9e342ef815c596a09e81eca4cbb6ce6d102ba247`。

| Workload | Atoms | Edges | 本实现 CPU | Vesin CPU reused | Vesin / 本实现 |
| --- | --: | --: | --: | --: | --: |
| 1,536-structure DataLoader epoch | 75,238 | 2,780,158 | 123.103 ms | 248.713 ms | 2.02× |
| 真实结构，1×1×1 | 64 | 744 | 0.0348 ms | 0.0457 ms | 1.32× |
| 派生 supercell，2×2×2 | 512 | 5,952 | 0.1643 ms | 0.2297 ms | 1.40× |
| 派生 supercell，3×3×3 | 1,728 | 20,088 | 0.8525 ms | 0.7150 ms | 0.84× |
| 派生 supercell，4×4×4 | 4,096 | 47,616 | 2.4268 ms | 1.6498 ms | 0.68× |
| 派生 supercell，6×6×6 | 13,824 | 160,704 | 8.4654 ms | 5.4847 ms | 0.65× |
| 派生 supercell，8×8×8 | 32,768 | 380,928 | 20.1076 ms | 13.0951 ms | 0.65× |

完整 epoch 的 11 个本实现 samples 位于 123.035–123.282 ms，Vesin samples 位于 248.576–248.866 ms；精确值以 JSON 为准。固定 core 与 2 秒 warmup 是必要的方法细节：较短 0.5 秒 warmup 时，Threadripper 在累计约 1.2 秒后出现明显 frequency plateau change，若只报告最小值或短序列中位数会把 governor 行为混入算法结论。

结果的正确解释是“真实样本分布上的小/中型调用吞吐优势”，不是“大体系 cell list 全面超过 Vesin”。样本最多 444 atoms，epoch 中大量 structure 落在 native metadata 和低固定开销占主导的区间，因此总体快 2.02×；64 与 512 atoms 也单独领先。到 1,728 atoms，Vesin 已快约 1.19×；32,768 atoms 时快约 1.54×。Production 可以在 ELFES 常见体系中提供价值，同时保留对大体系继续优化或继续使用 Vesin 的诚实空间。

## CPU crossover 与低成本优化证据

CPU exhaustive/cell-list crossover 按 `N² × image_count` 判断。开发期在同一 1,536-structure epoch 上扫描 candidate limits 2,048、8,192、16,384、32,768、131,072，对应约 138.8、115.4、115.0、117.0、162.2 ms，因此选择 16,384。该 quick sweep 发生在正式 pinning/warmup protocol 完成前，只用于相对选择，不与正式表格混用；所有阈值的 correctness 相同。

最初 CPU prototype 在 Python/Torch 中逐项构造 duals 和 image shifts，epoch 约 285.7 ms，慢于 Vesin 的约 232.7 ms。将公共 1–3 维 periodic metadata 移入一次 native CPU call 后，短 protocol 降至约 136.7 ms；删除每 structure 都会调用、但只在环境变量开启时才打印的临时 high-resolution profiler 后进一步降至约 117 ms。最终固定-core steady-state 数字为 123.1 ms，不能与不同 protocol 的开发值直接比较，但优化方向在相同阶段的 A/B runs 中成立。

两项直觉优化被真实 workload 否决。将 bins 从 cutoff 改为 cutoff/2 把 32,768-atom candidate visits 从约 222 万降到 114 万，却因 neighbor-bin loops 增多把 query 从约 19 ms 推到 36 ms；利用反向边对称性只做一半距离判断再成对写出，也从约 18.5 ms 退到 20.9 ms。两者均已撤回，工作记录保留原因。

## CUDA benchmark 方法与既有结果

CUDA workload 使用同一 1,536-structure sample，但 DataLoader `batch_size=32`、pinned CPU memory，并将每个完整 batch 一次 transfer 到 GPU。计时排除 DataLoader 与 H2D，包含公开 one-shot API metadata、同步、分配和 CUDA work。Vesin 0.6.1 在 CUDA tensors 上逐 structure 调用并拼接；Equiformer/FairChem-style baseline 独立实现 dense periodic pair/image materialization，并调整为本 API 的 strict cutoff 和 onsite semantics。

硬件是一张 NVIDIA RTX PRO 6000 Blackwell Workstation Edition，compute capability 12.0；software 为 PyTorch 2.12.1+cu130、Python 3.12.3，geometry float32、cutoff 5 Å。正式 record `benchmarks/results/rtx-pro-6000-blackwell.json` 的 implementation revision 为 `a20ee8960c27161a568e3f54a026d0f9a43779de`。

| Workload | Atoms | 本实现 batch CUDA | Vesin GPU/structure | Vesin / 本实现 | Dense PyTorch | Dense / 本实现 |
| --- | --: | --: | --: | --: | --: | --: |
| 1,536-structure DataLoader epoch | 75,238 | 36.584 ms | 539.100 ms | 14.74× | skipped | — |
| Median 32-structure batch | 1,126 | 0.918 ms | 9.970 ms | 10.86× | 43.215 ms | 47.07× |
| 真实结构，1×1×1 | 64 | 0.241 ms | 0.386 ms | 1.60× | 0.701 ms | 2.91× |
| 派生 supercell，2×2×2 | 512 | 0.305 ms | 0.641 ms | 2.10× | 6.675 ms | 21.86× |
| 派生 supercell，3×3×3 | 1,728 | 0.315 ms | 0.872 ms | 2.77× | 73.519 ms | 233.60× |
| 派生 supercell，4×4×4 | 4,096 | 0.310 ms | 0.848 ms | 2.73× | skipped | — |
| 派生 supercell，6×6×6 | 13,824 | 0.340 ms | 1.038 ms | 3.05× | skipped | — |
| 派生 supercell，8×8×8 | 32,768 | 0.414 ms | 1.607 ms | 3.88× | skipped | — |

Dense candidate estimate 从 1×1×1 的 110,592 增至 8×8×8 的 28,991,029,248，因此超过 150 million-candidate safety limit 的 runs 被跳过，避免无意义 OOM。Median batch 中 dense baseline 的 PyTorch allocator additional memory 为约 6.9 GB，本实现约 1.5 MB。CUDA 收益主要来自 batch-first execution 和不 materialize candidates，不能拿来预测 CPU 单 structure 倍数。

## CUDA profiling

最终 CUDA revision 的 Nsight Systems trace 在 32,768-atom workload 上测得每次调用约 0.136 ms 的全部 CUDA kernels，另有约 0.0065 ms CUDA memory operations；20 次 NVTX range 为 9.476 ms，即每次约 0.474 ms。最大的 kernels 是 fused representative wrapping/bounds 的约 48.5 µs、cell-list writing 的约 37.3 µs 和 counting 的约 36.4 µs。较早实验把 fused atomic bounds 换成独立 CUB reductions，bounds work 合计增至约 106.4 µs，NVTX wall range 也变差，因此撤销。

Machine-readable summary 与完整 kernel、memory operation、CUDA API、NVTX CSV 位于 `benchmarks/results/`；raw `.nsys-rep` 保持 ignored。CPU 没有使用 privileged `perf`，因为本机 `perf_event_paranoid=4`，本任务遵循不以 sudo 修改 host policy 的约束；CPU optimization 依据 wall time、显式临时 counters 和全量 correctness A/B。

## 结论边界

所有数字都只代表当前 workstation、固定 software revision、5 Å cutoff 和指定 workload。CPU 结果对 affinity、frequency warmup、dtype、structure distribution 与 Vesin reuse policy 敏感；CUDA 结果对 batch composition、GPU、CUDA/PyTorch version 和 memory pressure 敏感。Benchmark 数字不写成 unit-test 阈值，也不把一次历史快照当作可移植性能承诺。
