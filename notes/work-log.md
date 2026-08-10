# 工作记录

## 2026-08-09：物理约定与第一版 CUDA search

首先从 ELFES theory/design 固定不可妥协的物理约定：cell vectors 按行；periodicity 只由 `pbc[3]` 决定；active rows 可以形成 rank-1/2 lattice；每个 pair 的连续量是 `positions[source] - positions[target] + cell_shifts @ cell`；cutoff 使用严格 `<`；只排除 source、target 与 zero shift 同时相同的 onsite pair；periodic self-images 和同一 atom pair 的 multiple images 都必须保留。Vesin 0.6.1 只作为外部 correctness reference 与 baseline，没有复制或移植其源码。

第一版 CUDA 路径没有 materialize `N² × images` tensors，而是把候选映射到 CUDA threads，先计数、精确分配，再写出结果。小 workload 使用 exhaustive search；大 workload 使用 batched Cartesian cell list。Cell-list pipeline 在 device 上完成 representative wrapping、AABB、periodic source-image insertion、每 target 查询、prefix sum 与输出。异质 batch 从入口到输出只经过一组 native launch sequence，而不是 Python 逐结构循环。

早期 Python metadata 对代表性 batch 约 2.59 ms，明显超过 raw kernels，因此优化重点先放在边界层。将 1–3 维 periodic geometry 移入 native shared metadata 后，固定成本显著下降。大结构路径选择 cutoff-sized Cartesian bins；cutoff/2 bins 虽减少 visited nodes，却把 neighbor-bin loops 从最多 27 增到最多 125，端到端反而变慢，已经撤销。

Nsight Systems 还否决了一项看似合理的优化：把 fused wrapping/AABB atomics 拆成独立 CUB reductions 后，wrapping kernel 变小，但 reductions 本身更贵，总 GPU work 和 NVTX wall time都回退。这个结果促使项目始终以完整 one-shot call 为性能单位，而不是单看某个 kernel。

## 2026-08-09：真实数据与 CUDA 边界加固

ColabFit 直接 XYZ 下载在本机遇到 reCAPTCHA，因此改用同一官方数据的固定 Hugging Face Parquet revision，并保存文件大小与 SHA-256。`scripts/prepare_matbench.py` 从 `matbench_mp_e_form` 确定性抽取 1,536 个 structures，覆盖 1–444 atoms、1,343 个 formulas 和多种 cell shapes。Raw data 与 tensor cache 保持 ignored；仓库只提交 manifest 和复现脚本；没有下载 OMat24，也没有保留 labels。

第一轮独立审查发现：极端未 wrap representatives 可让 int32 `cell_shifts` 静默溢出；empty periodic structure 会按 tiny cell 枚举大量 images；NaN/Inf positions 或 inactive cell rows 会静默给出错误结果；过早的 exhaustive grid limit 会拒绝本可走 cell list 的输入。最终保持输出为 int32，并要求 representative wrap 与最终 shift 都可表示，否则直接报错；empty structure 使用零 image count；finite/range 检查进入 O(N) prepare；grid limit 只在实际选择 exhaustive path 时生效。

边界检查的第一版把 finite predicate 重复放进每个 candidate 并增加同步，真实 epoch 回退约 22%。修复后检查恢复为 O(N)，error status 与已有 synchronization 合并，性能回到原区间。这个过程确认 correctness hardening 也必须经过真实 workload profile。

## 2026-08-10：CPU backend 与共享 geometry

CPU 工作没有在 CUDA API 外加一个 device `if`。实现先拆出真正共享的 periodic geometry，再分别设计 native CPU hybrid search 与 CUDA schedule。CPU 可以接受 batch，但逐 structure 搜索；CUDA 继续整个 batch pipeline。Native build 始终包含 CPU extension，CUDA extension 按环境可选。

旧 metadata 使用多个小型 Torch operators；在 `batch_size=1` 的 1,536 次调用中 dispatcher 成为主要成本。新的 native geometry 在一个 C++ call 中完成 finite/rank check、active dual、repeat range 和 image enumeration，并用 long-double one-sided Jacobi SVD 直接处理最多 3×3 active rows，避免 Gram matrix 平方条件数。

CPU production 采用 hybrid search：候选规模小时 exhaustive，规模大时把相关 periodic source images 插入 cutoff-sized Cartesian bins，每个 target 扫描相邻 27 bins。内部 crossover 按 `N² × image_count` 的真实候选规模选择；完整 Matbench epoch sweep 后选定 16,384。CPU 不创建内部 thread pool，使 DDP、DataLoader workers 或调用者可以控制并行层级。

两项 CPU 优化被 benchmark 否决。利用 reverse-pair 对称性只计算一半距离增加了 branch 与 paired writes，32,768-atom case 反而从约 18.5 ms 退到 20.9 ms；把 linked nodes 改为额外排序的 contiguous bins 也没有端到端收益。真正有效的低成本改进包括 native metadata、合理 output reserve、紧凑 int32 nodes、只插入 target bounds 附近的 periodic images、`gil_scoped_release`，以及删除即使 profiler 未开启也会执行的高频 `steady_clock::now()`。

## 2026-08-10：CPU 独立审查与统一语义

CPU reviewer 找到四个核心反例：corner-bin pruning 在 strict boundary 附近只保留一个方向；empty structure 仍在 rank/repeat 前处理 tiny cell；Gram rank check 拒绝合法近共线 rows；wrapped-coordinate cutoff 对大 representatives 发生 cancellation。处理方式不是添加 magic epsilon，而是删除低收益 pruning、empty early return、直接 SVD，以及把 original positions/output shift 的公共公式作为 canonical predicate。

第二轮审查又找到 CUDA cell-list 与 CPU 对 float32 大共同晶格平移的 pair-set 分歧。为了避免复制另一套容易漂移的 boundary-shell 逻辑，CUDA prepare 检测 nonzero wraps 后将 whole call 路由到已有 canonical exhaustive path。随后又修复 batched sparse bin counts 单项不溢出但 cumsum 溢为负数的问题：超过实际用途的 bin count 在 device 上饱和为 safety limit + 1，host 稳定选择 sparse fallback。

最终 reviewer 对 12,000 个 rank/scale/condition-number geometry cases、数千个 strict-boundary/unwrapped/sparse differential cases、全部 1,536 个真实 structures 与 2,780,158 个 Vesin keys 做复核，在 clean revision `3851726124e9db81859682fc3f7e3c9a2231d310` 给出 PASS。

## 2026-08-10：第一性原理 API 重构为 tonari

项目随后重命名为 `tonari`，并删除旧 API，而不是保留 alias。公共面只剩 `find_neighbors(positions, cells, pbc, cutoff, offsets=None)`；`offsets=None` 是单结构，batched `offsets` 是拼接 positions 的 boundaries。返回 `pair_indices` 与 `cell_shifts`，方向统一为 source、target，shift 施加在 source 上。核心层只使用 neighbor、pair、source、target、cell shift、displacement 与 distance；只有未来 PyG/GNN adapter 才把结果映射为 `edge_index` 和 `edge_vectors`。

同一个入口现在按输入生态 dispatch。Torch 输入返回同 device 的 Torch tensors，CPU/CUDA 分别进入 native backend；NumPy 输入通过 `torch.from_numpy` 进入相同 native CPU backend，再以共享 storage 的安全方式返回 NumPy arrays。不可写、未对齐或 negative-stride arrays 只在必要时复制；NumPy/Torch 混用被明确拒绝。公共 docstring 使用 Torch style，覆盖 shape、dtype/device、batch、方向、strict cutoff、periodic images、单位、autograd 与可运行示例。

这次重构还发现一个不常见但真实的 CPU 性能因素：Ninja 按 object filename 排序链接，重命名后 hot function 地址变化，完整 epoch 从约 144 ms 退到约 165 ms，而 machine code bytes 完全相同。将共享实现命名为 `geometry.*` 后，link order 和 hot address 恢复，性能也恢复。这里只记录经同进程 direct native A/B 证明的现象，不把它包装成可移植的通用优化规律。

API 重构后共 68 项 CUDA-visible tests 全过，CPU-only 环境为 46 passed/22 skipped，公共 docstring 10 个 examples 全过。新增测试覆盖 NumPy/Torch 同一入口、ecosystem mixing rejection、single/batch 等价、CPU/CUDA/reference exact identity，以及 positions/cells/cutoff 一致缩放后的单位无关性。

## 2026-08-10：最终真实 benchmark

CPU 正式 run 使用 `DataLoader(batch_size=1)`、float64、固定 CPU 31、单线程、每 backend/workload 至少 2 秒 warmup。全部 Matbench keys 与复用 `NeighborList` 的 Vesin exact match。tonari 完整 epoch 为 143.802 ms，Vesin 为 248.076 ms，即 tonari 在真实小体系分布上快 1.73×；64 atoms 仍领先，约 512 atoms 到交叉区，1,728 atoms 以上 Vesin 明显更快。

CUDA 正式 run 使用 `DataLoader(batch_size=32)`、float32、整 batch H2D 后计时。tonari 完整 epoch 为 12.024 ms，逐结构 Vesin GPU 为 493.940 ms；median 32-structure batch 为 0.2227 ms，对应 Vesin 9.3111 ms 与 dense PyTorch 42.7788 ms。64–32,768 atom真实结构/派生 supercells 上，tonari CUDA 均领先逐结构 Vesin；dense baseline 从 4,096 atoms 起因候选规模超过安全限额跳过。

最终 Nsight profile 在 32,768 atoms 上测得全部 CUDA kernels 约 0.1400 ms/call，CUDA memory operations 约 0.0061 ms/call，NVTX one-shot range 约 0.3138 ms/call。Raw trace 保持 ignored；仓库提交复现脚本、summary 和完整 CSV aggregates。ELFES 只做只读需求核对：其当前 two-center 用法可由未来 adapter 在 scalar broad cutoff 后做 half-list canonicalization、species filtering 与 onsite addition，本轮没有修改或接入 ELFES。
