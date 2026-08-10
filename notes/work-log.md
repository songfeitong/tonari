# 工作记录

## 2026-08-09：约定与参考

先从 ELFES 的 theory/design 固定物理约定：cell vectors 按行，PBC 只由 `pbc[3]` 决定，active rows 可形成 rank-1/2 lattice，edge key 为 `(source, target, S)`，连续向量是 `r_source - r_target + S @ cell`，严格 `< cutoff`，仅排除 `(i, i, 0)`。Vesin 0.6.1 只用于外部 reference 和逐结构 baseline；实现没有复制或移植 Vesin 源码。Equiformer V3 参考 revision 为 `a7300c58df683dc99cb48027d5bfd4c887486c48`，Vesin 调研笔记记录的源码 revision 为 `ae2297613649e672e02537bd8eaea70dc5afcdb9`。

## 第一阶段：融合 exhaustive CUDA prototype

第一版没有 materialize `N^2 × images` tensors，而是把候选直接映射到 CUDA threads，先 block reduction 计数，再精确分配并用 block scan 写出。它很快得到 reference/ASE 精确正确性，并天然支持 batch 内不同 cell/PBC pattern；但算法工作量仍随 atom pair 数二次增长。最初 Python metadata 对代表性多结构 batch 约 2.59 ms，按 PBC pattern 批量计算 dual/repeat 并缓存重复 image ranges 后约 1.0 ms，显著高于同一时刻约 0.13 ms 的 raw kernel，因此先优化了边界层而不是盲调 CUDA 算术。

## 第二阶段：Cartesian cell list

大结构路径独立设计为整个 batch 一次 launch sequence：active-axis wrapping、Cartesian AABB、periodic source-image insertion、每 target warp 查询 27 bins、per-target count prefix sum、再写出。以 256 atoms 作为当前 crossover，小结构继续走低启动成本 exhaustive path。极端稀疏 finite geometry 会造成巨大 dense bin box，因此增加 `2^28` total bins 和 64 bins/node 两个明确的安全阈值，超限时回退 exhaustive path；这不是物理 tolerance。

在 Matbench 的 Si-like 真实结构派生 quick runs 中，cell-list 路径相对逐结构 Vesin GPU 从 512 atoms 的约 2×提升到 32,768 atoms 的约 3–4×，而 heterogeneous batch 的收益更大。所有 1,536 个正式样本随后逐 batch 与 Vesin 做完整 key equality，未发现重复或遗漏。

## 失败路线与 profile

Nsight Systems 指出 32,768 atoms 时最大的 device hotspot 是 wrapping kernel 内对六个 AABB 数值的 atomics，平均 48.3 µs。尝试把它拆成单 structure 的六次 CUB block reductions 后，wrapping 降到 1.6 µs，但 reduction 本身升到 104.8 µs；总 GPU time 和 NVTX wall range 都回退，故立即撤销。这里的 atomics 竞争直觉没有胜过 Blackwell 上的实际测量。

当前单次调用中 metadata 约 0.177 ms、extension 加同步约 0.323 ms。缓存 metadata 有明确潜在收益，但不能安全地仅按 Tensor object identity 缓存，因为原地修改 cell/PBC 会得到 stale graph；复制并 hash 又会吃掉收益。因此本轮保留无状态 one-shot API，把 reusable prepared metadata 作为需要单独所有权设计的后续方向。精确 edge allocation 的两次同步也可以用 workspace/over-allocation 改写，但会改变内存和 API contract，不属于低风险微优化。

## 数据与工具链

ColabFit 页面提供的直接 XYZ 下载在本机遇到 reCAPTCHA，因此使用同一官方数据页链接的 Hugging Face Parquet，并固定 repo commit、文件大小和 SHA-256。原始 123 MiB Parquet 与 1.5 MiB sample cache 位于 Git ignored `cache/`；仓库只提交约 992 KiB manifest 和可重复脚本。没有下载 OMat24，也没有保留 energy/force labels。

复用了 ELFES 的 Python 3.12/PyTorch 2.12.1+cu130/PyG 环境。最初 `setup.py build_ext --inplace` 成功，但第一次标准 `uv sync` editable build 暴露绝对 source path 不符合 setuptools wheel 规则；改为相对路径后，`uv sync --active --frozen --all-groups --inexact --no-build-isolation`、editable import、26 tests 和 CUDA memcheck 全部成功。系统 nvcc 13.2 与 wheel CUDA 13.0 有 minor mismatch warning，当前机器实测可用，但正式 toolchain 应尽量 minor 对齐。

## 独立终审与修复

GPT-5.6 high reasoning 独立审查确认了三个 correctness 边界：极端未 wrap representatives 会让 int32 shift 静默溢出；empty periodic structure 仍会按 tiny cell 枚举大量 images；NaN/Inf positions 或 inactive cell rows 会静默生成错图。最终选择保持 public `cell_shifts` 为 int32，并要求每个 representative periodic wrap 本身也能由 int32 表示，越界直接报错；per-atom prepare/wrap kernel 先验证 finite/range，再以 int64 只计算两个已验证 int32 wrap 的差。Empty structure 直接使用零 image count，cells finite 检查复用 metadata CPU copy，positions flag 融入 CUDA pass。

审查还发现 dense `block_ptr >= 2^31` 的早期检查会阻止本可走 cell list 的 741,456-atom finite workload。限制已移动到真正选择 exhaustive path 或 sparse-bin fallback 的位置；cell-list node 总数仍明确限制在 int32 range。

第一版 error flag 修复把 finite 检查重复放到每个 `N^2 × images` candidate，并增加多个 D2H/zero-fill launches，导致 Matbench epoch 从约 34.1 ms 回退到 40–41 ms。最终实现改为 O(N) per-atom wrap preparation，用 cumsum sentinel 将 error 与原 count 一次返回，并只对四个 8-byte status slots 使用 `cudaMemsetAsync`；独立复测恢复到 34.60 ms epoch、0.845 ms median batch 和 0.396 ms 的 32,768-atom case。该过程说明边界验证也必须进入真实 workload profile，不能只看功能测试。

在最终边界修复 commit `a20ee8960c27161a568e3f54a026d0f9a43779de` 上又完整跑了一次正式 benchmark：1,536 个结构的 epoch 为 36.584 ms，逐结构 Vesin 为 539.100 ms，即 14.74×；median batch 为 0.918 ms，对 Vesin 为 9.970 ms、对 dense baseline 为 43.215 ms。最终 Nsight trace 的 32,768-atom case 中，全部 kernels 平均每次约 0.136 ms，20-call NVTX range 为 9.476 ms。Raw trace 保持 ignored，但完整 kernel、memory operation、CUDA API 与 NVTX CSV summaries 已提交，避免性能结论只能依赖手写摘录。

## 2026-08-10：从 CUDA prototype 重构为 CPU/CUDA system

CPU 任务没有把一个 `if device == cpu` patch 塞进旧 CUDA metadata。先把原 `SearchMetadata` 拆成真正公共的 duals/image shifts/counts 和独立 `CudaSearchSchedule`，再把 native build 拆成始终存在的 `_C_cpu` 与可选 `_C_cuda`。公开函数、edge semantics、dtype 和 batched shapes 保持一套；CPU 可以处理 batch，但内部按 structure 顺序搜索，CUDA 保留整个 batch pipeline。这样最终目录结构表现得像 CPU/CUDA 从第一天就是共同需求，而不是一套主实现加一套附属 fallback。

旧 Python metadata 使用 PBC pattern grouping、`torch.linalg.svdvals/inv`、Python `product` 和 device tensor construction。它在单次 CUDA batch 中尚可接受，但 CPU `batch_size=1` 的 1,536 次调用会把小算子 dispatcher 变成主要成本。新 `_C_cpu.build_periodic_metadata_cpu` 在一个 C++ call 中完成 finite/rank check、active dual、repeat range 和 image enumeration；最终实现直接对最多 `3 x 3` active rows 做 long-double one-sided Jacobi SVD，避免 Gram matrix 把条件数平方。CPU/CUDA 都复用同一结果。

## CPU prototype 与 hybrid search

第一版 CPU exhaustive reference 很容易正确，但真实晶体的 `N² × images` 很快放大，因此 production 直接设计为 per-structure hybrid。候选数不超过 threshold 时直接遍历；否则把相关 periodic source images 插入 cutoff-sized Cartesian bins，每个 target 扫描相邻 27 bins。Cell list 使用 dense bin heads 与 int32 linked nodes；病态稀疏 finite coordinates 若需要过大 dense grid，就回退 exhaustive，优先避免空 grid OOM。早期 target-to-bin AABB pruning 后来因严格边界缺陷和收益过小被删除。

初始 Python metadata + native hybrid 在真实 epoch 为约 285.7 ms，Vesin 约 232.7 ms；native metadata 后降到约 136.7 ms。随后发现临时 profiler 即使环境变量未开启，也为每个 cell-list structure 调用了多个 `steady_clock::now()`；删除这些开发计时点后短 protocol 降到约 117 ms。这个优化非常朴素，却比许多内循环算术改写更有价值，说明高频 API 的 instrumentation 本身也必须接受 benchmark。

## CPU crossover sweep

Exhaustive 的成本是 `N² × image_count`，所以没有沿用 CUDA 的 256-atom threshold。对完整 Matbench epoch 扫描 candidate limits 2,048、8,192、16,384、32,768、131,072，quick-run medians 分别约为 138.8、115.4、115.0、117.0、162.2 ms；最终选择 16,384。131,072 会让 64-atom periodic structure 误走 exhaustive，单次从约 0.033 ms 退到 0.173 ms；2,048 又让太多很小 structures 过早支付 cell-list setup。该 sweep 只冻结 performance choice，不冻结算法输出。

## 被否决的 CPU 优化

临时 profiler 把 32,768-atom case 分成 bin layout、image insertion 和 query，显示 query 为主要热点。将 bin edge 从 cutoff 缩小到 cutoff/2 后，visited nodes 从约 2.22 million 降到 1.14 million，但邻近 bins 从最多 27 增加到最多 125，query 从约 19 ms 退到 35.7 ms，立即撤销。

另一个方案利用 full directed radius graph 的 reverse symmetry：只对 `(source,target,S)` 与 `(target,source,-S)` 中 canonical 的一半计算距离，命中后同时 append 两条 edges。它能少做距离算术，却仍需遍历 linked nodes，并增加 branch 和 paired vector writes；32,768-atom wall time从约 18.5 ms 退到 20.9 ms，完整 epoch也没有收益，因此撤销。还短暂尝试把 linked nodes 排序为 contiguous bins，cache locality 的理论优势同样没有转化为端到端改善。

保留的低成本优化包括：native periodic metadata、cutoff-sized bins、只插入 target bounds 附近的 periodic images、紧凑 int32 nodes、预留合理 output capacity，以及 pybind `gil_scoped_release`。CPU implementation 不启动内部线程，以便 DDP/DataLoader workers 能由调用方控制并行层级。

## CPU correctness 与真实数据

CPU tests 新增 mixed finite/partial/full batch、float32/float64、ASE triclinic partial PBC、small-cell multiple images、periodic self-images、strict cutoff、unwrapped representative relabel、continuous geometry backward、tiny-cell empty structure、cell-list path、sparse bounds fallback、NaN/Inf、dependent 与近共线满秩 active rows、int32 wrap/shift errors、periodic image resource limit、O(3) rotation/reflection 和 deterministic randomized differential。最终与既有 CUDA/reference tests 合计 53 项，另有 290 组开发期 CPU/reference/CUDA differential cases。

Formal benchmark 仍使用原来固定 revision 与 SHA-256 的 `matbench_mp_e_form` 1,536-structure sample，但 DataLoader 改为真实单 structure workflow：`batch_size=1`、deterministic shuffle、float64。Vesin baseline 使用一个跨重复复用的 `NeighborList(full_list=True, sorted=False, n_threads=1)`；计时前对全部 1,536 structures、2,780,158 keys 做 exact comparison，全部一致。

## CPU timing protocol 修正

第一次 formal run 未固定 affinity，完整 epoch 出现约 115/137 ms 两个平台。固定到 CPU 31 后迁移消失，但 0.5 秒 warmup 仍在累计约 1.2 秒处出现 frequency plateau change。最终 protocol 对每个 backend/workload warmup 至少 2 秒、固定 affinity、保存所有 samples，然后报告 median。边界加固前 revision `9e342ef815c596a09e81eca4cbb6ce6d102ba247` 的历史结果是 123.103 ms；独立审查后不沿用该数字。最终 clean revision `bd30fa1e50b785aea9cb9242d3889f171dd201db` 上，本实现 epoch 为 143.554 ms，复用 Vesin 为 248.190 ms，即 1.73×；64 atoms 为 0.0411 对 0.0457 ms，512 atoms 为 0.2391 对 0.2286 ms。从 1,728 atoms 起 Vesin 明显领先，32,768 atoms 为本实现 24.041 对 Vesin 13.137 ms。正式 JSON 同时记录了 worktree clean flag、cache hash 与实际 extension hash。

这个结果符合任务目标但也限定了结论：ELFES 当前 two-center 使用 scalar broad cutoff、单 Geometry 和 single-thread Vesin，CPU backend 在其常见小 structure 区间具有明确潜力；不过 ELFES 还需要 half-list canonicalization、onsite 和 species post-filter adapter，而且 512 atoms 左右已经到达本机交叉区，大体系 Vesin 更成熟。本轮保持 ELFES 完全只读，没有为了“去依赖”提前接入未冻结的 adapter。

## CPU 独立终审与边界加固

CPU 初版完成后，新的 GPT-5.6 high reasoning reviewer 只读检查了实现、真实 benchmark、数据 provenance、格式和 Git hygiene，并独立重跑 tests、随机差分与全部 Matbench/Vesin keys。它确认四个必须修复的问题：cell-list corner pruning 在 `nextafter(cutoff, 0)` 附近可能只保留一个方向；empty structure 仍会为极小 active cell 准备巨大 image range；Gram rank check 会拒绝 SVD 明确认定满秩的近共线 rows；大但仍在 int32 wrap 范围内的未 wrap representatives 会因 wrapped-coordinate cancellation 漏掉 cutoff 内 edge。

最终处理不是给复现加 magic epsilon：删除收益很小的 corner-bin pruning；empty structure 在 rank/repeat 前短路；以直接 one-sided Jacobi SVD 取代 Gram normal equations；cell list 用按输入量级推导的保守 broad-phase 误差带，并只在边界壳按原始 positions/output shift 重算 public strict-cutoff formula。另对 periodic image Cartesian product 增加 checked `2^24` resource limit，防止合法但极小的非空 cell 在 host 上不可控 OOM。

第一轮修复后 50 项 tests、290 组额外 differential cases 和全部 1,536 个真实结构/2,780,158 keys 均通过。正式 benchmark 从旧版 2.02× 收敛到最终 1.73×；团队接受这个变化，因为旧数字没有覆盖同等严格的数值边界，而最终结论仍然清晰。Benchmark runner 同时补上 clean-worktree enforcement 和 cache/binary SHA，回应 reviewer 对 provenance 的意见。

第二轮复审又发现 CPU 与 CUDA cell-list 对合法 float32 大共同晶胞平移有四个 key 的分歧：CPU/exhaustive CUDA 已按 original positions/output shift 判断，而 cell-list CUDA 仍把 wrapped arithmetic 当作 strict predicate。最终没有复制一份新的 CUDA boundary-shell 判据，而是在现有 prepare/status/cumsum 流程中检测 nonzero wrap，并将整个调用路由到已有 canonical exhaustive CUDA implementation。Standalone 256-atom regression 把旧差异放大到 508 keys，并固定两个漏边与两个伪边；修复后 CPU、新 reference 与 CUDA 都得到 64,264 个相同 keys。Reference 同时改为 public original-position formula，并获得与 production 相同的 `2^24` image guard。

最终 53 项 tests、完整 Matbench/Vesin 与 dense spot exact validation、CUDA memcheck 和 290 组 differential cases 全部通过。Reviewer 的 synthetic scaling 显示 unwrapped fallback 在 256/512/1,024/2,048 atoms 相对 well-wrapped cell list 慢约 1.03×/1.41×/2.88×/8.48×；它被保留为明确的非阻塞限制，因为验收要求统一正确语义，不要求未 wrap 大体系仍维持 cell-list 复杂度。

最后的资源复审构造了两个各 256 atoms、Cartesian extent 1,700,000 Å 的 finite members。旧版每个 structure 的 bin count 约 `4.913e18`，单独仍在 int64 内，但 batched cumsum 先溢成负数，未能进入预期的 sparse-bin fallback。最终 `define_bins_kernel` 不再计算超过用途的精确巨大数：任一 dimension/product 超过 `2^28` safety limit 就饱和为 `limit + 1`，host据此稳定选择 exhaustive；新增 batched regression 与 CPU key set 完全一致。
