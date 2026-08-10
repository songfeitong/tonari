# 工作记录

本文只记录项目为何形成当前设计，帮助理解主要取舍。它不是接口规范，也不保存每次调试、审查返工和 benchmark 波动；当前行为以 README、architecture、algorithm overview 和 design 为准。

## 2026-08-09：确定物理语义并实现 CUDA 原型

项目首先固定 neighbor-search 的物理约定：cell vectors 按行；periodicity 只由 `pbc` 决定；shift 施加在 target image；displacement 从 source 指向 target；cutoff 使用严格 `<`；periodic self-images 与 multiple images 必须保留。Vesin 只作为外部 reference 和 baseline，没有复制其实现。

第一版 CUDA search 已采用小体系 exhaustive、大体系 batched cell list 的总体路线。关键判断是把完整 batch 作为执行单位，不在 Python 中逐结构构图，也不 materialize `N² × images` candidate tensors。

早期 profiling 显示 Python/Torch metadata 的固定成本高于 raw kernels，因此 periodic geometry 和 schedule preparation 被逐步移入 native boundary。后续优化始终以完整 one-shot call 为测量单位，而不是只看单个 kernel。

## 2026-08-09：建立真实晶体 benchmark

`scripts/prepare_matbench.py` 从固定 revision 的 `matbench_mp_e_form` 中确定性抽取 1,536 个真实晶体，覆盖不同 atom counts、cell shapes 与 compositions。Raw data 和 cache 保持 Git ignored，仓库只提交 source 信息、抽样方法和 manifest；没有下载 OMat24，也没有保留无关 labels。

这套 workload 同时承担 correctness differential、真实 DataLoader epoch 和 real-derived supercell scaling，取代随机点与手工晶体作为主要性能证据。

## 2026-08-10：加入 CPU backend 与共享 geometry

CPU backend 没有包在 CUDA 路径外层，而是从公共 periodic geometry 和 pair policy 出发独立设计。小候选空间使用紧凑 exhaustive loop，大候选空间使用 Cartesian cell list；batch 内 structures 顺序处理，外层 workflow 决定并行度。

CPU 与 CUDA 共享 active-cell geometry、image range 和 pair identity，但保留各自的数据布局和调度。独立审查推动 canonical displacement predicate、empty-structure behavior、rank handling、unwrapped representatives 和资源边界在两端形成一致语义。

后续调查比较了大型周期体系的不同 cell-list 表示。每个原子只入表一次、查询时处理 periodic bin wrap 的设计在大型 CPU search 上具有更低常数，但当前实现已在常见小结构 workload 上表现良好，并与 CUDA 保持相近的数据流。考虑到真实需求与重构风险，项目选择暂不改变 CPU periodic-image strategy，等实际 CPU-only profile 出现明确瓶颈后再重新评估。

## 2026-08-10：统一公共 API 与 NumPy/PyTorch 语义

公共入口收敛为 `find_neighbors(positions, cells, pbc, cutoff, offsets=None)`，返回 `pair_indices` 与施加在 target 上的 `cell_shifts`。单结构与 batch、finite 与 periodic geometry 使用同一模型，旧接口不保留 alias。

NumPy 与 PyTorch 使用同一个 public function。随后项目进一步拆出 NumPy frontend、Torch frontend、独立 bindings 与 framework-neutral C++ core，使 NumPy CPU 不再借道 Tensor 或 LibTorch。CUDA 所需 metadata 和 schedule 也从 Python 收回 native provider。

这次重构确立了当前依赖方向：public API 只做 dispatch，frontend 处理数组生态，binding/provider 处理 native ownership，core 负责共享物理语义与 CPU search。

## 2026-08-10：补充真实分子 workload

周期晶体不能代表有限分子，因此项目选择 QMugs 而不是更小的 QM9。准备脚本从每个 ChEMBL 分子的 conformers 中确定性选择一个最低能结构，再构造自然分布和 size-balanced 两组互不重叠的 4,096-molecule samples。

结果显示 CPU 在自然分子分布上与 Vesin 基本打平，小分子区间领先、大分子区间逐渐落后；CUDA 的主要收益仍来自 batch amortization。这个趋势与 Matbench supercell scaling 一致，也明确了项目并不追求所有尺度上的 CPU 优势。

## 2026-08-10：原生 half list 与 zero-shift self

公共 API 增加 keyword-only `half_list` 与 `include_self`，默认行为不变。Full/half/self policy 进入共享 native candidate acceptance，而不是 Python 后处理。Vesin 与 ASE adapters 统一方向和 self 语义后，为四种模式提供外部 exact reference。

Half list 在 CPU 与 CUDA 上都真实减少 output pairs 和内存；zero-shift self 与 periodic self-images 保持清楚区分。独立审查重点验证这些不变量，没有围绕亚毫秒性能差异反复调优。

## 当前定位

项目现在是一套公共语义、三个 provider：NumPy CPU、Torch CPU 和 Torch CUDA。CPU 适合常见小结构、NumPy workflow 和无 GPU 环境；CUDA 面向模型入口的真实 batch；Vesin 与 ASE 保留为外部 baseline/reference。

尚未实现的 prepared workspace、Verlet cache、GNN adapter 和发布 wheel matrix 都应在出现真实需求后单独设计，不应通过继续扩大当前 API 来提前猜测。
