# 独立终审记录

## 审查原则

独立 reviewer 对 clean Git revision 做只读审查，不参与作者实现。审查范围包括 periodic geometry、strict cutoff、source/target 与 cell-shift 方向、CPU/CUDA/NumPy/Torch 一致性、stream/device/autograd、resource bounds、真实 Matbench/Vesin evidence、benchmark provenance、文档和 Git hygiene。审查发现以可重复输入和具体 key 差异为依据；性能问题使用同一 workload A/B，而不凭代码直觉定性。

## CPU backend correctness closure

CPU backend 初版审查确认并修复了以下问题：cell-list corner pruning 在 `nextafter(cutoff, 0)` 附近会破坏 full-directed 对称性；empty structure 会为极小 active cell 准备巨大 image range；Gram normal equations 会平方条件数并拒绝合法近共线 cells；大但仍合法的未 wrap representatives 会因 wrapped-coordinate cancellation 漏 pair；合法极小非空 cell 的 image Cartesian product缺少 fail-fast；CUDA cell-list 与 CPU canonical predicate 对 float32 大共同晶格平移不一致；batched sparse-bin counts 可在 int64 cumsum 中溢出。

最终实现删除无价值的 corner pruning，empty structure 在 rank/repeat 前短路，以 long-double one-sided Jacobi SVD 直接处理 active rows，cell-list broad phase 在数值边界回到 original positions 与 output shift 的公共公式，并对 image count 和 bin count 使用 checked/saturating resource guards。CUDA 检测到非零 representative wraps 时转入已有 canonical exhaustive path，避免维护第三份 strict-cutoff predicate。

修复后的独立证据包括：53 项当时的 production tests；1,200 个 inner-shell/strict-boundary cases；500 个 float32 individual-wrap cases；400 个 common-wrap Vesin cases；12,000 个 rank/scale/condition-number metadata fuzz；全部 1,536 个 Matbench structures 和 2,780,158 个 Vesin keys；mixed batch、non-default CUDA stream、extreme sparse inputs、CPU-only skip matrix、Ruff 与 Git checks。Reviewer 在 clean HEAD `3851726124e9db81859682fc3f7e3c9a2231d310` 给出 PASS，无 remaining correctness blocker。

## API 重构验收证据

项目随后从第一性原理重构为 `tonari`：公共面只保留 `find_neighbors`；`offsets=None` 与 batched `offsets` 是同一契约；返回值统一为 `pair_indices` 与施加在 source 上的 `cell_shifts`；Torch CPU、Torch CUDA 和 NumPy 共享同一个入口，NumPy 复用 native CPU backend；旧 package、symbol 和 alias 全部删除。

重构后的作者侧验证包括 74 项 CUDA-visible tests、50 项 CPU-only tests 加 24 项 expected skips、10 项可运行 docstring examples、Ruff、offline lock check、CPU/CUDA 全量 Matbench/Vesin exact differential、dense CUDA spot comparison、正式 CPU/CUDA benchmark 与 Nsight profile。特别增加了单结构/Batch、生态混用拒绝、NumPy zero-copy-safe adapter、CPU/CUDA一致性以及 `positions`、`cells`、`cutoff` 同比缩放后的 neighbor identity 不变性。

本轮独立 reviewer 又确认两个重构边界。第一，0-D Torch/NumPy positions 在默认 offsets 构造中先触发 `len()` TypeError，与 docstring 的 invalid-shape ValueError 不一致；shape validation 已移动到 normalization 之前，并以 0-D/1-D 两种生态回归固定。第二，CUDA cell-list 先把 cutoff cast 成 float32 再平方，而 CPU/CUDA exhaustive/reference 先在 double 中平方再 cast，导致 255/256-atom crossover 在 1 ulp strict boundary 得到不同 pairs；cell-list query 现在单独接收统一的 `cutoff_squared`，并由跨 CPU/reference/CUDA 与 exhaustive/cell-list 的确定性回归固定。

修复后的 clean revision `01eeac5683c2871d572338de437716d0689f5e50` 重新执行正式 CUDA 流程：全部 1,536 个 Matbench structures、2,780,158 个 Vesin keys 与 43,842 个 dense keys 精确一致；epoch 12.107 ms、median batch 0.2252 ms，相对修复前分别变化约 +0.7% 与 +1.1%。新的 32,768-atom Nsight profile 为全部 kernels 约 0.1356 ms/call、NVTX range 约 0.3031 ms/call，没有确认的性能回归。

最终独立 reviewer 的本轮结论将在上述修复的 clean delivery revision 审查完成后追加于此，避免把作者自测误写成独立认证。

## 已知限制

One-shot API 每次重建 metadata 并按真实 pair 数精确分配输出；prepared metadata 需要单独设计 ownership 与 invalidation contract。极小非空 periodic cell 的物理 neighbor set 本身可能包含大量 images，因此 metadata/output memory 仍会随真实结果增长。CUDA 对 large unwrapped representatives 使用 whole-call exhaustive fallback，保证统一语义但不保证大体系仍维持 cell-list complexity。CPU 大体系的 Vesin 实现当前更成熟；tonari CPU 的主要优势区间是材料数据中常见的小体系高频调用。Raw Nsight trace 保持 Git ignored，仓库提交可复现脚本、summary 和完整 CSV aggregates。
