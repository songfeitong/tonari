# 独立审查记录

本文汇总项目历次独立审查最终覆盖了什么，以及哪些结论已经成立。逐个反例的发现顺序、临时性能数字和中间提交不属于当前设计说明；需要追溯时可查 Git history。

## 最终结论

当前公开 API、NumPy/Torch CPU、Torch CUDA、exhaustive/cell-list paths、full/half/self policy 和真实数据 benchmark 均完成独立复核，未留下已确认的 correctness blocker。

审查重点始终是核心代码和公共语义，而不是把毫秒级波动、文档措辞或大型数据下载过程当作发布级安全审计。Reviewer 的职责是寻找能够改变 pair identity、破坏 API 一致性、造成资源失控或推翻主要性能结论的问题。

## Correctness 覆盖

独立检查覆盖 finite、partial/full PBC、triclinic cells、rank-deficient inactive rows、empty structures、multiple images、periodic self-images、strict cutoff、未 wrap representatives、mixed batches、int32/resource bounds 和 CUDA current stream。

Full/half/self 四种组合在 NumPy、Torch CPU、Torch CUDA、内部 reference、Vesin 与 ASE 之间统一方向后做 exact key comparison。关键不变量包括：

- `include_self` 只增加每个原子的 zero-shift self。
- Half list 等于 full list 中每个 reverse equivalence class 的 canonical 一侧。
- Half pairs 补 reverse 后恢复 full pairs。
- Exhaustive 与 cell-list crossover 不改变 strict-cutoff identity。
- 用原始 positions 和 output shift 重建的 displacement 满足公共公式。

## 真实数据证据

Matbench workload 的 1,536 个晶体、2,780,158 个 pair keys 在 CPU/CUDA 上与 Vesin exact match；representative CUDA batch 还与独立 dense reference exact match。

QMugs workload 的 8,192 个真实分子、15,144,842 个 pair keys 在 CPU/CUDA 上与 Vesin exact match；九个 representative batches 与 finite dense reference exact match。Dataset selection、cache 与 source archive 另做过确定性和逐结构核对，但这些重型数据审计不属于常规 CI。

## 架构审查

架构重构后，Reviewer 独立确认 NumPy 调用不导入或链接 Torch，NumPy/Torch CPU bindings 调用同一个 framework-neutral core，CUDA scheduling 留在 native provider，public API 不传递 implementation metadata。Native calls 的 GIL、exception translation、current-device/current-stream behavior 和 CPU-only build 也在审查范围内。

## 性能审查

Benchmark runner 固定比较语义、线程数、warmup 和输出口径，并把正式 revision、环境、data/cache/extension hashes 写入 machine-readable records。审查只要求没有明显回归、结果可复现且结论与数据一致，不围绕少量环境噪声反复调参。

当前可靠结论是：CPU 在常见小结构 workload 上有竞争力，大型单体系中 Vesin 更快；CUDA 在真实 batch workflow 中显著受益于 batched native execution。更详细的数字和边界见[benchmark](benchmark.md)。

## 已知但非阻断的限制

One-shot API 每次重建 geometry 与 workspace；CPU batch 内 structures 顺序执行；大规模未 wrap CUDA input 可能回退 exhaustive；极小周期晶胞的真实 image/output 数可能很大；当前没有 prepared cache、Verlet skin、sorting、neighbor cap 或 per-species cutoff。这些是明确的产品边界，不是被掩盖的 correctness defect。
