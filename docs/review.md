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

CPU 也在 reviewer 停止同核性能调用后，以 clean revision `052f207403d3c5c058dd844ad546115b99790500` 独占 CPU 31 重跑正式流程。全部 1,536 个 structures、2,780,158 个 keys 精确一致；tonari epoch 143.999 ms、Vesin 248.459 ms，相对旧正式结果均只变化约 0.1%。

## 最终结论

独立 reviewer 在 clean HEAD `715fcf2ffdb1ef00489d241212a6bb00c5b38184` 给出 PASS，无 confirmed blocker。它独立重跑 74 项 GPU-visible tests、50 passed/24 skipped 的 CPU-only matrix和 10 个 docstring examples，并额外完成 100 组 NumPy/Torch/CPU/CUDA/reference mixed-batch differential、30 组 `N≥256` unwrapped differential、200 组任意 float32 cutoff crossover；所有结果一致。

Reviewer 再次验证 CPU/CUDA 对全部 1,536 个 Matbench structures、2,780,158 个 Vesin keys 的 exact match，逐项重算 CPU/CUDA JSON 的 samples、medians、throughput、revision、clean flag 与 data/extension SHA，并确认 Nsight CSV 与 summary 的 0.135588 ms kernels、0.006111 ms memory operations一致。Public surface、docstring、旧 alias/package清理、英文源码、中文 Markdown、Ruff、Prettier、lock、Git status/diff/fsck 和磁盘级 legacy artifact 扫描均通过。

## QMugs finite-molecule 增量终审

QMugs benchmark 加入后，同一独立 reviewer 对官方来源、抽样、cache、CPU/CUDA/Vesin correctness、性能口径、文档与 provenance 做了新的只读终审。Reviewer 不使用项目 selection helper，流式重算完整 2.03 GB `summary.csv`：确认 665,911 个分子、1,992,984 个 conformers、每个 ChEMBL ID 的最低能量 conformer、4,096 个 population samples、互不重叠的八档各 512 个 size-balanced samples，以及 manifest/CSV/cache offsets 全部一致。它随后流式扫描官方 7.18 GB `structures.tar.gz`，独立解析 8,192 个 V2000 atom blocks，逐原子确认 float64 positions 与 int32 atomic numbers 和派生 cache 完全相等且没有 missing structure。

本轮审查推动闭合了四类工程问题：提前 EOF 时残缺下载曾在 size/SHA 验证前替换正式文件，现在 `.part` 只有通过双重校验后才原子替换并有 Range-resume 回归；数据出处补齐 QMugs 四位作者、ETH DOI、CC BY-SA 3.0 URL 与 ChEMBL 27 attribution；纯 benchmark helpers 移入不依赖 backend extension 的 `benchmarks/common.py`，CPU-only build 不再因可选 `_C_cuda` 缺失而无法导入；正式 CPU/CUDA records 重新绑定实际 Python、clean revision 与当前 extension SHA。CPU runner 还记录 governor、EPP、boost 与 frequency bounds，撤回无法在未记录 power policy 下复现的旧 1.16× headline。

最终 correctness 证据为全部 8,192 个 QMugs structures、15,144,842 个 CPU/CUDA/Vesin keys exact，以及九个 CUDA representative batches、1,322,646 个 finite dense keys exact。CPU 正式结果在明确的 `performance/performance` policy 下给出 population 169.700 ms 对 Vesin 169.554 ms；reviewer 独立复跑得到 166.717 对 169.943 ms，均支持“基本打平”而不是旧 headline。CUDA 正式 `batch_size=64` population epoch 为 6.396 ms 对逐结构 Vesin 905.611 ms，当前 source-tree binary SHA、正式 JSON 与 81 项 GPU-visible tests 使用同一 extension。

独立 reviewer 最终在 clean HEAD `50623cdab8229b4f9bb1c4ba8e5bd91e79b1bdb5` 给出 PASS，无 confirmed blocker。它确认 81 项 GPU-visible tests、57 passed/24 skipped 的 CPU-only matrix、Ruff、权威 `--prose-wrap never` Prettier、offline lock、Git status/diff/fsck、正式 JSON 的 revision/clean flag/hash/statistics 与中文文档全部通过。

## Half list与zero-shift self增量终审

本轮review按用户要求聚焦核心代码，不重新审计大型数据下载，也不把毫秒级benchmark波动或文档措辞当作blocker。审查范围是公开half/self语义、zero-shift self与periodic self-image区分、CPU/CUDA exhaustive与cell-list count/write、unwrapped fallback、NumPy/Torch/reference一致性，以及Vesin/ASE adapter的方向和self处理。

首轮核心证据包括115项GPU-visible tests、78 passed/37 skipped的CPU-only matrix、86组四mode CPU/CUDA/reference定向差分，以及192组tonari/Vesin/ASE exact differential。Reviewer未发现native count/write、默认hot path、external adapter或整体架构问题，只确认一个reference blocker：极小正cutoff的平方在geometry dtype下溢为零时，production按契约加入zero-shift self，但reference原先依赖distance comparison而漏掉diagonal。

修复把zero-image mask的diagonal按`include_self`显式置真或置假，不影响非零shift的periodic self-images；回归覆盖float32 `1e-30`、float64 `1e-200`与full/half。作者侧完整suite增至119项并全部通过；reviewer又独立复跑四种underflow组合，确认CPU/reference/CUDA都精确返回唯一`(0, 0, [0, 0, 0])`，相关pair-option tests为31 passed。最终在clean HEAD `8e1cccc56e4f2dbd4231ef49f175470c39d9c5e4`给出PASS，无新的core correctness、性能或架构blocker。

## Framework-neutral架构增量终审

本轮独立5.6-sol high reviewer聚焦架构与核心实现，未围绕毫秒级benchmark波动或历史provenance扩大审查。它确认NumPy和Torch CPU bindings调用同一个只依赖C++标准库的core；NumPy路径既不import Torch，也不链接Torch、ATen或c10；CUDA metadata、schedule和exhaustive/cell-list选择全部归native provider；旧Python planning模块、旧bindings与旧implementation symbols已经删除，暂定项目名没有泄漏到native namespace、注释或错误信息。

Reviewer在clean HEAD `b0d57b26e8bc3bc3895d78d983cfb9517c235b51`重跑119项完整tests，并额外执行80组包含empty、partial PBC、float32/float64和四种pair mode的NumPy CPU、Torch CPU、Torch CUDA batched differential，全部exact。它还在双GPU环境中让current device不同于输入device，并分别覆盖non-default stream下的exhaustive与cell-list路径；CPU/CUDA结果和device/stream行为均正确。NumPy input lifetime、GIL释放、Torch tensor ownership与native exception translation未发现缺陷，最终结论为PASS、无blocker。

唯一非阻断观察是CUDA pybind入口当前持有Python GIL，因此native流程中的两次必要host synchronization会阻塞同一进程的其他Python threads。这不影响correctness、device/current-stream语义或常见每进程单训练线程工作流，留作确有并发需求时再优化。

## 已知限制

One-shot API 每次重建 metadata 并按真实 pair 数精确分配输出；prepared metadata 需要单独设计 ownership 与 invalidation contract。极小非空 periodic cell 的物理 neighbor set 本身可能包含大量 images，因此 metadata/output memory 仍会随真实结果增长。CUDA 对 large unwrapped representatives 使用 whole-call exhaustive fallback，保证统一语义但不保证大体系仍维持 cell-list complexity。CPU 大体系的 Vesin 实现当前更成熟；QMugs 自然分布上两者基本打平，`tonari` 的 CPU 优势集中在更小分子。CPU timing 对 power policy 敏感，正式 JSON 已记录；CUDA JSON 保存 minimum/median/maximum 而不保存全部原始 samples。超过 9 GB 的 QMugs raw/cache 全量审计不适合常规 CI，因此依赖固定 URL、size、SHA 与本次正式审计。Benchmark 排除 data loading/H2D，Vesin GPU 受其逐 structure API 约束；本轮遵循真实工作流，没有人为构造超大单分子 scaling。Raw Nsight trace 保持 Git ignored，仓库提交可复现脚本、summary 和完整 CSV aggregates。
