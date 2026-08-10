# 独立终审记录

## 范围

完成实现、测试、真实 benchmark 和初版文档后，安排了一个全新 GPT-5.6 high reasoning subagent 对仓库做只读独立审查。提示没有预先灌输作者的设计判断，要求检查 periodic geometry、CUDA/PyTorch correctness、stream/device/autograd/indexing、Matbench 数据来源与抽样、Vesin 与 Equiformer/FairChem-style baseline 公平性、profiling evidence、过度工程、维护性和 Git hygiene。

## 已确认问题与处理

| 严重性 | 已确认问题 | 处理结果 |
| --- | --- | --- |
| 高 | Public int32 `cell_shifts` 对极端未 wrap representative 会 silent overflow、漏 edge；接近 ±2^63 的 int64 difference 也可再次 overflow | 明确要求每个 representative periodic wrap 和最终 shift 均在 int32 range；dense/cell 两路径在线性 per-atom pass 中验证并直接报错，difference 才使用 int64 temporary |
| 中 | Empty periodic structure 仍按 cell/cutoff 枚举 images，0.02 Å cubic cell 可产生约 109 万 shifts、约 654 MiB RSS | Empty structure 的 image count 直接设为零；production/reference 都增加 tiny-cell empty regression test |
| 中 | NaN/Inf positions 和 inactive NaN cell 可静默返回空图 | Cells 在既有 CPU metadata copy 上验证；positions 在 CUDA per-atom pass 中验证，并复用已有 count synchronization 报错 |
| 中 | 错误验证的初版实现使真实 epoch 回退约 22% | Profile 后改成 O(N) prepare、cumsum sentinel 和 8-byte async memset；独立复测恢复到旧版约 1–3% 波动范围 |
| 低 | 无条件 dense block-grid limit 会拒绝本可由 cell list 处理的 741,456-atom workload | 只在实际进入 exhaustive path 或 sparse-bin fallback 时检查 dense grid limit；metadata regression test 固定该行为 |
| 低 | Standard editable wheel build 因 `setup.py` absolute sources 失败 | Sources 改为相对路径，重新执行 `uv sync --active --frozen --all-groups --inexact --no-build-isolation` 并验证 import/test |

## 独立复测

审查者在最终边界修复版上运行了 26 项 tests、Ruff、极端 int32/int64 representative 复现、NaN/Inf、tiny-cell empty case、200 组 dense 随机差分和 40 组 cell-list 随机差分。它还重新对 Vesin 验证全部 1,536 个 Matbench structures 与 2,780,158 个 edge keys，并对 dense median batch 的 43,842 个 keys 精确比对，均通过。

15:11 最终性能修复版的独立 spot check 为 Matbench epoch 34.599 ms、median batch 0.845 ms、32,768-atom supercell 0.396 ms；审查者确认 sentinel/memset 路径没有发现新 race 或 indexing 问题。随后在固定 revision `a20ee8960c27161a568e3f54a026d0f9a43779de` 上重新执行完整正式流程，结果分别为 36.584 ms、0.918 ms 和 0.414 ms；全部 exact validation 仍通过。两组 timing 的差异用于展示短时系统噪声，仓库只把后者作为最终 machine-readable record。

## 残余风险

One-shot API 仍会在每次调用重建 metadata 并同步精确分配输出；prepared metadata 的 ownership/invalidation contract 尚未设计。极小非空 periodic cell 的真实 graph 本身可能包含大量 images，metadata/output memory 会按物理 edge 数增长。Nsight raw trace 因体积与本地路径保持 Git ignored，仓库保存可复现 profile script、完整命令、machine-readable summary，以及完整 kernel/memory/API/NVTX CSV records；性能证据仍只代表当前硬件与软件版本。

## CPU 后端独立终审

CPU/CUDA 一体化重构、CPU benchmark 与初版中文文档完成后，再启用同一个独立 reviewer 的新审查轮次。它从 public vector formula、严格 cutoff、rank/dual 数值稳定性、resource bounds、CPU/CUDA 一致性、真实 Matbench/Vesin evidence、benchmark provenance 和 Git hygiene 重新检查，而不是沿用第一次 CUDA 终审结论。

### 已确认问题与处理

| 严重性 | 已确认问题 | 处理结果 |
| --- | --- | --- |
| 高 | CPU cell-list 的 target-to-bin corner pruning 在 `nextafter(cutoff, 0)` 附近可能只保留一个方向 | 删除收益只有噪声量级的 corner pruning，固定扫描 27 bins；float32/float64 强制 cell-list 回归固定双向 strict semantics |
| 高 | Shared metadata 对 empty tiny cell 仍准备 images；Gram rank check 又会把合法近共线 rows 误判为退化 | Empty structure 在 rank/repeat 前短路；直接在 active rows 上做 long-double one-sided Jacobi SVD，不形成会平方条件数的 Gram matrix |
| 高 | CPU cell-list 用 wrapped coordinates 做最终 cutoff，对合法大未 wrap representatives 漏 edge | Wrapped distance 仅作带保守误差带的 broad phase，边界壳按 original positions/output shift 重算 public formula，unsafe 时 exhaustive |
| 中 | 合法极小非空 cell 可在 image Cartesian product 枚举前不可控 OOM | 以 checked multiplication 执行明确的 batched `2^24` image-shift resource limit；production/reference 同步 fail-fast |
| 高 | CPU/CUDA cell-list 对 float32 大共同晶胞平移产生漏边和伪边，破坏 unified API | CUDA prepare 发现任一 nonzero wrap 时，通过已有 status/cumsum host read 将 whole call 路由到 canonical exhaustive CUDA public predicate；新增 n=256 cross-device regression |
| 高 | 两个各自仍在 int64 内的巨大 sparse bin counts 在 batched cumsum 相加后溢为负数 | Device bin count 超过 `2^28` safety limit 即饱和为 `limit + 1`，不保留无用精确大数；host稳定进入既有 exhaustive sparse fallback |

### 独立复测证据

Reviewer 对 one-sided Jacobi 做了 12,000 个 rank-1/2/3、scale `1e-10`–`1e10`、condition number 到 `1e16` 的 metadata fuzz，并以 mpmath 80-digit arithmetic 复核 SVD decision 与 dual；未找到反例。CPU conservative broad phase 又经过 1,200 个 inner-shell/strict-boundary directed cases、500 个 float32 individual-wrap cases、400 个 common-wrap Vesin cases和独立 public-formula enumeration，未发现错误。

全部 1,536 个 Matbench structures、2,780,158 个 keys 再次与 Vesin 精确一致；CPU JSON 的 samples、median/min/max、throughput、ratio、manifest/cache/extension SHA、clean revision、CPU affinity、single-thread 和 reused `NeighborList` 都经独立重算。最终代码另通过 53 项 unit tests、290 组开发期 differential、CUDA memcheck、CPU-only skip matrix、Ruff、Markdown Prettier 和 Git checks。

### 新增残余限制

CUDA 对 unwrapped cell-list input 的 whole-call exhaustive fallback 保证语义统一，但在 256/512/1,024/2,048 atoms 的 synthetic scaling 中相对 well-wrapped cell list 分别慢约 1.03×/1.41×/2.88×/8.48×，并受 `< 2^31` exhaustive blocks 限制。任务验收不要求未 wrap 大体系保持 cell-list 复杂度，因此该项按已记录的性能边界保留，而不是用未经充分验证的第三套 cutoff predicate 换取速度。
