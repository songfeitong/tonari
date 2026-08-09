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
