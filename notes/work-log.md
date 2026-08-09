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
