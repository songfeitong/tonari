# torch-radius-graph

这是一个私有实验仓库，为 PyTorch CPU 和 CUDA tensors 构造完整的有向周期 cutoff graph。CPU 与 CUDA 共用一个公开 API、几何约定和 periodic metadata；CPU 对每个 structure 在 exhaustive 与 Cartesian cell list 之间切换，CUDA 则保留面向整个 heterogeneous batch 的 fused exhaustive 与 batched cell-list pipeline。Production implementation 不依赖 Vesin，Vesin 只作为开发期外部正确性参考和性能 baseline。

第一次阅读建议从[算法总览：一个接口，两套为硬件而生的搜索路径](docs/algorithm-overview.md)开始；具体实现见[当前设计](docs/design.md)，真实材料测量见[性能方法与结果](docs/benchmark.md)，完整实验过程见[工作记录](notes/work-log.md)。

## API

```python
import torch
from torch_radius_graph import radius_graph_pbc

edge_index, cell_shifts = radius_graph_pbc(
    positions=positions,  # float32/float64 [n_atoms_total, 3]，CPU 或 CUDA
    ptr=ptr,              # int64 [batch_size + 1]，与 positions 同 device
    cells=cells,          # 同浮点 dtype [batch_size, 3, 3]，同 device
    pbc=pbc,              # bool [batch_size, 3]，同 device
    cutoff=5.0,
)

source, target = edge_index
atom_batch = torch.repeat_interleave(
    torch.arange(len(ptr) - 1, device=positions.device), ptr[1:] - ptr[:-1]
)
edge_batch = atom_batch[target]
edge_vectors = positions[source] - positions[target] + torch.einsum(
    "ei,eij->ej", cell_shifts.to(positions.dtype), cells[edge_batch]
)
```

Cell vectors 按行保存。返回的 `edge_index` 为 int64，`cell_shifts` 为 int32；`cell_shifts[e]` 平移 edge `e` 的 source image，因此连续向量为 `positions[source] - positions[target] + cell_shifts @ cell`。Graph 包含平方距离严格小于 `cutoff**2` 的全部有向 atom-image edges；它只排除 `(i, i, [0, 0, 0])`，保留 periodic self-images 和 multiple images，不产生跨 batch member 的 edge，并保证 inactive PBC axes 上的 shift 为零。输出顺序没有接口保证。

`positions` 和 `cells` 可以设置 `requires_grad=True`。Connectivity 和 shifts 是离散整数输出，不提供 backward；按上例从原始浮点 tensors 重算 vectors，便可在 topology 固定时对连续几何正常求导。

## 构建与测试

CPU build 需要 Python 3.12、PyTorch 2.12.1、C++20 compiler 和 Ninja；如果 PyTorch 与本机 CUDA toolkit 都可用，`setup.py` 会同时构建可选 `_C_cuda` extension，否则只构建始终可用的 `_C_cpu`。CPU-only 安装不需要 CUDA toolkit。

本机执行复用了 ELFES 已有的 Python/PyTorch 环境，没有重复下载 wheel：

```bash
cd /home/ftsong/projects/elfes-workspace/torch-radius-graph
/home/ftsong/projects/elfes-workspace/elfes/.venv/bin/python setup.py build_ext --inplace

CUDA_VISIBLE_DEVICES=1 \
/home/ftsong/projects/elfes-workspace/elfes/.venv/bin/python -m pytest -q
```

系统 CUDA toolkit 为 13.2，而 PyTorch wheel 使用 CUDA 13.0 构建，因此 CUDA extension build 会给出 minor-version warning；当前机器上的编译、导入、测试、sanitizer 与 benchmark 均成功。Production toolchain 仍应优先让 toolkit minor version 与 PyTorch wheel 对齐。

## 真实材料 benchmark

主要 workload 是从 `matbench_mp_e_form` 确定性抽取的 1,536 个多样化真实晶体；完整原始数据和派生 tensor cache 均被 Git 忽略。CPU benchmark 使用真实 PyTorch `DataLoader(batch_size=1)` 的确定性顺序，固定到一个物理 core，双方均为单线程，并特意复用 Vesin `NeighborList`，从而不给本实现一次性调用上的不公平优势：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' \
/home/ftsong/projects/elfes-workspace/elfes/.venv/bin/python \
benchmarks/run_cpu_benchmark.py --cpu 31 --repeats 11 \
  --warmup-seconds 2 --require-clean \
  --output runs/reproduced-cpu-benchmark.json
```

AMD Ryzen Threadripper PRO 9975WX 上，完整 1,536-structure epoch 为 143.55 ms，Vesin 为 248.19 ms，本实现快 1.73×；64-atom real structure 为 0.0411 ms，对 Vesin 的 0.0457 ms，快 1.11×。512-atom real-derived supercell 已接近交叉点，本实现慢约 4.6%；到 1,728 atoms 后 Vesin 明显领先，32,768 atoms 时本实现为 24.04 ms、Vesin 为 13.14 ms。换言之，CPU backend 已在真实数据中占多数的常见小体系调用上形成优势，但当前 single-thread cell list 没有超过 Vesin 的大体系成熟度。

CUDA 的既有正式结果保持不变：RTX PRO 6000 Blackwell 上，32-structure DataLoader workload 相对逐 structure Vesin GPU 的最大价值来自 batch-first execution。CPU 与 CUDA 的完整方法、全部 samples、版本和结果分别见[性能文档](docs/benchmark.md)、`benchmarks/results/threadripper-pro-9975wx-cpu.json` 与 `benchmarks/results/rtx-pro-6000-blackwell.json`。

## 支持范围与限制

当前统一 API 支持 CPU/CUDA、float32/float64、同一 batch 内 mixed finite/partial/full PBC、不同 cell、active rows 线性独立的 rank-deficient cell、empty structures、有限 positions/cells、未 wrap atom representatives，以及 CUDA current stream。CPU extension 始终构建，CUDA extension 可选；CPU native search 会释放 Python GIL，但当前每个调用内部是单线程的，batched CPU 输入按 structure 顺序处理。

对每个 atom 和 active axis，由 representative 计算出的整数 periodic wrap 必须能由 int32 表示；返回 cell shift、cell-list node 和总 atom indexing 也受明确的 int32 bounds 约束，超出时直接报错，不静默截断。暂不提供 edge sorting、neighbor cap、per-edge/species cutoff、Verlet skin、prepared metadata cache、CUDA Graph capture、`torch.compile`/export 或跨 device dispatch。

ELFES 当前 two-center 路径以一个 scalar broad cutoff 调 Vesin half-list，再做 species cutoff 过滤并显式加入 onsite。新 CPU backend 已满足其 geometry、uniform cutoff、periodic images 和 strict-boundary 需求，但公开 API 返回 Torch full directed graph 且排除 onsite，因此还需要一个明确的 adapter/canonicalization design；本任务刻意没有修改或接入 ELFES。
