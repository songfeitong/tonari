# torch-radius-graph

这是一个私有 CUDA 实验仓库，用于为整个 PyTorch batch 构造完整的有向周期 cutoff graph。实现独立于 Vesin：小结构使用融合的 exhaustive 路径，大结构使用 Cartesian cell-list 路径。

第一次阅读建议从[算法总览：为什么这个 Radius Graph Builder 快](docs/algorithm-overview.md)开始；它集中介绍问题、整体架构、核心优化、真实性能、模型集成位置和已知边界。

## API

```python
import torch
from torch_radius_graph import radius_graph_pbc

edge_index, cell_shifts = radius_graph_pbc(
    positions=positions,  # float32/float64 [n_atoms_total, 3]，CUDA
    ptr=ptr,              # int64 [batch_size + 1]，CUDA
    cells=cells,          # 同浮点 dtype [batch_size, 3, 3]，CUDA
    pbc=pbc,              # bool [batch_size, 3]，CUDA
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

需要 Python 3.12、支持 CUDA 的 PyTorch 2.12.1、兼容的 CUDA toolkit、C++20 host compiler 和 Ninja。独立环境可用以下命令建立：

```bash
TORCH_CUDA_ARCH_LIST=12.0 uv sync --frozen --all-groups
CUDA_VISIBLE_DEVICES=0 uv run --frozen python -m pytest -q
```

请按目标 GPU 设置 `TORCH_CUDA_ARCH_LIST`。本机执行时复用了已经配置好的 ELFES Python/PyTorch 环境，没有再下载一份 PyTorch wheel；已验证的命令是：

```bash
VIRTUAL_ENV=/home/ftsong/projects/elfes-workspace/elfes/.venv \
CUDA_VISIBLE_DEVICES=1 TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=8 \
uv sync --active --frozen --all-groups --inexact --no-build-isolation

CUDA_VISIBLE_DEVICES=1 \
/home/ftsong/projects/elfes-workspace/elfes/.venv/bin/python -m pytest -q
```

本机系统 CUDA toolkit 是 13.2，而 PyTorch wheel 使用 CUDA 13.0 构建。PyTorch 会对此给出 minor-version warning；编译、导入、测试、sanitizer 检查和 benchmark 均已成功。Production build 仍应优先使用与 wheel minor version 一致的 toolkit。

## 真实材料 benchmark

主要 workload 是从 `matbench_mp_e_form` 确定性抽取的 1,536 个多样化结构；完整原始数据和派生 tensor cache 均被 Git 忽略。数据准备和 benchmark 复现命令为：

```bash
uv run --group data python scripts/prepare_matbench.py
CUDA_VISIBLE_DEVICES=1 uv run --all-groups python benchmarks/run_benchmark.py \
  --output runs/reproduced-benchmark.json
```

已提交的 manifest 保存每个 source configuration ID、固定的源 revision 与 SHA-256，以及完整抽样方法。正式结果将所有样本与逐结构 Vesin GPU 做了 exact key 对比，并在代表性 batch 上与独立的 Equiformer/FairChem-style dense PyTorch 实现对比。详见[性能方法与结果](docs/benchmark.md)、[当前设计](docs/design.md)和[工作记录](notes/work-log.md)。

## 支持范围与限制

当前实现支持 CUDA、float32/float64、同一 batch 内混合 finite/partial/full PBC、不同 cell、active rows 线性独立的 rank-deficient cell、empty structures、有限的 positions/cells、未 wrap 的 atom representatives，以及当前 PyTorch CUDA stream。对每个 atom 和 active axis，由 representative 计算出的整数 periodic wrap 必须能由 int32 表示；超出时直接报错，不进行截断或 int64 输出。暂不提供 production CPU backend、edge sorting、neighbor cap、per-edge/species cutoff、Verlet skin、CUDA Graph capture、`torch.compile`/export 或 multi-GPU dispatch。

One-shot API 每次调用都会重建 periodic search metadata，并为精确大小的输出 tensors 执行同步分配。对于 `ptr/cells/pbc/cutoff` 不变的重复构图，可复用 metadata object 可能提升性能，但安全的 ownership 与 mutation invalidation contract 尚未确定，因此没有加入隐式 cache。总 atom 数、cell-list node 数、representative wraps 和返回 shifts 均必须小于各自的 `2^31` indexing bound；超出时会直接报错。
