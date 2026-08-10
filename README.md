# tonari

`tonari` 是一个面向 NumPy 与 PyTorch、CPU 与 CUDA 的 neighbor-search 实验项目。它用同一个 `find_neighbors` 接口处理有限体系、周期体系、单结构和 batch，并返回 cutoff 内的 atom-image pairs。

项目当前关注三件事：清晰且一致的物理语义、适合真实 PyTorch batch 的 CUDA 执行方式，以及 NumPy/Torch CPU 共享的一份 native search implementation。Production code 不依赖 Vesin 或 ASE；它们只用于外部正确性验证和性能对照。

## 快速开始

```python
import torch

from tonari import find_neighbors

positions = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
cell = torch.eye(3) * 4.0
pbc = torch.tensor([False, False, False])

pair_indices, cell_shifts = find_neighbors(
    positions,
    cell,
    pbc,
    cutoff=1.0,
)

source, target = pair_indices
displacements = (
    positions[source]
    - positions[target]
    + cell_shifts.to(positions.dtype) @ cell
)
distances = torch.linalg.vector_norm(displacements, dim=1)
```

公共入口是：

```python
pair_indices, cell_shifts = find_neighbors(
    positions,
    cells,
    pbc,
    cutoff,
    offsets=None,
    *,
    half_list=False,
    include_self=False,
)
```

单结构输入使用 `positions: (N, 3)`、`cells: (3, 3)` 和 `pbc: (3,)`。Batch 输入把原子坐标拼接为 `positions: (N_total, 3)`，再用 `offsets: (B + 1,)` 标出各结构边界；此时 `cells` 和 `pbc` 分别为 `(B, 3, 3)` 与 `(B, 3)`。

`pair_indices` 的两行依次是 source 和 target，`cell_shifts` 施加在 source image 上。Cell vectors 按行存储，因此 displacement 定义为：

```text
positions[source] - positions[target] + cell_shifts @ cell
```

默认返回排除 zero-shift self 的 full directed pairs。`half_list=True` 只保留每对 reverse pairs 的 canonical 一侧；`include_self=True` 为每个原子加入一个 `(i, i, [0, 0, 0])`，不会删除 periodic self-images。Cutoff 使用严格 `<`，同一 atom pair 的多个周期镜像会分别保留，输出顺序不属于接口契约。

NumPy 输入返回 NumPy arrays，Torch 输入返回同 device 的 Torch tensors。所有 array 参数必须属于同一生态；Torch CPU、Torch CUDA 和 NumPy CPU 使用相同的公共语义。完整的 shape、dtype、device、异常和 autograd 说明见 `find_neighbors.__doc__` 与[设计文档](docs/design.md)。

## 实现概览

CPU 根据候选规模在 exhaustive search 与 cell list 之间选择；CUDA 对整个 heterogeneous batch 统一调度，并分别提供 fused exhaustive 与 batched cell-list 路径。NumPy 和 Torch CPU 通过不同的薄 binding 调用同一个 framework-neutral C++ core，NumPy 路径不经过 Torch runtime。

算法为什么这样组织、各路径适合什么 workload，见[算法总览](docs/algorithm-overview.md)。代码的层次与依赖方向见[架构介绍](docs/architecture.md)。

## 构建与验证

当前开发环境使用 Python 3.12、C++20、PyTorch、pybind11 和 Ninja；CUDA provider 仅在可用的 PyTorch/CUDA 工具链下构建。

```bash
cd /home/ftsong/projects/elfes-workspace/tonari
uv sync --group dev --group data
CUDA_VISIBLE_DEVICES=1 uv run python -m pytest -q
```

真实 benchmark 使用 `matbench_mp_e_form` 晶体和 QMugs 分子。主要结论是：CPU 在常见小结构 workload 上有竞争力，大型单体系中 Vesin 的成熟 cell list 更快；CUDA 的主要优势来自整批结构一次进入 native pipeline。完整方法、数据来源和结果见[benchmark 文档](docs/benchmark.md)。

## 文档

- [架构介绍](docs/architecture.md)：模块职责、依赖方向和发布边界。
- [算法总览](docs/algorithm-overview.md)：搜索策略、CPU/CUDA 路径和适用范围。
- [设计文档](docs/design.md)：公共契约、几何约定和 backend 行为。
- [Benchmark](docs/benchmark.md)：真实数据、对照方法和结果。
- [审查记录](docs/review.md)：独立审查覆盖范围与最终结论。
- [工作记录](notes/work-log.md)：主要设计决策的演进，不作为当前接口规范。

## 当前边界

当前只支持 scalar cutoff，不提供 sorting、neighbor cap、species-dependent cutoff、Verlet skin、prepared workspace、CUDA Graph capture 或 GNN/PyG adapter。Neighbor identity 是离散结果，不参与 autograd；需要梯度时，应使用原始 `positions`、`cells` 和返回的 `cell_shifts` 重建连续的 displacements。
