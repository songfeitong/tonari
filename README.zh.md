# tonari

[English](README.md) | **简体中文**

面向 NumPy 与 PyTorch、CPU 与 CUDA 的高性能原生 neighbor search。

`tonari` 用严格距离 cutoff 查找 atom-image pairs。同一个 `find_neighbors` 函数可以处理分子、周期材料、单结构和异质 batch。

## 主要功能

- NumPy 与 PyTorch 使用同一套 API。
- 原生 CPU search 与 batched CUDA execution。
- 支持有限、部分周期和完全周期体系。
- 支持正交与 triclinic cells，包括 inactive cell rows 导致的 rank-deficient cell。
- 返回 full directed list 或 canonical half list。
- 可选 zero-shift self pairs，同时保留 periodic self-images。
- 保留 cutoff 内的多个 periodic images，不隐式执行 minimum-image reduction。
- 支持 `float32` 与 `float64` geometry。

## 安装

项目目前从源码构建 native extensions，需要 Python 3.12 和支持 C++20 的编译器。

使用 NumPy：

```bash
python -m pip install .
```

使用 PyTorch：

```bash
python -m pip install ".[torch]"
```

如果环境中存在支持 CUDA 的 PyTorch 和 CUDA toolkit，安装时会同时构建 CUDA extension。CPU 功能不需要 CUDA toolkit。

## 快速开始

```python
import torch

from tonari import find_neighbors

positions = torch.tensor(
    [[0.1, 0.0, 0.0], [2.9, 0.0, 0.0]],
    dtype=torch.float64,
)
cell = torch.eye(3, dtype=torch.float64) * 3.0
pbc = torch.tensor([True, False, False])

pair_indices, cell_shifts = find_neighbors(
    positions,
    cell,
    pbc,
    cutoff=0.3,
)

source, target = pair_indices
displacements = (
    positions[source]
    - positions[target]
    + cell_shifts.to(positions.dtype) @ cell
)
distances = torch.linalg.vector_norm(displacements, dim=1)
```

`pair_indices` 的形状是 `(2, num_pairs)`，两行分别为 `source` 和 `target` indices。`cell_shifts` 的形状是 `(num_pairs, 3)`，表示施加在 source image 上的晶胞平移。Cell vectors 按行保存时，Cartesian displacement 为：

```text
positions[source] - positions[target] + cell_shifts @ cell
```

只有距离严格小于 `cutoff` 的 pairs 会被返回。输出顺序没有接口保证。

## NumPy

同一个函数可以直接接受并返回 NumPy arrays：

```python
import numpy as np

from tonari import find_neighbors

positions = np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
cell = np.eye(3) * 4.0
pbc = np.array([False, False, False])

pair_indices, cell_shifts = find_neighbors(
    positions,
    cell,
    pbc,
    cutoff=1.0,
)
```

同一次调用中的所有 array 参数必须属于同一生态，不能混用 NumPy arrays 与 PyTorch tensors。

## Batch

Batch 输入需要拼接所有原子坐标，并使用 `offsets` 标记各结构的边界：

```python
positions = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [0.8, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [0.0, 1.2, 0.0],
    ]
)
cells = torch.stack((torch.eye(3) * 4.0, torch.eye(3) * 5.0))
pbc = torch.tensor(
    [[False, False, False], [True, True, True]],
)
offsets = torch.tensor([0, 2, 5], dtype=torch.int64)

pair_indices, cell_shifts = find_neighbors(
    positions,
    cells,
    pbc,
    cutoff=1.5,
    offsets=offsets,
)
```

包含 `B` 个结构时，各参数形状为：

| 参数        | 形状           |
| ----------- | -------------- |
| `positions` | `(N_total, 3)` |
| `cells`     | `(B, 3, 3)`    |
| `pbc`       | `(B, 3)`       |
| `offsets`   | `(B + 1,)`     |

`offsets=None` 表示输入为单结构，等价于 `[0, len(positions)]`。返回的 pairs 不会跨结构边界。

## Half list 与 self pair

默认情况下，`find_neighbors` 返回 full directed list，并排除 zero-shift self pairs。

```python
pair_indices, cell_shifts = find_neighbors(
    positions,
    cells,
    pbc,
    cutoff=1.5,
    offsets=offsets,
    half_list=True,
    include_self=True,
)
```

`half_list=True` 对每组 pair/reverse-pair 只保留一个 canonical representative，不执行 minimum-image convention。`include_self=True` 为每个原子加入且只加入一个 `(i, i, [0, 0, 0])`；periodic self-images `(i, i, S != 0)` 仍作为普通 cutoff pairs 保留。

## PyTorch 与 autograd

Neighbor discovery 是离散操作，本身不可微；返回的 index 和 shift tensors 不携带梯度。如果需要在 neighbor identity 固定时对连续几何求导，请像快速开始示例一样，用原始 `positions`、`cell` 和返回的整数 tensors 重建 displacements。

Torch 输入与输出始终位于同一 device。因此可以先把完整 batch 一次传到 CUDA，再在模型使用 pairs 之前执行 neighbor search。

## API

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

使用 `help(find_neighbors)` 查看完整的 dtype、shape、validation 与 geometry contract。

## 许可证

本项目使用 [MIT License](LICENSE)。
