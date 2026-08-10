# tonari

**English** | [简体中文](README.zh.md)

Fast, native neighbor search for NumPy and PyTorch on CPU and CUDA.

`tonari` finds atom-image pairs inside a strict distance cutoff. The same `find_neighbors` function handles molecules, periodic materials, single structures, and heterogeneous batches.

## Features

- One API for NumPy and PyTorch.
- Native CPU search and batched CUDA execution.
- Finite, partially periodic, and fully periodic systems.
- Orthogonal and triclinic cells, including rank-deficient inactive cell rows.
- Full directed lists or canonical half lists.
- Optional zero-shift self pairs, without removing periodic self-images.
- Multiple periodic images are preserved; no implicit minimum-image reduction.
- `float32` and `float64` geometry.

## Installation

The project currently builds native extensions from source and requires Python 3.12 and a C++20 compiler.

For NumPy usage:

```bash
python -m pip install .
```

For PyTorch usage:

```bash
python -m pip install ".[torch]"
```

The CUDA extension is built when a CUDA-enabled PyTorch installation and a CUDA toolkit are available. CPU support does not require a CUDA toolkit.

## Quick start

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

`pair_indices` has shape `(2, num_pairs)` and contains `source` and `target` indices. `cell_shifts` has shape `(num_pairs, 3)` and translates the source image. With cell vectors stored as rows, the Cartesian displacement is

```text
positions[source] - positions[target] + cell_shifts @ cell
```

Pairs are included only when their distance is strictly smaller than `cutoff`. Output order is unspecified.

## NumPy

The same function accepts NumPy arrays and returns NumPy arrays:

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

All array arguments in one call must belong to the same ecosystem. NumPy and PyTorch arrays cannot be mixed.

## Batches

For a batch, concatenate atomic positions and use `offsets` to mark structure boundaries:

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

For `B` structures, the batched shapes are:

| Argument    | Shape          |
| ----------- | -------------- |
| `positions` | `(N_total, 3)` |
| `cells`     | `(B, 3, 3)`    |
| `pbc`       | `(B, 3)`       |
| `offsets`   | `(B + 1,)`     |

`offsets=None` denotes a single structure and is equivalent to `[0, len(positions)]`. Pairs never cross structure boundaries.

## Half lists and self pairs

By default, `find_neighbors` returns a full directed list and excludes zero-shift self pairs.

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

`half_list=True` keeps one canonical representative from each pair and reverse-pair class. It does not apply a minimum-image convention. `include_self=True` adds exactly one `(i, i, [0, 0, 0])` pair per atom; periodic self-images `(i, i, S != 0)` remain ordinary cutoff pairs.

## PyTorch and autograd

Neighbor discovery is discrete and is not differentiable. The returned index and shift tensors do not carry gradients. To differentiate continuous geometry while holding the neighbor identity fixed, reconstruct displacements from the original `positions` and `cell` tensors as shown in the quick-start example.

Torch inputs and outputs remain on the same device. A complete batch can therefore be transferred to CUDA once and searched immediately before a model uses the resulting pairs.

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

Use `help(find_neighbors)` for the complete dtype, shape, validation, and geometry contract.

## License

This project is available under the [MIT License](LICENSE).
