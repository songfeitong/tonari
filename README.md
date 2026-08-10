# tonari

**tonari** is a minimal Python package for fast neighbor-list construction in atomistic modelling. It supports batched inputs with CPU and CUDA backends and is particularly well suited to building radius graphs for atomistic GNNs.

> [!WARNING]
>
> **tonari** is 100% vibe coded. That said, it has a pretty thorough test suite, and its results have been checked against established neighbor-search implementations. Use at your own risk.

## API

The entire public API is one function:

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

## Installation

The project currently builds compiled extensions from source and requires Python 3.12 and a C++20 compiler.

```bash
# NumPy
python -m pip install .

# NumPy and PyTorch
python -m pip install ".[torch]"
```

The CUDA extension is built when both a CUDA-enabled PyTorch installation and a CUDA toolkit are available. CPU use does not require CUDA.

## Examples

### PyTorch Geometric batch

```python
import torch

from tonari import find_neighbors

edge_index, cell_shifts = find_neighbors(
    batch.pos,
    batch.cell,
    batch.pbc,
    cutoff=self.cutoff,
    offsets=batch.ptr,
)

source, target = edge_index
edge_cells = batch.cell[batch.batch[source]]
edge_vectors = (
    batch.pos[source]
    - batch.pos[target]
    + torch.einsum(
        "ni,nij->nj",
        cell_shifts.to(batch.pos.dtype),
        edge_cells,
    )
)
edge_lengths = torch.linalg.vector_norm(edge_vectors, dim=1)
```

### NumPy alongside ASE

```python
import numpy as np
from ase.neighborlist import neighbor_list

from tonari import find_neighbors

cutoff = 5.0

# tonari
pair_indices, cell_shifts = find_neighbors(
    atoms.positions,
    atoms.cell.array,
    atoms.pbc,
    cutoff,
)

# Equivalent ASE search, reordered to the same source-target convention
target, source, ase_cell_shifts = neighbor_list("ijS", atoms, cutoff)
ase_pair_indices = np.stack((source, target))
```

Pairs are included only when their distance is strictly smaller than `cutoff`. Output order is unspecified.

## Batches

For a batch, concatenate all atomic positions and use `offsets` to mark structure boundaries. `cells` and `pbc` then contain one entry per structure.

| Argument  | Shape        |
| --------- | ------------ |
| positions | (N_total, 3) |
| cells     | (B, 3, 3)    |
| pbc       | (B, 3)       |
| offsets   | (B + 1,)     |

`offsets=None` denotes a single structure and is equivalent to boundaries [0, N]. Returned pairs never cross structure boundaries.

## Options

- `half_list=True` returns only one direction for each neighbor pair. The default returns both directions.
- `include_self=True` adds one self pair for each atom. By default, self pairs within the same cell are omitted; periodic copies in neighboring cells are still treated as neighbors.

## License

This project is available under the [MIT License](LICENSE).
