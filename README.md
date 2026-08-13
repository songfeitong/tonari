# tonari

**tonari** (隣, “neighbor”) is a minimal Python package for fast neighbor-list construction. It supports batched structures and periodic boundary conditions on both CPU and CUDA.

[![CI](https://github.com/songfeitong/tonari/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/songfeitong/tonari/actions/workflows/ci.yml)

## API

The entire public API is one function:

```python
results = neighbor_list(
    quantities,
    positions,
    cell,
    pbc,
    cutoff,
    batch_ptr=None,
    *,
    algorithm="auto",
    cpu_threads=None,
    sorted=False,
    half_list=False,
    include_self=False,
)
```

### Inputs

Array inputs may be either NumPy arrays or PyTorch tensors, but they cannot be mixed in one call. PyTorch tensors must also share a device.

| Argument    | Single structure | Batch          |
| ----------- | ---------------- | -------------- |
| `positions` | `(N, 3)`         | `(N_total, 3)` |
| `cell`      | `(3, 3)`         | `(B, 3, 3)`    |
| `pbc`       | `(3,)`           | `(B, 3)`       |
| `batch_ptr` | `None`           | `(B + 1,)`     |

For a Batch, concatenate all positions and use `batch_ptr` to mark structure boundaries. Here, `N` is the number of atoms in one structure, `B` is the number of structures, and `N_total` is their total number of atoms. `batch_ptr` starts at zero and ends at `N_total`; returned pairs never cross its boundaries.

### Outputs

`quantities` selects the returned arrays. Results are always returned as a tuple in the requested order.

| Quantity | Shape | Meaning |
| --- | --- | --- |
| `i` | `(E,)` | Source atom indices |
| `j` | `(E,)` | Target atom indices |
| `P` | `(E, 2)` | Source and target indices together |
| `S` | `(E, 3)` | Integer cell shifts applied to the target atoms |
| `d` | `(E,)` | Distances |
| `D` | `(E, 3)` | Displacement vectors from source atoms to shifted target atoms |

### Options

- `algorithm` may be `"auto"` (default), `"brute_force"`, or `"cell_list"`. With `"auto"`:
  - CPU uses `"brute_force"` when it would test at most 16,384 atom pairs per structure and `"cell_list"` above that.
  - CUDA uses `"brute_force"` when the largest structure has fewer than 256 atoms and `"cell_list"` otherwise.
  - See [Algorithm selection](docs/algorithm-selection.md) for details and fallback behavior.
- `cpu_threads` is the positive CPU thread count, including the calling thread. `None` uses the conservative default of one thread on CPU and leaves the option unspecified on CUDA; CUDA rejects explicit integers. See [CPU multithreading](docs/cpu-multithreading.md) for workload and oversubscription guidance.
- `sorted=True` groups pairs by source index. Their order within each source is unspecified.
- `half_list=True` returns only one direction for each neighbor pair. The default returns both directions.
- `include_self=True` adds one self pair for each atom. By default, self pairs within the same cell are omitted.

## Install from source

The project builds its native code with CMake and packages it with scikit-build-core. Source installation requires Python 3.11–3.14, PyTorch, and a C++20 compiler. NumPy and Torch CPU support are always built. CUDA support is built by default and requires a CUDA-enabled PyTorch installation and a local CUDA toolkit containing `nvcc`.

```bash
# NumPy, Torch CPU, and Torch CUDA (default)
python -m pip install .

# NumPy and Torch CPU only
BUILD_CUDA=0 python -m pip install .
```

The default build fails instead of silently falling back when CUDA prerequisites are unavailable. Reinstall tonari after changing the PyTorch major/minor version or CUDA runtime. See [Source builds](docs/source-builds.md) for editable installs, build metadata, and CI behavior.

## Examples

### PyTorch Geometric batch

```python
import torch

from tonari import neighbor_list

pairs, edge_vectors = neighbor_list(
    "PD",
    batch.pos,
    batch.cell,
    batch.pbc,
    cutoff=self.cutoff,
    batch_ptr=batch.ptr,
)

edge_index = pairs.T.contiguous()
edge_lengths = torch.linalg.vector_norm(edge_vectors, dim=1)
```

### NumPy with ASE

```python
from ase.neighborlist import primitive_neighbor_list

from tonari import neighbor_list

cutoff = 5.0

source, target, cell_shifts = neighbor_list(
    "ijS",
    positions,
    cell,
    pbc,
    cutoff,
)

# Equivalent ASE search
ase_source, ase_target, ase_cell_shifts = primitive_neighbor_list(
    "ijS",
    pbc,
    cell,
    positions,
    cutoff,
)
```

## License

[MIT License](LICENSE).
