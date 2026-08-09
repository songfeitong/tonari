# torch-radius-graph

Private CUDA experiment for constructing a complete directed periodic cutoff graph for an entire PyTorch batch. The implementation is independent of Vesin and uses a hybrid fused exhaustive path for small structures and a Cartesian cell-list path for larger structures.

## API

```python
import torch
from torch_radius_graph import radius_graph_pbc

edge_index, cell_shifts = radius_graph_pbc(
    positions=positions,  # float32/float64 [n_atoms_total, 3], CUDA
    ptr=ptr,              # int64 [batch_size + 1], CUDA
    cells=cells,          # matching float dtype [batch_size, 3, 3], CUDA
    pbc=pbc,              # bool [batch_size, 3], CUDA
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

Cell vectors are rows. `cell_shifts[e]` translates the source image for edge `e`, so the continuous vector is `positions[source] - positions[target] + cell_shifts @ cell`. The graph contains every directed atom-image edge with squared distance strictly less than `cutoff**2`; it excludes only `(i, i, [0, 0, 0])`, retains periodic self-images and multiple images, never crosses batch members, and returns zero shifts on inactive PBC axes. Output order is intentionally unspecified.

`positions` and `cells` may require gradients. Connectivity and shifts are discrete integer outputs and have no backward; recomputing vectors from the original floating tensors as above differentiates the continuous geometry while holding topology fixed.

## Build and test

Python 3.12, a CUDA-enabled PyTorch 2.12.1 installation, a compatible CUDA toolkit, a C++20 host compiler, and Ninja are required. A standalone environment can be created with:

```bash
TORCH_CUDA_ARCH_LIST=12.0 uv sync --frozen --all-groups
CUDA_VISIBLE_DEVICES=0 uv run --frozen python -m pytest -q
```

Set `TORCH_CUDA_ARCH_LIST` for the target GPU. On this workstation the already provisioned ELFES Python/PyTorch environment was reused without downloading another PyTorch wheel; the verified command was:

```bash
VIRTUAL_ENV=/home/ftsong/projects/elfes-workspace/elfes/.venv \
CUDA_VISIBLE_DEVICES=1 TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=8 \
uv sync --active --frozen --all-groups --inexact --no-build-isolation

CUDA_VISIBLE_DEVICES=1 \
/home/ftsong/projects/elfes-workspace/elfes/.venv/bin/python -m pytest -q
```

The system CUDA toolkit here is 13.2 while the PyTorch wheel was built with CUDA 13.0. PyTorch reports this as a minor-version warning; compilation, import, tests, sanitizer checks, and benchmarks all succeeded. A production build should preferably use the wheel-matched toolkit.

## Real-material benchmark

The primary workload is a deterministic 1,536-structure diversity sample from `matbench_mp_e_form`; full raw data and the derived tensor cache are ignored by Git. Prepare it and reproduce the benchmark with:

```bash
uv run --group data python scripts/prepare_matbench.py
CUDA_VISIBLE_DEVICES=1 uv run --all-groups python benchmarks/run_benchmark.py \
  --output runs/reproduced-benchmark.json
```

The committed manifest records every source configuration ID, the pinned source revision and SHA-256, and the complete sampling method. The committed result compares all sampled structures exactly against per-structure Vesin GPU and compares a representative batch against an independent Equiformer/FairChem-style dense PyTorch implementation. See [benchmark methodology and results](docs/benchmark.md), [current design](docs/design.md), and [work log](notes/work-log.md).

## Supported scope and limitations

The current implementation supports CUDA, float32 and float64, mixed finite/partial/full PBC within one batch, different cells, rank-deficient cells whose active rows are independent, empty structures, unwrapped atom representatives, and the current PyTorch CUDA stream. It does not provide a CPU production backend, edge sorting, neighbor caps, per-edge or species cutoffs, a Verlet skin, CUDA Graph capture, `torch.compile`/export support, or multi-GPU dispatch.

The one-shot API rebuilds periodic search metadata and synchronizes to allocate exact-size output tensors on every call. A reusable metadata object could improve repeated graphs with unchanged `ptr/cells/pbc/cutoff`, but defining safe ownership and mutation invalidation is deliberately left open. The current implementation also uses int32 cell-list node and cell-shift storage and rejects searches that exceed its documented indexing bounds.
