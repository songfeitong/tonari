from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import isfinite

import torch
from torch import Tensor

_CUDA_BLOCK_SIZE = 256
_INT32_INDEX_LIMIT = 2**31


@dataclass(frozen=True, slots=True)
class NeighborSearchMetadata:
    duals: Tensor
    image_shifts: Tensor
    image_offsets: Tensor
    atom_counts: tuple[int, ...]
    image_counts: tuple[int, ...]
    maximum_atoms: int


@dataclass(frozen=True, slots=True)
class CudaSearchSchedule:
    block_offsets: Tensor
    node_offsets: Tensor
    total_blocks: int
    total_nodes: int


def validate_torch_inputs(
    positions: Tensor,
    cells: Tensor,
    pbc: Tensor,
    cutoff: float,
    offsets: Tensor,
) -> None:
    if positions.device.type not in ("cpu", "cuda"):
        raise ValueError("positions must be a CPU or CUDA tensor")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N_total, 3)")
    if positions.dtype not in (torch.float32, torch.float64):
        raise ValueError("positions must have dtype float32 or float64")
    if offsets.ndim != 1 or offsets.numel() == 0 or offsets.dtype != torch.int64:
        raise ValueError("offsets must be int64 with shape (B + 1,)")
    if cells.shape != (offsets.numel() - 1, 3, 3):
        raise ValueError("cells must have shape (B, 3, 3)")
    if cells.dtype != positions.dtype:
        raise ValueError("cells and positions must have the same dtype")
    if pbc.shape != (offsets.numel() - 1, 3) or pbc.dtype != torch.bool:
        raise ValueError("pbc must be bool with shape (B, 3)")
    if any(tensor.device != positions.device for tensor in (cells, pbc, offsets)):
        raise ValueError(
            "positions, cells, pbc, and offsets must be on the same device"
        )
    if not isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")


def build_search_metadata(
    offsets: Tensor,
    cells: Tensor,
    pbc: Tensor,
    cutoff: float,
    n_atoms_total: int,
) -> NeighborSearchMetadata:
    offsets_cpu = offsets.detach().cpu()
    if offsets_cpu[-1].item() != n_atoms_total:
        raise ValueError("offsets must end at N_total")
    cells_cpu = cells.detach().to(device="cpu", dtype=torch.float64)
    pbc_cpu = pbc.detach().cpu()
    if offsets_cpu[0].item() != 0 or offsets_cpu[-1].item() < 0:
        raise ValueError(
            "offsets must start at zero and contain nonnegative atom boundaries"
        )
    atom_counts_tensor = offsets_cpu[1:] - offsets_cpu[:-1]
    if torch.any(atom_counts_tensor < 0):
        raise ValueError("offsets must be nondecreasing")
    atom_counts = atom_counts_tensor.tolist()
    if n_atoms_total >= _INT32_INDEX_LIMIT:
        raise ValueError("the current implementation supports fewer than 2^31 atoms")
    try:
        from . import _C_cpu
    except ImportError as error:
        raise RuntimeError(
            "the tonari CPU extension is not built; install the project"
        ) from error
    try:
        duals, image_shifts, image_offsets = _C_cpu.build_periodic_metadata_cpu(
            cells_cpu.contiguous(),
            pbc_cpu.contiguous(),
            atom_counts_tensor.contiguous(),
            cutoff,
        )
    except RuntimeError as error:
        raise ValueError(str(error)) from None
    image_boundaries = image_offsets.tolist()
    image_counts = tuple(stop - start for start, stop in pairwise(image_boundaries))

    return NeighborSearchMetadata(
        duals=duals.to(device=cells.device, dtype=cells.dtype),
        image_shifts=image_shifts.to(device=cells.device),
        image_offsets=image_offsets.to(device=cells.device),
        atom_counts=tuple(atom_counts),
        image_counts=image_counts,
        maximum_atoms=max(atom_counts, default=0),
    )


def build_cuda_schedule(metadata: NeighborSearchMetadata) -> CudaSearchSchedule:
    block_offsets = [0]
    node_offsets = [0]
    for n_atoms, n_images in zip(
        metadata.atom_counts, metadata.image_counts, strict=True
    ):
        n_tasks = n_atoms * n_atoms * n_images
        block_offsets.append(
            block_offsets[-1] + (n_tasks + _CUDA_BLOCK_SIZE - 1) // _CUDA_BLOCK_SIZE
        )
        node_offsets.append(node_offsets[-1] + n_atoms * n_images)
    device = metadata.duals.device
    return CudaSearchSchedule(
        block_offsets=torch.tensor(block_offsets, dtype=torch.int64, device=device),
        node_offsets=torch.tensor(node_offsets, dtype=torch.int64, device=device),
        total_blocks=block_offsets[-1],
        total_nodes=node_offsets[-1],
    )
