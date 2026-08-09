from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import ceil

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class SearchMetadata:
    duals: Tensor
    image_shifts: Tensor
    image_ptr: Tensor
    block_ptr: Tensor
    total_blocks: int


def validate_inputs(
    positions: Tensor,
    ptr: Tensor,
    cells: Tensor,
    pbc: Tensor,
    cutoff: float,
    *,
    require_cuda: bool,
) -> None:
    if require_cuda and not positions.is_cuda:
        raise ValueError("positions must be a CUDA tensor")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms_total, 3)")
    if positions.dtype not in (torch.float32, torch.float64):
        raise ValueError("positions must have dtype float32 or float64")
    if ptr.ndim != 1 or ptr.dtype != torch.int64:
        raise ValueError("ptr must be an int64 tensor with shape (batch_size + 1,)")
    if cells.shape != (ptr.numel() - 1, 3, 3):
        raise ValueError("cells must have shape (batch_size, 3, 3)")
    if cells.dtype != positions.dtype:
        raise ValueError("cells and positions must have the same dtype")
    if pbc.shape != (ptr.numel() - 1, 3) or pbc.dtype != torch.bool:
        raise ValueError("pbc must be a bool tensor with shape (batch_size, 3)")
    tensors = (positions, ptr, cells, pbc)
    if any(tensor.device != positions.device for tensor in tensors):
        raise ValueError("positions, ptr, cells, and pbc must be on the same device")
    if not torch.isfinite(torch.as_tensor(cutoff)) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")


def build_search_metadata(
    ptr: Tensor,
    cells: Tensor,
    pbc: Tensor,
    cutoff: float,
) -> SearchMetadata:
    ptr_cpu = ptr.detach().cpu()
    cells_cpu = cells.detach().to(device="cpu", dtype=torch.float64)
    pbc_cpu = pbc.detach().cpu()
    if ptr_cpu[0].item() != 0 or ptr_cpu[-1].item() < 0:
        raise ValueError("ptr must start at zero and contain nonnegative atom offsets")
    atom_counts_tensor = ptr_cpu[1:] - ptr_cpu[:-1]
    if torch.any(atom_counts_tensor < 0):
        raise ValueError("ptr must be nondecreasing")

    duals = torch.zeros_like(cells_cpu)
    image_shifts: list[tuple[int, int, int]] = []
    image_ptr = [0]
    image_counts: list[int] = []
    for batch_index, periodic_axes in enumerate(pbc_cpu):
        active_axes = torch.nonzero(periodic_axes, as_tuple=False).flatten()
        repeats = [0, 0, 0]
        if active_axes.numel() > 0:
            active_cell = cells_cpu[batch_index, active_axes]
            singular_values = torch.linalg.svdvals(active_cell)
            tolerance = (
                torch.finfo(torch.float64).eps
                * max(active_cell.shape)
                * max(float(singular_values[0]), 1.0)
            )
            if float(singular_values[-1]) <= tolerance:
                raise ValueError("active periodic cell vectors must be linearly independent")
            active_duals = torch.linalg.pinv(active_cell)
            duals[batch_index, :, active_axes] = active_duals
            for local_axis, axis in enumerate(active_axes.tolist()):
                reciprocal_norm = float(torch.linalg.vector_norm(active_duals[:, local_axis]))
                repeats[axis] = ceil(cutoff * reciprocal_norm)

        structure_shifts = list(
            product(
                range(-repeats[0], repeats[0] + 1),
                range(-repeats[1], repeats[1] + 1),
                range(-repeats[2], repeats[2] + 1),
            )
        )
        image_shifts.extend(structure_shifts)
        image_counts.append(len(structure_shifts))
        image_ptr.append(len(image_shifts))

    if ptr_cpu[-1].item() >= 2**31:
        raise ValueError("the current CUDA implementation supports fewer than 2^31 atoms")
    blocks_per_structure = [
        (int(n_atoms) * int(n_atoms) * n_images + 255) // 256
        for n_atoms, n_images in zip(atom_counts_tensor.tolist(), image_counts, strict=True)
    ]
    block_ptr = [0]
    for n_blocks in blocks_per_structure:
        block_ptr.append(block_ptr[-1] + n_blocks)
    if block_ptr[-1] >= 2**31:
        raise ValueError("the dense CUDA path requires fewer than 2^31 thread blocks")
    return SearchMetadata(
        duals=duals.to(device=cells.device, dtype=cells.dtype),
        image_shifts=torch.tensor(
            image_shifts,
            dtype=torch.int32,
            device=cells.device,
        ).reshape(-1, 3),
        image_ptr=torch.tensor(image_ptr, dtype=torch.int64, device=cells.device),
        block_ptr=torch.tensor(block_ptr, dtype=torch.int64, device=cells.device),
        total_blocks=block_ptr[-1],
    )
