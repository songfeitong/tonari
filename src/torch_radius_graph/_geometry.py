from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite

import torch
from torch import Tensor

_CUDA_BLOCK_SIZE = 256
_INT32_INDEX_LIMIT = 2**31


@dataclass(frozen=True, slots=True)
class SearchMetadata:
    duals: Tensor
    image_shifts: Tensor
    image_ptr: Tensor
    block_ptr: Tensor
    node_ptr: Tensor
    total_blocks: int
    total_nodes: int
    maximum_atoms: int


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
    if any(tensor.device != positions.device for tensor in (ptr, cells, pbc)):
        raise ValueError("positions, ptr, cells, and pbc must be on the same device")
    if not isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")


def _periodic_geometry(
    cells: Tensor, pbc: Tensor, cutoff: float
) -> tuple[Tensor, list[list[int]]]:
    duals = torch.zeros_like(cells)
    repeats_by_structure = [[0, 0, 0] for _ in range(len(pbc))]
    pattern_groups: dict[tuple[bool, bool, bool], list[int]] = {}
    for batch_index, periodic_axes in enumerate(pbc.tolist()):
        pattern_groups.setdefault(tuple(periodic_axes), []).append(batch_index)

    for pattern, batch_indices in pattern_groups.items():
        active_axes = [axis for axis, periodic in enumerate(pattern) if periodic]
        if not active_axes:
            continue
        active_cells = cells[batch_indices][:, active_axes, :]
        singular_values = torch.linalg.svdvals(active_cells)
        scales = torch.maximum(
            singular_values[:, 0], torch.ones_like(singular_values[:, 0])
        )
        tolerances = (
            torch.finfo(torch.float64).eps * max(active_cells.shape[-2:]) * scales
        )
        if torch.any(singular_values[:, -1] <= tolerances):
            raise ValueError(
                "active periodic cell vectors must be linearly independent"
            )
        gram = active_cells @ active_cells.transpose(1, 2)
        active_duals = active_cells.transpose(1, 2) @ torch.linalg.inv(gram)
        repeat_values = torch.ceil(
            cutoff * torch.linalg.vector_norm(active_duals, dim=1)
        ).to(torch.int64)
        for local_axis, axis in enumerate(active_axes):
            duals[batch_indices, :, axis] = active_duals[:, :, local_axis]
        for batch_index, values in zip(
            batch_indices, repeat_values.tolist(), strict=True
        ):
            for axis, value in zip(active_axes, values, strict=True):
                repeats_by_structure[batch_index][axis] = value

    return duals, repeats_by_structure


def _enumerate_image_shifts(
    repeats_by_structure: list[list[int]], atom_counts: list[int]
) -> tuple[list[tuple[int, int, int]], list[int], list[int]]:
    image_shifts: list[tuple[int, int, int]] = []
    image_ptr = [0]
    image_counts: list[int] = []
    shift_cache: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for repeats, n_atoms in zip(repeats_by_structure, atom_counts, strict=True):
        repeat_key = tuple(repeats)
        if n_atoms == 0:
            structure_shifts = []
        else:
            structure_shifts = shift_cache.get(repeat_key)
            if structure_shifts is None:
                structure_shifts = list(
                    product(
                        range(-repeats[0], repeats[0] + 1),
                        range(-repeats[1], repeats[1] + 1),
                        range(-repeats[2], repeats[2] + 1),
                    )
                )
                shift_cache[repeat_key] = structure_shifts
        image_shifts.extend(structure_shifts)
        image_counts.append(len(structure_shifts))
        image_ptr.append(len(image_shifts))
    return image_shifts, image_ptr, image_counts


def build_search_metadata(
    ptr: Tensor,
    cells: Tensor,
    pbc: Tensor,
    cutoff: float,
    n_atoms_total: int,
) -> SearchMetadata:
    ptr_cpu = ptr.detach().cpu()
    if ptr_cpu[-1].item() != n_atoms_total:
        raise ValueError("ptr must end at n_atoms_total")
    cells_cpu = cells.detach().to(device="cpu", dtype=torch.float64)
    pbc_cpu = pbc.detach().cpu()
    if not torch.all(torch.isfinite(cells_cpu)):
        raise ValueError("cells must contain only finite values")
    if ptr_cpu[0].item() != 0 or ptr_cpu[-1].item() < 0:
        raise ValueError("ptr must start at zero and contain nonnegative atom offsets")
    atom_counts_tensor = ptr_cpu[1:] - ptr_cpu[:-1]
    if torch.any(atom_counts_tensor < 0):
        raise ValueError("ptr must be nondecreasing")
    atom_counts = atom_counts_tensor.tolist()
    if n_atoms_total >= _INT32_INDEX_LIMIT:
        raise ValueError(
            "the current CUDA implementation supports fewer than 2^31 atoms"
        )
    duals, repeats_by_structure = _periodic_geometry(cells_cpu, pbc_cpu, cutoff)
    image_shifts, image_ptr, image_counts = _enumerate_image_shifts(
        repeats_by_structure, atom_counts
    )

    block_ptr = [0]
    node_ptr = [0]
    for n_atoms, n_images in zip(atom_counts, image_counts, strict=True):
        n_tasks = n_atoms * n_atoms * n_images
        block_ptr.append(
            block_ptr[-1] + (n_tasks + _CUDA_BLOCK_SIZE - 1) // _CUDA_BLOCK_SIZE
        )
        node_ptr.append(node_ptr[-1] + int(n_atoms) * n_images)
    return SearchMetadata(
        duals=duals.to(device=cells.device, dtype=cells.dtype),
        image_shifts=torch.tensor(
            image_shifts,
            dtype=torch.int32,
            device=cells.device,
        ).reshape(-1, 3),
        image_ptr=torch.tensor(image_ptr, dtype=torch.int64, device=cells.device),
        block_ptr=torch.tensor(block_ptr, dtype=torch.int64, device=cells.device),
        node_ptr=torch.tensor(node_ptr, dtype=torch.int64, device=cells.device),
        total_blocks=block_ptr[-1],
        total_nodes=node_ptr[-1],
        maximum_atoms=max(atom_counts, default=0),
    )
