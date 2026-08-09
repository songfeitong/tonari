from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch
from torch import Tensor


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
    if not torch.all(torch.isfinite(cells_cpu)):
        raise ValueError("cells must contain only finite values")
    if ptr_cpu[0].item() != 0 or ptr_cpu[-1].item() < 0:
        raise ValueError("ptr must start at zero and contain nonnegative atom offsets")
    atom_counts_tensor = ptr_cpu[1:] - ptr_cpu[:-1]
    if torch.any(atom_counts_tensor < 0):
        raise ValueError("ptr must be nondecreasing")

    duals = torch.zeros_like(cells_cpu)
    repeats_by_structure = [[0, 0, 0] for _ in range(len(pbc_cpu))]
    pattern_groups: dict[tuple[bool, bool, bool], list[int]] = {}
    for batch_index, periodic_axes in enumerate(pbc_cpu.tolist()):
        pattern_groups.setdefault(tuple(periodic_axes), []).append(batch_index)
    for pattern, batch_indices in pattern_groups.items():
        active_axes = [axis for axis, periodic in enumerate(pattern) if periodic]
        if not active_axes:
            continue
        active_cells = cells_cpu[batch_indices][:, active_axes, :]
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
        reciprocal_norms = torch.linalg.vector_norm(active_duals, dim=1)
        repeats = torch.ceil(cutoff * reciprocal_norms).to(torch.int64)
        for local_axis, axis in enumerate(active_axes):
            duals[batch_indices, :, axis] = active_duals[:, :, local_axis]
            for group_index, batch_index in enumerate(batch_indices):
                repeats_by_structure[batch_index][axis] = int(
                    repeats[group_index, local_axis]
                )

    image_shifts: list[tuple[int, int, int]] = []
    image_ptr = [0]
    image_counts: list[int] = []
    shift_cache: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for repeats, n_atoms in zip(
        repeats_by_structure, atom_counts_tensor.tolist(), strict=True
    ):
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

    if ptr_cpu[-1].item() >= 2**31:
        raise ValueError(
            "the current CUDA implementation supports fewer than 2^31 atoms"
        )
    blocks_per_structure = [
        (int(n_atoms) * int(n_atoms) * n_images + 255) // 256
        for n_atoms, n_images in zip(
            atom_counts_tensor.tolist(), image_counts, strict=True
        )
    ]
    block_ptr = [0]
    for n_blocks in blocks_per_structure:
        block_ptr.append(block_ptr[-1] + n_blocks)
    node_ptr = [0]
    for n_atoms, n_images in zip(
        atom_counts_tensor.tolist(), image_counts, strict=True
    ):
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
        maximum_atoms=max(atom_counts_tensor.tolist(), default=0),
    )
