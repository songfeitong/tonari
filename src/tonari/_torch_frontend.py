from __future__ import annotations

from math import isfinite

import torch
from torch import Tensor

from ._extensions import load_torch_cpu, load_torch_cuda


def _require_torch_tensors(
    *,
    cell: object,
    pbc: object,
    batch_ptr: object | None,
) -> None:
    for name, tensor in (("cell", cell), ("pbc", pbc), ("batch_ptr", batch_ptr)):
        if tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError(
                "positions, cell, pbc, and batch_ptr must all be PyTorch tensors; "
                f"{name} has type {type(tensor).__name__}"
            )


def normalize_torch_inputs(
    positions: Tensor,
    cell: Tensor,
    pbc: Tensor,
    batch_ptr: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N_total, 3)")
    if batch_ptr is None:
        if cell.ndim != 2 or cell.shape != (3, 3):
            raise ValueError("single-structure cell must have shape (3, 3)")
        if pbc.ndim != 1 or pbc.shape != (3,):
            raise ValueError("single-structure pbc must have shape (3,)")
        batch_ptr = torch.tensor(
            [0, len(positions)], dtype=torch.int64, device=positions.device
        )
        cell = cell.unsqueeze(0)
        pbc = pbc.unsqueeze(0)
    else:
        if cell.ndim != 3:
            raise ValueError("batched cell must have shape (B, 3, 3)")
        if pbc.ndim != 2:
            raise ValueError("batched pbc must have shape (B, 3)")
    return positions, cell, pbc, batch_ptr


def validate_torch_inputs(
    positions: Tensor,
    cell: Tensor,
    pbc: Tensor,
    cutoff: float,
    batch_ptr: Tensor,
) -> None:
    if positions.device.type not in ("cpu", "cuda"):
        raise ValueError("positions must be a CPU or CUDA tensor")
    if positions.dtype not in (torch.float32, torch.float64):
        raise ValueError("positions must have dtype float32 or float64")
    if batch_ptr.ndim != 1 or batch_ptr.numel() == 0 or batch_ptr.dtype != torch.int64:
        raise ValueError("batch_ptr must be int64 with shape (B + 1,)")
    if cell.shape != (batch_ptr.numel() - 1, 3, 3):
        raise ValueError("cell must have shape (B, 3, 3)")
    if cell.dtype != positions.dtype:
        raise ValueError("cell and positions must have the same dtype")
    if pbc.shape != (batch_ptr.numel() - 1, 3) or pbc.dtype != torch.bool:
        raise ValueError("pbc must be bool with shape (B, 3)")
    if any(tensor.device != positions.device for tensor in (cell, pbc, batch_ptr)):
        raise ValueError(
            "positions, cell, pbc, and batch_ptr must be on the same device"
        )
    if not isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")


def _select_torch_quantities(
    quantities: str,
    positions: Tensor,
    cell: Tensor,
    batch_ptr: Tensor,
    pair_indices: Tensor,
    cell_shifts: Tensor,
) -> tuple[Tensor, ...]:
    values = {"P": pair_indices, "S": cell_shifts}
    source = pair_indices[:, 0]
    target = pair_indices[:, 1]
    if "i" in quantities:
        values["i"] = source.contiguous()
    if "j" in quantities:
        values["j"] = target.contiguous()
    if "d" in quantities or "D" in quantities:
        pair_batch = torch.bucketize(source.contiguous(), batch_ptr[1:], right=True)
        displacements = (
            positions[target]
            - positions[source]
            + torch.einsum(
                "ei,eij->ej", cell_shifts.to(positions.dtype), cell[pair_batch]
            )
        )
        values["D"] = displacements
        if "d" in quantities:
            values["d"] = torch.linalg.vector_norm(displacements, dim=1)
    return tuple(values[quantity] for quantity in quantities)


def neighbor_list_torch(
    quantities: str,
    positions: Tensor,
    cell: Tensor,
    pbc: Tensor,
    cutoff: float,
    batch_ptr: Tensor | None,
    *,
    algorithm: str,
    num_threads: int,
    sorted: bool,
    half_list: bool,
    include_self: bool,
) -> tuple[Tensor, ...]:
    _require_torch_tensors(cell=cell, pbc=pbc, batch_ptr=batch_ptr)
    positions, cell, pbc, batch_ptr = normalize_torch_inputs(
        positions, cell, pbc, batch_ptr
    )
    cutoff = float(cutoff)
    validate_torch_inputs(positions, cell, pbc, cutoff, batch_ptr)
    arguments = (
        positions.detach().contiguous(),
        batch_ptr.contiguous(),
        cell.detach().contiguous(),
        pbc.contiguous(),
        cutoff,
        half_list,
        include_self,
        algorithm,
    )
    if positions.is_cuda:
        if num_threads != 1:
            raise ValueError("num_threads only applies to CPU inputs")
        try:
            backend = load_torch_cuda()
        except ImportError as error:
            raise RuntimeError(f"the CUDA provider is unavailable: {error}") from error
        arguments += (sorted,)
    else:
        try:
            backend = load_torch_cpu()
        except ImportError as error:
            raise RuntimeError(
                f"the Torch CPU provider is unavailable: {error}"
            ) from error
        arguments += (num_threads,)
    pair_indices, cell_shifts = backend.neighbor_list(*arguments)
    return _select_torch_quantities(
        quantities, positions, cell, batch_ptr, pair_indices, cell_shifts
    )
