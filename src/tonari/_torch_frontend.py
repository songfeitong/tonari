from __future__ import annotations

from math import isfinite

import torch
from torch import Tensor

from ._extensions import load_torch_cpu, load_torch_cuda


def _require_torch_tensors(
    *,
    cells: object,
    pbc: object,
    batch_ptr: object | None,
) -> None:
    for name, tensor in (("cells", cells), ("pbc", pbc), ("batch_ptr", batch_ptr)):
        if tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError(
                "positions, cells, pbc, and batch_ptr must all be PyTorch tensors; "
                f"{name} has type {type(tensor).__name__}"
            )


def normalize_torch_inputs(
    positions: Tensor,
    cells: Tensor,
    pbc: Tensor,
    batch_ptr: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N_total, 3)")
    if batch_ptr is None:
        if cells.ndim != 2 or cells.shape != (3, 3):
            raise ValueError("single-structure cells must have shape (3, 3)")
        if pbc.ndim != 1 or pbc.shape != (3,):
            raise ValueError("single-structure pbc must have shape (3,)")
        batch_ptr = torch.tensor(
            [0, len(positions)], dtype=torch.int64, device=positions.device
        )
        cells = cells.unsqueeze(0)
        pbc = pbc.unsqueeze(0)
    else:
        if cells.ndim != 3:
            raise ValueError("batched cells must have shape (B, 3, 3)")
        if pbc.ndim != 2:
            raise ValueError("batched pbc must have shape (B, 3)")
    return positions, cells, pbc, batch_ptr


def validate_torch_inputs(
    positions: Tensor,
    cells: Tensor,
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
    if cells.shape != (batch_ptr.numel() - 1, 3, 3):
        raise ValueError("cells must have shape (B, 3, 3)")
    if cells.dtype != positions.dtype:
        raise ValueError("cells and positions must have the same dtype")
    if pbc.shape != (batch_ptr.numel() - 1, 3) or pbc.dtype != torch.bool:
        raise ValueError("pbc must be bool with shape (B, 3)")
    if any(tensor.device != positions.device for tensor in (cells, pbc, batch_ptr)):
        raise ValueError(
            "positions, cells, pbc, and batch_ptr must be on the same device"
        )
    if not isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")


def find_neighbors_torch(
    positions: Tensor,
    cells: Tensor,
    pbc: Tensor,
    cutoff: float,
    batch_ptr: Tensor | None,
    *,
    half_list: bool,
    include_self: bool,
) -> tuple[Tensor, Tensor]:
    _require_torch_tensors(cells=cells, pbc=pbc, batch_ptr=batch_ptr)
    positions, cells, pbc, batch_ptr = normalize_torch_inputs(
        positions, cells, pbc, batch_ptr
    )
    cutoff = float(cutoff)
    validate_torch_inputs(positions, cells, pbc, cutoff, batch_ptr)
    arguments = (
        positions.detach().contiguous(),
        batch_ptr.contiguous(),
        cells.detach().contiguous(),
        pbc.contiguous(),
        cutoff,
        half_list,
        include_self,
    )
    if positions.is_cuda:
        try:
            backend = load_torch_cuda()
        except ImportError as error:
            raise RuntimeError(
                "the CUDA extension is not built; install with a CUDA toolkit"
            ) from error
        return backend.find_neighbors(*arguments)
    try:
        backend = load_torch_cpu()
    except ImportError as error:
        raise RuntimeError("the Torch CPU extension is not built") from error
    return backend.find_neighbors(*arguments)
