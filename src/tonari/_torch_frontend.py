from __future__ import annotations

from math import isfinite

import torch
from torch import Tensor

from ._extensions import load_torch_cpu, load_torch_cuda


def _require_torch_tensors(
    *,
    cells: object,
    pbc: object,
    offsets: object | None,
) -> None:
    for name, tensor in (("cells", cells), ("pbc", pbc), ("offsets", offsets)):
        if tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError(
                "positions, cells, pbc, and offsets must all be PyTorch tensors; "
                f"{name} has type {type(tensor).__name__}"
            )


def normalize_torch_inputs(
    positions: Tensor,
    cells: Tensor,
    pbc: Tensor,
    offsets: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N_total, 3)")
    if offsets is None:
        if cells.ndim != 2 or cells.shape != (3, 3):
            raise ValueError("single-structure cells must have shape (3, 3)")
        if pbc.ndim != 1 or pbc.shape != (3,):
            raise ValueError("single-structure pbc must have shape (3,)")
        offsets = torch.tensor(
            [0, len(positions)], dtype=torch.int64, device=positions.device
        )
        cells = cells.unsqueeze(0)
        pbc = pbc.unsqueeze(0)
    else:
        if cells.ndim != 3:
            raise ValueError("batched cells must have shape (B, 3, 3)")
        if pbc.ndim != 2:
            raise ValueError("batched pbc must have shape (B, 3)")
    return positions, cells, pbc, offsets


def validate_torch_inputs(
    positions: Tensor,
    cells: Tensor,
    pbc: Tensor,
    cutoff: float,
    offsets: Tensor,
) -> None:
    if positions.device.type not in ("cpu", "cuda"):
        raise ValueError("positions must be a CPU or CUDA tensor")
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


def find_neighbors_torch(
    positions: Tensor,
    cells: Tensor,
    pbc: Tensor,
    cutoff: float,
    offsets: Tensor | None,
    *,
    half_list: bool,
    include_self: bool,
) -> tuple[Tensor, Tensor]:
    _require_torch_tensors(cells=cells, pbc=pbc, offsets=offsets)
    positions, cells, pbc, offsets = normalize_torch_inputs(
        positions, cells, pbc, offsets
    )
    cutoff = float(cutoff)
    validate_torch_inputs(positions, cells, pbc, cutoff, offsets)
    arguments = (
        positions.detach().contiguous(),
        offsets.contiguous(),
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
