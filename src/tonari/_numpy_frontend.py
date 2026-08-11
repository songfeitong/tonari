from __future__ import annotations

from math import isfinite

import numpy as np

from ._extensions import load_numpy_cpu


def _require_numpy_arrays(
    *,
    cell: object,
    pbc: object,
    batch_ptr: object | None,
) -> None:
    for name, array in (("cell", cell), ("pbc", pbc), ("batch_ptr", batch_ptr)):
        if array is not None and not isinstance(array, np.ndarray):
            raise TypeError(
                "positions, cell, pbc, and batch_ptr must all be NumPy arrays; "
                f"{name} has type {type(array).__name__}"
            )


def _normalize_numpy_inputs(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    batch_ptr: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N_total, 3)")
    if batch_ptr is None:
        if cell.ndim != 2 or cell.shape != (3, 3):
            raise ValueError("single-structure cell must have shape (3, 3)")
        if pbc.ndim != 1 or pbc.shape != (3,):
            raise ValueError("single-structure pbc must have shape (3,)")
        batch_ptr = np.array([0, len(positions)], dtype=np.int64)
        cell = cell[None]
        pbc = pbc[None]
    else:
        if cell.ndim != 3:
            raise ValueError("batched cell must have shape (B, 3, 3)")
        if pbc.ndim != 2:
            raise ValueError("batched pbc must have shape (B, 3)")
    return positions, cell, pbc, batch_ptr


def _validate_numpy_inputs(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    cutoff: float,
    batch_ptr: np.ndarray,
) -> None:
    if positions.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("positions must have dtype float32 or float64")
    if batch_ptr.ndim != 1 or batch_ptr.size == 0 or batch_ptr.dtype != np.int64:
        raise ValueError("batch_ptr must be int64 with shape (B + 1,)")
    if cell.shape != (batch_ptr.size - 1, 3, 3):
        raise ValueError("cell must have shape (B, 3, 3)")
    if cell.dtype != positions.dtype:
        raise ValueError("cell and positions must have the same dtype")
    if pbc.shape != (batch_ptr.size - 1, 3) or pbc.dtype != np.bool_:
        raise ValueError("pbc must be bool with shape (B, 3)")
    if not isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")


def _select_numpy_quantities(
    quantities: str,
    positions: np.ndarray,
    cell: np.ndarray,
    batch_ptr: np.ndarray,
    pair_indices: np.ndarray,
    cell_shifts: np.ndarray,
) -> tuple[np.ndarray, ...]:
    values: dict[str, np.ndarray] = {"P": pair_indices, "S": cell_shifts}
    source = pair_indices[:, 0]
    target = pair_indices[:, 1]
    if "i" in quantities:
        values["i"] = np.ascontiguousarray(source)
    if "j" in quantities:
        values["j"] = np.ascontiguousarray(target)
    if "d" in quantities or "D" in quantities:
        pair_batch = np.searchsorted(batch_ptr[1:], source, side="right")
        displacements = (
            positions[target]
            - positions[source]
            + np.einsum(
                "ei,eij->ej",
                cell_shifts.astype(positions.dtype, copy=False),
                cell[pair_batch],
            )
        )
        values["D"] = displacements
        if "d" in quantities:
            values["d"] = np.linalg.norm(displacements, axis=1)
    return tuple(values[quantity] for quantity in quantities)


def neighbor_list_numpy(
    quantities: str,
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    cutoff: float,
    batch_ptr: np.ndarray | None,
    *,
    half_list: bool,
    include_self: bool,
) -> tuple[np.ndarray, ...]:
    _require_numpy_arrays(cell=cell, pbc=pbc, batch_ptr=batch_ptr)
    positions, cell, pbc, batch_ptr = _normalize_numpy_inputs(
        positions, cell, pbc, batch_ptr
    )
    cutoff = float(cutoff)
    _validate_numpy_inputs(positions, cell, pbc, cutoff, batch_ptr)
    try:
        backend = load_numpy_cpu()
    except ImportError as error:
        raise RuntimeError(
            "the NumPy CPU extension is not built; install the project"
        ) from error
    pair_indices, cell_shifts = backend.neighbor_list(
        np.ascontiguousarray(positions),
        np.ascontiguousarray(batch_ptr),
        np.ascontiguousarray(cell),
        np.ascontiguousarray(pbc),
        cutoff,
        half_list,
        include_self,
    )
    return _select_numpy_quantities(
        quantities, positions, cell, batch_ptr, pair_indices, cell_shifts
    )
