from __future__ import annotations

from math import isfinite

import numpy as np

from ._extensions import load_numpy_cpu


def _require_numpy_arrays(
    *,
    cells: object,
    pbc: object,
    batch_ptr: object | None,
) -> None:
    for name, array in (("cells", cells), ("pbc", pbc), ("batch_ptr", batch_ptr)):
        if array is not None and not isinstance(array, np.ndarray):
            raise TypeError(
                "positions, cells, pbc, and batch_ptr must all be NumPy arrays; "
                f"{name} has type {type(array).__name__}"
            )


def _normalize_numpy_inputs(
    positions: np.ndarray,
    cells: np.ndarray,
    pbc: np.ndarray,
    batch_ptr: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N_total, 3)")
    if batch_ptr is None:
        if cells.ndim != 2 or cells.shape != (3, 3):
            raise ValueError("single-structure cells must have shape (3, 3)")
        if pbc.ndim != 1 or pbc.shape != (3,):
            raise ValueError("single-structure pbc must have shape (3,)")
        batch_ptr = np.array([0, len(positions)], dtype=np.int64)
        cells = cells[None]
        pbc = pbc[None]
    else:
        if cells.ndim != 3:
            raise ValueError("batched cells must have shape (B, 3, 3)")
        if pbc.ndim != 2:
            raise ValueError("batched pbc must have shape (B, 3)")
    return positions, cells, pbc, batch_ptr


def _validate_numpy_inputs(
    positions: np.ndarray,
    cells: np.ndarray,
    pbc: np.ndarray,
    cutoff: float,
    batch_ptr: np.ndarray,
) -> None:
    if positions.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("positions must have dtype float32 or float64")
    if batch_ptr.ndim != 1 or batch_ptr.size == 0 or batch_ptr.dtype != np.int64:
        raise ValueError("batch_ptr must be int64 with shape (B + 1,)")
    if cells.shape != (batch_ptr.size - 1, 3, 3):
        raise ValueError("cells must have shape (B, 3, 3)")
    if cells.dtype != positions.dtype:
        raise ValueError("cells and positions must have the same dtype")
    if pbc.shape != (batch_ptr.size - 1, 3) or pbc.dtype != np.bool_:
        raise ValueError("pbc must be bool with shape (B, 3)")
    if not isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")


def find_neighbors_numpy(
    positions: np.ndarray,
    cells: np.ndarray,
    pbc: np.ndarray,
    cutoff: float,
    batch_ptr: np.ndarray | None,
    *,
    half_list: bool,
    include_self: bool,
) -> tuple[np.ndarray, np.ndarray]:
    _require_numpy_arrays(cells=cells, pbc=pbc, batch_ptr=batch_ptr)
    positions, cells, pbc, batch_ptr = _normalize_numpy_inputs(
        positions, cells, pbc, batch_ptr
    )
    cutoff = float(cutoff)
    _validate_numpy_inputs(positions, cells, pbc, cutoff, batch_ptr)
    try:
        backend = load_numpy_cpu()
    except ImportError as error:
        raise RuntimeError(
            "the NumPy CPU extension is not built; install the project"
        ) from error
    return backend.find_neighbors(
        np.ascontiguousarray(positions),
        np.ascontiguousarray(batch_ptr),
        np.ascontiguousarray(cells),
        np.ascontiguousarray(pbc),
        cutoff,
        half_list,
        include_self,
    )
