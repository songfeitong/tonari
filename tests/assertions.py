from __future__ import annotations

import numpy as np
import torch

Array = np.ndarray | torch.Tensor
PairKey = tuple[int, int, int, int, int]


def pair_keys(
    pair_indices: Array | tuple[Array, Array],
    cell_shifts: Array | None = None,
) -> set[PairKey]:
    if cell_shifts is None:
        pair_indices, cell_shifts = pair_indices
    assert type(pair_indices) is type(cell_shifts)
    assert pair_indices.ndim == 2 and pair_indices.shape[1] == 2
    assert cell_shifts.ndim == 2 and cell_shifts.shape == (len(pair_indices), 3)
    if isinstance(pair_indices, torch.Tensor):
        assert pair_indices.dtype == torch.int64
        assert cell_shifts.dtype == torch.int32
        rows = (
            torch.cat((pair_indices, cell_shifts.to(torch.int64)), dim=1).cpu().tolist()
        )
    else:
        assert pair_indices.dtype == np.int64
        assert cell_shifts.dtype == np.int32
        rows = np.concatenate(
            (pair_indices, cell_shifts.astype(np.int64, copy=False)), axis=1
        ).tolist()
    keys = {tuple(row) for row in rows}
    assert len(keys) == len(rows), "neighbor list contains duplicate pair identities"
    return keys


def assert_sorted_by_source(pair_indices: Array) -> None:
    source = pair_indices[:, 0]
    if isinstance(source, np.ndarray):
        assert np.all(source[1:] >= source[:-1])
    else:
        assert torch.all(source[1:] >= source[:-1])
