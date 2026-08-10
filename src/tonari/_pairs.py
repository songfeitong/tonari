from __future__ import annotations

import torch
from torch import Tensor


def zero_shift_self_mask(pair_indices: Tensor, cell_shifts: Tensor) -> Tensor:
    source, target = pair_indices
    return (source == target) & torch.all(cell_shifts == 0, dim=1)


def canonical_half_mask(pair_indices: Tensor, cell_shifts: Tensor) -> Tensor:
    source, target = pair_indices
    same_atom = source == target
    shift_is_canonical = (cell_shifts[:, 0] < 0) | (
        (cell_shifts[:, 0] == 0)
        & (
            (cell_shifts[:, 1] < 0)
            | ((cell_shifts[:, 1] == 0) & (cell_shifts[:, 2] <= 0))
        )
    )
    return (source < target) | (same_atom & shift_is_canonical)


def canonicalize_half_pairs(
    pair_indices: Tensor, cell_shifts: Tensor
) -> tuple[Tensor, Tensor]:
    """Orient one representative per reverse class by the public key rule."""

    canonical = canonical_half_mask(pair_indices, cell_shifts)
    source, target = pair_indices
    return (
        torch.stack(
            (
                torch.where(canonical, source, target),
                torch.where(canonical, target, source),
            )
        ),
        torch.where(canonical[:, None], cell_shifts, -cell_shifts),
    )
