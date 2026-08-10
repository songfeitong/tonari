from __future__ import annotations

from itertools import product
from math import ceil

import torch
from torch import Tensor

from ._pairs import canonical_half_mask
from ._search import validate_torch_inputs
from .neighbors import _normalize_torch_inputs

_MAXIMUM_IMAGE_SHIFTS = 2**24


@torch.no_grad()
def find_neighbors_reference(
    positions: Tensor,
    cells: Tensor,
    pbc: Tensor,
    cutoff: float,
    offsets: Tensor | None = None,
    *,
    half_list: bool = False,
    include_self: bool = False,
) -> tuple[Tensor, Tensor]:
    """Find neighbors exhaustively for development-time correctness checks."""

    positions, cells, pbc, offsets = _normalize_torch_inputs(
        positions, cells, pbc, offsets
    )
    cutoff = float(cutoff)
    cutoff_squared = cutoff * cutoff
    int32_range = torch.iinfo(torch.int32)
    validate_torch_inputs(positions, cells, pbc, cutoff, offsets)
    offsets_cpu = offsets.detach().cpu()
    if offsets_cpu[0].item() != 0 or offsets_cpu[-1].item() != len(positions):
        raise ValueError("offsets must start at zero and end at N_total")
    if torch.any(offsets_cpu[1:] < offsets_cpu[:-1]):
        raise ValueError("offsets must be nondecreasing")
    if not torch.all(torch.isfinite(positions)):
        raise ValueError("positions must contain only finite values")
    if not torch.all(torch.isfinite(cells)):
        raise ValueError("cells must contain only finite values")

    pair_indices: list[Tensor] = []
    pair_shifts: list[Tensor] = []
    total_image_count = 0
    for batch_index in range(offsets.numel() - 1):
        start = int(offsets_cpu[batch_index])
        stop = int(offsets_cpu[batch_index + 1])
        structure_positions = positions[start:stop]
        if stop == start:
            continue
        active_axes = torch.nonzero(pbc[batch_index], as_tuple=False).flatten()
        atom_wrap = torch.zeros(
            (stop - start, 3), dtype=torch.int64, device=positions.device
        )
        repeats = [0, 0, 0]
        if active_axes.numel() > 0:
            active_cell = cells[batch_index, active_axes]
            active_duals = torch.linalg.pinv(active_cell)
            fractional = structure_positions @ active_duals
            floored_fractional = torch.floor(fractional)
            if torch.any(
                (floored_fractional < int32_range.min)
                | (floored_fractional > int32_range.max)
            ):
                raise ValueError(
                    "atom representatives require periodic wraps outside the int32 range"
                )
            active_wrap = floored_fractional.to(torch.int64)
            atom_wrap[:, active_axes] = active_wrap
            for local_axis, axis in enumerate(active_axes.tolist()):
                reciprocal_norm = float(
                    torch.linalg.vector_norm(active_duals[:, local_axis]).cpu()
                )
                repeats[axis] = ceil(cutoff * reciprocal_norm)

        image_count = 1
        for repeat in repeats:
            image_count *= 2 * repeat + 1
        if total_image_count + image_count > _MAXIMUM_IMAGE_SHIFTS:
            raise ValueError("periodic image count exceeds the 2^24 resource limit")
        total_image_count += image_count

        for shift_values in product(
            range(-repeats[0], repeats[0] + 1),
            range(-repeats[1], repeats[1] + 1),
            range(-repeats[2], repeats[2] + 1),
        ):
            image_shift = torch.tensor(
                shift_values, dtype=torch.int64, device=positions.device
            )
            shifts = (
                image_shift[None, None, :]
                - atom_wrap[:, None, :]
                + atom_wrap[None, :, :]
            )
            displacements = (
                structure_positions[:, None, :] - structure_positions[None, :, :]
            )
            for axis in range(3):
                displacements = (
                    displacements
                    + shifts[..., axis, None].to(positions.dtype)
                    * cells[batch_index, axis]
                )
            distance_squared = (
                displacements[..., 0] * displacements[..., 0]
                + displacements[..., 1] * displacements[..., 1]
                + displacements[..., 2] * displacements[..., 2]
            )
            within_cutoff = distance_squared < cutoff_squared
            if shift_values == (0, 0, 0):
                within_cutoff.fill_diagonal_(include_self)
            source, target = torch.nonzero(within_cutoff, as_tuple=True)
            if source.numel() == 0:
                continue
            selected_shifts = shifts[source, target]
            selected_pairs = torch.stack((source + start, target + start))
            if half_list:
                keep = canonical_half_mask(selected_pairs, selected_shifts)
                selected_pairs = selected_pairs[:, keep]
                selected_shifts = selected_shifts[keep]
                if selected_pairs.shape[1] == 0:
                    continue
            if torch.any(
                (selected_shifts < int32_range.min)
                | (selected_shifts > int32_range.max)
            ):
                raise ValueError(
                    "a cell shift required by the neighbor list exceeds the int32 output range"
                )
            pair_indices.append(selected_pairs)
            pair_shifts.append(selected_shifts.to(torch.int32))

    if not pair_indices:
        return (
            torch.empty((2, 0), dtype=torch.int64, device=positions.device),
            torch.empty((0, 3), dtype=torch.int32, device=positions.device),
        )
    return torch.cat(pair_indices, dim=1), torch.cat(pair_shifts, dim=0)
