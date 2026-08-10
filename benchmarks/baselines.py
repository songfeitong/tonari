from __future__ import annotations

from itertools import pairwise

import torch
from torch import Tensor
from vesin import NeighborList


def vesin_gpu_batch(
    positions: Tensor,
    cells: Tensor,
    pbc: Tensor,
    cutoff: float,
    offsets: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run Vesin 0.6.1 on each structure and concatenate the directed neighbor lists."""

    offsets_cpu = offsets.cpu().tolist()
    neighbor_list = NeighborList(cutoff=cutoff, full_list=True, sorted=False)
    pair_indices = []
    cell_shifts = []
    for batch_index, (start, stop) in enumerate(pairwise(offsets_cpu)):
        first, second, shifts = neighbor_list.compute(
            positions[start:stop], cells[batch_index], pbc[batch_index], "ijS"
        )
        pair_indices.append(
            torch.stack((second.to(torch.int64) + start, first.to(torch.int64) + start))
        )
        cell_shifts.append(shifts)
    if not pair_indices:
        return (
            torch.empty((2, 0), dtype=torch.int64, device=positions.device),
            torch.empty((0, 3), dtype=torch.int32, device=positions.device),
        )
    return torch.cat(pair_indices, dim=1), torch.cat(cell_shifts, dim=0)


def dense_candidate_count(offsets: Tensor, cells: Tensor, cutoff: float) -> int:
    counts = (offsets[1:] - offsets[:-1]).to(torch.int64)
    reciprocal_norms = torch.linalg.vector_norm(torch.linalg.inv(cells), dim=1)
    maximum_repeats = torch.ceil(cutoff * reciprocal_norms).to(torch.int64).amax(dim=0)
    n_images = int(torch.prod(2 * maximum_repeats + 1).cpu())
    return int(torch.sum(counts * counts).cpu()) * n_images


@torch.no_grad()
def torch_dense_batch(
    positions: Tensor,
    cells: Tensor,
    pbc: Tensor,
    cutoff: float,
    offsets: Tensor,
) -> tuple[Tensor, Tensor]:
    """Materialize all atom pairs and padded periodic images in the Equiformer/FairChem style.

    The comparison and onsite rule are adjusted to the exact API under test: strict cutoff, exclusion of only ``(i, i, 0)``, and integer shifts. This baseline requires homogeneous full PBC, matching the upstream batch limitation.
    """

    if not torch.all(pbc):
        raise ValueError("the dense batch baseline only supports homogeneous full PBC")
    device = positions.device
    counts = offsets[1:] - offsets[:-1]
    pair_counts = counts * counts
    pair_batch = torch.repeat_interleave(
        torch.arange(len(counts), device=device), pair_counts
    )
    pair_offsets = torch.cumsum(pair_counts, dim=0) - pair_counts
    local_pair = torch.arange(
        int(pair_counts.sum()), device=device
    ) - torch.repeat_interleave(pair_offsets, pair_counts)
    expanded_counts = torch.repeat_interleave(counts, pair_counts)
    atom_offsets = torch.repeat_interleave(offsets[:-1], pair_counts)
    target = (
        torch.div(local_pair, expanded_counts, rounding_mode="floor") + atom_offsets
    )
    source = local_pair % expanded_counts + atom_offsets

    reciprocal = torch.linalg.inv(cells)
    atom_batch = torch.repeat_interleave(
        torch.arange(len(counts), device=device), counts
    )
    fractional = torch.einsum("ai,aij->aj", positions, reciprocal[atom_batch])
    atom_wraps = torch.floor(fractional).to(torch.int64)
    wrapped_positions = positions - torch.einsum(
        "ai,aij->aj", atom_wraps.to(positions.dtype), cells[atom_batch]
    )

    reciprocal_norms = torch.linalg.vector_norm(reciprocal, dim=1)
    maximum_repeats = torch.ceil(cutoff * reciprocal_norms).to(torch.int64).amax(dim=0)
    image_shifts = torch.cartesian_prod(
        *(
            torch.arange(-int(repeat), int(repeat) + 1, device=device)
            for repeat in maximum_repeats.cpu()
        )
    ).to(torch.int64)
    if image_shifts.ndim == 1:
        image_shifts = image_shifts[:, None]
    n_images = len(image_shifts)
    source = source.repeat_interleave(n_images)
    target = target.repeat_interleave(n_images)
    pair_structure = pair_batch.repeat_interleave(n_images)
    wrapped_shifts = image_shifts.repeat(len(pair_batch), 1)
    output_shifts = wrapped_shifts - atom_wraps[source] + atom_wraps[target]
    displacements = (
        wrapped_positions[source]
        - wrapped_positions[target]
        + torch.einsum(
            "ei,eij->ej", wrapped_shifts.to(positions.dtype), cells[pair_structure]
        )
    )
    mask = torch.sum(displacements * displacements, dim=1) < cutoff * cutoff
    mask &= ~((source == target) & torch.all(output_shifts == 0, dim=1))
    return torch.stack((source[mask], target[mask])), output_shifts[mask].to(
        torch.int32
    )
