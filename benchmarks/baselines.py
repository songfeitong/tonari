from __future__ import annotations

from functools import cache
from itertools import pairwise

import torch
from torch import Tensor
from vesin import NeighborList


@cache
def vesin_gpu_neighbor_list(cutoff: float) -> NeighborList:
    return NeighborList(cutoff=cutoff, full_list=True, sorted=False)


def vesin_gpu_batch(
    positions: Tensor,
    cell: Tensor,
    pbc: Tensor,
    cutoff: float,
    batch_ptr: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run Vesin 0.6.1 on each structure and concatenate the directed neighbor lists."""

    batch_ptr_cpu = batch_ptr.cpu().tolist()
    neighbor_list = vesin_gpu_neighbor_list(cutoff)
    pair_indices = []
    cell_shifts = []
    for batch_index, (start, stop) in enumerate(pairwise(batch_ptr_cpu)):
        first, second, shifts = neighbor_list.compute(
            positions[start:stop], cell[batch_index], pbc[batch_index], "ijS"
        )
        pair_indices.append(
            torch.stack(
                (first.to(torch.int64) + start, second.to(torch.int64) + start),
                dim=1,
            )
        )
        cell_shifts.append(shifts)
    if not pair_indices:
        return (
            torch.empty((0, 2), dtype=torch.int64, device=positions.device),
            torch.empty((0, 3), dtype=torch.int32, device=positions.device),
        )
    return torch.cat(pair_indices, dim=0), torch.cat(cell_shifts, dim=0)


def dense_candidate_count(
    batch_ptr: Tensor, cell: Tensor, pbc: Tensor, cutoff: float
) -> int:
    counts = (batch_ptr[1:] - batch_ptr[:-1]).to(torch.int64)
    if not torch.any(pbc):
        return int(torch.sum(counts * counts).cpu())
    if not torch.all(pbc):
        raise ValueError(
            "the dense baseline requires uniformly finite or full-PBC data"
        )
    reciprocal_norms = torch.linalg.vector_norm(torch.linalg.inv(cell), dim=1)
    maximum_repeats = torch.ceil(cutoff * reciprocal_norms).to(torch.int64).amax(dim=0)
    n_images = int(torch.prod(2 * maximum_repeats + 1).cpu())
    return int(torch.sum(counts * counts).cpu()) * n_images


@torch.no_grad()
def torch_dense_batch(
    positions: Tensor,
    cell: Tensor,
    pbc: Tensor,
    cutoff: float,
    batch_ptr: Tensor,
) -> tuple[Tensor, Tensor]:
    """Materialize all atom pairs and, for full PBC, padded periodic images.

    The comparison and onsite rule match the API under test: strict cutoff, exclusion of only ``(i, i, 0)``, and integer shifts. Benchmark inputs must be uniformly finite or full-PBC.
    """

    device = positions.device
    counts = batch_ptr[1:] - batch_ptr[:-1]
    pair_counts = counts * counts
    pair_batch = torch.repeat_interleave(
        torch.arange(len(counts), device=device), pair_counts
    )
    pair_offsets = torch.cumsum(pair_counts, dim=0) - pair_counts
    local_pair = torch.arange(
        int(pair_counts.sum()), device=device
    ) - torch.repeat_interleave(pair_offsets, pair_counts)
    expanded_counts = torch.repeat_interleave(counts, pair_counts)
    atom_offsets = torch.repeat_interleave(batch_ptr[:-1], pair_counts)
    source = (
        torch.div(local_pair, expanded_counts, rounding_mode="floor") + atom_offsets
    )
    target = local_pair % expanded_counts + atom_offsets

    if not torch.any(pbc):
        displacements = positions[target] - positions[source]
        mask = torch.sum(displacements * displacements, dim=1) < cutoff * cutoff
        mask &= source != target
        source = source[mask]
        target = target[mask]
        return (
            torch.stack((source, target), dim=1),
            torch.zeros((source.shape[0], 3), dtype=torch.int32, device=device),
        )
    if not torch.all(pbc):
        raise ValueError(
            "the dense baseline requires uniformly finite or full-PBC data"
        )

    reciprocal = torch.linalg.inv(cell)
    atom_batch = torch.repeat_interleave(
        torch.arange(len(counts), device=device), counts
    )
    fractional = torch.einsum("ai,aij->aj", positions, reciprocal[atom_batch])
    atom_wraps = torch.floor(fractional).to(torch.int64)
    wrapped_positions = positions - torch.einsum(
        "ai,aij->aj", atom_wraps.to(positions.dtype), cell[atom_batch]
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
    output_shifts = wrapped_shifts - atom_wraps[target] + atom_wraps[source]
    displacements = (
        wrapped_positions[target]
        - wrapped_positions[source]
        + torch.einsum(
            "ei,eij->ej", wrapped_shifts.to(positions.dtype), cell[pair_structure]
        )
    )
    mask = torch.sum(displacements * displacements, dim=1) < cutoff * cutoff
    mask &= ~((source == target) & torch.all(output_shifts == 0, dim=1))
    return torch.stack((source[mask], target[mask]), dim=1), output_shifts[mask].to(
        torch.int32
    )
