from __future__ import annotations

from itertools import product
from math import ceil

import torch
from torch import Tensor

from ._geometry import validate_inputs


@torch.no_grad()
def reference_radius_graph_pbc(
    positions: Tensor,
    ptr: Tensor,
    cells: Tensor,
    pbc: Tensor,
    cutoff: float,
) -> tuple[Tensor, Tensor]:
    """Construct a complete directed periodic cutoff graph by exhaustive PyTorch operations.

    This implementation is intentionally independent of the CUDA kernel and intended for correctness checks on small inputs. Returned ``cell_shifts`` satisfy ``positions[source] - positions[target] + cell_shifts @ cell``. The zero-displacement onsite edge is excluded, periodic self-images are retained, and the cutoff comparison is strict.
    """

    cutoff = float(cutoff)
    validate_inputs(positions, ptr, cells, pbc, cutoff, require_cuda=False)
    ptr_cpu = ptr.detach().cpu()
    pbc_cpu = pbc.detach().cpu()
    if ptr_cpu[0].item() != 0 or ptr_cpu[-1].item() != len(positions):
        raise ValueError("ptr must start at zero and end at n_atoms_total")
    if torch.any(ptr_cpu[1:] < ptr_cpu[:-1]):
        raise ValueError("ptr must be nondecreasing")

    edge_indices: list[Tensor] = []
    edge_shifts: list[Tensor] = []
    for batch_index in range(ptr.numel() - 1):
        start = int(ptr_cpu[batch_index])
        stop = int(ptr_cpu[batch_index + 1])
        structure_positions = positions[start:stop]
        active_axes = torch.nonzero(pbc_cpu[batch_index], as_tuple=False).flatten()
        atom_wrap = torch.zeros(
            (stop - start, 3), dtype=torch.int64, device=positions.device
        )
        repeats = [0, 0, 0]
        wrapped_positions = structure_positions
        if active_axes.numel() > 0:
            active_cell = cells[batch_index, active_axes]
            active_duals = torch.linalg.pinv(active_cell)
            fractional = structure_positions @ active_duals
            active_wrap = torch.floor(fractional).to(torch.int64)
            atom_wrap[:, active_axes] = active_wrap
            wrapped_positions = structure_positions - active_wrap.to(
                positions.dtype
            ) @ active_cell
            for local_axis, axis in enumerate(active_axes.tolist()):
                reciprocal_norm = float(
                    torch.linalg.vector_norm(active_duals[:, local_axis]).cpu()
                )
                repeats[axis] = ceil(cutoff * reciprocal_norm)

        for shift_values in product(
            range(-repeats[0], repeats[0] + 1),
            range(-repeats[1], repeats[1] + 1),
            range(-repeats[2], repeats[2] + 1),
        ):
            image_shift = torch.tensor(
                shift_values, dtype=torch.int64, device=positions.device
            )
            translation = image_shift.to(positions.dtype) @ cells[batch_index]
            vectors = (
                wrapped_positions[:, None, :]
                - wrapped_positions[None, :, :]
                + translation
            )
            within_cutoff = torch.sum(vectors * vectors, dim=-1) < cutoff * cutoff
            if shift_values == (0, 0, 0):
                within_cutoff.fill_diagonal_(False)
            source, target = torch.nonzero(within_cutoff, as_tuple=True)
            if source.numel() == 0:
                continue
            shifts = (
                image_shift[None, :]
                - atom_wrap[source]
                + atom_wrap[target]
            )
            edge_indices.append(torch.stack((source + start, target + start)))
            edge_shifts.append(shifts.to(torch.int32))

    if not edge_indices:
        return (
            torch.empty((2, 0), dtype=torch.int64, device=positions.device),
            torch.empty((0, 3), dtype=torch.int32, device=positions.device),
        )
    return torch.cat(edge_indices, dim=1), torch.cat(edge_shifts, dim=0)

