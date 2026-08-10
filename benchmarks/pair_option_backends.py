from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ase.neighborlist import PrimitiveNeighborList
from torch import Tensor
from vesin import NeighborList as VesinNeighborList

from benchmarks.structure_data import StructureBatch
from tonari import find_neighbors
from tonari._pairs import canonicalize_half_pairs


@dataclass(frozen=True, slots=True)
class PairOptions:
    half_list: bool
    include_self: bool

    @property
    def name(self) -> str:
        list_kind = "half" if self.half_list else "full"
        self_kind = "with_self" if self.include_self else "without_self"
        return f"{list_kind}_{self_kind}"


def _require_single_structure(batch: StructureBatch) -> None:
    if len(batch.source_ids) != 1:
        raise ValueError("this CPU baseline accepts one structure per call")


def _empty_output() -> tuple[Tensor, Tensor]:
    return torch.empty((2, 0), dtype=torch.int64), torch.empty(
        (0, 3), dtype=torch.int32
    )


def _append_zero_shift_self_pairs(
    output: tuple[Tensor, Tensor], n_atoms: int
) -> tuple[Tensor, Tensor]:
    pair_indices, cell_shifts = output
    atoms = torch.arange(n_atoms, dtype=torch.int64)
    return (
        torch.cat((pair_indices, torch.stack((atoms, atoms))), dim=1),
        torch.cat((cell_shifts, torch.zeros((n_atoms, 3), dtype=torch.int32))),
    )


class TonariCpuBackend:
    def __init__(self, options: PairOptions) -> None:
        self.options = options

    def __call__(self, batch: StructureBatch, cutoff: float) -> tuple[Tensor, Tensor]:
        return find_neighbors(
            batch.positions,
            batch.cells,
            batch.pbc,
            cutoff,
            batch.offsets,
            half_list=self.options.half_list,
            include_self=self.options.include_self,
        )


class VesinCpuBackend:
    """Adapt Vesin's pair direction and missing zero-shift self pairs."""

    def __init__(self, cutoff: float, options: PairOptions) -> None:
        self.options = options
        self.neighbor_list = VesinNeighborList(
            cutoff=cutoff,
            full_list=not options.half_list,
            sorted=False,
            n_threads=1,
        )

    def __call__(self, batch: StructureBatch, cutoff: float) -> tuple[Tensor, Tensor]:
        _require_single_structure(batch)
        first, second, cell_shifts = self.neighbor_list.compute(
            batch.positions,
            batch.cells[0],
            batch.pbc[0],
            "ijS",
        )
        output = (
            torch.stack((second.to(torch.int64), first.to(torch.int64))),
            cell_shifts.to(torch.int32),
        )
        if self.options.half_list:
            output = canonicalize_half_pairs(*output)
        if self.options.include_self:
            output = _append_zero_shift_self_pairs(output, len(batch.positions))
        return output


class AseCpuBackend:
    """Rebuild ASE's primitive list while reusing configuration by atom count."""

    def __init__(self, cutoff: float, options: PairOptions) -> None:
        self.cutoff = cutoff
        self.options = options
        self._neighbor_lists: dict[int, PrimitiveNeighborList] = {}

    def _neighbor_list(self, n_atoms: int) -> PrimitiveNeighborList:
        if n_atoms not in self._neighbor_lists:
            self._neighbor_lists[n_atoms] = PrimitiveNeighborList(
                cutoffs=np.full(n_atoms, self.cutoff / 2),
                skin=0.0,
                sorted=False,
                self_interaction=self.options.include_self,
                bothways=not self.options.half_list,
                use_scaled_positions=False,
            )
        return self._neighbor_lists[n_atoms]

    def __call__(self, batch: StructureBatch, cutoff: float) -> tuple[Tensor, Tensor]:
        _require_single_structure(batch)
        if cutoff != self.cutoff:
            raise ValueError("ASE baseline cutoff differs from its configured cutoff")
        n_atoms = len(batch.positions)
        if n_atoms == 0:
            return _empty_output()
        neighbor_list = self._neighbor_list(n_atoms)
        neighbor_list.build(
            batch.pbc[0].numpy(),
            batch.cells[0].numpy(),
            batch.positions.numpy(),
        )
        pair_indices = []
        cell_shifts = []
        for target in range(n_atoms):
            sources, shifts = neighbor_list.get_neighbors(target)
            pair_indices.append(
                torch.stack(
                    (
                        torch.from_numpy(sources).to(torch.int64),
                        torch.full((len(sources),), target, dtype=torch.int64),
                    )
                )
            )
            cell_shifts.append(torch.from_numpy(shifts).to(torch.int32))
        output = torch.cat(pair_indices, dim=1), torch.cat(cell_shifts, dim=0)
        if self.options.half_list:
            output = canonicalize_half_pairs(*output)
        return output
