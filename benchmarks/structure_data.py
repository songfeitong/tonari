from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class StructureBatch:
    positions: Tensor
    offsets: Tensor
    cells: Tensor
    pbc: Tensor
    atomic_numbers: Tensor
    source_ids: tuple[str, ...]

    def pin_memory(self) -> StructureBatch:
        self.positions = self.positions.pin_memory()
        self.offsets = self.offsets.pin_memory()
        self.cells = self.cells.pin_memory()
        self.pbc = self.pbc.pin_memory()
        self.atomic_numbers = self.atomic_numbers.pin_memory()
        return self

    def to(self, device: torch.device) -> StructureBatch:
        return StructureBatch(
            positions=self.positions.to(device, non_blocking=True),
            offsets=self.offsets.to(device, non_blocking=True),
            cells=self.cells.to(device, non_blocking=True),
            pbc=self.pbc.to(device, non_blocking=True),
            atomic_numbers=self.atomic_numbers.to(device, non_blocking=True),
            source_ids=self.source_ids,
        )


def collate_structures(structures: list[dict[str, object]]) -> StructureBatch:
    counts = torch.tensor(
        [len(structure["positions"]) for structure in structures], dtype=torch.int64
    )
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int64), torch.cumsum(counts, dim=0))
    )
    return StructureBatch(
        positions=torch.cat([structure["positions"] for structure in structures]),
        offsets=offsets,
        cells=torch.stack([structure["cell"] for structure in structures]),
        pbc=torch.stack([structure["pbc"] for structure in structures]),
        atomic_numbers=torch.cat(
            [structure["atomic_numbers"] for structure in structures]
        ),
        source_ids=tuple(str(structure["source_id"]) for structure in structures),
    )
