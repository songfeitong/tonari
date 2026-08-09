from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(slots=True)
class StructureBatch:
    positions: Tensor
    ptr: Tensor
    cells: Tensor
    pbc: Tensor
    atomic_numbers: Tensor
    source_ids: tuple[str, ...]

    def pin_memory(self) -> StructureBatch:
        self.positions = self.positions.pin_memory()
        self.ptr = self.ptr.pin_memory()
        self.cells = self.cells.pin_memory()
        self.pbc = self.pbc.pin_memory()
        self.atomic_numbers = self.atomic_numbers.pin_memory()
        return self

    def to(self, device: torch.device) -> StructureBatch:
        return StructureBatch(
            positions=self.positions.to(device, non_blocking=True),
            ptr=self.ptr.to(device, non_blocking=True),
            cells=self.cells.to(device, non_blocking=True),
            pbc=self.pbc.to(device, non_blocking=True),
            atomic_numbers=self.atomic_numbers.to(device, non_blocking=True),
            source_ids=self.source_ids,
        )


class MatbenchStructureDataset(Dataset[dict[str, object]]):
    def __init__(
        self, cache_path: Path, manifest_path: Path, dtype: torch.dtype
    ) -> None:
        cache = np.load(cache_path)
        manifest = json.loads(manifest_path.read_text())
        entries = manifest["structures"]
        row_indices = cache["row_indices"]
        if len(entries) != len(row_indices) or any(
            entry["row_index"] != int(row_index)
            for entry, row_index in zip(entries, row_indices, strict=True)
        ):
            raise ValueError("Matbench cache and manifest select different source rows")
        self.positions = torch.from_numpy(cache["positions"]).to(dtype)
        self.ptr = torch.from_numpy(cache["ptr"])
        self.cells = torch.from_numpy(cache["cells"]).to(dtype)
        self.pbc = torch.from_numpy(cache["pbc"])
        self.atomic_numbers = torch.from_numpy(cache["atomic_numbers"])
        self.source_ids = tuple(entry["configuration_id"] for entry in entries)

    def __len__(self) -> int:
        return len(self.cells)

    def __getitem__(self, index: int) -> dict[str, object]:
        start = int(self.ptr[index])
        stop = int(self.ptr[index + 1])
        return {
            "positions": self.positions[start:stop],
            "cell": self.cells[index],
            "pbc": self.pbc[index],
            "atomic_numbers": self.atomic_numbers[start:stop],
            "source_id": self.source_ids[index],
        }


def select_scaling_structure(
    dataset: MatbenchStructureDataset,
) -> dict[str, object]:
    candidates = []
    for index in range(len(dataset)):
        structure = dataset[index]
        n_atoms = len(structure["positions"])
        lengths = torch.linalg.vector_norm(structure["cell"], dim=1)
        anisotropy = float(lengths.max() / lengths.min())
        candidates.append((abs(n_atoms - 64), abs(anisotropy - 1.5), index))
    return dataset[min(candidates)[2]]


def collate_structures(structures: list[dict[str, object]]) -> StructureBatch:
    counts = torch.tensor(
        [len(structure["positions"]) for structure in structures], dtype=torch.int64
    )
    ptr = torch.cat((torch.zeros(1, dtype=torch.int64), torch.cumsum(counts, dim=0)))
    return StructureBatch(
        positions=torch.cat([structure["positions"] for structure in structures]),
        ptr=ptr,
        cells=torch.stack([structure["cell"] for structure in structures]),
        pbc=torch.stack([structure["pbc"] for structure in structures]),
        atomic_numbers=torch.cat(
            [structure["atomic_numbers"] for structure in structures]
        ),
        source_ids=tuple(str(structure["source_id"]) for structure in structures),
    )


def repeat_structure(
    structure: dict[str, object], repetitions: tuple[int, int, int]
) -> dict[str, object]:
    positions = structure["positions"]
    cell = structure["cell"]
    translations = torch.cartesian_prod(
        *(torch.arange(repetition, dtype=cell.dtype) for repetition in repetitions)
    )
    if translations.ndim == 1:
        translations = translations[:, None]
    positions = (positions[None, :, :] + (translations @ cell)[:, None, :]).reshape(
        -1, 3
    )
    cell = cell * torch.tensor(repetitions, dtype=cell.dtype)[:, None]
    return {
        "positions": positions,
        "cell": cell,
        "pbc": structure["pbc"],
        "atomic_numbers": structure["atomic_numbers"].repeat(len(translations)),
        "source_id": f"{structure['source_id']}__{repetitions[0]}x{repetitions[1]}x{repetitions[2]}",
    }
