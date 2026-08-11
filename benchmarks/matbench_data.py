from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


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
        self.batch_ptr = torch.from_numpy(cache["batch_ptr"])
        self.cell = torch.from_numpy(cache["cell"]).to(dtype)
        self.pbc = torch.from_numpy(cache["pbc"])
        self.atomic_numbers = torch.from_numpy(cache["atomic_numbers"])
        self.source_ids = tuple(entry["configuration_id"] for entry in entries)

    def __len__(self) -> int:
        return len(self.cell)

    def __getitem__(self, index: int) -> dict[str, object]:
        start = int(self.batch_ptr[index])
        stop = int(self.batch_ptr[index + 1])
        return {
            "positions": self.positions[start:stop],
            "cell": self.cell[index],
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
