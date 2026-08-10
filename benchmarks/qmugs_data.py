from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class QmugsStructureDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        cache_path: Path,
        manifest_path: Path,
        dtype: torch.dtype,
    ) -> None:
        cache = np.load(cache_path)
        manifest = json.loads(manifest_path.read_text())
        with (manifest_path.parent / manifest["structures_file"]).open(
            newline=""
        ) as handle:
            entries = list(csv.DictReader(handle))
        for entry in entries:
            for field in ("n_atoms", "n_heavy_atoms", "heavy_atom_bin"):
                entry[field] = int(entry[field])
        self.heavy_atom_boundaries = tuple(
            manifest["sampling"]["heavy_atom_boundaries"]
        )
        offsets = cache["offsets"]
        if len(entries) + 1 != len(offsets):
            raise ValueError("QMugs cache and manifest contain different sample sizes")
        if list(cache["source_ids"]) != [entry["source_id"] for entry in entries]:
            raise ValueError("QMugs cache and manifest select different structures")
        if any(
            entry["n_atoms"] != int(offsets[index + 1] - offsets[index])
            for index, entry in enumerate(entries)
        ):
            raise ValueError("QMugs cache and manifest disagree on atom counts")

        self.positions = torch.from_numpy(cache["positions"]).to(dtype)
        self.offsets = torch.from_numpy(offsets)
        self.atomic_numbers = torch.from_numpy(cache["atomic_numbers"])
        self.entries = tuple(entries)
        self.zero_cell = torch.zeros((3, 3), dtype=dtype)
        self.nonperiodic = torch.zeros(3, dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, object]:
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        entry = self.entries[index]
        return {
            "positions": self.positions[start:stop],
            "cell": self.zero_cell,
            "pbc": self.nonperiodic,
            "atomic_numbers": self.atomic_numbers[start:stop],
            "source_id": entry["source_id"],
        }


def select_qmugs(
    dataset: QmugsStructureDataset,
    workload: str,
    heavy_atom_bin: int | None = None,
) -> Dataset[dict[str, object]]:
    indices = [
        index
        for index, entry in enumerate(dataset.entries)
        if entry["workload"] == workload
        and (heavy_atom_bin is None or entry["heavy_atom_bin"] == heavy_atom_bin)
    ]
    return Subset(dataset, indices)
