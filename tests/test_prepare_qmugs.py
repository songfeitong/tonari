from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Self
from unittest.mock import patch

import numpy as np
import pytest
import torch

from benchmarks.baselines import dense_candidate_count, torch_dense_batch
from benchmarks.qmugs_data import QmugsStructureDataset, select_qmugs
from scripts.prepare_qmugs import (
    ensure_download,
    parse_sdf,
    select_candidates,
    write_deterministic_npz,
)


class MockResponse:
    def __init__(self, contents: bytes, status: int) -> None:
        self.contents = contents
        self.status = status

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        del size
        contents, self.contents = self.contents, b""
        return contents


def test_download_keeps_truncated_response_resumable(tmp_path: Path) -> None:
    path = tmp_path / "source.bin"
    digest = hashlib.sha256(b"abcde").hexdigest()

    with (
        patch(
            "scripts.prepare_qmugs.urllib.request.urlopen",
            return_value=MockResponse(b"abc", 200),
        ),
        pytest.raises(RuntimeError, match="rerun to resume"),
    ):
        ensure_download(path, "https://example.test/source.bin", 5, digest)

    assert not path.exists()
    assert path.with_suffix(".bin.part").read_bytes() == b"abc"

    with patch(
        "scripts.prepare_qmugs.urllib.request.urlopen",
        return_value=MockResponse(b"de", 206),
    ) as urlopen:
        assert (
            ensure_download(path, "https://example.test/source.bin", 5, digest)
            == digest
        )

    assert path.read_bytes() == b"abcde"
    assert not path.with_suffix(".bin.part").exists()
    assert urlopen.call_args.args[0].get_header("Range") == "bytes=3-"


def test_parse_qmugs_v2000_sdf() -> None:
    contents = b"""example
  QMugs

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2500    0.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0
   -0.5000    0.7500    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END
$$$$
"""

    positions, atomic_numbers = parse_sdf(contents)

    np.testing.assert_array_equal(atomic_numbers, [6, 17, 1])
    np.testing.assert_allclose(
        positions,
        [[0.0, 0.0, 0.0], [1.25, 0.0, 0.0], [-0.5, 0.75, 0.0]],
    )


def test_selection_uses_lowest_energy_and_disjoint_workloads(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.csv"
    fieldnames = [
        "chembl_id",
        "conf_id",
        "atoms",
        "heavy_atoms",
        "heteroatoms",
        "rotatable_bonds",
        "rings",
        "GFN2_TOTAL_ENERGY",
    ]
    heavy_atom_counts = (5, 15, 25, 35, 45, 55, 70, 90)
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for bin_index, heavy_atoms in enumerate(heavy_atom_counts):
            for molecule_index in range(2):
                chembl_id = f"CHEMBL_{bin_index}_{molecule_index}"
                for conformer_index, energy in enumerate((-10.0, -11.0)):
                    writer.writerow(
                        {
                            "chembl_id": chembl_id,
                            "conf_id": f"conf_{conformer_index:02d}",
                            "atoms": heavy_atoms + 4,
                            "heavy_atoms": heavy_atoms,
                            "heteroatoms": 2,
                            "rotatable_bonds": 3,
                            "rings": 1,
                            "GFN2_TOTAL_ENERGY": energy,
                        }
                    )

    selected, counts = select_candidates(
        summary, seed=17, population_size=1, balanced_per_bin=1
    )

    assert counts == {"molecules": 16, "conformers": 32}
    assert len(selected) == 9
    assert len({candidate.chembl_id for candidate, _ in selected}) == 9
    assert {
        candidate.heavy_atom_bin
        for candidate, group in selected
        if group == "size_balanced"
    } == set(range(8))
    assert all(candidate.conformer_id == "conf_01" for candidate, _ in selected)


def test_dense_baseline_supports_finite_batches() -> None:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    batch_ptr = torch.tensor([0, 2, 3])
    cell = torch.zeros((2, 3, 3), dtype=torch.float64)
    pbc = torch.zeros((2, 3), dtype=torch.bool)

    pair_indices, cell_shifts = torch_dense_batch(positions, cell, pbc, 1.0, batch_ptr)

    assert dense_candidate_count(batch_ptr, cell, pbc, 1.0) == 5
    assert set(map(tuple, pair_indices.tolist())) == {(0, 1), (1, 0)}
    assert torch.count_nonzero(cell_shifts) == 0


def test_qmugs_cache_and_workload_selection(tmp_path: Path) -> None:
    cache = tmp_path / "sample.npz"
    manifest = tmp_path / "manifest.json"
    structures = tmp_path / "manifest_structures.csv"
    np.savez_compressed(
        cache,
        positions=np.zeros((5, 3), dtype=np.float64),
        batch_ptr=np.asarray([0, 2, 5], dtype=np.int64),
        atomic_numbers=np.asarray([1, 1, 6, 1, 1], dtype=np.int32),
        source_ids=np.asarray(["CHEMBL1/conf_00", "CHEMBL2/conf_01"]),
    )
    manifest.write_text(
        """{
  "sampling": {"heavy_atom_boundaries": [10, 20, 30, 40, 50, 65, 80]},
  "structures_file": "manifest_structures.csv"
}
"""
    )
    structures.write_text(
        "source_id,n_atoms,n_heavy_atoms,workload,heavy_atom_bin\n"
        "CHEMBL1/conf_00,2,1,population,0\n"
        "CHEMBL2/conf_01,3,1,size_balanced,1\n"
    )

    dataset = QmugsStructureDataset(cache, manifest, torch.float32)
    balanced = select_qmugs(dataset, "size_balanced", 1)

    assert len(dataset) == 2
    assert len(balanced) == 1
    assert balanced[0]["source_id"] == "CHEMBL2/conf_01"
    assert balanced[0]["positions"].shape == (3, 3)
    assert not torch.any(balanced[0]["pbc"])


def test_qmugs_cache_archive_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    arrays = {
        "positions": np.arange(12, dtype=np.float64).reshape(4, 3),
        "batch_ptr": np.asarray([0, 4], dtype=np.int64),
    }

    write_deterministic_npz(first, **arrays)
    write_deterministic_npz(second, **arrays)

    assert first.read_bytes() == second.read_bytes()
