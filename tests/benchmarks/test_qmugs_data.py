from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from benchmarks.qmugs_data import QmugsStructureDataset, select_qmugs


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
