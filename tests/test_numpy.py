from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.assertions import pair_keys
from tonari import neighbor_list


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_single_structure_matches_torch(dtype: type[np.floating]) -> None:
    positions = np.array(
        [[0.1, 0.2, 0.3], [0.9, 0.4, 0.8], [1.6, 1.0, 0.5]], dtype=dtype
    )
    cell = np.array([[1.7, 0.0, 0.0], [0.45, 1.6, 0.0], [0.0, 0.0, 0.0]], dtype=dtype)
    pbc = np.array([True, True, False])
    actual = neighbor_list("PS", positions, cell, pbc, 2.4)
    expected = neighbor_list(
        "PS",
        torch.from_numpy(positions),
        torch.from_numpy(cell),
        torch.from_numpy(pbc),
        2.4,
    )
    pair_indices, cell_shifts = actual
    assert pair_indices.dtype == np.int64
    assert cell_shifts.dtype == np.int32
    assert pair_keys(actual) == pair_keys(expected)


def test_numpy_batch_matches_torch() -> None:
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    cell = np.stack((np.zeros((3, 3)), np.diag([0.5, 8.0, 8.0])))
    pbc = np.array([[False, False, False], [True, False, False]])
    batch_ptr = np.array([0, 2, 3], dtype=np.int64)
    actual = neighbor_list("PS", positions, cell, pbc, 0.6, batch_ptr)
    expected = neighbor_list(
        "PS",
        torch.from_numpy(positions),
        torch.from_numpy(cell),
        torch.from_numpy(pbc),
        0.6,
        torch.from_numpy(batch_ptr),
    )
    assert pair_keys(actual) == pair_keys(expected)


def test_numpy_adapter_accepts_readonly_and_noncontiguous_arrays() -> None:
    positions = np.arange(18, dtype=np.float64).reshape(3, 6)[:, ::2]
    positions.setflags(write=False)
    cell = np.eye(3, dtype=np.float64)
    pbc = np.zeros(3, dtype=np.bool_)
    pair_indices, cell_shifts = neighbor_list("PS", positions, cell, pbc, 10.0)
    assert pair_indices.shape[1] == 2
    assert cell_shifts.shape == (pair_indices.shape[0], 3)


def test_numpy_frontend_does_not_import_torch() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc
import sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("Torch import deliberately blocked")
        return None

sys.meta_path.insert(0, BlockTorch())
import numpy as np
from tonari import neighbor_list

positions = np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
cell = np.eye(3) * 4.0
pbc = np.zeros(3, dtype=np.bool_)
pair_indices, cell_shifts = neighbor_list("PS", positions, cell, pbc, 1.0)
assert pair_indices.shape == (2, 2)
assert cell_shifts.shape == (2, 3)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository_root / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        check=True,
    )
