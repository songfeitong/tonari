from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_isolated(script: str) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository_root)
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        check=True,
    )


def test_numpy_frontend_does_not_import_torch() -> None:
    run_isolated(
        """
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
    )


def test_cpu_benchmarks_do_not_require_cuda_extension() -> None:
    run_isolated(
        """
import importlib.abc
import sys

class BlockCudaExtension(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.endswith("._torch_cuda"):
            raise ModuleNotFoundError("CUDA extension deliberately unavailable")
        return None

sys.meta_path.insert(0, BlockCudaExtension())
import benchmarks.run_cpu_thread_scaling
import benchmarks.run_qmugs_cpu_benchmark
"""
    )
