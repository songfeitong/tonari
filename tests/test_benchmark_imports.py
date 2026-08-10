from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_cpu_benchmark_import_does_not_require_cuda_extension() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc
import sys

class BlockCudaExtension(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "tonari._C_cuda":
            raise ModuleNotFoundError("CUDA extension deliberately unavailable")
        return None

sys.meta_path.insert(0, BlockCudaExtension())
import benchmarks.run_qmugs_cpu_benchmark
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(repository_root), str(repository_root / "src"))
    )

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        check=True,
    )
