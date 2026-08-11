from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_worktree_is_clean(path: Path) -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return not status.stdout


def canonical_keys(output: tuple[Tensor, Tensor]) -> np.ndarray:
    pair_indices, shifts = output
    keys = torch.cat((pair_indices, shifts.to(torch.int64)), dim=1).cpu().numpy()
    if len(keys) == 0:
        return keys
    order = np.lexsort(tuple(keys[:, column] for column in range(4, -1, -1)))
    return keys[order]


def cpu_frequency_policy(cpu: int | None) -> dict[str, object] | None:
    if cpu is None:
        return None
    root = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
    if not root.is_dir():
        return None

    def read(name: str) -> str | None:
        path = root / name
        return path.read_text().strip() if path.exists() else None

    boost_path = root.parent.parent / "cpufreq" / "boost"
    boost = boost_path.read_text().strip() if boost_path.exists() else None
    return {
        "driver": read("scaling_driver"),
        "governor": read("scaling_governor"),
        "energy_performance_preference": read("energy_performance_preference"),
        "scaling_min_khz": read("scaling_min_freq"),
        "scaling_max_khz": read("scaling_max_freq"),
        "boost": boost,
    }
