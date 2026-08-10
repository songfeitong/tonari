from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from vesin import NeighborList

from benchmarks.matbench_data import (
    MatbenchStructureDataset,
    StructureBatch,
    collate_structures,
    repeat_structure,
    select_scaling_structure,
)
from benchmarks.run_benchmark import canonical_keys, file_sha256, git_revision
from torch_radius_graph import _C_cpu, radius_graph_pbc

Backend = Callable[[StructureBatch, float], tuple[Tensor, Tensor]]


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    return platform.processor()


def git_worktree_is_clean(path: Path) -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return not status.stdout


class VesinCpuBackend:
    def __init__(self, cutoff: float) -> None:
        self.neighbor_list = NeighborList(
            cutoff=cutoff,
            full_list=True,
            sorted=False,
            n_threads=1,
        )

    def __call__(self, batch: StructureBatch, cutoff: float) -> tuple[Tensor, Tensor]:
        if len(batch.source_ids) != 1:
            raise ValueError("the CPU Vesin baseline accepts one structure per call")
        first, second, shifts = self.neighbor_list.compute(
            batch.positions,
            batch.cells[0],
            batch.pbc[0],
            "ijS",
        )
        return torch.stack((second.to(torch.int64), first.to(torch.int64))), shifts


def torch_radius_graph(batch: StructureBatch, cutoff: float) -> tuple[Tensor, Tensor]:
    return radius_graph_pbc(
        batch.positions,
        batch.ptr,
        batch.cells,
        batch.pbc,
        cutoff,
    )


def load_single_structure_batches(
    dataset: MatbenchStructureDataset, seed: int
) -> list[StructureBatch]:
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        collate_fn=collate_structures,
    )
    return list(loader)


def validate_external_reference(
    batches: Sequence[StructureBatch], cutoff: float
) -> dict[str, int | bool]:
    vesin = VesinCpuBackend(cutoff)
    total_edges = 0
    for batch_index, batch in enumerate(batches):
        actual = canonical_keys(torch_radius_graph(batch, cutoff))
        expected = canonical_keys(vesin(batch, cutoff))
        if not np.array_equal(actual, expected):
            missing = len(set(map(tuple, expected)) - set(map(tuple, actual)))
            extra = len(set(map(tuple, actual)) - set(map(tuple, expected)))
            raise AssertionError(
                f"Matbench structure {batch_index} differs from Vesin: {missing=} {extra=}"
            )
        total_edges += len(actual)
    return {
        "exact_key_match": True,
        "structures": len(batches),
        "edges": total_edges,
    }


def measure_backend(
    name: str,
    backend: Backend,
    batches: Sequence[StructureBatch],
    cutoff: float,
    repeats: int,
    warmup_seconds: float,
) -> dict[str, object]:
    warmup_start = time.perf_counter()
    warmup_traversals = 0
    while time.perf_counter() - warmup_start < warmup_seconds:
        for batch in batches:
            backend(batch, cutoff)
        warmup_traversals += 1

    elapsed_ms = []
    edge_count = 0
    for repeat in range(repeats):
        start = time.perf_counter()
        current_edges = 0
        for batch in batches:
            current_edges += backend(batch, cutoff)[0].shape[1]
        elapsed_ms.append((time.perf_counter() - start) * 1000)
        if repeat == 0:
            edge_count = current_edges
    median_ms = statistics.median(elapsed_ms)
    total_atoms = sum(len(batch.positions) for batch in batches)
    return {
        "backend": name,
        "median_ms": median_ms,
        "minimum_ms": min(elapsed_ms),
        "maximum_ms": max(elapsed_ms),
        "samples_ms": elapsed_ms,
        "warmup_traversals": warmup_traversals,
        "repeats": repeats,
        "structures": len(batches),
        "atoms": total_atoms,
        "edges": edge_count,
        "structures_per_second": 1000 * len(batches) / median_ms,
        "atoms_per_second": 1000 * total_atoms / median_ms,
        "edges_per_second": 1000 * edge_count / median_ms,
    }


def benchmark_workload(
    name: str,
    batches: Sequence[StructureBatch],
    cutoff: float,
    repeats: int,
    warmup_seconds: float,
) -> dict[str, object]:
    vesin = VesinCpuBackend(cutoff)
    source_ids = [batch.source_ids[0] for batch in batches]
    production = measure_backend(
        "torch_radius_graph_cpu",
        torch_radius_graph,
        batches,
        cutoff,
        repeats,
        warmup_seconds,
    )
    baseline = measure_backend(
        "vesin_cpu_reused",
        vesin,
        batches,
        cutoff,
        repeats,
        warmup_seconds,
    )
    return {
        "name": name,
        "source_ids": source_ids if len(source_ids) <= 16 else None,
        "source_id_count": len(source_ids),
        "source_id_sha256": hashlib.sha256("\n".join(source_ids).encode()).hexdigest(),
        "vesin_over_torch_radius_graph": (
            baseline["median_ms"] / production["median_ms"]
        ),
        "backends": {
            "torch_radius_graph_cpu": production,
            "vesin_cpu_reused": baseline,
        },
    }


def scaling_batches(
    dataset: MatbenchStructureDataset,
) -> tuple[str, list[StructureBatch]]:
    structure = select_scaling_structure(dataset)
    source_id = str(structure["source_id"])
    batches = [
        collate_structures([repeat_structure(structure, (factor, factor, factor))])
        for factor in (1, 2, 3, 4, 6, 8)
    ]
    return source_id, batches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the single-structure CPU radius-graph path."
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("cache/matbench_mp_e_form/sample-1536-seed-20260809.npz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/data/matbench_mp_e_form_sample.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("runs/cpu-benchmark.json"))
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20_260_809)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1 or args.warmup_seconds <= 0:
        raise ValueError("repeats and warmup-seconds must be positive")
    if args.cpu is not None:
        available_cpus = os.sched_getaffinity(0)
        if args.cpu not in available_cpus:
            raise ValueError(f"CPU {args.cpu} is outside the process affinity")
        os.sched_setaffinity(0, {args.cpu})
    repository_root = Path(__file__).resolve().parents[1]
    worktree_clean = git_worktree_is_clean(repository_root)
    if args.require_clean and not worktree_clean:
        raise RuntimeError("--require-clean needs a clean Git worktree")
    torch.set_num_threads(1)
    dataset = MatbenchStructureDataset(args.cache, args.manifest, torch.float64)
    batches = load_single_structure_batches(dataset, args.seed)
    validation = validate_external_reference(batches, args.cutoff)
    workloads = [
        benchmark_workload(
            "matbench_single_structure_epoch",
            batches,
            args.cutoff,
            args.repeats,
            args.warmup_seconds,
        )
    ]
    scaling_source_id, scaled = scaling_batches(dataset)
    for batch in scaled:
        workloads.append(
            benchmark_workload(
                f"matbench_supercell_{batch.source_ids[0].split('__')[-1]}",
                [batch],
                args.cutoff,
                max(args.repeats, 12),
                args.warmup_seconds,
            )
        )

    manifest = json.loads(args.manifest.read_text())
    report = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu": cpu_model(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "torch_num_threads": torch.get_num_threads(),
            "repository_revision": git_revision(repository_root),
            "repository_worktree_clean": worktree_clean,
            "cpu_extension_sha256": file_sha256(Path(_C_cpu.__file__)),
            "vesin_version": __import__("vesin").__version__,
        },
        "method": {
            "cutoff_angstrom": args.cutoff,
            "dtype": "float64",
            "batch_size": 1,
            "data_loading_timed": False,
            "dataloader": "map-style, deterministic shuffle, num_workers=0",
            "warmup_seconds_per_backend_and_workload": args.warmup_seconds,
            "statistic": "median wall time; minimum and maximum retained",
            "cpu_backend": "single-threaded hybrid exhaustive/cell-list; exhaustive candidate limit 16384",
            "output_order_compared": False,
            "exact_keys_compared": "(source, target, Sx, Sy, Sz)",
            "vesin": "one reused NeighborList with full_list=True, sorted=False, n_threads=1",
        },
        "dataset": {
            "manifest_sha256": file_sha256(args.manifest),
            "cache_sha256": file_sha256(args.cache),
            "sample_size": len(dataset),
            "sampling": manifest["sampling"],
            "source": manifest["dataset"],
            "supercell_source_id": scaling_source_id,
        },
        "validation": validation,
        "workloads": workloads,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
