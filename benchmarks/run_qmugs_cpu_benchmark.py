from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import torch

from benchmarks.common import (
    cpu_frequency_policy,
    file_sha256,
    git_revision,
    git_worktree_is_clean,
)
from benchmarks.qmugs_data import QmugsStructureDataset, select_qmugs
from benchmarks.run_cpu_benchmark import (
    benchmark_workload,
    cpu_model,
    load_single_structure_batches,
    validate_external_reference,
)
from benchmarks.structure_data import StructureBatch
from tonari._extensions import load_torch_cpu

CPU_EXTENSION = load_torch_cpu()


def load_workloads(
    dataset: QmugsStructureDataset, seed: int
) -> dict[str, list[StructureBatch]]:
    workloads = {
        "qmugs_population_single_structure_epoch": load_single_structure_batches(
            select_qmugs(dataset, "population"), seed
        ),
        "qmugs_size_balanced_single_structure_epoch": load_single_structure_batches(
            select_qmugs(dataset, "size_balanced"), seed
        ),
    }
    for bin_index in range(len(dataset.heavy_atom_boundaries) + 1):
        workloads[f"qmugs_size_balanced_bin_{bin_index}"] = (
            load_single_structure_batches(
                select_qmugs(dataset, "size_balanced", bin_index), seed
            )
        )
    return workloads


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark finite-molecule CPU neighbor search on QMugs."
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("cache/qmugs/sample-8192-seed-20260810.npz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/data/qmugs_sample.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/qmugs-cpu-benchmark.json")
    )
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20_260_810)
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
    dataset = QmugsStructureDataset(args.cache, args.manifest, torch.float64)
    batches_by_workload = load_workloads(dataset, args.seed)
    all_batches = (
        batches_by_workload["qmugs_population_single_structure_epoch"]
        + batches_by_workload["qmugs_size_balanced_single_structure_epoch"]
    )
    validation = validate_external_reference(all_batches, args.cutoff)
    workloads = [
        benchmark_workload(
            name,
            batches,
            args.cutoff,
            args.repeats,
            args.warmup_seconds,
        )
        for name, batches in batches_by_workload.items()
    ]

    manifest = json.loads(args.manifest.read_text())
    report = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu": cpu_model(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "cpu_frequency_policy": cpu_frequency_policy(args.cpu),
            "torch_num_threads": torch.get_num_threads(),
            "repository_revision": git_revision(repository_root),
            "repository_worktree_clean": worktree_clean,
            "cpu_extension_sha256": file_sha256(Path(CPU_EXTENSION.__file__)),
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
            "output_order_compared": False,
            "exact_keys_compared": "(source, target, Sx, Sy, Sz)",
            "vesin": "one reused NeighborList with full_list=True, sorted=False, n_threads=1",
        },
        "dataset": {
            "manifest_sha256": file_sha256(args.manifest),
            "structures_file_sha256": file_sha256(
                args.manifest.parent / manifest["structures_file"]
            ),
            "cache_sha256": file_sha256(args.cache),
            "sample_size": len(dataset),
            "sampling": manifest["sampling"],
            "source": manifest["dataset"],
        },
        "validation": validation,
        "workloads": workloads,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
