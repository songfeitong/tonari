from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch

from benchmarks.baselines import dense_candidate_count
from benchmarks.qmugs_data import QmugsStructureDataset, select_qmugs
from benchmarks.run_cuda_benchmark import (
    benchmark_workload,
    file_sha256,
    git_revision,
    git_worktree_is_clean,
    load_gpu_batches,
    validate_dense_baseline,
    validate_external_reference,
)
from benchmarks.structure_data import StructureBatch
from tonari import _C_cuda


def median_batch(batches: list[StructureBatch], cutoff: float) -> StructureBatch:
    candidate_counts = np.asarray(
        [
            dense_candidate_count(batch.offsets, batch.cells, batch.pbc, cutoff)
            for batch in batches
        ]
    )
    order = np.argsort(candidate_counts, kind="stable")
    return batches[int(order[len(batches) // 2])]


def load_workloads(
    dataset: QmugsStructureDataset,
    batch_size: int,
    batch_sizes: list[int],
    seed: int,
    cutoff: float,
    device: torch.device,
) -> tuple[dict[str, list[StructureBatch]], list[StructureBatch]]:
    population_by_batch_size = {
        current_batch_size: load_gpu_batches(
            select_qmugs(dataset, "population"),
            current_batch_size,
            seed,
            device,
        )
        for current_batch_size in batch_sizes
    }
    population = population_by_batch_size[batch_size]
    balanced = load_gpu_batches(
        select_qmugs(dataset, "size_balanced"), batch_size, seed, device
    )
    workloads = {
        f"qmugs_population_dataloader_epoch_bs{current_batch_size}": batches
        for current_batch_size, batches in population_by_batch_size.items()
    }
    workloads.update(
        {
            f"qmugs_size_balanced_dataloader_epoch_bs{batch_size}": balanced,
            f"qmugs_population_median_batch_bs{batch_size}": [
                median_batch(population, cutoff)
            ],
        }
    )
    dense_validation_batches = [
        workloads[f"qmugs_population_median_batch_bs{batch_size}"][0]
    ]
    for bin_index in range(len(dataset.heavy_atom_boundaries) + 1):
        batches = load_gpu_batches(
            select_qmugs(dataset, "size_balanced", bin_index),
            batch_size,
            seed,
            device,
        )
        representative = median_batch(batches, cutoff)
        workloads[
            f"qmugs_size_balanced_bin_{bin_index}_median_batch_bs{batch_size}"
        ] = [representative]
        dense_validation_batches.append(representative)
    return workloads, dense_validation_batches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark finite-molecule CUDA neighbor search on QMugs."
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
        "--output", type=Path, default=Path("runs/qmugs-cuda-benchmark.json")
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 32, 64, 128])
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20_260_810)
    parser.add_argument("--dense-candidate-limit", type=int, default=150_000_000)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    if args.batch_size not in args.batch_sizes:
        raise ValueError("--batch-sizes must contain --batch-size")
    if any(batch_size < 1 for batch_size in args.batch_sizes):
        raise ValueError("batch sizes must be positive")

    repository_root = Path(__file__).resolve().parents[1]
    worktree_clean = git_worktree_is_clean(repository_root)
    if args.require_clean and not worktree_clean:
        raise RuntimeError("--require-clean needs a clean Git worktree")
    device = torch.device("cuda")
    dataset = QmugsStructureDataset(args.cache, args.manifest, torch.float32)
    workloads_by_name, dense_validation_batches = load_workloads(
        dataset,
        args.batch_size,
        args.batch_sizes,
        args.seed,
        args.cutoff,
        device,
    )
    population_batches = workloads_by_name[
        f"qmugs_population_dataloader_epoch_bs{args.batch_size}"
    ]
    balanced_batches = workloads_by_name[
        f"qmugs_size_balanced_dataloader_epoch_bs{args.batch_size}"
    ]
    validation = {
        "vesin_all_sampled_structures": validate_external_reference(
            population_batches + balanced_batches, args.cutoff
        ),
        "dense_representative_batches": [
            validate_dense_baseline(batch, args.cutoff)
            for batch in dense_validation_batches
        ],
    }
    workloads = []
    for name, batches in workloads_by_name.items():
        is_epoch = "epoch" in name
        workloads.append(
            benchmark_workload(
                name,
                batches,
                args.cutoff,
                args.repeats if is_epoch else max(args.repeats, 21),
                0 if is_epoch else args.dense_candidate_limit,
            )
        )

    manifest = json.loads(args.manifest.read_text())
    report = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "repository_revision": git_revision(repository_root),
            "repository_worktree_clean": worktree_clean,
            "cuda_extension_sha256": file_sha256(Path(_C_cuda.__file__)),
            "vesin_version": __import__("vesin").__version__,
        },
        "method": {
            "cutoff_angstrom": args.cutoff,
            "dtype": "float32",
            "batch_size": args.batch_size,
            "population_batch_size_sweep": args.batch_sizes,
            "data_transfer_timed": False,
            "data_loading_timed": False,
            "dataloader": "map-style, deterministic shuffle, num_workers=0, pin_memory=True",
            "output_order_compared": False,
            "exact_keys_compared": "(source, target, Sx, Sy, Sz)",
            "dense_baseline": "independent finite-system N^2 tensor expansion with strict cutoff and onsite exclusion",
            "dense_candidate_limit": args.dense_candidate_limit,
            "vesin": "one reused NeighborList; per-structure CUDA compute followed by concatenation",
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
