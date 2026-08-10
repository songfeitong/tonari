from __future__ import annotations

import argparse
import gc
import hashlib
import json
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

from benchmarks.baselines import (
    dense_candidate_count,
    torch_dense_batch,
    vesin_gpu_batch,
)
from benchmarks.matbench_data import (
    MatbenchStructureDataset,
    StructureBatch,
    collate_structures,
    repeat_structure,
    select_scaling_structure,
)
from tonari import _C_cuda, find_neighbors

Backend = Callable[[Tensor, Tensor, Tensor, float, Tensor], tuple[Tensor, Tensor]]
BACKENDS: dict[str, Backend] = {
    "tonari_cuda": find_neighbors,
    "vesin_gpu_per_structure": vesin_gpu_batch,
    "torch_dense_batch": torch_dense_batch,
}


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
    keys = torch.cat((pair_indices.T, shifts.to(torch.int64)), dim=1).cpu().numpy()
    if len(keys) == 0:
        return keys
    order = np.lexsort(tuple(keys[:, column] for column in range(4, -1, -1)))
    return keys[order]


def call_backend(
    backend: Backend, batch: StructureBatch, cutoff: float
) -> tuple[Tensor, Tensor]:
    return backend(batch.positions, batch.cells, batch.pbc, cutoff, batch.offsets)


def validate_external_reference(
    batches: Sequence[StructureBatch], cutoff: float
) -> dict[str, int | bool]:
    total_pairs = 0
    for batch_index, batch in enumerate(batches):
        actual = canonical_keys(call_backend(find_neighbors, batch, cutoff))
        expected = canonical_keys(call_backend(vesin_gpu_batch, batch, cutoff))
        if not np.array_equal(actual, expected):
            missing = len(set(map(tuple, expected)) - set(map(tuple, actual)))
            extra = len(set(map(tuple, actual)) - set(map(tuple, expected)))
            raise AssertionError(
                f"Matbench batch {batch_index} differs from Vesin: {missing=} {extra=}"
            )
        total_pairs += len(actual)
    return {
        "exact_key_match": True,
        "batches": len(batches),
        "structures": sum(len(batch.source_ids) for batch in batches),
        "pairs": total_pairs,
    }


def validate_dense_baseline(
    batch: StructureBatch, cutoff: float
) -> dict[str, int | bool]:
    actual = canonical_keys(call_backend(find_neighbors, batch, cutoff))
    expected = canonical_keys(call_backend(torch_dense_batch, batch, cutoff))
    if not np.array_equal(actual, expected):
        raise AssertionError(
            "CUDA production path differs from the dense batch baseline"
        )
    return {"exact_key_match": True, "pairs": len(actual)}


def measure_backend(
    backend_name: str,
    batches: Sequence[StructureBatch],
    cutoff: float,
    repeats: int,
) -> dict[str, float | int | str]:
    backend = BACKENDS[backend_name]
    warmup_batches = batches[: min(3, len(batches))]
    for batch in warmup_batches:
        call_backend(backend, batch, cutoff)
    torch.cuda.synchronize()

    elapsed_ms = []
    pair_count = 0
    for repeat in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        current_pairs = 0
        for batch in batches:
            output = call_backend(backend, batch, cutoff)
            current_pairs += output[0].shape[1]
        torch.cuda.synchronize()
        elapsed_ms.append((time.perf_counter() - start) * 1000)
        if repeat == 0:
            pair_count = current_pairs

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()
    for batch in batches:
        output = call_backend(backend, batch, cutoff)
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() - baseline_memory
    del output
    total_structures = sum(len(batch.source_ids) for batch in batches)
    total_atoms = sum(len(batch.positions) for batch in batches)
    median_ms = statistics.median(elapsed_ms)
    return {
        "backend": backend_name,
        "median_ms": median_ms,
        "minimum_ms": min(elapsed_ms),
        "maximum_ms": max(elapsed_ms),
        "repeats": repeats,
        "structures": total_structures,
        "atoms": total_atoms,
        "pairs": pair_count,
        "structures_per_second": 1000 * total_structures / median_ms,
        "atoms_per_second": 1000 * total_atoms / median_ms,
        "pairs_per_second": 1000 * pair_count / median_ms,
        "torch_peak_bytes": peak_memory,
        "memory_note": (
            "Torch allocator only; Vesin native temporary allocations may not be visible"
            if backend_name == "vesin_gpu_per_structure"
            else "Torch allocator peak above resident inputs"
        ),
    }


def benchmark_workload(
    name: str,
    batches: Sequence[StructureBatch],
    cutoff: float,
    repeats: int,
    dense_candidate_limit: int,
) -> dict[str, object]:
    candidate_counts = [
        dense_candidate_count(batch.offsets, batch.cells, cutoff) for batch in batches
    ]
    source_ids = [source_id for batch in batches for source_id in batch.source_ids]
    results: dict[str, object] = {
        "name": name,
        "batch_count": len(batches),
        "source_id_count": len(source_ids),
        "source_id_sha256": hashlib.sha256("\n".join(source_ids).encode()).hexdigest(),
        "source_ids": source_ids if len(source_ids) <= 64 else None,
        "dense_candidate_count_max": max(candidate_counts),
        "backends": {},
    }
    for backend_name in ("tonari_cuda", "vesin_gpu_per_structure"):
        results["backends"][backend_name] = measure_backend(
            backend_name, batches, cutoff, repeats
        )
    if max(candidate_counts) <= dense_candidate_limit:
        results["backends"]["torch_dense_batch"] = measure_backend(
            "torch_dense_batch", batches, cutoff, repeats
        )
    else:
        results["backends"]["torch_dense_batch"] = {
            "status": "skipped",
            "reason": (
                f"estimated {max(candidate_counts):,} atom-pair-image candidates exceeds "
                f"the safety limit {dense_candidate_limit:,}"
            ),
        }
    return results


def load_gpu_batches(
    dataset: MatbenchStructureDataset,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[StructureBatch]:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_structures,
    )
    batches = [batch.to(device) for batch in loader]
    torch.cuda.synchronize()
    return batches


def scaling_batches(
    dataset: MatbenchStructureDataset, device: torch.device
) -> tuple[str, list[tuple[int, StructureBatch]]]:
    structure = select_scaling_structure(dataset)
    source_id = str(structure["source_id"])
    batches = []
    for factor in (1, 2, 3, 4, 6, 8):
        repeated = repeat_structure(structure, (factor, factor, factor))
        batches.append((factor, collate_structures([repeated]).pin_memory().to(device)))
    return source_id, batches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark periodic CUDA batch neighbor search."
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
    parser.add_argument(
        "--output", type=Path, default=Path("runs/final-benchmark.json")
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20_260_809)
    parser.add_argument("--dense-candidate-limit", type=int, default=150_000_000)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    worktree_clean = git_worktree_is_clean(repository_root)
    if args.require_clean and not worktree_clean:
        raise RuntimeError("--require-clean needs a clean Git worktree")
    device = torch.device("cuda")
    dataset = MatbenchStructureDataset(args.cache, args.manifest, torch.float32)
    batches = load_gpu_batches(dataset, args.batch_size, args.seed, device)
    candidate_counts = np.asarray(
        [
            dense_candidate_count(batch.offsets, batch.cells, args.cutoff)
            for batch in batches
        ]
    )
    median_batch = batches[int(np.argsort(candidate_counts)[len(batches) // 2])]
    validation = {
        "vesin_all_sampled_structures": validate_external_reference(
            batches, args.cutoff
        ),
        "dense_median_batch": validate_dense_baseline(median_batch, args.cutoff),
    }

    workloads = [
        benchmark_workload(
            f"matbench_dataloader_epoch_bs{args.batch_size}",
            batches,
            args.cutoff,
            args.repeats,
            0,
        ),
        benchmark_workload(
            f"matbench_median_batch_bs{args.batch_size}",
            [median_batch],
            args.cutoff,
            max(args.repeats, 12),
            args.dense_candidate_limit,
        ),
    ]
    scaling_source_id, scaled = scaling_batches(dataset, device)
    for factor, batch in scaled:
        workloads.append(
            benchmark_workload(
                f"matbench_supercell_{factor}x{factor}x{factor}",
                [batch],
                args.cutoff,
                max(args.repeats, 12),
                args.dense_candidate_limit,
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
            "equiformer_v3_reference_revision": git_revision(
                repository_root.parent / "references" / "repos" / "equiformer_v3"
            ),
        },
        "method": {
            "cutoff_angstrom": args.cutoff,
            "dtype": "float32",
            "batch_size": args.batch_size,
            "data_transfer_timed": False,
            "output_order_compared": False,
            "exact_keys_compared": "(source, target, Sx, Sy, Sz)",
            "dense_baseline": "independent Equiformer/FairChem-style N^2 x padded-images tensor expansion adjusted to strict cutoff and onsite policy",
            "dense_candidate_limit": args.dense_candidate_limit,
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
