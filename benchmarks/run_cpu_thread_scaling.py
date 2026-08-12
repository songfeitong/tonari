from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from vesin import NeighborList

from benchmarks.common import (
    canonical_keys,
    cpu_frequency_policy,
    file_sha256,
    git_revision,
    git_worktree_is_clean,
)
from benchmarks.matbench_data import (
    MatbenchStructureDataset,
    repeat_structure,
    select_scaling_structure,
)
from benchmarks.qmugs_data import QmugsStructureDataset, select_qmugs
from benchmarks.run_cpu_benchmark import cpu_model
from benchmarks.structure_data import StructureBatch, collate_structures
from tonari import neighbor_list
from tonari._extensions import load_torch_cpu

CPU_EXTENSION = load_torch_cpu()
Backend = Callable[[], int]


def parse_thread_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(item) for item in value.split(","))
    if (
        1 not in counts
        or any(count < 1 for count in counts)
        or len(set(counts)) != len(counts)
    ):
        raise argparse.ArgumentTypeError(
            "thread counts must be distinct positive comma-separated integers including 1"
        )
    return counts


def parse_cpus(value: str) -> tuple[int, ...]:
    cpus = tuple(int(item) for item in value.split(","))
    if not cpus or len(set(cpus)) != len(cpus):
        raise argparse.ArgumentTypeError(
            "CPUs must be distinct comma-separated integers"
        )
    return cpus


def production_backend(
    batch: StructureBatch, cutoff: float, num_threads: int
) -> Backend:
    def run() -> int:
        pairs, _ = neighbor_list(
            "PS",
            batch.positions,
            batch.cell,
            batch.pbc,
            cutoff,
            batch.batch_ptr,
            num_threads=num_threads,
        )
        return len(pairs)

    return run


def vesin_backend(batch: StructureBatch, cutoff: float, num_threads: int) -> Backend:
    search = NeighborList(
        cutoff=cutoff,
        full_list=True,
        sorted=False,
        n_threads=num_threads,
    )
    boundaries = batch.batch_ptr.tolist()

    def run() -> int:
        pair_count = 0
        for structure, (start, stop) in enumerate(pairwise(boundaries)):
            first, _, _ = search.compute(
                batch.positions[start:stop],
                batch.cell[structure],
                batch.pbc[structure],
                "ijS",
            )
            pair_count += len(first)
        return pair_count

    return run


def measure(
    backend: Backend,
    repeats: int,
    warmup_seconds: float,
) -> dict[str, object]:
    warmup_start = time.perf_counter()
    warmup_runs = 0
    while time.perf_counter() - warmup_start < warmup_seconds:
        backend()
        warmup_runs += 1

    samples_ms = []
    pair_count = 0
    for repeat in range(repeats):
        start = time.perf_counter()
        current_pairs = backend()
        samples_ms.append((time.perf_counter() - start) * 1000)
        if repeat == 0:
            pair_count = current_pairs
        elif current_pairs != pair_count:
            raise RuntimeError("backend pair count changed between benchmark repeats")
    return {
        "median_ms": statistics.median(samples_ms),
        "minimum_ms": min(samples_ms),
        "maximum_ms": max(samples_ms),
        "samples_ms": samples_ms,
        "warmup_runs": warmup_runs,
        "repeats": repeats,
        "pairs": pair_count,
    }


def canonical_output_digest(
    pair_indices: Tensor, cell_shifts: Tensor, batch_ptr: Tensor
) -> tuple[str, int]:
    boundaries = batch_ptr.tolist()
    edge_boundaries = torch.searchsorted(
        pair_indices[:, 0].contiguous(), batch_ptr, right=False
    ).tolist()
    digest = hashlib.sha256()
    total_pairs = 0
    for structure in range(len(boundaries) - 1):
        edge_start = edge_boundaries[structure]
        edge_stop = edge_boundaries[structure + 1]
        keys = canonical_keys(
            (
                pair_indices[edge_start:edge_stop],
                cell_shifts[edge_start:edge_stop],
            )
        )
        digest.update(np.asarray([len(keys)], dtype="<i8").tobytes())
        digest.update(keys.astype("<i8", copy=False).tobytes())
        total_pairs += len(keys)
    return digest.hexdigest(), total_pairs


def validate_against_vesin(
    batch: StructureBatch,
    cutoff: float,
    thread_counts: tuple[int, ...],
) -> dict[str, object]:
    actual_pairs, actual_shifts = neighbor_list(
        "PS",
        batch.positions,
        batch.cell,
        batch.pbc,
        cutoff,
        batch.batch_ptr,
        num_threads=1,
        sorted=True,
    )
    search = NeighborList(
        cutoff=cutoff,
        full_list=True,
        sorted=False,
        n_threads=1,
    )
    boundaries = batch.batch_ptr.tolist()
    edge_boundaries = torch.searchsorted(
        actual_pairs[:, 0].contiguous(), batch.batch_ptr, right=False
    ).tolist()
    digest = hashlib.sha256()
    total_pairs = 0
    for structure, (start, stop) in enumerate(pairwise(boundaries)):
        edge_start = edge_boundaries[structure]
        edge_stop = edge_boundaries[structure + 1]
        actual = canonical_keys(
            (
                actual_pairs[edge_start:edge_stop],
                actual_shifts[edge_start:edge_stop],
            )
        )
        first, second, shifts = search.compute(
            batch.positions[start:stop],
            batch.cell[structure],
            batch.pbc[structure],
            "ijS",
        )
        expected_pairs = torch.stack(
            (first.to(torch.int64) + start, second.to(torch.int64) + start), dim=1
        )
        expected = canonical_keys((expected_pairs, shifts))
        if not np.array_equal(actual, expected):
            missing = len(set(map(tuple, expected)) - set(map(tuple, actual)))
            extra = len(set(map(tuple, actual)) - set(map(tuple, expected)))
            raise AssertionError(
                f"structure {structure} differs from Vesin: {missing=} {extra=}"
            )
        digest.update(np.asarray([len(actual)], dtype="<i8").tobytes())
        digest.update(actual.astype("<i8", copy=False).tobytes())
        total_pairs += len(actual)
    if total_pairs != len(actual_pairs):
        raise AssertionError("production batch contains cross-structure pairs")
    reference_digest = digest.hexdigest()
    thread_digests = {"1": reference_digest}
    for num_threads in thread_counts:
        if num_threads == 1:
            continue
        threaded_pairs, threaded_shifts = neighbor_list(
            "PS",
            batch.positions,
            batch.cell,
            batch.pbc,
            cutoff,
            batch.batch_ptr,
            num_threads=num_threads,
            sorted=True,
        )
        threaded_digest, threaded_pair_count = canonical_output_digest(
            threaded_pairs, threaded_shifts, batch.batch_ptr
        )
        if threaded_pair_count != total_pairs or threaded_digest != reference_digest:
            raise AssertionError(
                f"Tonari differs from its exact Vesin-validated reference at "
                f"{num_threads} threads"
            )
        thread_digests[str(num_threads)] = threaded_digest
    return {
        "exact_key_match": True,
        "canonical_key_sha256": reference_digest,
        "thread_count_key_match": {
            str(num_threads): thread_digests[str(num_threads)] == reference_digest
            for num_threads in thread_counts
        },
        "structures": len(batch.source_ids),
        "atoms": len(batch.positions),
        "pairs": total_pairs,
    }


def benchmark_workload(
    name: str,
    batch: StructureBatch,
    cutoff: float,
    thread_counts: tuple[int, ...],
    repeats: int,
    warmup_seconds: float,
) -> dict[str, object]:
    validation = validate_against_vesin(batch, cutoff, thread_counts)
    measurements: dict[str, dict[str, dict[str, object]]] = {}
    for num_threads in thread_counts:
        production = measure(
            production_backend(batch, cutoff, num_threads),
            repeats,
            warmup_seconds,
        )
        vesin = measure(
            vesin_backend(batch, cutoff, num_threads),
            repeats,
            warmup_seconds,
        )
        if production["pairs"] != vesin["pairs"]:
            raise AssertionError(
                f"pair counts differ for {name} at {num_threads} threads"
            )
        measurements[str(num_threads)] = {
            "tonari": production,
            "vesin": vesin,
        }
    one_thread = measurements["1"]
    for num_threads, result in measurements.items():
        result["tonari"]["speedup_over_one_thread"] = (
            one_thread["tonari"]["median_ms"] / result["tonari"]["median_ms"]
        )
        result["vesin"]["speedup_over_one_thread"] = (
            one_thread["vesin"]["median_ms"] / result["vesin"]["median_ms"]
        )
        result["vesin_over_tonari"] = (
            result["vesin"]["median_ms"] / result["tonari"]["median_ms"]
        )
    source_ids = batch.source_ids
    return {
        "name": name,
        "source_ids": list(source_ids) if len(source_ids) <= 16 else None,
        "source_id_count": len(source_ids),
        "source_id_sha256": hashlib.sha256("\n".join(source_ids).encode()).hexdigest(),
        "validation": validation,
        "measurements": measurements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CPU thread scaling on real Tonari workloads."
    )
    parser.add_argument(
        "--matbench-cache",
        type=Path,
        default=Path("cache/matbench_mp_e_form/sample-1536-seed-20260809.npz"),
    )
    parser.add_argument(
        "--matbench-manifest",
        type=Path,
        default=Path("benchmarks/data/matbench_mp_e_form_sample.json"),
    )
    parser.add_argument(
        "--qmugs-cache",
        type=Path,
        default=Path("cache/qmugs/sample-8192-seed-20260810.npz"),
    )
    parser.add_argument(
        "--qmugs-manifest",
        type=Path,
        default=Path("benchmarks/data/qmugs_sample.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/cpu-thread-scaling.json")
    )
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--threads", type=parse_thread_counts, default=(1, 2, 4, 8))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--cpus", type=parse_cpus)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1 or args.warmup_seconds <= 0:
        raise ValueError("repeats and warmup-seconds must be positive")
    if args.cpus is not None:
        available = os.sched_getaffinity(0)
        if not set(args.cpus) <= available:
            raise ValueError("requested CPUs are outside the process affinity")
        if max(args.threads) > len(args.cpus):
            raise ValueError("maximum thread count exceeds the selected CPU affinity")
        os.sched_setaffinity(0, args.cpus)

    repository_root = Path(__file__).resolve().parents[1]
    worktree_clean = git_worktree_is_clean(repository_root)
    if args.require_clean and not worktree_clean:
        raise RuntimeError("--require-clean needs a clean Git worktree")
    torch.set_num_threads(1)

    matbench = MatbenchStructureDataset(
        args.matbench_cache, args.matbench_manifest, torch.float64
    )
    qmugs = QmugsStructureDataset(args.qmugs_cache, args.qmugs_manifest, torch.float64)
    qmugs_population = select_qmugs(qmugs, "population")
    scaling_structure = select_scaling_structure(matbench)
    workloads = (
        (
            "matbench_1536_structure_batch",
            collate_structures([matbench[index] for index in range(len(matbench))]),
        ),
        (
            "qmugs_population_4096_structure_batch",
            collate_structures(
                [qmugs_population[index] for index in range(len(qmugs_population))]
            ),
        ),
        (
            "matbench_32768_atom_supercell",
            collate_structures([repeat_structure(scaling_structure, (8, 8, 8))]),
        ),
    )
    results = [
        benchmark_workload(
            name,
            batch,
            args.cutoff,
            args.threads,
            args.repeats,
            args.warmup_seconds,
        )
        for name, batch in workloads
    ]
    affinity = sorted(os.sched_getaffinity(0))
    report = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu": cpu_model(),
            "cpu_affinity": affinity,
            "cpu_frequency_policy": {
                str(cpu): cpu_frequency_policy(cpu) for cpu in affinity
            },
            "torch_num_threads": torch.get_num_threads(),
            "repository_revision": git_revision(repository_root),
            "repository_worktree_clean": worktree_clean,
            "cpu_extension_sha256": file_sha256(Path(CPU_EXTENSION.__file__)),
            "vesin_version": __import__("vesin").__version__,
        },
        "method": {
            "cutoff_angstrom": args.cutoff,
            "dtype": "float64",
            "thread_counts": list(args.threads),
            "data_loading_timed": False,
            "warmup_seconds_per_backend_workload_and_thread_count": args.warmup_seconds,
            "statistic": "median wall time; minimum, maximum, and samples retained",
            "tonari": "one native batch call; num_threads includes the caller",
            "vesin": "one reused NeighborList per measurement; n_threads matches Tonari; batches require one public compute call per structure",
            "output_order_compared": False,
            "exact_keys_compared": "(source, target, Sx, Sy, Sz)",
            "threaded_validation": "num_threads=1 compared exactly with Vesin per structure; every other thread count must have the same per-structure canonical-key SHA-256",
        },
        "datasets": {
            "matbench_manifest_sha256": file_sha256(args.matbench_manifest),
            "matbench_cache_sha256": file_sha256(args.matbench_cache),
            "qmugs_manifest_sha256": file_sha256(args.qmugs_manifest),
            "qmugs_cache_sha256": file_sha256(args.qmugs_cache),
        },
        "workloads": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
