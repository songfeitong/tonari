from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from vesin import NeighborList

from benchmarks.common import (
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
from benchmarks.structure_data import collate_structures
from tonari import neighbor_list
from tonari._extensions import load_numpy_cpu

CPU_EXTENSION = load_numpy_cpu()
Output = tuple[np.ndarray, np.ndarray, np.ndarray]
Backend = Callable[[], Output]


@dataclass
class NumpyBatch:
    positions: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray
    batch_ptr: np.ndarray
    source_ids: tuple[str, ...]

    @classmethod
    def from_structures(cls, structures: list[dict[str, object]]) -> NumpyBatch:
        batch = collate_structures(structures)
        return cls(
            *(
                np.ascontiguousarray(getattr(batch, key).numpy())
                for key in ("positions", "cell", "pbc", "batch_ptr")
            ),
            batch.source_ids,
        )


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


def production_backend(batch: NumpyBatch, cutoff: float, num_threads: int) -> Backend:
    def run() -> Output:
        return neighbor_list(
            "ijS",
            batch.positions,
            batch.cell,
            batch.pbc,
            cutoff,
            batch.batch_ptr,
            cpu_threads=num_threads,
        )

    return run


def vesin_backend(batch: NumpyBatch, cutoff: float, num_threads: int) -> Backend:
    search = NeighborList(
        cutoff=cutoff, full_list=True, sorted=False, n_threads=num_threads
    )
    boundaries = batch.batch_ptr.tolist()

    def run() -> Output:
        if len(batch.source_ids) == 1:
            return search.compute(batch.positions, batch.cell[0], batch.pbc[0], "ijS")
        outputs = []
        for structure, (start, stop) in enumerate(pairwise(boundaries)):
            first, second, shifts = search.compute(
                batch.positions[start:stop],
                batch.cell[structure],
                batch.pbc[structure],
                "ijS",
            )
            outputs.append((first + start, second + start, shifts))
        return tuple(
            np.concatenate([output[column] for output in outputs], axis=0)
            for column in range(3)
        )

    return run


def measure(backend: Backend, repeats: int, warmup_seconds: float) -> dict[str, object]:
    warmup_start = time.perf_counter()
    warmup_runs = 0
    while time.perf_counter() - warmup_start < warmup_seconds:
        output = backend()
        del output
        warmup_runs += 1
    samples_ms = []
    pair_count = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        output = backend()
        samples_ms.append((time.perf_counter_ns() - start) / 1e6)
        assert all(isinstance(array, np.ndarray) for array in output)
        if pair_count is None:
            pair_count = len(output[0])
        elif len(output[0]) != pair_count:
            raise RuntimeError("backend pair count changed between repeats")
        del output
    return {
        "median_ms": statistics.median(samples_ms),
        "minimum_ms": min(samples_ms),
        "maximum_ms": max(samples_ms),
        "samples_ms": samples_ms,
        "warmup_runs": warmup_runs,
        "repeats": repeats,
        "pairs": pair_count,
    }


def canonical(output: Output) -> np.ndarray:
    first, second, shifts = output
    assert all(isinstance(array, np.ndarray) for array in output)
    keys = np.column_stack((first, second, shifts)).astype("<i8", copy=False)
    return keys[np.lexsort(tuple(keys[:, column] for column in range(4, -1, -1)))]


def validate_against_vesin(
    batch: NumpyBatch, cutoff: float, thread_counts: tuple[int, ...]
) -> dict[str, object]:
    expected = canonical(production_backend(batch, cutoff, 1)())
    digest = hashlib.sha256(expected.tobytes()).hexdigest()
    matches = {}
    for threads in thread_counts:
        matches[str(threads)] = {}
        for name, factory in (("tonari", production_backend), ("vesin", vesin_backend)):
            actual = canonical(factory(batch, cutoff, threads)())
            if not np.array_equal(expected, actual):
                raise AssertionError(f"{name} differs at {threads} threads")
            matches[str(threads)][name] = hashlib.sha256(actual.tobytes()).hexdigest()
    return {
        "exact_key_match": True,
        "canonical_key_sha256": digest,
        "backend_thread_key_sha256": matches,
        "thread_count_key_match": {
            str(n): all(value == digest for value in matches[str(n)].values())
            for n in thread_counts
        },
        "structures": len(batch.source_ids),
        "atoms": len(batch.positions),
        "pairs": len(expected),
    }


def benchmark_workload(
    name: str,
    batch: NumpyBatch,
    cutoff: float,
    thread_counts: tuple[int, ...],
    repeats: int,
    warmup_seconds: float,
) -> dict[str, object]:
    print(f"{name}: validating NumPy outputs at all thread counts", flush=True)
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
        print(
            f"{name}: {num_threads} threads: tonari={production['median_ms']:.3f} ms, vesin={vesin['median_ms']:.3f} ms",
            flush=True,
        )
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
            NumpyBatch.from_structures(
                [matbench[index] for index in range(len(matbench))]
            ),
        ),
        (
            "qmugs_population_4096_structure_batch",
            NumpyBatch.from_structures(
                [qmugs_population[index] for index in range(len(qmugs_population))]
            ),
        ),
        (
            "matbench_32768_atom_supercell",
            NumpyBatch.from_structures(
                [repeat_structure(scaling_structure, (8, 8, 8))]
            ),
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
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "numpy": np.__version__,
            "script_sha256": file_sha256(Path(__file__)),
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
            "numpy_extension_sha256": file_sha256(Path(CPU_EXTENSION.__file__)),
            "vesin_version": __import__("vesin").__version__,
        },
        "method": {
            "cutoff_angstrom": args.cutoff,
            "dtype": "float64",
            "thread_counts": list(args.threads),
            "data_loading_timed": False,
            "array_api": "NumPy inputs and outputs; Torch used only for data preparation outside timing",
            "quantities": "ijS",
            "output_release_timed": False,
            "warmup_seconds_per_backend_workload_and_thread_count": args.warmup_seconds,
            "statistic": "median wall time; minimum, maximum, and samples retained",
            "tonari": "one native NumPy batch call; the requested thread count includes the caller",
            "vesin": "one reused NeighborList per measurement; n_threads matches Tonari; NumPy compute per structure; batch offsets and concatenation timed; single structure uses direct compute",
            "output_order_compared": False,
            "exact_keys_compared": "(source, target, Sx, Sy, Sz)",
            "threaded_validation": "both NumPy backends at every thread count are exact-compared against canonical single-thread Tonari keys; SHA-256 recorded for each",
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
