from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import ase
import numpy as np
import torch

from benchmarks.common import (
    canonical_keys,
    cpu_frequency_policy,
    file_sha256,
    git_revision,
    git_worktree_is_clean,
)
from benchmarks.matbench_data import MatbenchStructureDataset
from benchmarks.pair_option_backends import (
    AseCpuBackend,
    NativeCpuBackend,
    PairOptions,
    VesinCpuBackend,
)
from benchmarks.run_cpu_benchmark import (
    cpu_model,
    load_single_structure_batches,
)
from benchmarks.structure_data import StructureBatch
from tonari._extensions import load_torch_cpu

CPU_EXTENSION = load_torch_cpu()

Backend = Callable[[StructureBatch, float], tuple[torch.Tensor, torch.Tensor]]


def validate_backends(
    batches: Sequence[StructureBatch], cutoff: float, options: PairOptions
) -> dict[str, object]:
    backends: dict[str, Backend] = {
        "vesin_cpu_reused": VesinCpuBackend(cutoff, options),
        "ase_primitive_rebuilt": AseCpuBackend(cutoff, options),
    }
    production = NativeCpuBackend(options)
    total_pairs = 0
    for batch_index, batch in enumerate(batches):
        actual = canonical_keys(production(batch, cutoff))
        for name, backend in backends.items():
            expected = canonical_keys(backend(batch, cutoff))
            if not np.array_equal(actual, expected):
                raise AssertionError(
                    f"structure {batch_index} differs from {name} in {options.name}"
                )
        total_pairs += len(actual)
    return {
        "exact_key_match": True,
        "structures": len(batches),
        "pairs": total_pairs,
        "external_references": sorted(backends),
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

    samples_ms = []
    pair_count = 0
    output_bytes = 0
    for repeat in range(repeats):
        start = time.perf_counter()
        current_pairs = 0
        current_bytes = 0
        for batch in batches:
            pair_indices, cell_shifts = backend(batch, cutoff)
            current_pairs += pair_indices.shape[0]
            current_bytes += pair_indices.numel() * pair_indices.element_size()
            current_bytes += cell_shifts.numel() * cell_shifts.element_size()
        samples_ms.append((time.perf_counter() - start) * 1000)
        if repeat == 0:
            pair_count = current_pairs
            output_bytes = current_bytes
    median_ms = statistics.median(samples_ms)
    return {
        "backend": name,
        "median_ms": median_ms,
        "minimum_ms": min(samples_ms),
        "maximum_ms": max(samples_ms),
        "samples_ms": samples_ms,
        "warmup_traversals": warmup_traversals,
        "repeats": repeats,
        "pairs": pair_count,
        "output_bytes": output_bytes,
        "structures_per_second": 1000 * len(batches) / median_ms,
    }


def benchmark_mode(
    batches: Sequence[StructureBatch],
    cutoff: float,
    options: PairOptions,
    repeats: int,
    warmup_seconds: float,
) -> dict[str, object]:
    backends: dict[str, Backend] = {
        "native_cpu": NativeCpuBackend(options),
        "vesin_cpu_reused": VesinCpuBackend(cutoff, options),
        "ase_primitive_rebuilt": AseCpuBackend(cutoff, options),
    }
    results = {
        name: measure_backend(name, backend, batches, cutoff, repeats, warmup_seconds)
        for name, backend in backends.items()
    }
    return {
        "name": options.name,
        "half_list": options.half_list,
        "include_self": options.include_self,
        "backends": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CPU full/half lists and zero-shift self pairs."
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
        "--output", type=Path, default=Path("runs/pair-options-cpu.json")
    )
    parser.add_argument("--structures", type=int, default=256)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20_260_809)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()

    if args.structures < 1 or args.repeats < 1 or args.warmup_seconds <= 0:
        raise ValueError("structures, repeats, and warmup-seconds must be positive")
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
    batches = load_single_structure_batches(dataset, args.seed)[: args.structures]
    options = [
        PairOptions(half_list=False, include_self=False),
        PairOptions(half_list=False, include_self=True),
        PairOptions(half_list=True, include_self=False),
        PairOptions(half_list=True, include_self=True),
    ]
    validation = {
        option.name: validate_backends(batches, args.cutoff, option)
        for option in options
    }
    workloads = [
        benchmark_mode(
            batches,
            args.cutoff,
            option,
            args.repeats,
            args.warmup_seconds,
        )
        for option in options
    ]
    source_ids = [batch.source_ids[0] for batch in batches]
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
            "ase_version": ase.__version__,
        },
        "method": {
            "cutoff_angstrom": args.cutoff,
            "dtype": "float64",
            "batch_size": 1,
            "data_loading_timed": False,
            "warmup_seconds_per_backend_and_mode": args.warmup_seconds,
            "statistic": "median wall time; all samples retained",
            "native": "public one-shot neighbor_list output",
            "vesin": (
                "one NeighborList reused per mode; full_list matches the requested "
                "mode; sorted=False; n_threads=1; zero-shift self pairs are added "
                "by the timed adapter"
            ),
            "ase": (
                "PrimitiveNeighborList with skin=0, sorted=False, native "
                "self_interaction/bothways; configuration reused by atom count, "
                "build called for every structure"
            ),
            "half_list_normalization": (
                "external one-way pairs are reoriented by the public "
                "lexicographic canonical rule inside the timed adapters"
            ),
        },
        "dataset": {
            "name": "matbench_mp_e_form",
            "manifest_sha256": file_sha256(args.manifest),
            "cache_sha256": file_sha256(args.cache),
            "sample_size": len(batches),
            "selection": "first structures from the existing deterministic shuffled sample",
            "source_id_sha256": hashlib.sha256(
                "\n".join(source_ids).encode()
            ).hexdigest(),
        },
        "validation": validation,
        "workloads": workloads,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
