"""Compare one-shot NumPy CPU neighbor search on periodic supercells."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import ase
import numpy as np
import torch
import vesin
from ase.neighborlist import primitive_neighbor_list

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
from benchmarks.run_cpu_benchmark import cpu_model
from tonari import neighbor_list
from tonari._extensions import load_numpy_cpu


def canonical(output: tuple[np.ndarray, ...]) -> np.ndarray:
    first, second, shifts = output
    keys = np.column_stack((first, second, shifts)).astype(np.int64)
    return keys[np.lexsort(tuple(keys[:, column] for column in range(4, -1, -1)))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=int, default=31)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--factors", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    parser.add_argument(
        "--output", type=Path, default=Path("runs/cpu-single-structure.json")
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup_seconds <= 0 or min(args.factors) < 1:
        parser.error("repeats, warmup and factors must be positive")
    os.sched_setaffinity(0, {args.core})
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    cache = Path("cache/matbench_mp_e_form/sample-1536-seed-20260809.npz")
    manifest = Path("benchmarks/data/matbench_mp_e_form_sample.json")
    dataset = MatbenchStructureDataset(cache, manifest, torch.float64)
    source = select_scaling_structure(dataset)
    report = {
        "environment": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "cpu": cpu_model(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "cpu_frequency_policy": cpu_frequency_policy(args.core),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "ase": ase.__version__,
            "vesin": vesin.__version__,
            "repository_revision": git_revision(root),
            "repository_worktree_clean": git_worktree_is_clean(root),
            "numpy_extension_sha256": file_sha256(Path(load_numpy_cpu().__file__)),
            "script_sha256": file_sha256(Path(__file__)),
        },
        "method": {
            "cutoff_angstrom": args.cutoff,
            "dtype": "float64",
            "threads": 1,
            "quantities": "ijS",
            "full_list": True,
            "include_self": False,
            "repeats": args.repeats,
            "warmup_seconds_per_backend": args.warmup_seconds,
            "timing": "single public NumPy call; perf_counter_ns; output release outside timing; rotating backend order across repeats",
            "statistic": "median wall time; raw samples retained",
            "data_loading_and_structure_preparation_timed": False,
            "tonari": "NumPy provider, algorithm=auto, cpu_threads=1, sorted=False",
            "vesin": "reused NeighborList, full_list=True, sorted=False, n_threads=1; NumPy compute each call",
            "ase": "primitive_neighbor_list, scalar cutoff, self_interaction=False, use_scaled_positions=False; default max_nbins; fresh search each call; native output ordering retained",
            "validation": "exact canonical (i,j,Sx,Sy,Sz) comparison for all three backends at every size, outside timing",
        },
        "dataset": {
            "source_id": source["source_id"],
            "manifest_sha256": file_sha256(manifest),
            "cache_sha256": file_sha256(cache),
        },
        "workloads": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for factor in args.factors:
        structure = repeat_structure(source, (factor, factor, factor))
        positions = np.ascontiguousarray(structure["positions"].numpy())
        cell = np.ascontiguousarray(structure["cell"].numpy())
        pbc = np.ascontiguousarray(structure["pbc"].numpy())
        reference = vesin.NeighborList(
            cutoff=args.cutoff, full_list=True, sorted=False, n_threads=1
        )
        backends = {
            "tonari": partial(
                neighbor_list, "ijS", positions, cell, pbc, args.cutoff, cpu_threads=1
            ),
            "vesin": partial(reference.compute, positions, cell, pbc, "ijS"),
            "ase": partial(
                primitive_neighbor_list,
                "ijS",
                pbc,
                cell,
                positions,
                args.cutoff,
                self_interaction=False,
            ),
        }
        print(f"{len(positions)} atoms: validation", flush=True)
        expected = canonical(backends["tonari"]())
        for name, call in backends.items():
            actual = canonical(call())
            if not np.array_equal(expected, actual):
                raise AssertionError(f"{name} differs at factor={factor}")
        record = {
            "name": f"matbench_supercell_{factor}x{factor}x{factor}",
            "atoms": len(positions),
            "pairs": len(expected),
            "structures": 1,
            "validation": {
                "exact_key_match": True,
                "canonical_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
            },
            "backends": {},
        }
        del actual, expected
        for name, call in backends.items():
            start = time.perf_counter()
            count = 0
            while time.perf_counter() - start < args.warmup_seconds:
                output = call()
                del output
                count += 1
            record["backends"][name] = {"warmup_calls": count, "samples_ms": []}
        names = list(backends)
        for repeat in range(args.repeats):
            offset = repeat % len(names)
            for name in names[offset:] + names[:offset]:
                start = time.perf_counter_ns()
                output = backends[name]()
                elapsed = (time.perf_counter_ns() - start) / 1e6
                record["backends"][name]["samples_ms"].append(elapsed)
                del output
        for data in record["backends"].values():
            samples = data["samples_ms"]
            data.update(
                median_ms=statistics.median(samples),
                minimum_ms=min(samples),
                maximum_ms=max(samples),
            )
        report["workloads"].append(record)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(
            len(positions),
            {name: data["median_ms"] for name, data in record["backends"].items()},
            flush=True,
        )


if __name__ == "__main__":
    main()
