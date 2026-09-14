"""Measure synchronized single-batch latency on real molecules and crystals."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from benchmarks.common import file_sha256, git_revision, git_worktree_is_clean
from benchmarks.matbench_data import MatbenchStructureDataset
from benchmarks.qmugs_data import QmugsStructureDataset, select_qmugs
from benchmarks.run_cuda_benchmark import (
    BACKENDS,
    CUDA_EXTENSION,
    call_backend,
    validate_external_reference,
)
from benchmarks.structure_data import collate_structures


def summary(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": float(np.median(values)),
        "p10_ms": float(np.percentile(values, 10)),
        "p90_ms": float(np.percentile(values, 90)),
        "minimum_ms": min(values),
        "maximum_ms": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[8, 32, 64, 128, 256, 512, 1024]
    )
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/cuda-batch-scaling.json")
    )
    args = parser.parse_args()
    if min(*args.batch_sizes, args.batches, args.repeats, args.warmup) < 1:
        parser.error("batch sizes, batches, repeats and warmup must be positive")
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(1)
    sources = {
        "qmugs_population": (
            "cache/qmugs/sample-8192-seed-20260810.npz",
            "benchmarks/data/qmugs_sample.json",
        ),
        "matbench": (
            "cache/matbench_mp_e_form/sample-1536-seed-20260809.npz",
            "benchmarks/data/matbench_mp_e_form_sample.json",
        ),
    }
    report = {
        "environment": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_num_threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "nvidia_smi": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,driver_version,pstate,temperature.gpu,power.draw",
                    "--format=csv",
                ],
                text=True,
            ),
            "repository_revision": git_revision(root),
            "repository_worktree_clean": git_worktree_is_clean(root),
            "cuda_extension_sha256": file_sha256(Path(CUDA_EXTENSION.__file__)),
            "benchmark_script_sha256": file_sha256(Path(__file__)),
            "vesin_version": __import__("vesin").__version__,
        },
        "method": {
            "batch_sizes": args.batch_sizes,
            "batches_per_size": args.batches,
            "repeats_per_batch": args.repeats,
            "warmup_calls_per_backend_per_batch": args.warmup,
            "seed": args.seed,
            "sampling": "Each seed generates a random permutation without replacement; take its batch-size prefix. Same seeds across sizes, nested inputs; overlap between batches is allowed. No partial batches.",
            "statistic": "Median across per-batch latency medians; p10/p90 across those medians describe input-composition variation, not a confidence interval.",
            "timing": "perf_counter_ns around one public API call, CUDA synchronize before and after; output release outside timing; alternating backend order across repeats",
            "cutoff_angstrom": args.cutoff,
            "dtype": "float32",
            "quantities": "PS",
            "algorithm": "auto",
            "half_list": False,
            "include_self": False,
            "sorted": False,
            "data_loading_and_h2d_timed": False,
            "vesin": "reused NeighborList; CUDA per-structure compute with batch offsets and concatenation included",
            "validation": "Every measured batch exact-compared with Vesin using sorted (source,target,Sx,Sy,Sz) keys outside timing",
        },
        "datasets": {},
        "workloads": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, (cache, manifest) in sources.items():
        if name == "qmugs_population":
            dataset = select_qmugs(
                QmugsStructureDataset(Path(cache), Path(manifest), torch.float32),
                "population",
            )
        else:
            dataset = MatbenchStructureDataset(
                Path(cache), Path(manifest), torch.float32
            )
        if max(args.batch_sizes) > len(dataset):
            parser.error(f"batch size exceeds {name} population")
        report["datasets"][name] = {
            "structures": len(dataset),
            "cache_sha256": file_sha256(Path(cache)),
            "manifest_sha256": file_sha256(Path(manifest)),
        }
        permutations = [
            torch.randperm(
                len(dataset), generator=torch.Generator().manual_seed(args.seed + index)
            ).tolist()
            for index in range(args.batches)
        ]
        for size in args.batch_sizes:
            records = []
            print(f"{name} bs={size}: starting", flush=True)
            for index, permutation in enumerate(permutations):
                batch = (
                    collate_structures([dataset[i] for i in permutation[:size]])
                    .pin_memory()
                    .to(torch.device("cuda"))
                )
                torch.cuda.synchronize()
                validation = validate_external_reference([batch], args.cutoff)
                backend_names = ["production_cuda", "vesin_gpu_per_structure"]
                samples = {backend: [] for backend in backend_names}
                for backend in backend_names:
                    for _ in range(args.warmup):
                        output = call_backend(BACKENDS[backend], batch, args.cutoff)
                        torch.cuda.synchronize()
                        del output
                for repeat in range(args.repeats):
                    order = (
                        backend_names
                        if (repeat + index) % 2 == 0
                        else backend_names[::-1]
                    )
                    for backend in order:
                        torch.cuda.synchronize()
                        start = time.perf_counter_ns()
                        output = call_backend(BACKENDS[backend], batch, args.cutoff)
                        torch.cuda.synchronize()
                        samples[backend].append((time.perf_counter_ns() - start) / 1e6)
                        del output
                records.append(
                    {
                        "seed": args.seed + index,
                        "structures": size,
                        "atoms": len(batch.positions),
                        "source_ids": batch.source_ids,
                        "source_id_sha256": hashlib.sha256(
                            "\n".join(batch.source_ids).encode()
                        ).hexdigest(),
                        "validation": validation,
                        "backends": {
                            backend: {**summary(times), "samples_ms": times}
                            for backend, times in samples.items()
                        },
                    }
                )
                del batch
            aggregates = {
                backend: summary(
                    [record["backends"][backend]["median_ms"] for record in records]
                )
                for backend in backend_names
            }
            report["workloads"].append(
                {
                    "dataset": name,
                    "batch_size": size,
                    "backends": aggregates,
                    "batches": records,
                }
            )
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(f"{name} bs={size}: {aggregates}", flush=True)


if __name__ == "__main__":
    main()
