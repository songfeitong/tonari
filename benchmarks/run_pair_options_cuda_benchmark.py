from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import Subset

from benchmarks.baselines import vesin_gpu_batch
from benchmarks.common import (
    canonical_keys,
    file_sha256,
    git_revision,
    git_worktree_is_clean,
)
from benchmarks.matbench_data import MatbenchStructureDataset
from benchmarks.pair_option_backends import PairOptions
from benchmarks.run_cuda_benchmark import load_gpu_batches
from benchmarks.structure_data import StructureBatch
from tonari import _C_cuda, find_neighbors


def call_tonari(
    batch: StructureBatch, cutoff: float, options: PairOptions
) -> tuple[torch.Tensor, torch.Tensor]:
    return find_neighbors(
        batch.positions,
        batch.cells,
        batch.pbc,
        cutoff,
        batch.offsets,
        half_list=options.half_list,
        include_self=options.include_self,
    )


def key_set(output: tuple[torch.Tensor, torch.Tensor]) -> set[tuple[int, ...]]:
    return set(map(tuple, canonical_keys(output)))


def reverse_key(key: tuple[int, ...]) -> tuple[int, ...]:
    source, target, shift_x, shift_y, shift_z = key
    return target, source, -shift_x, -shift_y, -shift_z


def validate_modes(batches: list[StructureBatch], cutoff: float) -> dict[str, object]:
    total_pairs = {option.name: 0 for option in PAIR_OPTIONS}
    for batch_index, batch in enumerate(batches):
        outputs = {
            option.name: key_set(call_tonari(batch, cutoff, option))
            for option in PAIR_OPTIONS
        }
        full = outputs["full_without_self"]
        full_with_self = outputs["full_with_self"]
        half = outputs["half_without_self"]
        half_with_self = outputs["half_with_self"]
        zero_self = {(atom, atom, 0, 0, 0) for atom in range(len(batch.positions))}
        if full_with_self - full != zero_self:
            raise AssertionError(f"batch {batch_index} has incorrect self pairs")
        if half_with_self - half != zero_self:
            raise AssertionError(f"batch {batch_index} has incorrect half self pairs")
        if half != {min(key, reverse_key(key)) for key in full}:
            raise AssertionError(f"batch {batch_index} has a noncanonical half list")
        if half_with_self != {min(key, reverse_key(key)) for key in full_with_self}:
            raise AssertionError(
                f"batch {batch_index} has a noncanonical half list with self"
            )
        vesin = key_set(
            vesin_gpu_batch(
                batch.positions,
                batch.cells,
                batch.pbc,
                cutoff,
                batch.offsets,
            )
        )
        if full != vesin:
            raise AssertionError(f"batch {batch_index} differs from Vesin")
        for name, keys in outputs.items():
            total_pairs[name] += len(keys)
    return {
        "exact_mode_invariants": True,
        "full_without_self_matches_vesin": True,
        "batches": len(batches),
        "structures": sum(len(batch.source_ids) for batch in batches),
        "pairs": total_pairs,
    }


def measure_mode(
    batches: list[StructureBatch],
    cutoff: float,
    options: PairOptions,
    repeats: int,
) -> dict[str, object]:
    for batch in batches[: min(3, len(batches))]:
        call_tonari(batch, cutoff, options)
    torch.cuda.synchronize()

    samples_ms = []
    pair_count = 0
    output_bytes = 0
    for repeat in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        current_pairs = 0
        current_bytes = 0
        for batch in batches:
            pair_indices, cell_shifts = call_tonari(batch, cutoff, options)
            current_pairs += pair_indices.shape[1]
            current_bytes += pair_indices.numel() * pair_indices.element_size()
            current_bytes += cell_shifts.numel() * cell_shifts.element_size()
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1000)
        if repeat == 0:
            pair_count = current_pairs
            output_bytes = current_bytes

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()
    for batch in batches:
        output = call_tonari(batch, cutoff, options)
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() - baseline_memory
    del output
    median_ms = statistics.median(samples_ms)
    return {
        "name": options.name,
        "half_list": options.half_list,
        "include_self": options.include_self,
        "median_ms": median_ms,
        "minimum_ms": min(samples_ms),
        "maximum_ms": max(samples_ms),
        "repeats": repeats,
        "pairs": pair_count,
        "output_bytes": output_bytes,
        "torch_peak_bytes": peak_memory,
        "samples_ms": samples_ms,
    }


PAIR_OPTIONS = [
    PairOptions(half_list=False, include_self=False),
    PairOptions(half_list=False, include_self=True),
    PairOptions(half_list=True, include_self=False),
    PairOptions(half_list=True, include_self=True),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA full/half lists and zero-shift self pairs."
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
        "--output", type=Path, default=Path("runs/pair-options-cuda.json")
    )
    parser.add_argument("--structures", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20_260_809)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()

    if args.structures < 1 or args.batch_size < 1 or args.repeats < 1:
        raise ValueError("structures, batch-size, and repeats must be positive")
    repository_root = Path(__file__).resolve().parents[1]
    worktree_clean = git_worktree_is_clean(repository_root)
    if args.require_clean and not worktree_clean:
        raise RuntimeError("--require-clean needs a clean Git worktree")
    device = torch.device("cuda")
    full_dataset = MatbenchStructureDataset(args.cache, args.manifest, torch.float32)
    dataset = Subset(full_dataset, range(min(args.structures, len(full_dataset))))
    batches = load_gpu_batches(dataset, args.batch_size, args.seed, device)
    validation = validate_modes(batches, args.cutoff)
    workloads = [
        measure_mode(batches, args.cutoff, option, args.repeats)
        for option in PAIR_OPTIONS
    ]
    source_ids = [source_id for batch in batches for source_id in batch.source_ids]
    report = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "repository_revision": git_revision(repository_root),
            "repository_worktree_clean": worktree_clean,
            "cuda_extension_sha256": file_sha256(Path(_C_cuda.__file__)),
            "vesin_version": __import__("vesin").__version__,
        },
        "method": {
            "cutoff_angstrom": args.cutoff,
            "dtype": "float32",
            "batch_size": args.batch_size,
            "data_loading_and_h2d_timed": False,
            "statistic": "median synchronized wall time; all samples retained",
            "validation": (
                "all four modes checked by public reverse/self invariants; default "
                "full list also compared exactly with Vesin"
            ),
            "memory": "Torch allocator peak above resident inputs",
        },
        "dataset": {
            "name": "matbench_mp_e_form",
            "manifest_sha256": file_sha256(args.manifest),
            "cache_sha256": file_sha256(args.cache),
            "sample_size": len(dataset),
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
