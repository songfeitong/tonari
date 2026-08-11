from __future__ import annotations

import argparse
from pathlib import Path

import torch

from benchmarks.matbench_data import (
    MatbenchStructureDataset,
    collate_structures,
    repeat_structure,
    select_scaling_structure,
)
from tonari import neighbor_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nsight entry point for one real supercell."
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
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--cutoff", type=float, default=5.0)
    args = parser.parse_args()
    dataset = MatbenchStructureDataset(args.cache, args.manifest, torch.float32)
    structure = select_scaling_structure(dataset)
    structure = repeat_structure(structure, (args.factor, args.factor, args.factor))
    batch = collate_structures([structure]).to(torch.device("cuda"))
    for _ in range(3):
        neighbor_list(
            "PS", batch.positions, batch.cell, batch.pbc, args.cutoff, batch.batch_ptr
        )
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("profile_neighbor_list")
    for _ in range(args.iterations):
        neighbor_list(
            "PS", batch.positions, batch.cell, batch.pbc, args.cutoff, batch.batch_ptr
        )
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
