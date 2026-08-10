from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np

DATASET_ID = "DS_5drebe4tktiu_0"
DATASET_PAGE = f"https://materials.colabfit.org/id/{DATASET_ID}"
HF_REPOSITORY = "colabfit/Matbench_mp_e_form"
HF_COMMIT = "9880d5b9b62877ec5aa14d1a4c2a9ff4ee870b8d"
PARQUET_PATH = "co/co_0.parquet"
PARQUET_BYTES = 128_655_162
PARQUET_SHA256 = "4b815791cc31862895b23cda7339d96217c37815c8f183949dc59b3035ee2afd"
DEFAULT_SAMPLE_SIZE = 1_536
DEFAULT_SEED = 20_260_809
ATOM_BOUNDARIES = (4, 8, 16, 32, 64, 128, 256)
ANISOTROPY_BOUNDARIES = (1.2, 1.5, 2.0, 3.0, 6.0)
SKEW_BOUNDARIES = (0.05, 0.2, 0.4, 0.7)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_parquet(path: Path, endpoint: str) -> str:
    url = (
        f"{endpoint.rstrip('/')}/datasets/{HF_REPOSITORY}/resolve/"
        f"{HF_COMMIT}/{PARQUET_PATH}?download=true"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if (
        path.exists()
        and path.stat().st_size == PARQUET_BYTES
        and sha256(path) == PARQUET_SHA256
    ):
        return url
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url)
    resume_bytes = temporary.stat().st_size if temporary.exists() else 0
    if resume_bytes:
        request.add_header("Range", f"bytes={resume_bytes}-")
    with urllib.request.urlopen(request) as response:
        mode = "ab" if resume_bytes and response.status == 206 else "wb"
        with temporary.open(mode) as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    temporary.replace(path)
    if path.stat().st_size != PARQUET_BYTES or sha256(path) != PARQUET_SHA256:
        raise RuntimeError(
            "downloaded Matbench parquet does not match the pinned size and SHA-256"
        )
    return url


def cell_metrics(
    cell: np.ndarray,
) -> tuple[list[float], list[float], float, float, float]:
    lengths = np.linalg.norm(cell, axis=1)
    if np.any(lengths <= 0):
        raise ValueError("Matbench structure contains a zero cell vector")
    cosines = np.array(
        [
            np.dot(cell[1], cell[2]) / (lengths[1] * lengths[2]),
            np.dot(cell[0], cell[2]) / (lengths[0] * lengths[2]),
            np.dot(cell[0], cell[1]) / (lengths[0] * lengths[1]),
        ]
    )
    cosines = np.clip(cosines, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosines))
    anisotropy = float(lengths.max() / lengths.min())
    skew = float(np.abs(cosines).max())
    volume = float(abs(np.linalg.det(cell)))
    return lengths.tolist(), angles.tolist(), anisotropy, skew, volume


def sample_rows(
    table, sample_size: int, seed: int
) -> tuple[list[int], list[dict[str, object]]]:
    configuration_ids = table["configuration_id"].to_pylist()
    names = table["names"].to_pylist()
    cells = table["cell"].to_pylist()
    pbc_values = table["pbc"].to_pylist()
    nsites = table["nsites"].to_pylist()
    nelements = table["nelements"].to_pylist()
    formulas = table["chemical_formula_reduced"].to_pylist()
    buckets: dict[
        tuple[int, int, int, int], list[tuple[str, int, dict[str, object]]]
    ] = defaultdict(list)
    for row_index, configuration_id in enumerate(configuration_ids):
        cell = np.asarray(cells[row_index], dtype=np.float64)
        lengths, angles, anisotropy, skew, volume = cell_metrics(cell)
        atom_bucket = bisect_right(ATOM_BOUNDARIES, nsites[row_index])
        anisotropy_bucket = bisect_right(ANISOTROPY_BOUNDARIES, anisotropy)
        skew_bucket = bisect_right(SKEW_BOUNDARIES, skew)
        element_bucket = min(int(nelements[row_index]), 6)
        stratum = (atom_bucket, anisotropy_bucket, skew_bucket, element_bucket)
        score = hashlib.sha256(f"{seed}:{configuration_id}".encode()).hexdigest()
        matbench_ids = [
            name for name in names[row_index] if name.startswith("matbench_mp_e_form_")
        ]
        entry = {
            "row_index": row_index,
            "configuration_id": configuration_id,
            "matbench_id": matbench_ids[0] if matbench_ids else None,
            "formula": formulas[row_index],
            "n_atoms": int(nsites[row_index]),
            "n_elements": int(nelements[row_index]),
            "pbc": [bool(value) for value in pbc_values[row_index]],
            "cell_lengths_angstrom": [round(value, 8) for value in lengths],
            "cell_angles_degree": [round(value, 8) for value in angles],
            "cell_anisotropy": round(anisotropy, 8),
            "cell_skew": round(skew, 8),
            "cell_volume_angstrom3": round(volume, 8),
            "stratum": list(stratum),
        }
        buckets[stratum].append((score, row_index, entry))
    for candidates in buckets.values():
        candidates.sort(key=lambda candidate: candidate[0])

    selected: list[tuple[int, dict[str, object]]] = []
    round_index = 0
    ordered_strata = sorted(buckets)
    while len(selected) < sample_size:
        added = 0
        for stratum in ordered_strata:
            candidates = buckets[stratum]
            if round_index < len(candidates):
                _, row_index, entry = candidates[round_index]
                selected.append((row_index, entry))
                added += 1
                if len(selected) == sample_size:
                    break
        if added == 0:
            raise ValueError(
                f"requested {sample_size} samples from a dataset with too few rows"
            )
        round_index += 1
    return [row_index for row_index, _ in selected], [entry for _, entry in selected]


def write_sample_cache(
    parquet_path: Path, sample_path: Path, row_indices: list[int]
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    structures = pq.read_table(
        parquet_path,
        columns=["cell", "positions", "pbc", "atomic_numbers"],
    ).take(pa.array(row_indices, type=pa.int64()))
    positions_rows = structures["positions"].to_pylist()
    numbers_rows = structures["atomic_numbers"].to_pylist()
    positions = np.concatenate(
        [np.asarray(values, dtype=np.float64) for values in positions_rows], axis=0
    )
    atomic_numbers = np.concatenate(
        [np.asarray(values, dtype=np.int32) for values in numbers_rows], axis=0
    )
    counts = np.asarray([len(values) for values in positions_rows], dtype=np.int64)
    batch_ptr = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    cells = np.asarray(structures["cell"].to_pylist(), dtype=np.float64)
    pbc = np.asarray(structures["pbc"].to_pylist(), dtype=np.bool_)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sample_path,
        positions=positions,
        batch_ptr=batch_ptr,
        cells=cells,
        pbc=pbc,
        atomic_numbers=atomic_numbers,
        row_indices=np.asarray(row_indices, dtype=np.int64),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and deterministically sample Matbench MP formation-energy structures."
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache/matbench_mp_e_form")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/data/matbench_mp_e_form_sample.json"),
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--endpoint",
        default="https://huggingface.co",
        help="Hugging Face endpoint; use https://hf-mirror.com where the origin is unavailable.",
    )
    args = parser.parse_args()
    if not 1_000 <= args.sample_size <= 2_000:
        raise ValueError("sample-size must be between 1000 and 2000")

    parquet_path = args.cache_dir / "co_0.parquet"
    download_url = download_parquet(parquet_path, args.endpoint)
    import pyarrow.parquet as pq

    metadata = pq.read_table(
        parquet_path,
        columns=[
            "configuration_id",
            "names",
            "cell",
            "pbc",
            "nsites",
            "nelements",
            "chemical_formula_reduced",
        ],
    )
    row_indices, entries = sample_rows(metadata, args.sample_size, args.seed)
    sample_path = args.cache_dir / f"sample-{args.sample_size}-seed-{args.seed}.npz"
    write_sample_cache(parquet_path, sample_path, row_indices)
    manifest = {
        "dataset": {
            "name": "matbench_mp_e_form",
            "colabfit_id": DATASET_ID,
            "page": DATASET_PAGE,
            "configuration_count_observed": metadata.num_rows,
            "hf_repository": HF_REPOSITORY,
            "hf_commit": HF_COMMIT,
            "parquet_path": PARQUET_PATH,
            "download_url_used": download_url,
            "parquet_bytes": PARQUET_BYTES,
            "parquet_sha256": PARQUET_SHA256,
        },
        "sampling": {
            "sample_size": args.sample_size,
            "seed": args.seed,
            "score": "sha256(f'{seed}:{configuration_id}')",
            "method": "round-robin over fixed atom-count, cell-anisotropy, cell-skew, and element-count strata; stable hash order within each stratum",
            "atom_count_boundaries": list(ATOM_BOUNDARIES),
            "anisotropy_boundaries": list(ANISOTROPY_BOUNDARIES),
            "skew_boundaries": list(SKEW_BOUNDARIES),
            "unique_formulas": len({entry["formula"] for entry in entries}),
            "minimum_atoms": min(entry["n_atoms"] for entry in entries),
            "maximum_atoms": max(entry["n_atoms"] for entry in entries),
        },
        "cache_file": sample_path.name,
        "structures": entries,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "cache": str(sample_path),
                "sample_size": args.sample_size,
                "unique_formulas": manifest["sampling"]["unique_formulas"],
                "minimum_atoms": manifest["sampling"]["minimum_atoms"],
                "maximum_atoms": manifest["sampling"]["maximum_atoms"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
