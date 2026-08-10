from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import io
import json
import tarfile
import urllib.request
import zipfile
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATASET_DOI = "https://doi.org/10.3929/ethz-b-000482129"
DATASET_PAGE = "https://www.research-collection.ethz.ch/handle/20.500.11850/482129"
DOWNLOAD_ROOT = "https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM/download"
SUMMARY_URL = f"{DOWNLOAD_ROOT}?path=%2F&files=summary.csv"
STRUCTURES_URL = f"{DOWNLOAD_ROOT}?path=%2F&files=structures.tar.gz"
SUMMARY_BYTES = 2_026_848_085
STRUCTURES_BYTES = 7_180_016_346
SUMMARY_SHA256 = "b6d7b54fa4d290ceace81c644f20b2ddfd68c21ecb1f4b5c00e8913cd608bcfd"
STRUCTURES_SHA256 = "264102bf1c036d077a72ab558168be4c5c6054e6aeecb8a7768be36df87ad46b"
DEFAULT_SEED = 20_260_810
DEFAULT_POPULATION_SIZE = 4_096
DEFAULT_BALANCED_PER_BIN = 512
HEAVY_ATOM_BOUNDARIES = (10, 20, 30, 40, 50, 65, 80)
ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}
ELEMENT_SYMBOLS = {number: symbol for symbol, number in ATOMIC_NUMBERS.items()}


@dataclass(frozen=True, slots=True)
class Candidate:
    chembl_id: str
    conformer_id: str
    n_atoms: int
    n_heavy_atoms: int
    n_heteroatoms: int
    n_rotatable_bonds: int
    n_rings: int
    gfn2_total_energy: float
    score: int

    @property
    def heavy_atom_bin(self) -> int:
        return bisect_left(HEAVY_ATOM_BOUNDARIES, self.n_heavy_atoms)

    @property
    def source_id(self) -> str:
        return f"{self.chembl_id}/{self.conformer_id}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_download(
    path: Path, url: str, expected_bytes: int, expected_sha256: str
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if path.exists():
        final_size = path.stat().st_size
        if final_size == expected_bytes:
            actual_sha256 = sha256(path)
            if expected_sha256 and actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"{path} has an unexpected SHA-256; remove it and rerun"
                )
            return actual_sha256
        if final_size < expected_bytes and not temporary.exists():
            path.replace(temporary)
        else:
            raise RuntimeError(f"{path} has an unexpected size; remove it and rerun")

    resume_bytes = temporary.stat().st_size if temporary.exists() else 0
    if resume_bytes > expected_bytes:
        raise RuntimeError(f"{temporary} is larger than expected; remove it and rerun")
    if resume_bytes < expected_bytes:
        request = urllib.request.Request(url)
        if resume_bytes:
            request.add_header("Range", f"bytes={resume_bytes}-")
        with urllib.request.urlopen(request) as response:
            append = resume_bytes > 0 and response.status == 206
            with temporary.open("ab" if append else "wb") as output:
                while chunk := response.read(4 * 1024 * 1024):
                    output.write(chunk)

    if temporary.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"downloaded {path.name} has an unexpected size; rerun to resume"
        )
    actual_sha256 = sha256(temporary)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"downloaded {path.name} has an unexpected SHA-256; "
            f"remove {temporary} and rerun"
        )
    temporary.replace(path)
    return actual_sha256


def candidate_from_row(row: dict[str, str], seed: int) -> Candidate:
    chembl_id = row["chembl_id"]
    score = int.from_bytes(
        hashlib.sha256(f"{seed}:{chembl_id}".encode()).digest(), "big"
    )
    return Candidate(
        chembl_id=chembl_id,
        conformer_id=row["conf_id"],
        n_atoms=int(row["atoms"]),
        n_heavy_atoms=int(row["heavy_atoms"]),
        n_heteroatoms=int(row["heteroatoms"]),
        n_rotatable_bonds=int(row["rotatable_bonds"]),
        n_rings=int(row["rings"]),
        gfn2_total_energy=float(row["GFN2_TOTAL_ENERGY"]),
        score=score,
    )


def keep_smallest(
    heap: list[tuple[int, str, Candidate]], candidate: Candidate, capacity: int
) -> None:
    item = (-candidate.score, candidate.chembl_id, candidate)
    if len(heap) < capacity:
        heapq.heappush(heap, item)
    elif candidate.score < -heap[0][0]:
        heapq.heapreplace(heap, item)


def select_candidates(
    summary_path: Path,
    seed: int,
    population_size: int,
    balanced_per_bin: int,
) -> tuple[list[tuple[Candidate, str]], dict[str, int]]:
    population_heap: list[tuple[int, str, Candidate]] = []
    per_bin_capacity = population_size + balanced_per_bin
    bin_heaps: list[list[tuple[int, str, Candidate]]] = [
        [] for _ in range(len(HEAVY_ATOM_BOUNDARIES) + 1)
    ]
    molecule_count = 0
    conformer_count = 0
    current: Candidate | None = None
    seen: set[str] = set()

    def finalize(candidate: Candidate) -> None:
        nonlocal molecule_count
        if candidate.chembl_id in seen:
            raise ValueError("QMugs summary is not grouped by ChEMBL ID")
        seen.add(candidate.chembl_id)
        molecule_count += 1
        keep_smallest(population_heap, candidate, population_size)
        keep_smallest(bin_heaps[candidate.heavy_atom_bin], candidate, per_bin_capacity)

    with summary_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            conformer_count += 1
            candidate = candidate_from_row(row, seed)
            if current is None:
                current = candidate
            elif candidate.chembl_id != current.chembl_id:
                finalize(current)
                current = candidate
            elif (
                candidate.gfn2_total_energy,
                candidate.conformer_id,
            ) < (
                current.gfn2_total_energy,
                current.conformer_id,
            ):
                current = candidate
    if current is not None:
        finalize(current)

    population = sorted(
        (item[2] for item in population_heap),
        key=lambda candidate: (candidate.score, candidate.chembl_id),
    )
    population_ids = {candidate.chembl_id for candidate in population}
    balanced: list[Candidate] = []
    for bin_index, heap in enumerate(bin_heaps):
        candidates = sorted(
            (item[2] for item in heap),
            key=lambda candidate: (candidate.score, candidate.chembl_id),
        )
        selected = [
            candidate
            for candidate in candidates
            if candidate.chembl_id not in population_ids
        ][:balanced_per_bin]
        if len(selected) != balanced_per_bin:
            raise ValueError(
                f"heavy-atom bin {bin_index} has too few molecules for the sample"
            )
        balanced.extend(selected)
    return (
        [(candidate, "population") for candidate in population]
        + [(candidate, "size_balanced") for candidate in balanced],
        {"molecules": molecule_count, "conformers": conformer_count},
    )


def parse_sdf(contents: bytes) -> tuple[np.ndarray, np.ndarray]:
    lines = contents.decode("utf-8").splitlines()
    if len(lines) < 4 or "V2000" not in lines[3]:
        raise ValueError("QMugs sample is not an MDL V2000 structure")
    n_atoms = int(lines[3][:3])
    atom_lines = lines[4 : 4 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise ValueError("QMugs SDF atom block is truncated")
    positions = np.empty((n_atoms, 3), dtype=np.float64)
    atomic_numbers = np.empty(n_atoms, dtype=np.int32)
    for index, line in enumerate(atom_lines):
        fields = line.split()
        if len(fields) < 4 or fields[3] not in ATOMIC_NUMBERS:
            raise ValueError("QMugs SDF contains an unsupported atom record")
        positions[index] = [float(value) for value in fields[:3]]
        atomic_numbers[index] = ATOMIC_NUMBERS[fields[3]]
    return positions, atomic_numbers


def extract_selected_structures(
    archive_path: Path, selected: list[tuple[Candidate, str]]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    wanted = {candidate.source_id for candidate, _ in selected}
    structures: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".sdf"):
                continue
            path = Path(member.name)
            if len(path.parts) < 2:
                continue
            source_id = f"{path.parent.name}/{path.stem}"
            if source_id not in wanted:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"could not read {member.name}")
            structures[source_id] = parse_sdf(extracted.read())
            if len(structures) == len(wanted):
                break
    missing = sorted(wanted - structures.keys())
    if missing:
        raise ValueError(f"QMugs archive is missing {len(missing)} selected structures")
    return structures


def geometry_metrics(positions: np.ndarray) -> tuple[float, list[float]]:
    centered = positions - positions.mean(axis=0)
    radius_of_gyration = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if singular_values[0] == 0:
        shape_ratios = np.zeros(3)
    else:
        shape_ratios = singular_values / singular_values[0]
    return radius_of_gyration, [round(float(value), 8) for value in shape_ratios]


def write_deterministic_npz(path: Path, **arrays: np.ndarray) -> None:
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compresslevel=9)


def write_sample(
    cache_path: Path,
    manifest_path: Path,
    selected: list[tuple[Candidate, str]],
    structures: dict[str, tuple[np.ndarray, np.ndarray]],
    counts: dict[str, int],
    source_hashes: dict[str, str],
    seed: int,
    population_size: int,
    balanced_per_bin: int,
) -> None:
    positions_rows = []
    number_rows = []
    entries = []
    for candidate, workload in selected:
        positions, atomic_numbers = structures[candidate.source_id]
        if len(positions) != candidate.n_atoms:
            raise ValueError(f"atom count mismatch for {candidate.source_id}")
        positions_rows.append(positions)
        number_rows.append(atomic_numbers)
        radius, shape_ratios = geometry_metrics(positions)
        entries.append(
            {
                "source_id": candidate.source_id,
                "chembl_id": candidate.chembl_id,
                "conformer_id": candidate.conformer_id,
                "workload": workload,
                "n_atoms": candidate.n_atoms,
                "n_heavy_atoms": candidate.n_heavy_atoms,
                "heavy_atom_bin": candidate.heavy_atom_bin,
                "n_heteroatoms": candidate.n_heteroatoms,
                "n_rotatable_bonds": candidate.n_rotatable_bonds,
                "n_rings": candidate.n_rings,
                "elements": [
                    ELEMENT_SYMBOLS[number]
                    for number in sorted(set(map(int, atomic_numbers)))
                ],
                "radius_of_gyration_angstrom": round(radius, 8),
                "shape_singular_value_ratios": shape_ratios,
            }
        )

    atom_counts = np.asarray([len(row) for row in positions_rows], dtype=np.int64)
    batch_ptr = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(atom_counts)))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_npz(
        cache_path,
        positions=np.concatenate(positions_rows),
        batch_ptr=batch_ptr,
        atomic_numbers=np.concatenate(number_rows),
        source_ids=np.asarray([entry["source_id"] for entry in entries]),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    structures_path = manifest_path.with_name(f"{manifest_path.stem}_structures.csv")
    structure_fields = list(entries[0])
    with structures_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=structure_fields, lineterminator="\n"
        )
        writer.writeheader()
        for entry in entries:
            row = entry.copy()
            row["elements"] = "|".join(row["elements"])
            row["shape_singular_value_ratios"] = "|".join(
                map(str, row["shape_singular_value_ratios"])
            )
            writer.writerow(row)

    manifest = {
        "dataset": {
            "name": "QMugs",
            "release": "original QMugs data collection",
            "archive_changelog_date": "2021-07-30",
            "doi": DATASET_DOI,
            "page": DATASET_PAGE,
            "license": "CC BY-SA 3.0",
            "chembl_source": "ChEMBL 27",
            "summary_url": SUMMARY_URL,
            "summary_bytes": SUMMARY_BYTES,
            "summary_sha256": source_hashes["summary"],
            "structures_url": STRUCTURES_URL,
            "structures_bytes": STRUCTURES_BYTES,
            "structures_sha256": source_hashes["structures"],
            "molecule_count_observed": counts["molecules"],
            "conformer_count_observed": counts["conformers"],
        },
        "sampling": {
            "seed": seed,
            "score": "sha256(f'{seed}:{chembl_id}')",
            "conformer_selection": "minimum GFN2_TOTAL_ENERGY, then conformer ID",
            "population_size": population_size,
            "population_method": "globally smallest stable hash scores",
            "size_balanced_size": balanced_per_bin * (len(HEAVY_ATOM_BOUNDARIES) + 1),
            "size_balanced_per_bin": balanced_per_bin,
            "heavy_atom_boundaries": list(HEAVY_ATOM_BOUNDARIES),
            "size_balanced_method": "smallest stable hash scores in each heavy-atom bin, excluding population sample",
            "samples_are_disjoint": True,
        },
        "cache_file": cache_path.name,
        "structures_file": structures_path.name,
        "structure_count": len(entries),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and deterministically sample QMugs molecular structures."
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/qmugs"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/data/qmugs_sample.json"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument(
        "--balanced-per-bin", type=int, default=DEFAULT_BALANCED_PER_BIN
    )
    args = parser.parse_args()
    if args.population_size < 1 or args.balanced_per_bin < 1:
        raise ValueError("sample sizes must be positive")

    summary_path = args.cache_dir / "summary.csv"
    archive_path = args.cache_dir / "structures.tar.gz"
    source_hashes = {
        "summary": ensure_download(
            summary_path, SUMMARY_URL, SUMMARY_BYTES, SUMMARY_SHA256
        ),
        "structures": ensure_download(
            archive_path, STRUCTURES_URL, STRUCTURES_BYTES, STRUCTURES_SHA256
        ),
    }
    selected, counts = select_candidates(
        summary_path,
        args.seed,
        args.population_size,
        args.balanced_per_bin,
    )
    structures = extract_selected_structures(archive_path, selected)
    sample_size = len(selected)
    cache_path = args.cache_dir / f"sample-{sample_size}-seed-{args.seed}.npz"
    write_sample(
        cache_path,
        args.manifest,
        selected,
        structures,
        counts,
        source_hashes,
        args.seed,
        args.population_size,
        args.balanced_per_bin,
    )
    print(
        json.dumps(
            {
                "cache": str(cache_path),
                "manifest": str(args.manifest),
                "molecules_observed": counts["molecules"],
                "conformers_observed": counts["conformers"],
                "sample_size": sample_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
