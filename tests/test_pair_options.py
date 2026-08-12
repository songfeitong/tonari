from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.neighborlist import NeighborList as AseNeighborList
from ase.neighborlist import PrimitiveNeighborList
from vesin import NeighborList as VesinNeighborList

from tests.support.assertions import PairKey, pair_keys
from tests.support.reference import neighbor_list_reference
from tonari import neighbor_list


def reverse_key(key: PairKey) -> PairKey:
    source, target, shift_x, shift_y, shift_z = key
    return target, source, -shift_x, -shift_y, -shift_z


def canonical_key(key: PairKey) -> PairKey:
    return min(key, reverse_key(key))


def canonicalize(keys: Iterable[PairKey]) -> set[PairKey]:
    return {canonical_key(key) for key in keys}


def expand_half_list(keys: Iterable[PairKey]) -> set[PairKey]:
    expanded = set(keys)
    expanded.update(reverse_key(key) for key in keys)
    return expanded


def zero_shift_self_keys(n_atoms: int) -> set[PairKey]:
    return {(atom, atom, 0, 0, 0) for atom in range(n_atoms)}


def vesin_keys(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    cutoff: float,
    *,
    half_list: bool,
    include_self: bool,
) -> set[PairKey]:
    neighbor_list = VesinNeighborList(
        cutoff=cutoff,
        full_list=not half_list,
        sorted=False,
        n_threads=1,
    )
    first, second, shifts = neighbor_list.compute(
        positions, cell, pbc, quantities="ijS"
    )
    keys = {
        (int(i), int(j), int(shift[0]), int(shift[1]), int(shift[2]))
        for i, j, shift in zip(first, second, shifts, strict=True)
    }
    if half_list:
        keys = canonicalize(keys)
    if include_self:
        # Vesin deliberately excludes zero-distance pairs, so the adapter adds
        # Public zero-shift self pairs are added after native neighbor search.
        keys |= zero_shift_self_keys(len(positions))
    return keys


def ase_keys(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    cutoff: float,
    *,
    half_list: bool,
    include_self: bool,
) -> set[PairKey]:
    atoms = Atoms(
        numbers=np.ones(len(positions), dtype=np.int32),
        positions=positions,
        cell=cell,
        pbc=pbc,
    )
    neighbor_list = AseNeighborList(
        np.full(len(positions), cutoff / 2),
        skin=0.0,
        sorted=False,
        self_interaction=include_self,
        bothways=not half_list,
        primitive=PrimitiveNeighborList,
    )
    neighbor_list.update(atoms)
    keys: set[PairKey] = set()
    for source in range(len(positions)):
        targets, shifts = neighbor_list.get_neighbors(source)
        keys.update(
            (source, int(target), int(shift[0]), int(shift[1]), int(shift[2]))
            for target, shift in zip(targets, shifts, strict=True)
        )
    return canonicalize(keys) if half_list else keys


@pytest.fixture
def triclinic_structure() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    positions = np.array(
        [[0.1, 0.2, 0.3], [0.9, 0.4, 0.8], [1.6, 1.0, 0.5]],
        dtype=np.float64,
    )
    cell = np.array(
        [[1.7, 0.0, 0.0], [0.45, 1.6, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    return positions, cell, np.array([True, True, False]), 2.4


@pytest.mark.parametrize("half_list", [False, True])
@pytest.mark.parametrize("include_self", [False, True])
def test_cpu_numpy_reference_vesin_and_ase_agree(
    triclinic_structure: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    half_list: bool,
    include_self: bool,
) -> None:
    positions, cell, pbc, cutoff = triclinic_structure
    options = {"half_list": half_list, "include_self": include_self}
    actual_numpy = pair_keys(
        neighbor_list("PS", positions, cell, pbc, cutoff, **options)
    )
    actual_torch = pair_keys(
        neighbor_list(
            "PS",
            torch.from_numpy(positions),
            torch.from_numpy(cell),
            torch.from_numpy(pbc),
            cutoff,
            **options,
        )
    )
    reference = pair_keys(
        neighbor_list_reference(
            torch.from_numpy(positions),
            torch.from_numpy(cell)[None],
            torch.from_numpy(pbc)[None],
            cutoff,
            torch.tensor([0, len(positions)]),
            **options,
        )
    )

    assert actual_numpy == actual_torch == reference
    assert actual_torch == vesin_keys(positions, cell, pbc, cutoff, **options)
    assert actual_torch == ase_keys(positions, cell, pbc, cutoff, **options)


def test_pair_options_obey_reverse_and_self_invariants(
    triclinic_structure: tuple[np.ndarray, np.ndarray, np.ndarray, float],
) -> None:
    positions, cell, pbc, cutoff = triclinic_structure
    full = pair_keys(neighbor_list("PS", positions, cell, pbc, cutoff))
    full_with_self = pair_keys(
        neighbor_list("PS", positions, cell, pbc, cutoff, include_self=True)
    )
    half = pair_keys(neighbor_list("PS", positions, cell, pbc, cutoff, half_list=True))
    half_with_self = pair_keys(
        neighbor_list(
            "PS",
            positions,
            cell,
            pbc,
            cutoff,
            half_list=True,
            include_self=True,
        )
    )
    self_keys = zero_shift_self_keys(len(positions))

    assert full_with_self - full == self_keys
    assert half_with_self - half == self_keys
    assert half == canonicalize(full)
    assert half_with_self == canonicalize(full_with_self)
    assert expand_half_list(half) == full
    assert expand_half_list(half_with_self) == full_with_self


def test_periodic_self_images_are_independent_of_zero_shift_self_option() -> None:
    positions = np.zeros((1, 3), dtype=np.float64)
    cell = np.diag([0.4, 8.0, 8.0])
    pbc = np.array([True, False, False])
    full = pair_keys(neighbor_list("PS", positions, cell, pbc, 1.0))
    half = pair_keys(neighbor_list("PS", positions, cell, pbc, 1.0, half_list=True))

    assert full == {
        (0, 0, -2, 0, 0),
        (0, 0, -1, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 2, 0, 0),
    }
    assert half == {(0, 0, -2, 0, 0), (0, 0, -1, 0, 0)}
    assert pair_keys(
        neighbor_list("PS", positions, cell, pbc, 1.0, include_self=True)
    ) == (full | {(0, 0, 0, 0, 0)})


@pytest.mark.parametrize("half_list", [False, True])
def test_strict_boundary_and_distinct_coincident_atoms_are_not_self_pairs(
    half_list: bool,
) -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    options = {"half_list": half_list, "include_self": True}
    keys = pair_keys(
        neighbor_list(
            "PS",
            positions,
            torch.zeros((3, 3)),
            torch.zeros(3, dtype=torch.bool),
            1.0,
            **options,
        )
    )
    expected = zero_shift_self_keys(3)
    expected.add((0, 1, 0, 0, 0))
    if not half_list:
        expected.add((1, 0, 0, 0, 0))
    assert keys == expected


@pytest.mark.parametrize(
    ("dtype", "cutoff"),
    [(torch.float32, 1e-30), (torch.float64, 1e-200)],
)
@pytest.mark.parametrize("half_list", [False, True])
def test_zero_shift_self_does_not_depend_on_squared_cutoff_underflow(
    dtype: torch.dtype,
    cutoff: float,
    half_list: bool,
) -> None:
    positions = torch.zeros((1, 3), dtype=dtype)
    cell = torch.zeros((3, 3), dtype=dtype)
    pbc = torch.zeros(3, dtype=torch.bool)
    options = {"half_list": half_list, "include_self": True}
    expected = {(0, 0, 0, 0, 0)}

    assert (
        pair_keys(neighbor_list("PS", positions, cell, pbc, cutoff, **options))
        == expected
    )
    assert (
        pair_keys(
            neighbor_list_reference(
                positions,
                cell[None],
                pbc[None],
                cutoff,
                torch.tensor([0, len(positions)]),
                **options,
            )
        )
        == expected
    )


@pytest.mark.parametrize("half_list", [False, True])
@pytest.mark.parametrize("include_self", [False, True])
def test_mixed_batch_pair_options_match_individual_structures(
    half_list: bool,
    include_self: bool,
) -> None:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    cell = torch.stack(
        (
            torch.zeros((3, 3), dtype=torch.float64),
            torch.diag(torch.tensor([0.5, 8.0, 8.0])),
        )
    )
    pbc = torch.tensor([[False, False, False], [True, False, False]])
    batch_ptr = torch.tensor([0, 0, 2, 3])
    cell = torch.cat((torch.zeros((1, 3, 3), dtype=torch.float64), cell))
    pbc = torch.cat((torch.zeros((1, 3), dtype=torch.bool), pbc))
    options = {"half_list": half_list, "include_self": include_self}

    batched = pair_keys(
        neighbor_list("PS", positions, cell, pbc, 0.6, batch_ptr, **options)
    )
    first = pair_keys(
        neighbor_list("PS", positions[:2], cell[1], pbc[1], 0.6, **options)
    )
    second = {
        (source + 2, target + 2, shift_x, shift_y, shift_z)
        for source, target, shift_x, shift_y, shift_z in pair_keys(
            neighbor_list("PS", positions[2:], cell[2], pbc[2], 0.6, **options)
        )
    }
    assert batched == first | second


@pytest.mark.parametrize("option", ["sorted", "half_list", "include_self"])
def test_boolean_options_require_bool(option: str) -> None:
    options = {option: 1}
    with pytest.raises(
        TypeError, match="sorted, half_list, and include_self must be bool"
    ):
        neighbor_list(
            "PS",
            np.zeros((1, 3)),
            np.zeros((3, 3)),
            np.zeros(3, dtype=np.bool_),
            1.0,
            **options,
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("n_atoms", "algorithm"), [(12, "brute_force"), (256, "cell_list")]
)
@pytest.mark.parametrize("half_list", [False, True])
@pytest.mark.parametrize("include_self", [False, True])
def test_cuda_brute_force_and_cell_paths_match_cpu_and_reference(
    n_atoms: int,
    algorithm: str,
    half_list: bool,
    include_self: bool,
) -> None:
    generator = torch.Generator().manual_seed(1841 + n_atoms)
    positions = torch.rand((n_atoms, 3), generator=generator, dtype=torch.float32) * 8
    cell = torch.zeros((3, 3), dtype=torch.float32)
    pbc = torch.zeros(3, dtype=torch.bool)
    options = {"half_list": half_list, "include_self": include_self}
    cpu = pair_keys(
        neighbor_list("PS", positions, cell, pbc, 1.1, algorithm=algorithm, **options)
    )
    cuda = pair_keys(
        neighbor_list(
            "PS",
            positions.cuda(),
            cell.cuda(),
            pbc.cuda(),
            1.1,
            algorithm=algorithm,
            **options,
        )
    )
    reference = pair_keys(
        neighbor_list_reference(
            positions,
            cell[None],
            pbc[None],
            1.1,
            torch.tensor([0, len(positions)]),
            **options,
        )
    )
    assert cuda == cpu == reference


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("half_list", [False, True])
@pytest.mark.parametrize("include_self", [False, True])
def test_cuda_auto_fallback_preserves_pair_options(
    half_list: bool,
    include_self: bool,
) -> None:
    generator = torch.Generator().manual_seed(9041)
    cell = torch.tensor(
        [[8.0, 0.2, 0.1], [0.1, 8.5, 0.3], [0.2, 0.1, 9.0]],
        dtype=torch.float64,
    )
    positions = torch.rand((256, 3), generator=generator, dtype=torch.float64) @ cell
    positions[0] += 2 * cell[0] - 3 * cell[2]
    pbc = torch.ones(3, dtype=torch.bool)
    options = {"half_list": half_list, "include_self": include_self}

    cpu = pair_keys(neighbor_list("PS", positions, cell, pbc, 0.8, **options))
    cuda = pair_keys(
        neighbor_list("PS", positions.cuda(), cell.cuda(), pbc.cuda(), 0.8, **options)
    )
    reference = pair_keys(
        neighbor_list_reference(
            positions,
            cell[None],
            pbc[None],
            0.8,
            torch.tensor([0, len(positions)]),
            **options,
        )
    )
    assert cuda == cpu == reference
