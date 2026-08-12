from __future__ import annotations

import numpy as np
import pytest
import torch
from ase.neighborlist import primitive_neighbor_list

from tests.assertions import pair_keys
from tests.reference import neighbor_list_reference
from tonari import neighbor_list


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_matches_reference_for_mixed_batch(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(712)
    finite = torch.rand((7, 3), generator=generator, dtype=dtype) * 4.0
    partial_cell = torch.tensor(
        [[2.1, 0.2, 0.1], [0.0, 2.5, 0.3], [0.0, 0.0, 0.0]], dtype=dtype
    )
    periodic_cell = torch.tensor(
        [[1.7, 0.2, 0.0], [0.1, 1.9, 0.3], [0.2, 0.1, 2.2]], dtype=dtype
    )
    positions = torch.cat(
        (
            finite,
            torch.rand((5, 3), generator=generator, dtype=dtype)
            @ torch.tensor(
                [[2.1, 0.2, 0.1], [0.0, 2.5, 0.3], [0.0, 0.0, 8.0]], dtype=dtype
            ),
            torch.rand((4, 3), generator=generator, dtype=dtype) @ periodic_cell,
        )
    )
    batch_ptr = torch.tensor([0, 7, 12, 16])
    cell = torch.stack((torch.zeros((3, 3), dtype=dtype), partial_cell, periodic_cell))
    pbc = torch.tensor([[False, False, False], [True, True, False], [True, True, True]])
    expected = neighbor_list_reference("PS", positions, cell, pbc, 1.35, batch_ptr)
    actual = neighbor_list("PS", positions, cell, pbc, 1.35, batch_ptr)
    assert pair_keys(*actual) == pair_keys(*expected)


def test_cpu_matches_ase_for_partial_triclinic_multiple_images() -> None:
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [0.9, 0.4, 0.8], [1.6, 1.0, 0.5]], dtype=torch.float64
    )
    cell = torch.tensor(
        [[1.7, 0.0, 0.0], [0.45, 1.6, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
    )
    pbc = torch.tensor([True, True, False])
    cutoff = 2.4
    first, second, shifts = primitive_neighbor_list(
        "ijS",
        pbc.numpy(),
        cell.numpy(),
        positions.numpy(),
        cutoff,
        self_interaction=False,
    )
    expected = {
        (int(i), int(j), int(shift[0]), int(shift[1]), int(shift[2]))
        for i, j, shift in zip(first, second, shifts, strict=True)
    }
    actual = neighbor_list(
        "PS",
        positions,
        cell[None],
        pbc[None],
        cutoff,
        torch.tensor([0, len(positions)]),
    )
    assert pair_keys(*actual) == expected


def test_cpu_strict_cutoff_and_periodic_self_images() -> None:
    common = (
        torch.zeros((3, 3)),
        torch.zeros(3, dtype=torch.bool),
        1.0,
    )
    boundary = neighbor_list(
        "PS", torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), *common
    )
    assert pair_keys(*boundary) == set()
    just_inside = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))
    inside = neighbor_list(
        "PS", torch.tensor([[0.0, 0.0, 0.0], [just_inside, 0.0, 0.0]]), *common
    )
    assert pair_keys(*inside) == {(0, 1, 0, 0, 0), (1, 0, 0, 0, 0)}
    periodic = neighbor_list(
        "PS",
        torch.zeros((1, 3)),
        torch.diag(torch.tensor([0.4, 8.0, 8.0]))[None],
        torch.tensor([[True, False, False]]),
        1.0,
        torch.tensor([0, 1]),
    )
    assert pair_keys(*periodic) == {
        (0, 0, -2, 0, 0),
        (0, 0, -1, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 2, 0, 0),
    }


def test_cpu_relabels_unwrapped_representatives() -> None:
    cell = torch.tensor(
        [[2.0, 0.1, 0.0], [0.2, 2.5, 0.0], [0.1, 0.3, 3.0]], dtype=torch.float64
    )
    positions = torch.tensor([[0.2, 0.1, 0.4], [1.8, 0.2, 0.5]], dtype=torch.float64)
    translated = positions.clone()
    translated[0] -= 3 * cell[1]
    translated[1] += 2 * cell[0] - cell[2]
    common = (
        cell,
        torch.ones(3, dtype=torch.bool),
        1.0,
    )
    first_pairs, first_shifts = neighbor_list("PS", positions, *common)
    second_pairs, second_shifts = neighbor_list("PS", translated, *common)
    first_displacements = (
        positions[first_pairs[:, 1]]
        - positions[first_pairs[:, 0]]
        + first_shifts.to(positions.dtype) @ cell
    )
    second_displacements = (
        translated[second_pairs[:, 1]]
        - translated[second_pairs[:, 0]]
        + second_shifts.to(positions.dtype) @ cell
    )
    assert pair_keys(first_pairs, first_shifts) != pair_keys(
        second_pairs, second_shifts
    )
    assert sorted(map(tuple, np.round(first_displacements.numpy(), 12))) == sorted(
        map(tuple, np.round(second_displacements.numpy(), 12))
    )


def test_cpu_rotation_and_reflection_covariance() -> None:
    cell = torch.tensor(
        [[2.0, 0.3, 0.1], [0.2, 2.4, 0.4], [0.1, 0.2, 2.8]], dtype=torch.float64
    )
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [1.7, 0.4, 0.5], [0.6, 1.8, 1.1]], dtype=torch.float64
    )
    orthogonal = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float64
    )
    common = (torch.ones(3, dtype=torch.bool), 1.3)
    first = neighbor_list("PS", positions, cell, *common)
    second = neighbor_list("PS", positions @ orthogonal, cell @ orthogonal, *common)
    assert pair_keys(*first) == pair_keys(*second)


def test_cpu_randomized_differential() -> None:
    generator = torch.Generator().manual_seed(29381)
    pbc_patterns = torch.tensor(
        [
            [False, False, False],
            [True, False, False],
            [True, False, True],
            [True, True, False],
            [True, True, True],
        ]
    )
    for case in range(40):
        n_atoms = case + 1
        diagonal = 1.8 + 2.5 * torch.rand(3, generator=generator)
        cell = torch.diag(diagonal)
        cell += 0.35 * torch.tril(torch.rand((3, 3), generator=generator), diagonal=-1)
        positions = torch.rand((n_atoms, 3), generator=generator) @ cell
        pbc = pbc_patterns[case % len(pbc_patterns)]
        if case % 4 == 0:
            positions[0] += 3 * cell[0]
            if pbc[2]:
                positions[-1] -= 2 * cell[2]
        cutoff = 0.55 + 0.8 * float(torch.rand((), generator=generator))
        arguments = (
            positions,
            cell,
            pbc,
            cutoff,
        )
        assert pair_keys(*neighbor_list("PS", *arguments)) == pair_keys(
            *neighbor_list_reference("PS", *arguments)
        )


def test_cpu_allows_continuous_geometry_backward() -> None:
    positions = torch.tensor(
        [[0.1, 0.0, 0.0], [1.8, 0.2, 0.0]], dtype=torch.float64, requires_grad=True
    )
    cell = torch.tensor(
        [[[2.0, 0.0, 0.0], [0.1, 2.5, 0.0], [0.0, 0.0, 3.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    pair_indices, shifts = neighbor_list(
        "PS",
        positions,
        cell,
        torch.ones((1, 3), dtype=torch.bool),
        0.8,
        torch.tensor([0, 2]),
    )
    displacements = (
        positions[pair_indices[:, 1]]
        - positions[pair_indices[:, 0]]
        + shifts.to(positions.dtype) @ cell[0]
    )
    torch.sum(displacements.square()).backward()
    assert pair_indices.grad_fn is None
    assert shifts.grad_fn is None
    assert positions.grad is not None and torch.all(torch.isfinite(positions.grad))
    assert cell.grad is not None and torch.all(torch.isfinite(cell.grad))
    assert torch.count_nonzero(cell.grad) > 0


def test_cpu_empty_periodic_structure_skips_tiny_cell_images() -> None:
    pair_indices, shifts = neighbor_list(
        "PS",
        torch.empty((0, 3), dtype=torch.float64),
        (1e-12 * torch.eye(3, dtype=torch.float64))[None],
        torch.ones((1, 3), dtype=torch.bool),
        1.0,
        torch.tensor([0, 0]),
    )
    assert pair_indices.shape == (0, 2)
    assert shifts.shape == (0, 3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_cell_list_path_matches_reference(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(1842)
    n_atoms = 288
    cell = torch.tensor(
        [[8.0, 0.4, 0.1], [0.2, 7.5, 0.5], [0.3, 0.1, 9.0]], dtype=dtype
    )
    positions = torch.rand((n_atoms, 3), generator=generator, dtype=dtype) @ cell
    batch_ptr = torch.tensor([0, n_atoms])
    pbc = torch.ones((1, 3), dtype=torch.bool)
    expected = neighbor_list_reference("PS", positions, cell[None], pbc, 1.2, batch_ptr)
    actual = neighbor_list("PS", positions, cell[None], pbc, 1.2, batch_ptr)
    assert pair_keys(*actual) == pair_keys(*expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_cell_list_preserves_nextafter_inside_pairs(dtype: torch.dtype) -> None:
    positions = torch.zeros((129, 3), dtype=dtype)
    positions[1, 0] = torch.nextafter(
        torch.tensor(1.0, dtype=dtype), torch.tensor(0.0, dtype=dtype)
    )
    positions[2:, 1] = 3 * torch.arange(2, 129, dtype=dtype)
    pair_indices, shifts = neighbor_list(
        "PS",
        positions,
        torch.zeros((1, 3, 3), dtype=dtype),
        torch.zeros((1, 3), dtype=torch.bool),
        1.0,
        torch.tensor([0, len(positions)]),
    )
    assert pair_keys(pair_indices, shifts) == {(0, 1, 0, 0, 0), (1, 0, 0, 0, 0)}


def test_cpu_cell_list_falls_back_for_extremely_sparse_bounds() -> None:
    positions = torch.zeros((256, 3))
    positions[:, 0] = torch.arange(256) * 10000.0
    pair_indices, shifts = neighbor_list(
        "PS",
        positions,
        torch.zeros((1, 3, 3)),
        torch.zeros((1, 3), dtype=torch.bool),
        1.0,
        torch.tensor([0, 256]),
    )
    assert pair_keys(pair_indices, shifts) == set()


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_cpu_rejects_nonfinite_positions(value: float) -> None:
    positions = torch.zeros((2, 3), dtype=torch.float64)
    positions[0, 0] = value
    with pytest.raises(RuntimeError, match="positions must"):
        neighbor_list(
            "PS",
            positions,
            torch.zeros((1, 3, 3), dtype=torch.float64),
            torch.zeros((1, 3), dtype=torch.bool),
            0.5,
            torch.tensor([0, 2]),
        )


def test_cpu_rejects_representative_wrap_outside_int32_range() -> None:
    positions = torch.tensor(
        [[0.1, 0.0, 0.0], [2**31 + 0.2, 0.0, 0.0]], dtype=torch.float64
    )
    with pytest.raises(RuntimeError, match="wraps.*int32"):
        neighbor_list(
            "PS",
            positions,
            torch.eye(3, dtype=torch.float64)[None],
            torch.tensor([[True, False, False]]),
            0.5,
            torch.tensor([0, 2]),
        )


def test_cpu_rejects_output_shift_outside_int32_range() -> None:
    positions = torch.tensor(
        [[-(2**31) + 0.1, 0.0, 0.0], [2**31 - 0.8, 0.0, 0.0]], dtype=torch.float64
    )
    with pytest.raises(RuntimeError, match="cell shift.*int32"):
        neighbor_list(
            "PS",
            positions,
            torch.eye(3, dtype=torch.float64)[None],
            torch.tensor([[True, False, False]]),
            0.5,
            torch.tensor([0, 2]),
        )


def test_cpu_rejects_dependent_active_cell_rows() -> None:
    with pytest.raises(ValueError, match="linearly independent"):
        neighbor_list(
            "PS",
            torch.zeros((1, 3), dtype=torch.float64),
            torch.tensor(
                [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 4.0]]],
                dtype=torch.float64,
            ),
            torch.tensor([[True, True, False]]),
            0.5,
            torch.tensor([0, 1]),
        )


def test_cpu_rejects_nonfinite_inactive_cell_row() -> None:
    cell = torch.eye(3, dtype=torch.float64)
    cell[2, 0] = torch.nan
    with pytest.raises(ValueError, match="cell must contain only finite values"):
        neighbor_list(
            "PS",
            torch.zeros((1, 3), dtype=torch.float64),
            cell[None],
            torch.tensor([[True, True, False]]),
            0.5,
            torch.tensor([0, 1]),
        )


def test_cpu_accepts_ill_conditioned_full_rank_active_rows() -> None:
    cell = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 1e-08, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
    )
    arguments = (
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 5e-10, 0.0]], dtype=torch.float64),
        cell,
        torch.tensor([True, True, False]),
        1e-09,
    )
    assert pair_keys(*neighbor_list("PS", *arguments)) == pair_keys(
        *neighbor_list_reference("PS", *arguments)
    )


def test_cpu_large_unwrapped_representatives_use_original_geometry() -> None:
    cell = torch.tensor(
        [
            [-0.7647425305009836, -2.672179555955561, 2.8765648407042885],
            [-1.0423290504477831, -3.665756214439444, -3.2514359787147056],
            [5.609321251949969, -2.2087783511424566, -0.0013679155882546645],
        ],
        dtype=torch.float64,
    )
    positions = torch.tensor(
        [
            [598110346.0828383, -8096602.288435191, 242913231.3592053],
            [598110345.1706604, -8096602.339372153, 242913230.9525894],
        ],
        dtype=torch.float64,
    )
    pair_indices, shifts = neighbor_list(
        "PS",
        positions,
        cell[None],
        torch.ones((1, 3), dtype=torch.bool),
        1.0,
        torch.tensor([0, 2]),
    )
    assert pair_keys(pair_indices, shifts) == {(0, 1, 0, 0, 0), (1, 0, 0, 0, 0)}


def test_cpu_cell_list_large_common_lattice_translation_matches_reference() -> None:
    generator = torch.Generator().manual_seed(9184)
    cell = torch.tensor(
        [[4.1, 0.3, 0.2], [0.4, 4.7, 0.1], [0.2, 0.5, 5.3]], dtype=torch.float64
    )
    positions = torch.rand((129, 3), generator=generator, dtype=torch.float64) @ cell
    positions += (
        torch.tensor([100000000, -80000000, 60000000], dtype=torch.float64) @ cell
    )
    arguments = (
        positions,
        cell,
        torch.ones(3, dtype=torch.bool),
        1.0,
    )
    assert pair_keys(*neighbor_list("PS", *arguments)) == pair_keys(
        *neighbor_list_reference("PS", *arguments)
    )


def test_cpu_rejects_pathological_periodic_image_count() -> None:
    with pytest.raises(ValueError, match="image count.*resource limit"):
        neighbor_list(
            "PS",
            torch.zeros((1, 3), dtype=torch.float64),
            (0.001 * torch.eye(3, dtype=torch.float64))[None],
            torch.ones((1, 3), dtype=torch.bool),
            1.0,
            torch.tensor([0, 1]),
        )
