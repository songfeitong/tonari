from __future__ import annotations

import numpy as np
import pytest
import torch
from ase.neighborlist import primitive_neighbor_list

from torch_radius_graph import radius_graph_pbc, reference_radius_graph_pbc


def edge_keys(
    edge_index: torch.Tensor, shifts: torch.Tensor
) -> set[tuple[int, ...]]:
    rows = torch.cat((edge_index.T, shifts.to(torch.int64)), dim=1).tolist()
    assert len(rows) == len({tuple(row) for row in rows})
    return {tuple(row) for row in rows}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_matches_reference_for_mixed_batch(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(712)
    finite = torch.rand((7, 3), generator=generator, dtype=dtype) * 4.0
    partial_cell = torch.tensor(
        [[2.1, 0.2, 0.1], [0.0, 2.5, 0.3], [0.0, 0.0, 0.0]],
        dtype=dtype,
    )
    periodic_cell = torch.tensor(
        [[1.7, 0.2, 0.0], [0.1, 1.9, 0.3], [0.2, 0.1, 2.2]],
        dtype=dtype,
    )
    positions = torch.cat(
        (
            finite,
            torch.rand((5, 3), generator=generator, dtype=dtype)
            @ torch.tensor(
                [[2.1, 0.2, 0.1], [0.0, 2.5, 0.3], [0.0, 0.0, 8.0]],
                dtype=dtype,
            ),
            torch.rand((4, 3), generator=generator, dtype=dtype) @ periodic_cell,
        )
    )
    ptr = torch.tensor([0, 7, 12, 16])
    cells = torch.stack((torch.zeros((3, 3), dtype=dtype), partial_cell, periodic_cell))
    pbc = torch.tensor(
        [[False, False, False], [True, True, False], [True, True, True]]
    )
    expected = reference_radius_graph_pbc(positions, ptr, cells, pbc, 1.35)
    actual = radius_graph_pbc(positions, ptr, cells, pbc, 1.35)
    assert edge_keys(*actual) == edge_keys(*expected)


def test_cpu_matches_ase_for_partial_triclinic_multiple_images() -> None:
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [0.9, 0.4, 0.8], [1.6, 1.0, 0.5]],
        dtype=torch.float64,
    )
    cell = torch.tensor(
        [[1.7, 0.0, 0.0], [0.45, 1.6, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
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
        (int(j), int(i), int(shift[0]), int(shift[1]), int(shift[2]))
        for i, j, shift in zip(first, second, shifts, strict=True)
    }
    actual = radius_graph_pbc(
        positions,
        torch.tensor([0, len(positions)]),
        cell[None],
        pbc[None],
        cutoff,
    )
    assert edge_keys(*actual) == expected


def test_cpu_strict_cutoff_and_periodic_self_images() -> None:
    common = (
        torch.tensor([0, 2]),
        torch.zeros((1, 3, 3)),
        torch.zeros((1, 3), dtype=torch.bool),
        1.0,
    )
    boundary = radius_graph_pbc(
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), *common
    )
    assert edge_keys(*boundary) == set()

    just_inside = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))
    inside = radius_graph_pbc(
        torch.tensor([[0.0, 0.0, 0.0], [just_inside, 0.0, 0.0]]), *common
    )
    assert edge_keys(*inside) == {
        (0, 1, 0, 0, 0),
        (1, 0, 0, 0, 0),
    }

    periodic = radius_graph_pbc(
        torch.zeros((1, 3)),
        torch.tensor([0, 1]),
        torch.diag(torch.tensor([0.4, 8.0, 8.0]))[None],
        torch.tensor([[True, False, False]]),
        1.0,
    )
    assert edge_keys(*periodic) == {
        (0, 0, -2, 0, 0),
        (0, 0, -1, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 2, 0, 0),
    }


def test_cpu_relabels_unwrapped_representatives() -> None:
    cell = torch.tensor(
        [[2.0, 0.1, 0.0], [0.2, 2.5, 0.0], [0.1, 0.3, 3.0]],
        dtype=torch.float64,
    )
    positions = torch.tensor(
        [[0.2, 0.1, 0.4], [1.8, 0.2, 0.5]], dtype=torch.float64
    )
    translated = positions.clone()
    translated[0] -= 3 * cell[1]
    translated[1] += 2 * cell[0] - cell[2]
    common = (
        torch.tensor([0, 2]),
        cell[None],
        torch.ones((1, 3), dtype=torch.bool),
        1.0,
    )
    first_edges, first_shifts = radius_graph_pbc(positions, *common)
    second_edges, second_shifts = radius_graph_pbc(translated, *common)
    first_vectors = (
        positions[first_edges[0]]
        - positions[first_edges[1]]
        + first_shifts.to(positions.dtype) @ cell
    )
    second_vectors = (
        translated[second_edges[0]]
        - translated[second_edges[1]]
        + second_shifts.to(positions.dtype) @ cell
    )
    assert edge_keys(first_edges, first_shifts) != edge_keys(
        second_edges, second_shifts
    )
    assert sorted(map(tuple, np.round(first_vectors.numpy(), 12))) == sorted(
        map(tuple, np.round(second_vectors.numpy(), 12))
    )


def test_cpu_rotation_and_reflection_covariance() -> None:
    cell = torch.tensor(
        [[2.0, 0.3, 0.1], [0.2, 2.4, 0.4], [0.1, 0.2, 2.8]],
        dtype=torch.float64,
    )
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [1.7, 0.4, 0.5], [0.6, 1.8, 1.1]],
        dtype=torch.float64,
    )
    orthogonal = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=torch.float64,
    )
    common = (
        torch.tensor([0, len(positions)]),
        torch.ones((1, 3), dtype=torch.bool),
        1.3,
    )
    first = radius_graph_pbc(positions, common[0], cell[None], *common[1:])
    second = radius_graph_pbc(
        positions @ orthogonal,
        common[0],
        (cell @ orthogonal)[None],
        *common[1:],
    )
    assert edge_keys(*first) == edge_keys(*second)


def test_cpu_randomized_differential() -> None:
    generator = torch.Generator().manual_seed(29_381)
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
        cell += 0.35 * torch.tril(
            torch.rand((3, 3), generator=generator), diagonal=-1
        )
        positions = torch.rand((n_atoms, 3), generator=generator) @ cell
        pbc = pbc_patterns[case % len(pbc_patterns)]
        if case % 4 == 0:
            positions[0] += 3 * cell[0]
            if pbc[2]:
                positions[-1] -= 2 * cell[2]
        cutoff = 0.55 + 0.8 * float(torch.rand((), generator=generator))
        arguments = (
            positions,
            torch.tensor([0, n_atoms]),
            cell[None],
            pbc[None],
            cutoff,
        )
        assert edge_keys(*radius_graph_pbc(*arguments)) == edge_keys(
            *reference_radius_graph_pbc(*arguments)
        )


def test_cpu_allows_continuous_geometry_backward() -> None:
    positions = torch.tensor(
        [[0.1, 0.0, 0.0], [1.8, 0.2, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    cells = torch.tensor(
        [[[2.0, 0.0, 0.0], [0.1, 2.5, 0.0], [0.0, 0.0, 3.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    edge_index, shifts = radius_graph_pbc(
        positions,
        torch.tensor([0, 2]),
        cells,
        torch.ones((1, 3), dtype=torch.bool),
        0.8,
    )
    vectors = (
        positions[edge_index[0]]
        - positions[edge_index[1]]
        + shifts.to(positions.dtype) @ cells[0]
    )
    torch.sum(vectors.square()).backward()
    assert edge_index.grad_fn is None
    assert shifts.grad_fn is None
    assert positions.grad is not None and torch.all(torch.isfinite(positions.grad))
    assert cells.grad is not None and torch.all(torch.isfinite(cells.grad))
    assert torch.count_nonzero(cells.grad) > 0


def test_cpu_empty_periodic_structure_skips_tiny_cell_images() -> None:
    edge_index, shifts = radius_graph_pbc(
        torch.empty((0, 3), dtype=torch.float64),
        torch.tensor([0, 0]),
        (0.02 * torch.eye(3, dtype=torch.float64))[None],
        torch.ones((1, 3), dtype=torch.bool),
        1.0,
    )
    assert edge_index.shape == (2, 0)
    assert shifts.shape == (0, 3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_cell_list_path_matches_reference(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(1842)
    n_atoms = 288
    cell = torch.tensor(
        [[8.0, 0.4, 0.1], [0.2, 7.5, 0.5], [0.3, 0.1, 9.0]], dtype=dtype
    )
    positions = torch.rand((n_atoms, 3), generator=generator, dtype=dtype) @ cell
    positions[:5] += 3 * cell[0] - 2 * cell[1]
    ptr = torch.tensor([0, n_atoms])
    pbc = torch.ones((1, 3), dtype=torch.bool)
    expected = reference_radius_graph_pbc(positions, ptr, cell[None], pbc, 1.2)
    actual = radius_graph_pbc(positions, ptr, cell[None], pbc, 1.2)
    assert edge_keys(*actual) == edge_keys(*expected)


def test_cpu_cell_list_falls_back_for_extremely_sparse_bounds() -> None:
    positions = torch.zeros((256, 3))
    positions[:, 0] = torch.arange(256) * 10_000.0
    edge_index, shifts = radius_graph_pbc(
        positions,
        torch.tensor([0, 256]),
        torch.zeros((1, 3, 3)),
        torch.zeros((1, 3), dtype=torch.bool),
        1.0,
    )
    assert edge_keys(edge_index, shifts) == set()


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_cpu_rejects_nonfinite_positions(value: float) -> None:
    positions = torch.zeros((2, 3), dtype=torch.float64)
    positions[0, 0] = value
    with pytest.raises(RuntimeError, match="positions must"):
        radius_graph_pbc(
            positions,
            torch.tensor([0, 2]),
            torch.zeros((1, 3, 3), dtype=torch.float64),
            torch.zeros((1, 3), dtype=torch.bool),
            0.5,
        )


def test_cpu_rejects_representative_wrap_outside_int32_range() -> None:
    positions = torch.tensor(
        [[0.1, 0.0, 0.0], [2**31 + 0.2, 0.0, 0.0]], dtype=torch.float64
    )
    with pytest.raises(RuntimeError, match="wraps.*int32"):
        radius_graph_pbc(
            positions,
            torch.tensor([0, 2]),
            torch.eye(3, dtype=torch.float64)[None],
            torch.tensor([[True, False, False]]),
            0.5,
        )


def test_cpu_rejects_output_shift_outside_int32_range() -> None:
    positions = torch.tensor(
        [[-(2**31) + 0.1, 0.0, 0.0], [2**31 - 0.8, 0.0, 0.0]],
        dtype=torch.float64,
    )
    with pytest.raises(RuntimeError, match="cell shift.*int32"):
        radius_graph_pbc(
            positions,
            torch.tensor([0, 2]),
            torch.eye(3, dtype=torch.float64)[None],
            torch.tensor([[True, False, False]]),
            0.5,
        )


def test_cpu_rejects_dependent_active_cell_rows() -> None:
    with pytest.raises(ValueError, match="linearly independent"):
        radius_graph_pbc(
            torch.zeros((1, 3), dtype=torch.float64),
            torch.tensor([0, 1]),
            torch.tensor(
                [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 4.0]]],
                dtype=torch.float64,
            ),
            torch.tensor([[True, True, False]]),
            0.5,
        )


def test_cpu_rejects_nonfinite_inactive_cell_row() -> None:
    cell = torch.eye(3, dtype=torch.float64)
    cell[2, 0] = torch.nan
    with pytest.raises(ValueError, match="cells must contain only finite values"):
        radius_graph_pbc(
            torch.zeros((1, 3), dtype=torch.float64),
            torch.tensor([0, 1]),
            cell[None],
            torch.tensor([[True, True, False]]),
            0.5,
        )
