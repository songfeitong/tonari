from __future__ import annotations

import numpy as np
import pytest
import torch
from ase.neighborlist import primitive_neighbor_list

from torch_radius_graph import radius_graph_pbc, reference_radius_graph_pbc

pytestmark = pytest.mark.cuda


def edge_keys(edge_index: torch.Tensor, shifts: torch.Tensor) -> set[tuple[int, ...]]:
    rows = torch.cat((edge_index.T, shifts.to(torch.int64)), dim=1).cpu().tolist()
    assert len(rows) == len({tuple(row) for row in rows})
    return {tuple(row) for row in rows}


def cuda_graph(
    positions: torch.Tensor,
    ptr: torch.Tensor,
    cells: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return radius_graph_pbc(
        positions.cuda(),
        ptr.cuda(),
        cells.cuda(),
        pbc.cuda(),
        cutoff,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_matches_reference_for_mixed_boundary_conditions(
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator().manual_seed(712)
    positions = torch.cat(
        (
            torch.rand((7, 3), generator=generator, dtype=dtype) * 4.0,
            torch.rand((5, 3), generator=generator, dtype=dtype)
            @ torch.tensor(
                [[2.1, 0.2, 0.1], [0.0, 2.5, 0.3], [0.0, 0.0, 8.0]],
                dtype=dtype,
            ),
            torch.rand((4, 3), generator=generator, dtype=dtype)
            @ torch.tensor(
                [[1.7, 0.2, 0.0], [0.1, 1.9, 0.3], [0.2, 0.1, 2.2]],
                dtype=dtype,
            ),
        )
    )
    ptr = torch.tensor([0, 7, 12, 16])
    cells = torch.stack(
        (
            torch.zeros((3, 3), dtype=dtype),
            torch.tensor(
                [[2.1, 0.2, 0.1], [0.0, 2.5, 0.3], [0.0, 0.0, 0.0]],
                dtype=dtype,
            ),
            torch.tensor(
                [[1.7, 0.2, 0.0], [0.1, 1.9, 0.3], [0.2, 0.1, 2.2]],
                dtype=dtype,
            ),
        )
    )
    pbc = torch.tensor([[False, False, False], [True, True, False], [True, True, True]])
    expected = reference_radius_graph_pbc(positions, ptr, cells, pbc, 1.35)
    actual = cuda_graph(positions, ptr, cells, pbc, 1.35)
    assert edge_keys(*actual) == edge_keys(*expected)


def test_cuda_matches_ase_for_partial_triclinic_multiple_images() -> None:
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
    actual = cuda_graph(
        positions,
        torch.tensor([0, len(positions)]),
        cell[None],
        pbc[None],
        cutoff,
    )
    assert edge_keys(*actual) == expected
    assert any(
        source == target and shift != (0, 0, 0) for source, target, *shift in expected
    )


def test_cuda_relabels_unwrapped_representatives() -> None:
    dtype = torch.float64
    cell = torch.tensor(
        [[2.0, 0.1, 0.0], [0.2, 2.5, 0.0], [0.1, 0.3, 3.0]], dtype=dtype
    )
    positions = torch.tensor([[0.2, 0.1, 0.4], [1.8, 0.2, 0.5]], dtype=dtype)
    translated = positions.clone()
    translated[0] -= 3 * cell[1]
    translated[1] += 2 * cell[0] - cell[2]
    ptr = torch.tensor([0, 2])
    pbc = torch.ones((1, 3), dtype=torch.bool)
    first_edges, first_shifts = cuda_graph(positions, ptr, cell[None], pbc, 1.0)
    second_edges, second_shifts = cuda_graph(translated, ptr, cell[None], pbc, 1.0)
    first_vectors = (
        positions.cuda()[first_edges[0]]
        - positions.cuda()[first_edges[1]]
        + first_shifts.to(dtype) @ cell.cuda()
    )
    second_vectors = (
        translated.cuda()[second_edges[0]]
        - translated.cuda()[second_edges[1]]
        + second_shifts.to(dtype) @ cell.cuda()
    )
    assert edge_keys(first_edges, first_shifts) != edge_keys(
        second_edges, second_shifts
    )
    first_vectors = sorted(map(tuple, np.round(first_vectors.cpu().numpy(), 12)))
    second_vectors = sorted(map(tuple, np.round(second_vectors.cpu().numpy(), 12)))
    assert first_vectors == second_vectors


def test_cuda_strict_cutoff_and_periodic_self_images() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    finite_edges, finite_shifts = cuda_graph(
        positions,
        torch.tensor([0, 2]),
        torch.zeros((1, 3, 3)),
        torch.zeros((1, 3), dtype=torch.bool),
        1.0,
    )
    assert edge_keys(finite_edges, finite_shifts) == set()

    periodic_edges, periodic_shifts = cuda_graph(
        positions[:1],
        torch.tensor([0, 1]),
        torch.diag(torch.tensor([0.4, 8.0, 8.0]))[None],
        torch.tensor([[True, False, False]]),
        1.0,
    )
    assert edge_keys(periodic_edges, periodic_shifts) == {
        (0, 0, -2, 0, 0),
        (0, 0, -1, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 2, 0, 0),
    }


def test_graph_allows_continuous_geometry_backward() -> None:
    positions = torch.tensor(
        [[0.1, 0.0, 0.0], [1.8, 0.2, 0.0]],
        device="cuda",
        dtype=torch.float64,
        requires_grad=True,
    )
    cells = torch.tensor(
        [[[2.0, 0.0, 0.0], [0.1, 2.5, 0.0], [0.0, 0.0, 3.0]]],
        device="cuda",
        dtype=torch.float64,
        requires_grad=True,
    )
    ptr = torch.tensor([0, 2], device="cuda")
    pbc = torch.ones((1, 3), device="cuda", dtype=torch.bool)
    edge_index, shifts = radius_graph_pbc(positions, ptr, cells, pbc, 0.8)
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


def test_nondefault_stream_and_empty_structure() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], device="cuda")
    ptr = torch.tensor([0, 0, 2], device="cuda")
    cells = torch.zeros((2, 3, 3), device="cuda")
    pbc = torch.zeros((2, 3), dtype=torch.bool, device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        positions = positions + 0.0
        edge_index, shifts = radius_graph_pbc(positions, ptr, cells, pbc, 0.5)
    torch.cuda.current_stream().wait_stream(stream)
    assert edge_keys(edge_index, shifts) == {
        (0, 1, 0, 0, 0),
        (1, 0, 0, 0, 0),
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cell_list_path_matches_reference(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(1842)
    n_atoms = 288
    cell = torch.tensor(
        [[8.0, 0.4, 0.1], [0.2, 7.5, 0.5], [0.3, 0.1, 9.0]], dtype=dtype
    )
    positions = torch.rand((n_atoms, 3), generator=generator, dtype=dtype) @ cell
    positions[:5] += 3 * cell[0] - 2 * cell[1]
    ptr = torch.tensor([0, n_atoms])
    pbc = torch.tensor([[True, True, True]])
    expected = reference_radius_graph_pbc(positions, ptr, cell[None], pbc, 1.2)
    actual = cuda_graph(positions, ptr, cell[None], pbc, 1.2)
    assert edge_keys(*actual) == edge_keys(*expected)


def test_cell_list_falls_back_for_extremely_sparse_bounds() -> None:
    positions = torch.zeros((256, 3))
    positions[:, 0] = torch.arange(256) * 10_000.0
    edge_index, shifts = cuda_graph(
        positions,
        torch.tensor([0, 256]),
        torch.zeros((1, 3, 3)),
        torch.zeros((1, 3), dtype=torch.bool),
        1.0,
    )
    assert edge_keys(edge_index, shifts) == set()
