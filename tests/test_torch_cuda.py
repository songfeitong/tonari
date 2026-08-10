from __future__ import annotations

import numpy as np
import pytest
import torch
from ase.neighborlist import primitive_neighbor_list

from tests.reference import find_neighbors_reference
from tonari import find_neighbors

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def pair_keys(pair_indices: torch.Tensor, shifts: torch.Tensor) -> set[tuple[int, ...]]:
    rows = torch.cat((pair_indices.T, shifts.to(torch.int64)), dim=1).cpu().tolist()
    assert len(rows) == len({tuple(row) for row in rows})
    return {tuple(row) for row in rows}


def cuda_neighbors(
    positions: torch.Tensor,
    cells: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    batch_ptr: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return find_neighbors(
        positions.cuda(),
        cells.cuda(),
        pbc.cuda(),
        cutoff,
        None if batch_ptr is None else batch_ptr.cuda(),
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
                [[2.1, 0.2, 0.1], [0.0, 2.5, 0.3], [0.0, 0.0, 8.0]], dtype=dtype
            ),
            torch.rand((4, 3), generator=generator, dtype=dtype)
            @ torch.tensor(
                [[1.7, 0.2, 0.0], [0.1, 1.9, 0.3], [0.2, 0.1, 2.2]], dtype=dtype
            ),
        )
    )
    batch_ptr = torch.tensor([0, 7, 12, 16])
    cells = torch.stack(
        (
            torch.zeros((3, 3), dtype=dtype),
            torch.tensor(
                [[2.1, 0.2, 0.1], [0.0, 2.5, 0.3], [0.0, 0.0, 0.0]], dtype=dtype
            ),
            torch.tensor(
                [[1.7, 0.2, 0.0], [0.1, 1.9, 0.3], [0.2, 0.1, 2.2]], dtype=dtype
            ),
        )
    )
    pbc = torch.tensor([[False, False, False], [True, True, False], [True, True, True]])
    expected = find_neighbors_reference(positions, cells, pbc, 1.35, batch_ptr)
    actual = cuda_neighbors(positions, cells, pbc, 1.35, batch_ptr)
    assert pair_keys(*actual) == pair_keys(*expected)


def test_cuda_matches_ase_for_partial_triclinic_multiple_images() -> None:
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
    actual = cuda_neighbors(
        positions, cell[None], pbc[None], cutoff, torch.tensor([0, len(positions)])
    )
    assert pair_keys(*actual) == expected
    assert any(
        (
            source == target and tuple(shift) != (0, 0, 0)
            for source, target, *shift in expected
        )
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
    batch_ptr = torch.tensor([0, 2])
    pbc = torch.ones((1, 3), dtype=torch.bool)
    first_pairs, first_shifts = cuda_neighbors(
        positions, cell[None], pbc, 1.0, batch_ptr
    )
    second_pairs, second_shifts = cuda_neighbors(
        translated, cell[None], pbc, 1.0, batch_ptr
    )
    first_displacements = (
        positions.cuda()[first_pairs[1]]
        - positions.cuda()[first_pairs[0]]
        + first_shifts.to(dtype) @ cell.cuda()
    )
    second_displacements = (
        translated.cuda()[second_pairs[1]]
        - translated.cuda()[second_pairs[0]]
        + second_shifts.to(dtype) @ cell.cuda()
    )
    assert pair_keys(first_pairs, first_shifts) != pair_keys(
        second_pairs, second_shifts
    )
    first_displacements = sorted(
        map(tuple, np.round(first_displacements.cpu().numpy(), 12))
    )
    second_displacements = sorted(
        map(tuple, np.round(second_displacements.cpu().numpy(), 12))
    )
    assert first_displacements == second_displacements


def test_cuda_strict_cutoff_and_periodic_self_images() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    finite_pairs, finite_shifts = cuda_neighbors(
        positions,
        torch.zeros((1, 3, 3)),
        torch.zeros((1, 3), dtype=torch.bool),
        1.0,
        torch.tensor([0, 2]),
    )
    assert pair_keys(finite_pairs, finite_shifts) == set()
    just_inside = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))
    inside_pairs, inside_shifts = cuda_neighbors(
        torch.tensor([[0.0, 0.0, 0.0], [just_inside, 0.0, 0.0]]),
        torch.zeros((1, 3, 3)),
        torch.zeros((1, 3), dtype=torch.bool),
        1.0,
        torch.tensor([0, 2]),
    )
    assert pair_keys(inside_pairs, inside_shifts) == {(0, 1, 0, 0, 0), (1, 0, 0, 0, 0)}
    periodic_pairs, periodic_shifts = cuda_neighbors(
        positions[:1],
        torch.diag(torch.tensor([0.4, 8.0, 8.0]))[None],
        torch.tensor([[True, False, False]]),
        1.0,
        torch.tensor([0, 1]),
    )
    assert pair_keys(periodic_pairs, periodic_shifts) == {
        (0, 0, -2, 0, 0),
        (0, 0, -1, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 2, 0, 0),
    }


@pytest.mark.parametrize("n_atoms", [255, 256])
def test_cuda_paths_use_the_same_float32_cutoff_rounding(n_atoms: int) -> None:
    cutoff = 353.2019167901003
    positions = torch.zeros((n_atoms, 3), dtype=torch.float32)
    positions[1, 0] = torch.tensor(353.2019, dtype=torch.float32)
    positions[2:, 1] = 1000 * torch.arange(2, n_atoms, dtype=torch.float32)
    cell = torch.zeros((3, 3), dtype=torch.float32)
    pbc = torch.zeros(3, dtype=torch.bool)
    expected = {(0, 1, 0, 0, 0), (1, 0, 0, 0, 0)}

    cpu = find_neighbors(positions, cell, pbc, cutoff)
    cuda = find_neighbors(positions.cuda(), cell.cuda(), pbc.cuda(), cutoff)
    reference = find_neighbors_reference(
        positions,
        cell[None],
        pbc[None],
        cutoff,
        torch.tensor([0, n_atoms]),
    )

    assert pair_keys(*cpu) == expected
    assert pair_keys(*cuda) == expected
    assert pair_keys(*reference) == expected


def test_cuda_allows_continuous_geometry_backward() -> None:
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
    batch_ptr = torch.tensor([0, 2], device="cuda")
    pbc = torch.ones((1, 3), device="cuda", dtype=torch.bool)
    pair_indices, shifts = find_neighbors(positions, cells, pbc, 0.8, batch_ptr)
    displacements = (
        positions[pair_indices[1]]
        - positions[pair_indices[0]]
        + shifts.to(positions.dtype) @ cells[0]
    )
    torch.sum(displacements.square()).backward()
    assert pair_indices.grad_fn is None
    assert shifts.grad_fn is None
    assert positions.grad is not None and torch.all(torch.isfinite(positions.grad))
    assert cells.grad is not None and torch.all(torch.isfinite(cells.grad))
    assert torch.count_nonzero(cells.grad) > 0


def test_nondefault_stream_and_empty_structure() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], device="cuda")
    batch_ptr = torch.tensor([0, 0, 2], device="cuda")
    cells = torch.zeros((2, 3, 3), device="cuda")
    pbc = torch.zeros((2, 3), dtype=torch.bool, device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        positions = positions + 0.0
        pair_indices, shifts = find_neighbors(positions, cells, pbc, 0.5, batch_ptr)
    torch.cuda.current_stream().wait_stream(stream)
    assert pair_keys(pair_indices, shifts) == {(0, 1, 0, 0, 0), (1, 0, 0, 0, 0)}


def test_empty_periodic_structure_does_not_enumerate_tiny_cell_images() -> None:
    pair_indices, shifts = cuda_neighbors(
        torch.empty((0, 3), dtype=torch.float64),
        (1e-12 * torch.eye(3, dtype=torch.float64))[None],
        torch.ones((1, 3), dtype=torch.bool),
        1.0,
        torch.tensor([0, 0]),
    )
    assert pair_indices.shape == (2, 0)
    assert shifts.shape == (0, 3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cell_list_path_matches_reference(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(1842)
    n_atoms = 288
    cell = torch.tensor(
        [[8.0, 0.4, 0.1], [0.2, 7.5, 0.5], [0.3, 0.1, 9.0]], dtype=dtype
    )
    positions = torch.rand((n_atoms, 3), generator=generator, dtype=dtype) @ cell
    positions[:5] += 3 * cell[0] - 2 * cell[1]
    batch_ptr = torch.tensor([0, n_atoms])
    pbc = torch.tensor([[True, True, True]])
    expected = find_neighbors_reference(positions, cell[None], pbc, 1.2, batch_ptr)
    actual = cuda_neighbors(positions, cell[None], pbc, 1.2, batch_ptr)
    assert pair_keys(*actual) == pair_keys(*expected)


def test_cell_list_path_handles_mixed_finite_and_partial_pbc_batch() -> None:
    generator = torch.Generator().manual_seed(5801)
    counts = (256, 257)
    cells = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[2.1, 0.3, 0.2], [0.4, 4.0, 0.1], [0.2, 0.5, 5.0]],
        ],
        dtype=torch.float64,
    )
    positions = torch.cat(
        (
            torch.rand((counts[0], 3), generator=generator, dtype=torch.float64) * 8,
            torch.rand((counts[1], 3), generator=generator, dtype=torch.float64)
            @ cells[1],
        )
    )
    positions[-3:] += 4 * cells[1, 0]
    batch_ptr = torch.tensor([0, counts[0], sum(counts)])
    pbc = torch.tensor([[False, False, False], [True, False, False]])
    expected = find_neighbors_reference(positions, cells, pbc, 0.55, batch_ptr)
    actual = cuda_neighbors(positions, cells, pbc, 0.55, batch_ptr)
    assert pair_keys(*actual) == pair_keys(*expected)
    pair_indices, shifts = actual
    source_batch = torch.bucketize(pair_indices[0], batch_ptr[1:].cuda(), right=True)
    target_batch = torch.bucketize(pair_indices[1], batch_ptr[1:].cuda(), right=True)
    assert torch.equal(source_batch, target_batch)
    displacements = (
        positions.cuda()[pair_indices[1]]
        - positions.cuda()[pair_indices[0]]
        + torch.einsum(
            "ei,eij->ej", shifts.to(torch.float64), cells.cuda()[source_batch]
        )
    )
    assert torch.all(torch.sum(displacements.square(), dim=1) < 0.55**2)
    assert torch.all(shifts[source_batch == 1, 1:] == 0)


def test_cell_list_large_common_translation_uses_public_vector_formula() -> None:
    cell = torch.tensor(
        [
            [-0.9499396681785583, 2.3687520027160645, 1.7597240209579468],
            [-3.4348089694976807, -1.8801195621490479, 1.9097744226455688],
            [2.5219004154205322, -1.9322474002838135, 4.843184471130371],
        ]
    )
    positions = torch.tensor([-2138.380859375, -11887.5810546875, 10318.5625]).repeat(
        256, 1
    )
    positions[52] = torch.tensor(
        [-2135.800048828125, -11886.6826171875, 10317.2880859375]
    )
    positions[40] = torch.tensor(
        [-2141.306396484375, -11885.267578125, 10315.8291015625]
    )
    positions[61] = torch.tensor([-2135.84033203125, -11887.59375, 10317.1181640625])
    arguments = (
        positions,
        cell,
        torch.ones(3, dtype=torch.bool),
        1.4481067657470703,
    )
    expected = find_neighbors(*arguments)
    actual = cuda_neighbors(*arguments)
    expected_keys = pair_keys(*expected)
    assert (52, 23, 0, -1, 0) in expected_keys
    assert (23, 52, 0, 1, 0) in expected_keys
    assert (61, 40, -1, -1, 1) not in expected_keys
    assert (40, 61, 1, 1, -1) not in expected_keys
    assert pair_keys(*actual) == expected_keys


def test_rejects_dependent_active_cell_rows() -> None:
    with pytest.raises(ValueError, match="linearly independent"):
        cuda_neighbors(
            torch.zeros((1, 3), dtype=torch.float64),
            torch.tensor(
                [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                dtype=torch.float64,
            ),
            torch.tensor([[True, True, False]]),
            1.0,
            torch.tensor([0, 1]),
        )


@pytest.mark.parametrize("n_atoms", [2, 256])
def test_rejects_representative_wrap_outside_int32_range(n_atoms: int) -> None:
    positions = torch.zeros((n_atoms, 3), dtype=torch.float64)
    positions[0, 0] = 0.1
    positions[1, 0] = 2**31 + 0.2
    if n_atoms > 2:
        positions[2:, 1] = 2 * torch.arange(2, n_atoms, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="wraps.*int32"):
        cuda_neighbors(
            positions,
            torch.eye(3, dtype=torch.float64)[None],
            torch.tensor([[True, False, False]]),
            0.5,
            torch.tensor([0, n_atoms]),
        )


@pytest.mark.parametrize("n_atoms", [2, 256])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_rejects_nonfinite_positions(n_atoms: int, value: float) -> None:
    positions = torch.zeros((n_atoms, 3), dtype=torch.float64)
    positions[0, 0] = value
    if n_atoms > 2:
        positions[:, 1] = 2 * torch.arange(n_atoms, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="positions must"):
        cuda_neighbors(
            positions,
            torch.zeros((1, 3, 3), dtype=torch.float64),
            torch.zeros((1, 3), dtype=torch.bool),
            0.5,
            torch.tensor([0, n_atoms]),
        )


def test_rejects_nonfinite_inactive_cell_row() -> None:
    cell = torch.eye(3, dtype=torch.float64)
    cell[2, 0] = torch.nan
    with pytest.raises(ValueError, match="cells must contain only finite values"):
        cuda_neighbors(
            torch.zeros((1, 3), dtype=torch.float64),
            cell[None],
            torch.tensor([[True, True, False]]),
            0.5,
            torch.tensor([0, 1]),
        )


def test_cell_list_falls_back_for_extremely_sparse_bounds() -> None:
    positions = torch.zeros((256, 3))
    positions[:, 0] = torch.arange(256) * 10000.0
    pair_indices, shifts = cuda_neighbors(
        positions,
        torch.zeros((1, 3, 3)),
        torch.zeros((1, 3), dtype=torch.bool),
        1.0,
        torch.tensor([0, 256]),
    )
    assert pair_keys(pair_indices, shifts) == set()


def test_batched_sparse_bin_counts_saturate_before_cumsum() -> None:
    n_atoms = 256
    structure = torch.full((n_atoms, 3), 1700000.0, dtype=torch.float64)
    structure[0] = 0.0
    positions = torch.cat((structure, structure))
    arguments = (
        positions,
        torch.zeros((2, 3, 3), dtype=torch.float64),
        torch.zeros((2, 3), dtype=torch.bool),
        1.0,
        torch.tensor([0, n_atoms, 2 * n_atoms]),
    )
    expected = find_neighbors(*arguments)
    actual = cuda_neighbors(*arguments)
    assert pair_keys(*actual) == pair_keys(*expected)
