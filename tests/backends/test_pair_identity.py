from __future__ import annotations

import numpy as np
import pytest
import torch
from ase.neighborlist import primitive_neighbor_list

from tests.support.assertions import pair_keys
from tests.support.reference import neighbor_list_reference
from tonari import neighbor_list


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mixed_boundary_batch_matches_reference(
    torch_device: torch.device, dtype: torch.dtype
) -> None:
    generator = torch.Generator().manual_seed(712)
    partial_cell = torch.tensor(
        [[2.1, 0.2, 0.1], [0.0, 2.5, 0.3], [0.0, 0.0, 0.0]], dtype=dtype
    )
    periodic_cell = torch.tensor(
        [[1.7, 0.2, 0.0], [0.1, 1.9, 0.3], [0.2, 0.1, 2.2]], dtype=dtype
    )
    positions = torch.cat(
        (
            torch.rand((7, 3), generator=generator, dtype=dtype) * 4.0,
            torch.rand((5, 3), generator=generator, dtype=dtype)
            @ torch.tensor(
                [[2.1, 0.2, 0.1], [0.0, 2.5, 0.3], [0.0, 0.0, 8.0]],
                dtype=dtype,
            ),
            torch.rand((4, 3), generator=generator, dtype=dtype) @ periodic_cell,
        )
    )
    batch_ptr = torch.tensor([0, 7, 12, 16])
    cell = torch.stack((torch.zeros((3, 3), dtype=dtype), partial_cell, periodic_cell))
    pbc = torch.tensor([[False, False, False], [True, True, False], [True, True, True]])
    expected = neighbor_list_reference(positions, cell, pbc, 1.35, batch_ptr)
    actual = neighbor_list(
        "PS",
        positions.to(torch_device),
        cell.to(torch_device),
        pbc.to(torch_device),
        1.35,
        batch_ptr.to(torch_device),
    )
    assert pair_keys(*actual) == pair_keys(*expected)


def test_partial_triclinic_multiple_images_match_ase(
    torch_device: torch.device,
) -> None:
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
        positions.to(torch_device),
        cell.to(torch_device),
        pbc.to(torch_device),
        cutoff,
    )
    assert pair_keys(*actual) == expected


def test_strict_cutoff_and_periodic_self_images(torch_device: torch.device) -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cell = torch.zeros((3, 3))
    pbc = torch.zeros(3, dtype=torch.bool)
    boundary = neighbor_list(
        "PS",
        positions.to(torch_device),
        cell.to(torch_device),
        pbc.to(torch_device),
        1.0,
    )
    assert pair_keys(*boundary) == set()
    just_inside = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))
    positions[1, 0] = just_inside
    inside = neighbor_list(
        "PS",
        positions.to(torch_device),
        cell.to(torch_device),
        pbc.to(torch_device),
        1.0,
    )
    assert pair_keys(*inside) == {(0, 1, 0, 0, 0), (1, 0, 0, 0, 0)}
    periodic = neighbor_list(
        "PS",
        torch.zeros((1, 3), device=torch_device),
        torch.diag(torch.tensor([0.4, 8.0, 8.0], device=torch_device)),
        torch.tensor([True, False, False], device=torch_device),
        1.0,
    )
    assert pair_keys(*periodic) == {
        (0, 0, -2, 0, 0),
        (0, 0, -1, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 2, 0, 0),
    }


def test_unwrapped_representatives_relabel_shifts(torch_device: torch.device) -> None:
    cell = torch.tensor(
        [[2.0, 0.1, 0.0], [0.2, 2.5, 0.0], [0.1, 0.3, 3.0]], dtype=torch.float64
    )
    positions = torch.tensor([[0.2, 0.1, 0.4], [1.8, 0.2, 0.5]], dtype=torch.float64)
    translated = positions.clone()
    translated[0] -= 3 * cell[1]
    translated[1] += 2 * cell[0] - cell[2]
    pbc = torch.ones(3, dtype=torch.bool)
    first_pairs, first_shifts = neighbor_list(
        "PS",
        positions.to(torch_device),
        cell.to(torch_device),
        pbc.to(torch_device),
        1.0,
    )
    second_pairs, second_shifts = neighbor_list(
        "PS",
        translated.to(torch_device),
        cell.to(torch_device),
        pbc.to(torch_device),
        1.0,
    )
    first_displacements = (
        positions.to(torch_device)[first_pairs[:, 1]]
        - positions.to(torch_device)[first_pairs[:, 0]]
        + first_shifts.to(torch.float64) @ cell.to(torch_device)
    )
    second_displacements = (
        translated.to(torch_device)[second_pairs[:, 1]]
        - translated.to(torch_device)[second_pairs[:, 0]]
        + second_shifts.to(torch.float64) @ cell.to(torch_device)
    )
    assert pair_keys(first_pairs, first_shifts) != pair_keys(
        second_pairs, second_shifts
    )
    first_rows = sorted(map(tuple, np.round(first_displacements.cpu().numpy(), 12)))
    second_rows = sorted(map(tuple, np.round(second_displacements.cpu().numpy(), 12)))
    assert first_rows == second_rows


def test_rotation_and_reflection_covariance(torch_device: torch.device) -> None:
    cell = torch.tensor(
        [[2.0, 0.3, 0.1], [0.2, 2.4, 0.4], [0.1, 0.2, 2.8]], dtype=torch.float64
    )
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [1.7, 0.4, 0.5], [0.6, 1.8, 1.1]], dtype=torch.float64
    )
    orthogonal = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float64
    )
    pbc = torch.ones(3, dtype=torch.bool)
    first = neighbor_list(
        "PS",
        positions.to(torch_device),
        cell.to(torch_device),
        pbc.to(torch_device),
        1.3,
    )
    second = neighbor_list(
        "PS",
        (positions @ orthogonal).to(torch_device),
        (cell @ orthogonal).to(torch_device),
        pbc.to(torch_device),
        1.3,
    )
    assert pair_keys(*first) == pair_keys(*second)


def test_randomized_differential(torch_device: torch.device) -> None:
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
    for case in range(30):
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
        batch_ptr = torch.tensor([0, n_atoms])
        expected = neighbor_list_reference(
            positions, cell[None], pbc[None], cutoff, batch_ptr
        )
        actual = neighbor_list(
            "PS",
            positions.to(torch_device),
            cell.to(torch_device),
            pbc.to(torch_device),
            cutoff,
        )
        assert pair_keys(*actual) == pair_keys(*expected)


def test_continuous_geometry_preserves_autograd(torch_device: torch.device) -> None:
    positions = torch.tensor(
        [[0.1, 0.0, 0.0], [1.8, 0.2, 0.0]],
        device=torch_device,
        dtype=torch.float64,
        requires_grad=True,
    )
    cell = torch.tensor(
        [[2.0, 0.0, 0.0], [0.1, 2.5, 0.0], [0.0, 0.0, 3.0]],
        device=torch_device,
        dtype=torch.float64,
        requires_grad=True,
    )
    pbc = torch.ones(3, device=torch_device, dtype=torch.bool)
    pair_indices, shifts = neighbor_list("PS", positions, cell, pbc, 0.8)
    displacements = (
        positions[pair_indices[:, 1]]
        - positions[pair_indices[:, 0]]
        + shifts.to(positions.dtype) @ cell
    )
    torch.sum(displacements.square()).backward()
    assert pair_indices.grad_fn is None
    assert shifts.grad_fn is None
    assert positions.grad is not None and torch.all(torch.isfinite(positions.grad))
    assert cell.grad is not None and torch.all(torch.isfinite(cell.grad))
    assert torch.count_nonzero(cell.grad) > 0


def test_empty_periodic_structure_skips_image_enumeration(
    torch_device: torch.device,
) -> None:
    pair_indices, shifts = neighbor_list(
        "PS",
        torch.empty((0, 3), dtype=torch.float64, device=torch_device),
        (1e-12 * torch.eye(3, dtype=torch.float64, device=torch_device))[None],
        torch.ones((1, 3), dtype=torch.bool, device=torch_device),
        1.0,
        torch.tensor([0, 0], device=torch_device),
    )
    assert pair_keys(pair_indices, shifts) == set()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forced_cell_list_matches_reference(
    torch_device: torch.device, dtype: torch.dtype
) -> None:
    generator = torch.Generator().manual_seed(1842)
    n_atoms = 288
    cell = torch.tensor(
        [[8.0, 0.4, 0.1], [0.2, 7.5, 0.5], [0.3, 0.1, 9.0]], dtype=dtype
    )
    positions = torch.rand((n_atoms, 3), generator=generator, dtype=dtype) @ cell
    pbc = torch.ones(3, dtype=torch.bool)
    batch_ptr = torch.tensor([0, n_atoms])
    expected = neighbor_list_reference(positions, cell[None], pbc[None], 1.2, batch_ptr)
    actual = neighbor_list(
        "PS",
        positions.to(torch_device),
        cell.to(torch_device),
        pbc.to(torch_device),
        1.2,
        algorithm="cell_list",
    )
    assert pair_keys(*actual) == pair_keys(*expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forced_cell_list_preserves_nextafter_inside_pair(
    torch_device: torch.device, dtype: torch.dtype
) -> None:
    positions = torch.zeros((256, 3), dtype=dtype)
    positions[1, 0] = torch.nextafter(
        torch.tensor(1.0, dtype=dtype), torch.tensor(0.0, dtype=dtype)
    )
    positions[2:, 1] = 3 * torch.arange(2, 256, dtype=dtype)
    output = neighbor_list(
        "PS",
        positions.to(torch_device),
        torch.zeros((3, 3), dtype=dtype, device=torch_device),
        torch.zeros(3, dtype=torch.bool, device=torch_device),
        1.0,
        algorithm="cell_list",
    )
    assert pair_keys(*output) == {(0, 1, 0, 0, 0), (1, 0, 0, 0, 0)}


def test_large_unwrapped_representatives_use_original_geometry(
    torch_device: torch.device,
) -> None:
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
    output = neighbor_list(
        "PS",
        positions.to(torch_device),
        cell.to(torch_device),
        torch.ones(3, dtype=torch.bool, device=torch_device),
        1.0,
    )
    assert pair_keys(*output) == {(0, 1, 0, 0, 0), (1, 0, 0, 0, 0)}
