from __future__ import annotations

import pytest
import torch

import tonari
from tonari import find_neighbors


def pair_keys(output: tuple[torch.Tensor, torch.Tensor]) -> set[tuple[int, ...]]:
    pair_indices, cell_shifts = output
    rows = torch.cat((pair_indices.T, cell_shifts.to(torch.int64)), dim=1)
    return {tuple(row) for row in rows.cpu().tolist()}


def test_public_surface_contains_only_find_neighbors() -> None:
    assert tonari.__all__ == ["find_neighbors"]
    assert not hasattr(tonari, "radius_graph_pbc")
    assert not hasattr(tonari, "reference_radius_graph_pbc")


def test_single_structure_default_matches_explicit_batch() -> None:
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [1.7, 0.4, 0.5], [0.6, 1.8, 1.1]],
        dtype=torch.float64,
    )
    cell = torch.tensor(
        [[2.0, 0.3, 0.1], [0.2, 2.4, 0.4], [0.1, 0.2, 2.8]],
        dtype=torch.float64,
    )
    pbc = torch.tensor([True, False, True])
    single = find_neighbors(positions, cell, pbc, 1.3)
    batched = find_neighbors(
        positions,
        cell[None],
        pbc[None],
        1.3,
        offsets=torch.tensor([0, len(positions)]),
    )
    assert pair_keys(single) == pair_keys(batched)


@pytest.mark.parametrize("scale", [1e-3, 1e3])
def test_neighbor_identity_is_independent_of_length_unit(scale: float) -> None:
    positions = torch.tensor(
        [[0.2, 0.3, 0.1], [1.6, 0.2, 0.4], [0.7, 1.5, 1.0]],
        dtype=torch.float64,
    )
    cell = torch.tensor(
        [[2.1, 0.2, 0.1], [0.3, 2.5, 0.2], [0.1, 0.4, 2.9]],
        dtype=torch.float64,
    )
    pbc = torch.tensor([True, True, True])
    baseline = find_neighbors(positions, cell, pbc, 1.25)
    scaled = find_neighbors(positions * scale, cell * scale, pbc, 1.25 * scale)
    assert pair_keys(scaled) == pair_keys(baseline)


@pytest.mark.parametrize(
    ("cells", "pbc", "offsets", "message"),
    [
        (
            torch.zeros((1, 3, 3)),
            torch.zeros(3, dtype=torch.bool),
            None,
            "single-structure cells",
        ),
        (
            torch.zeros((3, 3)),
            torch.zeros((1, 3), dtype=torch.bool),
            None,
            "single-structure pbc",
        ),
        (
            torch.zeros((3, 3)),
            torch.zeros(3, dtype=torch.bool),
            torch.tensor([0, 2]),
            "batched cells",
        ),
    ],
)
def test_single_and_batch_shapes_are_unambiguous(
    cells: torch.Tensor,
    pbc: torch.Tensor,
    offsets: torch.Tensor | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        find_neighbors(torch.zeros((2, 3)), cells, pbc, 1.0, offsets)


def test_batch_offsets_define_structure_boundaries() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0]])
    cells = torch.zeros((2, 3, 3))
    pbc = torch.zeros((2, 3), dtype=torch.bool)
    pair_indices, _ = find_neighbors(
        positions, cells, pbc, 0.6, offsets=torch.tensor([0, 2, 3])
    )
    assert torch.all(pair_indices < 2)
