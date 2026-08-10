from __future__ import annotations

import pytest
import torch

from tonari._reference import find_neighbors_reference


def pair_keys(pair_indices: torch.Tensor, shifts: torch.Tensor) -> set[tuple[int, ...]]:
    keys = torch.cat((pair_indices.T, shifts.to(torch.int64)), dim=1).cpu().tolist()
    return {tuple(row) for row in keys}


def test_finite_directed_pairs_exclude_onsite_and_strict_boundary() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    pair_indices, shifts = find_neighbors_reference(
        positions,
        torch.zeros((1, 3, 3)),
        torch.zeros((1, 3), dtype=torch.bool),
        cutoff=1.0,
        offsets=torch.tensor([0, 3]),
    )
    assert pair_keys(pair_indices, shifts) == {(1, 2, 0, 0, 0), (2, 1, 0, 0, 0)}


def test_periodic_small_cell_retains_self_images_and_multiple_images() -> None:
    pair_indices, shifts = find_neighbors_reference(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.diag(torch.tensor([1.0, 9.0, 9.0]))[None],
        torch.tensor([[True, False, False]]),
        cutoff=2.1,
        offsets=torch.tensor([0, 1]),
    )
    assert pair_keys(pair_indices, shifts) == {
        (0, 0, -2, 0, 0),
        (0, 0, -1, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 2, 0, 0),
    }


def test_mixed_batch_uses_each_structure_cell_and_never_crosses() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0]])
    pair_indices, shifts = find_neighbors_reference(
        positions,
        torch.stack((torch.zeros((3, 3)), torch.diag(torch.tensor([0.5, 8.0, 8.0])))),
        torch.tensor([[False, False, False], [True, False, False]]),
        cutoff=0.6,
        offsets=torch.tensor([0, 2, 3]),
    )
    keys = pair_keys(pair_indices, shifts)
    assert {(0, 1, 0, 0, 0), (1, 0, 0, 0, 0)} <= keys
    assert (2, 2, -1, 0, 0) in keys
    assert (2, 2, 1, 0, 0) in keys
    assert all(
        (
            source < 2 and target < 2 or source == target == 2
            for source, target, *_ in keys
        )
    )


def test_representative_translation_relabels_shift_without_changing_vectors() -> None:
    cell = torch.diag(torch.tensor([2.0, 3.0, 4.0]))
    positions = torch.tensor([[0.2, 0.1, 0.0], [1.8, 0.1, 0.0]])
    translated = positions.clone()
    translated[1] += cell[0]
    common = (cell, torch.tensor([True, True, True]), 0.5)
    first_pairs, first_shifts = find_neighbors_reference(positions, *common)
    second_pairs, second_shifts = find_neighbors_reference(translated, *common)
    first_displacements = (
        positions[first_pairs[0]]
        - positions[first_pairs[1]]
        + first_shifts.to(positions.dtype) @ cell
    )
    second_displacements = (
        translated[second_pairs[0]]
        - translated[second_pairs[1]]
        + second_shifts.to(positions.dtype) @ cell
    )
    assert pair_keys(first_pairs, first_shifts) != pair_keys(
        second_pairs, second_shifts
    )
    assert torch.allclose(
        torch.sort(first_displacements[:, 0]).values,
        torch.sort(second_displacements[:, 0]).values,
    )


def test_empty_periodic_reference_skips_tiny_cell_image_enumeration() -> None:
    pair_indices, shifts = find_neighbors_reference(
        torch.empty((0, 3), dtype=torch.float64),
        (0.02 * torch.eye(3, dtype=torch.float64))[None],
        torch.ones((1, 3), dtype=torch.bool),
        1.0,
        torch.tensor([0, 0]),
    )
    assert pair_indices.shape == (2, 0)
    assert shifts.shape == (0, 3)


def test_reference_rejects_pathological_periodic_image_count() -> None:
    with pytest.raises(ValueError, match="image count.*resource limit"):
        find_neighbors_reference(
            torch.zeros((1, 3), dtype=torch.float64),
            (0.001 * torch.eye(3, dtype=torch.float64))[None],
            torch.ones((1, 3), dtype=torch.bool),
            1.0,
            torch.tensor([0, 1]),
        )
