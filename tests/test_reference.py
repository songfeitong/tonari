from __future__ import annotations

import pytest
import torch

from tests.assertions import pair_keys
from tests.reference import neighbor_list_reference


def test_finite_directed_pairs_exclude_onsite_and_strict_boundary() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    pair_indices, shifts = neighbor_list_reference(
        "PS",
        positions,
        torch.zeros((1, 3, 3)),
        torch.zeros((1, 3), dtype=torch.bool),
        cutoff=1.0,
        batch_ptr=torch.tensor([0, 3]),
    )
    assert pair_keys(pair_indices, shifts) == {(1, 2, 0, 0, 0), (2, 1, 0, 0, 0)}


def test_periodic_small_cell_retains_self_images_and_multiple_images() -> None:
    pair_indices, shifts = neighbor_list_reference(
        "PS",
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.diag(torch.tensor([1.0, 9.0, 9.0]))[None],
        torch.tensor([[True, False, False]]),
        cutoff=2.1,
        batch_ptr=torch.tensor([0, 1]),
    )
    assert pair_keys(pair_indices, shifts) == {
        (0, 0, -2, 0, 0),
        (0, 0, -1, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 2, 0, 0),
    }


def test_mixed_batch_uses_each_structure_cell_and_never_crosses() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0]])
    pair_indices, shifts = neighbor_list_reference(
        "PS",
        positions,
        torch.stack((torch.zeros((3, 3)), torch.diag(torch.tensor([0.5, 8.0, 8.0])))),
        torch.tensor([[False, False, False], [True, False, False]]),
        cutoff=0.6,
        batch_ptr=torch.tensor([0, 2, 3]),
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
    first_pairs, first_shifts = neighbor_list_reference("PS", positions, *common)
    second_pairs, second_shifts = neighbor_list_reference("PS", translated, *common)
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
    assert torch.allclose(
        torch.sort(first_displacements[:, 0]).values,
        torch.sort(second_displacements[:, 0]).values,
    )


def test_empty_periodic_reference_skips_tiny_cell_image_enumeration() -> None:
    pair_indices, shifts = neighbor_list_reference(
        "PS",
        torch.empty((0, 3), dtype=torch.float64),
        (0.02 * torch.eye(3, dtype=torch.float64))[None],
        torch.ones((1, 3), dtype=torch.bool),
        1.0,
        torch.tensor([0, 0]),
    )
    assert pair_indices.shape == (0, 2)
    assert shifts.shape == (0, 3)


def test_reference_rejects_pathological_periodic_image_count() -> None:
    with pytest.raises(ValueError, match="image count.*resource limit"):
        neighbor_list_reference(
            "PS",
            torch.zeros((1, 3), dtype=torch.float64),
            (0.001 * torch.eye(3, dtype=torch.float64))[None],
            torch.ones((1, 3), dtype=torch.bool),
            1.0,
            torch.tensor([0, 1]),
        )
