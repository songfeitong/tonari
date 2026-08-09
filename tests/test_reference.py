from __future__ import annotations

import torch

from torch_radius_graph import reference_radius_graph_pbc


def edge_keys(edge_index: torch.Tensor, shifts: torch.Tensor) -> set[tuple[int, ...]]:
    keys = torch.cat((edge_index.T, shifts.to(torch.int64)), dim=1).cpu().tolist()
    return {tuple(row) for row in keys}


def test_finite_directed_graph_excludes_onsite_and_strict_boundary() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    edge_index, shifts = reference_radius_graph_pbc(
        positions,
        torch.tensor([0, 3]),
        torch.zeros((1, 3, 3)),
        torch.zeros((1, 3), dtype=torch.bool),
        cutoff=1.0,
    )
    assert edge_keys(edge_index, shifts) == {
        (1, 2, 0, 0, 0),
        (2, 1, 0, 0, 0),
    }


def test_periodic_small_cell_retains_self_images_and_multiple_images() -> None:
    edge_index, shifts = reference_radius_graph_pbc(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([0, 1]),
        torch.diag(torch.tensor([1.0, 9.0, 9.0]))[None],
        torch.tensor([[True, False, False]]),
        cutoff=2.1,
    )
    assert edge_keys(edge_index, shifts) == {
        (0, 0, -2, 0, 0),
        (0, 0, -1, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 2, 0, 0),
    }


def test_mixed_batch_uses_each_structure_cell_and_never_crosses() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0]])
    edge_index, shifts = reference_radius_graph_pbc(
        positions,
        torch.tensor([0, 2, 3]),
        torch.stack((torch.zeros((3, 3)), torch.diag(torch.tensor([0.5, 8.0, 8.0])))),
        torch.tensor([[False, False, False], [True, False, False]]),
        cutoff=0.6,
    )
    keys = edge_keys(edge_index, shifts)
    assert {(0, 1, 0, 0, 0), (1, 0, 0, 0, 0)} <= keys
    assert (2, 2, -1, 0, 0) in keys
    assert (2, 2, 1, 0, 0) in keys
    assert all(
        (source < 2 and target < 2) or (source == target == 2)
        for source, target, *_ in keys
    )


def test_representative_translation_relabels_shift_without_changing_vectors() -> None:
    cell = torch.diag(torch.tensor([2.0, 3.0, 4.0]))
    positions = torch.tensor([[0.2, 0.1, 0.0], [1.8, 0.1, 0.0]])
    translated = positions.clone()
    translated[1] += cell[0]
    common = (
        torch.tensor([0, 2]),
        cell[None],
        torch.tensor([[True, True, True]]),
        0.5,
    )
    first_edges, first_shifts = reference_radius_graph_pbc(positions, *common)
    second_edges, second_shifts = reference_radius_graph_pbc(translated, *common)
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
    assert torch.allclose(
        torch.sort(first_vectors[:, 0]).values,
        torch.sort(second_vectors[:, 0]).values,
    )
