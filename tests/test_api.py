from __future__ import annotations

import inspect
from importlib.metadata import version

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


def test_version_matches_distribution_metadata() -> None:
    assert tonari.__version__ == version(tonari.__name__)


def test_pair_options_are_keyword_only_and_default_to_existing_behavior() -> None:
    signature = inspect.signature(find_neighbors)
    assert signature.parameters["half_list"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["include_self"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["half_list"].default is False
    assert signature.parameters["include_self"].default is False


@pytest.mark.parametrize("positions", [torch.tensor(1.0), torch.zeros(3)])
def test_invalid_torch_positions_shape_raises_value_error(
    positions: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="positions must have shape"):
        find_neighbors(
            positions,
            torch.eye(3),
            torch.zeros(3, dtype=torch.bool),
            1.0,
        )


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


def test_cell_shift_translates_the_target_image() -> None:
    positions = torch.tensor([[0.9, 0.0, 0.0], [0.1, 0.0, 0.0]])
    cell = torch.diag(torch.tensor([1.0, 4.0, 4.0]))
    pbc = torch.tensor([True, False, False])

    output = find_neighbors(positions, cell, pbc, 0.3)

    assert pair_keys(output) == {
        (0, 1, 1, 0, 0),
        (1, 0, -1, 0, 0),
    }


@pytest.mark.parametrize("scale", [1e-3, 1e3])
@pytest.mark.parametrize("half_list", [False, True])
@pytest.mark.parametrize("include_self", [False, True])
def test_neighbor_identity_is_independent_of_length_unit(
    scale: float, half_list: bool, include_self: bool
) -> None:
    positions = torch.tensor(
        [[0.2, 0.3, 0.1], [1.6, 0.2, 0.4], [0.7, 1.5, 1.0]],
        dtype=torch.float64,
    )
    cell = torch.tensor(
        [[2.1, 0.2, 0.1], [0.3, 2.5, 0.2], [0.1, 0.4, 2.9]],
        dtype=torch.float64,
    )
    pbc = torch.tensor([True, True, True])
    options = {"half_list": half_list, "include_self": include_self}
    baseline = find_neighbors(positions, cell, pbc, 1.25, **options)
    scaled = find_neighbors(
        positions * scale, cell * scale, pbc, 1.25 * scale, **options
    )
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
