from __future__ import annotations

import inspect
from importlib.metadata import version

import pytest
import torch

import tonari
from tonari import neighbor_list


def pair_keys(output: tuple[torch.Tensor, torch.Tensor]) -> set[tuple[int, ...]]:
    pair_indices, cell_shifts = output
    rows = torch.cat((pair_indices, cell_shifts.to(torch.int64)), dim=1)
    return {tuple(row) for row in rows.cpu().tolist()}


def test_public_surface_contains_only_neighbor_list() -> None:
    assert tonari.__all__ == ["neighbor_list"]
    assert not hasattr(tonari, "find_neighbors")


def test_version_matches_distribution_metadata() -> None:
    assert tonari.__version__ == version(tonari.__name__)


def test_options_are_keyword_only_and_default_to_existing_behavior() -> None:
    signature = inspect.signature(neighbor_list)
    assert tuple(signature.parameters)[:5] == (
        "quantities",
        "positions",
        "cell",
        "pbc",
        "cutoff",
    )
    assert signature.parameters["algorithm"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["half_list"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["include_self"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["algorithm"].default == "auto"
    assert signature.parameters["half_list"].default is False
    assert signature.parameters["include_self"].default is False


@pytest.mark.parametrize(
    ("algorithm", "error", "message"),
    [
        (None, TypeError, "algorithm must be a string"),
        ("fast", ValueError, "algorithm must be 'auto'"),
    ],
)
def test_invalid_algorithm_is_rejected(
    algorithm: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        neighbor_list(
            "PS",
            torch.zeros((1, 3)),
            torch.zeros((3, 3)),
            torch.zeros(3, dtype=torch.bool),
            1.0,
            algorithm=algorithm,
        )


@pytest.mark.parametrize("positions", [torch.tensor(1.0), torch.zeros(3)])
def test_invalid_torch_positions_shape_raises_value_error(
    positions: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="positions must have shape"):
        neighbor_list(
            "PS",
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
    single = neighbor_list("PS", positions, cell, pbc, 1.3)
    batched = neighbor_list(
        "PS",
        positions,
        cell[None],
        pbc[None],
        1.3,
        batch_ptr=torch.tensor([0, len(positions)]),
    )
    assert pair_keys(single) == pair_keys(batched)


def test_cell_shift_translates_the_target_image() -> None:
    positions = torch.tensor([[0.9, 0.0, 0.0], [0.1, 0.0, 0.0]])
    cell = torch.diag(torch.tensor([1.0, 4.0, 4.0]))
    pbc = torch.tensor([True, False, False])

    output = neighbor_list("PS", positions, cell, pbc, 0.3)

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
    baseline = neighbor_list("PS", positions, cell, pbc, 1.25, **options)
    scaled = neighbor_list(
        "PS", positions * scale, cell * scale, pbc, 1.25 * scale, **options
    )
    assert pair_keys(scaled) == pair_keys(baseline)


@pytest.mark.parametrize(
    ("cell", "pbc", "batch_ptr", "message"),
    [
        (
            torch.zeros((1, 3, 3)),
            torch.zeros(3, dtype=torch.bool),
            None,
            "single-structure cell",
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
            "batched cell",
        ),
    ],
)
def test_single_and_batch_shapes_are_unambiguous(
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_ptr: torch.Tensor | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        neighbor_list("PS", torch.zeros((2, 3)), cell, pbc, 1.0, batch_ptr)


def test_batch_ptr_defines_structure_boundaries() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0]])
    cell = torch.zeros((2, 3, 3))
    pbc = torch.zeros((2, 3), dtype=torch.bool)
    pair_indices, _ = neighbor_list(
        "PS", positions, cell, pbc, 0.6, batch_ptr=torch.tensor([0, 2, 3])
    )
    assert torch.all(pair_indices < 2)
