from __future__ import annotations

import pytest
import torch

from tests.support.assertions import pair_keys
from tests.support.reference import neighbor_list_reference
from tonari import neighbor_list


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_rejects_nonfinite_positions(torch_device: torch.device, value: float) -> None:
    positions = torch.zeros((2, 3), dtype=torch.float64, device=torch_device)
    positions[0, 0] = value
    with pytest.raises(RuntimeError, match="positions must"):
        neighbor_list(
            "PS",
            positions,
            torch.zeros((3, 3), dtype=torch.float64, device=torch_device),
            torch.zeros(3, dtype=torch.bool, device=torch_device),
            0.5,
        )


def test_rejects_representative_wrap_outside_int32_range(
    torch_device: torch.device,
) -> None:
    positions = torch.tensor(
        [[0.1, 0.0, 0.0], [2**31 + 0.2, 0.0, 0.0]],
        dtype=torch.float64,
        device=torch_device,
    )
    with pytest.raises(RuntimeError, match="wraps.*int32"):
        neighbor_list(
            "PS",
            positions,
            torch.eye(3, dtype=torch.float64, device=torch_device),
            torch.tensor([True, False, False], device=torch_device),
            0.5,
        )


def test_rejects_output_shift_outside_int32_range(
    torch_device: torch.device,
) -> None:
    positions = torch.tensor(
        [[-(2**31) + 0.1, 0.0, 0.0], [2**31 - 0.8, 0.0, 0.0]],
        dtype=torch.float64,
        device=torch_device,
    )
    with pytest.raises(RuntimeError, match="cell shift.*int32"):
        neighbor_list(
            "PS",
            positions,
            torch.eye(3, dtype=torch.float64, device=torch_device),
            torch.tensor([True, False, False], device=torch_device),
            0.5,
        )


def test_rejects_dependent_active_cell_rows(torch_device: torch.device) -> None:
    cell = torch.tensor(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 4.0]],
        dtype=torch.float64,
        device=torch_device,
    )
    with pytest.raises(ValueError, match="linearly independent"):
        neighbor_list(
            "PS",
            torch.zeros((1, 3), dtype=torch.float64, device=torch_device),
            cell,
            torch.tensor([True, True, False], device=torch_device),
            0.5,
        )


def test_rejects_nonfinite_inactive_cell_row(torch_device: torch.device) -> None:
    cell = torch.eye(3, dtype=torch.float64, device=torch_device)
    cell[2, 0] = torch.nan
    with pytest.raises(ValueError, match="cell must contain only finite values"):
        neighbor_list(
            "PS",
            torch.zeros((1, 3), dtype=torch.float64, device=torch_device),
            cell,
            torch.tensor([True, True, False], device=torch_device),
            0.5,
        )


def test_accepts_ill_conditioned_full_rank_active_rows(
    torch_device: torch.device,
) -> None:
    cell = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 1e-08, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
    )
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 5e-10, 0.0]], dtype=torch.float64)
    pbc = torch.tensor([True, True, False])
    expected = neighbor_list_reference(
        positions,
        cell[None],
        pbc[None],
        1e-09,
        torch.tensor([0, len(positions)]),
    )
    actual = neighbor_list(
        "PS",
        positions.to(torch_device),
        cell.to(torch_device),
        pbc.to(torch_device),
        1e-09,
    )
    assert pair_keys(*actual) == pair_keys(*expected)


def test_rejects_pathological_periodic_image_count(
    torch_device: torch.device,
) -> None:
    with pytest.raises(ValueError, match="image count.*resource limit"):
        neighbor_list(
            "PS",
            torch.zeros((1, 3), dtype=torch.float64, device=torch_device),
            0.001 * torch.eye(3, dtype=torch.float64, device=torch_device),
            torch.ones(3, dtype=torch.bool, device=torch_device),
            1.0,
        )
