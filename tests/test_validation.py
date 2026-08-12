from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import torch

from tests.support.assertions import pair_keys
from tonari import neighbor_list

Ecosystem = Literal["numpy", "torch"]


def as_ecosystem(
    tensor: torch.Tensor, ecosystem: Ecosystem
) -> np.ndarray | torch.Tensor:
    return tensor.numpy() if ecosystem == "numpy" else tensor


def valid_batch(
    ecosystem: Ecosystem,
    batch_ptr: tuple[int, ...] = (0, 1, 2),
) -> tuple[object, object, object, object]:
    batch_size = len(batch_ptr) - 1
    tensors = (
        torch.zeros((2, 3), dtype=torch.float64),
        torch.zeros((batch_size, 3, 3), dtype=torch.float64),
        torch.zeros((batch_size, 3), dtype=torch.bool),
        torch.tensor(batch_ptr, dtype=torch.int64),
    )
    return tuple(as_ecosystem(tensor, ecosystem) for tensor in tensors)


@pytest.mark.parametrize("ecosystem", ["numpy", "torch"])
@pytest.mark.parametrize("argument", ["cell", "pbc", "batch_ptr"])
def test_array_arguments_must_use_one_ecosystem(
    ecosystem: Ecosystem, argument: str
) -> None:
    positions, cell, pbc, batch_ptr = valid_batch(ecosystem)
    arguments = {"cell": cell, "pbc": pbc, "batch_ptr": batch_ptr}
    value = arguments[argument]
    if ecosystem == "numpy":
        arguments[argument] = torch.from_numpy(value)
        message = "must all be NumPy arrays"
    else:
        arguments[argument] = value.numpy()
        message = "must all be PyTorch tensors"
    with pytest.raises(TypeError, match=message):
        neighbor_list(
            "PS",
            positions,
            arguments["cell"],
            arguments["pbc"],
            1.0,
            arguments["batch_ptr"],
        )


@pytest.mark.parametrize("ecosystem", ["numpy", "torch"])
@pytest.mark.parametrize(
    ("positions", "message"),
    [
        (torch.tensor(1.0), "positions must have shape"),
        (torch.zeros(3), "positions must have shape"),
        (torch.zeros((1, 2)), "positions must have shape"),
        (torch.zeros((1, 3), dtype=torch.int64), "positions must have dtype"),
    ],
)
def test_positions_contract(
    ecosystem: Ecosystem, positions: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        neighbor_list(
            "PS",
            as_ecosystem(positions, ecosystem),
            as_ecosystem(torch.eye(3, dtype=torch.float64), ecosystem),
            as_ecosystem(torch.zeros(3, dtype=torch.bool), ecosystem),
            1.0,
        )


@pytest.mark.parametrize("ecosystem", ["numpy", "torch"])
@pytest.mark.parametrize(
    ("cell", "pbc", "message"),
    [
        (
            torch.zeros((1, 3, 3)),
            torch.zeros(3, dtype=torch.bool),
            "single-structure cell",
        ),
        (
            torch.zeros((3, 3)),
            torch.zeros((1, 3), dtype=torch.bool),
            "single-structure pbc",
        ),
    ],
)
def test_single_structure_metadata_shapes(
    ecosystem: Ecosystem,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        neighbor_list(
            "PS",
            as_ecosystem(torch.zeros((2, 3)), ecosystem),
            as_ecosystem(cell, ecosystem),
            as_ecosystem(pbc, ecosystem),
            1.0,
        )


@pytest.mark.parametrize("ecosystem", ["numpy", "torch"])
@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("batch_ptr", torch.empty(0, dtype=torch.int64), "batch_ptr must be int64"),
        ("batch_ptr", torch.tensor([[0, 1, 2]]), "batch_ptr must be int64"),
        (
            "batch_ptr",
            torch.tensor([0, 1, 2], dtype=torch.int32),
            "batch_ptr must be int64",
        ),
        ("cell", torch.zeros((2, 3, 2), dtype=torch.float64), "cell must have shape"),
        ("cell", torch.zeros((2, 3, 3), dtype=torch.float32), "same dtype"),
        ("pbc", torch.zeros((2, 2), dtype=torch.bool), "pbc must be bool"),
        ("pbc", torch.zeros((2, 3), dtype=torch.int8), "pbc must be bool"),
    ],
)
def test_batched_metadata_contract(
    ecosystem: Ecosystem,
    field: str,
    invalid: torch.Tensor,
    message: str,
) -> None:
    positions, cell, pbc, batch_ptr = valid_batch(ecosystem)
    arguments = {
        "positions": positions,
        "cell": cell,
        "pbc": pbc,
        "batch_ptr": batch_ptr,
    }
    arguments[field] = as_ecosystem(invalid, ecosystem)
    with pytest.raises(ValueError, match=message):
        neighbor_list(
            "PS",
            arguments["positions"],
            arguments["cell"],
            arguments["pbc"],
            1.0,
            arguments["batch_ptr"],
        )


@pytest.mark.parametrize("ecosystem", ["numpy", "torch"])
@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("cell", torch.zeros((3, 3), dtype=torch.float64), "batched cell"),
        ("pbc", torch.zeros(3, dtype=torch.bool), "batched pbc"),
    ],
)
def test_explicit_batch_ptr_requires_batched_metadata(
    ecosystem: Ecosystem,
    field: str,
    invalid: torch.Tensor,
    message: str,
) -> None:
    positions, cell, pbc, batch_ptr = valid_batch(ecosystem)
    arguments = {"cell": cell, "pbc": pbc}
    arguments[field] = as_ecosystem(invalid, ecosystem)
    with pytest.raises(ValueError, match=message):
        neighbor_list(
            "PS",
            positions,
            arguments["cell"],
            arguments["pbc"],
            1.0,
            batch_ptr,
        )


@pytest.mark.parametrize("ecosystem", ["numpy", "torch"])
@pytest.mark.parametrize(
    ("batch_ptr", "message"),
    [
        ((1, 1, 2), "batch_ptr must start at zero"),
        ((0, 1, 1), "batch_ptr must end at N_total"),
        ((0, 2, 1, 2), "batch_ptr must be nondecreasing"),
    ],
)
def test_batch_ptr_boundaries(
    ecosystem: Ecosystem, batch_ptr: tuple[int, ...], message: str
) -> None:
    positions, cell, pbc, batch_ptr_array = valid_batch(ecosystem, batch_ptr)
    with pytest.raises(ValueError, match=message):
        neighbor_list("PS", positions, cell, pbc, 1.0, batch_ptr_array)


@pytest.mark.parametrize("ecosystem", ["numpy", "torch"])
@pytest.mark.parametrize("cutoff", [0.0, -1.0, float("nan"), float("inf")])
def test_cutoff_must_be_finite_and_positive(
    ecosystem: Ecosystem, cutoff: float
) -> None:
    positions, cell, pbc, batch_ptr = valid_batch(ecosystem)
    with pytest.raises(ValueError, match="cutoff must be finite and positive"):
        neighbor_list("PS", positions, cell, pbc, cutoff, batch_ptr)


def test_positions_must_be_a_supported_array() -> None:
    with pytest.raises(TypeError, match="PyTorch tensor or NumPy array"):
        neighbor_list("PS", [[0.0, 0.0, 0.0]], np.eye(3), np.zeros(3), 1.0)


def test_torch_rejects_unsupported_or_mixed_devices() -> None:
    with pytest.raises(ValueError, match="CPU or CUDA tensor"):
        neighbor_list(
            "PS",
            torch.zeros((1, 3), device="meta"),
            torch.zeros((3, 3), device="meta"),
            torch.zeros(3, dtype=torch.bool, device="meta"),
            1.0,
        )
    positions, cell, pbc, batch_ptr = valid_batch("torch")
    with pytest.raises(ValueError, match="must be on the same device"):
        neighbor_list("PS", positions, cell.to("meta"), pbc, 1.0, batch_ptr)


def test_torch_accepts_noncontiguous_inputs() -> None:
    positions = torch.arange(18, dtype=torch.float64).reshape(3, 6)[:, ::2]
    assert not positions.is_contiguous()
    output = neighbor_list(
        "PS",
        positions,
        torch.eye(3, dtype=torch.float64).T,
        torch.zeros(3, dtype=torch.bool),
        11.0,
    )
    assert len(pair_keys(output)) > 0
