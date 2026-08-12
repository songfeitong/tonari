from __future__ import annotations

import numpy as np
import pytest
import torch

from tonari import neighbor_list


def test_quantities_follow_requested_order_and_layout() -> None:
    positions = torch.tensor([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=torch.float64)
    cell = torch.diag(torch.tensor([1.0, 4.0, 4.0], dtype=torch.float64))
    pbc = torch.tensor([True, False, False])

    target, source, pairs, shifts, distances, displacements, source_again = (
        neighbor_list("jiPSdDi", positions, cell, pbc, 0.3)
    )

    assert pairs.shape == (2, 2)
    assert shifts.shape == (2, 3)
    assert pairs.dtype == source.dtype == target.dtype == torch.int64
    assert shifts.dtype == torch.int32
    assert distances.dtype == displacements.dtype == positions.dtype
    assert torch.equal(source, pairs[:, 0])
    assert torch.equal(target, pairs[:, 1])
    assert torch.equal(source_again, source)
    expected_displacements = (
        positions[target] - positions[source] + shifts.to(positions.dtype) @ cell
    )
    assert torch.equal(displacements, expected_displacements)
    assert torch.equal(
        distances, torch.linalg.vector_norm(expected_displacements, dim=1)
    )


def test_quantities_always_return_a_tuple_and_validate_characters() -> None:
    positions = torch.zeros((1, 3))
    cell = torch.zeros((3, 3))
    pbc = torch.zeros(3, dtype=torch.bool)

    single = neighbor_list("i", positions, cell, pbc, 1.0)
    assert isinstance(single, tuple) and len(single) == 1
    assert neighbor_list("", positions, cell, pbc, 1.0) == ()
    with pytest.raises(TypeError, match="quantities must be a string"):
        neighbor_list(None, positions, cell, pbc, 1.0)
    with pytest.raises(ValueError, match="unsupported quantities"):
        neighbor_list("ix", positions, cell, pbc, 1.0)


def test_numpy_and_torch_quantities_match() -> None:
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [1.3, 0.2, 0.0]],
        dtype=np.float32,
    )
    cell = np.diag(np.array([2.0, 3.0, 4.0], dtype=np.float32))
    pbc = np.array([True, False, False])

    numpy_output = neighbor_list("ijPSdD", positions, cell, pbc, 0.9)
    torch_output = neighbor_list(
        "ijPSdD",
        torch.from_numpy(positions),
        torch.from_numpy(cell),
        torch.from_numpy(pbc),
        0.9,
    )

    for numpy_array, torch_tensor in zip(numpy_output, torch_output, strict=True):
        np.testing.assert_array_equal(numpy_array, torch_tensor.numpy())


def test_batch_displacements_use_each_structure_cell() -> None:
    positions = torch.tensor(
        [[0.9, 0.0, 0.0], [0.1, 0.0, 0.0], [1.8, 0.0, 0.0], [0.2, 0.0, 0.0]],
        dtype=torch.float64,
    )
    cell = torch.stack(
        (
            torch.diag(torch.tensor([1.0, 4.0, 4.0], dtype=torch.float64)),
            torch.diag(torch.tensor([2.0, 4.0, 4.0], dtype=torch.float64)),
        )
    )
    pbc = torch.tensor([[True, False, False], [True, False, False]])
    batch_ptr = torch.tensor([0, 2, 4])

    pairs, shifts, displacements = neighbor_list(
        "PSD", positions, cell, pbc, 0.5, batch_ptr
    )
    pair_batch = torch.bucketize(pairs[:, 0].contiguous(), batch_ptr[1:], right=True)
    expected = (
        positions[pairs[:, 1]]
        - positions[pairs[:, 0]]
        + torch.einsum("ei,eij->ej", shifts.to(positions.dtype), cell[pair_batch])
    )
    assert torch.equal(displacements, expected)


def test_torch_distances_and_displacements_preserve_autograd(
    torch_device: torch.device,
) -> None:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
        dtype=torch.float64,
        device=torch_device,
        requires_grad=True,
    )
    cell = (
        torch.eye(3, dtype=torch.float64, device=torch_device) * 4.0
    ).requires_grad_()
    pbc = torch.zeros(3, dtype=torch.bool, device=torch_device)

    distances, displacements = neighbor_list("dD", positions, cell, pbc, 1.0)
    (distances.sum() + displacements.sum()).backward()

    assert positions.grad is not None
    assert cell.grad is not None


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_quantities_match_cpu() -> None:
    positions = torch.tensor([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=torch.float64)
    cell = torch.diag(torch.tensor([1.0, 4.0, 4.0], dtype=torch.float64))
    pbc = torch.tensor([True, False, False])

    cpu = neighbor_list("ijPSdD", positions, cell, pbc, 0.3, sorted=True)
    cuda = neighbor_list(
        "ijPSdD", positions.cuda(), cell.cuda(), pbc.cuda(), 0.3, sorted=True
    )

    def sorted_rows(output: tuple[torch.Tensor, ...]) -> np.ndarray:
        source, target, pairs, shifts, distances, displacements = output
        assert torch.equal(source, pairs[:, 0])
        assert torch.equal(target, pairs[:, 1])
        assert torch.all(source[1:] >= source[:-1])
        rows = (
            torch.cat(
                (
                    pairs,
                    shifts.to(pairs.dtype),
                    distances[:, None],
                    displacements,
                ),
                dim=1,
            )
            .cpu()
            .numpy()
        )
        order = np.lexsort(tuple(rows[:, column] for column in range(4, -1, -1)))
        return rows[order]

    np.testing.assert_allclose(sorted_rows(cpu), sorted_rows(cuda))
