from __future__ import annotations

import pytest
import torch

from tests.support.assertions import pair_keys
from tests.support.reference import neighbor_list_reference
from tonari import neighbor_list

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def cuda_neighbors(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    batch_ptr: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return neighbor_list(
        "PS",
        positions.cuda(),
        cell.cuda(),
        pbc.cuda(),
        cutoff,
        None if batch_ptr is None else batch_ptr.cuda(),
    )


@pytest.mark.parametrize("n_atoms", [255, 256])
def test_paths_use_the_same_float32_cutoff_rounding(n_atoms: int) -> None:
    cutoff = 353.2019167901003
    positions = torch.zeros((n_atoms, 3), dtype=torch.float32)
    positions[1, 0] = torch.tensor(353.2019, dtype=torch.float32)
    positions[2:, 1] = 1000 * torch.arange(2, n_atoms, dtype=torch.float32)
    cell = torch.zeros((3, 3), dtype=torch.float32)
    pbc = torch.zeros(3, dtype=torch.bool)
    expected = {(0, 1, 0, 0, 0), (1, 0, 0, 0, 0)}
    cpu = neighbor_list("PS", positions, cell, pbc, cutoff)
    cuda = cuda_neighbors(positions, cell, pbc, cutoff)
    reference = neighbor_list_reference(
        positions,
        cell[None],
        pbc[None],
        cutoff,
        torch.tensor([0, n_atoms]),
    )
    assert pair_keys(*cpu) == expected
    assert pair_keys(*cuda) == expected
    assert pair_keys(*reference) == expected


def test_nondefault_stream_and_empty_batch_member() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], device="cuda")
    batch_ptr = torch.tensor([0, 0, 2], device="cuda")
    cell = torch.zeros((2, 3, 3), device="cuda")
    pbc = torch.zeros((2, 3), dtype=torch.bool, device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        positions = positions + 0.0
        pair_indices, shifts = neighbor_list(
            "PS", positions, cell, pbc, 0.5, batch_ptr, sorted=True
        )
    torch.cuda.current_stream().wait_stream(stream)
    assert pair_keys(pair_indices, shifts) == {(0, 1, 0, 0, 0), (1, 0, 0, 0, 0)}
    assert torch.all(pair_indices[1:, 0] >= pair_indices[:-1, 0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_auto_fallback_for_unwrapped_representatives_matches_reference(
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator().manual_seed(1842)
    n_atoms = 288
    cell = torch.tensor(
        [[8.0, 0.4, 0.1], [0.2, 7.5, 0.5], [0.3, 0.1, 9.0]], dtype=dtype
    )
    positions = torch.rand((n_atoms, 3), generator=generator, dtype=dtype) @ cell
    positions[:5] += 3 * cell[0] - 2 * cell[1]
    batch_ptr = torch.tensor([0, n_atoms])
    pbc = torch.ones((1, 3), dtype=torch.bool)
    expected = neighbor_list_reference(positions, cell[None], pbc, 1.2, batch_ptr)
    actual = cuda_neighbors(positions, cell[None], pbc, 1.2, batch_ptr)
    assert pair_keys(*actual) == pair_keys(*expected)


def test_auto_fallback_handles_mixed_finite_and_partial_pbc_batch() -> None:
    generator = torch.Generator().manual_seed(5801)
    counts = (256, 257)
    cell = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[2.1, 0.3, 0.2], [0.4, 4.0, 0.1], [0.2, 0.5, 5.0]],
        ],
        dtype=torch.float64,
    )
    positions = torch.cat(
        (
            torch.rand((counts[0], 3), generator=generator, dtype=torch.float64) * 8,
            torch.rand((counts[1], 3), generator=generator, dtype=torch.float64)
            @ cell[1],
        )
    )
    positions[-3:] += 4 * cell[1, 0]
    batch_ptr = torch.tensor([0, counts[0], sum(counts)])
    pbc = torch.tensor([[False, False, False], [True, False, False]])
    expected = neighbor_list_reference(positions, cell, pbc, 0.55, batch_ptr)
    actual = cuda_neighbors(positions, cell, pbc, 0.55, batch_ptr)
    assert pair_keys(*actual) == pair_keys(*expected)
    pair_indices, shifts = actual
    source_batch = torch.bucketize(
        pair_indices[:, 0].contiguous(), batch_ptr[1:].cuda(), right=True
    )
    target_batch = torch.bucketize(
        pair_indices[:, 1].contiguous(), batch_ptr[1:].cuda(), right=True
    )
    assert torch.equal(source_batch, target_batch)
    displacements = (
        positions.cuda()[pair_indices[:, 1]]
        - positions.cuda()[pair_indices[:, 0]]
        + torch.einsum(
            "ei,eij->ej", shifts.to(torch.float64), cell.cuda()[source_batch]
        )
    )
    assert torch.all(torch.sum(displacements.square(), dim=1) < 0.55**2)
    assert torch.all(shifts[source_batch == 1, 1:] == 0)


def test_cell_list_large_common_translation_uses_public_vector_formula() -> None:
    cell = torch.tensor(
        [
            [-0.9499396681785583, 2.3687520027160645, 1.7597240209579468],
            [-3.4348089694976807, -1.8801195621490479, 1.9097744226455688],
            [2.5219004154205322, -1.9322474002838135, 4.843184471130371],
        ]
    )
    positions = torch.tensor([-2138.380859375, -11887.5810546875, 10318.5625]).repeat(
        256, 1
    )
    positions[52] = torch.tensor(
        [-2135.800048828125, -11886.6826171875, 10317.2880859375]
    )
    positions[40] = torch.tensor(
        [-2141.306396484375, -11885.267578125, 10315.8291015625]
    )
    positions[61] = torch.tensor([-2135.84033203125, -11887.59375, 10317.1181640625])
    arguments = (positions, cell, torch.ones(3, dtype=torch.bool), 1.4481067657470703)
    expected_keys = pair_keys(neighbor_list("PS", *arguments))
    actual = cuda_neighbors(*arguments)
    assert (52, 23, 0, -1, 0) in expected_keys
    assert (23, 52, 0, 1, 0) in expected_keys
    assert (61, 40, -1, -1, 1) not in expected_keys
    assert (40, 61, 1, 1, -1) not in expected_keys
    assert pair_keys(*actual) == expected_keys


def test_cell_list_rejects_representative_wrap_outside_int32_range() -> None:
    n_atoms = 256
    positions = torch.zeros((n_atoms, 3), dtype=torch.float64)
    positions[0, 0] = 0.1
    positions[1, 0] = 2**31 + 0.2
    positions[2:, 1] = 2 * torch.arange(2, n_atoms, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="wraps.*int32"):
        cuda_neighbors(
            positions,
            torch.eye(3, dtype=torch.float64)[None],
            torch.tensor([[True, False, False]]),
            0.5,
            torch.tensor([0, n_atoms]),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_cell_list_rejects_nonfinite_positions(value: float) -> None:
    n_atoms = 256
    positions = torch.zeros((n_atoms, 3), dtype=torch.float64)
    positions[0, 0] = value
    positions[:, 1] = 2 * torch.arange(n_atoms, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="positions must"):
        cuda_neighbors(
            positions,
            torch.zeros((1, 3, 3), dtype=torch.float64),
            torch.zeros((1, 3), dtype=torch.bool),
            0.5,
            torch.tensor([0, n_atoms]),
        )


@pytest.mark.parametrize(
    ("batch_ptr", "message"),
    [
        ((1, 1, 2), "start at zero"),
        ((0, 1, 1), "end at N_total"),
        ((0, 2, 1, 2), "nondecreasing"),
    ],
)
def test_rejects_invalid_batch_ptr_contents(
    batch_ptr: tuple[int, ...], message: str
) -> None:
    batch_size = len(batch_ptr) - 1
    with pytest.raises(ValueError, match=message):
        cuda_neighbors(
            torch.zeros((2, 3)),
            torch.zeros((batch_size, 3, 3)),
            torch.zeros((batch_size, 3), dtype=torch.bool),
            1.0,
            torch.tensor(batch_ptr),
        )


def test_cell_list_falls_back_for_extremely_sparse_bounds() -> None:
    positions = torch.zeros((256, 3))
    positions[:, 0] = torch.arange(256) * 10000.0
    output = cuda_neighbors(
        positions,
        torch.zeros((1, 3, 3)),
        torch.zeros((1, 3), dtype=torch.bool),
        1.0,
        torch.tensor([0, 256]),
    )
    assert pair_keys(*output) == set()


def test_batched_sparse_bin_counts_saturate_before_cumsum() -> None:
    n_atoms = 256
    structure = torch.full((n_atoms, 3), 1700000.0, dtype=torch.float64)
    structure[0] = 0.0
    positions = torch.cat((structure, structure))
    arguments = (
        positions,
        torch.zeros((2, 3, 3), dtype=torch.float64),
        torch.zeros((2, 3), dtype=torch.bool),
        1.0,
        torch.tensor([0, n_atoms, 2 * n_atoms]),
    )
    expected = neighbor_list("PS", *arguments)
    actual = cuda_neighbors(*arguments)
    assert pair_keys(*actual) == pair_keys(*expected)
